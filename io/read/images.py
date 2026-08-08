"""Image metadata and read-side helpers for xentools."""

from __future__ import annotations

import os
import io
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

try:
    from ...core.images import AlignedImage
except ImportError:
    from core.images import AlignedImage

__all__ = [
    "_parse_ome_xml",
    "_detect_linked_protein_images",
    "_image_extent_um",
    "_source_series_level_arrays",
    "find_alignment_file",
    "load_aligned_image",
]


def _numeric_identifiers(value):
    identifiers = set()
    for token in re.findall(r"\d{5,}", str(value)):
        identifiers.add(str(int(token)))
    return identifiers


def find_alignment_file(image_path, xenium_folder):
    """Find one unambiguous Explorer alignment bundle using shared sample IDs."""
    image_path = Path(image_path).expanduser().resolve()
    search_dirs = [image_path.parent, Path(xenium_folder).expanduser().resolve().parent]
    candidates = {
        path
        for folder in search_dirs
        for path in folder.iterdir()
        if "align" in path.name.lower()
        and (path.suffix.lower() == ".zip" or (path.is_dir() and (path / "matrix.csv").is_file()))
    }
    candidates = sorted(candidates)
    if not candidates:
        raise FileNotFoundError(
            "No alignment ZIP or extracted alignment folder was found beside "
            "the image or Xenium output. "
            "Pass alignment_file=<path> explicitly."
        )

    image_ids = _numeric_identifiers(image_path.stem)
    xenium_ids = _numeric_identifiers(Path(xenium_folder).name)
    scored = []
    for candidate in candidates:
        candidate_ids = _numeric_identifiers(candidate.stem)
        score = 2 * len(candidate_ids & image_ids) + len(candidate_ids & xenium_ids)
        if score:
            scored.append((score, candidate))
    if not scored:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            "Alignment auto-matching found no shared numeric sample identifier. "
            f"Candidates: {names}. Pass alignment_file=<path> explicitly."
        )
    best_score = max(score for score, _ in scored)
    best = [path for score, path in scored if score == best_score]
    if len(best) != 1:
        names = ", ".join(path.name for path in best)
        raise ValueError(f"Alignment match is ambiguous: {names}. Pass alignment_file explicitly.")
    return str(best[0])


def _read_alignment_bundle(alignment_file):
    path = Path(alignment_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    keypoints = None
    if path.is_dir():
        matrix_path = path / "matrix.csv"
        if not matrix_path.is_file():
            raise ValueError(f"Alignment folder '{path.name}' does not contain matrix.csv.")
        matrix = np.loadtxt(matrix_path, delimiter=",")
        keypoint_path = path / "keypoints.csv"
        if keypoint_path.is_file():
            import pandas as pd

            keypoints = pd.read_csv(keypoint_path)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = {Path(name).name.lower(): name for name in archive.namelist()}
            if "matrix.csv" not in names:
                raise ValueError(f"Alignment ZIP '{path.name}' does not contain matrix.csv.")
            matrix = np.loadtxt(io.BytesIO(archive.read(names["matrix.csv"])), delimiter=",")
            if "keypoints.csv" in names:
                import pandas as pd

                keypoints = pd.read_csv(io.BytesIO(archive.read(names["keypoints.csv"])))
    else:
        # Explorer exposes matrix.csv and keypoints.csv together. Accept either
        # member as the bundle entry point so selecting the human-readable
        # keypoint table does not fall through to an opaque NumPy parse error.
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig") as handle:
                first_line = handle.readline().strip()
            columns = {value.strip() for value in first_line.split(",")}
            keypoint_columns = {"fixedX", "fixedY", "alignmentX", "alignmentY"}
            if keypoint_columns.issubset(columns):
                matrix_path = path.with_name("matrix.csv")
                if not matrix_path.is_file():
                    raise ValueError(
                        f"'{path.name}' is an Explorer keypoint table, but its sibling "
                        "matrix.csv was not found. Pass the alignment ZIP, extracted "
                        "alignment folder, or matrix.csv."
                    )
                import pandas as pd

                keypoints = pd.read_csv(path)
                matrix = np.loadtxt(matrix_path, delimiter=",")
            else:
                try:
                    matrix = np.loadtxt(path, delimiter=",")
                except ValueError as exc:
                    raise ValueError(
                        f"Could not read '{path.name}' as a 3×3 alignment matrix. "
                        "Pass the Explorer alignment ZIP, extracted alignment folder, "
                        "matrix.csv, or keypoints.csv with a sibling matrix.csv."
                    ) from exc
                sibling_keypoints = path.with_name("keypoints.csv")
                if path.name.lower() == "matrix.csv" and sibling_keypoints.is_file():
                    import pandas as pd

                    keypoints = pd.read_csv(sibling_keypoints)
        else:
            try:
                matrix = np.loadtxt(path, delimiter=",")
            except ValueError as exc:
                raise ValueError(
                    f"Could not read '{path.name}' as a 3×3 alignment matrix. "
                    "Pass the Explorer alignment ZIP, extracted alignment folder, "
                    "or matrix.csv."
                ) from exc
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Alignment matrix must contain a finite 3×3 homogeneous transform.")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-8):
        raise ValueError("Alignment matrix last row must be [0, 0, 1].")
    if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError("Alignment matrix is singular and cannot register an image.")
    return matrix, keypoints, str(path)


def _validate_keypoints(matrix, keypoints):
    required = {"fixedX", "fixedY", "alignmentX", "alignmentY"}
    if keypoints is None or not required.issubset(keypoints.columns):
        return matrix, 0, None, None
    fixed = keypoints[["fixedX", "fixedY"]].to_numpy(dtype=float)
    moving = keypoints[["alignmentX", "alignmentY"]].to_numpy(dtype=float)

    def errors(candidate):
        homogeneous = np.column_stack([moving, np.ones(len(moving))])
        predicted = (candidate @ homogeneous.T).T
        predicted = predicted[:, :2] / predicted[:, 2, None]
        return np.linalg.norm(predicted - fixed, axis=1)

    direct_errors = errors(matrix)
    inverse = np.linalg.inv(matrix)
    inverse_errors = errors(inverse)
    if np.sqrt(np.mean(inverse_errors**2)) < np.sqrt(np.mean(direct_errors**2)):
        matrix, direct_errors = inverse, inverse_errors
    rmse = float(np.sqrt(np.mean(direct_errors**2)))
    max_error = float(direct_errors.max())
    return matrix, len(direct_errors), rmse, max_error


def _ome_image_metadata(image_path):
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        axes = series.axes
        shape = series.shape
        ome_xml = tif.ome_metadata
    y_index = axes.find("Y")
    x_index = axes.find("X")
    if y_index < 0 or x_index < 0:
        raise ValueError(f"Image axes {axes!r} do not contain both Y and X.")
    height, width = int(shape[y_index]), int(shape[x_index])
    pixel_x = pixel_y = None
    channel_names = []
    if ome_xml:
        root, ns = _parse_ome_xml(ome_xml)
        pixels = root.find(".//ome:Pixels", ns)
        if pixels is not None:
            pixel_x = float(pixels.attrib["PhysicalSizeX"]) if "PhysicalSizeX" in pixels.attrib else None
            pixel_y = float(pixels.attrib["PhysicalSizeY"]) if "PhysicalSizeY" in pixels.attrib else None
        channel_names = [
            node.attrib.get("Name", f"Channel {i}")
            for i, node in enumerate(root.findall(".//ome:Channel", ns))
        ]
    return (height, width), axes, (pixel_x, pixel_y), channel_names


def load_aligned_image(
    image_path,
    alignment_file,
    *,
    name,
    fixed_shape,
    fixed_pixel_size,
    poor_alignment_rmse_px=25.0,
    metadata=None,
):
    """Read an external image and Explorer affine into an AlignedImage."""
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    matrix, keypoints, alignment_path = _read_alignment_bundle(alignment_file)
    matrix, n_points, rmse, max_error = _validate_keypoints(matrix, keypoints)
    if rmse is not None and rmse > poor_alignment_rmse_px:
        warnings.warn(
            f"Alignment keypoint RMSE is {rmse:.2f} pixels, above the "
            f"{poor_alignment_rmse_px:.2f}-pixel diagnostic threshold.",
            UserWarning,
            stacklevel=2,
        )
    source_shape, axes, source_pixel_size, channel_names = _ome_image_metadata(image_path)
    return AlignedImage(
        name=str(name),
        path=str(image_path),
        source_to_fixed=matrix,
        source_shape=source_shape,
        source_axes=axes,
        source_pixel_size=source_pixel_size,
        fixed_shape=tuple(map(int, fixed_shape)),
        fixed_pixel_size=float(fixed_pixel_size),
        alignment_file=alignment_path,
        channel_names=channel_names,
        keypoint_count=n_points,
        keypoint_rmse_px=rmse,
        keypoint_max_error_px=max_error,
        metadata={} if metadata is None else dict(metadata),
    )


def _parse_ome_xml(ome_xml):
    root = ET.fromstring(ome_xml)
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    return root, ns


def _detect_linked_protein_images(xenium_folder):
    folder = os.path.join(xenium_folder, "morphology_focus")
    if not os.path.isdir(folder):
        return None

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".ome.tif", ".ome.tiff"))
    )
    if not files:
        return None

    first_file = os.path.join(folder, files[0])
    try:
        import tifffile

        with tifffile.TiffFile(first_file) as tif:
            ome_xml = tif.ome_metadata
    except Exception:
        return None

    if not ome_xml:
        return None

    root, ns = _parse_ome_xml(ome_xml)
    pixels = root.find(".//ome:Pixels", ns)
    if pixels is None:
        return None

    channel_names = []
    for channel in root.findall(".//ome:Channel", ns):
        channel_names.append(channel.attrib.get("Name", f"Channel {len(channel_names)}"))

    file_map = {}
    for tiffdata in root.findall(".//ome:TiffData", ns):
        first_c = int(tiffdata.attrib.get("FirstC", 0))
        uuid_node = tiffdata.find("ome:UUID", ns)
        if uuid_node is not None and uuid_node.attrib.get("FileName"):
            file_map[first_c] = uuid_node.attrib["FileName"]

    filenames = [file_map.get(i, files[i] if i < len(files) else None) for i in range(len(channel_names))]
    if any(name is None for name in filenames):
        return None
    if any(not os.path.exists(os.path.join(folder, name)) for name in filenames):
        return None

    return {
        "folder": folder,
        "files": [os.path.join(folder, name) for name in filenames],
        "filenames": filenames,
        "channel_names": channel_names,
        "axes": "CYX",
        "shape": (
            int(pixels.attrib.get("SizeC", len(channel_names))),
            int(pixels.attrib.get("SizeY", 0)),
            int(pixels.attrib.get("SizeX", 0)),
        ),
        "pixel_size": float(pixels.attrib.get("PhysicalSizeX", 1.0)),
        "linked": True,
        "source_file": first_file,
        "ome_xml_template": ome_xml,
    }


def _image_extent_um(image_path, pixel_size):
    """Return the full-resolution image extent in microns."""
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        shape = series.shape
    return (0.0, shape[-1] * pixel_size, 0.0, shape[-2] * pixel_size)


def _source_series_level_arrays(series, zarr_store=None):
    import zarr

    if zarr_store is None:
        zarr_store = series.aszarr()
    root = zarr.open(zarr_store, mode="r")
    if hasattr(root, "shape"):
        return [root], zarr_store
    keys = sorted((int(k), k) for k in root.keys() if str(k).isdigit())
    arrays = [root[k] for _, k in keys]
    if not arrays:
        raise ValueError("Could not find pyramid level arrays in zarr store.")
    return arrays, zarr_store
