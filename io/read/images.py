"""Image metadata and read-side helpers for xentools."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

__all__ = [
    "_parse_ome_xml",
    "_detect_linked_protein_images",
    "_image_extent_um",
    "_source_series_level_arrays",
]


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
