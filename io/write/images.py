"""Image export and crop helpers for xentools."""

from __future__ import annotations

import os
import re
import shutil
import sys
import importlib.util
import uuid

import numpy as np


def _load_local_module(module_name, relative_path):
    """Load a sibling xentools module by file path when imported outside a package."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = os.path.join(os.path.dirname(__file__), relative_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load local module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from ...core.rois import _shapely_bounds
except ImportError:
    _shapely_bounds = _load_local_module(
        "_xentools_core_rois",
        os.path.join("..", "..", "core", "rois.py"),
    )._shapely_bounds

try:
    from ..read.images import _source_series_level_arrays
except ImportError:
    _source_series_level_arrays = _load_local_module(
        "_xentools_io_read_images",
        os.path.join("..", "read", "images.py"),
    )._source_series_level_arrays

__all__ = [
    "_roi_bounds_in_pixels",
    "_pixel_aligned_bounds_um",
    "_scale_bounds_for_level",
    "_cropped_shape_from_bounds",
    "_iter_cropped_tiles",
    "_build_pyramid_levels",
    "_write_pyramidal_ome_tiff",
    "_write_pyramidal_ome_tiff_from_levels",
    "_linked_ome_xml",
    "_write_linked_pyramidal_ome_tiffs",
    "_write_linked_pyramidal_ome_tiffs_from_levels",
    "_write_morphology_for_slice",
    "_write_protein_images_for_slice",
    "_write_ome_tiff",
    "write_ome_tiff",
]


def _roi_bounds_in_pixels(roi_geometry, pixel_size, image_shape):
    minx, miny, maxx, maxy = _shapely_bounds(roi_geometry)
    x0 = max(0, int(np.floor(minx / pixel_size)))
    y0 = max(0, int(np.floor(miny / pixel_size)))
    x1 = min(int(image_shape[-1]), int(np.ceil(maxx / pixel_size)))
    y1 = min(int(image_shape[-2]), int(np.ceil(maxy / pixel_size)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI bounding box does not overlap the morphology image.")
    return x0, x1, y0, y1


def _pixel_aligned_bounds_um(bounds, pixel_size):
    """Snap micron bounds to the pixel grid used for image crops."""
    xmin, xmax, ymin, ymax = map(float, bounds)
    x0 = np.floor(xmin / pixel_size) * pixel_size
    y0 = np.floor(ymin / pixel_size) * pixel_size
    x1 = np.ceil(xmax / pixel_size) * pixel_size
    y1 = np.ceil(ymax / pixel_size) * pixel_size
    return (x0, x1, y0, y1)


def _scale_bounds_for_level(bounds, level):
    x0, x1, y0, y1 = bounds
    scale = 2 ** level
    return (
        int(np.floor(x0 / scale)),
        int(np.ceil(x1 / scale)),
        int(np.floor(y0 / scale)),
        int(np.ceil(y1 / scale)),
    )


def _cropped_shape_from_bounds(source_array, bounds):
    x0, x1, y0, y1 = bounds
    if source_array.ndim == 2:
        return (y1 - y0, x1 - x0)
    if source_array.ndim == 3:
        return (source_array.shape[0], y1 - y0, x1 - x0)
    raise ValueError(f"Unsupported source array ndim: {source_array.ndim}")


def _iter_cropped_tiles(source_array, bounds, tile=(1024, 1024)):
    x0, x1, y0, y1 = bounds
    tile_y, tile_x = tile
    for y in range(y0, y1, tile_y):
        for x in range(x0, x1, tile_x):
            ys = slice(y, min(y + tile_y, y1))
            xs = slice(x, min(x + tile_x, x1))
            if source_array.ndim == 2:
                yield np.asarray(source_array[ys, xs])
            elif source_array.ndim == 3:
                yield np.asarray(source_array[:, ys, xs])
            else:
                raise ValueError(f"Unsupported source array ndim: {source_array.ndim}")


def _build_pyramid_levels(image, scale_factor=2):
    if scale_factor < 2:
        raise ValueError("scale_factor must be >= 2 for pyramid generation.")

    levels = []
    current = image
    while min(current.shape[-2:]) > 1:
        next_level = current[:, ::scale_factor, ::scale_factor]
        if next_level.shape[-2:] == current.shape[-2:]:
            break
        levels.append(next_level)
        current = next_level
    return levels


def _write_pyramidal_ome_tiff(image,
                              output_path,
                              pixel_size,
                              tile=(1024, 1024),
                              pyramid_scale=2,
                              compression="jpeg2000"):
    import tifffile

    pyramid_levels = _build_pyramid_levels(image, scale_factor=pyramid_scale)
    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        tif.write(
            image,
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeY": pixel_size,
            },
            tile=tile,
            compression=compression,
            subifds=len(pyramid_levels),
        )
        for level in pyramid_levels:
            tif.write(
                level,
                tile=tile,
                compression=compression,
                subfiletype=1,
            )


def _write_pyramidal_ome_tiff_from_levels(level_arrays,
                                          output_path,
                                          base_bounds,
                                          pixel_size,
                                          tile=(1024, 1024),
                                          compression="jpeg2000"):
    import tifffile

    if not level_arrays:
        raise ValueError("level_arrays must not be empty.")

    cropped_levels = []
    for level, level_array in enumerate(level_arrays):
        level_bounds = _scale_bounds_for_level(base_bounds, level)
        cropped_levels.append(
            np.asarray(level_array[:, level_bounds[2]:level_bounds[3], level_bounds[0]:level_bounds[1]])
        )

    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        tif.write(
            cropped_levels[0],
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeY": pixel_size,
            },
            tile=tile,
            compression=compression,
            subifds=max(0, len(cropped_levels) - 1),
        )
        for level in cropped_levels[1:]:
            tif.write(
                level,
                tile=tile,
                compression=compression,
                subfiletype=1,
            )


def _linked_ome_xml(channel_names,
                    filenames,
                    file_uuids,
                    size_x,
                    size_y,
                    pixel_size,
                    root_uuid,
                    template_xml=None,
                    dtype_name="uint16"):
    if template_xml is not None:
        xml_str = template_xml
        xml_str = re.sub(r'UUID="urn:uuid:[^"]+"', f'UUID="{root_uuid}"', xml_str, count=1)
        xml_str = re.sub(r'Type="[^"]+"', f'Type="{dtype_name}"', xml_str, count=1)
        xml_str = re.sub(r'SizeX="[^"]+"', f'SizeX="{size_x}"', xml_str, count=1)
        xml_str = re.sub(r'SizeY="[^"]+"', f'SizeY="{size_y}"', xml_str, count=1)
        xml_str = re.sub(r'SizeZ="[^"]+"', 'SizeZ="1"', xml_str, count=1)
        xml_str = re.sub(r'SizeC="[^"]+"', f'SizeC="{len(channel_names)}"', xml_str, count=1)
        xml_str = re.sub(r'SizeT="[^"]+"', 'SizeT="1"', xml_str, count=1)
        xml_str = re.sub(r'PhysicalSizeX="[^"]+"', f'PhysicalSizeX="{pixel_size}"', xml_str, count=1)
        xml_str = re.sub(r'PhysicalSizeY="[^"]+"', f'PhysicalSizeY="{pixel_size}"', xml_str, count=1)
        xml_str = xml_str.replace('PhysicalSizeXUnit="µm"', 'PhysicalSizeXUnit="&#181;m"')
        xml_str = xml_str.replace('PhysicalSizeYUnit="µm"', 'PhysicalSizeYUnit="&#181;m"')
        xml_str = xml_str.replace('WellOriginXUnit="µm"', 'WellOriginXUnit="&#181;m"')
        xml_str = xml_str.replace('WellOriginYUnit="µm"', 'WellOriginYUnit="&#181;m"')

        for i, name in enumerate(channel_names):
            xml_str = re.sub(
                rf'(<Channel[^>]*ID="Channel:{i}"[^>]*Name=")[^"]+(")',
                rf'\g<1>{name}\2',
                xml_str,
                count=1,
            )

        uuid_pattern = re.compile(r'(<UUID FileName=")([^"]+)(">)(urn:uuid:[^<]+)(</UUID>)')
        replacements = iter(zip(filenames, file_uuids))

        def _replace_uuid(match):
            try:
                filename, file_uuid = next(replacements)
            except StopIteration:
                return match.group(0)
            return f'{match.group(1)}{filename}{match.group(3)}{file_uuid}{match.group(5)}'

        xml_str = uuid_pattern.sub(_replace_uuid, xml_str)

        return xml_str.encode("ascii", "xmlcharrefreplace").decode("ascii")

    channel_entries = "\n".join(
        f'      <Channel ID="Channel:{i}" Name="{name}" SamplesPerPixel="1"/>'
        for i, name in enumerate(channel_names)
    )
    tiffdata_entries = "\n".join(
        f'      <TiffData FirstZ="0" FirstT="0" FirstC="{i}" PlaneCount="1"><UUID FileName="{filenames[i]}">{file_uuids[i]}</UUID></TiffData>'
        for i in range(len(channel_names))
    )
    xml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" UUID="{root_uuid}">
  <Instrument ID="Instrument:0">
    <Microscope Manufacturer="10x Genomics" Model="Xenium"/>
  </Instrument>
  <Image ID="Image:0">
    <InstrumentRef ID="Instrument:0"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="{dtype_name}" SizeX="{size_x}" SizeY="{size_y}" SizeZ="1" SizeC="{len(channel_names)}" SizeT="1" PhysicalSizeX="{pixel_size}" PhysicalSizeXUnit="um" PhysicalSizeY="{pixel_size}" PhysicalSizeYUnit="um">
{channel_entries}
{tiffdata_entries}
    </Pixels>
  </Image>
</OME>'''
    return xml_str.encode("ascii", "xmlcharrefreplace").decode("ascii")


def _write_linked_pyramidal_ome_tiffs(image,
                                      output_dir,
                                      filenames,
                                      channel_names,
                                      pixel_size,
                                      template_xml=None,
                                      tile=(1024, 1024),
                                      pyramid_scale=2,
                                      compression="jpeg2000"):
    import tifffile
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image.ndim != 3:
        raise ValueError("Expected protein image array with shape (C, Y, X).")
    if image.shape[0] != len(channel_names) or image.shape[0] != len(filenames):
        raise ValueError("Protein image channel count does not match filenames/channel names.")

    file_uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in filenames]

    for idx, filename in enumerate(filenames):
        channel_image = image[idx]
        pyramid_levels = _build_pyramid_levels(channel_image[np.newaxis, ...], scale_factor=pyramid_scale)
        ome_xml = _linked_ome_xml(
            channel_names=channel_names,
            filenames=filenames,
            file_uuids=file_uuids,
            size_x=image.shape[2],
            size_y=image.shape[1],
            pixel_size=pixel_size,
            root_uuid=file_uuids[idx],
            template_xml=template_xml,
            dtype_name=image.dtype.name,
        )
        with tifffile.TiffWriter(output_dir / filename, bigtiff=True) as tif:
            tif.write(
                channel_image,
                description=ome_xml,
                tile=tile,
                compression=compression,
                subifds=len(pyramid_levels),
            )
            for level in pyramid_levels:
                tif.write(
                    level[0],
                    tile=tile,
                    compression=compression,
                    subfiletype=1,
                )


def _write_linked_pyramidal_ome_tiffs_from_levels(level_arrays_by_channel,
                                                  output_dir,
                                                  filenames,
                                                  channel_names,
                                                  pixel_size,
                                                  base_bounds,
                                                  template_xml=None,
                                                  tile=(1024, 1024),
                                                  compression="jpeg2000"):
    import tifffile
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(level_arrays_by_channel) != len(channel_names) or len(level_arrays_by_channel) != len(filenames):
        raise ValueError("Protein channel count does not match level arrays.")

    file_uuids = [f"urn:uuid:{uuid.uuid4()}" for _ in filenames]
    n_levels = len(level_arrays_by_channel[0])

    for idx, filename in enumerate(filenames):
        channel_levels = level_arrays_by_channel[idx]
        if len(channel_levels) != n_levels:
            raise ValueError("Protein channel pyramid depth mismatch.")
        ome_xml = _linked_ome_xml(
            channel_names=channel_names,
            filenames=filenames,
            file_uuids=file_uuids,
            size_x=base_bounds[1] - base_bounds[0],
            size_y=base_bounds[3] - base_bounds[2],
            pixel_size=pixel_size,
            root_uuid=file_uuids[idx],
            template_xml=template_xml,
            dtype_name=channel_levels[0].dtype.name,
        )
        with tifffile.TiffWriter(output_dir / filename, bigtiff=True) as tif:
            tif.write(
                data=_iter_cropped_tiles(channel_levels[0], base_bounds, tile=tile),
                shape=_cropped_shape_from_bounds(channel_levels[0], base_bounds),
                dtype=channel_levels[0].dtype,
                description=ome_xml,
                metadata=None,
                tile=tile,
                compression=compression,
                subifds=max(0, n_levels - 1),
            )
            for level, level_array in enumerate(channel_levels[1:], start=1):
                level_bounds = _scale_bounds_for_level(base_bounds, level)
                tif.write(
                    data=_iter_cropped_tiles(level_array, level_bounds, tile=tile),
                    shape=_cropped_shape_from_bounds(level_array, level_bounds),
                    dtype=level_array.dtype,
                    metadata=None,
                    tile=tile,
                    compression=compression,
                    subfiletype=1,
                )


def _write_morphology_for_slice(xdata,
                                output_path,
                                crop=True,
                                pyramidal=True,
                                pyramid_scale=2,
                                tile=(1024, 1024)):
    import tifffile

    morphology_path = os.path.join(xdata.xenium_folder, "morphology.ome.tif")
    if not os.path.exists(morphology_path):
        raise FileNotFoundError(f"Morphology image not found: {morphology_path}")

    subset_roi = getattr(xdata, "subset_roi", None)
    if not crop or subset_roi is None:
        shutil.copy2(morphology_path, output_path)
        return

    with tifffile.TiffFile(morphology_path) as tif:
        series = tif.series[0]
        level_arrays, zarr_store = _source_series_level_arrays(series)
        try:
            base_bounds = _roi_bounds_in_pixels(subset_roi, xdata.pixel_size, level_arrays[0].shape)
            if pyramidal:
                _write_pyramidal_ome_tiff_from_levels(
                    level_arrays,
                    output_path=output_path,
                    base_bounds=base_bounds,
                    pixel_size=xdata.pixel_size,
                    tile=tile,
                )
            else:
                cropped = np.asarray(level_arrays[0][:, base_bounds[2]:base_bounds[3], base_bounds[0]:base_bounds[1]])
                tifffile.imwrite(
                    output_path,
                    cropped,
                    ome=True,
                    metadata={
                        "axes": "ZYX",
                        "PhysicalSizeX": xdata.pixel_size,
                        "PhysicalSizeY": xdata.pixel_size,
                    },
                )
        finally:
            zarr_store.close()


def _write_protein_images_for_slice(xdata,
                                    output_dir,
                                    crop=True,
                                    pyramid_scale=2,
                                    tile=(1024, 1024)):
    import tifffile
    import zarr
    from pathlib import Path

    protein_info = xdata.protein_images
    if protein_info is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_roi = getattr(xdata, "subset_roi", None)
    if not crop or subset_roi is None:
        for src, name in zip(protein_info["files"], protein_info["filenames"]):
            shutil.copy2(src, output_dir / name)
        return

    level_arrays_by_channel = []
    base_bounds = None
    for src in protein_info["files"]:
        with tifffile.TiffFile(src) as tif:
            page = tif.pages[0]
            levels = [zarr.open(page.aszarr(), mode="r")]
            if hasattr(page, "pages") and page.pages is not None:
                for subpage in page.pages:
                    levels.append(zarr.open(subpage.aszarr(), mode="r"))
            level_arrays_by_channel.append(levels)
            if base_bounds is None:
                base_bounds = _roi_bounds_in_pixels(subset_roi, xdata.pixel_size, levels[0].shape)

    _write_linked_pyramidal_ome_tiffs_from_levels(
        level_arrays_by_channel,
        output_dir=output_dir,
        filenames=protein_info["filenames"],
        channel_names=protein_info["channel_names"],
        pixel_size=xdata.pixel_size,
        base_bounds=base_bounds,
        template_xml=protein_info.get("ome_xml_template"),
        tile=tile,
    )


def _write_ome_tiff(image_array,
                    channel_names,
                    channel_ids=None,
                    channel_colors=None,
                    physical_size_x=5,
                    physical_size_y=5,
                    significant_bits=12,
                    compression="zlib",
                    output_path=None,
                    pyramidal: bool = True,
                    pyramid_scale: int = 2,
                    tile: tuple = (1024, 1024)):
    import tifffile

    y_size, x_size, n_channels = image_array.shape
    if len(channel_names) != n_channels:
        raise ValueError("Length of channel_names must equal the number of channels in image_array.")
    if channel_ids is not None and len(channel_ids) != n_channels:
        raise ValueError("Length of channel_ids must equal the number of channels in image_array.")
    if channel_colors is not None and len(channel_colors) != n_channels:
        raise ValueError("Length of channel_colors must equal the number of channels in image_array.")

    if channel_ids is None:
        channel_ids = [f"Channel:{i}" for i in range(n_channels)]

    arr = image_array.transpose(2, 0, 1)[np.newaxis, :, np.newaxis, :, :]

    channel_entries = []
    channel_maxes = image_array.reshape(-1, image_array.shape[2]).max(axis=0)
    for i, name in enumerate(channel_names):
        max_val = float(channel_maxes[i])
        color_attr = f' Color="{channel_colors[i]}"' if channel_colors is not None else ""
        entry = f'''      <Channel ID="{channel_ids[i]}" Name="{name}" SamplesPerPixel="1"{color_attr}>
            <DisplaySettings>
            <DisplayRangeMin>0</DisplayRangeMin>
            <DisplayRangeMax>{max_val}</DisplayRangeMax>
            </DisplaySettings>
        </Channel>'''
        channel_entries.append(entry)
    channel_entries_str = "\n".join(channel_entries)

    ome_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
    <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" ID="Pixels:0" Type="{image_array.dtype.name}"
            SizeX="{x_size}"
            SizeY="{y_size}"
            SizeC="{n_channels}"
            SizeZ="1"
            SizeT="1"
            PhysicalSizeX="{physical_size_x}"
            PhysicalSizeY="{physical_size_y}"
            SignificantBits="{significant_bits}">
            {channel_entries_str}
    </Pixels>
    </Image>
    </OME>'''

    if output_path is None:
        output_path = f"multilayer_{physical_size_x}um.ome.tiff"

    if pyramidal:
        arr_cyx = image_array.transpose(2, 0, 1)
        pyramid_levels = _build_pyramid_levels(arr_cyx, scale_factor=pyramid_scale)

        def _to_tczyx(a_cyx):
            return a_cyx[np.newaxis, :, np.newaxis, :, :]

        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            tif.write(
                _to_tczyx(arr_cyx),
                photometric="minisblack",
                compression=compression,
                description=ome_xml,
                metadata=None,
                tile=tile,
                subifds=len(pyramid_levels),
            )
            for level in pyramid_levels:
                tif.write(
                    _to_tczyx(level),
                    photometric="minisblack",
                    compression=compression,
                    metadata=None,
                    tile=tile,
                    subfiletype=1,
                )
    else:
        tifffile.imwrite(
            output_path,
            arr,
            photometric="minisblack",
            compression=compression,
            description=ome_xml,
            metadata=None,
        )


def _create_multilayer_image_from_binned(xdata, genes):
    if not hasattr(xdata, "binned_adata"):
        raise AttributeError(
            "No binned AnnData found. Run create_binned_adata() before write_ome_tiff()."
        )

    data = xdata.binned_adata
    if "spatial" not in data.obsm:
        raise ValueError("binned_adata must contain spatial coordinates in obsm['spatial'].")

    if genes is None:
        genes = data.var_names.tolist()
    elif isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(genes)

    missing_genes = [gene for gene in genes if gene not in data.var_names]
    if missing_genes:
        raise ValueError(
            "The following genes are not present in the binned AnnData object: "
            + ", ".join(missing_genes)
        )

    coords = data.obsm["spatial"]
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x_min = x_coords.min()
    y_min = y_coords.min()
    x_idx = (x_coords - x_min).astype(int)
    y_idx = (y_coords - y_min).astype(int)
    width = int(x_coords.max() - x_min) + 1
    height = int(y_coords.max() - y_min) + 1
    image = np.zeros((height, width, len(genes)), dtype=np.uint16)

    matrix = data[:, genes].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    if np.max(matrix) > 255:
        matrix = np.clip(matrix, 0, 2**16 - 1).astype(np.uint16)
    else:
        matrix = np.clip(matrix, 0, 255).astype(np.uint8)

    image[y_idx, x_idx, :] = matrix
    return image[::-1, :, :], genes


def write_ome_tiff(
    xdata,
    genes,
    output_path,
    flip_y=False,
    compression="zlib",
    pyramidal: bool = True,
    pyramid_scale: int = 2,
    tile: tuple = (1024, 1024),
):
    """
    Write a multi-layer OME-TIFF image from ``xdata.binned_adata``.
    """
    if not hasattr(xdata, "binned_adata"):
        raise AttributeError(
            "No binned AnnData found. Run create_binned_adata() before write_ome_tiff()."
        )

    output_path = os.fspath(output_path)
    if not output_path.endswith(".ome.tiff"):
        raise ValueError("Output path must end with .ome.tiff")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    bin_size = xdata.binned_adata.uns["bin_size"]
    image, genes = _create_multilayer_image_from_binned(xdata, genes)
    if flip_y:
        image = image[::-1, :, :]

    _write_ome_tiff(
        image,
        channel_names=genes,
        compression=compression,
        physical_size_x=bin_size,
        physical_size_y=bin_size,
        output_path=output_path,
        pyramidal=pyramidal,
        pyramid_scale=pyramid_scale,
        tile=tile,
    )
