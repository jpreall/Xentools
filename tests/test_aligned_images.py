from __future__ import annotations

import io
import zipfile

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest
import tifffile


def _write_rgb_ome(path, shape=(20, 30, 3)):
    y, x = np.indices(shape[:2])
    image = np.stack(
        [(x * 7) % 255, (y * 11) % 255, ((x + y) * 5) % 255], axis=-1
    ).astype(np.uint8)
    tifffile.imwrite(
        path,
        image,
        ome=True,
        metadata={
            "axes": "YXS",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
        },
    )
    return image


def _write_alignment_zip(path, matrix, moving_points=None):
    if moving_points is None:
        moving_points = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    homogeneous = np.column_stack([moving_points, np.ones(len(moving_points))])
    fixed = (matrix @ homogeneous.T).T[:, :2]
    keypoints = pd.DataFrame(
        {
            "fixedX": fixed[:, 0],
            "fixedY": fixed[:, 1],
            "alignmentX": moving_points[:, 0],
            "alignmentY": moving_points[:, 1],
        }
    )
    with zipfile.ZipFile(path, "w") as archive:
        matrix_buffer = io.StringIO()
        np.savetxt(matrix_buffer, matrix, delimiter=",")
        archive.writestr("matrix.csv", matrix_buffer.getvalue())
        archive.writestr("keypoints.csv", keypoints.to_csv(index=False))


def test_load_aligned_image_validates_explorer_keypoints(tmp_path):
    from xentools.io.read.images import load_aligned_image

    image_path = tmp_path / "sample_117451_HE.ome.tif"
    alignment_path = tmp_path / "117451_HE_alignment_files.zip"
    _write_rgb_ome(image_path)
    matrix = np.array([[0, 1.2, 5], [-1.2, 0, 40], [0, 0, 1]], dtype=float)
    _write_alignment_zip(alignment_path, matrix)

    registered = load_aligned_image(
        image_path,
        alignment_path,
        name="H&E",
        fixed_shape=(100, 80),
        fixed_pixel_size=0.25,
    )

    np.testing.assert_allclose(registered.source_to_fixed, matrix)
    assert registered.source_shape == (20, 30)
    assert registered.source_axes == "YXS"
    assert registered.keypoint_count == 3
    assert registered.keypoint_rmse_px == pytest.approx(0)
    assert registered.channel_names


def test_find_alignment_file_matches_leading_zero_sample_ids(tmp_path):
    from xentools.io.read.images import find_alignment_file

    image = tmp_path / "20260803_117451_HE.ome.tif"
    image.touch()
    xenium = tmp_path / "output-XETG__0117451__Region_1"
    xenium.mkdir()
    match = tmp_path / "117451_HE_alignment_files.zip"
    other = tmp_path / "117469_HE_alignment_files.zip"
    match.touch()
    other.touch()
    assert find_alignment_file(image, xenium) == str(match)


def test_load_aligned_image_accepts_extracted_alignment_folder(tmp_path):
    from xentools.io.read.images import load_aligned_image

    image_path = tmp_path / "sample_117451_HE.ome.tif"
    alignment_path = tmp_path / "117451_HE_alignment_files"
    alignment_path.mkdir()
    _write_rgb_ome(image_path)
    matrix = np.array([[1, 0, 3], [0, 1, 7], [0, 0, 1]], dtype=float)
    np.savetxt(alignment_path / "matrix.csv", matrix, delimiter=",")

    registered = load_aligned_image(
        image_path,
        alignment_path,
        name="H&E",
        fixed_shape=(100, 80),
        fixed_pixel_size=0.25,
    )
    np.testing.assert_allclose(registered.source_to_fixed, matrix)


def test_load_aligned_image_accepts_keypoints_csv_as_bundle_entry(tmp_path):
    from xentools.io.read.images import load_aligned_image

    image_path = tmp_path / "sample_117451_HE.ome.tif"
    alignment_path = tmp_path / "117451_HE_alignment_files"
    alignment_path.mkdir()
    _write_rgb_ome(image_path)
    matrix = np.array([[1, 0, 3], [0, 1, 7], [0, 0, 1]], dtype=float)
    np.savetxt(alignment_path / "matrix.csv", matrix, delimiter=",")
    moving = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    fixed = (matrix @ np.column_stack([moving, np.ones(3)]).T).T[:, :2]
    pd.DataFrame(
        {
            "fixedX": fixed[:, 0],
            "fixedY": fixed[:, 1],
            "alignmentX": moving[:, 0],
            "alignmentY": moving[:, 1],
        }
    ).to_csv(alignment_path / "keypoints.csv", index=False)

    registered = load_aligned_image(
        image_path,
        alignment_path / "keypoints.csv",
        name="H&E",
        fixed_shape=(100, 80),
        fixed_pixel_size=0.25,
    )
    np.testing.assert_allclose(registered.source_to_fixed, matrix)
    assert registered.keypoint_count == 3
    assert registered.keypoint_rmse_px == pytest.approx(0)


def test_keypoints_csv_without_matrix_has_helpful_error(tmp_path):
    from xentools.io.read.images import _read_alignment_bundle

    keypoints = tmp_path / "keypoints.csv"
    pd.DataFrame(
        columns=["fixedX", "fixedY", "alignmentX", "alignmentY"]
    ).to_csv(keypoints, index=False)
    with pytest.raises(ValueError, match="sibling matrix.csv was not found"):
        _read_alignment_bundle(keypoints)


def test_find_alignment_file_accepts_extracted_folder(tmp_path):
    from xentools.io.read.images import find_alignment_file

    image = tmp_path / "20260803_117451_HE.ome.tif"
    image.touch()
    xenium = tmp_path / "output-XETG__0117451__Region_1"
    xenium.mkdir()
    match = tmp_path / "117451_HE_alignment_files"
    match.mkdir()
    np.savetxt(match / "matrix.csv", np.eye(3), delimiter=",")
    assert find_alignment_file(image, xenium) == str(match)


def test_show_aligned_image_reads_rgb_region_and_uses_micron_coordinates(tmp_path):
    import xentools
    from xentools.io.read.images import load_aligned_image

    image_path = tmp_path / "image.ome.tif"
    alignment_path = tmp_path / "alignment.zip"
    _write_rgb_ome(image_path)
    matrix = np.eye(3)
    _write_alignment_zip(alignment_path, matrix)
    registered = load_aligned_image(
        image_path,
        alignment_path,
        name="H&E",
        fixed_shape=(20, 30),
        fixed_pixel_size=1.0,
    )

    ax = xentools.show_aligned_image(registered, verbose=False)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape == (20, 30, 3)
    assert tuple(ax.get_xlim()) == pytest.approx((0, 30))
    assert tuple(ax.get_ylim()) == pytest.approx((20, 0))


def test_aligned_image_crop_does_not_change_global_transform(tmp_path):
    import xentools
    from xentools.io.read.images import load_aligned_image

    image_path = tmp_path / "image.ome.tif"
    alignment_path = tmp_path / "alignment.zip"
    _write_rgb_ome(image_path)
    _write_alignment_zip(alignment_path, np.eye(3))
    registered = load_aligned_image(
        image_path,
        alignment_path,
        name="H&E",
        fixed_shape=(20, 30),
        fixed_pixel_size=1.0,
    )
    before = registered.source_to_global.copy()
    ax = xentools.show_aligned_image(
        registered, bounds=(5, 15, 4, 12), level=0, verbose=False
    )

    np.testing.assert_allclose(registered.source_to_global, before)
    assert tuple(ax.get_xlim()) == pytest.approx((5, 15))
    assert tuple(ax.get_ylim()) == pytest.approx((12, 4))


def test_xendata_import_aligned_image_registers_named_layer(xdata, tmp_path):
    image_path = tmp_path / "20260803_117451_HE.ome.tif"
    alignment_path = tmp_path / "117451_HE_alignment_files.zip"
    _write_rgb_ome(image_path)
    _write_alignment_zip(alignment_path, np.eye(3))

    registered = xdata.import_aligned_image(
        image_path,
        alignment_file=alignment_path,
        name="test H&E",
    )

    assert xdata.images["test H&E"] is registered
    np.testing.assert_allclose(
        xdata.transforms.get("image:test H&E"), registered.source_to_global
    )
    assert "AlignedImage" in repr(registered)
    with pytest.raises(ValueError, match="already registered"):
        xdata.import_aligned_image(
            image_path,
            alignment_file=alignment_path,
            name="test H&E",
        )
