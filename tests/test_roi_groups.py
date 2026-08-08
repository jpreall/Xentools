from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from xentools.core.rois import ROI, ROICollection


def _collection():
    collection = ROICollection()
    collection.add(ROI.from_geometry(box(0, 0, 2, 2), name="liver_a"))
    collection.add(ROI.from_geometry(box(8, 0, 10, 2), name="liver_b"))
    return collection


def test_roi_group_preserves_members_and_exposes_union():
    collection = _collection()
    group = collection.create_group(
        "liver",
        ["liver_a", "liver_b"],
        metadata={"tissue": "liver"},
    )

    assert group.member_names == ["liver_a", "liver_b"]
    assert collection.group_names() == ["liver"]
    assert collection["liver"] is group
    assert collection.resolve("liver").geometry.geom_type == "MultiPolygon"
    np.testing.assert_array_equal(
        group.contains_points(np.array([1, 5, 9]), np.array([1, 1, 1])),
        [True, False, True],
    )
    assert collection.summary()["groups"] == {"liver": ["liver_a", "liver_b"]}


def test_roi_collection_repr_is_compact_and_shows_unclassified_rois():
    collection = _collection()
    collection.create_group("all_liver", ["liver_a", "liver_b"])

    text = repr(collection)
    assert "2 selections · 0 classified · 2 unclassified" in text
    assert "liver_a" in text
    assert "all_liver" in text
    assert "MultiPolygon" in text
    assert "Unclassified" in text
    assert "POLYGON ((" not in text

    html = collection._repr_html_()
    assert "<strong>ROICollection</strong>" in html
    assert "Unclassified" in html
    assert "all_liver" in html


def test_roi_collection_repr_shows_classes_and_limits_long_lists():
    collection = ROICollection()
    for i in range(15):
        collection.add(
            ROI.from_geometry(
                box(i * 2, 0, i * 2 + 1, 1),
                name=f"piece_{i}",
                class_name="liver" if i < 3 else None,
            )
        )
    text = repr(collection)
    assert "3 classified · 12 unclassified" in text
    assert "liver" in text
    assert "… 3 more" in text


def test_multipolygon_plot_draws_every_component():
    merged = _collection().create_group("liver", ["liver_a", "liver_b"]).union
    _, ax = plt.subplots()
    merged.plot(ax)
    assert len(ax.patches) == 2


def test_roi_plot_uses_native_global_coordinates():
    roi = ROI.from_geometry(box(10, 20, 30, 40), name="test")
    _, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    roi.plot(ax)

    plotted = ax.patches[0].get_xy()
    assert plotted[:, 1].min() == pytest.approx(20)
    assert plotted[:, 1].max() == pytest.approx(40)


def test_roi_collection_plot_can_label_selections():
    collection = _collection()
    _, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    collection.plot(
        ax,
        labels=True,
        edgecolor="cyan",
        label_kwargs={"fontsize": 12},
    )

    assert [text.get_text() for text in ax.texts] == ["liver_a", "liver_b"]
    assert all(text.get_color() == "cyan" for text in ax.texts)
    assert all(text.get_fontsize() == 12 for text in ax.texts)
    assert ax.texts[0].get_position() == pytest.approx((1, 1))


def test_roi_collection_plot_labels_each_group_once():
    collection = _collection()
    collection.create_group("all_liver", ["liver_a", "liver_b"])
    _, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    collection.plot(ax, level="group", labels=True)

    assert [text.get_text() for text in ax.texts] == ["all_liver"]


def test_group_geojson_preserves_multipolygon_and_members():
    collection = _collection()
    collection.create_group("liver", ["liver_a", "liver_b"])
    feature_collection = collection.to_geojson_feature_collection(level="group")
    feature = feature_collection["features"][0]
    assert feature["geometry"]["type"] == "MultiPolygon"
    assert feature["properties"]["members"] == ["liver_a", "liver_b"]


def test_lazy_transcript_query_applies_exact_group_geometry():
    merged = _collection().create_group("liver", ["liver_a", "liver_b"]).union

    class FakeLazyTranscripts:
        def query(self, **kwargs):
            return pd.DataFrame(
                {
                    "x_location": [1.0, 5.0, 9.0],
                    "y_location": [1.0, 1.0, 1.0],
                    "feature_name": ["A", "A", "A"],
                }
            )

    result = merged.query_lazy_transcripts(FakeLazyTranscripts())
    assert result["x_location"].tolist() == [1.0, 9.0]


def test_group_creation_rejects_empty_and_conflicting_names():
    collection = _collection()
    with pytest.raises(ValueError, match="at least one ROI"):
        collection.create_group("empty", [])
    with pytest.raises(ValueError, match="conflicts"):
        collection.create_group("liver_a", ["liver_b"])


def test_xendata_can_crop_to_named_roi_group(xdata):
    from xentools import XenData

    # Use a fresh object because other session-scoped tests may intentionally
    # attach non-copyable interactive resources (for example an open HDF5
    # gene-set picker) to the shared fixture.
    data = XenData(xdata.xenium_folder, verbose=False, lazy_boundaries=True)
    coords = np.asarray(data.adata.obsm["spatial"], dtype=float)
    first = coords[0]
    last = coords[-1]
    radius = 0.01
    data.rois.add(
        ROI.from_geometry(
            box(first[0] - radius, first[1] - radius, first[0] + radius, first[1] + radius),
            name="group_crop_a",
        )
    )
    data.rois.add(
        ROI.from_geometry(
            box(last[0] - radius, last[1] - radius, last[0] + radius, last[1] + radius),
            name="group_crop_b",
        )
    )
    group = data.create_roi_group("group_crop", ["group_crop_a", "group_crop_b"])

    cropped = data.subset_to_roi("group_crop", inplace=False)

    assert group.member_names == ["group_crop_a", "group_crop_b"]
    assert cropped.subset_roi.name == "group_crop"
    assert cropped.subset_roi.geometry.geom_type in {"Polygon", "MultiPolygon"}
    assert 1 <= cropped.adata.n_obs <= 2
    assert data.rois["group_crop"] is group
    assert not hasattr(data, "ROIs")
