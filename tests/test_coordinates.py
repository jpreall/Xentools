from __future__ import annotations

import numpy as np
import pytest


def test_default_global_coordinate_system_is_platform_neutral():
    from xentools import TransformRegistry

    registry = TransformRegistry()
    system = registry.systems["global"]
    assert system.name == "global"
    assert system.axes == ("x", "y")
    assert system.units == ("micrometer", "micrometer")
    assert system.y_direction == "down"


def test_transform_registry_round_trip():
    from xentools import TransformRegistry

    registry = TransformRegistry()
    matrix = np.array([[2, 0, 10], [0, 3, 20], [0, 0, 1]], dtype=float)
    registry.register("image:test", matrix, source="pixels:test")
    points = np.array([[0, 0], [4, 5]], dtype=float)
    mapped = registry.apply("image:test", points)
    np.testing.assert_allclose(mapped, [[10, 20], [18, 35]])
    np.testing.assert_allclose(registry.apply("image:test", mapped, inverse=True), points)

    with pytest.raises(ValueError, match="already exists"):
        registry.register("image:test", matrix)


def test_xendata_registers_native_elements_and_dapi(xdata):
    expected = {
        "points:transcripts",
        "points:cells",
        "shapes:cells",
        "shapes:nuclei",
        "shapes:rois",
        "image:DAPI",
    }
    assert expected.issubset(set(xdata.transforms._transforms))
    dapi = xdata.transforms.get("image:DAPI")
    assert dapi[0, 0] == pytest.approx(xdata.pixel_size)
    assert dapi[1, 1] == pytest.approx(xdata.pixel_size)
