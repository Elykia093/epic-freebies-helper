from types import SimpleNamespace

import pytest
from hcaptcha_challenger.models import PointCoordinate, SpatialPath

from extensions.spatial_coordinates import (
    same_bounds,
    to_page_paths,
    to_page_point,
    unique_points,
    uses_normalized_coordinates,
)


BBOX = {"x": 710.0, "y": 365.0, "width": 500.0, "height": 470.0}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("Qwen/Qwen3-VL-32B-Instruct", True),
        ("Qwen3-VL-8B-Thinking", True),
        ("provider/qWeN3-vL-30B-A3B-Instruct", True),
        ("Qwen/Qwen3.5-122B-A10B", False),
        ("Qwen/Qwen2.5-VL-32B-Instruct", False),
        ("zai-org/GLM-4.5V", False),
        ("provider/custom-qwen3-vl-model", False),
        ("", False),
    ],
)
def test_normalized_model_selection_is_limited_to_qwen3_vl(model, expected):
    assert uses_normalized_coordinates(model) is expected


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [((0, 0), (710, 365)), ((500, 500), (960, 600)), ((1000, 1000), (1210, 835))],
)
def test_to_page_point_maps_normalized_image_coordinates(coordinates, expected):
    point = PointCoordinate(x=coordinates[0], y=coordinates[1])
    result = to_page_point(point, BBOX)

    assert (result.x, result.y) == expected
    assert (point.x, point.y) == coordinates


def test_to_page_point_preserves_fractional_bbox_until_final_rounding():
    bbox = {"x": 710.25, "y": 365.3, "width": 500.5, "height": 470.5}

    result = to_page_point(PointCoordinate(x=250, y=750), bbox)

    assert (result.x, result.y) == (835, 718)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("value", [-1, 1001, float("nan"), float("inf"), -float("inf")])
def test_to_page_point_rejects_invalid_original_coordinates(axis, value):
    coordinates = {"x": 500, "y": 500, axis: value}

    with pytest.raises(ValueError):
        to_page_point(SimpleNamespace(**coordinates), BBOX)


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        {},
        {**BBOX, "width": 0},
        {**BBOX, "height": -1},
        {**BBOX, "x": float("nan")},
        {**BBOX, "y": float("inf")},
        {**BBOX, "width": float("inf")},
        {**BBOX, "height": float("nan")},
    ],
)
def test_to_page_point_rejects_invalid_bounds(bbox):
    with pytest.raises(ValueError):
        to_page_point(PointCoordinate(x=500, y=500), bbox)


def test_to_page_point_rejects_overflow_in_projected_coordinates():
    bbox = {"x": 1e308, "y": 0, "width": 1e308, "height": 1}

    with pytest.raises(ValueError, match="page.x"):
        to_page_point(PointCoordinate(x=1000, y=500), bbox)


def test_to_page_paths_converts_both_ends_without_changing_input():
    path = SpatialPath(
        start_point=PointCoordinate(x=0, y=500),
        end_point=PointCoordinate(x=750, y=1000),
    )

    result = to_page_paths([path], BBOX)

    assert result == [
        SpatialPath(
            start_point=PointCoordinate(x=710, y=600),
            end_point=PointCoordinate(x=1085, y=835),
        )
    ]
    assert path.start_point == PointCoordinate(x=0, y=500)
    assert path.end_point == PointCoordinate(x=750, y=1000)
    assert to_page_paths([], BBOX) == []


def test_to_page_paths_rejects_invalid_endpoint():
    path = SpatialPath(
        start_point=PointCoordinate(x=500, y=500),
        end_point=PointCoordinate(x=1001, y=0),
    )

    with pytest.raises(ValueError, match="0..1000"):
        to_page_paths([path], BBOX)


def test_unique_points_keeps_first_occurrence_and_exact_order():
    first = PointCoordinate(x=844, y=580)
    second = PointCoordinate(x=844, y=780)
    nearby = PointCoordinate(x=845, y=580)
    points = [first, second, PointCoordinate(x=844, y=580), second, nearby]

    result = unique_points(points)

    assert result == [first, second, nearby]
    assert result[0] is first
    assert len(points) == 5
    assert unique_points([]) == []


def test_same_bounds_accepts_small_position_and_size_differences():
    assert same_bounds(BBOX, dict(BBOX))
    assert same_bounds(BBOX, {name: value + 1 for name, value in BBOX.items()})
    assert same_bounds(BBOX, dict(BBOX), tolerance=0)


@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
def test_same_bounds_rejects_movement_or_resize_beyond_tolerance(field):
    assert not same_bounds(BBOX, {**BBOX, field: BBOX[field] + 1.01})


@pytest.mark.parametrize("invalid", [None, {}, {**BBOX, "width": 0}, {**BBOX, "y": float("nan")}])
def test_same_bounds_rejects_missing_or_invalid_bounds(invalid):
    assert not same_bounds(BBOX, invalid)
    assert not same_bounds(invalid, BBOX)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_same_bounds_rejects_invalid_tolerance(tolerance):
    assert not same_bounds(BBOX, BBOX, tolerance=tolerance)
