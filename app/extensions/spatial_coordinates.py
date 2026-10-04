from __future__ import annotations

from math import isfinite

from hcaptcha_challenger.models import PointCoordinate, SpatialPath

NORMALIZED_SPATIAL_INSTRUCTION = (
    "Use only the single original challenge image supplied in this request. "
    "Return x and y as integers normalized to 0..1000 relative to the entire image: "
    "the top-left is (0, 0) and the bottom-right is (1000, 1000). "
    "Do not use webpage coordinates or coordinate-grid labels; no grid image is supplied. "
    "Inspect the whole task and return every matching interactive target, not reference "
    "examples, labels, or instructions. Return each selected point only once; never repeat "
    "a point to satisfy a requested count. Return one path per required drag. "
    "For drag paths, start_point is the center "
    "of the movable piece and end_point is the center of its intended destination outline."
)

_Bounds = tuple[float, float, float, float]


def uses_normalized_coordinates(model: str) -> bool:
    return model.rsplit("/", 1)[-1].lower().startswith("qwen3-vl-")


def _finite_number(value: object, name: str) -> float:
    try:
        if isinstance(value, bool) or not isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return float(value)
    except (TypeError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None


def _bounds_values(bbox: dict | None) -> _Bounds:
    if not isinstance(bbox, dict):
        raise ValueError("A valid challenge bounding box is required")
    values = tuple(
        _finite_number(bbox.get(name), f"bbox.{name}") for name in ("x", "y", "width", "height")
    )
    left, top, width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("Challenge bounding box width and height must be positive")
    return left, top, width, height


def _project_point(point: PointCoordinate, bounds: _Bounds) -> PointCoordinate:
    x = _finite_number(point.x, "point.x")
    y = _finite_number(point.y, "point.y")
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Normalized point coordinates must be within 0..1000")
    left, top, width, height = bounds
    page_x = _finite_number(left + x / 1000 * width, "page.x")
    page_y = _finite_number(top + y / 1000 * height, "page.y")
    return PointCoordinate(x=round(page_x), y=round(page_y))


def to_page_point(point: PointCoordinate, bbox: dict | None) -> PointCoordinate:
    """Map a normalized point to page coordinates without clamping invalid answers."""
    return _project_point(point, _bounds_values(bbox))


def to_page_paths(paths: list[SpatialPath], bbox: dict | None) -> list[SpatialPath]:
    bounds = _bounds_values(bbox)
    return [
        SpatialPath(
            start_point=_project_point(path.start_point, bounds),
            end_point=_project_point(path.end_point, bounds),
        )
        for path in paths
    ]


def unique_points(points: list[PointCoordinate]) -> list[PointCoordinate]:
    """Keep the first occurrence of each exact coordinate within one response."""
    seen: set[tuple[int, int]] = set()
    result: list[PointCoordinate] = []
    for point in points:
        coordinates = point.x, point.y
        if coordinates not in seen:
            seen.add(coordinates)
            result.append(point)
    return result


def same_bounds(old: dict | None, new: dict | None, tolerance: float = 1.0) -> bool:
    try:
        old_values = _bounds_values(old)
        new_values = _bounds_values(new)
        tolerance = _finite_number(tolerance, "tolerance")
    except ValueError:
        return False
    return tolerance >= 0 and all(
        abs(before - after) <= tolerance for before, after in zip(old_values, new_values)
    )
