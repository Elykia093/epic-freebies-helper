"""Outline candidates must exclude draggable sprites regardless of layout side."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from extensions import hcaptcha_adapter as adapter


# Sanitized 500px challenge screenshot from run53. No account, URL, or session payload.
REAL_CHALLENGE = Path(__file__).parent / "fixtures" / "hcaptcha_outline_sources_left.png"


def source_entities():
    return [
        SimpleNamespace(coords=[59, 55], size=[85, 85]),
        SimpleNamespace(coords=[59, 154], size=[85, 85]),
    ]


def test_real_left_source_sprites_are_not_returned_as_destination_outlines():
    image = cv2.imread(str(REAL_CHALLENGE))
    assert image.shape[1] == 500
    assert adapter._detect_task_canvas_origin(REAL_CHALLENGE) == (10, 136)

    targets = adapter._extract_outline_targets(REAL_CHALLENGE, source_entities=source_entities())

    # The legacy extractor returned two contours centered on the movable sprites.
    # This sample has no safely extracted targets after those regions are excluded.
    assert targets == []


def synthetic_challenge(tmp_path, monkeypatch, *, mirrored=False, bottom=False):
    image = np.full((400, 500, 3), (60, 30, 10), dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (80, 100), (240, 240, 240), 4)
    target_top, target_bottom = (360, 396) if bottom else (200, 250)
    cv2.rectangle(image, (360, target_top), (410, target_bottom), (240, 240, 240), 4)
    source_x = 55
    target_x = 385
    if mirrored:
        image = cv2.flip(image, 1)
        source_x = 499 - source_x
        target_x = 499 - target_x
    path = tmp_path / "outline-layout.png"
    assert cv2.imwrite(str(path), image)
    monkeypatch.setattr(adapter, "_detect_task_canvas_origin", lambda _path: (0, 0))
    entity = SimpleNamespace(coords=[source_x, 70], size=[70, 80])
    return path, entity, (target_x, (target_top + target_bottom) / 2)


@pytest.mark.parametrize("mirrored", [False, True])
def test_outline_targets_support_sources_on_either_side(tmp_path, monkeypatch, mirrored):
    path, entity, expected = synthetic_challenge(tmp_path, monkeypatch, mirrored=mirrored)

    targets = adapter._extract_outline_targets(path, source_entities=[entity])

    assert len(targets) == 1
    assert targets[0][1] == pytest.approx(expected, abs=1)


def test_outline_targets_are_not_discarded_near_canvas_bottom(tmp_path, monkeypatch):
    path, entity, expected = synthetic_challenge(tmp_path, monkeypatch, bottom=True)

    targets = adapter._extract_outline_targets(path, source_entities=[entity])

    assert len(targets) == 1
    assert targets[0][1] == pytest.approx(expected, abs=1)


@pytest.mark.parametrize(
    "entity",
    [
        SimpleNamespace(coords=None, size=[70, 80]),
        SimpleNamespace(coords=[55, 70], size=None),
        SimpleNamespace(coords=[55], size=[70, 80]),
        SimpleNamespace(coords=["59.5", 70], size=[70, 80]),
        SimpleNamespace(coords=[float("nan"), 70], size=[70, 80]),
        SimpleNamespace(coords=[55, 70], size=[float("inf"), 80]),
        SimpleNamespace(coords=[55, 70], size=[0, 80]),
        SimpleNamespace(coords=[55, 70], size=[70, -1]),
        SimpleNamespace(coords=[10, 70], size=[70, 80]),
        SimpleNamespace(coords=[490, 70], size=[70, 80]),
        SimpleNamespace(coords=[55, 395], size=[70, 80]),
    ],
)
def test_unreliable_source_regions_disable_local_target_extraction(tmp_path, monkeypatch, entity):
    path, _, _ = synthetic_challenge(tmp_path, monkeypatch)

    assert adapter._extract_outline_targets(path, source_entities=[entity]) == []


def test_missing_source_regions_disable_local_target_extraction(tmp_path, monkeypatch):
    path, _, _ = synthetic_challenge(tmp_path, monkeypatch)

    assert adapter._extract_outline_targets(path, source_entities=[]) == []


def test_contour_overlapping_a_source_region_is_rejected(tmp_path, monkeypatch):
    image = np.full((200, 200, 3), (60, 30, 10), dtype=np.uint8)
    cv2.rectangle(image, (40, 60), (100, 120), (240, 240, 240), 4)
    path = tmp_path / "overlapping.png"
    assert cv2.imwrite(str(path), image)
    monkeypatch.setattr(adapter, "_detect_task_canvas_origin", lambda _path: (0, 0))
    source = SimpleNamespace(coords=[50, 70], size=[40, 40])

    assert adapter._extract_outline_targets(path, source_entities=[source]) == []


def test_real_sample_falls_back_before_downloading_entity_images(monkeypatch):
    payload = SimpleNamespace(
        tasklist=[SimpleNamespace(entities=source_entities())],
        get_requester_question=lambda: "Put the correct animal into its matching outline",
    )

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("No network is needed when safe targets are unavailable")

    monkeypatch.setattr(adapter.httpx, "AsyncClient", unexpected_network)

    result = asyncio.run(
        adapter._resolve_outline_paths(
            captcha_payload=payload,
            crumb_id=0,
            challenge_screenshot=REAL_CHALLENGE,
            challenge_bbox={"x": 710, "y": 314.8, "width": 500, "height": 470},
        )
    )

    assert result is None


@pytest.mark.parametrize("bad_coords", [None, ["59.5", 70]])
def test_bad_geometry_falls_back_before_source_point_conversion(tmp_path, monkeypatch, bad_coords):
    path, entity, _ = synthetic_challenge(tmp_path, monkeypatch)
    payload = SimpleNamespace(
        tasklist=[
            SimpleNamespace(entities=[entity, SimpleNamespace(coords=bad_coords, size=[70, 80])])
        ],
        get_requester_question=lambda: "Put animals into matching outlines",
    )

    def unexpected_conversion(**_kwargs):
        raise AssertionError("Invalid source geometry must not reach coordinate conversion")

    monkeypatch.setattr(adapter, "_payload_source_points", unexpected_conversion)
    monkeypatch.setattr(adapter.httpx, "AsyncClient", unexpected_conversion)
    result = asyncio.run(
        adapter._resolve_outline_paths(
            captcha_payload=payload,
            crumb_id=0,
            challenge_screenshot=path,
            challenge_bbox={"x": 0, "y": 0, "width": 500, "height": 400},
        )
    )

    assert result is None


def test_outline_matching_keeps_existing_confidence_threshold(monkeypatch):
    contour = np.array([[[0, 0]], [[40, 0]], [[40, 40]], [[0, 40]]], dtype=np.int32)
    monkeypatch.setattr(adapter.cv2, "matchShapes", lambda *_args: 0.17)

    assert adapter._match_outline_contours([contour], [(contour, (50, 50))]) is None
