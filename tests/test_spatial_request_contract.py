import asyncio
import itertools
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cv2
import numpy as np
import pytest
from google.genai import types
from hcaptcha_challenger.models import (
    ImageAreaSelectChallenge,
    ImageDragDropChallenge,
    PointCoordinate,
    SpatialPath,
)
from hcaptcha_challenger.tools.internal.providers.gemini import GeminiProvider
from hcaptcha_challenger.tools.spatial.path import SpatialPathReasoner
from hcaptcha_challenger.tools.spatial.point import SpatialPointReasoner

import extensions.hcaptcha_adapter as adapter
from extensions.llm_adapter import _GLMAsyncModels, _PatchedResponse
from extensions.spatial_coordinates import (
    NORMALIZED_SPATIAL_INSTRUCTION,
    uses_normalized_coordinates,
)

QWEN_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BBOX = {"x": 1200.0, "y": 800.0, "width": 500.0, "height": 400.0}


def _answer(coordinates):
    return ImageAreaSelectChallenge(
        challenge_prompt="Select circles",
        points=[PointCoordinate(x=x, y=y) for x, y in coordinates],
    )


def _offline_reasoner(answer, model=QWEN_MODEL, *, reasoner_type=SpatialPointReasoner):
    uploads = []
    requests = []
    bridge = _GLMAsyncModels(settings=None, storage={})

    async def upload(*, file):
        uploads.append(file)
        return types.File(uri=f"https://example.invalid/{file.name}", mime_type="image/png")

    async def generate_content(*, model, contents, config):
        # Keep the real SDK config and GLM message construction; replace only network I/O.
        requests.append(
            {"model": model, "config": config, "messages": bridge._build_messages(contents, config)}
        )
        text = answer.model_dump_json()
        return _PatchedResponse(
            text=text, parsed=answer, raw={"choices": [{"message": {"content": text}}]}
        )

    provider = GeminiProvider(api_key="offline-test-key", model=model)
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(
            files=SimpleNamespace(upload=upload),
            models=SimpleNamespace(generate_content=generate_content),
        )
    )
    reasoner = reasoner_type("offline-test-key", model=model, provider=provider)
    return reasoner, uploads, requests


def _images(tmp_path):
    raw, projection = tmp_path / "raw.png", tmp_path / "grid.png"
    assert cv2.imwrite(str(raw), np.zeros((400, 500, 3), dtype=np.uint8))
    assert cv2.imwrite(str(projection), np.zeros((800, 1000, 3), dtype=np.uint8))
    return raw, projection


def test_qwen_single_image_instruction_survives_real_google_config(tmp_path):
    raw, projection = _images(tmp_path)
    reasoner, uploads, requests = _offline_reasoner(_answer([(500, 500)]))

    result = asyncio.run(
        adapter._request_spatial_response(
            reasoner,
            raw,
            projection,
            "Select circles",
            ImageAreaSelectChallenge,
            normalized=uses_normalized_coordinates(QWEN_MODEL),
        )
    )

    assert uploads == [raw]
    assert len(requests) == 1
    config = requests[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert isinstance(config.system_instruction, str)
    assert config.system_instruction == NORMALIZED_SPATIAL_INSTRUCTION
    assert config.response_schema is ImageAreaSelectChallenge
    assert requests[0]["model"] == QWEN_MODEL
    messages = requests[0]["messages"]
    assert NORMALIZED_SPATIAL_INSTRUCTION in messages[0]["content"]
    assert "gray coordinate grid" not in json.dumps(messages)
    images = [item for item in messages[1]["content"] if item["type"] == "image_url"]
    assert images == [
        {"type": "image_url", "image_url": {"url": "https://example.invalid/raw.png"}}
    ]
    assert [(point.x, point.y) for point in result.points] == [(500, 500)]


def test_non_qwen_preserves_original_reasoner_and_two_images(tmp_path):
    raw, projection = _images(tmp_path)
    model = "zai-org/GLM-4.5V"
    reasoner, uploads, requests = _offline_reasoner(_answer([(1450, 1000)]), model=model)

    asyncio.run(
        adapter._request_spatial_response(
            reasoner,
            raw,
            projection,
            "Select circles",
            ImageAreaSelectChallenge,
            normalized=uses_normalized_coordinates(model),
        )
    )

    assert uploads == [raw, projection]
    assert len(requests) == 1
    assert requests[0]["config"].system_instruction == reasoner.description
    assert "gray coordinate grid" in requests[0]["messages"][0]["content"]
    images = [item for item in requests[0]["messages"][1]["content"] if item["type"] == "image_url"]
    assert len(images) == 2


@pytest.fixture
def point_flow(monkeypatch, tmp_path):
    class OfflineArm:
        async def challenge_image_label_select(self, job_type):
            raise AssertionError("The point patch was not installed")

        async def challenge_image_drag_drop(self, job_type):
            raise AssertionError("The drag method is outside this point test")

    # Install the production point patch on an isolated class, not the global browser class.
    monkeypatch.setattr(adapter, "RoboticArm", OfflineArm)
    monkeypatch.setattr(adapter, "_apply_empty_checkcaptcha_patch", lambda: None)
    monkeypatch.setattr(adapter, "_detect_task_canvas_bounds", lambda _raw: (0, 0, 500, 400))
    adapter.apply_hcaptcha_drag_patch()
    original_validation = adapter._point_answer_validation_error

    def build(coordinates, *, bounds=None, model=QWEN_MODEL):
        raw, projection = _images(tmp_path)
        cache_key = tmp_path / "capture"
        cache_key.mkdir()
        answer = _answer(coordinates)
        reasoner, uploads, requests = _offline_reasoner(answer, model=model)
        sequence = bounds or [dict(BBOX)]
        view = SimpleNamespace(
            bounding_box=AsyncMock(
                side_effect=itertools.chain(sequence, itertools.repeat(sequence[-1]))
            )
        )
        submit = object()
        frame = SimpleNamespace(
            locator=Mock(
                side_effect=lambda selector: view if "challenge-view" in selector else submit
            )
        )
        mouse = SimpleNamespace(click=AsyncMock())
        page = SimpleNamespace(mouse=mouse, wait_for_timeout=AsyncMock())
        arm = SimpleNamespace(
            page=page,
            config=SimpleNamespace(
                SPATIAL_POINT_REASONER_MODEL=model,
                WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS=0,
                create_cache_key=lambda _payload: cache_key,
            ),
            captcha_payload=object(),
            get_challenge_frame_locator=AsyncMock(return_value=frame),
            check_crumb_count=AsyncMock(return_value=1),
            _capture_spatial_mapping=AsyncMock(return_value=(raw, projection)),
            _match_user_prompt=lambda _job: "Select circles",
            _spatial_point_reasoner=reasoner,
            click_by_mouse=AsyncMock(),
        )
        validations = []

        def record_validation(points, **kwargs):
            validations.append([(point.x, point.y) for point in points])
            return original_validation(points, **kwargs)

        monkeypatch.setattr(adapter, "_point_answer_validation_error", record_validation)
        return SimpleNamespace(
            run=lambda: asyncio.run(OfflineArm.challenge_image_label_select(arm, "select")),
            run_drag=lambda: asyncio.run(OfflineArm.challenge_image_drag_drop(arm, "drag")),
            arm=arm,
            uploads=uploads,
            requests=requests,
            answer=answer,
            validations=validations,
            raw=raw,
            cache_key=cache_key,
        )

    return build


def test_point_flow_maps_before_validation_and_clicks_duplicates_once(point_flow):
    flow = point_flow([(500, 500), (500, 500), (100, 750)])

    flow.run()

    expected = [(1450, 1000), (1250, 1100)]
    assert flow.validations == [expected]
    assert [call.args for call in flow.arm.page.mouse.click.await_args_list] == expected
    assert flow.arm.click_by_mouse.await_count == 1
    assert flow.uploads == [flow.raw]
    assert [(point.x, point.y) for point in flow.answer.points] == [
        (500, 500),
        (500, 500),
        (100, 750),
    ]
    cached = json.loads((flow.cache_key / "capture_0_model_answer.json").read_text("utf-8"))
    assert [(point["x"], point["y"]) for point in cached["parsed"]["points"]] == [
        (500, 500),
        (500, 500),
        (100, 750),
    ]
    metadata = json.loads((flow.cache_key / "capture_0_coordinate_frame.json").read_text("utf-8"))
    assert metadata == {"space": "image_1000", "image": "raw.png", "bbox": BBOX}


def test_non_qwen_point_flow_keeps_page_coordinates(point_flow):
    flow = point_flow([(1450, 1000)], model="zai-org/GLM-4.5V")

    flow.run()

    assert flow.validations == [[(1450, 1000)]]
    assert flow.arm.page.mouse.click.await_args.args == (1450, 1000)
    assert len(flow.uploads) == 2
    metadata = json.loads((flow.cache_key / "capture_0_coordinate_frame.json").read_text("utf-8"))
    assert metadata["space"] == "page"


@pytest.mark.parametrize("point", [(1000, 500), (500, 1000), (999, 500), (500, 999)])
def test_normalized_points_on_or_rounded_to_css_edges_are_not_clicked(point_flow, point):
    flow = point_flow([point])

    with pytest.raises(ValueError, match="outside challenge bounds"):
        flow.run()

    flow.arm.page.mouse.click.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()


def test_bounds_change_between_clicks_stops_remaining_targets(point_flow):
    moved = {**BBOX, "x": BBOX["x"] + 20}
    flow = point_flow([(500, 500), (100, 750)], bounds=[BBOX] * 4 + [moved])

    with pytest.raises(ValueError, match="before clicking"):
        flow.run()

    assert [call.args for call in flow.arm.page.mouse.click.await_args_list] == [(1450, 1000)]
    flow.arm.click_by_mouse.assert_not_awaited()


@pytest.mark.parametrize("stage", ["capture", "model"])
def test_changed_bounds_prevent_all_clicks(point_flow, stage):
    moved = {**BBOX, "x": BBOX["x"] + 20}
    bounds = [BBOX, moved] if stage == "capture" else [BBOX, BBOX, moved]
    flow = point_flow([(500, 500)], bounds=bounds)

    with pytest.raises(ValueError, match="Challenge bounds changed"):
        flow.run()

    flow.arm.page.mouse.click.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()
    assert flow.validations == []
    assert len(flow.requests) == (0 if stage == "capture" else 1)


@pytest.mark.parametrize("coordinates", [[(1001, 500)], [(500, -1)]])
def test_out_of_range_normalized_answer_is_cached_but_never_clicked(point_flow, coordinates):
    flow = point_flow(coordinates)

    with pytest.raises(ValueError, match="0..1000"):
        flow.run()

    flow.arm.page.mouse.click.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()
    assert flow.validations == []
    cached = json.loads((flow.cache_key / "capture_0_model_answer.json").read_text("utf-8"))
    assert [(point["x"], point["y"]) for point in cached["parsed"]["points"]] == coordinates


def _paths(coordinates):
    return [
        SpatialPath(
            start_point=PointCoordinate(x=start[0], y=start[1]),
            end_point=PointCoordinate(x=end[0], y=end[1]),
        )
        for start, end in coordinates
    ]


def _path_pairs(paths):
    return [
        ((path.start_point.x, path.start_point.y), (path.end_point.x, path.end_point.y))
        for path in paths
    ]


@pytest.fixture
def drag_flow(monkeypatch, point_flow):
    original_correction = adapter._correct_drag_source_points

    def build(coordinates, *, bounds=None, local_solver=None):
        flow = point_flow([(500, 500)], bounds=bounds)
        answer = ImageDragDropChallenge(
            challenge_prompt="Drag the matching pieces", paths=_paths(coordinates)
        )
        reasoner, uploads, requests = _offline_reasoner(answer, reasoner_type=SpatialPathReasoner)
        flow.arm.config.SPATIAL_PATH_REASONER_MODEL = QWEN_MODEL
        flow.arm.captcha_payload = SimpleNamespace(tasklist=[SimpleNamespace(entities=[])])
        flow.arm._spatial_path_reasoner = reasoner
        flow.arm._perform_drag_drop = AsyncMock()
        flow.answer, flow.uploads, flow.requests = answer, uploads, requests
        flow.run = flow.run_drag
        flow.correction_inputs = []
        flow.line_solver = Mock(return_value=answer.paths if local_solver == "line" else None)
        flow.outline_solver = AsyncMock(
            return_value=answer.paths if local_solver == "outline" else None
        )
        monkeypatch.setattr(adapter, "_resolve_line_path", flow.line_solver)
        monkeypatch.setattr(adapter, "_resolve_outline_paths", flow.outline_solver)

        def record_correction(paths, **kwargs):
            flow.correction_inputs.append(_path_pairs(paths))
            return original_correction(paths, **kwargs)

        monkeypatch.setattr(adapter, "_correct_drag_source_points", record_correction)
        return flow

    return build


def test_qwen_drag_maps_both_endpoints_before_source_correction_and_execution(drag_flow):
    coordinates = [((100, 200), (700, 800)), ((200, 300), (600, 700))]
    flow = drag_flow(coordinates)

    flow.run()

    expected = [((1250, 880), (1550, 1120)), ((1300, 920), (1500, 1080))]
    assert flow.correction_inputs == [expected]
    executed = [call.args[0] for call in flow.arm._perform_drag_drop.await_args_list]
    assert _path_pairs(executed) == expected
    assert flow.validations == [[(1250, 880), (1550, 1120), (1300, 920), (1500, 1080)]]
    assert flow.uploads == [flow.raw]
    assert len(flow.requests) == 1
    assert flow.requests[0]["config"].response_schema is ImageDragDropChallenge
    assert flow.requests[0]["config"].system_instruction == NORMALIZED_SPATIAL_INSTRUCTION
    assert "gray coordinate grid" not in json.dumps(flow.requests[0]["messages"])
    assert _path_pairs(flow.answer.paths) == coordinates
    cached = json.loads((flow.cache_key / "capture_0_model_answer.json").read_text("utf-8"))
    assert cached["parsed"]["paths"] == [path.model_dump() for path in flow.answer.paths]
    flow.arm.click_by_mouse.assert_awaited_once()


@pytest.mark.parametrize(
    "bad_coords",
    [None, [], ["59.5", 55], [float("nan"), 55], [float("inf"), 55], [59.5, 55], [-1, 55]],
)
def test_bad_payload_coordinates_reach_model_and_preserve_its_drag_paths(
    drag_flow, monkeypatch, bad_coords
):
    real_line_solver = adapter._resolve_line_path
    real_outline_solver = adapter._resolve_outline_paths
    coordinates = [((100, 200), (700, 800)), ((200, 300), (600, 700))]
    flow = drag_flow(coordinates)
    monkeypatch.setattr(adapter, "_resolve_line_path", real_line_solver)
    monkeypatch.setattr(adapter, "_resolve_outline_paths", real_outline_solver)
    flow.arm._match_user_prompt = lambda _job: "Put animals into matching outlines"
    flow.arm.captcha_payload = SimpleNamespace(
        tasklist=[
            SimpleNamespace(
                entities=[
                    SimpleNamespace(coords=[59, 55], size=[85, 85]),
                    SimpleNamespace(coords=bad_coords, size=[85, 85]),
                ]
            )
        ],
        get_requester_question=lambda: "Put animals into matching outlines",
    )

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("Unsafe local geometry must not download entity images")

    monkeypatch.setattr(adapter.httpx, "AsyncClient", unexpected_network)

    flow.run()

    expected = [((1250, 880), (1550, 1120)), ((1300, 920), (1500, 1080))]
    assert len(flow.requests) == 1
    assert flow.uploads == [flow.raw]
    assert flow.correction_inputs == [expected]
    executed = [call.args[0] for call in flow.arm._perform_drag_drop.await_args_list]
    assert _path_pairs(executed) == expected
    assert flow.validations == [[(1250, 880), (1550, 1120), (1300, 920), (1500, 1080)]]
    assert _path_pairs(flow.answer.paths) == coordinates
    flow.arm.click_by_mouse.assert_awaited_once()


@pytest.mark.parametrize("local_solver", ["line", "outline"])
def test_local_drag_solution_keeps_page_coordinates_and_skips_model(drag_flow, local_solver):
    coordinates = [((1250, 880), (1550, 1120))]
    flow = drag_flow(coordinates, local_solver=local_solver)

    flow.run()

    assert flow.requests == []
    assert flow.uploads == []
    assert flow.correction_inputs == []
    assert _path_pairs([flow.arm._perform_drag_drop.await_args.args[0]]) == coordinates
    assert _path_pairs(flow.answer.paths) == coordinates
    assert flow.validations == [[(1250, 880), (1550, 1120)]]
    flow.line_solver.assert_called_once()
    if local_solver == "line":
        flow.outline_solver.assert_not_awaited()
    else:
        flow.outline_solver.assert_awaited_once()
    flow.arm.click_by_mouse.assert_awaited_once()


def test_bounds_change_between_drags_stops_remaining_paths(drag_flow):
    moved = {**BBOX, "y": BBOX["y"] + 20}
    flow = drag_flow(
        [((100, 200), (700, 800)), ((200, 300), (600, 700))], bounds=[BBOX] * 4 + [moved]
    )

    with pytest.raises(ValueError, match="before performing a drag"):
        flow.run()

    executed = [call.args[0] for call in flow.arm._perform_drag_drop.await_args_list]
    assert _path_pairs(executed) == [((1250, 880), (1550, 1120))]
    flow.arm.click_by_mouse.assert_not_awaited()


@pytest.mark.parametrize("invalid_path", [((1001, 500), (800, 800)), ((200, 200), (500, -1))])
def test_any_out_of_range_normalized_endpoint_prevents_all_drags(drag_flow, invalid_path):
    flow = drag_flow([((100, 200), (700, 800)), invalid_path])

    with pytest.raises(ValueError, match="0..1000"):
        flow.run()

    assert flow.correction_inputs == []
    assert flow.validations == []
    flow.arm._perform_drag_drop.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()
