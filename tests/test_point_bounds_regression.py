"""Point actions use DOM bounds without assuming every target has a colored background."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cv2
import numpy as np
import pytest
from hcaptcha_challenger.models import ImageAreaSelectChallenge, PointCoordinate

from extensions import hcaptcha_adapter as adapter

BBOX = {"x": 100, "y": 100, "width": 500, "height": 470}


@pytest.fixture
def point_flow(monkeypatch):
    class Arm:
        async def challenge_image_label_select(self, _job):
            raise AssertionError("Point patch was not installed")

        async def challenge_image_drag_drop(self, _job):
            raise AssertionError("Unexpected drag")

    monkeypatch.setattr(adapter, "RoboticArm", Arm)
    monkeypatch.setattr(adapter, "_apply_empty_checkcaptcha_patch", lambda: None)
    adapter.apply_hcaptcha_drag_patch()

    def build(
        model, coordinates, *, count_layout=False, prompt=None, source_question=None, responses=None
    ):
        image = np.full((470, 500, 3), 245, dtype=np.uint8)
        image[:108] = (143, 131, 0)
        if count_layout:
            image[130:460, 10:490] = (80, 120, 160)
            for y in (165, 275, 385):
                cv2.ellipse(image, (65, y), (29, 18), 0, 0, 360, (20, 20, 20), -1)
            default_prompt = "Click the requested animal count"
        else:
            image[130:460, 10:490] = 255
            image[220:460, 10:490] = (60, 130, 70)
            cv2.circle(image, (250, 170), 20, (0, 0, 220), -1)
            default_prompt = "Click the red circle"
        prompt = prompt or default_prompt
        monkeypatch.setattr(adapter.cv2, "imread", lambda _path: image)
        answer = ImageAreaSelectChallenge(
            challenge_prompt=prompt, points=[PointCoordinate(x=coordinates[0], y=coordinates[1])]
        )
        request = AsyncMock(side_effect=responses or [answer])
        monkeypatch.setattr(adapter, "_request_spatial_response", request)
        view = SimpleNamespace(bounding_box=AsyncMock(return_value=dict(BBOX)))
        page = SimpleNamespace(
            wait_for_timeout=AsyncMock(), mouse=SimpleNamespace(click=AsyncMock())
        )
        arm = SimpleNamespace(
            page=page,
            config=SimpleNamespace(
                create_cache_key=lambda _payload: Path("in-memory-challenge"),
                SPATIAL_POINT_REASONER_MODEL=model,
                WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS=0,
            ),
            captcha_payload=SimpleNamespace(
                get_requester_question=lambda: source_question or prompt
            ),
            get_challenge_frame_locator=AsyncMock(
                return_value=SimpleNamespace(locator=lambda _selector: view)
            ),
            check_crumb_count=AsyncMock(return_value=1),
            _capture_spatial_mapping=AsyncMock(
                return_value=(Path("in-memory-raw.png"), Path("in-memory-grid.png"))
            ),
            _match_user_prompt=lambda _job: prompt,
            _spatial_point_reasoner=SimpleNamespace(cache_response=Mock()),
            click_by_mouse=AsyncMock(),
        )
        return SimpleNamespace(
            run=lambda: asyncio.run(Arm.challenge_image_label_select(arm, "select")),
            arm=arm,
            request=request,
        )

    return build


@pytest.mark.parametrize(
    "model,coordinates", [("Qwen/Qwen3-VL-32B-Instruct", (500, 362)), ("glm-4.6v", (350, 270))]
)
def test_white_background_target_is_not_excluded_by_unrelated_colored_region(
    point_flow, model, coordinates
):
    flow = point_flow(model, coordinates)
    assert adapter._detect_task_canvas_bounds(Path("in-memory-raw.png")) == (10, 220, 489, 459)

    flow.run()

    flow.arm.page.mouse.click.assert_awaited_once_with(350, 270, delay=180)
    flow.arm.click_by_mouse.assert_awaited_once()
    assert "interactive area" not in flow.request.await_args.args[3]


@pytest.mark.parametrize(
    "model,coordinates", [("Qwen/Qwen3-VL-32B-Instruct", (1000, 362)), ("glm-4.6v", (601, 270))]
)
def test_removing_color_heuristic_keeps_dom_boundary_enforcement(point_flow, model, coordinates):
    flow = point_flow(model, coordinates)

    with pytest.raises(ValueError, match="outside challenge bounds"):
        flow.run()

    flow.arm.page.mouse.click.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()


def test_known_count_layout_still_excludes_reference_strip(point_flow):
    flow = point_flow("glm-4.6v", (165, 300), count_layout=True)

    with pytest.raises(ValueError, match="outside clickable grid"):
        flow.run()

    flow.arm.page.mouse.click.assert_not_awaited()
    flow.arm.click_by_mouse.assert_not_awaited()


def test_point_answer_is_reasked_when_it_omits_an_explicit_target(point_flow):
    prompt = "Select the targets"
    source_question = "Please click on the gap on each of the two lines"
    responses = [
        ImageAreaSelectChallenge(challenge_prompt=prompt, points=[PointCoordinate(x=500, y=500)]),
        ImageAreaSelectChallenge(
            challenge_prompt=prompt,
            points=[PointCoordinate(x=400, y=500), PointCoordinate(x=600, y=500)],
        ),
    ]
    flow = point_flow(
        "Qwen/Qwen3-VL-32B-Instruct",
        (500, 500),
        prompt=prompt,
        source_question=source_question,
        responses=responses,
    )

    flow.run()

    assert flow.request.await_count == 2
    assert adapter._explicit_point_count("Click on the TWO shortest towers") == 2
    assert "gap on each of the two lines" in flow.request.await_args_list[0].args[3]
    assert "exactly 2 distinct click points" in flow.request.await_args_list[0].args[3]
    assert "previous answer was incomplete" in flow.request.await_args_list[1].args[3]
    assert [call.args[:2] for call in flow.arm.page.mouse.click.await_args_list] == [
        (300, 335),
        (400, 335),
    ]


def test_man_made_selection_prompt_excludes_natural_objects(point_flow):
    flow = point_flow(
        "Qwen/Qwen3-VL-32B-Instruct",
        (500, 500),
        prompt="Select the targets",
        source_question="Tap on all things that people made",
    )

    flow.run()

    prompt = flow.request.await_args.args[3]
    assert "Select only objects manufactured by people" in prompt
    assert "pine cones" in prompt
    assert "not its tile or collage background" in prompt
