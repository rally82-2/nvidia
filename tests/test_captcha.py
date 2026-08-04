from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import requests
from playwright.async_api import Page

from captcha import (
    CaptchaRunSolver,
    LLMCaptchaAction,
    LLMCaptchaCapture,
    LLMCaptchaDecision,
    LLMCaptchaSolver,
    ManualCaptchaSolver,
    YesCaptchaSolver,
    _apply_llm_action,
    _captcha_action_error,
    _capture_llm_captcha,
    _click_hcaptcha_checkbox,
    _click_hcaptcha_submit,
    _coerce_dom_grid_decision,
    _fallback_grid_action_error,
    _get_site_key,
    _is_actionable_submit_label,
    _is_checkbox_capture,
    _is_find_all_challenge,
    _is_multi_select_challenge,
    _is_new_hcaptcha_challenge,
    _is_repeated_click,
    _is_two_arrow_challenge,
    _is_unequal_slices_challenge,
    _llm_captcha_prompt,
    _llm_retry_feedback,
    _llm_timeout_error,
    _parse_arrow_grounding_actions,
    _parse_find_grounding_actions,
    _parse_grounded_click_actions,
    _parse_llm_decision,
    _parse_pattern_grounding,
    _parse_verbose_click_decision,
    _refresh_llm_capture_geometry,
    _chat_completions_endpoint,
    _responses_endpoint,
    build_captcha_solver,
    reset_captcha_state,
    start_capturing_sitekey,
)
from config import CaptchaConfig


def _make_sse_response(protocol: str, text: str, response_id: str) -> Mock:
    """Build a Mock requests.Response that streams *text* back as SSE.

    Mirrors what an OpenAI-compatible provider returns when the request is sent
    with ``stream: True``. Used because captcha._request_decision now consumes
    the stream via _consume_sse_response instead of response.json().
    """
    if protocol == "chat_completions":
        data = {"id": response_id, "choices": [{"delta": {"content": text}}]}
        lines = ["data: " + json.dumps(data), "data: [DONE]"]
    else:
        delta_event = {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "delta": text,
        }
        completed_event = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "output_text": text,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
            },
        }
        lines = [
            "event: response.output_text.delta",
            "data: " + json.dumps(delta_event),
            "event: response.completed",
            "data: " + json.dumps(completed_event),
        ]
    response = Mock(ok=True, status_code=200)
    response.iter_lines = Mock(return_value=iter(lines))
    response.close = Mock()
    return response


class FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def click(self, x: float, y: float, delay: int) -> None:
        self.calls.append(("click", x, y, delay))

    async def move(self, x: float, y: float, steps: int = 1) -> None:
        self.calls.append(("move", x, y, steps))

    async def down(self) -> None:
        self.calls.append(("down",))

    async def up(self) -> None:
        self.calls.append(("up",))


class FakePage:
    viewport_size = {"width": 1280, "height": 800}

    def __init__(self) -> None:
        self.mouse = FakeMouse()
        self.frames: list[Any] = []
        self.screenshot_count = 0

    async def screenshot(self, type: str, clip=None) -> bytes:
        self.screenshot_count += 1
        self.last_clip = clip
        return f"screenshot-{self.screenshot_count}-{type}".encode()

    async def evaluate(self, expression: str):
        return False


class FakeRequestPage:
    def __init__(self) -> None:
        self.request_handler = None

    def on(self, event: str, handler) -> None:
        if event == "request":
            self.request_handler = handler

    def capture_sitekey(self, sitekey: str) -> None:
        request = Mock(
            url=f"https://hcaptcha.com/checksiteconfig?sitekey={sitekey}"
        )
        self.request_handler(request)


class LLMCaptchaTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, mode: str) -> CaptchaConfig:
        return CaptchaConfig(
            mode=mode,
            yescaptcha_client_key="yes-key",
            yescaptcha_api_url="https://yes.example.test",
            captcharun_token="run-token",
            captcharun_api_url="https://run.example.test",
            llm_model="vision-model",
            llm_api_base="https://llm.example.test/v1",
            llm_api_key="llm-key",
            llm_api_protocol="responses",
            llm_reasoning_effort=None,
            llm_call_delay_seconds=5,
            llm_action_delay_seconds=5,
            llm_calls_per_attempt=10,
            llm_max_attempts=2,
            llm_max_output_tokens=1200,
            llm_max_concurrency=1,
            llm_artifact_dir=None,
            poll_interval_seconds=3,
            timeout_seconds=180,
        )

    def test_build_solver_keeps_all_modes_available(self) -> None:
        expected = {
            "manual": ManualCaptchaSolver,
            "yescaptcha": YesCaptchaSolver,
            "captcharun": CaptchaRunSolver,
            "llm": LLMCaptchaSolver,
        }

        for mode, solver_type in expected.items():
            with self.subTest(mode=mode):
                self.assertIsInstance(build_captcha_solver(self._config(mode)), solver_type)

    async def test_sitekey_capture_is_isolated_per_page(self) -> None:
        first = FakeRequestPage()
        second = FakeRequestPage()
        reset_captcha_state()
        start_capturing_sitekey(cast(Page, first))
        start_capturing_sitekey(cast(Page, second))

        first.capture_sitekey("first-key")
        second.capture_sitekey("second-key")

        self.assertEqual(await _get_site_key(cast(Page, first)), "first-key")
        self.assertEqual(await _get_site_key(cast(Page, second)), "second-key")

    def test_prompt_requires_one_action_from_fresh_screenshot(self) -> None:
        prompt = " ".join(
            _llm_captcha_prompt(
                1280,
                800,
                1,
                2,
                "Drag ONE animal to the matching silhouette",
                "Check",
                "hcaptcha_canvas",
                1,
            ).split()
        )

        self.assertIn("at most ONE action", prompt)
        self.assertIn("send a fresh screenshot", prompt)
        self.assertIn("green checked hCaptcha checkbox", prompt)
        self.assertIn("outer-page submit controls", prompt)
        self.assertIn("visible hCaptcha canvas cropped", prompt)
        self.assertIn("Drag ONE animal", prompt)
        self.assertIn('status "verify"', prompt)
        self.assertIn("Actions already executed during this challenge attempt: 1", prompt)
        self.assertIn("two arrows that break the chain", prompt)
        self.assertIn("teal circle with a white check", prompt)
        self.assertIn("Both markers mean SELECTED", prompt)

    def test_submit_label_distinguishes_skip_from_check(self) -> None:
        self.assertFalse(_is_actionable_submit_label("跳过 skip challenge"))
        self.assertTrue(_is_actionable_submit_label("检查 submit answer"))
        self.assertFalse(_is_actionable_submit_label("下一个 next challenge"))

    def test_new_challenge_boundary_clears_after_check_to_skip(self) -> None:
        self.assertTrue(
            _is_new_hcaptcha_challenge(
                "",
                "",
                "Place the correct animal",
                "跳过 跳过挑战",
            )
        )
        self.assertTrue(
            _is_new_hcaptcha_challenge(
                "Click the two arrows",
                "检查 验证答案",
                "Click the two arrows",
                "跳过 跳过挑战",
            )
        )
        self.assertFalse(
            _is_new_hcaptcha_challenge(
                "Click the two arrows",
                "检查 验证答案",
                "Click the two arrows",
                "检查 验证答案",
            )
        )

    def test_retry_feedback_names_selected_points(self) -> None:
        feedback = _llm_retry_feedback(
            "pick another",
            [(120.0, 240.0)],
            width=416,
            height=616,
        )

        self.assertIn("pick another", feedback)
        self.assertIn("(288.5, 389.6)", feedback)
        self.assertIn("normalized_1000", feedback)
        self.assertIn("Do not click", feedback)

    def test_identifies_two_arrow_challenge(self) -> None:
        self.assertTrue(_is_two_arrow_challenge("Click the two arrows that break the chain"))
        self.assertTrue(_is_two_arrow_challenge("点击两个箭头"))
        self.assertFalse(_is_two_arrow_challenge("Place the correct animal"))

    def test_identifies_common_find_and_unequal_slice_variants(self) -> None:
        self.assertTrue(_is_find_all_challenge("Find all animals based on the number provided"))
        self.assertTrue(_is_find_all_challenge("Find ALL animals the exact number of times indicated"))
        self.assertTrue(_is_unequal_slices_challenge("Please click on the circle with uneven slices"))

    def test_rejects_fallback_grid_clicks_outside_tile_area(self) -> None:
        decision = LLMCaptchaDecision(
            "actions",
            (LLMCaptchaAction("click", 294, 111, None, None),),
            "",
        )
        capture = LLMCaptchaCapture(
            b"png",
            416,
            617,
            source="hcaptcha_iframe",
        )

        self.assertIsNotNone(_fallback_grid_action_error(decision, capture))

    def test_checks_every_action_in_dom_grid_batch(self) -> None:
        decision = LLMCaptchaDecision(
            "actions",
            (
                LLMCaptchaAction("click", 208, 199, None, None),
                LLMCaptchaAction("click", 208, 90, None, None),
            ),
            "",
        )
        capture = LLMCaptchaCapture(
            b"png",
            416,
            617,
            source="hcaptcha_iframe",
        )

        self.assertIsNotNone(_fallback_grid_action_error(decision, capture))

    def test_repairs_mixed_pixel_and_normalized_dom_grid_axes(self) -> None:
        decision = _parse_llm_decision(
            '{"status":"actions","actions":[{"kind":"click",'
            '"start_x":500,"start_y":199,"end_x":null,"end_y":null}],'
            '"message":"top middle","coordinate_space":"normalized_1000"}',
            416,
            616,
        )
        capture = LLMCaptchaCapture(
            b"png",
            416,
            616,
            source="hcaptcha_iframe",
        )

        repaired = _coerce_dom_grid_decision(decision, capture)

        self.assertEqual(repaired.actions[0].start_x, 208)
        self.assertEqual(repaired.actions[0].start_y, 199)

    def test_repairs_upscaled_dom_grid_axes(self) -> None:
        decision = _parse_llm_decision(
            '{"status":"actions","actions":[{"kind":"click",'
            '"start_x":500,"start_y":534,"end_x":null,"end_y":null}],'
            '"message":"middle tile","coordinate_space":"normalized_1000"}',
            832,
            1232,
        )
        capture = LLMCaptchaCapture(
            b"png",
            832,
            1232,
            source="hcaptcha_iframe",
        )

        repaired = _coerce_dom_grid_decision(decision, capture)

        self.assertAlmostEqual(repaired.actions[0].start_x, 416)
        self.assertEqual(repaired.actions[0].start_y, 658)

    def test_grid_row_and_column_override_inconsistent_model_coordinates(self) -> None:
        decision = _parse_llm_decision(
            '{"status":"actions","actions":[{"kind":"click",'
            '"start_x":500,"start_y":500,"end_x":null,"end_y":null,'
            '"grid_row":1,"grid_column":1}],"message":"top-left",'
            '"coordinate_space":"normalized_1000"}',
            416,
            616,
            allow_grid_coordinates=True,
        )
        capture = LLMCaptchaCapture(
            b"png",
            416,
            616,
            source="hcaptcha_iframe",
        )

        repaired = _coerce_dom_grid_decision(decision, capture)

        self.assertEqual(repaired.actions[0].start_x, 79.04)
        self.assertEqual(repaired.actions[0].start_y, 199)

    def test_grid_row_and_column_are_ignored_for_canvas_capture(self) -> None:
        decision = _parse_llm_decision(
            '{"status":"actions","actions":[{"kind":"click",'
            '"start_x":500,"start_y":500,"end_x":null,"end_y":null,'
            '"grid_row":1,"grid_column":1}],"message":"canvas target",'
            '"coordinate_space":"normalized_1000"}',
            1000,
            700,
        )

        self.assertEqual(decision.actions[0].start_x, 500)
        self.assertEqual(decision.actions[0].start_y, 350)

    def test_identifies_small_hcaptcha_checkbox_capture(self) -> None:
        self.assertTrue(
            _is_checkbox_capture(
                LLMCaptchaCapture(b"png", 318, 92, source="hcaptcha_iframe")
            )
        )
        self.assertFalse(
            _is_checkbox_capture(
                LLMCaptchaCapture(
                    b"png",
                    318,
                    92,
                    source="hcaptcha_iframe",
                    challenge_prompt="Select all buses",
                )
            )
        )

    def test_repeated_click_detects_selected_item_neighborhood(self) -> None:
        self.assertTrue(
            _is_repeated_click(
                LLMCaptchaAction("click", 505, 492, None, None),
                [(500, 500)],
                1000,
                940,
            )
        )
        self.assertFalse(
            _is_repeated_click(
                LLMCaptchaAction("click", 800, 500, None, None),
                [(500, 500)],
                1000,
                940,
            )
        )

    def test_timeout_error_preserves_last_failure(self) -> None:
        error = _llm_timeout_error(16, 14, "Responses API HTTP 429")

        self.assertIn("16 decision rounds", error)
        self.assertIn("14 API requests", error)
        self.assertIn("last result: Responses API HTTP 429", error)

    def test_parser_accepts_dom_verify_request(self) -> None:
        decision = _parse_llm_decision(
            '{"status":"verify","actions":[],"message":"selection complete"}',
            1000,
            940,
        )

        self.assertEqual(decision.status, "verify")
        self.assertEqual(decision.actions, ())

    def test_rejects_action_kind_that_conflicts_with_instruction(self) -> None:
        decision = LLMCaptchaDecision(
            "actions",
            (LLMCaptchaAction("drag", 100, 100, 200, 200),),
            "",
        )

        error = _captcha_action_error(decision, "Click the two arrows")

        self.assertIn("requires click", error or "")

    def test_parses_pixel_grounded_pattern_action(self) -> None:
        action = _parse_pattern_grounding(
            {
                "sourceIndex": 2,
                "targetRow": 4,
                "targetColumn": 3,
                "startX": 140,
                "startY": 536,
                "endX": 700,
                "endY": 827,
            },
            1000,
            940,
        )

        self.assertEqual(action, LLMCaptchaAction("drag", 140, 536, 700, 827))

    def test_parses_two_pixel_grounded_arrow_actions(self) -> None:
        actions = _parse_arrow_grounding_actions(
            {
                "kind": "arrow",
                "targets": [{"x": 185, "y": 383}, {"x": 815, "y": 348}],
            },
            1000,
            700,
        )

        self.assertEqual(
            actions,
            (
                LLMCaptchaAction("click", 185, 383, None, None),
                LLMCaptchaAction("click", 815, 348, None, None),
            ),
        )

    def test_rejects_invalid_pixel_grounded_arrow_actions(self) -> None:
        self.assertEqual(
            _parse_arrow_grounding_actions(
                {
                    "kind": "arrow",
                    "targets": [{"x": 185, "y": 383}],
                },
                1000,
                700,
            ),
            (),
        )

    def test_rejects_duplicate_pixel_grounded_find_targets(self) -> None:
        self.assertEqual(
            _parse_find_grounding_actions(
                {
                    "kind": "find_all",
                    "targets": [{"x": 280, "y": 357}, {"x": 280, "y": 357}],
                },
                1000,
                940,
            ),
            (),
        )

    def test_accepts_distinct_pixel_grounded_find_targets(self) -> None:
        actions = _parse_find_grounding_actions(
            {
                "kind": "find_all",
                "targets": [{"x": 425, "y": 437}, {"x": 715, "y": 117}],
            },
            1000,
            700,
        )

        self.assertEqual(
            actions,
            (
                LLMCaptchaAction("click", 425, 437, None, None),
                LLMCaptchaAction("click", 715, 117, None, None),
            ),
        )

    def test_parses_pixel_grounded_unequal_slice_target(self) -> None:
        actions = _parse_grounded_click_actions(
            {
                "kind": "unequal_slices",
                "target": {"x": 537, "y": 507},
            },
            1000,
            700,
        )

        self.assertEqual(
            actions,
            (LLMCaptchaAction("click", 537, 507, None, None),),
        )
        self.assertTrue(
            _is_unequal_slices_challenge(
                "Please click on the circle where each slice is a different size"
            )
        )

    def test_parser_keeps_only_first_action_for_fresh_feedback(self) -> None:
        decision = _parse_llm_decision(
            """{
                "status": "actions",
                "actions": [
                    {"kind": "click", "start_x": 120, "start_y": 80,
                     "end_x": null, "end_y": null},
                    {"kind": "drag", "start_x": 200, "start_y": 220,
                     "end_x": 420, "end_y": 260}
                ],
                "message": "act in order"
            }""",
            1280,
            800,
        )

        self.assertEqual([action.kind for action in decision.actions], ["click"])
        self.assertEqual(decision.ignored_action_count, 1)

    def test_parser_extracts_json_from_compatibility_wrapper(self) -> None:
        decision = _parse_llm_decision(
            """I inspected the image.\n```json
            {"status":"actions","actions":[],"message":"wait"}
            ```""",
            1280,
            800,
        )

        self.assertEqual(decision.status, "actions")
        self.assertEqual(decision.actions, ())

    def test_recovers_explicit_click_conclusion_from_verbose_output(self) -> None:
        decision = _parse_verbose_click_decision(
            "I inspected every tile. I'll click the bottom right tile at "
            "approximately x=347, y=532.",
            416,
            617,
            "Pick all foods that need to be kept cold in a fridge",
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.actions[0].start_x, 347)  # type: ignore[union-attr]
        self.assertEqual(decision.actions[0].start_y, 532)  # type: ignore[union-attr]

    def test_does_not_guess_from_ambiguous_or_drag_reasoning(self) -> None:
        ambiguous = _parse_verbose_click_decision(
            "Tile centers are (70, 194), (208, 363), and (347, 532).",
            416,
            617,
            "Pick all foods",
        )
        drag = _parse_verbose_click_decision(
            "I'll click the candidate at x=140, y=357.",
            1000,
            940,
            "Place the correct animal into the empty spot",
        )

        self.assertIsNone(ambiguous)
        self.assertIsNone(drag)

    def test_rejects_action_outside_viewport(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the viewport"):
            _parse_llm_decision(
                """{
                    "status": "actions",
                    "actions": [{"kind": "click", "start_x": 1281, "start_y": 80,
                                 "end_x": null, "end_y": null}],
                    "message": ""
                }""",
                1280,
                800,
            )

    def test_converts_normalized_grounding_coordinates(self) -> None:
        decision = _parse_llm_decision(
            """{
                "status": "actions",
                "actions": [{"kind": "drag", "start_x": 803, "start_y": 306,
                             "end_x": 477, "end_y": 392}],
                "message": "drag the rooster"
            }""",
            536,
            587,
        )

        self.assertEqual(decision.coordinate_space, "normalized_1000")
        self.assertEqual(decision.actions[0].start_x, 430.41)
        self.assertEqual(decision.actions[0].start_y, 179.62)
        self.assertEqual(decision.actions[0].end_x, 255.67)
        self.assertEqual(decision.actions[0].end_y, 230.1)

    def test_explicit_normalized_space_converts_in_bounds_values(self) -> None:
        decision = _parse_llm_decision(
            """{
                "status": "actions",
                "actions": [{"kind": "click", "start_x": 186, "start_y": 341,
                             "end_x": 329, "end_y": 467}],
                "message": "top left",
                "coordinate_space": "normalized_1000"
            }""",
            416,
            617,
        )

        self.assertEqual(decision.coordinate_space, "normalized_1000")
        self.assertEqual(decision.actions[0].start_x, 77.38)
        self.assertEqual(decision.actions[0].start_y, 210.4)
        self.assertIsNone(decision.actions[0].end_x)

    def test_responses_endpoint_accepts_base_or_full_url(self) -> None:
        self.assertEqual(
            _responses_endpoint("https://api.example.test/v1"),
            "https://api.example.test/v1/responses",
        )
        self.assertEqual(
            _responses_endpoint("https://api.example.test/v1/responses"),
            "https://api.example.test/v1/responses",
        )

    def test_chat_completions_endpoint_accepts_base_or_full_url(self) -> None:
        self.assertEqual(
            _chat_completions_endpoint("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        self.assertEqual(
            _chat_completions_endpoint(
                "http://127.0.0.1:11434/v1/chat/completions"
            ),
            "http://127.0.0.1:11434/v1/chat/completions",
        )

    def test_request_uses_image_and_strict_json_schema(self) -> None:
        response = _make_sse_response(
            "responses",
            '{"status":"solved","actions":[],"message":"done"}',
            "response-1",
        )
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            reasoning_effort="high",
            max_output_tokens=2048,
        )

        with patch("captcha.requests.post", return_value=response) as post:
            decision = solver._request_decision(b"png", 1280, 800, 1, 1)

        self.assertEqual(decision.status, "solved")
        self.assertEqual(decision.response_id, "response-1")
        request = post.call_args
        self.assertEqual(request.args[0], "https://api.example.test/v1/responses")
        body = request.kwargs["json"]
        self.assertEqual(body["model"], "vision-model")
        self.assertTrue(body.get("stream"))
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["max_output_tokens"], 2048)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertEqual(body["text"]["format"]["schema"]["properties"]["actions"]["maxItems"], 1)
        action_format = body["text"]["format"]["schema"]["properties"]["actions"]["items"]
        self.assertIn("grid_row", action_format["required"])
        self.assertIn("grid_column", action_format["required"])
        self.assertIn(
            "coordinate_space",
            body["text"]["format"]["schema"]["required"],
        )
        self.assertTrue(
            body["input"][0]["content"][1]["image_url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_dom_multi_select_request_allows_a_full_grid_batch(self) -> None:
        response = _make_sse_response(
            "responses",
            '{"status":"actions","actions":[{"kind":"click",'
            '"start_x":500,"start_y":500,"end_x":null,"end_y":null,'
            '"grid_row":1,"grid_column":1}],"message":"top-left",'
            '"coordinate_space":"normalized_1000"}',
            "response-1",
        )
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
        )

        with patch("captcha.requests.post", return_value=response) as post:
            decision = solver._request_decision(
                b"png",
                416,
                616,
                1,
                1,
                "Click on items that are primarily glass",
                "Check",
                "hcaptcha_iframe",
            )

        schema = post.call_args.kwargs["json"]["text"]["format"]["schema"]
        self.assertEqual(schema["properties"]["actions"]["maxItems"], 9)
        self.assertEqual(decision.actions[0].start_x, 79.04)
        self.assertEqual(decision.actions[0].start_y, 198.97)

    def test_chat_completions_request_uses_vision_and_json_schema(self) -> None:
        response = _make_sse_response(
            "chat_completions",
            '{"status":"solved","actions":[],"message":"done",'
            '"coordinate_space":"normalized_1000"}',
            "chatcmpl-1",
        )
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://chat.example.test/v1",
            "api-key",
            30,
            reasoning_effort="high",
            max_output_tokens=2048,
            api_protocol="chat_completions",
        )

        with patch("captcha.requests.post", return_value=response) as post:
            decision = solver._request_decision(b"png", 1280, 800, 1, 1)

        self.assertEqual(decision.status, "solved")
        self.assertEqual(decision.response_id, "chatcmpl-1")
        request = post.call_args
        self.assertEqual(
            request.args[0],
            "https://chat.example.test/v1/chat/completions",
        )
        body = request.kwargs["json"]
        self.assertEqual(body["model"], "vision-model")
        self.assertTrue(body.get("stream"))
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["max_tokens"], 2048)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            body["response_format"]["json_schema"]["schema"]["properties"]
            ["actions"]["maxItems"],
            1,
        )
        self.assertTrue(
            body["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_request_retries_one_transient_network_failure(self) -> None:
        response = _make_sse_response(
            "responses",
            '{"status":"solved","actions":[],"message":"done",'
            '"coordinate_space":"normalized_1000"}',
            "response-1",
        )
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
        )

        with (
            patch(
                "captcha.requests.post",
                side_effect=[requests.exceptions.SSLError("unexpected eof"), response],
            ) as post,
            patch("captcha.time.sleep") as sleep,
        ):
            decision = solver._request_decision(b"png", 100, 100, 1, 1)

        self.assertEqual(decision.status, "solved")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    async def test_solver_recaptures_between_click_and_drag(self) -> None:
        page = FakePage()
        click_decision = LLMCaptchaDecision(
            "actions",
            (LLMCaptchaAction("click", 100, 120, None, None),),
            "",
        )
        drag_decision = LLMCaptchaDecision(
            "actions",
            (LLMCaptchaAction("drag", 200, 220, 360, 260),),
            "",
        )
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=3,
            max_attempts=1,
        )

        with (
            patch.object(
                LLMCaptchaSolver,
                "_request_decision",
                side_effect=[click_decision, drag_decision],
            ),
            patch("captcha.asyncio.sleep", new=AsyncMock()),
            patch(
                "captcha._click_hcaptcha_checkbox",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, False, False, True]),
            ),
        ):
            solved = await solver.solve(cast(Page, page))

        self.assertTrue(solved)
        self.assertEqual(page.mouse.calls[0], ("click", 100, 120, 80))
        self.assertEqual(page.mouse.calls[1], ("move", 200, 220, 1))
        self.assertEqual(page.mouse.calls[2], ("down",))
        self.assertEqual(page.mouse.calls[4], ("up",))
        self.assertEqual(page.screenshot_count, 2)

    async def test_llm_request_semaphore_serializes_compatibility_api_calls(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            request_semaphore=asyncio.Semaphore(1),
        )
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def request(*_args, **_kwargs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return LLMCaptchaDecision("solved", (), "done")

        with patch.object(solver, "_request_decision", side_effect=request):
            await asyncio.gather(
                *(
                    solver._request_decision_async(b"png", 100, 100, 1, index)
                    for index in range(1, 5)
                )
            )

        self.assertEqual(maximum_active, 1)

    async def test_solver_sessions_overlap_with_shared_request_semaphore(self) -> None:
        semaphore = asyncio.Semaphore(1)
        solvers = [
            LLMCaptchaSolver(
                "vision-model",
                "https://api.example.test/v1",
                "key",
                30,
                request_semaphore=semaphore,
            )
            for _ in range(2)
        ]
        active = 0
        maximum_active = 0

        async def solve_session() -> bool:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return True

        async def solve_first(_page: Page) -> bool:
            return await solve_session()

        async def solve_second(_page: Page) -> bool:
            return await solve_session()

        with (
            patch.object(
                solvers[0],
                "_solve_session",
                side_effect=solve_first,
            ),
            patch.object(
                solvers[1],
                "_solve_session",
                side_effect=solve_second,
            ),
        ):
            results = await asyncio.gather(
                *(solver.solve(cast(Page, FakePage())) for solver in solvers)
            )

        self.assertEqual(results, [True, True])
        self.assertEqual(maximum_active, 2)

    async def test_solver_resets_early_after_three_unchanged_captures(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=5,
            max_attempts=2,
        )
        capture = LLMCaptchaCapture(b"same-image", 100, 100)

        with (
            patch(
                "captcha._capture_llm_captcha",
                new=AsyncMock(return_value=capture),
            ),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(
                    return_value=LLMCaptchaDecision("failed", (), "no action")
                ),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch("captcha._is_captcha_target_enabled", new=AsyncMock(return_value=False)),
            patch("captcha._reset_hcaptcha", new=AsyncMock(return_value=True)) as reset,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertFalse(solved)
        self.assertEqual(request.await_count, 4)
        reset.assert_awaited_once()

    async def test_solver_waits_for_model_before_submitting_actionable_button(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=2,
            max_attempts=1,
        )
        capture = LLMCaptchaCapture(
            b"selected-answers",
            100,
            100,
            submit_label="Check",
        )

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(return_value=capture)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(
                    return_value=LLMCaptchaDecision("verify", (), "ready")
                ),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch("captcha._click_hcaptcha_submit", new=AsyncMock(return_value=True)) as submit,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        request.assert_awaited_once()
        submit.assert_awaited_once()

    async def test_dom_multi_select_drains_one_model_batch_then_submits(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=3,
            max_attempts=1,
        )
        captures = [
            LLMCaptchaCapture(
                b"before-click",
                416,
                616,
                source="hcaptcha_iframe",
                challenge_prompt="Select all items that are in the reference",
                submit_label="Check",
            ),
            LLMCaptchaCapture(
                b"after-click",
                416,
                616,
                source="hcaptcha_iframe",
                challenge_prompt="Select all items that are in the reference",
                submit_label="Check",
            ),
        ]
        decision = LLMCaptchaDecision(
            "actions",
            (
                LLMCaptchaAction("click", 79, 199, None, None),
                LLMCaptchaAction("click", 208, 329, None, None),
            ),
            "all matching tiles",
        )

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(side_effect=captures)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(return_value=decision),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, False, False, False, True]),
            ),
            patch("captcha._click_hcaptcha_submit", new=AsyncMock(return_value=True)) as submit,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        request.assert_awaited_once()
        submit.assert_awaited_once()

    async def test_checkbox_capture_is_never_sent_to_model(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=2,
            max_attempts=1,
        )
        capture = LLMCaptchaCapture(
            b"checkbox",
            318,
            92,
            source="hcaptcha_iframe",
        )

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(return_value=capture)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch("captcha.asyncio.sleep", new=AsyncMock()),
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        request.assert_not_awaited()

    def test_artifacts_deduplicate_captures_and_remove_success_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            solver = LLMCaptchaSolver(
                "vision-model",
                "https://api.example.test/v1",
                "key",
                30,
                artifact_dir=Path(directory),
            )
            solver._start_artifact_session()
            session_dir = solver._active_artifact_dir

            first = solver._save_capture("first.png", b"same-image")
            duplicate = solver._save_capture("duplicate.png", b"same-image")

            self.assertEqual(first, "first.png")
            self.assertEqual(duplicate, "first.png")
            self.assertEqual(len(list(Path(directory).glob("*/*.png"))), 1)
            solver._finish_artifact_session(success=True)
            self.assertIsNotNone(session_dir)
            self.assertFalse(session_dir.exists())  # type: ignore[union-attr]

    def test_artifact_pruning_keeps_only_recent_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            for index in range(22):
                session_dir = artifact_dir / f"20260801-000000-{index:08x}"
                session_dir.mkdir()
                (session_dir / "trace.jsonl").write_text(
                    '{"event":"failed"}\n',
                    encoding="utf-8",
                )
            solved_dir = artifact_dir / "20260801-000001-ffffffff"
            solved_dir.mkdir()
            (solved_dir / "trace.jsonl").write_text(
                '{"event":"solved"}\n',
                encoding="utf-8",
            )
            active_dir = artifact_dir / "20260801-000002-eeeeeeee"
            active_dir.mkdir()
            (active_dir / "trace.jsonl").write_text(
                '{"event":"request"}\n',
                encoding="utf-8",
            )
            stale_dir = artifact_dir / "20260801-000003-dddddddd"
            stale_dir.mkdir()
            stale_trace = stale_dir / "trace.jsonl"
            stale_trace.write_text('{"event":"request"}\n', encoding="utf-8")
            os.utime(stale_trace, (1, 1))
            os.utime(stale_dir, (1, 1))
            solver = LLMCaptchaSolver(
                "vision-model",
                "https://api.example.test/v1",
                "key",
                30,
                artifact_dir=artifact_dir,
            )

            solver._prune_artifact_sessions()

            terminal_failures = []
            for path in artifact_dir.glob("*/trace.jsonl"):
                if '"failed"' in path.read_text(encoding="utf-8"):
                    terminal_failures.append(path)
            self.assertEqual(len(terminal_failures), 20)
            self.assertFalse(solved_dir.exists())
            self.assertTrue(active_dir.exists())
            self.assertFalse(stale_dir.exists())

    async def test_solver_advances_next_after_model_action_without_auto_check(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=3,
            max_attempts=1,
        )
        captures = [
            LLMCaptchaCapture(b"first", 100, 100),
            LLMCaptchaCapture(b"next", 100, 100, submit_label="下一个 next challenge"),
        ]
        decision = LLMCaptchaDecision(
            "actions",
            (LLMCaptchaAction("click", 50, 50, None, None),),
            "select one",
        )

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(side_effect=captures)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(return_value=decision),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, False, False, True]),
            ),
            patch("captcha._click_hcaptcha_submit", new=AsyncMock(return_value=True)) as submit,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        request.assert_awaited_once()
        submit.assert_awaited_once()

    async def test_solver_processes_same_prompt_after_submit_when_image_changes(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=2,
            max_attempts=1,
        )
        captures = [
            LLMCaptchaCapture(
                b"before-submit",
                100,
                100,
                source="hcaptcha_iframe",
                challenge_prompt="Find all animals",
                submit_label="Check",
            ),
            LLMCaptchaCapture(
                b"new-challenge",
                100,
                100,
                source="hcaptcha_iframe",
                challenge_prompt="Find all animals",
                submit_label="Check",
            ),
            LLMCaptchaCapture(
                b"new-challenge",
                100,
                100,
                source="hcaptcha_iframe",
                challenge_prompt="Find all animals",
                submit_label="Check",
            ),
        ]
        decisions = [
            LLMCaptchaDecision("verify", (), "ready"),
            LLMCaptchaDecision(
                "actions",
                (LLMCaptchaAction("click", 50, 50, None, None),),
                "select the target",
            ),
        ]

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(side_effect=captures)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(side_effect=decisions),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, False, False, True]),
            ),
            patch("captcha._click_hcaptcha_submit", new=AsyncMock(return_value=True)) as submit,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        self.assertEqual(request.await_count, 2)
        submit.assert_awaited_once()

    async def test_solver_does_not_requery_model_while_submit_frame_is_stable(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=1,
            max_attempts=1,
        )
        capture = LLMCaptchaCapture(
            b"stable-selected-state",
            416,
            616,
            source="hcaptcha_iframe",
            challenge_prompt="Select all items that are in the reference",
            submit_label="Check",
        )

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(return_value=capture)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(return_value=LLMCaptchaDecision("verify", (), "ready")),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch("captcha._is_captcha_target_enabled", new=AsyncMock(return_value=False)),
            patch("captcha._click_hcaptcha_submit", new=AsyncMock(return_value=True)) as submit,
            patch("captcha.asyncio.sleep", new=AsyncMock()),
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertFalse(solved)
        request.assert_awaited_once()
        submit.assert_awaited_once()

    async def test_solver_processes_stable_unequal_slices_probe(self) -> None:
        solver = LLMCaptchaSolver(
            "vision-model",
            "https://api.example.test/v1",
            "key",
            30,
            call_delay_seconds=0,
            action_delay_seconds=0,
            calls_per_attempt=3,
            max_attempts=1,
        )
        target = LLMCaptchaAction("click", 200, 200, None, None)
        captures = [
            LLMCaptchaCapture(
                b"before-submit",
                1000,
                700,
                source="hcaptcha_canvas",
                challenge_prompt="Please click on the circle where each slice is a different size",
                submit_label="Check",
            ),
            LLMCaptchaCapture(
                b"new-challenge",
                1000,
                700,
                source="hcaptcha_canvas",
                challenge_prompt="Please click on the circle where each slice is a different size",
                submit_label="Check",
                grounded_action=target,
                grounding={
                    "kind": "unequal_slices",
                    "target": {"x": 200, "y": 200},
                },
            ),
            LLMCaptchaCapture(
                b"new-challenge",
                1000,
                700,
                source="hcaptcha_canvas",
                challenge_prompt="Please click on the circle where each slice is a different size",
                submit_label="Check",
                grounded_action=target,
                grounding={
                    "kind": "unequal_slices",
                    "target": {"x": 200, "y": 200},
                },
            ),
        ]

        with (
            patch("captcha._capture_llm_captcha", new=AsyncMock(side_effect=captures)),
            patch.object(
                solver,
                "_request_decision_async",
                new=AsyncMock(
                    return_value=LLMCaptchaDecision("verify", (), "ready")
                ),
            ) as request,
            patch("captcha._click_hcaptcha_checkbox", new=AsyncMock(return_value=False)),
            patch(
                "captcha._is_captcha_target_enabled",
                new=AsyncMock(side_effect=[False, False, False, False, True]),
            ),
            patch(
                "captcha._click_hcaptcha_submit",
                new=AsyncMock(return_value=True),
            ) as submit,
        ):
            solved = await solver.solve(cast(Page, FakePage()))

        self.assertTrue(solved)
        request.assert_awaited_once()
        self.assertEqual(submit.await_count, 2)

    def test_multi_select_classifies_click_on_items_prompts(self) -> None:
        self.assertTrue(
            _is_multi_select_challenge(
                "Click on items that are primarily glass",
                "hcaptcha_iframe",
            )
        )

    async def test_click_checkbox_waits_for_hcaptcha_frame(self) -> None:
        checkbox = Mock()
        checkbox.count = AsyncMock(return_value=1)
        checkbox.is_visible = AsyncMock(return_value=True)
        checkbox.get_attribute = AsyncMock(return_value="false")
        checkbox.click = AsyncMock()
        frame = Mock(url="https://newassets.hcaptcha.com/captcha/v1/checkbox")
        frame.locator.return_value = checkbox
        page = Mock(frames=[frame])

        clicked = await _click_hcaptcha_checkbox(cast(Page, page), timeout_seconds=1)

        self.assertTrue(clicked)
        checkbox.click.assert_awaited_once_with(delay=80, timeout=1000)

    async def test_clicks_non_skip_hcaptcha_submit_control(self) -> None:
        button = Mock()
        button.count = AsyncMock(return_value=1)
        button.is_visible = AsyncMock(return_value=True)
        button.evaluate = AsyncMock(return_value="检查 submit answer")
        button.get_attribute = AsyncMock(return_value=None)
        button.click = AsyncMock()
        locator = Mock()
        locator.first = button
        frame = Mock(
            url="https://newassets.hcaptcha.com/captcha/static/hcaptcha.html#frame=challenge"
        )
        frame.locator.return_value = locator
        page = Mock(frames=[frame])

        clicked = await _click_hcaptcha_submit(cast(Page, page), timeout_seconds=0)

        self.assertTrue(clicked)
        button.click.assert_awaited_once_with(delay=80, timeout=5000)

    async def test_action_maps_native_canvas_coordinates_to_css_page(self) -> None:
        page = FakePage()

        await _apply_llm_action(
            cast(Page, page),
            LLMCaptchaAction("click", 600, 400, None, None),
            offset_x=100,
            offset_y=50,
            scale_x=0.5,
            scale_y=0.5,
        )

        self.assertEqual(page.mouse.calls, [("click", 400, 250, 80)])

    async def test_refreshes_moving_hcaptcha_iframe_before_action(self) -> None:
        page = FakePage()
        page.evaluate = AsyncMock(return_value={"x": 421, "y": 423})
        capture = LLMCaptchaCapture(
            b"png",
            318,
            92,
            offset_x=421,
            offset_y=549,
            source="hcaptcha_iframe",
        )

        refreshed = await _refresh_llm_capture_geometry(cast(Page, page), capture)

        self.assertEqual((refreshed.offset_x, refreshed.offset_y), (421, 423))

    async def test_capture_prefers_native_hcaptcha_canvas(self) -> None:
        png = b"\x89PNG\r\n\x1a\nnative"
        canvas = Mock()
        canvas.count = AsyncMock(return_value=1)
        canvas.is_visible = AsyncMock(return_value=True)
        canvas.evaluate = AsyncMock(
            return_value={
                "dataUrl": "data:image/png;base64," + base64.b64encode(png).decode(),
                "width": 1000,
                "height": 700,
                "cssX": 10,
                "cssY": 10,
                "cssWidth": 500,
                "cssHeight": 350,
                "cropX": 0,
                "cropY": 240,
                "prompt": "Drag ONE animal to the matching silhouette",
                "submitLabel": "跳过 skip challenge",
                "patternGrounding": {
                    "sourceIndex": 1,
                    "targetRow": 3,
                    "targetColumn": 1,
                    "startX": 140,
                    "startY": 117,
                    "endX": 405,
                    "endY": 435,
                    "rowRatio": 3.2,
                    "cellRatio": 2.0,
                    "candidateRatio": 1.8,
                },
            }
        )
        locator = Mock()
        locator.first = canvas
        frame = Mock(
            url="https://newassets.hcaptcha.com/captcha/static/hcaptcha.html#frame=challenge"
        )
        frame.locator.return_value = locator
        page = FakePage()
        page.frames = [frame]
        page.evaluate = AsyncMock(
            return_value={"x": 300, "y": 100, "width": 500, "height": 560}
        )

        capture = await _capture_llm_captcha(cast(Page, page))

        self.assertEqual(capture.source, "hcaptcha_canvas")
        self.assertEqual(capture.content, png)
        self.assertEqual((capture.width, capture.height), (1000, 700))
        self.assertEqual((capture.offset_x, capture.offset_y), (310, 230))
        self.assertEqual((capture.scale_x, capture.scale_y), (0.5, 0.5))
        self.assertEqual(
            capture.challenge_prompt,
            "Drag ONE animal to the matching silhouette",
        )
        self.assertEqual(
            capture.grounded_action,
            LLMCaptchaAction("drag", 140, 117, 405, 435),
        )

    async def test_capture_crops_visible_hcaptcha_iframe(self) -> None:
        page = FakePage()
        page.evaluate = AsyncMock(
            return_value={"x": 300, "y": 100, "width": 500, "height": 600}
        )

        capture = await _capture_llm_captcha(cast(Page, page))

        self.assertEqual(capture.source, "hcaptcha_iframe")
        self.assertEqual((capture.width, capture.height), (500, 600))
        self.assertEqual((capture.offset_x, capture.offset_y), (300, 100))
        self.assertEqual(
            page.last_clip,
            {"x": 300.0, "y": 100.0, "width": 500.0, "height": 600.0},
        )


if __name__ == "__main__":
    unittest.main()
