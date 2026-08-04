from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from urllib.parse import parse_qs, urlparse
from uuid import uuid4
from weakref import WeakKeyDictionary

import requests
from playwright.async_api import Page

from config import CaptchaConfig

_T = TypeVar("_T")


async def _run_with_timeout(
    operation: str,
    timeout_seconds: float,
    callback: Callable[[], Awaitable[_T]],
) -> _T:
    """Put a hard asyncio deadline around Playwright operations."""
    if timeout_seconds <= 0:
        raise TimeoutError(f"{operation} timed out before it started")
    try:
        async with asyncio.timeout(timeout_seconds):
            return await callback()
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{operation} timed out after {timeout_seconds:.1f}s"
        ) from exc


async def _run_until_deadline(
    operation: str,
    deadline: float,
    callback: Callable[[], Awaitable[_T]],
    max_seconds: float,
) -> _T:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"{operation} timed out before it started")
    return await _run_with_timeout(
        operation,
        min(remaining, max_seconds),
        callback,
    )


class CaptchaSolver(Protocol):
    async def solve(self, page: Page) -> bool:
        ...


class ManualCaptchaSolver:
    async def solve(self, page: Page) -> bool:
        print("\n[2/4] Please solve the hCaptcha manually...")
        for i in range(120):
            if await _is_register_button_enabled(page):
                print(f"  hCaptcha solved ({i}s)")
                return True
            await asyncio.sleep(1)
        print("  hCaptcha timeout")
        return False


LLM_CALLS_PER_ATTEMPT = 8
LLM_MAX_ATTEMPTS = 2
LLM_MAX_ACTIONS_PER_CALL = 1
LLM_MAX_MULTI_ACTIONS_PER_CALL = 9
LLM_CALL_DELAY_SECONDS = 1
LLM_ACTION_DELAY_SECONDS = 1
LLM_REQUEST_MAX_ATTEMPTS = 2
LLM_REQUEST_MAX_SECONDS = 30
LLM_TRANSIENT_HTTP_STATUSES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)
LLM_ARTIFACT_CAPTURE_LIMIT = 16
LLM_ARTIFACT_FAILURE_LIMIT = 20
LLM_ARTIFACT_STALE_MIN_SECONDS = 600

LLM_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "actions", "message", "coordinate_space"],
    "properties": {
        "status": {"type": "string", "enum": ["actions", "verify", "solved"]},
        "actions": {
            "type": "array",
            "maxItems": LLM_MAX_ACTIONS_PER_CALL,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "grid_row",
                    "grid_column",
                ],
                "properties": {
                    "kind": {"type": "string", "enum": ["click", "drag"]},
                    "start_x": {"type": "number"},
                    "start_y": {"type": "number"},
                    "end_x": {"type": ["number", "null"]},
                    "end_y": {"type": ["number", "null"]},
                    "grid_row": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "grid_column": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 3,
                    },
                },
            },
        },
        "message": {"type": "string"},
        "coordinate_space": {
            "type": "string",
            "enum": ["normalized_1000"],
        },
    },
}


@dataclass(frozen=True)
class LLMCaptchaAction:
    kind: str
    start_x: float
    start_y: float
    end_x: float | None
    end_y: float | None


@dataclass(frozen=True)
class LLMCaptchaDecision:
    status: str
    actions: tuple[LLMCaptchaAction, ...]
    message: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    raw_output: str = ""
    ignored_action_count: int = 0
    coordinate_space: str = "pixels"


@dataclass(frozen=True)
class LLMCaptchaCapture:
    content: bytes
    width: int
    height: int
    offset_x: float = 0
    offset_y: float = 0
    scale_x: float = 1
    scale_y: float = 1
    source: str = "viewport"
    challenge_prompt: str = ""
    submit_label: str = ""
    grounded_action: LLMCaptchaAction | None = None
    grounding: dict[str, Any] | None = None


class LLMOutputError(ValueError):
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


@dataclass
class LLMCaptchaSolver:
    model: str
    api_base: str
    api_key: str
    timeout_seconds: int
    reasoning_effort: str | None = None
    call_delay_seconds: int = LLM_CALL_DELAY_SECONDS
    action_delay_seconds: int = LLM_ACTION_DELAY_SECONDS
    calls_per_attempt: int = LLM_CALLS_PER_ATTEMPT
    max_attempts: int = LLM_MAX_ATTEMPTS
    max_output_tokens: int = 1200
    api_protocol: str = "responses"
    artifact_dir: Path | None = None
    request_semaphore: asyncio.Semaphore | None = None
    last_error: str | None = None
    _active_artifact_dir: Path | None = field(default=None, init=False, repr=False)
    _saved_capture_names: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _artifact_capture_count: int = field(default=0, init=False, repr=False)

    def _start_artifact_session(self) -> None:
        self._active_artifact_dir = None
        self._saved_capture_names.clear()
        self._artifact_capture_count = 0
        if not self.artifact_dir:
            return
        self._prune_artifact_sessions()
        session_name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        session_dir = self.artifact_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._active_artifact_dir = session_dir
        print(f"  LLM captcha artifacts: {session_dir}")

    def _save_capture(self, filename: str, content: bytes) -> str | None:
        if not self._active_artifact_dir:
            return None
        digest = hashlib.sha256(content).hexdigest()
        if existing_name := self._saved_capture_names.get(digest):
            return existing_name
        if (
            self._artifact_capture_count >= LLM_ARTIFACT_CAPTURE_LIMIT
            and filename != "final.png"
        ):
            return None
        path = self._active_artifact_dir / filename
        path.write_bytes(content)
        path.chmod(0o600)
        self._saved_capture_names[digest] = filename
        self._artifact_capture_count += 1
        return filename

    def _finish_artifact_session(self, success: bool) -> None:
        session_dir = self._active_artifact_dir
        self._active_artifact_dir = None
        if not session_dir:
            return
        if success:
            try:
                shutil.rmtree(session_dir)
            except OSError as exc:
                print(f"  LLM artifact cleanup failed: {exc}")
        self._prune_artifact_sessions()

    def _prune_artifact_sessions(self) -> None:
        if not self.artifact_dir or not self.artifact_dir.is_dir():
            return
        session_pattern = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
        failed_sessions: list[Path] = []
        solved_sessions: list[Path] = []
        stale_before = time.time() - max(
            LLM_ARTIFACT_STALE_MIN_SECONDS,
            self.timeout_seconds * 2,
        )
        for session_dir in self.artifact_dir.iterdir():
            if (
                not session_dir.is_dir()
                or session_dir == self._active_artifact_dir
                or not session_pattern.fullmatch(session_dir.name)
            ):
                continue
            trace_path = session_dir / "trace.jsonl"
            try:
                is_stale = session_dir.stat().st_mtime < stale_before
                lines = trace_path.read_text(encoding="utf-8").splitlines()
                terminal_event = json.loads(lines[-1]).get("event") if lines else None
            except (OSError, json.JSONDecodeError):
                try:
                    if session_dir.stat().st_mtime < stale_before:
                        failed_sessions.append(session_dir)
                except OSError:
                    pass
                continue
            if terminal_event == "solved":
                solved_sessions.append(session_dir)
            elif terminal_event in {"failed", "timeout", "aborted"} or is_stale:
                failed_sessions.append(session_dir)

        failed_sessions.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for session_dir in solved_sessions + failed_sessions[LLM_ARTIFACT_FAILURE_LIMIT:]:
            try:
                shutil.rmtree(session_dir)
            except OSError as exc:
                print(f"  LLM artifact cleanup failed: {exc}")

    def _trace(self, event: str, **details: Any) -> None:
        if not self._active_artifact_dir:
            return
        path = self._active_artifact_dir / "trace.jsonl"
        entry = {"event": event, "time": time.time(), **details}
        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            path.chmod(0o600)
        except OSError as exc:
            print(f"  LLM artifact write failed: {exc}")

    async def solve(self, page: Page) -> bool:
        success = False
        try:
            success = await self._solve_session(page)
            return success
        except Exception as exc:
            self._trace("aborted", error=str(exc))
            raise
        finally:
            self._finish_artifact_session(success)

    async def _solve_session(self, page: Page) -> bool:
        print("\n[2/4] Solving visual captcha with LLM Responses API...")
        self.last_error = None
        self._start_artifact_session()
        deadline = time.monotonic() + self.timeout_seconds
        total_calls = 0
        api_requests = 0
        last_detail = ""
        next_delay = self.call_delay_seconds

        async def page_call(
            operation: str,
            callback: Callable[[], Awaitable[Any]],
            max_seconds: float,
        ) -> Any:
            return await _run_until_deadline(
                operation,
                deadline,
                callback,
                max_seconds,
            )

        for attempt in range(1, self.max_attempts + 1):
            capture_digest_counts: dict[str, int] = {}
            attempt_api_requests = 0
            clicked_points: list[tuple[float, float]] = []
            repeated_action_count = 0
            executed_action_count = 0
            previous_prompt = ""
            previous_submit_label = ""
            retry_feedback = ""
            pending_grounded_actions: list[LLMCaptchaAction] = []
            pending_model_actions: list[LLMCaptchaAction] = []
            pending_batch_submit = False
            awaiting_submit_result = False
            awaiting_submit_digest: str | None = None
            awaiting_submit_probe = False
            awaiting_submit_probe_digest: str | None = None
            awaiting_submit_wait_count = 0
            awaiting_batch_result = False
            print(f"  LLM captcha attempt {attempt}/{self.max_attempts}")
            self._trace("attempt_start", attempt=attempt)
            print("    checking hCaptcha checkbox...", flush=True)
            checkbox_clicked = await _click_hcaptcha_checkbox(
                page,
                timeout_seconds=3,
            )
            print(
                f"    hCaptcha checkbox probe clicked={checkbox_clicked}",
                flush=True,
            )
            if checkbox_clicked:
                print("  hCaptcha checkbox clicked directly; waiting for challenge")
                self._trace("checkbox_click", attempt=attempt)
                next_delay = self.action_delay_seconds
            # A multi-select response can contain up to nine clicks. Once the
            # request budget is exhausted, allow that already-returned queue to
            # drain and confirm the result without issuing another API request.
            max_rounds = self.calls_per_attempt + LLM_MAX_MULTI_ACTIONS_PER_CALL + 2
            for call_number in range(1, max_rounds + 1):
                if attempt_api_requests >= self.calls_per_attempt and not (
                    pending_model_actions
                    or pending_grounded_actions
                    or awaiting_submit_result
                    or (
                        pending_batch_submit
                        and executed_action_count > 0
                    )
                ):
                    break
                await asyncio.sleep(
                    min(next_delay, max(0, deadline - time.monotonic()))
                )
                next_delay = self.call_delay_seconds

                if time.monotonic() >= deadline:
                    print("  LLM captcha timeout")
                    self.last_error = _llm_timeout_error(
                        total_calls,
                        api_requests,
                        last_detail,
                    )
                    self._trace(
                        "timeout",
                        calls=total_calls,
                        api_requests=api_requests,
                        last_detail=last_detail,
                        error=self.last_error,
                    )
                    return False
                try:
                    target_enabled = await page_call(
                        "hCaptcha completion check",
                        lambda: _is_captcha_target_enabled(page),
                        3,
                    )
                except TimeoutError as exc:
                    last_detail = str(exc)
                    self.last_error = _llm_timeout_error(
                        total_calls,
                        api_requests,
                        last_detail,
                    )
                    print(f"  {self.last_error}")
                    self._trace(
                        "timeout",
                        attempt=attempt,
                        call=call_number,
                        calls=total_calls,
                        api_requests=api_requests,
                        last_detail=last_detail,
                        error=self.last_error,
                    )
                    return False
                if target_enabled:
                    print("  captcha target is enabled")
                    self._trace("solved", attempt=attempt, call=call_number)
                    return True

                try:
                    print(
                        f"    capturing hCaptcha ({attempt}/{self.max_attempts}, "
                        f"call {call_number}/{self.calls_per_attempt})...",
                        flush=True,
                    )
                    capture = await page_call(
                        "hCaptcha screenshot capture",
                        lambda: _capture_llm_captcha(page),
                        10,
                    )
                except TimeoutError as exc:
                    last_detail = str(exc)
                    self.last_error = _llm_timeout_error(
                        total_calls,
                        api_requests,
                        last_detail,
                    )
                    print(f"  {self.last_error}")
                    self._trace(
                        "timeout",
                        attempt=attempt,
                        call=call_number,
                        calls=total_calls,
                        api_requests=api_requests,
                        last_detail=last_detail,
                        error=self.last_error,
                    )
                    return False
                screenshot_name = self._save_capture(
                    f"attempt-{attempt:02d}-call-{call_number:02d}.png",
                    capture.content,
                )
                if _is_checkbox_capture(capture):
                    checkbox_clicked = await _click_hcaptcha_checkbox(
                        page,
                        timeout_seconds=1,
                    )
                    self._trace(
                        "checkbox_retry",
                        attempt=attempt,
                        call=call_number,
                        screenshot=screenshot_name,
                        clicked=checkbox_clicked,
                    )
                    # Never ask the model about a checkbox frame. The challenge iframe
                    # can replace it while an API request is in flight, which would map
                    # the old checkbox coordinates into the new challenge dialog.
                    next_delay = max(self.action_delay_seconds, 1)
                    continue
                total_calls += 1
                capture_digest = hashlib.sha256(capture.content).hexdigest()
                challenge_changed = _is_new_hcaptcha_challenge(
                    previous_prompt,
                    previous_submit_label,
                    capture.challenge_prompt,
                    capture.submit_label,
                )
                if challenge_changed:
                    clicked_points.clear()
                    repeated_action_count = 0
                    executed_action_count = 0
                    retry_feedback = ""
                    pending_grounded_actions.clear()
                    pending_model_actions.clear()
                    pending_batch_submit = False
                    # A new challenge may reuse the same prompt and button label.
                    awaiting_submit_result = False
                    awaiting_submit_digest = None
                    awaiting_submit_probe = False
                    awaiting_submit_probe_digest = None
                    awaiting_submit_wait_count = 0
                    awaiting_batch_result = False
                    self._trace(
                        "challenge_changed",
                        attempt=attempt,
                        call=call_number,
                        previous_prompt=previous_prompt,
                        previous_submit_label=previous_submit_label,
                        prompt=capture.challenge_prompt,
                        submit_label=capture.submit_label,
                    )
                previous_prompt = capture.challenge_prompt
                previous_submit_label = capture.submit_label
                capture_digest_counts[capture_digest] = (
                    capture_digest_counts.get(capture_digest, 0) + 1
                )
                repeated_capture_count = capture_digest_counts[capture_digest]
                capture_details = {
                    "source": capture.source,
                    "width": capture.width,
                    "height": capture.height,
                    "offset_x": capture.offset_x,
                    "offset_y": capture.offset_y,
                    "scale_x": capture.scale_x,
                    "scale_y": capture.scale_y,
                    "challenge_prompt": capture.challenge_prompt,
                    "submit_label": capture.submit_label,
                    "grounding": capture.grounding,
                    "sha256": capture_digest,
                    "repeated_capture_count": repeated_capture_count,
                }
                self._trace(
                    "request",
                    attempt=attempt,
                    call=call_number,
                    screenshot=screenshot_name,
                    capture=capture_details,
                )
                if (
                    executed_action_count > 0
                    and _is_next_submit_label(capture.submit_label)
                    and not pending_grounded_actions
                    and not pending_model_actions
                ):
                    submitted = await _click_hcaptcha_submit(
                        page,
                        timeout_seconds=1,
                    )
                    print(f"    hCaptcha next requested (clicked={submitted})")
                    self._trace(
                        "captcha_submit",
                        attempt=attempt,
                        call=call_number,
                        clicked=submitted,
                        automatic=True,
                        label=capture.submit_label,
                        action="next",
                    )
                    if submitted:
                        capture_digest_counts.clear()
                        clicked_points.clear()
                        repeated_action_count = 0
                        executed_action_count = 0
                        pending_grounded_actions.clear()
                        pending_model_actions.clear()
                        pending_batch_submit = False
                        awaiting_submit_result = True
                        awaiting_submit_digest = capture_digest
                        awaiting_submit_probe = False
                        awaiting_submit_probe_digest = None
                        awaiting_submit_wait_count = 0
                        awaiting_batch_result = False
                        next_delay = self.action_delay_seconds
                        continue
                if (
                    pending_batch_submit
                    and executed_action_count > 0
                    and not pending_model_actions
                    and not pending_grounded_actions
                    and _is_actionable_submit_label(capture.submit_label)
                ):
                    submitted = await _click_hcaptcha_submit(
                        page,
                        timeout_seconds=1,
                    )
                    print(f"    hCaptcha multi-select submitted (clicked={submitted})")
                    self._trace(
                        "captcha_submit",
                        attempt=attempt,
                        call=call_number,
                        clicked=submitted,
                        automatic=True,
                        label=capture.submit_label,
                        action="multi_select_batch_complete",
                    )
                    if submitted:
                        capture_digest_counts.clear()
                        clicked_points.clear()
                        repeated_action_count = 0
                        executed_action_count = 0
                        pending_grounded_actions.clear()
                        pending_model_actions.clear()
                        pending_batch_submit = False
                        awaiting_submit_result = True
                        awaiting_submit_digest = capture_digest
                        awaiting_submit_probe = False
                        awaiting_submit_probe_digest = None
                        awaiting_submit_wait_count = 0
                        awaiting_batch_result = True
                        next_delay = self.action_delay_seconds
                        continue
                if awaiting_submit_result:
                    if awaiting_submit_digest is not None and capture_digest == awaiting_submit_digest:
                        awaiting_submit_wait_count += 1
                        self._trace(
                            "submit_transition_wait",
                            attempt=attempt,
                            call=call_number,
                            stable_frames=awaiting_submit_wait_count,
                        )
                        if awaiting_submit_wait_count >= 3:
                            last_detail = (
                                "hCaptcha did not transition after submit for "
                                f"{awaiting_submit_wait_count} stable frames"
                            )
                            self._trace(
                                "submit_transition_stalled",
                                attempt=attempt,
                                call=call_number,
                                error=last_detail,
                            )
                            break
                        next_delay = max(self.action_delay_seconds, 1)
                        continue
                    if (
                        awaiting_submit_probe
                        and awaiting_submit_probe_digest is not None
                        and capture_digest == awaiting_submit_probe_digest
                    ):
                        if (
                            awaiting_batch_result
                            and capture.source == "hcaptcha_iframe"
                            and _is_multi_select_challenge(
                                capture.challenge_prompt,
                                capture.source,
                            )
                        ):
                            last_detail = (
                                "hCaptcha kept the submitted multi-select batch open"
                            )
                            self._trace(
                                "submit_rejected",
                                attempt=attempt,
                                call=call_number,
                                error=last_detail,
                            )
                            break
                        awaiting_submit_result = False
                        awaiting_submit_digest = None
                        awaiting_submit_probe = False
                        awaiting_submit_probe_digest = None
                        awaiting_submit_wait_count = 0
                        awaiting_batch_result = False
                        self._trace(
                            "submit_transition_ready",
                            attempt=attempt,
                            call=call_number,
                            probe=True,
                        )
                        next_delay = self.action_delay_seconds
                    elif not awaiting_submit_probe:
                        # The first changed screenshot can be the same challenge with
                        # hCaptcha's selection marker or loading state painted on it.
                        # Give every challenge one more render cycle before acting on it.
                        awaiting_submit_probe = True
                        awaiting_submit_probe_digest = capture_digest
                        awaiting_submit_wait_count = 0
                        self._trace(
                            "submit_transition_probe",
                            attempt=attempt,
                            call=call_number,
                            previous_digest=awaiting_submit_digest,
                            current_digest=capture_digest,
                        )
                        next_delay = max(self.action_delay_seconds, 1)
                        continue
                    awaiting_submit_result = False
                    awaiting_submit_digest = None
                    awaiting_submit_probe = False
                    awaiting_submit_probe_digest = None
                    awaiting_submit_wait_count = 0
                    awaiting_batch_result = False
                multi_select = _is_multi_select_challenge(
                    capture.challenge_prompt,
                    capture.source,
                )
                if repeated_capture_count >= 3:
                    last_detail = (
                        f"challenge image recurred "
                        f"{repeated_capture_count} times"
                    )
                    print(f"  {last_detail}; ending attempt early")
                    self._trace(
                        "stalled",
                        attempt=attempt,
                        call=call_number,
                        capture=capture_details,
                        error=last_detail,
                    )
                    break
                grounded_actions = _parse_arrow_grounding_actions(
                    capture.grounding,
                    capture.width,
                    capture.height,
                )
                if not grounded_actions:
                    grounded_actions = _parse_find_grounding_actions(
                        capture.grounding,
                        capture.width,
                        capture.height,
                    )
                if grounded_actions and not pending_grounded_actions:
                    pending_grounded_actions.extend(grounded_actions)
                grounded = bool(pending_grounded_actions) or capture.grounded_action is not None
                using_pending_model_action = False
                if pending_model_actions:
                    grounded = False
                    using_pending_model_action = True
                    decision = LLMCaptchaDecision(
                        "actions",
                        (pending_model_actions.pop(0),),
                        "queued multi-select tile",
                        coordinate_space="pixels",
                    )
                elif pending_grounded_actions:
                    decision = LLMCaptchaDecision(
                        "actions",
                        (pending_grounded_actions.pop(0),),
                        "pixel-grounded arrow break",
                        coordinate_space="pixels",
                    )
                elif capture.grounded_action:
                    decision = LLMCaptchaDecision(
                        "actions",
                        (capture.grounded_action,),
                        "pixel-grounded animal pattern",
                        coordinate_space="pixels",
                    )
                else:
                    if attempt_api_requests >= self.calls_per_attempt:
                        last_detail = (
                            "LLM API request budget exhausted after "
                            f"{attempt_api_requests} requests in attempt {attempt}"
                        )
                        self._trace(
                            "request_budget_exhausted",
                            attempt=attempt,
                            call=call_number,
                            calls=total_calls,
                            api_requests=api_requests,
                            attempt_api_requests=attempt_api_requests,
                            error=last_detail,
                        )
                        break
                    try:
                        remaining_seconds = deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            raise TimeoutError("captcha deadline reached before LLM request")
                        api_requests += 1
                        attempt_api_requests += 1
                        print(
                            f"    sending LLM request ({attempt}/{self.max_attempts}, "
                            f"call {call_number}/{self.calls_per_attempt})...",
                            flush=True,
                        )
                        decision = await self._request_decision_async(
                            capture.content,
                            capture.width,
                            capture.height,
                            attempt,
                            call_number,
                            capture.challenge_prompt,
                            capture.submit_label,
                            capture.source,
                            executed_action_count,
                            remaining_seconds,
                            feedback=_llm_retry_feedback(
                                retry_feedback,
                                clicked_points,
                                capture.width,
                                capture.height,
                            ),
                        )
                    except Exception as exc:
                        last_detail = str(exc)
                        print(
                            f"  LLM call {call_number}/{self.calls_per_attempt} failed: {exc}"
                        )
                        raw_output = getattr(exc, "raw_output", "")
                        if raw_output:
                            print(f"    raw output: {_output_preview(raw_output)}")
                        self._trace(
                            "response_error",
                            attempt=attempt,
                            call=call_number,
                            screenshot=screenshot_name,
                            capture=capture_details,
                            error=str(exc),
                            raw_output=raw_output,
                        )
                        continue

                decision = _coerce_dom_grid_decision(decision, capture)
                action_error = _captcha_action_error(decision, capture.challenge_prompt)
                if not action_error:
                    action_error = _fallback_grid_action_error(decision, capture)
                if action_error:
                    last_detail = action_error
                    retry_feedback = (
                        f"{action_error}. The next response MUST use kind=drag with "
                        "both endpoints: start at the draggable animal center and end "
                        "at the matching empty grid cell center. Do not return a click."
                        if "requires drag" in action_error
                        else action_error
                    )
                    print(f"  LLM call {call_number}/{self.calls_per_attempt} rejected: {action_error}")
                    self._trace(
                        "response_error",
                        attempt=attempt,
                        call=call_number,
                        screenshot=screenshot_name,
                        capture=capture_details,
                        error=action_error,
                        raw_output=decision.raw_output,
                    )
                    continue

                last_detail = decision.message or decision.status
                source_label = "pixel grounding" if grounded else "LLM call"
                print(
                    f"  {source_label} {call_number}/{self.calls_per_attempt}: "
                    f"{decision.status}, {len(decision.actions)} action(s)"
                )
                if decision.ignored_action_count:
                    print(
                        f"    ignored {decision.ignored_action_count} stale action(s); "
                        "a fresh screenshot will be used"
                    )
                if decision.coordinate_space != "pixels":
                    print(f"    converted {decision.coordinate_space} coordinates to pixels")
                self._trace(
                    "response",
                    attempt=attempt,
                    call=call_number,
                    screenshot=screenshot_name,
                    capture=capture_details,
                    status=decision.status,
                    message=decision.message,
                    actions=[
                        {
                            "kind": action.kind,
                            "start_x": action.start_x,
                            "start_y": action.start_y,
                            "end_x": action.end_x,
                            "end_y": action.end_y,
                        }
                        for action in decision.actions
                    ],
                    ignored_action_count=decision.ignored_action_count,
                    coordinate_space=decision.coordinate_space,
                    response_id=decision.response_id,
                    usage=decision.usage,
                    raw_output=decision.raw_output,
                )
                if decision.status == "failed":
                    print("    model returned no action; retrying with a fresh call")
                    continue

                if (
                    multi_select
                    and decision.status == "actions"
                    and not using_pending_model_action
                ):
                    # A model can include a tile that was already selected in a
                    # multi-action response. Filter those actions before queuing;
                    # otherwise the controller can submit a partial batch after
                    # rejecting the duplicate first click.
                    batch_actions: list[LLMCaptchaAction] = []
                    batch_had_repeated_click = False
                    batch_points = list(clicked_points)
                    for candidate in decision.actions:
                        if _is_repeated_click(
                            candidate,
                            batch_points,
                            capture.width,
                            capture.height,
                        ):
                            batch_had_repeated_click = True
                            continue
                        batch_actions.append(candidate)
                        if candidate.kind == "click":
                            batch_points.append((candidate.start_x, candidate.start_y))
                    if batch_had_repeated_click:
                        retry_feedback = (
                            "The previous multi-select response included an already selected tile. "
                            "Use only currently unselected tiles from the fresh screenshot, and "
                            "do not verify until every required tile is selected."
                        )
                    decision = replace(decision, actions=tuple(batch_actions))
                    # The schema asks the model for every matching tile in one fresh
                    # screenshot. Drain that batch and submit it without asking the
                    # model to reinterpret its own selection markers.
                    pending_batch_submit = bool(batch_actions)
                    if len(batch_actions) > 1:
                        pending_model_actions.extend(batch_actions[1:])
                        print(
                            f"    queued {len(batch_actions) - 1} additional "
                            "multi-select tile(s)"
                        )
                    if not batch_actions:
                        next_delay = 0
                        continue

                if decision.status == "verify":
                    pending_model_actions.clear()
                    pending_batch_submit = False
                    submitted = await _click_hcaptcha_submit(page)
                    print(f"    hCaptcha submit requested (clicked={submitted})")
                    self._trace(
                        "captcha_submit",
                        attempt=attempt,
                        call=call_number,
                        clicked=submitted,
                    )
                    if submitted:
                        capture_digest_counts.clear()
                        clicked_points.clear()
                        repeated_action_count = 0
                        executed_action_count = 0
                        retry_feedback = ""
                        pending_grounded_actions.clear()
                        pending_batch_submit = False
                        awaiting_submit_result = True
                        awaiting_submit_digest = capture_digest
                        awaiting_submit_probe = False
                        awaiting_submit_probe_digest = None
                        awaiting_submit_wait_count = 0
                        awaiting_batch_result = False
                    next_delay = self.action_delay_seconds
                    continue

                if decision.actions:
                    action = decision.actions[0]
                    if _is_repeated_click(
                        action,
                        clicked_points,
                        capture.width,
                        capture.height,
                    ) and not _is_checkbox_capture(capture):
                        repeated_action_count += 1
                        last_detail = (
                            "model tried to click an already selected item; "
                            "the circled X is a selection marker, not a close button"
                        )
                        print(f"    rejected repeated click ({repeated_action_count})")
                        self._trace(
                            "action_rejected",
                            attempt=attempt,
                            call=call_number,
                            error=last_detail,
                            action={
                                "kind": action.kind,
                                "start_x": action.start_x,
                                "start_y": action.start_y,
                            },
                        )
                        retry_feedback = (
                            "The previous proposed click was already selected. "
                            "Choose a different unselected target from this fresh screenshot, "
                            "or return verify if the required selection is complete. "
                            "For the DOM 3x3 grid, valid tile centers in this crop are "
                            "approximately x=(70, 208, 347) and y=(199, 329, 459) pixels; "
                            "convert one of those centers to normalized_1000 and never "
                            "reuse the selected center."
                        )
                        next_delay = 0
                        continue
                    action_capture = await _refresh_llm_capture_geometry(page, capture)
                    await _apply_llm_action(
                        page,
                        action,
                        offset_x=action_capture.offset_x,
                        offset_y=action_capture.offset_y,
                        scale_x=action_capture.scale_x,
                        scale_y=action_capture.scale_y,
                    )
                    executed_action_count += 1
                    if action.kind == "click":
                        clicked_points.append((action.start_x, action.start_y))
                    retry_feedback = ""
                    self._trace(
                        "action",
                        attempt=attempt,
                        call=call_number,
                        action={
                            "kind": action.kind,
                            "start_x": (
                                action.start_x * action_capture.scale_x
                                + action_capture.offset_x
                            ),
                            "start_y": (
                                action.start_y * action_capture.scale_y
                                + action_capture.offset_y
                            ),
                            "end_x": (
                                action.end_x * action_capture.scale_x
                                + action_capture.offset_x
                                if action.end_x is not None
                                else None
                            ),
                            "end_y": (
                                action.end_y * action_capture.scale_y
                                + action_capture.offset_y
                                if action.end_y is not None
                                else None
                            ),
                        },
                        capture_geometry={
                            "offset_x": action_capture.offset_x,
                            "offset_y": action_capture.offset_y,
                            "scale_x": action_capture.scale_x,
                            "scale_y": action_capture.scale_y,
                        },
                    )
                    try:
                        target_enabled = await page_call(
                            "hCaptcha completion check after action",
                            lambda: _is_captcha_target_enabled(page),
                            3,
                        )
                    except TimeoutError as exc:
                        last_detail = str(exc)
                        self.last_error = _llm_timeout_error(
                            total_calls,
                            api_requests,
                            last_detail,
                        )
                        print(f"  {self.last_error}")
                        self._trace(
                            "timeout",
                            attempt=attempt,
                            call=call_number,
                            calls=total_calls,
                            api_requests=api_requests,
                            last_detail=last_detail,
                            error=self.last_error,
                        )
                        return False
                    if target_enabled:
                        print("  captcha solved by LLM actions")
                        self._trace("solved", attempt=attempt, call=call_number)
                        return True
                    if (
                        action.kind == "click"
                        and _is_actionable_submit_label(capture.submit_label)
                        and (
                            (
                                _is_two_arrow_challenge(capture.challenge_prompt)
                                and len(clicked_points) >= 2
                            )
                            or (
                                _is_find_all_challenge(capture.challenge_prompt)
                                and not pending_grounded_actions
                                and not pending_model_actions
                                and executed_action_count > 0
                            )
                            or (
                                grounded
                                and _is_unequal_slices_challenge(capture.challenge_prompt)
                            )
                            or (
                                pending_batch_submit
                                and not pending_model_actions
                                and executed_action_count > 0
                            )
                        )
                    ):
                        submitted = await _click_hcaptcha_submit(
                            page,
                            timeout_seconds=1,
                        )
                        submit_reason = (
                            "multi_select_batch_complete"
                            if pending_batch_submit
                            else "grounded_selection_complete"
                        )
                        print(
                            "    selection complete; "
                            f"hCaptcha verify clicked={submitted}"
                        )
                        self._trace(
                            "captcha_submit",
                            attempt=attempt,
                            call=call_number,
                            clicked=submitted,
                            automatic=True,
                            label=capture.submit_label,
                            action=submit_reason,
                        )
                        if submitted:
                            capture_digest_counts.clear()
                            clicked_points.clear()
                            repeated_action_count = 0
                            executed_action_count = 0
                            retry_feedback = ""
                            pending_grounded_actions.clear()
                            pending_model_actions.clear()
                            pending_batch_submit = False
                            awaiting_submit_result = True
                            awaiting_submit_digest = capture_digest
                            awaiting_submit_probe = False
                            awaiting_submit_probe_digest = None
                            awaiting_submit_wait_count = 0
                            awaiting_batch_result = (
                                submit_reason == "multi_select_batch_complete"
                            )
                        next_delay = self.action_delay_seconds
                        continue
                    if action.kind == "drag":
                        # The canvas needs a short animation window before its Check
                        # control accepts the newly placed tile.
                        await asyncio.sleep(max(1.5, self.action_delay_seconds))
                        submitted = await _click_hcaptcha_submit(page)
                        print(f"    drag complete; hCaptcha submit clicked={submitted}")
                        self._trace(
                            "captcha_submit",
                            attempt=attempt,
                            call=call_number,
                            clicked=submitted,
                            after_action="drag",
                        )
                        if submitted:
                            capture_digest_counts.clear()
                            clicked_points.clear()
                            repeated_action_count = 0
                            executed_action_count = 0
                            pending_grounded_actions.clear()
                            pending_model_actions.clear()
                            pending_batch_submit = False
                            awaiting_submit_result = True
                            awaiting_submit_digest = capture_digest
                            awaiting_submit_probe = False
                            awaiting_submit_probe_digest = None
                            awaiting_submit_wait_count = 0
                            awaiting_batch_result = False
                    next_delay = self.action_delay_seconds
                    continue

                try:
                    target_enabled = await page_call(
                        "hCaptcha completion check",
                        lambda: _is_captcha_target_enabled(page),
                        3,
                    )
                except TimeoutError as exc:
                    last_detail = str(exc)
                    self.last_error = _llm_timeout_error(
                        total_calls,
                        api_requests,
                        last_detail,
                    )
                    print(f"  {self.last_error}")
                    self._trace(
                        "timeout",
                        attempt=attempt,
                        call=call_number,
                        calls=total_calls,
                        api_requests=api_requests,
                        last_detail=last_detail,
                        error=self.last_error,
                    )
                    return False
                if target_enabled:
                    print("  captcha solved by LLM")
                    self._trace("solved", attempt=attempt, call=call_number)
                    return True
                if decision.status == "solved":
                    print("    model reported solved; waiting for browser confirmation")

            if attempt < self.max_attempts:
                reset = await _reset_hcaptcha(page)
                print(f"  starting next LLM captcha attempt (reset={reset})")
                self._trace("reset", attempt=attempt, reset=reset)
                next_delay = self.call_delay_seconds

        suffix = f"; last result: {last_detail}" if last_detail else ""
        self.last_error = (
            f"LLM captcha failed after {self.max_attempts} attempts and "
            f"{total_calls} decision rounds ({api_requests} API requests){suffix}"
        )
        print(f"  {self.last_error}")
        try:
            final_capture = await _capture_llm_captcha(page)
            final_name = self._save_capture("final.png", final_capture.content)
        except Exception:
            final_name = None
        self._trace("failed", calls=total_calls, error=self.last_error, screenshot=final_name)
        return False

    async def _request_decision_async(
        self,
        screenshot: bytes,
        width: int,
        height: int,
        attempt: int,
        call_number: int,
        challenge_prompt: str = "",
        submit_label: str = "",
        capture_source: str = "viewport",
        executed_action_count: int = 0,
        request_timeout_seconds: float | None = None,
        feedback: str = "",
    ) -> LLMCaptchaDecision:
        async def request(timeout_seconds: float | None) -> LLMCaptchaDecision:
            return await asyncio.to_thread(
                self._request_decision,
                screenshot,
                width,
                height,
                attempt,
                call_number,
                challenge_prompt,
                submit_label,
                capture_source,
                executed_action_count,
                timeout_seconds,
                feedback,
            )

        started_at = time.monotonic()
        acquired = False
        if self.request_semaphore is not None:
            if request_timeout_seconds is None:
                await self.request_semaphore.acquire()
            else:
                await _run_with_timeout(
                    "LLM request queue",
                    request_timeout_seconds,
                    self.request_semaphore.acquire,
                )
            acquired = True
        try:
            remaining = request_timeout_seconds
            if remaining is not None:
                remaining -= time.monotonic() - started_at
                if remaining <= 0:
                    raise TimeoutError("LLM request deadline reached in queue")
            # requests.post has its own bounded timeout. Do not cancel to_thread:
            # cancellation would release the semaphore while the HTTP thread kept
            # running, violating the configured API concurrency limit.
            return await request(remaining)
        finally:
            if acquired and self.request_semaphore is not None:
                self.request_semaphore.release()

    def _request_decision(
        self,
        screenshot: bytes,
        width: int,
        height: int,
        attempt: int,
        call_number: int,
        challenge_prompt: str = "",
        submit_label: str = "",
        capture_source: str = "viewport",
        executed_action_count: int = 0,
        request_timeout_seconds: float | None = None,
        feedback: str = "",
    ) -> LLMCaptchaDecision:
        image_data = base64.b64encode(screenshot).decode("ascii")
        allow_multiple_actions = _is_multi_select_challenge(
            challenge_prompt,
            capture_source,
        )
        action_schema = copy.deepcopy(LLM_ACTION_SCHEMA)
        max_actions = LLM_MAX_ACTIONS_PER_CALL
        if allow_multiple_actions:
            max_actions = LLM_MAX_MULTI_ACTIONS_PER_CALL
            action_schema["properties"]["actions"]["maxItems"] = max_actions
        prompt = _llm_captcha_prompt(
            width,
            height,
            attempt,
            call_number,
            challenge_prompt,
            submit_label,
            capture_source,
            executed_action_count,
            feedback,
            allow_multiple_actions,
        )
        if self.api_protocol == "chat_completions":
            request_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "captcha_actions",
                        "strict": True,
                        "schema": action_schema,
                    },
                },
                "max_tokens": self.max_output_tokens,
                "stream": True,
            }
            if self.reasoning_effort:
                request_payload["reasoning_effort"] = self.reasoning_effort
            endpoint = _chat_completions_endpoint(self.api_base)
            api_name = "Chat Completions API"
        else:
            request_payload = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{image_data}",
                                "detail": "high",
                            },
                        ],
                    }
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "captcha_actions",
                        "strict": True,
                        "schema": action_schema,
                    }
                },
                "max_output_tokens": self.max_output_tokens,
                "stream": True,
            }
            if self.reasoning_effort:
                request_payload["reasoning"] = {"effort": self.reasoning_effort}
            endpoint = _responses_endpoint(self.api_base)
            api_name = "Responses API"
        request_budget = min(
            float(LLM_REQUEST_MAX_SECONDS),
            max(
                0.1,
                request_timeout_seconds
                if request_timeout_seconds is not None
                else float(self.timeout_seconds),
            ),
        )
        request_deadline = time.monotonic() + request_budget
        response: requests.Response | None = None
        last_request_error = ""
        for request_attempt in range(1, LLM_REQUEST_MAX_ATTEMPTS + 1):
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    last_request_error or f"{api_name} request deadline reached"
                )
            try:
                response = requests.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    stream=True,
                    timeout=max(0.1, remaining),
                )
            except requests.RequestException as exc:
                last_request_error = f"{api_name} network error: {exc}"
                response = None
            else:
                if response.ok:
                    break
                detail = response.text[:300].replace("\n", " ")
                last_request_error = (
                    f"{api_name} HTTP {response.status_code}: {detail}"
                )
                if response.status_code not in LLM_TRANSIENT_HTTP_STATUSES:
                    raise RuntimeError(last_request_error)

            if request_attempt >= LLM_REQUEST_MAX_ATTEMPTS:
                raise RuntimeError(last_request_error)
            retry_delay = min(
                float(request_attempt),
                max(0.0, request_deadline - time.monotonic()),
            )
            if (
                response is not None
                and response.status_code == 429
                and isinstance(response.headers, Mapping)
            ):
                try:
                    retry_delay = min(
                        max(0.0, float(response.headers.get("Retry-After", retry_delay))),
                        max(0.0, request_deadline - time.monotonic()),
                    )
                except (TypeError, ValueError):
                    pass
            if retry_delay > 0:
                time.sleep(retry_delay)

        if response is None or not response.ok:
            raise RuntimeError(last_request_error or f"{api_name} request failed")
        try:
            streamed_text, payload = _consume_sse_response(
                response, self.api_protocol, request_deadline
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"{api_name} stream error: {exc}") from exc
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise RuntimeError(f"{api_name} returned a non-object JSON payload")
        output_text = (
            _chat_completion_output_text(payload)
            if self.api_protocol == "chat_completions"
            else _responses_output_text(payload)
        )
        if not output_text:
            output_text = streamed_text
        try:
            decision = _parse_llm_decision(
                output_text,
                width,
                height,
                max_actions=max_actions,
                allow_grid_coordinates=(
                    capture_source == "hcaptcha_iframe"
                    and width >= 400
                    and height >= 500
                ),
            )
        except ValueError as exc:
            decision = _parse_verbose_click_decision(
                output_text,
                width,
                height,
                challenge_prompt,
            )
            if decision is None:
                raise LLMOutputError(str(exc), output_text) from exc
        return replace(
            decision,
            response_id=str(payload["id"]) if payload.get("id") else None,
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            raw_output=output_text,
        )


@dataclass(frozen=True)
class YesCaptchaSolver:
    client_key: str
    api_url: str
    poll_interval_seconds: int
    timeout_seconds: int

    async def solve(self, page: Page) -> bool:
        print("\n[2/4] Solving hCaptcha with YesCaptcha...")
        site_key = await _get_site_key(page)
        if not site_key:
            print("  hCaptcha sitekey not found")
            return False

        task_id = await asyncio.to_thread(self._create_task, page.url, site_key)
        token = await asyncio.to_thread(self._poll_task_result, task_id)
        if not token:
            return False

        await _inject_hcaptcha_token(page, token)
        for i in range(20):
            if await _is_register_button_enabled(page):
                print(f"  hCaptcha solved by YesCaptcha ({i}s)")
                return True
            await asyncio.sleep(1)
        print("  hCaptcha token injected, but #register_button stayed disabled")
        return False

    def _create_task(self, website_url: str, website_key: str) -> str:
        response = requests.post(
            f"{self.api_url}/createTask",
            json={
                "clientKey": self.client_key,
                "task": {
                    "type": "HCaptchaTaskProxyless",
                    "websiteURL": website_url,
                    "websiteKey": website_key,
                },
            },
            timeout=30,
        )
        data = response.json()
        if data.get("errorId"):
            raise RuntimeError(f"YesCaptcha createTask failed: {data}")
        task_id = data.get("taskId")
        if not task_id:
            raise RuntimeError(f"YesCaptcha createTask missing taskId: {data}")
        return str(task_id)

    def _poll_task_result(self, task_id: str) -> str | None:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            response = requests.post(
                f"{self.api_url}/getTaskResult",
                json={"clientKey": self.client_key, "taskId": task_id},
                timeout=30,
            )
            data = response.json()
            if data.get("errorId"):
                print(f"  YesCaptcha getTaskResult failed: {data}")
                return None
            if data.get("status") == "ready":
                solution = data.get("solution") or {}
                return solution.get("gRecaptchaResponse") or solution.get("token")
            time.sleep(self.poll_interval_seconds)
        print("  YesCaptcha timeout")
        return None


@dataclass(frozen=True)
class CaptchaRunSolver:
    token: str
    api_url: str
    poll_interval_seconds: int
    timeout_seconds: int

    async def solve(self, page: Page) -> bool:
        print("\n[2/4] Solving hCaptcha with CaptchaRun...")
        site_key = await _get_site_key(page)
        if not site_key:
            print("  hCaptcha sitekey not found")
            return False

        user_agent = await page.evaluate("() => navigator.userAgent")
        task_id, token = await asyncio.to_thread(
            self._create_task,
            page.url,
            site_key,
            user_agent,
        )
        if task_id and not token:
            token = await asyncio.to_thread(self._poll_task_result, task_id)
        if not token:
            return False

        await _inject_hcaptcha_token(page, token)
        for i in range(20):
            if await _is_register_button_enabled(page):
                print(f"  hCaptcha solved by CaptchaRun ({i}s)")
                return True
            await asyncio.sleep(1)
        print("  hCaptcha token injected, but #register_button stayed disabled")
        return False

    def _create_task(self, website_url: str, website_key: str, user_agent: str) -> tuple[str | None, str | None]:
        response = requests.post(
            f"{self.api_url}/v2/tasks",
            headers=self._headers(),
            json={
                "captchaType": "HCaptcha",
                "siteKey": website_key,
                "siteReferer": _site_referer(website_url),
                "userAgent": user_agent,
                "fallbackToActualUA": True,
            },
            timeout=30,
        )
        data = _response_json(response)
        if not response.ok:
            raise RuntimeError(f"CaptchaRun create task failed: {data}")
        task_id = data.get("taskId")
        result = data.get("result") or {}
        token = _extract_hcaptcha_token(result)
        if not task_id and not token:
            raise RuntimeError(f"CaptchaRun create task missing taskId/result: {data}")
        return str(task_id) if task_id else None, token

    def _poll_task_result(self, task_id: str) -> str | None:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            response = requests.get(
                f"{self.api_url}/v2/tasks/{task_id}",
                headers=self._headers(content_type=False),
                timeout=30,
            )
            data = _response_json(response)
            if not response.ok:
                print(f"  CaptchaRun get task result failed: {data}")
                return None

            status = str(data.get("status", "")).lower()
            if status == "success":
                return _extract_hcaptcha_token(data.get("response") or data.get("result") or {})
            if status == "fail":
                print(f"  CaptchaRun failed: {data.get('reason') or data}")
                return None
            time.sleep(self.poll_interval_seconds)
        print("  CaptchaRun timeout")
        return None

    def _headers(self, content_type: bool = True) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers


async def _click_hcaptcha_checkbox(page: Page, timeout_seconds: float = 15) -> bool:
    async def scan(deadline: float | None) -> bool:
        for frame in getattr(page, "frames", []):
            if "hcaptcha" not in frame.url.lower():
                continue
            for selector in ("#checkbox", '[role="checkbox"]'):
                if deadline is None:
                    click_timeout_ms = 5000
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    click_timeout_ms = max(
                        1,
                        min(5000, math.ceil(remaining * 1000)),
                    )
                checkbox = frame.locator(selector)
                try:
                    if await checkbox.count() < 1 or not await checkbox.is_visible():
                        continue
                    if await checkbox.get_attribute("aria-checked") == "true":
                        return False
                    await checkbox.click(delay=80, timeout=click_timeout_ms)
                    return True
                except Exception:
                    continue
        return False

    # timeout=0 is useful for a single immediate probe and is kept for callers
    # that explicitly request one scan.
    if timeout_seconds <= 0:
        return await scan(None)

    deadline = time.monotonic() + timeout_seconds
    poll_seconds = 0.5
    while True:
        if await scan(deadline):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(poll_seconds, remaining))


async def _read_hcaptcha_challenge_metadata(page: Page) -> tuple[str, str]:
    for frame in getattr(page, "frames", []):
        if "hcaptcha" not in frame.url.lower():
            continue
        prompt_text = ""
        submit_label = ""
        try:
            prompt = frame.locator("#prompt-question").first
            if await prompt.count() > 0 and await prompt.is_visible():
                prompt_text = (await prompt.inner_text(timeout=500)).strip()
        except Exception:
            pass
        try:
            submit = frame.locator(".button-submit").first
            if await submit.count() > 0 and await submit.is_visible():
                submit_label = str(
                    await submit.evaluate(
                        """(element) => [element.textContent, element.getAttribute('aria-label')]
                            .filter(Boolean).join(' ').replace(/\\s+/g, ' ').trim()"""
                    )
                )
        except Exception:
            pass
        if prompt_text or submit_label:
            return prompt_text, submit_label
    return "", ""


async def _capture_llm_captcha(page: Page) -> LLMCaptchaCapture:
    native_capture = await _capture_native_hcaptcha_canvas(page)
    if native_capture:
        return native_capture

    viewport = page.viewport_size or await page.evaluate(
        "() => ({width: window.innerWidth, height: window.innerHeight})"
    )
    viewport_width = int(viewport["width"])
    viewport_height = int(viewport["height"])
    rect = await page.evaluate(
        """() => {
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const candidates = Array.from(document.querySelectorAll('iframe'))
                .map((iframe) => {
                    const marker = [iframe.src, iframe.title, iframe.name]
                        .filter(Boolean).join(' ').toLowerCase();
                    if (!marker.includes('hcaptcha')) return null;
                    const style = getComputedStyle(iframe);
                    const box = iframe.getBoundingClientRect();
                    if (style.display === 'none' || style.visibility === 'hidden' ||
                        Number(style.opacity || 1) === 0 || box.width < 20 || box.height < 20) {
                        return null;
                    }
                    const left = Math.max(0, box.left);
                    const top = Math.max(0, box.top);
                    const right = Math.min(viewportWidth, box.right);
                    const bottom = Math.min(viewportHeight, box.bottom);
                    if (right <= left || bottom <= top) return null;
                    return {left, top, right, bottom, area: (right - left) * (bottom - top)};
                })
                .filter(Boolean)
                .sort((a, b) => b.area - a.area);
            if (!candidates.length) return null;
            const box = candidates[0];
            const padding = 8;
            const x = Math.max(0, Math.floor(box.left) - padding);
            const y = Math.max(0, Math.floor(box.top) - padding);
            const right = Math.min(viewportWidth, Math.ceil(box.right) + padding);
            const bottom = Math.min(viewportHeight, Math.ceil(box.bottom) + padding);
            return {x, y, width: right - x, height: bottom - y};
        }"""
    )
    if isinstance(rect, dict):
        values = [rect.get(key) for key in ("x", "y", "width", "height")]
        if all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in values
        ):
            x, y, width, height = (
                float(cast(int | float, value)) for value in values
            )
            if width >= 20 and height >= 20:
                content = await page.screenshot(
                    type="png",
                    clip={"x": x, "y": y, "width": width, "height": height},
                )
                challenge_prompt, submit_label = await _read_hcaptcha_challenge_metadata(page)
                image_width, image_height = _png_dimensions(content) or (
                    round(width),
                    round(height),
                )
                return LLMCaptchaCapture(
                    content=content,
                    width=image_width,
                    height=image_height,
                    offset_x=x,
                    offset_y=y,
                    scale_x=width / image_width,
                    scale_y=height / image_height,
                    source="hcaptcha_iframe",
                    challenge_prompt=challenge_prompt,
                    submit_label=submit_label,
                )

    content = await page.screenshot(type="png")
    return LLMCaptchaCapture(content, viewport_width, viewport_height)


def _png_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    return (width, height) if width > 0 and height > 0 else None


async def _refresh_llm_capture_geometry(
    page: Page,
    capture: LLMCaptchaCapture,
) -> LLMCaptchaCapture:
    """Refresh a moving iframe's page offset immediately before mouse input."""
    if capture.source != "hcaptcha_iframe":
        return capture
    try:
        rect = await page.evaluate(
            """() => {
                const viewportWidth = window.innerWidth;
                const viewportHeight = window.innerHeight;
                const candidates = Array.from(document.querySelectorAll('iframe'))
                    .map((iframe) => {
                        const marker = [iframe.src, iframe.title, iframe.name]
                            .filter(Boolean).join(' ').toLowerCase();
                        if (!marker.includes('hcaptcha')) return null;
                        const style = getComputedStyle(iframe);
                        const box = iframe.getBoundingClientRect();
                        if (style.display === 'none' || style.visibility === 'hidden' ||
                            Number(style.opacity || 1) === 0 || box.width < 20 || box.height < 20) {
                            return null;
                        }
                        const left = Math.max(0, box.left);
                        const top = Math.max(0, box.top);
                        const right = Math.min(viewportWidth, box.right);
                        const bottom = Math.min(viewportHeight, box.bottom);
                        if (right <= left || bottom <= top) return null;
                        return {left, top, right, bottom, area: (right - left) * (bottom - top)};
                    })
                    .filter(Boolean)
                    .sort((a, b) => b.area - a.area);
                if (!candidates.length) return null;
                const box = candidates[0];
                return {
                    x: Math.max(0, Math.floor(box.left) - 8),
                    y: Math.max(0, Math.floor(box.top) - 8),
                };
            }"""
        )
        if not isinstance(rect, dict):
            return capture
        x = rect.get("x")
        y = rect.get("y")
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or isinstance(y, bool)
            or not isinstance(y, (int, float))
        ):
            return capture
        return replace(capture, offset_x=float(x), offset_y=float(y))
    except Exception:
        return capture


async def _capture_native_hcaptcha_canvas(page: Page) -> LLMCaptchaCapture | None:
    challenge_frame = next(
        (
            frame
            for frame in getattr(page, "frames", [])
            if "hcaptcha" in frame.url.lower() and "frame=challenge" in frame.url.lower()
        ),
        None,
    )
    if challenge_frame is None:
        return None

    try:
        iframe_rect = await page.evaluate(
            """() => {
                const candidates = Array.from(document.querySelectorAll('iframe'))
                    .map((iframe) => {
                        const marker = [iframe.src, iframe.title, iframe.name]
                            .filter(Boolean).join(' ').toLowerCase();
                        if (!marker.includes('hcaptcha') || !marker.includes('frame=challenge')) {
                            return null;
                        }
                        const style = getComputedStyle(iframe);
                        const box = iframe.getBoundingClientRect();
                        if (style.display === 'none' || style.visibility === 'hidden' ||
                            Number(style.opacity || 1) === 0 || box.width < 20 || box.height < 20) {
                            return null;
                        }
                        return {x: box.left, y: box.top, width: box.width, height: box.height};
                    })
                    .filter(Boolean)
                    .sort((a, b) => b.width * b.height - a.width * a.height);
                return candidates[0] || null;
            }"""
        )
        if not isinstance(iframe_rect, dict):
            return None

        canvas = challenge_frame.locator("canvas").first
        if await canvas.count() < 1 or not await canvas.is_visible():
            return None
        canvas_data = await canvas.evaluate(
            r"""(canvas) => {
                const box = canvas.getBoundingClientRect();
                const prompt = document.querySelector('#prompt-question');
                const submit = document.querySelector('.button-submit');
                const promptText = prompt
                    ? prompt.textContent.replace(/\s+/g, ' ').trim()
                    : '';
                const label = submit
                    ? [submit.textContent, submit.getAttribute('aria-label')]
                        .filter(Boolean).join(' ').replace(/\s+/g, ' ').trim()
                    : '';
                let patternGrounding = null;
                let arrowGrounding = null;
                let findGrounding = null;
                let circleGrounding = null;
                const patternPrompt = /(?:place|drag) the correct animal.*empty spot/i.test(promptText);
                const actionableLabel = /skip/i.test(label) || label.includes('跳过');
                const probe = document.createElement('canvas');
                probe.width = canvas.width;
                probe.height = canvas.height;
                const context = probe.getContext('2d', {willReadFrequently: true});
                context.drawImage(canvas, 0, 0);
                const pixels = context.getImageData(
                    0, 0, probe.width, probe.height
                ).data;
                // Grounding coordinates must use the same cropped image that is sent to
                // the model. The native canvas often has a transparent header before the
                // visible challenge, so calculating from probe.height shifts every row.
                let cropX = 0;
                let cropY = 0;
                let cropWidth = canvas.width;
                let cropHeight = canvas.height;
                try {
                    const sourceContext = canvas.getContext('2d', {willReadFrequently: true});
                    const sourcePixels = sourceContext.getImageData(
                        0, 0, canvas.width, canvas.height
                    ).data;
                    let left = canvas.width;
                    let top = canvas.height;
                    let right = 0;
                    let bottom = 0;
                    for (let y = 0; y < canvas.height; y += 1) {
                        for (let x = 0; x < canvas.width; x += 1) {
                            if (sourcePixels[(y * canvas.width + x) * 4 + 3] <= 8) continue;
                            left = Math.min(left, x);
                            top = Math.min(top, y);
                            right = Math.max(right, x + 1);
                            bottom = Math.max(bottom, y + 1);
                        }
                    }
                    if (right - left >= 20 && bottom - top >= 20) {
                        cropX = left;
                        cropY = top;
                        cropWidth = right - left;
                        cropHeight = bottom - top;
                    }
                } catch (_) {}
                const groundingWidth = Math.max(1, cropWidth);
                const groundingHeight = Math.max(1, cropHeight);
                const groundingX = (value) => cropX + Math.round(value * groundingWidth);
                const groundingY = (value) => cropY + Math.round(value * groundingHeight);
                if (patternPrompt && actionableLabel) {
                    try {
                        const gridXs = [0.405, 0.55, 0.70, 0.85]
                            .map((value) => groundingX(value));
                        const gridYs = [0.40, 0.56, 0.72, 0.88]
                            .map((value) => groundingY(value));
                        const candidateX = groundingX(0.14);
                        const candidateYs = [0.38, 0.56]
                            .map((value) => groundingY(value));
                        const radius = Math.max(20, Math.round(0.038 * groundingWidth));
                        const histogram = (centerX, centerY) => {
                            const bins = 4;
                            const values = Array(bins * bins * bins).fill(0);
                            let samples = 0;
                            for (let dy = -radius; dy <= radius; dy += 2) {
                                for (let dx = -radius; dx <= radius; dx += 2) {
                                    if (dx * dx + dy * dy > radius * radius) continue;
                                    const x = centerX + dx;
                                    const y = centerY + dy;
                                    if (x < 0 || y < 0 || x >= probe.width || y >= probe.height) continue;
                                    const offset = (y * probe.width + x) * 4;
                                    const red = Math.min(bins - 1, pixels[offset] * bins >> 8);
                                    const green = Math.min(bins - 1, pixels[offset + 1] * bins >> 8);
                                    const blue = Math.min(bins - 1, pixels[offset + 2] * bins >> 8);
                                    values[(red * bins + green) * bins + blue] += 1;
                                    samples += 1;
                                }
                            }
                            return values.map((value) => value / Math.max(1, samples));
                        };
                        const histogramDistance = (first, second) => first.reduce(
                            (total, value, index) => total + Math.abs(value - second[index]),
                            0,
                        );
                        const cellHistograms = gridYs.map((gridY) =>
                            gridXs.map((gridX) => histogram(gridX, gridY))
                        );
                        const candidates = candidateYs.map((candidateY) =>
                            histogram(candidateX, candidateY)
                        );
                        const lineMatches = [];
                        for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
                            for (const orientation of ['row', 'column']) {
                                for (let line = 0; line < 4; line += 1) {
                                    const distances = [];
                                    for (let item = 0; item < 4; item += 1) {
                                        const cell = orientation === 'row'
                                            ? cellHistograms[line][item]
                                            : cellHistograms[item][line];
                                        distances.push(histogramDistance(candidates[candidateIndex], cell));
                                    }
                                    const ranked = [...distances].sort((a, b) => a - b);
                                    const matchAverage = (ranked[0] + ranked[1] + ranked[2]) / 3;
                                    const outlierGap = ranked[3] - matchAverage;
                                    lineMatches.push({
                                        candidateIndex,
                                        orientation,
                                        line,
                                        distances,
                                        matchAverage,
                                        outlierGap,
                                    });
                                }
                            }
                        }
                        lineMatches.sort((first, second) =>
                            first.matchAverage - second.matchAverage ||
                            second.outlierGap - first.outlierGap
                        );
                        const best = lineMatches[0];
                        const second = lineMatches[1];
                        if (best && second &&
                            best.outlierGap >= Math.max(0.12, best.matchAverage * 0.25) &&
                            second.matchAverage - best.matchAverage >= 0.12) {
                            const targetIndex = best.distances.indexOf(Math.max(...best.distances));
                            const targetRow = best.orientation === 'row' ? best.line : targetIndex;
                            const targetColumn = best.orientation === 'row' ? targetIndex : best.line;
                            patternGrounding = {
                                kind: 'pattern',
                                sourceIndex: best.candidateIndex + 1,
                                targetRow: targetRow + 1,
                                targetColumn: targetColumn + 1,
                                startX: candidateX,
                                startY: candidateYs[best.candidateIndex],
                                endX: gridXs[targetColumn],
                                endY: gridYs[targetRow],
                                matchAverage: best.matchAverage,
                                outlierGap: best.outlierGap,
                            };
                        }
                    } catch (_) {}
                }
                if (/click the two arrows|两个箭头/i.test(promptText) && label) {
                    try {
                        // Arrow anomalies are the minority shape in the chain. Bright neutral
                        // outline components are stable even when the canvas background changes.
                        const sampleStep = 2;
                        const sampleWidth = Math.ceil(probe.width / sampleStep);
                        const sampleHeight = Math.ceil(probe.height / sampleStep);
                        const mask = new Uint8Array(sampleWidth * sampleHeight);
                        const isBrightNeutral = (x, y) => {
                            const offset = (y * probe.width + x) * 4;
                            const red = pixels[offset];
                            const green = pixels[offset + 1];
                            const blue = pixels[offset + 2];
                            const minimum = Math.min(red, green, blue);
                            const maximum = Math.max(red, green, blue);
                            return pixels[offset + 3] > 8 && minimum >= 150 && maximum - minimum <= 55;
                        };
                        for (let y = 0; y < sampleHeight; y += 1) {
                            for (let x = 0; x < sampleWidth; x += 1) {
                                let found = false;
                                for (let dy = 0; dy < sampleStep && !found; dy += 1) {
                                    for (let dx = 0; dx < sampleStep; dx += 1) {
                                        const sourceX = x * sampleStep + dx;
                                        const sourceY = y * sampleStep + dy;
                                        if (sourceX < probe.width && sourceY < probe.height &&
                                            isBrightNeutral(sourceX, sourceY)) {
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                mask[y * sampleWidth + x] = found ? 1 : 0;
                            }
                        }
                        const seen = new Uint8Array(mask.length);
                        const components = [];
                        for (let y = 0; y < sampleHeight; y += 1) {
                            for (let x = 0; x < sampleWidth; x += 1) {
                                const start = y * sampleWidth + x;
                                if (!mask[start] || seen[start]) continue;
                                const stack = [start];
                                seen[start] = 1;
                                let count = 0;
                                let minX = x;
                                let maxX = x;
                                let minY = y;
                                let maxY = y;
                                while (stack.length) {
                                    const current = stack.pop();
                                    const currentY = Math.floor(current / sampleWidth);
                                    const currentX = current - currentY * sampleWidth;
                                    count += 1;
                                    minX = Math.min(minX, currentX);
                                    maxX = Math.max(maxX, currentX);
                                    minY = Math.min(minY, currentY);
                                    maxY = Math.max(maxY, currentY);
                                    for (let dy = -1; dy <= 1; dy += 1) {
                                        for (let dx = -1; dx <= 1; dx += 1) {
                                            if (dx === 0 && dy === 0) continue;
                                            const nextX = currentX + dx;
                                            const nextY = currentY + dy;
                                            if (nextX < 0 || nextY < 0 ||
                                                nextX >= sampleWidth || nextY >= sampleHeight) continue;
                                            const next = nextY * sampleWidth + nextX;
                                            if (mask[next] && !seen[next]) {
                                                seen[next] = 1;
                                                stack.push(next);
                                            }
                                        }
                                    }
                                }
                                const componentWidth = (maxX - minX + 1) * sampleStep;
                                const componentHeight = (maxY - minY + 1) * sampleStep;
                                const componentArea = componentWidth * componentHeight;
                                if (count >= 100 && count <= 3000 &&
                                    componentWidth >= 20 && componentWidth <= 150 &&
                                    componentHeight >= 15 && componentHeight <= 130 &&
                                    componentArea >= 1600 && componentArea <= 12000) {
                                    components.push({
                                        x: (minX + maxX + 1) * sampleStep / 2,
                                        y: (minY + maxY + 1) * sampleStep / 2,
                                        area: componentArea,
                                    });
                                }
                            }
                        }
                        components.sort((first, second) => second.area - first.area);
                        const arrows = [];
                        for (const component of components) {
                            if (arrows.every((arrow) =>
                                Math.hypot(arrow.x - component.x, arrow.y - component.y) >= 32
                            )) {
                                arrows.push(component);
                            }
                        }
                        if (arrows.length >= 7 && arrows.length <= 12) {
                            arrows.sort((first, second) => first.area - second.area);
                            let split = -1;
                            let splitRatio = 0;
                            for (let index = 1; index < arrows.length; index += 1) {
                                const leftCount = index;
                                const rightCount = arrows.length - index;
                                if (leftCount !== 2 && rightCount !== 2) continue;
                                const leftArea = arrows[index - 1].area;
                                const rightArea = arrows[index].area;
                                const ratio = Math.max(leftArea, rightArea) /
                                    Math.max(1, Math.min(leftArea, rightArea));
                                if (ratio > splitRatio) {
                                    splitRatio = ratio;
                                    split = index;
                                }
                            }
                            if (split >= 0 && splitRatio >= 1.35) {
                                const anomalies = split === 2
                                    ? arrows.slice(0, 2)
                                    : arrows.slice(split);
                                if (anomalies.length === 2) {
                                    arrowGrounding = {
                                        kind: 'arrow',
                                        targets: anomalies.map((arrow) => ({
                                            x: Math.round(arrow.x),
                                            y: Math.round(arrow.y),
                                        })),
                                        candidateCount: arrows.length,
                                        sizeRatio: splitRatio,
                                    };
                                }
                            }
                        }
                    } catch (_) {}
                }
                if (/(?:find all (?:animal )?(?:icons?|animals?)|找到所有动物)/i.test(promptText) && label) {
                    try {
                        const radius = Math.max(20, Math.round(0.038 * groundingWidth));
                        const histogram = (centerX, centerY) => {
                            const bins = 4;
                            const values = Array(bins * bins * bins).fill(0);
                            let samples = 0;
                            for (let dy = -radius; dy <= radius; dy += 2) {
                                for (let dx = -radius; dx <= radius; dx += 2) {
                                    if (dx * dx + dy * dy > radius * radius) continue;
                                    const x = centerX + dx;
                                    const y = centerY + dy;
                                    if (x < 0 || y < 0 || x >= probe.width || y >= probe.height) continue;
                                    const offset = (y * probe.width + x) * 4;
                                    const red = Math.min(bins - 1, pixels[offset] * bins >> 8);
                                    const green = Math.min(bins - 1, pixels[offset + 1] * bins >> 8);
                                    const blue = Math.min(bins - 1, pixels[offset + 2] * bins >> 8);
                                    values[(red * bins + green) * bins + blue] += 1;
                                    samples += 1;
                                }
                            }
                            return values.map((value) => value / Math.max(1, samples));
                        };
                        const distance = (first, second) => first.reduce(
                            (total, value, index) => total + Math.abs(value - second[index]),
                            0,
                        );
                        // hCaptcha uses more than one native layout for this task. The
                        // 5x3 layout is common, while older challenges use 4x3 or 4x4.
                        // Try all known geometries and keep the distinct low-score match.
                        const layouts = [
                            {
                                gridXs: [0.208, 0.352, 0.496, 0.640, 0.784],
                                gridYs: [0.170, 0.375, 0.580],
                                legendXs: [0.215, 0.500, 0.785],
                                legendY: 0.860,
                            },
                            {
                                gridXs: [0.280, 0.425, 0.570, 0.715],
                                gridYs: [0.170, 0.375, 0.580],
                                legendXs: [0.330, 0.800],
                                legendY: 0.860,
                            },
                            {
                                gridXs: [0.405, 0.550, 0.700, 0.850],
                                gridYs: [0.200, 0.400, 0.600, 0.800],
                                legendXs: [0.250, 0.750],
                                legendY: 0.900,
                            },
                        ];
                        let bestLayout = null;
                        for (const layout of layouts) {
                            const gridXs = layout.gridXs
                                .map((value) => groundingX(value));
                            const gridYs = layout.gridYs
                                .map((value) => groundingY(value));
                            const legendXs = layout.legendXs
                                .map((value) => groundingX(value));
                            const legendY = groundingY(layout.legendY);
                            const cells = [];
                            for (let row = 0; row < gridYs.length; row += 1) {
                                for (let column = 0; column < gridXs.length; column += 1) {
                                    cells.push({
                                        row,
                                        column,
                                        signature: histogram(gridXs[column], gridYs[row]),
                                    });
                                }
                            }
                            const targets = [];
                            for (let targetIndex = 0; targetIndex < legendXs.length; targetIndex += 1) {
                                const targetSignature = histogram(legendXs[targetIndex], legendY);
                                const ranked = cells.map((cell) => ({
                                    ...cell,
                                    score: distance(targetSignature, cell.signature),
                                })).sort((first, second) => first.score - second.score);
                                const best = ranked[0];
                                if (best && best.score <= 1.15) {
                                    targets.push({
                                        x: gridXs[best.column],
                                        y: gridYs[best.row],
                                        score: best.score,
                                    });
                                }
                            }
                            const distinctTargets = targets.length === legendXs.length &&
                                targets.every((target, index) =>
                                    targets.slice(0, index).every((previous) =>
                                        Math.hypot(target.x - previous.x, target.y - previous.y) >= 32
                                    )
                                );
                            if (!distinctTargets) continue;
                            const score = targets.reduce(
                                (total, target) => total + target.score,
                                0,
                            ) / Math.max(1, targets.length);
                            if (!bestLayout ||
                                score < bestLayout.score - 0.02 ||
                                (Math.abs(score - bestLayout.score) <= 0.02 &&
                                    targets.length > bestLayout.targets.length)) {
                                bestLayout = {targets, score};
                            }
                        }
                        if (bestLayout) {
                            arrowGrounding = null;
                            patternGrounding = null;
                            findGrounding = {
                                kind: 'find_all',
                                targets: bestLayout.targets,
                                score: bestLayout.score,
                            };
                        }
                    } catch (_) {}
                }
                if (/circle where each slice is a different size|circle with uneven slices|每个.*(?:切片|扇形).*大小.*不同/i.test(promptText) && label) {
                    try {
                        // Pie charts are dark wedges separated by bright radial lines.
                        const sampleStep = 2;
                        const sampleWidth = Math.ceil(probe.width / sampleStep);
                        const sampleHeight = Math.ceil(probe.height / sampleStep);
                        const mask = new Uint8Array(sampleWidth * sampleHeight);
                        const isDarkObject = (x, y) => {
                            const offset = (y * probe.width + x) * 4;
                            const red = pixels[offset];
                            const green = pixels[offset + 1];
                            const blue = pixels[offset + 2];
                            const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;
                            return pixels[offset + 3] > 8 && luminance < 140;
                        };
                        for (let y = 0; y < sampleHeight; y += 1) {
                            for (let x = 0; x < sampleWidth; x += 1) {
                                let found = false;
                                for (let dy = 0; dy < sampleStep && !found; dy += 1) {
                                    for (let dx = 0; dx < sampleStep; dx += 1) {
                                        const sourceX = x * sampleStep + dx;
                                        const sourceY = y * sampleStep + dy;
                                        if (sourceX < probe.width && sourceY < probe.height &&
                                            isDarkObject(sourceX, sourceY)) {
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                                mask[y * sampleWidth + x] = found ? 1 : 0;
                            }
                        }
                        const seen = new Uint8Array(mask.length);
                        const components = [];
                        for (let y = 0; y < sampleHeight; y += 1) {
                            for (let x = 0; x < sampleWidth; x += 1) {
                                const start = y * sampleWidth + x;
                                if (!mask[start] || seen[start]) continue;
                                const stack = [start];
                                seen[start] = 1;
                                let count = 0;
                                let sumX = 0;
                                let sumY = 0;
                                let minX = x;
                                let maxX = x;
                                let minY = y;
                                let maxY = y;
                                while (stack.length) {
                                    const current = stack.pop();
                                    const currentY = Math.floor(current / sampleWidth);
                                    const currentX = current - currentY * sampleWidth;
                                    count += 1;
                                    sumX += currentX;
                                    sumY += currentY;
                                    minX = Math.min(minX, currentX);
                                    maxX = Math.max(maxX, currentX);
                                    minY = Math.min(minY, currentY);
                                    maxY = Math.max(maxY, currentY);
                                    for (let dy = -1; dy <= 1; dy += 1) {
                                        for (let dx = -1; dx <= 1; dx += 1) {
                                            if (dx === 0 && dy === 0) continue;
                                            const nextX = currentX + dx;
                                            const nextY = currentY + dy;
                                            if (nextX < 0 || nextY < 0 ||
                                                nextX >= sampleWidth || nextY >= sampleHeight) continue;
                                            const next = nextY * sampleWidth + nextX;
                                            if (mask[next] && !seen[next]) {
                                                seen[next] = 1;
                                                stack.push(next);
                                            }
                                        }
                                    }
                                }
                                const componentWidth = (maxX - minX + 1) * sampleStep;
                                const componentHeight = (maxY - minY + 1) * sampleStep;
                                if (count >= 500 && count <= 4000 &&
                                    componentWidth >= 40 && componentWidth <= 180 &&
                                    componentHeight >= 40 && componentHeight <= 180) {
                                    components.push({
                                        x: sumX / count * sampleStep,
                                        y: sumY / count * sampleStep,
                                        minX: minX * sampleStep,
                                        maxX: (maxX + 1) * sampleStep,
                                        minY: minY * sampleStep,
                                        maxY: (maxY + 1) * sampleStep,
                                    });
                                }
                            }
                        }
                        const clusters = [];
                        for (const component of components) {
                            let cluster = clusters.find((candidate) =>
                                Math.hypot(candidate.x - component.x, candidate.y - component.y) < 115
                            );
                            if (!cluster) {
                                cluster = {
                                    parts: [],
                                    x: component.x,
                                    y: component.y,
                                    minX: component.minX,
                                    maxX: component.maxX,
                                    minY: component.minY,
                                    maxY: component.maxY,
                                };
                                clusters.push(cluster);
                            }
                            cluster.parts.push(component);
                            cluster.x = cluster.parts.reduce((sum, part) => sum + part.x, 0) /
                                cluster.parts.length;
                            cluster.y = cluster.parts.reduce((sum, part) => sum + part.y, 0) /
                                cluster.parts.length;
                            cluster.minX = Math.min(cluster.minX, component.minX);
                            cluster.maxX = Math.max(cluster.maxX, component.maxX);
                            cluster.minY = Math.min(cluster.minY, component.minY);
                            cluster.maxY = Math.max(cluster.maxY, component.maxY);
                        }
                        const angleDistance = (first, second) => {
                            const difference = Math.abs(first - second) % 360;
                            return Math.min(difference, 360 - difference);
                        };
                        const candidates = [];
                        for (const cluster of clusters) {
                            if (cluster.parts.length < 2 || cluster.parts.length > 4) continue;
                            const centerX = (cluster.minX + cluster.maxX) / 2;
                            const centerY = (cluster.minY + cluster.maxY) / 2;
                            const radius = Math.min(
                                cluster.maxX - cluster.minX,
                                cluster.maxY - cluster.minY,
                            ) / 2;
                            if (radius < 35) continue;
                            const scores = [];
                            const isBrightNeutral = (x, y) => {
                                const sampleX = Math.max(0, Math.min(
                                    probe.width - 1,
                                    Math.round(x),
                                ));
                                const sampleY = Math.max(0, Math.min(
                                    probe.height - 1,
                                    Math.round(y),
                                ));
                                const offset = (sampleY * probe.width + sampleX) * 4;
                                const red = pixels[offset];
                                const green = pixels[offset + 1];
                                const blue = pixels[offset + 2];
                                return pixels[offset + 3] > 8 &&
                                    Math.min(red, green, blue) >= 150 &&
                                    Math.max(red, green, blue) - Math.min(red, green, blue) <= 60;
                            };
                            for (let angle = 0; angle < 360; angle += 2) {
                                const radians = angle * Math.PI / 180;
                                let score = 0;
                                for (let distance = Math.round(radius * 0.2);
                                    distance <= Math.round(radius * 0.86);
                                    distance += 2) {
                                    if (isBrightNeutral(
                                        centerX + Math.cos(radians) * distance,
                                        centerY + Math.sin(radians) * distance,
                                    )) {
                                        score += 1;
                                    }
                                }
                                scores.push({angle, score});
                            }
                            const localMaxima = scores.filter((item, index) => {
                                if (item.score < 4) return false;
                                const previous = scores[(index + scores.length - 1) % scores.length];
                                const next = scores[(index + 1) % scores.length];
                                return item.score >= previous.score && item.score >= next.score;
                            }).sort((first, second) => second.score - first.score);
                            const peaks = [];
                            for (const peak of localMaxima) {
                                if (peaks.every((selected) =>
                                    angleDistance(selected.angle, peak.angle) >= 20
                                )) {
                                    peaks.push(peak);
                                }
                                if (peaks.length === 3) break;
                            }
                            if (peaks.length !== 3) continue;
                            const angles = peaks.map((peak) => peak.angle).sort((a, b) => a - b);
                            const gaps = [
                                angles[1] - angles[0],
                                angles[2] - angles[1],
                                angles[0] + 360 - angles[2],
                            ].sort((a, b) => a - b);
                            const minimumDifference = Math.min(
                                gaps[1] - gaps[0],
                                gaps[2] - gaps[1],
                            );
                            if (minimumDifference >= 12) {
                                candidates.push({
                                    x: centerX,
                                    y: centerY,
                                    gaps,
                                    score: minimumDifference + gaps[2] - gaps[0],
                                });
                            }
                        }
                        candidates.sort((first, second) => second.score - first.score);
                        const best = candidates[0];
                        if (best) {
                            circleGrounding = {
                                kind: 'unequal_slices',
                                target: {
                                    x: Math.round(best.x),
                                    y: Math.round(best.y),
                                },
                                gaps: best.gaps,
                                candidateCount: candidates.length,
                            };
                        }
                    } catch (_) {}
                }
                if (patternGrounding) {
                    patternGrounding.startX -= cropX;
                    patternGrounding.startY -= cropY;
                    patternGrounding.endX -= cropX;
                    patternGrounding.endY -= cropY;
                }
                if (arrowGrounding) {
                    arrowGrounding.targets = arrowGrounding.targets.map((target) => ({
                        x: target.x - cropX,
                        y: target.y - cropY,
                    }));
                }
                if (findGrounding) {
                    findGrounding.targets = findGrounding.targets.map((target) => ({
                        x: target.x - cropX,
                        y: target.y - cropY,
                        score: target.score,
                    }));
                }
                if (circleGrounding) {
                    circleGrounding.target.x -= cropX;
                    circleGrounding.target.y -= cropY;
                }
                const output = document.createElement('canvas');
                output.width = cropWidth;
                output.height = cropHeight;
                output.getContext('2d').drawImage(
                    canvas,
                    cropX,
                    cropY,
                    cropWidth,
                    cropHeight,
                    0,
                    0,
                    cropWidth,
                    cropHeight,
                );
                return {
                    dataUrl: output.toDataURL('image/png'),
                    width: cropWidth,
                    height: cropHeight,
                    cropX,
                    cropY,
                    cssX: box.x,
                    cssY: box.y,
                    cssWidth: box.width * cropWidth / canvas.width,
                    cssHeight: box.height * cropHeight / canvas.height,
                    prompt: promptText,
                    submitLabel: label,
                    patternGrounding,
                    arrowGrounding,
                    findGrounding,
                    circleGrounding,
                };
            }"""
        )
        if not isinstance(canvas_data, dict):
            return None
        numeric_keys = ("width", "height", "cssX", "cssY", "cssWidth", "cssHeight")
        if not all(
            not isinstance(canvas_data.get(key), bool)
            and isinstance(canvas_data.get(key), (int, float))
            for key in numeric_keys
        ):
            return None
        width = int(canvas_data["width"])
        height = int(canvas_data["height"])
        css_width = float(canvas_data["cssWidth"])
        css_height = float(canvas_data["cssHeight"])
        crop_x = canvas_data.get("cropX", 0)
        crop_y = canvas_data.get("cropY", 0)
        if (
            isinstance(crop_x, bool)
            or not isinstance(crop_x, (int, float))
            or isinstance(crop_y, bool)
            or not isinstance(crop_y, (int, float))
            or not math.isfinite(float(crop_x))
            or not math.isfinite(float(crop_y))
            or float(crop_x) < 0
            or float(crop_y) < 0
        ):
            return None
        if width < 20 or height < 20 or css_width < 20 or css_height < 20:
            return None
        data_url = canvas_data.get("dataUrl")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            return None
        content = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        iframe_x = iframe_rect.get("x")
        iframe_y = iframe_rect.get("y")
        if (
            isinstance(iframe_x, bool)
            or not isinstance(iframe_x, (int, float))
            or isinstance(iframe_y, bool)
            or not isinstance(iframe_y, (int, float))
        ):
            return None
        grounding = (
            canvas_data.get("patternGrounding")
            or canvas_data.get("arrowGrounding")
            or canvas_data.get("findGrounding")
            or canvas_data.get("circleGrounding")
        )
        grounded_action = _parse_pattern_grounding(grounding, width, height)
        if grounded_action is None:
            grounded_actions = _parse_grounded_click_actions(grounding, width, height)
            grounded_action = grounded_actions[0] if grounded_actions else None
        return LLMCaptchaCapture(
            content=content,
            width=width,
            height=height,
            offset_x=(
                float(iframe_x)
                + float(canvas_data["cssX"])
                + float(crop_x) * (css_width / width)
            ),
            offset_y=(
                float(iframe_y)
                + float(canvas_data["cssY"])
                + float(crop_y) * (css_height / height)
            ),
            scale_x=css_width / width,
            scale_y=css_height / height,
            source="hcaptcha_canvas",
            challenge_prompt=str(canvas_data.get("prompt") or ""),
            submit_label=str(canvas_data.get("submitLabel") or ""),
            grounded_action=grounded_action,
            grounding=grounding if isinstance(grounding, dict) else None,
        )
    except Exception:
        return None


def _parse_grounded_click_actions(
    grounding: Any,
    width: int,
    height: int,
) -> tuple[LLMCaptchaAction, ...]:
    if not isinstance(grounding, dict) or grounding.get("kind") not in {
        "arrow",
        "find_all",
        "unequal_slices",
    }:
        return ()
    raw_targets = (
        [grounding.get("target")]
        if grounding.get("kind") == "unequal_slices"
        else grounding.get("targets")
    )
    if not isinstance(raw_targets, list) or not raw_targets:
        return ()
    actions: list[LLMCaptchaAction] = []
    for index, target in enumerate(raw_targets, 1):
        if not isinstance(target, dict):
            return ()
        start_x = target.get("x")
        start_y = target.get("y")
        if (
            isinstance(start_x, bool)
            or not isinstance(start_x, (int, float))
            or not math.isfinite(float(start_x))
            or isinstance(start_y, bool)
            or not isinstance(start_y, (int, float))
            or not math.isfinite(float(start_y))
            or not (0 <= float(start_x) <= width)
            or not (0 <= float(start_y) <= height)
        ):
            return ()
        actions.append(
            LLMCaptchaAction(
                "click",
                round(float(start_x), 2),
                round(float(start_y), 2),
                None,
                None,
            )
        )
    if grounding.get("kind") == "find_all":
        for index, action in enumerate(actions):
            if any(
                math.hypot(
                    action.start_x - previous.start_x,
                    action.start_y - previous.start_y,
                ) < 32
                for previous in actions[:index]
            ):
                return ()
    return tuple(actions)


def _parse_arrow_grounding_actions(
    grounding: Any,
    width: int,
    height: int,
) -> tuple[LLMCaptchaAction, ...]:
    if not isinstance(grounding, dict) or grounding.get("kind") != "arrow":
        return ()
    if not isinstance(grounding.get("targets"), list) or len(grounding["targets"]) != 2:
        return ()
    return _parse_grounded_click_actions(grounding, width, height)


def _parse_find_grounding_actions(
    grounding: Any,
    width: int,
    height: int,
) -> tuple[LLMCaptchaAction, ...]:
    if not isinstance(grounding, dict) or grounding.get("kind") != "find_all":
        return ()
    return _parse_grounded_click_actions(grounding, width, height)


def _parse_pattern_grounding(
    grounding: Any,
    width: int,
    height: int,
) -> LLMCaptchaAction | None:
    if not isinstance(grounding, dict):
        return None
    if grounding.get("sourceIndex") not in {1, 2}:
        return None
    if grounding.get("targetRow") not in {1, 2, 3, 4}:
        return None
    if grounding.get("targetColumn") not in {1, 2, 3, 4}:
        return None
    values = [grounding.get(key) for key in ("startX", "startY", "endX", "endY")]
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in values
    ):
        return None
    start_x, start_y, end_x, end_y = (
        float(cast(int | float, value)) for value in values
    )
    if not (0 <= start_x <= width and 0 <= end_x <= width):
        return None
    if not (0 <= start_y <= height and 0 <= end_y <= height):
        return None
    return LLMCaptchaAction("drag", start_x, start_y, end_x, end_y)


def _output_preview(output: str, limit: int = 500) -> str:
    compact = " ".join(output.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _llm_timeout_error(
    decision_rounds: int,
    api_requests: int,
    last_detail: str,
) -> str:
    detail = f"; last result: {last_detail}" if last_detail else ""
    return (
        f"LLM captcha timed out after {decision_rounds} decision rounds "
        f"({api_requests} API requests){detail}"
    )


def _responses_endpoint(api_base: str) -> str:
    normalized = api_base.rstrip("/")
    return normalized if normalized.endswith("/responses") else f"{normalized}/responses"


def _chat_completions_endpoint(api_base: str) -> str:
    normalized = api_base.rstrip("/")
    return (
        normalized
        if normalized.endswith("/chat/completions")
        else f"{normalized}/chat/completions"
    )


def _is_actionable_submit_label(label: str) -> bool:
    normalized = " ".join(label.lower().split())
    if not normalized or "skip" in normalized or "跳过" in normalized:
        return False
    return any(
        marker in normalized
        for marker in ("verify", "check", "检查", "验证", "提交")
    )


def _is_next_submit_label(label: str) -> bool:
    normalized = " ".join(label.lower().split())
    if not normalized or "skip" in normalized or "跳过" in normalized:
        return False
    return any(
        marker in normalized
        for marker in ("next", "下一个", "下一步", "下一页")
    )


def _is_two_arrow_challenge(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return "two arrows" in normalized or "两个箭头" in normalized


def _is_find_all_challenge(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return (
        "find all animal icons" in normalized
        or "find all animals" in normalized
        or "找到所有动物" in normalized
    )


def _is_multi_select_challenge(prompt: str, capture_source: str = "") -> bool:
    """Return whether one fresh image can safely contain several tile clicks."""
    normalized = " ".join(prompt.lower().split())
    if capture_source not in {"hcaptcha_iframe", "hcaptcha_canvas"}:
        return False
    if normalized.startswith(
        (
            "click on items",
            "click items",
            "select items",
            "pick items",
        )
    ):
        return True
    return any(
        marker in normalized
        for marker in (
            "select all",
            "pick all",
            "click on all",
            "click all",
            "find all",
        )
    )


def _is_unequal_slices_challenge(prompt: str) -> bool:
    normalized = " ".join(prompt.lower().split())
    return (
        "circle where each slice is a different size" in normalized
        or "circle with uneven slices" in normalized
        or ("每个" in normalized and ("切片" in normalized or "扇形" in normalized)
            and "大小" in normalized and "不同" in normalized)
    )


def _is_checkbox_capture(capture: LLMCaptchaCapture) -> bool:
    return (
        capture.source == "hcaptcha_iframe"
        and capture.width <= 400
        and capture.height <= 130
        and not capture.challenge_prompt
    )


def _is_repeated_click(
    action: LLMCaptchaAction,
    clicked_points: list[tuple[float, float]],
    width: int,
    height: int,
) -> bool:
    if action.kind != "click":
        return False
    tolerance = max(20.0, min(width, height) * 0.06)
    return any(
        math.hypot(action.start_x - x, action.start_y - y) <= tolerance
        for x, y in clicked_points
    )


def _is_new_hcaptcha_challenge(
    previous_prompt: str,
    previous_submit_label: str,
    prompt: str,
    submit_label: str,
) -> bool:
    """Detect a new page after Check without mistaking selection feedback for one."""
    if not previous_prompt and prompt:
        return True
    if previous_prompt and prompt and previous_prompt.strip() != prompt.strip():
        return True
    if not previous_submit_label or not submit_label:
        return False
    return (
        _is_actionable_submit_label(previous_submit_label)
        and not _is_actionable_submit_label(submit_label)
        and not _is_next_submit_label(submit_label)
    )


def _llm_retry_feedback(
    feedback: str,
    clicked_points: list[tuple[float, float]],
    width: int | None = None,
    height: int | None = None,
) -> str:
    details = feedback.strip()
    if clicked_points:
        if width and height:
            points = ", ".join(
                f"({round(x * 1000 / width, 1)}, {round(y * 1000 / height, 1)})"
                for x, y in clicked_points[-6:]
            )
            coordinate_detail = "in normalized_1000 coordinates"
        else:
            points = ", ".join(
                f"({round(x, 1)}, {round(y, 1)})" for x, y in clicked_points[-6:]
            )
            coordinate_detail = "in the current capture's pixel coordinates"
        selected = (
            "Known selected click centers in the current challenge image are "
            f"{points} {coordinate_detail}. Do not click within the selected markers."
        )
        details = f"{details} {selected}".strip()
    return details


def _captcha_action_error(
    decision: LLMCaptchaDecision,
    challenge_prompt: str,
) -> str | None:
    if not decision.actions or not challenge_prompt:
        return None
    prompt = challenge_prompt.strip().lower()
    expected_kind: str | None = None
    if prompt.startswith(("click", "please click", "pick ", "find all", "select ")):
        expected_kind = "click"
    elif prompt.startswith(("drag", "move ", "place ")):
        expected_kind = "drag"
    if expected_kind:
        for action in decision.actions:
            if action.kind != expected_kind:
                return (
                    f"challenge instruction requires {expected_kind}, but model returned "
                    f"{action.kind}"
                )
    return None


def _fallback_grid_action_error(
    decision: LLMCaptchaDecision,
    capture: LLMCaptchaCapture,
) -> str | None:
    """Reject clicks in the prompt/footer of the DOM-based 3x3 challenge."""
    if (
        capture.source != "hcaptcha_iframe"
        or capture.width < 400
        or capture.height < 500
        or not decision.actions
        or decision.actions[0].kind != "click"
    ):
        return None
    for action in decision.actions:
        if not (15 <= action.start_x <= capture.width - 15):
            return (
                "this hCaptcha image is a 3x3 tile grid; click a tile, not the prompt or footer"
            )
        if not (
            capture.height * 0.20
            <= action.start_y
            <= min(capture.height * 0.86, capture.height - 10)
        ):
            return (
                "this hCaptcha image is a 3x3 tile grid; click a tile, not the prompt or footer"
            )
    return None


def _coerce_dom_grid_decision(
    decision: LLMCaptchaDecision,
    capture: LLMCaptchaCapture,
) -> LLMCaptchaDecision:
    """Repair mixed normalized/pixel coordinates common on the DOM grid crop."""
    if (
        decision.coordinate_space != "normalized_1000"
        or capture.source != "hcaptcha_iframe"
        or capture.width < 400
        or capture.height < 500
        or not decision.actions
    ):
        return decision

    x_centers = tuple(
        round(capture.width * ratio, 2) for ratio in (0.19, 0.50, 0.81)
    )
    y_centers = tuple(
        round(capture.height * ratio)
        for ratio in (199 / 616, 329 / 616, 459 / 616)
    )

    def snap(value: float, alternate: float, limit: float, centers: tuple[float, ...]) -> float:
        options = [(value, min(abs(value - center) for center in centers))]
        if 0 <= alternate <= limit:
            options.append((alternate, min(abs(alternate - center) for center in centers)))
        candidate, distance = min(options, key=lambda item: item[1])
        if distance > 82:
            return value
        return min(centers, key=lambda center: abs(center - candidate))

    actions: list[LLMCaptchaAction] = []
    changed = False
    for action in decision.actions:
        if action.kind != "click":
            actions.append(action)
            continue
        raw_x = action.start_x * 1000 / capture.width
        raw_y = action.start_y * 1000 / capture.height
        start_x = snap(action.start_x, raw_x, capture.width, x_centers)
        start_y = snap(action.start_y, raw_y, capture.height, y_centers)
        changed = changed or start_x != action.start_x or start_y != action.start_y
        actions.append(replace(action, start_x=start_x, start_y=start_y))
    return replace(decision, actions=tuple(actions)) if changed else decision


def _llm_captcha_prompt(
    width: int,
    height: int,
    attempt: int,
    call_number: int,
    challenge_prompt: str = "",
    submit_label: str = "",
    capture_source: str = "viewport",
    executed_action_count: int = 0,
    feedback: str = "",
    allow_multiple_actions: bool = False,
) -> str:
    prompt_detail = challenge_prompt or "Read the visible hCaptcha instruction from the image."
    submit_detail = submit_label or "not detected"
    feedback_detail = (
        f"Controller feedback for this retry: {feedback.strip()}"
        if feedback.strip()
        else ""
    )
    source_detail = (
        "This is the visible hCaptcha canvas cropped from its native backing resolution. The prompt may "
        "be rendered outside the canvas, so use the extracted instruction below."
        if capture_source == "hcaptcha_canvas"
        else "This is a browser screenshot containing the visible hCaptcha UI."
    )
    grid_detail = ""
    if capture_source == "hcaptcha_iframe" and width >= 400 and height >= 500:
        grid_x = [round(value * 1000) for value in (0.19, 0.50, 0.81)]
        grid_y = [round(value * 1000) for value in (0.323, 0.534, 0.745)]
        grid_detail = (
            "This DOM-based challenge uses a 3x3 image grid. For every click, set "
            "grid_row and grid_column to the target's 1-based row and column; the "
            "controller trusts those fields over approximate coordinates. Tile-center "
            f"coordinates in normalized_1000 are x={grid_x} and y={grid_y}. Never "
            "target the prompt, reference image, or footer."
        )
    action_limit_detail = (
        "This is a multi-select image task. Return one click action for every clearly "
        "matching, currently unselected tile visible in this screenshot; use only tile "
        "centers and do not include the reference image, prompt, footer, or Verify button."
        if allow_multiple_actions
        else "Return at most ONE action based on this exact screenshot. The browser will execute it, "
        "wait for the UI to update, and send a fresh screenshot before any next action."
    )
    return f"""Analyze the attached image and operate only the visible hCaptcha UI.
The image is exactly {width} by {height} pixels, but every returned coordinate MUST use normalized_1000
space: (0, 0) is the image's top-left and (1000, 1000) is its bottom-right. Set coordinate_space to
"normalized_1000". Never return pixel coordinates. This is overall attempt {attempt} and model call
{call_number}. The browser has already confirmed that the CAPTCHA is not complete at the time of this
screenshot, regardless of any earlier model response.

	Capture details: {source_detail}
	{grid_detail}
	Extracted hCaptcha instruction: {prompt_detail}
    Current hCaptcha submit control label: {submit_detail}
    {feedback_detail}
    Actions already executed during this challenge attempt: {executed_action_count}. Previously selected
DOM-grid tiles shrink slightly and show a teal circle with a white check at the upper-right; canvas tasks
may use a white circled X. Both markers mean SELECTED, not close or cancel. Never click a marked or already
selected item, because doing so deselects the answer. There is no close button inside the challenge canvas.

    {action_limit_detail} The browser will execute returned clicks in order. For a click, use the target center for
    start_x/start_y and null for end_x/end_y. For a drag, use the draggable object's center as the start and
    the destination center as the end. On a DOM 3x3 grid, set grid_row/grid_column to 1-3. Set both fields
    to null for canvas clicks, checkbox clicks, and drags.

Follow these hCaptcha rules:
- If an unchecked hCaptcha checkbox is visible, click only its checkbox.
- If an image-selection challenge is visible, click clearly matching tiles. In multi-select mode,
  return every clearly matching, currently unselected tile from this screenshot in one response; the
  controller will submit the batch. Otherwise return one tile and wait for a fresh image.
- For "given number of times" challenges, use the reference icons and counts shown in the legend, then
  select every matching grid icon one at a time.
- For "two arrows that break the chain" challenges, identify both orientation/shape breaks and select
  one still-unselected break per screenshot. Do not verify until both have been selected.
- For pattern challenges, compare rows and columns and select the single icon that breaks the repeated
  pattern.
- If a drag challenge is visible, return one drag action.
- If the challenge selection is complete, return status "verify" with no actions. The browser will
  click the hCaptcha Verify or Check control by DOM; that control may be outside the attached canvas.
- Verify is a CAPTCHA control. Never click account, registration, login, navigation, browser, or other
  outer-page submit controls.

Return "solved" with no actions only when a green checked hCaptcha checkbox or explicit success state is
visible and no challenge dialog remains. A visible Verify button is not success. If you can describe or
name the correct target, you MUST return its click or drag coordinates with status "actions"; never
describe an intended action only in message. If uncertain, choose the best visible CAPTCHA action rather
than returning no action. In a drag puzzle, the destination must be the center of the matching silhouette
or a currently empty target cell, never an occupied animal. Keep reasoning brief, keep coordinates inside
the image, and return only the requested JSON object."""


def _responses_output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "refusal":
                    refusal = part.get("refusal")
                    raise RuntimeError(
                        f"Responses API refused the image request: {str(refusal)[:200]}"
                    )
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    if chunks:
        return "\n".join(chunks)

    error = payload.get("error")
    if error:
        raise RuntimeError(f"Responses API error: {str(error)[:300]}")
    raise RuntimeError("Responses API response contained no output text")


def _chat_completion_output_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                refusal = message.get("refusal")
                if isinstance(refusal, str) and refusal.strip():
                    raise RuntimeError(
                        f"Chat Completions API refused the image request: {refusal[:200]}"
                    )
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    chunks = [
                        str(part["text"]).strip()
                        for part in content
                        if isinstance(part, dict)
                        and isinstance(part.get("text"), str)
                        and part["text"].strip()
                    ]
                    if chunks:
                        return "\n".join(chunks)

    error = payload.get("error")
    if error:
        raise RuntimeError(f"Chat Completions API error: {str(error)[:300]}")
    raise RuntimeError("Chat Completions API response contained no output text")


def _consume_sse_response(
    response: "requests.Response",
    api_protocol: str,
    deadline: float | None,
) -> tuple[str, dict[str, Any]]:
    """Consume a streaming (SSE) LLM response and return (full_text, payload).

    The request is sent with ``stream: True``; the provider replies with
    Server-Sent Events. We assemble the incremental text deltas and rebuild a
    payload compatible with the existing ``_chat_completion_output_text`` /
    ``responses_output_text`` parsers so the rest of the pipeline is unchanged.

    Handles both OpenAI-compatible protocols:
    - ``chat_completions``: ``data: {"choices":[{"delta":{"content":"..."}}]}``
      ending with ``data: [DONE]``.
    - ``responses``: ``event: response.output_text.delta`` +
      ``data: {"type":"response.output_text.delta","delta":"..."}``, with a
      final ``event: response.completed`` carrying the full response object.
    """
    text_chunks: list[str] = []
    final_payload: dict[str, Any] = {}
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    event_type = ""

    def _enforce_deadline() -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise RuntimeError("LLM SSE stream exceeded request deadline")

    response.encoding = "utf-8"
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            _enforce_deadline()
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue

            rid = data.get("response_id") or data.get("id")
            if isinstance(rid, str):
                response_id = rid
            u = data.get("usage")
            if isinstance(u, dict):
                usage = u

            if api_protocol == "chat_completions":
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    delta = (
                        choices[0].get("delta")
                        if isinstance(choices[0], dict)
                        else None
                    )
                    if isinstance(delta, dict):
                        piece = delta.get("content")
                        if isinstance(piece, str):
                            text_chunks.append(piece)
            else:
                etype = data.get("type") or event_type
                if etype == "response.completed":
                    completed = data.get("response")
                    if isinstance(completed, dict):
                        final_payload = completed
                piece = data.get("delta")
                if isinstance(piece, str) and etype in (
                    "response.output_text.delta",
                    "response.refusal.delta",
                ):
                    text_chunks.append(piece)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"LLM SSE stream interrupted: {exc}") from exc

    text = "".join(text_chunks).strip()
    if api_protocol == "chat_completions":
        payload: dict[str, Any] = {
            "id": response_id,
            "choices": [{"message": {"content": text, "refusal": None}}],
            "usage": usage,
        }
    else:
        payload = final_payload or {
            "id": response_id,
            "output_text": text,
            "output": [{"content": [{"type": "output_text", "text": text}]}],
            "usage": usage,
        }
    return text, payload


def _parse_llm_decision(
    text: str,
    width: int,
    height: int,
    max_actions: int = LLM_MAX_ACTIONS_PER_CALL,
    allow_grid_coordinates: bool = False,
) -> LLMCaptchaDecision:
    clean = text.strip()
    try:
        payload = json.loads(clean)
    except json.JSONDecodeError:
        payload = None
        decoder = json.JSONDecoder()
        for index, character in enumerate(clean):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(clean[index:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, dict)
                and "status" in candidate
                and "actions" in candidate
            ):
                payload = candidate
                break
        if payload is None:
            raise ValueError("LLM captcha output is not valid JSON")
    if not isinstance(payload, dict):
        raise ValueError("LLM captcha output must be a JSON object")

    status = payload.get("status")
    if status not in {"actions", "verify", "solved", "failed"}:
        raise ValueError("LLM captcha status must be actions, verify, solved, or failed")
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("LLM captcha actions must be an array")
    if max_actions < 1:
        raise ValueError("LLM captcha max_actions must be positive")
    ignored_action_count = max(0, len(raw_actions) - max_actions)
    raw_actions = raw_actions[:max_actions]
    if raw_actions and status != "actions":
        status = "actions"
    reported_coordinate_space = payload.get("coordinate_space")
    if reported_coordinate_space not in {None, "pixels", "normalized_1000"}:
        raise ValueError("LLM captcha coordinate_space is invalid")

    actions: list[LLMCaptchaAction] = []
    coordinate_space = "pixels"
    for index, raw_action in enumerate(raw_actions, 1):
        if not isinstance(raw_action, dict):
            raise ValueError(f"LLM captcha action {index} must be an object")
        kind = raw_action.get("kind")
        if kind not in {"click", "drag"}:
            raise ValueError(f"LLM captcha action {index} has an invalid kind")
        start_x = _raw_model_coordinate(raw_action.get("start_x"), index, "start_x")
        start_y = _raw_model_coordinate(raw_action.get("start_y"), index, "start_y")
        grid_row = raw_action.get("grid_row")
        grid_column = raw_action.get("grid_column")
        grid_cell_specified = allow_grid_coordinates and (
            grid_row is not None or grid_column is not None
        )
        if grid_cell_specified:
            if kind != "click":
                raise ValueError(
                    f"LLM captcha action {index} grid coordinates require click"
                )
            if (
                isinstance(grid_row, bool)
                or not isinstance(grid_row, int)
                or isinstance(grid_column, bool)
                or not isinstance(grid_column, int)
                or not 1 <= grid_row <= 3
                or not 1 <= grid_column <= 3
            ):
                raise ValueError(
                    f"LLM captcha action {index} grid row and column must be 1-3"
                )
            start_x = (190.0, 500.0, 810.0)[grid_column - 1]
            start_y = (323.0, 534.0, 745.0)[grid_row - 1]
        if kind == "drag":
            end_x = _raw_model_coordinate(raw_action.get("end_x"), index, "end_x")
            end_y = _raw_model_coordinate(raw_action.get("end_y"), index, "end_y")
        else:
            end_x = None
            end_y = None
        coordinate_pairs = [(start_x, width), (start_y, height)]
        if end_x is not None and end_y is not None:
            coordinate_pairs.extend(((end_x, width), (end_y, height)))
        use_normalized = (
            reported_coordinate_space == "normalized_1000" or grid_cell_specified
        )
        if not use_normalized and any(
            value > limit for value, limit in coordinate_pairs
        ):
            if any(value > 1000 for value, _ in coordinate_pairs):
                raise ValueError(
                    f"LLM captcha action {index} coordinates are outside the viewport"
                )
            use_normalized = True
        if use_normalized:
            if any(value > 1000 for value, _ in coordinate_pairs):
                raise ValueError(
                    f"LLM captcha action {index} normalized coordinates exceed 1000"
                )
            coordinate_space = "normalized_1000"
            start_x = round(start_x / 1000 * width, 2)
            start_y = round(start_y / 1000 * height, 2)
            if end_x is not None and end_y is not None:
                end_x = round(end_x / 1000 * width, 2)
                end_y = round(end_y / 1000 * height, 2)
        actions.append(LLMCaptchaAction(kind, start_x, start_y, end_x, end_y))

    message = payload.get("message", "")
    if not isinstance(message, str):
        raise ValueError("LLM captcha message must be a string")
    return LLMCaptchaDecision(
        status,
        tuple(actions),
        message,
        ignored_action_count=ignored_action_count,
        coordinate_space=coordinate_space,
    )


_VERBOSE_CLICK_PATTERN = re.compile(
    r"(?:i(?:'ll|\s+will|\s+should|\s+need\s+to)\s+click|"
    r"i(?:'ll|\s+will)\s+(?:go\s+with|pick)|let\s+me\s+pick)"
    r"[^.\n]{0,240}?(?:\bat\b|coordinates?)\s*"
    r"(?:approximately\s*)?(?:coordinates?\s*)?\(?\s*"
    r"(?:x\s*=\s*)?(\d+(?:\.\d+)?)\s*[,，]\s*"
    r"(?:y\s*=\s*)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_verbose_click_decision(
    text: str,
    width: int,
    height: int,
    challenge_prompt: str,
) -> LLMCaptchaDecision | None:
    """Recover only an explicit, in-bounds click conclusion from non-JSON output."""
    prompt = challenge_prompt.strip().lower()
    output = text.lower()
    if any(
        marker in output
        for marker in (
            "place the correct animal",
            "drag the correct",
            "drag challenge",
        )
    ):
        return None
    if prompt and not prompt.startswith(
        ("click", "please click", "pick ", "find all", "select ")
    ):
        return None
    matches = list(_VERBOSE_CLICK_PATTERN.finditer(text))
    if not matches:
        return None
    start_x, start_y = (float(value) for value in matches[-1].groups())
    if not (0 <= start_x <= width and 0 <= start_y <= height):
        return None
    return LLMCaptchaDecision(
        status="actions",
        actions=(LLMCaptchaAction("click", start_x, start_y, None, None),),
        message="recovered explicit click conclusion from compatibility output",
        raw_output=text,
        coordinate_space="pixels",
    )


def _raw_model_coordinate(value: Any, action_index: int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LLM captcha action {action_index} {field} must be numeric")
    coordinate = float(value)
    if not math.isfinite(coordinate) or coordinate < 0:
        raise ValueError(f"LLM captcha action {action_index} {field} is outside the viewport")
    return round(coordinate, 2)


async def _apply_llm_action(
    page: Page,
    action: LLMCaptchaAction,
    offset_x: float = 0,
    offset_y: float = 0,
    scale_x: float = 1,
    scale_y: float = 1,
) -> None:
    start_x = action.start_x * scale_x + offset_x
    start_y = action.start_y * scale_y + offset_y
    if action.kind == "click":
        await page.mouse.click(start_x, start_y, delay=80)
        return
    assert action.end_x is not None and action.end_y is not None
    end_x = action.end_x * scale_x + offset_x
    end_y = action.end_y * scale_y + offset_y
    distance = math.hypot(end_x - start_x, end_y - start_y)
    steps = max(6, min(30, int(distance / 12)))
    await page.mouse.move(start_x, start_y)
    await page.mouse.down()
    try:
        await page.mouse.move(end_x, end_y, steps=steps)
    finally:
        await page.mouse.up()


async def _click_hcaptcha_submit(page: Page, timeout_seconds: float = 3) -> bool:
    async def scan(deadline: float | None) -> bool:
        skip_labels = ("skip", "跳过")
        for frame in getattr(page, "frames", []):
            if "hcaptcha" not in frame.url.lower() or "frame=challenge" not in frame.url.lower():
                continue
            button = frame.locator(".button-submit").first
            if deadline is None:
                click_timeout_ms = 5000
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                click_timeout_ms = max(
                    1,
                    min(5000, math.ceil(remaining * 1000)),
                )
            try:
                if await button.count() < 1 or not await button.is_visible():
                    continue
                label = await button.evaluate(
                    r"""(element) => [element.textContent, element.getAttribute('aria-label')]
                        .filter(Boolean).join(' ').replace(/\s+/g, ' ').trim().toLowerCase()"""
                )
                if not isinstance(label, str) or not label:
                    continue
                if any(skip_label in label for skip_label in skip_labels):
                    continue
                if await button.get_attribute("aria-disabled") == "true":
                    continue
                await button.click(delay=80, timeout=click_timeout_ms)
                return True
            except Exception:
                continue
        return False

    if timeout_seconds <= 0:
        return await scan(None)

    deadline = time.monotonic() + timeout_seconds
    poll_seconds = 0.2
    while True:
        if await scan(deadline):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(poll_seconds, remaining))


async def _is_captcha_target_enabled(page: Page) -> bool:
    result = await page.evaluate(
        """() => {
            const register = document.querySelector('#register_button');
            return register
                ? !register.disabled && register.getAttribute('aria-disabled') !== 'true'
                : false;
        }"""
    )
    return bool(result)


async def _reset_hcaptcha(page: Page) -> bool:
    try:
        result = await page.evaluate(
            """() => {
                try {
                    if (window.hcaptcha && typeof window.hcaptcha.reset === 'function') {
                        window.hcaptcha.reset();
                        return true;
                    }
                } catch (_) {}
                return false;
            }"""
        )
        return bool(result)
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  sitekey 捕获（render=explicit 模式下 DOM 无 sitekey，只能从网络请求获取）
# ---------------------------------------------------------------------------

_captured_sitekeys: WeakKeyDictionary[Page, str] = WeakKeyDictionary()


def reset_captcha_state(page: Page | None = None) -> None:
    """Reset one page's sitekey cache, or all caches for compatibility."""
    if page is None:
        _captured_sitekeys.clear()
    else:
        _captured_sitekeys.pop(page, None)


def start_capturing_sitekey(page: Page) -> None:
    """注册网络请求监听器，从 checksiteconfig 请求中捕获 hCaptcha sitekey。

    必须在 create-account 页加载前调用。
    """
    def _on_request(req):
        if page in _captured_sitekeys:
            return
        url = req.url
        if "checksiteconfig" in url and "sitekey=" in url:
            try:
                sk = parse_qs(urlparse(url).query).get("sitekey", [None])[0]
                if sk:
                    _captured_sitekeys[page] = sk
                    print(f"  sitekey captured: {sk}")
            except Exception:
                pass

    page.on("request", _on_request)


async def _get_site_key(page: Page) -> str | None:
    """获取 sitekey（仅从网络请求缓存中读取）。"""
    if sitekey := _captured_sitekeys.get(page):
        return sitekey
    # 等待网络请求捕获（hCaptcha iframe 可能还在加载）
    for _ in range(30):
        if sitekey := _captured_sitekeys.get(page):
            return sitekey
        await asyncio.sleep(1)
    return None


# ---------------------------------------------------------------------------
#  token 注入（通过拦截的 Angular 回调直接触发 onSuccess）
# ---------------------------------------------------------------------------


async def _inject_hcaptcha_token(page: Page, token: str) -> None:
    """调用拦截的 __hCaptchaCallback 触发 Angular onSuccess，使 #register_button enable。

    回调由 main.py 的 _ensure_hcaptcha_hook 通过 addInitScript 在
    hcaptcha.render 调用时捕获到 window.__hCaptchaCallback。
    """
    result = await page.evaluate(
        r"""(token) => {
            if (typeof window.__hCaptchaCallback === 'function') {
                window.__hCaptchaCallback(token);
                return true;
            }
            return false;
        }""",
        token,
    )
    print(f"  callback called: {result}")


# ---------------------------------------------------------------------------
#  辅助
# ---------------------------------------------------------------------------


async def _is_register_button_enabled(page: Page) -> bool:
    """检查 #register_button 是否 enabled（hCaptcha 通过后按钮才会 enable）。"""
    result = await page.evaluate(
        """() => {
            const btn = document.querySelector('#register_button');
            return btn ? !btn.disabled : false;
        }"""
    )
    return bool(result)


def _site_referer(website_url: str) -> str:
    parsed = urlparse(website_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/"
    return website_url


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {"status_code": response.status_code, "text": response.text}
    return data if isinstance(data, dict) else {"data": data}


def _extract_hcaptcha_token(data: dict[str, Any]) -> str | None:
    token = data.get("gRecaptchaResponse") or data.get("token")
    return str(token) if token else None


def build_captcha_solver(
    config: CaptchaConfig,
    request_semaphore: asyncio.Semaphore | None = None,
) -> CaptchaSolver:
    if config.mode == "manual":
        return ManualCaptchaSolver()
    if config.mode == "yescaptcha":
        if not config.yescaptcha_client_key:
            raise ValueError("yescaptcha_client_key is required")
        return YesCaptchaSolver(
            client_key=config.yescaptcha_client_key,
            api_url=config.yescaptcha_api_url,
            poll_interval_seconds=config.poll_interval_seconds,
            timeout_seconds=config.timeout_seconds,
        )
    if config.mode == "llm":
        if not config.llm_model or not config.llm_api_key:
            raise ValueError("llm_model and llm_api_key are required")
        return LLMCaptchaSolver(
            model=config.llm_model,
            api_base=config.llm_api_base,
            api_key=config.llm_api_key,
            api_protocol=config.llm_api_protocol,
            timeout_seconds=config.timeout_seconds,
            reasoning_effort=config.llm_reasoning_effort,
            call_delay_seconds=config.llm_call_delay_seconds,
            action_delay_seconds=config.llm_action_delay_seconds,
            calls_per_attempt=config.llm_calls_per_attempt,
            max_attempts=config.llm_max_attempts,
            max_output_tokens=config.llm_max_output_tokens,
            artifact_dir=config.llm_artifact_dir,
            request_semaphore=request_semaphore,
        )
    if config.mode == "captcharun":
        if not config.captcharun_token:
            raise ValueError("captcharun_token is required")
        return CaptchaRunSolver(
            token=config.captcharun_token,
            api_url=config.captcharun_api_url,
            poll_interval_seconds=config.poll_interval_seconds,
            timeout_seconds=config.timeout_seconds,
        )
    raise ValueError(f"Unsupported captcha mode: {config.mode}")
