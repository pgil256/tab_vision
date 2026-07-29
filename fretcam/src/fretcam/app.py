"""FastAPI application for the local FretCam live HUD."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import time
from contextlib import asynccontextmanager
from importlib.resources import files
from urllib.parse import urlparse
from typing import AsyncIterator, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fretcam.diagnostic_capture import (
    DiagnosticCaptureError,
    FailureExpectation,
    LocalCaptureSession,
)
from fretcam.processing import FrameProcessorFactory, HudFrameProcessor

DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024
STATIC_DIR = files("fretcam").joinpath("static")
CAPTURE_CONTROL_TYPES = frozenset(
    {
        "trace_start",
        "trace_save",
        "trace_cancel",
        "failure_buffer",
        "failure_mark",
        "frame_context",
    }
)
FRAME_CONTEXT_FIELDS = frozenset(
    {
        "sequence",
        "session_offset_ms",
        "source_width",
        "source_height",
        "inference_width",
        "inference_height",
        "jpeg_quality",
        "payload_bytes",
    }
)
TRACE_MUTATING_CONTROL_TYPES = frozenset(
    {
        "settings",
        "calibrate",
        "calibrate_two_point",
        "continue_calibration",
        "reset_calibration",
        "reacquire",
    }
)


def _is_loopback_host(value: str) -> bool:
    if value in {"localhost", "testclient", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _capture_controls_allowed(websocket: WebSocket) -> bool:
    client = websocket.client
    if client is None or not _is_loopback_host(client.host):
        return False
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    origin_url = urlparse(origin)
    host_header = websocket.headers.get("host")
    if host_header is None:
        return False
    target_url = urlparse(f"//{host_header}")
    return bool(
        origin_url.hostname
        and target_url.hostname
        and _is_loopback_host(origin_url.hostname)
        and origin_url.hostname.lower() == target_url.hostname.lower()
        and origin_url.port == target_url.port
    )


def _frame_context(payload: dict[str, object]) -> dict[str, object]:
    unknown = set(payload) - FRAME_CONTEXT_FIELDS - {"type"}
    if unknown:
        raise DiagnosticCaptureError(f"unknown frame context fields: {sorted(unknown)}")
    missing = FRAME_CONTEXT_FIELDS - set(payload)
    if missing:
        raise DiagnosticCaptureError(f"missing frame context fields: {sorted(missing)}")
    context: dict[str, object] = {}
    for name in FRAME_CONTEXT_FIELDS:
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosticCaptureError(f"{name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise DiagnosticCaptureError(f"{name} must be finite and non-negative")
        if name in {
            "sequence",
            "source_width",
            "source_height",
            "inference_width",
            "inference_height",
            "payload_bytes",
        }:
            if number > 10_000_000 or not number.is_integer():
                raise DiagnosticCaptureError(f"{name} must be a bounded integer")
            context[name] = int(number)
        elif name == "jpeg_quality":
            if number > 1.0:
                raise DiagnosticCaptureError("jpeg_quality must be between 0 and 1")
            context[name] = number
        else:
            context[name] = number
    return context


def create_app(
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    echo_mode: bool = False,
    processor_factory: FrameProcessorFactory | None = None,
    capture_factory: Callable[[], LocalCaptureSession] | None = None,
) -> FastAPI:
    """Create the HUD app, or the retained F1 echo harness for its benchmark."""

    if max_frame_bytes < 1:
        raise ValueError("max_frame_bytes must be positive")
    factory = processor_factory or HudFrameProcessor
    make_capture = capture_factory or LocalCaptureSession

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if echo_mode:
            yield
            return
        processor = await asyncio.to_thread(factory)
        try:
            await asyncio.to_thread(processor.warmup)
            application.state.frame_processor = processor
            yield
        finally:
            await asyncio.to_thread(processor.close)

    app = FastAPI(title="FretCam", version="0.1.0", lifespan=lifespan)
    session_lock = asyncio.Lock()
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR.joinpath("index.html"))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "echo" if echo_mode else "hud"}

    @app.websocket("/ws")
    async def live_frames(websocket: WebSocket) -> None:
        await websocket.accept()
        capture: LocalCaptureSession | None = None
        pending_capture: tuple[int | None, int] | None = None
        last_client_capture_sequence = 0
        player_handedness = "right"

        def get_capture() -> LocalCaptureSession:
            nonlocal capture
            if capture is None:
                capture = make_capture()
            return capture

        try:
            if echo_mode:
                while True:
                    frame = await websocket.receive_bytes()
                    if len(frame) > max_frame_bytes:
                        await websocket.close(code=1009, reason="frame too large")
                        return
                    await websocket.send_bytes(frame)
            async with session_lock:
                processor = app.state.frame_processor
                await asyncio.to_thread(processor.reset)
                try:
                    while True:
                        message = await websocket.receive()
                        observed_at_s = time.monotonic()
                        if message.get("type") == "websocket.disconnect":
                            return
                        frame = message.get("bytes")
                        text = message.get("text")
                        capture_control = False
                        send_response = True
                        capture_warning: str | None = None
                        try:
                            if frame is not None:
                                if len(frame) > max_frame_bytes:
                                    await websocket.close(
                                        code=1009, reason="frame too large"
                                    )
                                    return
                                if pending_capture is not None:
                                    if capture is not None:
                                        capture.disconnect()
                                    pending_capture = None
                                    capture_warning = (
                                        "local capture stopped because the previous "
                                        "frame context was missing"
                                    )
                                response = await asyncio.to_thread(
                                    processor.process_jpeg,
                                    frame,
                                    timestamp_s=observed_at_s,
                                )
                                if capture is not None:
                                    status = capture.status()
                                else:
                                    status = None
                                if status is not None and (
                                    status.trace_enabled or status.failure_enabled
                                ):
                                    detection = response.get("detection", {})
                                    processor_timestamp_s = (
                                        detection.get("timestamp_s")
                                        if isinstance(detection, dict)
                                        else None
                                    )
                                    try:
                                        capture_sequence = capture.record_frame(
                                            frame,
                                            response,
                                            observed_at_s=observed_at_s,
                                            processor_timestamp_s=(
                                                float(processor_timestamp_s)
                                                if isinstance(
                                                    processor_timestamp_s,
                                                    (int, float),
                                                )
                                                and not isinstance(
                                                    processor_timestamp_s,
                                                    bool,
                                                )
                                                else observed_at_s
                                            ),
                                            server_metadata={
                                                "payload_bytes": len(frame),
                                                "processed_frame": response.get(
                                                    "frame",
                                                    {},
                                                ),
                                            },
                                        )
                                        # Even a frozen non-rolling trace must
                                        # consume and validate the browser's
                                        # context for this packet. A ``None``
                                        # sequence means no enabled buffer
                                        # retained the already-processed JPEG.
                                        pending_capture = (
                                            capture_sequence,
                                            len(frame),
                                        )
                                    except DiagnosticCaptureError as exc:
                                        capture.disconnect()
                                        pending_capture = None
                                        response = dict(response)
                                        response["capture_warning"] = str(exc)
                                if capture_warning is not None:
                                    response = dict(response)
                                    response["capture_warning"] = capture_warning
                            elif text is not None:
                                payload = json.loads(text)
                                if not isinstance(payload, dict):
                                    raise ValueError(
                                        "control payload must be an object"
                                    )
                                message_type = payload.get("type")
                                if message_type in CAPTURE_CONTROL_TYPES:
                                    capture_control = True
                                    if not _capture_controls_allowed(websocket):
                                        raise DiagnosticCaptureError(
                                            "local capture controls require a "
                                            "loopback page and connection"
                                        )
                                    if message_type == "frame_context":
                                        if capture is None or pending_capture is None:
                                            raise DiagnosticCaptureError(
                                                "frame context has no preceding "
                                                "captured frame"
                                            )
                                        try:
                                            context = _frame_context(payload)
                                            sequence, payload_bytes = pending_capture
                                            if (
                                                context["payload_bytes"]
                                                != payload_bytes
                                            ):
                                                raise DiagnosticCaptureError(
                                                    "frame context payload_bytes does "
                                                    "not match the exact JPEG"
                                                )
                                            client_sequence = int(context["sequence"])
                                            if (
                                                client_sequence
                                                <= last_client_capture_sequence
                                            ):
                                                raise DiagnosticCaptureError(
                                                    "frame context sequence must "
                                                    "increase"
                                                )
                                            if sequence is not None:
                                                capture.attach_client_metadata(
                                                    sequence,
                                                    context,
                                                )
                                        except DiagnosticCaptureError:
                                            capture.disconnect()
                                            pending_capture = None
                                            raise
                                        last_client_capture_sequence = client_sequence
                                        pending_capture = None
                                        send_response = False
                                        response = None
                                    elif message_type == "trace_start":
                                        await asyncio.to_thread(processor.reset)
                                        status = get_capture().start_trace(
                                            session_metadata={
                                                "source": "browser_live",
                                                "hud_version": 2,
                                            },
                                            replay_controls=(
                                                {
                                                    "type": "settings",
                                                    "player_handedness": (
                                                        player_handedness
                                                    ),
                                                },
                                            ),
                                        )
                                        pending_capture = None
                                        response = {
                                            "type": "control",
                                            "status": "trace_started",
                                            "capture": status.as_dict(),
                                        }
                                    elif message_type == "trace_save":
                                        if pending_capture is not None:
                                            raise DiagnosticCaptureError(
                                                "the most recent captured frame is "
                                                "still missing browser context"
                                            )
                                        path = await asyncio.to_thread(
                                            get_capture().save_trace,
                                            confirm=(
                                                payload.get("confirm_save") is True
                                            ),
                                        )
                                        response = {
                                            "type": "control",
                                            "status": "trace_saved",
                                            "capture": get_capture().status().as_dict(),
                                            "package_id": path.name,
                                        }
                                    elif message_type == "trace_cancel":
                                        status = get_capture().cancel_trace()
                                        pending_capture = None
                                        response = {
                                            "type": "control",
                                            "status": "trace_cancelled",
                                            "capture": status.as_dict(),
                                        }
                                    elif message_type == "failure_buffer":
                                        status = get_capture().set_failure_buffer(
                                            payload.get("enabled")  # type: ignore[arg-type]
                                        )
                                        if not status.failure_enabled:
                                            pending_capture = None
                                        response = {
                                            "type": "control",
                                            "status": (
                                                "failure_buffer_enabled"
                                                if status.failure_enabled
                                                else "failure_buffer_disabled"
                                            ),
                                            "capture": status.as_dict(),
                                        }
                                    elif message_type == "failure_mark":
                                        if pending_capture is not None:
                                            raise DiagnosticCaptureError(
                                                "the most recent captured frame is "
                                                "still missing browser context"
                                            )
                                        expectation = payload.get("expectation")
                                        if not isinstance(expectation, dict):
                                            raise DiagnosticCaptureError(
                                                "failure expectation must be an object"
                                            )
                                        path = await asyncio.to_thread(
                                            get_capture().mark_failure,
                                            FailureExpectation.from_mapping(
                                                expectation
                                            ),
                                            confirm=(
                                                payload.get("confirm_save") is True
                                            ),
                                        )
                                        response = {
                                            "type": "control",
                                            "status": "failure_saved",
                                            "capture": get_capture().status().as_dict(),
                                            "package_id": path.name,
                                        }
                                else:
                                    if (
                                        capture is not None
                                        and capture.status().trace_enabled
                                        and message_type in TRACE_MUTATING_CONTROL_TYPES
                                    ):
                                        capture_control = True
                                        raise DiagnosticCaptureError(
                                            "save or cancel the exact trace before "
                                            "changing handedness or calibration"
                                        )
                                    response = await asyncio.to_thread(
                                        processor.handle_control,
                                        payload,
                                    )
                                    if (
                                        message_type == "settings"
                                        and response.get("status") == "settings_applied"
                                    ):
                                        player_handedness = str(
                                            response.get(
                                                "player_handedness",
                                                player_handedness,
                                            )
                                        )
                            else:
                                raise ValueError(
                                    "expected a binary frame or JSON control"
                                )
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            error_response: dict[str, object] = {
                                "type": "error",
                                "scope": ("capture" if capture_control else "live"),
                                "message": str(exc),
                            }
                            if capture_control and capture is not None:
                                error_response["capture"] = capture.status().as_dict()
                            await websocket.send_json(error_response)
                            continue
                        except OSError as exc:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "scope": "capture",
                                    "message": f"local diagnostic save failed: {exc}",
                                }
                            )
                            continue
                        if send_response:
                            assert response is not None
                            await websocket.send_json(response)
                finally:
                    if capture is not None:
                        capture.disconnect()
        except WebSocketDisconnect:
            return

    return app


app = create_app()
