"""FastAPI application for the local FretCam live HUD."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fretcam.processing import FrameProcessorFactory, HudFrameProcessor

DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024
STATIC_DIR = files("fretcam").joinpath("static")


def create_app(
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    echo_mode: bool = False,
    processor_factory: FrameProcessorFactory | None = None,
) -> FastAPI:
    """Create the HUD app, or the retained F1 echo harness for its benchmark."""

    if max_frame_bytes < 1:
        raise ValueError("max_frame_bytes must be positive")
    factory = processor_factory or HudFrameProcessor

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
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        return
                    frame = message.get("bytes")
                    text = message.get("text")
                    try:
                        if frame is not None:
                            if len(frame) > max_frame_bytes:
                                await websocket.close(
                                    code=1009, reason="frame too large"
                                )
                                return
                            response = await asyncio.to_thread(
                                processor.process_jpeg, frame
                            )
                        elif text is not None:
                            payload = json.loads(text)
                            if not isinstance(payload, dict):
                                raise ValueError("control payload must be an object")
                            response = await asyncio.to_thread(
                                processor.handle_control, payload
                            )
                        else:
                            raise ValueError("expected a binary frame or JSON control")
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        await websocket.send_json(
                            {"type": "error", "message": str(exc)}
                        )
                        continue
                    await websocket.send_json(response)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
