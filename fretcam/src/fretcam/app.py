"""FastAPI application for the FretCam loopback scaffold."""

from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024
STATIC_DIR = files("fretcam").joinpath("static")


def create_app(*, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> FastAPI:
    """Create the echo-mode FretCam application.

    F1 accepts one binary JPEG payload at a time and echoes the exact bytes.
    Later phases replace the echo with frame processing while preserving the
    browser transport. Frames are held only for the duration of a request.
    """

    app = FastAPI(title="FretCam", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR.joinpath("index.html"))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "echo"}

    @app.websocket("/ws")
    async def echo_frames(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                frame = await websocket.receive_bytes()
                if len(frame) > max_frame_bytes:
                    await websocket.close(code=1009, reason="frame too large")
                    return
                await websocket.send_bytes(frame)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
