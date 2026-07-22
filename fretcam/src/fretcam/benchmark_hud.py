"""Measure the complete localhost HUD path on public cached GAPS frames."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import socket
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import uvicorn
from websockets.asyncio.client import connect

from fretcam.app import create_app

DEFAULT_CLIP = "031_vpswc"


@dataclass(frozen=True)
class HudBenchmarkMetrics:
    clip: str
    rounds: int
    warmup: int
    input_max_dimension_px: int
    median_payload_bytes: int
    throughput_fps: float
    e2e_median_ms: float
    e2e_p95_ms: float
    e2e_max_ms: float
    server_median_ms: float
    server_p95_ms: float
    detector_frames: int
    neck_locked_frames: int
    target_fps: float = 10.0
    target_latency_ms: float = 150.0

    @property
    def verdict(self) -> str:
        return (
            "pass"
            if self.throughput_fps >= self.target_fps
            and self.e2e_p95_ms <= self.target_latency_ms
            else "fail"
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_listening(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError(f"FretCam HUD server did not listen on port {port}")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _jpeg_frames(
    video_path: Path,
    *,
    count: int,
    start_s: float,
    sample_fps: float,
) -> list[bytes]:
    if not video_path.exists():
        raise FileNotFoundError(f"GAPS cache miss: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    payloads: list[bytes] = []
    try:
        for index in range(count):
            timestamp_s = start_s + index / sample_fps
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame at {timestamp_s:.3f}s")
            height, width = frame.shape[:2]
            scale = min(1.0, 640.0 / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            encoded, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
            )
            if not encoded:
                raise RuntimeError("could not encode benchmark frame")
            payloads.append(jpeg.tobytes())
    finally:
        capture.release()
    return payloads


async def _round_trip(
    uri: str,
    payloads: list[bytes],
    *,
    warmup: int,
) -> tuple[list[float], list[dict[str, object]]]:
    latencies: list[float] = []
    responses: list[dict[str, object]] = []
    async with connect(uri, max_size=4 * 1024 * 1024) as websocket:
        for index, payload in enumerate(payloads):
            started = time.perf_counter()
            await websocket.send(payload)
            raw = await websocket.recv()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(raw, str):
                raise RuntimeError("HUD server returned a binary response")
            response = json.loads(raw)
            if response.get("type") != "hud":
                raise RuntimeError(f"HUD server error: {response}")
            if index >= warmup:
                latencies.append(elapsed_ms)
                responses.append(response)
    return latencies, responses


def run_hud_benchmark(
    *,
    clip: str = DEFAULT_CLIP,
    cache_dir: Path = Path.home() / ".tabvision" / "cache" / "gaps_video",
    rounds: int = 30,
    warmup: int = 10,
    start_s: float = 2.0,
    sample_fps: float = 10.0,
) -> HudBenchmarkMetrics:
    if rounds < 1 or warmup < 0 or start_s < 0.0 or sample_fps <= 0.0:
        raise ValueError("invalid benchmark bounds")
    payloads = _jpeg_frames(
        cache_dir / f"{clip}.mp4",
        count=rounds + warmup,
        start_s=start_s,
        sample_fps=sample_fps,
    )

    port = _free_port()
    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="fretcam-hud-benchmark", daemon=True)
    thread.start()
    try:
        _wait_until_listening(port)
        latencies, responses = asyncio.run(
            _round_trip(
                f"ws://127.0.0.1:{port}/ws",
                payloads,
                warmup=warmup,
            )
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("FretCam HUD server did not stop")

    server_ms = [float(response["server_ms"]) for response in responses]
    detector_frames = sum(
        bool(response["detection"]["detector_ran"])  # type: ignore[index]
        for response in responses
    )
    neck_locked_frames = sum(
        bool(response["detection"]["neck_locked"])  # type: ignore[index]
        for response in responses
    )
    return HudBenchmarkMetrics(
        clip=clip,
        rounds=rounds,
        warmup=warmup,
        input_max_dimension_px=640,
        median_payload_bytes=round(statistics.median(map(len, payloads[warmup:]))),
        throughput_fps=round(1000.0 * rounds / sum(latencies), 3),
        e2e_median_ms=round(statistics.median(latencies), 3),
        e2e_p95_ms=round(_percentile(latencies, 0.95), 3),
        e2e_max_ms=round(max(latencies), 3),
        server_median_ms=round(statistics.median(server_ms), 3),
        server_p95_ms=round(_percentile(server_ms, 0.95), 3),
        detector_frames=detector_frames,
        neck_locked_frames=neck_locked_frames,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_video",
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    metrics = run_hud_benchmark(
        clip=args.clip,
        cache_dir=args.cache_dir,
        rounds=args.rounds,
        warmup=args.warmup,
        start_s=args.start,
        sample_fps=args.sample_fps,
    )
    report = {"verdict": metrics.verdict, **asdict(metrics)}
    print(json.dumps(report, indent=2))
    if metrics.verdict != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
