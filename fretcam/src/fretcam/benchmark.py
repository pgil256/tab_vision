"""Measured localhost WebSocket round trip for the F1 echo scaffold."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import socket
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import uvicorn
from websockets.asyncio.client import connect

from fretcam.app import create_app

# Valid 1x1 white JPEG, kept in memory and never written to disk.
SYNTHETIC_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
    "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAEf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EB//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EB//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EB//2Q=="
)


@dataclass(frozen=True)
class LoopbackMetrics:
    rounds: int
    payload_bytes: int
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_listening(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError(f"FretCam loopback server did not listen on port {port}")


async def _round_trip(uri: str, rounds: int, warmup: int) -> list[float]:
    latencies: list[float] = []
    async with connect(uri, max_size=4 * 1024 * 1024) as websocket:
        for index in range(rounds + warmup):
            started = time.perf_counter_ns()
            await websocket.send(SYNTHETIC_JPEG)
            echoed = await websocket.recv()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            if echoed != SYNTHETIC_JPEG:
                raise RuntimeError("loopback payload changed in transit")
            if index >= warmup:
                latencies.append(elapsed_ms)
    return latencies


def run_loopback_benchmark(*, rounds: int = 100, warmup: int = 10) -> LoopbackMetrics:
    if rounds < 1 or warmup < 0:
        raise ValueError("rounds must be positive and warmup must be non-negative")

    port = _free_port()
    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="fretcam-benchmark", daemon=True)
    thread.start()
    try:
        _wait_until_listening(port)
        latencies = asyncio.run(
            _round_trip(f"ws://127.0.0.1:{port}/ws", rounds, warmup)
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("FretCam loopback server did not stop")

    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return LoopbackMetrics(
        rounds=rounds,
        payload_bytes=len(SYNTHETIC_JPEG),
        median_ms=round(statistics.median(ordered), 3),
        p95_ms=round(ordered[p95_index], 3),
        min_ms=round(ordered[0], 3),
        max_ms=round(ordered[-1], 3),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            asdict(run_loopback_benchmark(rounds=args.rounds, warmup=args.warmup)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
