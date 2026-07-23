"""Replay labeled public clips through FretCam's real localhost WebSocket path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import socket
import statistics
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from itertools import product
from pathlib import Path
from typing import Literal, TypeVar

import cv2
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from fretcam.app import create_app
from fretcam.detection import ConfidenceFactors
from fretcam.position_benchmark import (
    BenchmarkManifest,
    BenchmarkSequence,
    FramePrediction,
    PositionLabel,
    Split,
    _apply_annotation_transform,
    _find_label,
    _sample_times,
    default_manifest_path,
    load_manifest,
    score_predictions,
    validate_manifest,
)
from fretcam.processing import FrameProcessorFactory

LightingTransform = Literal["native", "bright", "dim", "warm", "cool", "uneven"]
Perturbation = Literal["none", "occlusion", "camera_motion"]
MatrixMode = Literal["coverage", "cartesian"]

DEFAULT_FPS = (2.0, 5.0, 10.0, 20.0)
DEFAULT_JPEG_QUALITIES = (50, 72, 90)
DEFAULT_INFERENCE_SIZES = (320, 480, 640)
DEFAULT_LIGHTING: tuple[LightingTransform, ...] = (
    "native",
    "bright",
    "dim",
    "warm",
    "cool",
    "uneven",
)
DEFAULT_PERTURBATIONS: tuple[Perturbation, ...] = (
    "none",
    "occlusion",
    "camera_motion",
)
BASELINE_FPS = 10.0
BASELINE_JPEG_QUALITY = 72
BASELINE_INFERENCE_SIZE = 640
BASELINE_LIGHTING: LightingTransform = "native"
BASELINE_PERTURBATION: Perturbation = "none"
TEMPORARY_OCCLUSION_RECT = (0.42, 0.04, 0.88, 0.74)
_T = TypeVar("_T")


@dataclass(frozen=True)
class LiveCondition:
    """One deterministic browser-path replay condition."""

    sample_fps: float
    jpeg_quality: int
    inference_size_px: int
    lighting: LightingTransform
    perturbation: Perturbation

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_fps) or not 0.0 < self.sample_fps <= 120.0:
            raise ValueError("sample_fps must be finite and in (0, 120]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if not 64 <= self.inference_size_px <= 4096:
            raise ValueError("inference_size_px must be in [64, 4096]")
        if self.lighting not in {
            "native",
            "bright",
            "dim",
            "warm",
            "cool",
            "uneven",
        }:
            raise ValueError(f"unsupported lighting transform: {self.lighting}")
        if self.perturbation not in {"none", "occlusion", "camera_motion"}:
            raise ValueError(f"unsupported perturbation: {self.perturbation}")

    @property
    def condition_id(self) -> str:
        fps = f"{self.sample_fps:g}".replace(".", "p")
        return (
            f"fps-{fps}_q-{self.jpeg_quality}_px-{self.inference_size_px}_"
            f"{self.lighting}_{self.perturbation}"
        )

    def as_dict(self) -> dict[str, object]:
        return {"condition_id": self.condition_id, **asdict(self)}


@dataclass(frozen=True)
class _PreparedFrame:
    timestamp_s: float
    payload: bytes


@dataclass(frozen=True)
class _SequenceTransport:
    predictions: tuple[FramePrediction, ...]
    payload_bytes: tuple[int, ...]
    e2e_ms: tuple[float, ...]
    server_ms: tuple[float, ...]
    schedule_lag_ms: tuple[float, ...]
    active_elapsed_s: float


def _preferred(values: Sequence[_T], preferred: _T) -> _T:
    if not values:
        raise ValueError("condition dimensions must not be empty")
    return preferred if preferred in values else values[0]


def build_conditions(
    *,
    fps_values: Sequence[float] = DEFAULT_FPS,
    jpeg_qualities: Sequence[int] = DEFAULT_JPEG_QUALITIES,
    inference_sizes: Sequence[int] = DEFAULT_INFERENCE_SIZES,
    lighting_values: Sequence[LightingTransform] = DEFAULT_LIGHTING,
    perturbations: Sequence[Perturbation] = DEFAULT_PERTURBATIONS,
    matrix: MatrixMode = "coverage",
) -> tuple[LiveCondition, ...]:
    """Build either a bounded one-factor suite or the explicit full Cartesian grid."""
    if matrix not in {"coverage", "cartesian"}:
        raise ValueError("matrix must be 'coverage' or 'cartesian'")
    fps = tuple(float(value) for value in fps_values)
    qualities = tuple(int(value) for value in jpeg_qualities)
    sizes = tuple(int(value) for value in inference_sizes)
    lighting = tuple(lighting_values)
    effects = tuple(perturbations)
    baseline = LiveCondition(
        sample_fps=_preferred(fps, BASELINE_FPS),
        jpeg_quality=_preferred(qualities, BASELINE_JPEG_QUALITY),
        inference_size_px=_preferred(sizes, BASELINE_INFERENCE_SIZE),
        lighting=_preferred(lighting, BASELINE_LIGHTING),
        perturbation=_preferred(effects, BASELINE_PERTURBATION),
    )
    if matrix == "cartesian":
        return tuple(
            LiveCondition(
                sample_fps=sample_fps,
                jpeg_quality=quality,
                inference_size_px=size,
                lighting=light,
                perturbation=effect,
            )
            for sample_fps, quality, size, light, effect in product(
                fps,
                qualities,
                sizes,
                lighting,
                effects,
            )
        )

    candidates = [baseline]
    candidates.extend(replace(baseline, sample_fps=value) for value in fps)
    candidates.extend(replace(baseline, jpeg_quality=value) for value in qualities)
    candidates.extend(replace(baseline, inference_size_px=value) for value in sizes)
    candidates.extend(replace(baseline, lighting=value) for value in lighting)
    candidates.extend(replace(baseline, perturbation=value) for value in effects)
    unique: dict[tuple[object, ...], LiveCondition] = {}
    for condition in candidates:
        key = (
            condition.sample_fps,
            condition.jpeg_quality,
            condition.inference_size_px,
            condition.lighting,
            condition.perturbation,
        )
        unique.setdefault(key, condition)
    return tuple(unique.values())


def _apply_lighting(
    frame: np.ndarray,
    lighting: LightingTransform,
) -> np.ndarray:
    """Apply a deterministic BGR lighting stress transform."""
    if lighting == "native":
        return frame.copy()
    work = frame.astype(np.float32)
    if lighting == "bright":
        work = work * 1.18 + 14.0
    elif lighting == "dim":
        work = work * 0.42
    elif lighting == "warm":
        work *= np.asarray([0.72, 0.96, 1.18], dtype=np.float32)
    elif lighting == "cool":
        work *= np.asarray([1.18, 0.98, 0.74], dtype=np.float32)
    elif lighting == "uneven":
        height, width = work.shape[:2]
        horizontal = np.linspace(0.35, 1.25, width, dtype=np.float32)
        vertical = np.linspace(1.10, 0.72, height, dtype=np.float32)
        gain = vertical[:, None] * horizontal[None, :]
        work *= gain[:, :, None]
    else:  # pragma: no cover - guarded by LiveCondition
        raise ValueError(f"unsupported lighting transform: {lighting}")
    return np.clip(work, 0.0, 255.0).astype(np.uint8)


def _apply_camera_motion(frame: np.ndarray, progress: float) -> np.ndarray:
    """Apply bounded periodic affine motion with deterministic reflected borders."""
    height, width = frame.shape[:2]
    phase = 4.0 * math.pi * float(np.clip(progress, 0.0, 1.0))
    angle = 2.75 * math.sin(phase)
    shift_x = 0.025 * width * math.sin(phase * 0.73)
    shift_y = 0.018 * height * math.cos(phase * 0.91)
    transform = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle,
        1.0,
    )
    transform[0, 2] += shift_x
    transform[1, 2] += shift_y
    return cv2.warpAffine(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_temporary_occlusion(frame: np.ndarray, progress: float) -> np.ndarray:
    """Mask a fixed screen region only during the middle fifth of a sequence."""
    output = frame.copy()
    if not 0.40 <= progress < 0.60:
        return output
    height, width = output.shape[:2]
    left, top, right, bottom = TEMPORARY_OCCLUSION_RECT
    x0, x1 = round(left * width), round(right * width)
    y0, y1 = round(top * height), round(bottom * height)
    output[y0:y1, x0:x1] = 0
    return output


def _fit_max_dimension(frame: np.ndarray, max_dimension_px: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension_px / max(height, width, 1))
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _prepare_frame(
    frame: np.ndarray,
    *,
    label: PositionLabel | None,
    condition: LiveCondition,
    progress: float,
) -> bytes:
    output = frame
    if condition.perturbation == "camera_motion":
        output = _apply_camera_motion(output, progress)
    output = _apply_lighting(output, condition.lighting)
    output = _apply_annotation_transform(output, label)
    if condition.perturbation == "occlusion":
        output = _apply_temporary_occlusion(output, progress)
    output = _fit_max_dimension(output, condition.inference_size_px)
    encoded, jpeg = cv2.imencode(
        ".jpg",
        output,
        [cv2.IMWRITE_JPEG_QUALITY, condition.jpeg_quality],
    )
    if not encoded:
        raise RuntimeError("could not encode live benchmark frame")
    return jpeg.tobytes()


def _prepare_sequence_payloads(
    sequence: BenchmarkSequence,
    condition: LiveCondition,
    video_cache: Path,
) -> list[_PreparedFrame]:
    """Read one manifest-declared cached public clip and prepare browser JPEGs."""
    video_path = video_cache / f"{sequence.source}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"GAPS cache miss: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    prepared: list[_PreparedFrame] = []
    duration = sequence.end_s - sequence.start_s
    try:
        for timestamp_s in _sample_times(
            sequence.start_s,
            sequence.end_s,
            condition.sample_fps,
        ):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame at {timestamp_s:.3f}s")
            progress = (timestamp_s - sequence.start_s) / max(duration, 1e-9)
            payload = _prepare_frame(
                frame,
                label=_find_label(sequence, timestamp_s),
                condition=condition,
                progress=progress,
            )
            prepared.append(
                _PreparedFrame(
                    timestamp_s=round(timestamp_s, 6),
                    payload=payload,
                )
            )
    finally:
        capture.release()
    return prepared


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_listening(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError(f"FretCam live benchmark did not listen on port {port}")


@contextmanager
def _serve_live_path(
    *,
    processor_factory: FrameProcessorFactory | None,
    startup_timeout_s: float,
) -> Iterator[str]:
    port = _free_port()
    app = create_app(processor_factory=processor_factory)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        name="fretcam-live-accuracy",
        daemon=True,
    )
    thread.start()
    try:
        _wait_until_listening(port, startup_timeout_s)
        yield f"ws://127.0.0.1:{port}/ws"
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)
        if thread.is_alive():
            raise RuntimeError("FretCam live benchmark server did not stop")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"HUD response field {field!r} is not an object")
    return value


def _confidence_factors(raw: object) -> ConfidenceFactors:
    data = _mapping(raw, "detection.confidence_factors")
    blockers = data.get("blockers", ())
    if not isinstance(blockers, (list, tuple)):
        raise RuntimeError("confidence factor blockers are not an array")
    values = {
        field.name: float(data.get(field.name, 0.0))
        for field in fields(ConfidenceFactors)
        if field.name != "blockers"
    }
    return ConfidenceFactors(  # type: ignore[arg-type]
        **values,
        blockers=tuple(str(value) for value in blockers),
    )


def _prediction_from_hud(
    response: Mapping[str, object],
    *,
    sequence: BenchmarkSequence,
    timestamp_s: float,
) -> FramePrediction:
    if response.get("type") != "hud":
        raise RuntimeError(f"HUD server returned an unexpected response: {response}")
    detection = _mapping(response.get("detection"), "detection")
    position = _mapping(response.get("position"), "position")
    raw_position = position.get("position")
    return FramePrediction(
        sequence_id=sequence.sequence_id,
        source=sequence.source,
        split=sequence.split,
        timestamp_s=round(timestamp_s, 6),
        state=str(position.get("state", "lost")),
        position=None if raw_position is None else int(raw_position),
        confidence=float(position.get("confidence", 0.0)),
        observation_valid=position.get("raw_index_fret") is not None,
        confidence_factors=_confidence_factors(detection.get("confidence_factors")),
        geometry_status=str(detection.get("geometry_status", "unknown")),
        geometry_age_ms=float(detection.get("geometry_age_ms", 0.0)),
    )


async def _receive_json(websocket: object) -> Mapping[str, object]:
    raw = await websocket.recv()  # type: ignore[attr-defined]
    if not isinstance(raw, str):
        raise RuntimeError("HUD server returned a binary response")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise RuntimeError("HUD server returned a non-object JSON response")
    return decoded


async def _stream_sequence(
    uri: str,
    *,
    sequence: BenchmarkSequence,
    frames: Sequence[_PreparedFrame],
    player_handedness: str,
    pace: bool,
) -> _SequenceTransport:
    predictions: list[FramePrediction] = []
    payload_sizes: list[int] = []
    e2e_ms: list[float] = []
    server_ms: list[float] = []
    schedule_lag_ms: list[float] = []
    active_started: float | None = None
    active_finished: float | None = None
    async with connect(uri, max_size=8 * 1024 * 1024) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "settings",
                    "player_handedness": player_handedness,
                }
            )
        )
        control = await _receive_json(websocket)
        if control.get("type") != "control":
            raise RuntimeError(f"settings control failed: {control}")
        schedule_started = asyncio.get_running_loop().time()
        for frame in frames:
            target = schedule_started + frame.timestamp_s - sequence.start_s
            if pace:
                delay = target - asyncio.get_running_loop().time()
                if delay > 0.0:
                    await asyncio.sleep(delay)
            lag_ms = max(
                0.0,
                (asyncio.get_running_loop().time() - target) * 1000.0,
            )
            started = time.perf_counter()
            if active_started is None:
                active_started = started
            await websocket.send(frame.payload)
            response = await _receive_json(websocket)
            finished = time.perf_counter()
            active_finished = finished
            predictions.append(
                _prediction_from_hud(
                    response,
                    sequence=sequence,
                    timestamp_s=frame.timestamp_s,
                )
            )
            payload_sizes.append(len(frame.payload))
            e2e_ms.append((finished - started) * 1000.0)
            server_ms.append(float(response.get("server_ms", 0.0)))
            schedule_lag_ms.append(lag_ms)
    active_elapsed = (
        0.0
        if active_started is None or active_finished is None
        else max(0.0, active_finished - active_started)
    )
    return _SequenceTransport(
        predictions=tuple(predictions),
        payload_bytes=tuple(payload_sizes),
        e2e_ms=tuple(e2e_ms),
        server_ms=tuple(server_ms),
        schedule_lag_ms=tuple(schedule_lag_ms),
        active_elapsed_s=active_elapsed,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _transport_summary(
    *,
    condition: LiveCondition,
    streams: Sequence[_SequenceTransport],
) -> dict[str, object]:
    payloads = [value for stream in streams for value in stream.payload_bytes]
    latencies = [value for stream in streams for value in stream.e2e_ms]
    server = [value for stream in streams for value in stream.server_ms]
    schedule_lag = [value for stream in streams for value in stream.schedule_lag_ms]
    active_elapsed = sum(stream.active_elapsed_s for stream in streams)
    frames = len(latencies)
    return {
        "frames": frames,
        "target_fps": condition.sample_fps,
        "effective_fps": (
            round(frames / active_elapsed, 3) if active_elapsed > 0.0 else None
        ),
        "active_elapsed_s": round(active_elapsed, 3),
        "payload_median_bytes": round(statistics.median(payloads)),
        "payload_p95_bytes": round(_percentile(payloads, 0.95)),
        "e2e_median_ms": round(statistics.median(latencies), 3),
        "e2e_p95_ms": round(_percentile(latencies, 0.95), 3),
        "server_median_ms": round(statistics.median(server), 3),
        "server_p95_ms": round(_percentile(server, 0.95), 3),
        "schedule_lag_p95_ms": round(_percentile(schedule_lag, 0.95), 3),
        "schedule_lag_max_ms": round(max(schedule_lag), 3),
    }


async def _run_conditions(
    uri: str,
    *,
    manifest: BenchmarkManifest,
    conditions: Sequence[LiveCondition],
    video_cache: Path,
    player_handedness: str,
    pace: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for condition in conditions:
        condition_manifest = replace(manifest, sample_fps=condition.sample_fps)
        streams: list[_SequenceTransport] = []
        for sequence in condition_manifest.sequences:
            prepared = _prepare_sequence_payloads(
                sequence,
                condition,
                video_cache,
            )
            streams.append(
                await _stream_sequence(
                    uri,
                    sequence=sequence,
                    frames=prepared,
                    player_handedness=player_handedness,
                    pace=pace,
                )
            )
        predictions = [
            prediction for stream in streams for prediction in stream.predictions
        ]
        results.append(
            {
                "condition": condition.as_dict(),
                "transport": _transport_summary(
                    condition=condition,
                    streams=streams,
                ),
                "metrics": score_predictions(condition_manifest, predictions),
                "predictions": [asdict(prediction) for prediction in predictions],
            }
        )
    return results


def run_live_benchmark(
    manifest: BenchmarkManifest,
    *,
    video_cache: Path,
    conditions: Sequence[LiveCondition],
    splits: set[Split] | None = None,
    sequence_ids: set[str] | None = None,
    player_handedness: str = "right",
    processor_factory: FrameProcessorFactory | None = None,
    startup_timeout_s: float = 90.0,
    pace: bool = True,
) -> dict[str, object]:
    """Run selected manifest windows through a real uvicorn/WebSocket HUD server."""
    validate_manifest(manifest)
    if player_handedness not in {"right", "left"}:
        raise ValueError("player_handedness must be 'right' or 'left'")
    if not conditions:
        raise ValueError("at least one live condition is required")
    selected_splits = splits or {"dev"}
    selected_sequences = tuple(
        sequence
        for sequence in manifest.sequences
        if sequence.split in selected_splits
        and (sequence_ids is None or sequence.sequence_id in sequence_ids)
    )
    if not selected_sequences:
        raise ValueError("no benchmark sequences matched the selection")
    if sequence_ids is not None:
        found = {sequence.sequence_id for sequence in selected_sequences}
        missing = sequence_ids - found
        if missing:
            raise ValueError(f"unknown or unselected sequence ids: {sorted(missing)}")
    selected_manifest = replace(manifest, sequences=selected_sequences)
    with _serve_live_path(
        processor_factory=processor_factory,
        startup_timeout_s=startup_timeout_s,
    ) as uri:
        condition_results = asyncio.run(
            _run_conditions(
                uri,
                manifest=selected_manifest,
                conditions=conditions,
                video_cache=video_cache,
                player_handedness=player_handedness,
                pace=pace,
            )
        )
    return {
        "schema_version": 1,
        "benchmark": "fretcam-live-position-websocket",
        "corpus": manifest.corpus,
        "corpus_license": manifest.corpus_license,
        "public_only": manifest.public_only,
        "paced": pace,
        "player_handedness": player_handedness,
        "sequence_ids": [sequence.sequence_id for sequence in selected_sequences],
        "conditions": condition_results,
    }


def summarize_results(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a compact console summary while full predictions remain optional."""
    raw_conditions = payload.get("conditions", ())
    if not isinstance(raw_conditions, Sequence):
        raise ValueError("live benchmark payload has no condition sequence")
    summaries = []
    for raw in raw_conditions:
        result = _mapping(raw, "conditions[]")
        condition = _mapping(result.get("condition"), "condition")
        transport = _mapping(result.get("transport"), "transport")
        metrics = _mapping(result.get("metrics"), "metrics")
        overall = _mapping(metrics.get("overall"), "metrics.overall")
        blockers = _mapping(metrics.get("blockers"), "metrics.blockers")
        blocker_overall = _mapping(blockers.get("overall"), "blockers.overall")
        summaries.append(
            {
                "condition_id": condition.get("condition_id"),
                "displayed_position_precision": overall.get(
                    "displayed_position_precision"
                ),
                "coverage": overall.get("coverage"),
                "valid_observation_rate": overall.get("valid_observation_rate"),
                "blocker_counts": blocker_overall.get("counts"),
                "e2e_p95_ms": transport.get("e2e_p95_ms"),
                "schedule_lag_p95_ms": transport.get("schedule_lag_p95_ms"),
            }
        )
    return {
        "schema_version": payload.get("schema_version"),
        "benchmark": payload.get("benchmark"),
        "public_only": payload.get("public_only"),
        "paced": payload.get("paced"),
        "conditions": summaries,
    }


def _selected_conditions(
    conditions: Sequence[LiveCondition],
    only: Sequence[str] | None,
) -> tuple[LiveCondition, ...]:
    if not only:
        return tuple(conditions)
    by_id = {condition.condition_id: condition for condition in conditions}
    missing = set(only) - by_id.keys()
    if missing:
        raise ValueError(f"unknown condition ids: {sorted(missing)}")
    return tuple(by_id[value] for value in only)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument(
        "--video-cache",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_video",
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--sequence", action="append", dest="sequence_ids")
    parser.add_argument("--fps", nargs="+", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--jpeg-quality",
        nargs="+",
        type=int,
        default=DEFAULT_JPEG_QUALITIES,
    )
    parser.add_argument(
        "--inference-size",
        nargs="+",
        type=int,
        default=DEFAULT_INFERENCE_SIZES,
    )
    parser.add_argument(
        "--lighting",
        nargs="+",
        choices=DEFAULT_LIGHTING,
        default=DEFAULT_LIGHTING,
    )
    parser.add_argument(
        "--perturbation",
        nargs="+",
        choices=DEFAULT_PERTURBATIONS,
        default=DEFAULT_PERTURBATIONS,
    )
    parser.add_argument(
        "--matrix",
        choices=("coverage", "cartesian"),
        default="coverage",
    )
    parser.add_argument("--only", nargs="+", metavar="CONDITION_ID")
    parser.add_argument(
        "--player-handedness",
        choices=("right", "left"),
        default="right",
    )
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)

    conditions = _selected_conditions(
        build_conditions(
            fps_values=args.fps,
            jpeg_qualities=args.jpeg_quality,
            inference_sizes=args.inference_size,
            lighting_values=args.lighting,
            perturbations=args.perturbation,
            matrix=args.matrix,
        ),
        args.only,
    )
    if args.list_conditions:
        print(
            json.dumps(
                {
                    "matrix": args.matrix,
                    "count": len(conditions),
                    "conditions": [condition.as_dict() for condition in conditions],
                },
                indent=2,
            )
        )
        return

    manifest = load_manifest(args.manifest)
    splits: set[Split] = {"dev", "test"} if args.split == "all" else {args.split}
    payload = run_live_benchmark(
        manifest,
        video_cache=args.video_cache,
        conditions=conditions,
        splits=splits,
        sequence_ids=(
            set(args.sequence_ids) if args.sequence_ids is not None else None
        ),
        player_handedness=args.player_handedness,
    )
    payload["manifest"] = str(args.manifest.resolve())
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summarize_results(payload), indent=2))


if __name__ == "__main__":
    main()
