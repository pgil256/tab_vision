"""Replay the F2 chain against three public GAPS clips from the local cache."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from fretcam.detection import DetectionChain, FrameDetection

DEFAULT_CLIPS = ("027_Zpswc", "031_vpswc", "043_bc1wc")


@dataclass(frozen=True)
class ClipGate:
    clip: str
    passed: bool
    attempts: int
    video_timestamp_s: float | None
    center_fret: float | None
    anchor_confidence: float
    homography_confidence: float
    fret_map_locked: bool
    tracking_frame_locked: bool


def _read_frame(
    capture: cv2.VideoCapture, timestamp_s: float
) -> tuple[np.ndarray, float]:
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
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    return frame, fps


def _plausible(result: FrameDetection, max_fret: int) -> bool:
    anchor = result.anchor
    return (
        result.neck_locked
        and anchor.confidence > 0.0
        and all(
            math.isfinite(value)
            for value in (anchor.min_fret, anchor.center_fret, anchor.max_fret)
        )
        and 0.5 < anchor.center_fret < max_fret - 0.5
        and 0.0 <= anchor.min_fret <= anchor.center_fret <= anchor.max_fret <= max_fret
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(results: list[FrameDetection]) -> dict[str, dict[str, float]]:
    fields = ("detector_ms", "homography_ms", "hand_ms", "anchor_ms", "total_ms")
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(getattr(result.stage_latency, field)) for result in results]
        if field in {"detector_ms", "homography_ms"}:
            values = [value for value in values if value > 0.0]
        if not values:
            continue
        summary[field] = {
            "median": round(statistics.median(values), 3),
            "p95": round(_percentile(values, 0.95), 3),
            "max": round(max(values), 3),
        }
    return summary


def run_replay(
    clips: Sequence[str],
    *,
    cache_dir: Path,
    start_s: float = 2.0,
    stop_s: float = 30.0,
    step_s: float = 0.5,
) -> dict[str, object]:
    if not clips:
        raise ValueError("at least one clip is required")
    if start_s < 0.0 or stop_s <= start_s or step_s <= 0.0:
        raise ValueError("expected 0 <= start < stop and step > 0")

    clip_gates: list[ClipGate] = []
    observations: list[FrameDetection] = []
    with DetectionChain(detector_hz=2.0) as chain:
        for stem in clips:
            video_path = cache_dir / f"{stem}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(f"GAPS cache miss: {video_path}")
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open {video_path}")
            chain.reset()
            attempts = 0
            success: FrameDetection | None = None
            success_time: float | None = None
            tracking_locked = False
            try:
                for timestamp_s in np.arange(start_s, stop_s, step_s):
                    frame, fps = _read_frame(capture, float(timestamp_s))
                    result = chain.process_frame(frame, timestamp_s=float(timestamp_s))
                    observations.append(result)
                    attempts += 1
                    if not _plausible(result, chain.guitar_config.max_fret):
                        continue
                    success = result
                    success_time = float(timestamp_s)
                    next_frame, _ = _read_frame(capture, float(timestamp_s) + 1.0 / fps)
                    tracked = chain.process_frame(
                        next_frame, timestamp_s=float(timestamp_s) + 1.0 / fps
                    )
                    observations.append(tracked)
                    tracking_locked = tracked.neck_locked and not tracked.detector_ran
                    break
            finally:
                capture.release()

            clip_gates.append(
                ClipGate(
                    clip=stem,
                    passed=success is not None and tracking_locked,
                    attempts=attempts,
                    video_timestamp_s=success_time,
                    center_fret=(
                        round(success.anchor.center_fret, 3) if success else None
                    ),
                    anchor_confidence=(
                        round(success.anchor.confidence, 3) if success else 0.0
                    ),
                    homography_confidence=(
                        round(success.homography_confidence, 3) if success else 0.0
                    ),
                    fret_map_locked=(success.fret_map_locked if success else False),
                    tracking_frame_locked=tracking_locked,
                )
            )

    passed_clips = sum(gate.passed for gate in clip_gates)
    return {
        "verdict": "pass" if passed_clips >= 3 else "fail",
        "passed_clips": passed_clips,
        "required_clips": 3,
        "input_max_dimension_px": 640,
        "clips": [asdict(gate) for gate in clip_gates],
        "stage_ms": _latency_summary(observations),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", default=list(DEFAULT_CLIPS))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_video",
    )
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--stop", type=float, default=30.0)
    parser.add_argument(
        "--step",
        type=float,
        default=0.5,
        help="sample cadence in seconds (default 0.5 = detector's configured 2 Hz)",
    )
    args = parser.parse_args(argv)
    report = run_replay(
        args.clips,
        cache_dir=args.cache_dir,
        start_s=args.start,
        stop_s=args.stop,
        step_s=args.step,
    )
    print(json.dumps(report, indent=2))
    if report["verdict"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
