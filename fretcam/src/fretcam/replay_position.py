"""Render an F3 position-state diagnostic over one public cached GAPS clip."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from fretcam.detection import DetectionChain, FrameDetection
from fretcam.position import PositionEstimate, PositionEstimator

DEFAULT_CLIP = "031_vpswc"


def _read_frame(capture: cv2.VideoCapture, timestamp_s: float) -> np.ndarray:
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
    return frame


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _diagnostic_overlay(
    frame: np.ndarray,
    detection: FrameDetection,
    estimate: PositionEstimate,
) -> np.ndarray:
    output = frame.copy()
    panel = output.copy()
    cv2.rectangle(panel, (8, 8), (430, 116), (12, 12, 12), thickness=-1)
    cv2.addWeighted(panel, 0.72, output, 0.28, 0.0, output)
    colour = {
        "locked": (70, 220, 70),
        "holding": (0, 210, 255),
        "shifting": (0, 170, 255),
        "acquiring": (0, 170, 255),
        "lost": (70, 70, 230),
    }[estimate.state]
    label = estimate.label.replace("…", "...")
    raw = "--" if estimate.raw_index_fret is None else f"{estimate.raw_index_fret:.2f}"
    smooth = (
        "--"
        if estimate.smoothed_index_fret is None
        else f"{estimate.smoothed_index_fret:.2f}"
    )
    window = ",".join(str(fret) for fret in estimate.window_frets)
    lines = (
        f"{label}  [{estimate.state}]",
        f"index fret {raw}   EMA {smooth}",
        f"window {{{window}}}   conf {estimate.confidence:.2f}",
    )
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (20, 38 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            colour if index == 0 else (235, 235, 235),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )

    index_points = [point for point in detection.hand_points if point.name == "index"]
    if index_points:
        point = index_points[0]
        cv2.circle(output, (round(point.x), round(point.y)), 7, colour, 2, cv2.LINE_AA)
    return output


def render_position_replay(
    stem: str,
    *,
    cache_dir: Path,
    output_path: Path,
    still_path: Path,
    start_s: float = 2.0,
    duration_s: float = 6.0,
    sample_fps: float = 10.0,
) -> dict[str, object]:
    """Run F2+F3 on sampled public frames and write a diagnostic MP4/still."""
    if start_s < 0.0 or duration_s <= 0.0 or sample_fps <= 0.0:
        raise ValueError("expected start >= 0, duration > 0, and sample_fps > 0")
    video_path = cache_dir / f"{stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    writer: cv2.VideoWriter | None = None
    estimator = PositionEstimator()
    state_counts: Counter[str] = Counter()
    positions: Counter[int] = Counter()
    estimator_ms: list[float] = []
    transitions: list[dict[str, object]] = []
    previous_state: str | None = None
    first_locked_s: float | None = None
    still_written = False

    try:
        with DetectionChain(detector_hz=2.0) as chain:
            for timestamp_s in np.arange(
                start_s, start_s + duration_s, 1.0 / sample_fps
            ):
                frame = _read_frame(capture, float(timestamp_s))
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        sample_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not create {output_path}")

                detection = chain.process_frame(frame, timestamp_s=float(timestamp_s))
                started = time.perf_counter()
                estimate = estimator.update(
                    index_fret=detection.index_fret,
                    vision_confidence=(
                        detection.anchor.confidence if detection.neck_locked else 0.0
                    ),
                    timestamp_s=float(timestamp_s),
                )
                estimator_ms.append((time.perf_counter() - started) * 1000.0)
                state_counts[estimate.state] += 1
                if estimate.position is not None:
                    positions[estimate.position] += 1
                if estimate.state != previous_state:
                    transitions.append(
                        {
                            "t_s": round(float(timestamp_s), 3),
                            "state": estimate.state,
                            "position": estimate.position,
                        }
                    )
                    previous_state = estimate.state
                if estimate.state == "locked" and first_locked_s is None:
                    first_locked_s = float(timestamp_s)

                rendered = _diagnostic_overlay(frame, detection, estimate)
                writer.write(rendered)
                if estimate.state == "locked" and not still_written:
                    if not cv2.imwrite(str(still_path), rendered):
                        raise RuntimeError(f"could not create {still_path}")
                    still_written = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if not still_written:
        raise RuntimeError("replay never reached a locked state")
    return {
        "clip": stem,
        "start_s": start_s,
        "duration_s": duration_s,
        "sample_fps": sample_fps,
        "frames": sum(state_counts.values()),
        "state_counts": dict(sorted(state_counts.items())),
        "locked_positions": {
            str(key): value for key, value in sorted(positions.items())
        },
        "first_lock_delay_s": (
            round(first_locked_s - start_s, 3) if first_locked_s is not None else None
        ),
        "transitions": transitions,
        "estimator_ms": {
            "median": round(statistics.median(estimator_ms), 4),
            "p95": round(_percentile(estimator_ms, 0.95), 4),
            "max": round(max(estimator_ms), 4),
        },
        "overlay_video": str(output_path.resolve()),
        "overlay_still": str(still_path.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default=DEFAULT_CLIP)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_video",
    )
    artifact_dir = Path.home() / ".tabvision" / "cache" / "fretcam_artifacts"
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_dir / "f3_031_vpswc_overlay.mp4",
    )
    parser.add_argument(
        "--still",
        type=Path,
        default=artifact_dir / "f3_031_vpswc_overlay.png",
    )
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            render_position_replay(
                args.clip,
                cache_dir=args.cache_dir,
                output_path=args.output,
                still_path=args.still,
                start_s=args.start,
                duration_s=args.duration,
                sample_fps=args.sample_fps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
