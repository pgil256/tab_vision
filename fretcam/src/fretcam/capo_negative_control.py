r"""False-positive control for capo detection on real, capo-free footage.

There is no capo ground truth in this repository — GAPS carries no capo column
and none of its 404 MusicXML scores encode ``<capo>`` — so the true-positive
rate on real capos cannot be measured here. What *can* be measured, and is the
thing that actually gates safety for a reporting feature, is the **false
positive rate on real footage that certainly has no capo**.

GAPS clean-12 is solo classical guitar. Classical players do not use capos, and
none of these scores declare one. A detector that reports a capo on any of them
would put a wrong number in front of a human, which is worse than staying quiet.

This performs real inference and takes a few minutes; frames are sampled at
:data:`SAMPLE_HZ` rather than every frame, because a capo is static and the
estimate is session-level.

Reproduce from ``tabvision/`` with the sibling FretCam package installed:

    $env:PYTHONPATH = ((Resolve-Path '../fretcam/src').Path + ';' + (Get-Location).Path)
    .\.venv\Scripts\python -m fretcam.capo_negative_control
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cv2

from fretcam.capo import CapoDetector, CapoObservation
from fretcam.detection import DetectionChain
from scripts.acquire.gaps_video import CLEAN_12
from tabvision.types import GuitarConfig

VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video"
SAMPLE_HZ = 2.0
"""Frames sampled per second. A capo does not move; 2 Hz is ample."""

MAX_FRAMES = 240
"""Cap per clip so the control stays a few minutes, not an hour."""


def analyze(stem: str, *, sample_hz: float, max_frames: int) -> CapoObservation:
    path = VIDEO_CACHE / f"{stem}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / max(sample_hz, 1e-6))))

    cfg = GuitarConfig()
    chain = DetectionChain(
        guitar_config=cfg,
        detector_hz=2.0,
        background_detector=False,
        crop_hand=True,
    )
    detector = CapoDetector()
    try:
        index = 0
        used = 0
        while used < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            timestamp = index / fps
            index += 1

            height, width = frame.shape[:2]
            scale = min(1.0, 640 / max(width, 1), 480 / max(height, 1))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            detection = chain.process_frame(frame, timestamp_s=timestamp)
            detector.observe(
                frame,
                detection.fret_ticks,
                neck_quad=detection.neck_quad,
                body_joint_fret=detection.body_joint_fret,
            )
            used += 1
    finally:
        chain.close()
        capture.release()
    return detector.estimate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", default="clean12")
    parser.add_argument("--sample-hz", type=float, default=SAMPLE_HZ)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )

    rows: list[dict[str, object]] = []
    false_positives = 0
    for stem in stems:
        result = analyze(stem, sample_hz=args.sample_hz, max_frames=args.max_frames)
        if result.detected:
            false_positives += 1
        rows.append(
            {
                "clip": stem,
                "fret": result.fret,
                "confidence": round(result.confidence, 4),
                "frames_observed": result.frames_observed,
                "frames_supporting": result.frames_supporting,
                "margin": round(result.margin, 4),
                "reason": result.reason,
            }
        )
        flag = "FALSE POSITIVE" if result.detected else "ok"
        print(
            f"  {stem:12s} {flag:14s} fret={result.fret} conf={result.confidence:.3f} "
            f"frames={result.frames_observed:3d} margin={result.margin:.3f} "
            f"({result.reason})",
            flush=True,
        )

    print()
    print(f"false positives: {false_positives} / {len(rows)} capo-free clips")
    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "population": "gaps-clean12-known-capo-free",
                    "sample_hz": args.sample_hz,
                    "max_frames": args.max_frames,
                    "false_positives": false_positives,
                    "clips": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
