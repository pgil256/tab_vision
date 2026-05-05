"""Render a debug video that overlays YOLO-OBB detections on the input.

Per SPEC §7 Phase 3 deliverable 10. Reads an input video, runs the
fine-tuned YOLO-OBB backend on each frame, and writes an MP4 with the
detected ``neck`` / ``fret`` / ``nut`` oriented bounding boxes drawn
on top, colour-coded by class plus confidence labels.

Usage::

    python -m scripts.viz.overlay_guitar input.mov
    python -m scripts.viz.overlay_guitar input.mov --out /tmp/dbg.mp4
    python -m scripts.viz.overlay_guitar input.mov --stride 5 --conf 0.3

By default the output goes next to the input as ``<stem>_overlay.mp4``.
``--stride`` skips frames between detections (the skipped frames are
still written but show the most recent overlay) — useful for fast
turnaround on long takes; ``--stride 1`` runs detection on every frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tabvision.video.guitar.yolo_backend import (
    CLASS_FRET,
    CLASS_NECK,
    CLASS_NUT,
    OBBDetection,
    OBBPredictions,
    YoloOBBBackend,
)

# BGR colours; chosen to be distinct on a typical wood-grain fretboard.
COLOR_NECK = (0, 255, 0)      # green
COLOR_FRET = (0, 165, 255)    # orange
COLOR_NUT = (255, 0, 255)     # magenta
COLOR_TEXT = (255, 255, 255)  # white


def render_overlay(  # noqa: PLR0913 — wraps a clear set of CLI flags
    input_path: Path,
    output_path: Path,
    *,
    checkpoint: Path | None = None,
    conf: float = 0.25,
    stride: int = 1,
    max_frames: int | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    """Render an overlay video. Returns a small stats dict for logging."""
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    backend = YoloOBBBackend(checkpoint_path=checkpoint, conf=conf)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {output_path}")

    last_preds = OBBPredictions()
    frame_idx = 0
    detect_count = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % stride == 0:
                last_preds = backend.predict_all(frame)
                detect_count += 1

            annotated = _draw_predictions(frame, last_preds)
            writer.write(annotated)
            frame_idx += 1

            if show_progress and frame_idx % 30 == 0 and total:
                pct = 100 * frame_idx / total
                rate = frame_idx / (time.time() - t0)
                print(
                    f"[overlay_guitar] {frame_idx}/{total} ({pct:.0f}%) — "
                    f"{rate:.1f} fps, {detect_count} detections",
                    file=sys.stderr,
                )
    finally:
        cap.release()
        writer.release()

    return {
        "frames_written": frame_idx,
        "detections_run": detect_count,
        "elapsed_s": int(time.time() - t0),
    }


def _draw_predictions(frame: np.ndarray, preds: OBBPredictions) -> np.ndarray:
    """Annotate ``frame`` (BGR, in-place safe) with all detections in ``preds``."""
    out = frame.copy()
    # Order matters visually: draw the big neck first so smaller fret/nut
    # boxes stay on top.
    if neck := preds.best_neck():
        _draw_obb(out, neck, COLOR_NECK)
    for det in preds.frets:
        _draw_obb(out, det, COLOR_FRET, label_short=True)
    if nut := preds.best_nut():
        _draw_obb(out, nut, COLOR_NUT)

    _draw_legend(out, preds)
    return out


def _draw_obb(
    frame: np.ndarray,
    obb: OBBDetection,
    color: tuple[int, int, int],
    *,
    label_short: bool = False,
) -> None:
    """Draw an oriented bbox with a label."""
    import cv2

    corners = _obb_corners(obb).astype(np.int32)
    cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=2)

    label = f"{obb.confidence:.2f}" if label_short else f"{obb.class_name} {obb.confidence:.2f}"
    label_pos = (corners[0, 0], max(15, corners[0, 1] - 4))
    cv2.putText(
        frame,
        label,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )


def _obb_corners(obb: OBBDetection) -> np.ndarray:
    """4×2 corners of an OBB in image coordinates.

    Mirrors ``video.fretboard.keypoint._obb_to_corners`` but kept local
    so the viz tools have no internal dep on the fretboard module.
    """
    rad = float(np.radians(obb.rotation_deg))
    cos_r, sin_r = float(np.cos(rad)), float(np.sin(rad))
    half_w, half_h = obb.w / 2.0, obb.h / 2.0
    local = np.array(
        [
            [+half_w, +half_h],
            [-half_w, +half_h],
            [-half_w, -half_h],
            [+half_w, -half_h],
        ],
        dtype=np.float64,
    )
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)
    rotated = local @ rot.T
    rotated[:, 0] += obb.cx
    rotated[:, 1] += obb.cy
    return rotated


def _draw_legend(frame: np.ndarray, preds: OBBPredictions) -> None:
    """Top-left legend with detection counts per class."""
    import cv2

    lines = [
        (f"{CLASS_NECK}: {len(preds.neck)}", COLOR_NECK),
        (f"{CLASS_FRET}: {len(preds.frets)}", COLOR_FRET),
        (f"{CLASS_NUT}: {len(preds.nut)}", COLOR_NUT),
    ]
    for i, (text, color) in enumerate(lines):
        y = 20 + 18 * i
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("input", type=Path, help="input video file (mp4/mov)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output mp4 path (default: <stem>_overlay.mp4 next to input)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="path to fine-tuned .pt; defaults to TABVISION_GUITAR_YOLO_CHECKPOINT "
        "or ~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="run detection every Nth frame (1 = every frame)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap output to N frames (debug aid for long takes)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.stride < 1:
        print("error: --stride must be ≥ 1", file=sys.stderr)
        return 2

    output = args.out or args.input.with_name(f"{args.input.stem}_overlay.mp4")
    print(f"[overlay_guitar] {args.input} -> {output}", file=sys.stderr)

    stats = render_overlay(
        args.input,
        output,
        checkpoint=args.checkpoint,
        conf=args.conf,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(f"[overlay_guitar] done: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
