"""Render a debug video that overlays the fretboard homography on the input.

Per SPEC §7 Phase 3 deliverable 10. For each frame, computes a per-frame
:class:`Homography` (defaults to the keypoint backend, falls back to
geometric when keypoint confidence is below ``--fallback-conf``) and
projects:

- the canonical fretboard rectangle [0,1]×[0,1] back into image space
  (drawn as the magenta quadrilateral),
- a 6-line string grid (canonical y = i/5 for i ∈ {0..5}),
- vertical guides at canonical fret positions for frets 1..12 using
  equal-tempered spacing, with the assumption that x=1.0 corresponds
  to the 12th fret (typical neck-body junction on acoustic guitars).
  These lines won't hit real frets exactly when the detected fretboard
  extends past or stops short of the 12th fret, but they're a useful
  visual sanity-check of the homography's orientation.

A small HUD shows the homography ``method`` and ``confidence`` per frame.

Usage::

    python -m scripts.viz.overlay_fretboard input.mov
    python -m scripts.viz.overlay_fretboard input.mov --backend geometric
    python -m scripts.viz.overlay_fretboard input.mov --no-fret-grid
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tabvision.types import GuitarBBox, Homography
from tabvision.video.fretboard.geometric import GeometricFretboardBackend
from tabvision.video.fretboard.keypoint import KeypointFretboardBackend

# BGR colours.
COLOR_QUAD = (255, 0, 255)      # magenta — fretboard rectangle
COLOR_STRING = (0, 200, 255)    # yellow-orange — string guides
COLOR_FRET_LINE = (180, 180, 180)  # light grey — fret guides
COLOR_HUD_BG = (0, 0, 0)
COLOR_HUD_FG = (255, 255, 255)


def render_overlay(  # noqa: PLR0913 — wraps a clear set of CLI flags
    input_path: Path,
    output_path: Path,
    *,
    backend_name: str = "keypoint",
    checkpoint: Path | None = None,
    fallback_conf: float = 0.3,
    stride: int = 1,
    max_frames: int | None = None,
    draw_fret_grid: bool = True,
    show_progress: bool = True,
) -> dict[str, int | float]:
    """Render an overlay video. Returns a small stats dict for logging."""
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    primary, secondary = _build_backends(backend_name, checkpoint)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {output_path}")

    last_homog: Homography | None = None
    last_method = ""
    fallback_used = 0
    frame_idx = 0
    detect_count = 0
    full_frame_box = GuitarBBox(0.0, 0.0, float(width), float(height), 1.0)
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % stride == 0:
                last_homog, last_method, fell_back = _run_backends(
                    primary, secondary, frame, full_frame_box, fallback_conf
                )
                if fell_back:
                    fallback_used += 1
                detect_count += 1

            annotated = _draw(frame, last_homog, last_method, draw_fret_grid)
            writer.write(annotated)
            frame_idx += 1

            if show_progress and frame_idx % 30 == 0 and total:
                pct = 100 * frame_idx / total
                rate = frame_idx / (time.time() - t0)
                print(
                    f"[overlay_fretboard] {frame_idx}/{total} ({pct:.0f}%) — "
                    f"{rate:.1f} fps, {fallback_used} fallback frames",
                    file=sys.stderr,
                )
    finally:
        cap.release()
        writer.release()

    return {
        "frames_written": frame_idx,
        "detections_run": detect_count,
        "fallback_used": fallback_used,
        "elapsed_s": int(time.time() - t0),
    }


def _build_backends(
    name: str, checkpoint: Path | None
) -> tuple[KeypointFretboardBackend | GeometricFretboardBackend,
           KeypointFretboardBackend | GeometricFretboardBackend | None]:
    """Build the primary backend and optional secondary fallback per the spec
    decision (keypoint primary, geometric fallback)."""
    if name == "keypoint":
        from tabvision.video.guitar.yolo_backend import YoloOBBBackend
        yolo = YoloOBBBackend(checkpoint_path=checkpoint) if checkpoint else None
        primary = KeypointFretboardBackend(backend=yolo)
        secondary: GeometricFretboardBackend | None = GeometricFretboardBackend()
    elif name == "geometric":
        primary = GeometricFretboardBackend()
        secondary = None
    else:
        raise ValueError(f"unknown backend {name!r} — choose keypoint or geometric")
    return primary, secondary


def _run_backends(
    primary,  # noqa: ANN001 — duck-typed Backend protocol
    secondary,  # noqa: ANN001
    frame: np.ndarray,
    bbox: GuitarBBox,
    fallback_conf: float,
) -> tuple[Homography, str, bool]:
    """Run primary; if confidence is below ``fallback_conf`` and a
    secondary backend is configured, also run it and pick the higher-
    confidence result. Returns ``(homography, method_label, fell_back)``."""
    primary_h = primary.detect(frame, bbox)
    if primary_h.confidence >= fallback_conf or secondary is None:
        return primary_h, primary.name, False

    secondary_h = secondary.detect(frame, bbox)
    if secondary_h.confidence > primary_h.confidence:
        return secondary_h, f"{secondary.name} (fallback)", True
    return primary_h, primary.name, False


def _draw(
    frame: np.ndarray,
    homog: Homography | None,
    method_label: str,
    draw_fret_grid: bool,
) -> np.ndarray:
    """Annotate ``frame`` with the homography projection."""
    out = frame.copy()
    if homog is None or homog.confidence == 0.0:
        _draw_hud(out, "no detection", 0.0)
        return out

    # The unit-square corners projected back to image space.
    quad = _project(
        homog,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )
    _draw_quad(out, quad, COLOR_QUAD)
    _draw_string_grid(out, homog)
    if draw_fret_grid:
        _draw_fret_grid(out, homog)
    _draw_hud(out, method_label, homog.confidence)
    return out


def _project(homog: Homography, pts_canon: np.ndarray) -> np.ndarray:
    """Map (N, 2) canonical points through homography H -> image px."""
    homog_pts = np.hstack([pts_canon, np.ones((pts_canon.shape[0], 1))])
    proj = (homog.H @ homog_pts.T).T
    return proj[:, :2] / proj[:, 2:]


def _draw_quad(frame: np.ndarray, quad: np.ndarray, color: tuple[int, int, int]) -> None:
    import cv2
    cv2.polylines(frame, [quad.astype(np.int32)], isClosed=True, color=color, thickness=2)


def _draw_string_grid(frame: np.ndarray, homog: Homography) -> None:
    """Draw 6 horizontal lines at canonical y = i/5 for i ∈ {0..5}."""
    import cv2
    for i in range(6):
        y = i / 5.0
        line = _project(homog, np.array([[0.0, y], [1.0, y]]))
        cv2.line(
            frame,
            tuple(line[0].astype(int)),
            tuple(line[1].astype(int)),
            COLOR_STRING,
            1,
            cv2.LINE_AA,
        )


# Equal-tempered fret positions (rule of 18). Canonical x assumed to span
# nut (x=0) to fret 12 (x=1). Off when the detected fretboard ends before
# or extends past the 12th fret, but visually informative.
_FRET_X_CANON = [
    (1 - 1.0 / (2 ** (k / 12.0))) / (1 - 1.0 / (2 ** 1.0))
    for k in range(1, 13)
]


def _draw_fret_grid(frame: np.ndarray, homog: Homography) -> None:
    """Vertical guides at frets 1..12, assuming x=1 is at fret 12."""
    import cv2
    for k, x in enumerate(_FRET_X_CANON, start=1):
        if not 0.0 <= x <= 1.0:
            continue
        line = _project(homog, np.array([[x, 0.0], [x, 1.0]]))
        cv2.line(
            frame,
            tuple(line[0].astype(int)),
            tuple(line[1].astype(int)),
            COLOR_FRET_LINE,
            1,
            cv2.LINE_AA,
        )
        # Small fret-number label near the top edge.
        label_pt = _project(homog, np.array([[x, -0.05]]))[0]
        cv2.putText(
            frame,
            str(k),
            (int(label_pt[0]) - 4, max(12, int(label_pt[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_FRET_LINE,
            1,
            cv2.LINE_AA,
        )


def _draw_hud(frame: np.ndarray, method_label: str, confidence: float) -> None:
    """Bottom-left HUD with method + confidence."""
    import cv2

    text = f"method: {method_label}  conf: {confidence:.2f}"
    h = frame.shape[0]
    pad = 8
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame,
        (pad - 4, h - pad - th - 4),
        (pad + tw + 4, h - pad + 4),
        COLOR_HUD_BG,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (pad, h - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_HUD_FG,
        1,
        cv2.LINE_AA,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("input", type=Path, help="input video file (mp4/mov)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output mp4 path (default: <stem>_fretboard.mp4 next to input)",
    )
    parser.add_argument(
        "--backend",
        choices=["keypoint", "geometric"],
        default="keypoint",
        help="primary fretboard backend (default: keypoint, with geometric fallback)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="path to fine-tuned YOLO .pt for the keypoint backend",
    )
    parser.add_argument(
        "--fallback-conf",
        type=float,
        default=0.3,
        help="confidence below which to try the geometric fallback (keypoint backend only)",
    )
    parser.add_argument("--stride", type=int, default=1, help="run detection every Nth frame")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap output to N frames (debug aid)",
    )
    parser.add_argument(
        "--no-fret-grid",
        action="store_true",
        help="skip the projected 1..12 fret-line guides",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.stride < 1:
        print("error: --stride must be ≥ 1", file=sys.stderr)
        return 2

    output = args.out or args.input.with_name(f"{args.input.stem}_fretboard.mp4")
    print(
        f"[overlay_fretboard] {args.input} -> {output}  (backend={args.backend})",
        file=sys.stderr,
    )

    stats = render_overlay(
        args.input,
        output,
        backend_name=args.backend,
        checkpoint=args.checkpoint,
        fallback_conf=args.fallback_conf,
        stride=args.stride,
        max_frames=args.max_frames,
        draw_fret_grid=not args.no_fret_grid,
    )
    print(f"[overlay_fretboard] done: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
