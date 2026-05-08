"""Render a debug video that overlays the per-frame fingering heat-map.

Per SPEC §7 Phase 4 deliverable 3. For each frame, computes a
:class:`Homography` (keypoint backend) and a :class:`FrameFingering`
(MediaPipe + fingertip-to-fret posterior), then projects the
``(string, fret)`` marginal back into image space as a colour-coded
overlay.

Usage::

    python -m scripts.viz.overlay_hand input.mov
    python -m scripts.viz.overlay_hand input.mov --no-fingertips
    python -m scripts.viz.overlay_hand input.mov --max-frames 90 --stride 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tabvision.types import FrameFingering, GuitarBBox, GuitarConfig, Homography
from tabvision.video.fretboard.geometric import GeometricFretboardBackend
from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS, marginal_string_fret
from tabvision.video.hand.mediapipe_backend import MediaPipeHandBackend

# BGR colours; viridis-ish ramp for the heat-map plus per-finger fingertip dots.
HEATMAP_LUT_BGR = [
    (84, 1, 68),       # dark purple (low)
    (139, 35, 59),
    (165, 80, 49),
    (123, 144, 33),
    (54, 200, 50),
    (39, 235, 175),
    (37, 231, 253),    # bright yellow (high)
]
COLOR_FINGERTIPS = [
    (0, 255, 255),     # index — yellow
    (0, 165, 255),     # middle — orange
    (0, 0, 255),       # ring — red
    (255, 0, 255),     # pinky — magenta
]
COLOR_HUD_BG = (0, 0, 0)
COLOR_HUD_FG = (255, 255, 255)


def render_overlay(  # noqa: PLR0913 — wraps a clear set of CLI flags
    input_path: Path,
    output_path: Path,
    *,
    yolo_checkpoint: Path | None = None,
    hand_model: Path | None = None,
    cfg: GuitarConfig | None = None,
    fallback_conf: float = 0.3,
    stride: int = 1,
    max_frames: int | None = None,
    draw_fingertips: bool = True,
    heatmap_alpha: float = 0.45,
    show_progress: bool = True,
) -> dict[str, int | float]:
    """Render a per-frame fingering overlay video."""
    import cv2

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cfg = cfg or GuitarConfig()

    fb_primary = KeypointFretboardBackend()
    fb_fallback = GeometricFretboardBackend()
    hand_backend = MediaPipeHandBackend(model_path=hand_model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open writer for {output_path}")

    last_homog: Homography | None = None
    last_fingering: FrameFingering | None = None
    last_method = ""
    frame_idx = 0
    detect_count = 0
    hand_detected = 0
    full_box = GuitarBBox(0.0, 0.0, float(width), float(height), 1.0)
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % stride == 0:
                last_homog, last_method = _run_fretboard(
                    fb_primary, fb_fallback, frame, full_box, fallback_conf
                )
                last_fingering = hand_backend.detect(frame, last_homog, cfg)
                if last_fingering.homography_confidence > 0:
                    hand_detected += 1
                detect_count += 1

            annotated = _draw(
                frame, last_homog, last_fingering, last_method, cfg,
                heatmap_alpha=heatmap_alpha,
                draw_fingertips=draw_fingertips,
            )
            writer.write(annotated)
            frame_idx += 1

            if show_progress and frame_idx % 30 == 0 and total:
                pct = 100 * frame_idx / total
                rate = frame_idx / (time.time() - t0)
                print(
                    f"[overlay_hand] {frame_idx}/{total} ({pct:.0f}%) — "
                    f"{rate:.1f} fps, {hand_detected} hand-frames",
                    file=sys.stderr,
                )
    finally:
        hand_backend.close()
        cap.release()
        writer.release()

    return {
        "frames_written": frame_idx,
        "detections_run": detect_count,
        "hand_detected_frames": hand_detected,
        "elapsed_s": int(time.time() - t0),
    }


def _run_fretboard(
    primary: KeypointFretboardBackend,
    fallback: GeometricFretboardBackend,
    frame: np.ndarray,
    bbox: GuitarBBox,
    fallback_conf: float,
) -> tuple[Homography, str]:
    """Try the keypoint backend first; fall back to geometric on low confidence."""
    primary_h = primary.detect(frame, bbox)
    if primary_h.confidence >= fallback_conf:
        return primary_h, primary.name
    fallback_h = fallback.detect(frame, bbox)
    if fallback_h.confidence > primary_h.confidence:
        return fallback_h, f"{fallback.name} (fallback)"
    return primary_h, primary.name


def _draw(  # noqa: PLR0913
    frame: np.ndarray,
    homog: Homography | None,
    fingering: FrameFingering | None,
    method_label: str,
    cfg: GuitarConfig,
    *,
    heatmap_alpha: float,
    draw_fingertips: bool,
) -> np.ndarray:
    out = frame.copy()
    if homog is None or homog.confidence == 0.0 or fingering is None:
        _draw_hud(out, "no homography", 0.0, hand_detected=False)
        return out

    if fingering.homography_confidence == 0.0:
        # Homography exists but no hand detected.
        _draw_hud(out, method_label, homog.confidence, hand_detected=False)
        _draw_quad(out, homog)
        return out

    marginal = marginal_string_fret(fingering.finger_pos_logits)  # (S, F)
    _blit_heatmap(out, marginal, homog, alpha=heatmap_alpha)
    _draw_quad(out, homog)
    if draw_fingertips:
        _draw_finger_argmax_dots(out, fingering, homog, cfg)
    _draw_hud(out, method_label, homog.confidence, hand_detected=True)
    return out


def _blit_heatmap(
    frame: np.ndarray,
    marginal: np.ndarray,
    homog: Homography,
    *,
    alpha: float,
) -> None:
    """Paint a colour-mapped heat-map of ``marginal`` over the canonical
    fretboard rectangle, alpha-blended with ``frame``."""
    import cv2

    n_strings, n_frets = marginal.shape
    # Normalise the marginal into [0, 1] by its own max so weak frames are
    # still visible (the absolute scale of the softmax is uninformative).
    peak = float(marginal.max())
    if peak <= 0:
        return
    norm = marginal / peak  # (S, F)

    # Build an (n_strings * cell, n_frets * cell) BGR image, then warp it
    # through ``homog.H`` into image space and alpha-blend.
    cell = 16  # canonical cell pixel size (small — warp is what gives final scale)
    canon_h = n_strings * cell
    canon_w = n_frets * cell
    canvas = np.zeros((canon_h, canon_w, 3), dtype=np.uint8)
    for s in range(n_strings):
        for f in range(n_frets):
            colour = _heat_colour(norm[s, f])
            canvas[s * cell : (s + 1) * cell, f * cell : (f + 1) * cell] = colour

    # Warp canonical canvas -> frame space.  The canonical canvas spans
    # x ∈ [0, canon_w] mapping to canonical x ∈ [0, 1]; same for y.  Build
    # the canvas-to-image homography by composing scale * homog.H.
    scale = np.array(
        [[1.0 / canon_w, 0, 0], [0, 1.0 / canon_h, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    M = homog.H @ scale  # noqa: N806 — math-convention name
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(canvas, M, (w, h))
    mask = (warped.sum(axis=-1) > 0)
    if not mask.any():
        return
    blended = cv2.addWeighted(frame, 1 - alpha, warped, alpha, 0)
    frame[mask] = blended[mask]


def _heat_colour(t: float) -> tuple[int, int, int]:
    """Linear interpolation through the heat-map LUT for ``t ∈ [0, 1]``."""
    t = max(0.0, min(1.0, float(t)))
    n = len(HEATMAP_LUT_BGR) - 1
    pos = t * n
    lo = int(pos)
    frac = pos - lo
    if lo >= n:
        return HEATMAP_LUT_BGR[-1]
    a = HEATMAP_LUT_BGR[lo]
    b = HEATMAP_LUT_BGR[lo + 1]
    return (
        int(a[0] + frac * (b[0] - a[0])),
        int(a[1] + frac * (b[1] - a[1])),
        int(a[2] + frac * (b[2] - a[2])),
    )


def _draw_quad(frame: np.ndarray, homog: Homography) -> None:
    import cv2

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    homo_pts = np.hstack([src, np.ones((4, 1))])
    proj = (homog.H @ homo_pts.T).T
    proj = proj[:, :2] / proj[:, 2:]
    cv2.polylines(
        frame,
        [proj.astype(np.int32)],
        isClosed=True,
        color=(255, 0, 255),
        thickness=2,
    )


def _draw_finger_argmax_dots(
    frame: np.ndarray,
    fingering: FrameFingering,
    homog: Homography,
    cfg: GuitarConfig,
) -> None:
    """Plot a dot for each finger at its argmax cell, colour-coded."""
    import cv2

    n_fingers = fingering.finger_pos_logits.shape[0]
    for fi in range(n_fingers):
        flat = fingering.finger_pos_logits[fi].reshape(-1)
        s_arg, f_arg = np.unravel_index(int(flat.argmax()), fingering.finger_pos_logits.shape[1:])
        # Convert (s_arg, f_arg) → canonical (x, y) via the same convention
        # the posterior used.
        canon_x = (f_arg + 0.5) / (cfg.max_fret + 1)
        canon_y = (cfg.n_strings - 1 - s_arg + 0.5) / cfg.n_strings
        pt = homog.H @ np.array([canon_x, canon_y, 1.0])
        px, py = int(pt[0] / pt[2]), int(pt[1] / pt[2])
        colour = COLOR_FINGERTIPS[fi % len(COLOR_FINGERTIPS)]
        cv2.circle(frame, (px, py), 6, colour, thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(
            frame, FRETTING_FINGERS[fi][0].upper(),
            (px - 4, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
        )


def _draw_hud(
    frame: np.ndarray,
    method_label: str,
    confidence: float,
    *,
    hand_detected: bool,
) -> None:
    import cv2

    parts = [f"fretboard: {method_label} ({confidence:.2f})"]
    parts.append("hand: yes" if hand_detected else "hand: no")
    text = "  ".join(parts)
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
        frame, text, (pad, h - pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HUD_FG, 1, cv2.LINE_AA,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("input", type=Path, help="input video file (mp4/mov)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output mp4 path (default: <stem>_hand.mp4 next to input)",
    )
    parser.add_argument(
        "--yolo-checkpoint",
        type=Path,
        default=None,
        help="fine-tuned YOLO-OBB .pt (defaults to ~/.tabvision/data/models/...)",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=None,
        help="MediaPipe hand_landmarker.task path (defaults to ~/.mediapipe/...)",
    )
    parser.add_argument(
        "--fallback-conf",
        type=float,
        default=0.3,
        help="confidence below which to fall back from keypoint to geometric",
    )
    parser.add_argument("--stride", type=int, default=1, help="run detection every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-fingertips",
        action="store_true",
        help="hide the per-finger argmax dots (heatmap only)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="heatmap opacity (0=transparent, 1=opaque)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if args.stride < 1:
        print("error: --stride must be ≥ 1", file=sys.stderr)
        return 2

    output = args.out or args.input.with_name(f"{args.input.stem}_hand.mp4")
    print(f"[overlay_hand] {args.input} -> {output}", file=sys.stderr)

    stats = render_overlay(
        args.input,
        output,
        yolo_checkpoint=args.yolo_checkpoint,
        hand_model=args.hand_model,
        fallback_conf=args.fallback_conf,
        stride=args.stride,
        max_frames=args.max_frames,
        draw_fingertips=not args.no_fingertips,
        heatmap_alpha=args.alpha,
    )
    print(f"[overlay_hand] done: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
