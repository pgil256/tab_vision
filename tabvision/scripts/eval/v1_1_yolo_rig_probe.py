"""v1.1 chunk-2: does the trained YOLO-OBB fretboard detector generalize to the Kaggle rig?

The keypoint homography backend needs a per-frame ``neck`` OBB (plus nut/fret bonuses). The
detector is fine-tuned on the Roboflow ``b101/guitar-3`` set — a *different* rig than the Kaggle
UT-Austin seated front-angle footage. This probe runs the trained ``YoloOBBBackend`` +
``KeypointFretboardBackend`` on a sample of Kaggle frames and reports whether the neck is found,
its confidence, and — crucially — whether the resulting homography quad localizes the actual
(diagonal, narrow) neck rather than collapsing to a full-frame box (the failure mode the v0
geometric detector showed on this rig).

Point it at a checkpoint with --checkpoint (or TABVISION_GUITAR_YOLO_CHECKPOINT).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
from tabvision.video.guitar.yolo_backend import YoloOBBBackend

_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)


def _quad_extent(homography: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """Project the canonical unit square through homography and return its bbox extent
    as a fraction of (frame_w, frame_h). A real neck is a narrow diagonal strip
    (small in at least one axis); a full-frame mis-fire is ~1.0 x ~1.0."""
    corners = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64).T
    proj = homography @ corners
    proj = proj[:2] / proj[2]
    dx = (proj[0].max() - proj[0].min()) / w
    dy = (proj[1].max() - proj[1].min()) / h
    return float(dx), float(dy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None, help="YOLO-OBB .pt (or set env)")
    ap.add_argument("--clips", default="0,1,5,10,15,20,24", help="comma clip ids")
    ap.add_argument("--per-clip", type=int, default=3, help="frames per clip")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    fb = KeypointFretboardBackend(backend=yolo)
    frames_dir = args.root / "tablature_frames"

    print(f"checkpoint: {yolo.checkpoint_path}")
    print(
        f"{'frame':>8} {'neck':>5} {'conf':>5} {'frets':>5} {'nut':>4} "
        f"{'H.conf':>6} {'quad_w%':>7} {'quad_h%':>7}"
    )
    neck_hits = 0
    good_quad = 0
    total = 0
    for cid in args.clips.split(","):
        paths = sorted(
            frames_dir.glob(f"{cid}_*.png"), key=lambda p: int(p.stem.split("_")[1])
        )
        for p in paths[:: max(1, len(paths) // args.per_clip)][: args.per_clip]:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            total += 1
            h, w = frame.shape[:2]
            preds = yolo.predict_all(frame)
            neck = preds.best_neck()
            homography = fb.detect(frame, None)  # GuitarBBox unused by keypoint backend
            qw, qh = (
                _quad_extent(homography.H, w, h)
                if homography.confidence > 0
                else (0.0, 0.0)
            )
            if neck is not None:
                neck_hits += 1
            # "good" = localized strip: covers <85% of at least one axis and H is confident
            if homography.confidence > 0 and (qw < 0.85 or qh < 0.85):
                good_quad += 1
            print(
                f"{p.stem:>8} {('Y' if neck else 'n'):>5} "
                f"{(neck.confidence if neck else 0):>5.2f} {len(preds.frets):>5} "
                f"{('Y' if preds.best_nut() else 'n'):>4} {homography.confidence:>6.2f} "
                f"{qw:>7.2f} {qh:>7.2f}"
            )

    print(
        f"\nneck detected {neck_hits}/{total} ({neck_hits / max(1, total):.0%}); "
        f"localized (non-full-frame) homography {good_quad}/{total} "
        f"({good_quad / max(1, total):.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
