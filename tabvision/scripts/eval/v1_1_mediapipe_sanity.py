"""v1.1 chunk-2: MediaPipe hand-detection sanity check on the Kaggle UT-Austin footage.

The open unknown for chunk-2 is whether the iPhone-tuned CV stack survives a *different rig*
(seated front-angle acoustic, varied lighting/skin/room) than the footage MediaPipe + the
fretboard detector were built against. Before wiring the full chain, this script answers the
narrowest question first: **does MediaPipe even find the fretting hand on these PNG frames, and
where do its landmarks land?**

Pure MediaPipe — no homography, no fingertip→fret, no fusion. For a sample of frames it reports
per-frame detection: hand count, the chosen fretting hand's handedness label + confidence, and
the fretting-hand wrist / fingertip positions as a fraction of the frame (so we can eyeball that
the hand sits on the neck, not the strumming hand over the soundhole).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from tabvision.types import GuitarConfig
from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS
from tabvision.video.hand.mediapipe_backend import MediaPipeHandBackend

_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)


def _clip_frame_paths(root: Path, clip_id: str) -> list[Path]:
    frames = root / "tablature_frames"
    paths = sorted(
        frames.glob(f"{clip_id}_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--clip", default="0", help="clip id (default 0)")
    ap.add_argument("--stride", type=int, default=20, help="sample every Nth frame")
    ap.add_argument("--limit", type=int, default=20, help="max frames to report")
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    backend = MediaPipeHandBackend()

    paths = _clip_frame_paths(args.root, args.clip)
    sampled = paths[:: args.stride][: args.limit]
    print(f"clip {args.clip}: {len(paths)} frames, sampling {len(sampled)} (stride {args.stride})\n")

    print(f"{'frame':>8} {'hand?':>5} {'label':>6} {'conf':>5} {'wrist_xy%':>14} {'fingertips_x%'}")
    detected = 0
    labels: Counter[str] = Counter()
    for p in sampled:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"{p.stem:>8}  <unreadable>")
            continue
        h, w = frame.shape[:2]
        hand = backend._extract_fretting_hand(frame)  # noqa: SLF001 — sanity probe
        if hand is None:
            print(f"{p.stem:>8} {'no':>5}")
            continue
        detected += 1
        label = "left" if hand.is_left_hand else "right"
        labels[label] += 1
        wx, wy = hand.wrist_xy
        tip_xs = [
            f"{hand.fingers[f].tip_xy[0] / w:.2f}" for f in FRETTING_FINGERS if f in hand.fingers
        ]
        print(
            f"{p.stem:>8} {'yes':>5} {label:>6} {hand.confidence:>5.2f}"
            f" {f'({wx / w:.2f},{wy / h:.2f})':>14}  {','.join(tip_xs)}"
        )

    n = len(sampled)
    print(
        f"\ndetected {detected}/{n} ({detected / n:.0%})"
        f"  fretting-hand labels: {dict(labels)}"
    )
    print(
        "NOTE: 'left'=HandSample.is_left_hand (MediaPipe 'Right', mirror convention)."
        " On this rig the neck/fretting hand should sit at higher wrist_x% (toward image right)."
    )
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
