"""v1.1 chunk-2 deliverable: the REAL CV chain end-to-end on the Kaggle rig.

Replaces the chunk-1 *oracle* ``FrameFingering`` with one produced by the actual vision stack:

    PNG frame -> YOLO-OBB neck -> KeypointFretboardBackend homography
              -> MediaPipe hands -> (geometric fretting-hand selection) -> fingertip_to_fret
              -> FrameFingering

then fuses it with gold-pitch ``AudioEvent``s (string/fret stripped, exactly like the oracle
probe) so the measured Tab F1 isolates the *string* axis — the v1.1 lever. Compares three
conditions per clip: audio-only, +real-video, +oracle (the chunk-1 ceiling).

Hand-selection fix (chunk-2 Finding 2): the Kaggle footage is NOT mirrored, so v0's "pick the
MediaPipe-'Right' hand" grabs the strumming hand. Here we instead project each detected hand's
fingertips through the homography and keep the hand whose tips land *on* the canonical
fretboard ``[0,1]x[0,1]`` — rig-agnostic, and only possible now that we have a real homography.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from scripts.eval.v1_1_kaggle_oracle_probe import (
    _events_from_gold,
    _load_timestamps,
    _oracle_fingerings,
    parse_clip,
)
from tabvision.eval.metrics import tab_f1
from tabvision.fusion import fuse
from tabvision.types import FrameFingering, GuitarConfig
from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
from tabvision.video.guitar.yolo_backend import YoloOBBBackend
from tabvision.video.hand.fingertip_to_fret import (
    FRETTING_FINGERS,
    HandSample,
    compute_fingering,
)
from tabvision.video.hand.mediapipe_backend import (
    _FINGER_LANDMARKS,
    _build_hand_sample,
)

_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)


def _select_fretting_hand_geometric(hands: list[HandSample], H_inv: np.ndarray) -> HandSample | None:
    """Pick the hand whose fingertips project nearest the canonical fretboard center.

    The strumming hand (over the soundhole) projects far outside [0,1]^2; the fretting hand
    lands on the board. Score = mean distance of fingertips from the [0,1]^2 box (0 = inside).
    """
    best: HandSample | None = None
    best_score = float("inf")
    for hand in hands:
        cs = []
        for f in FRETTING_FINGERS:
            s = hand.fingers.get(f)
            if s is None:
                continue
            pt = H_inv @ np.array([s.tip_xy[0], s.tip_xy[1], 1.0])
            if abs(pt[2]) < 1e-9:
                continue
            cx, cy = pt[0] / pt[2], pt[1] / pt[2]
            # distance outside the unit box (0 if inside)
            dx = max(0.0, -cx, cx - 1.0)
            dy = max(0.0, -cy, cy - 1.0)
            cs.append((dx * dx + dy * dy) ** 0.5)
        if cs:
            score = float(np.mean(cs))
            if score < best_score:
                best_score, best = score, hand
    return best


def _real_fingerings_for_clip(
    clip_id: str,
    root: Path,
    onsets: list[float],
    cfg: GuitarConfig,
    yolo: YoloOBBBackend,
    fb: KeypointFretboardBackend,
    landmarker,  # noqa: ANN001 — mediapipe HandLandmarker
) -> list[FrameFingering]:
    """One real FrameFingering per gold onset (nearest available frame)."""
    import mediapipe as mp

    frames_dir = root / "tablature_frames"
    ts = _load_timestamps(root)
    # map onset time -> nearest frame index for this clip
    frame_times = {
        int(k.split("_")[1].split(".")[0]): v
        for k, v in ts.items()
        if k.split("_")[0] == clip_id
    }
    out: list[FrameFingering] = []
    empty = np.zeros((len(FRETTING_FINGERS), cfg.n_strings, cfg.max_fret + 1))
    for onset in onsets:
        idx = min(frame_times, key=lambda i: abs(frame_times[i] - onset)) if frame_times else None
        p = frames_dir / f"{clip_id}_{idx}.png" if idx is not None else None
        if p is None or not p.exists():
            out.append(FrameFingering(t=onset, finger_pos_logits=empty.copy(), homography_confidence=0.0))
            continue
        frame = cv2.imread(str(p))
        H = fb.detect(frame, None)
        if H.confidence <= 0.0:
            out.append(FrameFingering(t=onset, finger_pos_logits=empty.copy(), homography_confidence=0.0))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            out.append(FrameFingering(t=onset, finger_pos_logits=empty.copy(), homography_confidence=0.0))
            continue
        h, w = frame.shape[:2]
        hands = [
            _build_hand_sample(lm, hd, frame_width=w, frame_height=h)
            for lm, hd in zip(res.hand_landmarks, res.handedness)
        ]
        H_inv = np.linalg.inv(H.H)
        hand = _select_fretting_hand_geometric(hands, H_inv)
        if hand is None:
            out.append(FrameFingering(t=onset, finger_pos_logits=empty.copy(), homography_confidence=0.0))
            continue
        ff = compute_fingering(hand, H, cfg)
        logits = ff.finger_pos_logits
        # Orientation correction for this rig: keypoint.py's nut/body + high-E/low-E
        # heuristics assume iPhone lap framing; the Kaggle rig is inverted on both axes
        # (nut at high-X, strings reversed). See v1_1_chunk2_cv_chain report.
        if FLIP_STRING:
            logits = logits[:, ::-1, :]
        if FLIP_FRET:
            logits = logits[:, :, ::-1]
        out.append(FrameFingering(t=onset, finger_pos_logits=logits, homography_confidence=ff.homography_confidence))
    return out


FLIP_FRET = False
FLIP_STRING = False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--clips", default=None, help="comma clip ids; default all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--flip-fret", action="store_true", help="invert canonical fret axis (nut/body)")
    ap.add_argument("--flip-string", action="store_true", help="invert canonical string axis (E/e)")
    args = ap.parse_args(argv)

    global FLIP_FRET, FLIP_STRING
    FLIP_FRET = args.flip_fret
    FLIP_STRING = args.flip_string

    cfg = GuitarConfig()
    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    fb = KeypointFretboardBackend(backend=yolo)

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = os.path.expanduser("~/.mediapipe/models/hand_landmarker.task")
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model), num_hands=2
        )
    )

    ts = _load_timestamps(args.root)
    label_dir = args.root / "tablature_labels"
    clip_ids = (
        args.clips.split(",")
        if args.clips
        else sorted((p.stem for p in label_dir.glob("*.npy")), key=int)
    )

    print(f"{'clip':>5} {'notes':>6} {'audio':>8} {'+real':>8} {'+oracle':>8} {'real-d':>8}")
    rows = []
    for cid in clip_ids:
        gold = parse_clip(cid, args.root, ts, cfg)
        if not gold:
            continue
        ev = _events_from_gold(gold)
        onsets = [g.onset_s for g in gold]
        fa = tab_f1(fuse(ev, [], cfg), gold).f1
        real = _real_fingerings_for_clip(cid, args.root, onsets, cfg, yolo, fb, landmarker)
        fr = tab_f1(fuse(ev, real, cfg), gold).f1
        fo = tab_f1(fuse(ev, _oracle_fingerings(gold, cfg), cfg), gold).f1
        rows.append((cid, len(gold), fa, fr, fo))
        print(f"{cid:>5} {len(gold):>6} {fa:>8.4f} {fr:>8.4f} {fo:>8.4f} {fr - fa:>+8.4f}")

    if rows:
        n = len(rows)
        ma = sum(r[2] for r in rows) / n
        mr = sum(r[3] for r in rows) / n
        mo = sum(r[4] for r in rows) / n
        print(f"{'ALL':>5} {sum(r[1] for r in rows):>6} {ma:>8.4f} {mr:>8.4f} {mo:>8.4f} {mr - ma:>+8.4f}  ({n} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
