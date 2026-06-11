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
from dataclasses import replace
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
from tabvision.fusion.vision_evidence import (
    ORIENTATION_BY_NAME,
    Orientation,
    choose_orientation,
    combine_fingerings,
    empty_fingering,
    gate_fingering_to_audio,
    orient_fingering,
)
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig, SessionConfig, TabEvent
from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
from tabvision.video.guitar.yolo_backend import YoloOBBBackend
from tabvision.video.hand.fingertip_to_fret import (
    FRETTING_FINGERS,
    HandSample,
    compute_fingering,
)
from tabvision.video.hand.mediapipe_backend import _build_hand_sample

_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)


def _select_fretting_hand_geometric(
    hands: list[HandSample], h_inv: np.ndarray
) -> HandSample | None:
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
            pt = h_inv @ np.array([s.tip_xy[0], s.tip_xy[1], 1.0])
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


def _frame_indices_near_onset(
    frame_times: dict[int, float],
    onset_s: float,
    *,
    max_frames: int,
    window_s: float,
) -> list[int]:
    """Nearest frame indices around an onset, sorted by timestamp."""

    if not frame_times:
        return []
    ranked = sorted(frame_times, key=lambda i: abs(frame_times[i] - onset_s))
    in_window = [i for i in ranked if abs(frame_times[i] - onset_s) <= window_s]
    chosen = in_window[:max_frames] if in_window else ranked[:1]
    return sorted(chosen, key=lambda i: frame_times[i])


def _raw_fingering_for_frame(
    frame_path: Path,
    cfg: GuitarConfig,
    fb: KeypointFretboardBackend,
    landmarker,  # noqa: ANN001 - mediapipe HandLandmarker
) -> FrameFingering | None:
    """Run YOLO homography + MediaPipe hand detection on one stored frame."""

    import mediapipe as mp

    frame = cv2.imread(str(frame_path))
    if frame is None:
        return None
    homography = fb.detect(frame, None)
    if homography.confidence <= 0.0:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    hands = [
        _build_hand_sample(lm, hd, frame_width=w, frame_height=h)
        for lm, hd in zip(res.hand_landmarks, res.handedness, strict=False)
    ]
    h_inv = np.linalg.inv(homography.H)
    hand = _select_fretting_hand_geometric(hands, h_inv)
    if hand is None:
        return None
    return compute_fingering(hand, homography, cfg)


def _robust_real_fingerings_for_clip(
    clip_id: str,
    root: Path,
    audio_events: list[AudioEvent],
    cfg: GuitarConfig,
    fb: KeypointFretboardBackend,
    landmarker,  # noqa: ANN001 - mediapipe HandLandmarker
    *,
    orientation: Orientation | None,
    gate: bool,
    vote_frames: int,
    vote_window_s: float,
    min_homography_confidence: float,
    min_candidate_support: float,
    min_best_ratio: float,
    min_clip_coverage: float,
) -> tuple[list[FrameFingering], Orientation, dict[str, float], int]:
    """One robust ``FrameFingering`` per audio event."""

    frames_dir = root / "tablature_frames"
    ts = _load_timestamps(root)
    frame_times = {
        int(k.split("_")[1].split(".")[0]): v for k, v in ts.items() if k.split("_")[0] == clip_id
    }

    raw_by_event: list[list[FrameFingering]] = []
    for event in audio_events:
        raw: list[FrameFingering] = []
        for idx in _frame_indices_near_onset(
            frame_times,
            event.onset_s,
            max_frames=vote_frames,
            window_s=vote_window_s,
        ):
            p = frames_dir / f"{clip_id}_{idx}.png"
            if not p.exists():
                continue
            ff = _raw_fingering_for_frame(p, cfg, fb, landmarker)
            if ff is not None:
                raw.append(replace(ff, t=frame_times[idx]))
        raw_by_event.append(raw)

    chosen_orientation = orientation
    orientation_scores: dict[str, float] = {}
    if chosen_orientation is None:
        chosen_orientation, orientation_scores = choose_orientation(raw_by_event, audio_events, cfg)

    out: list[FrameFingering] = []
    kept = 0
    for event, raw in zip(audio_events, raw_by_event, strict=False):
        if not raw:
            out.append(empty_fingering(event.onset_s, cfg))
            continue
        oriented = [orient_fingering(f, chosen_orientation) for f in raw]
        voted = combine_fingerings(oriented, cfg, t=event.onset_s)
        if gate:
            voted = gate_fingering_to_audio(
                event,
                voted,
                cfg,
                min_homography_confidence=min_homography_confidence,
                min_candidate_support=min_candidate_support,
                min_best_ratio=min_best_ratio,
            )
        if voted.homography_confidence > 0.0 and (voted.finger_pos_logits != 0).any():
            kept += 1
        out.append(voted)
    coverage = kept / len(audio_events) if audio_events else 0.0
    if gate and coverage < min_clip_coverage:
        out = [empty_fingering(event.onset_s, cfg) for event in audio_events]
        kept = 0
    return out, chosen_orientation, orientation_scores, kept


def _load_highres_audio_events(
    clip_id: str,
    root: Path,
    backend,  # noqa: ANN001 - HighResBackend, imported lazily
) -> list[AudioEvent]:
    """Run highres on the UT-Austin WAV for a clip."""

    import soundfile as sf

    wav_path = root / "tablature_audio" / f"{clip_id}.wav"
    wav, sr = sf.read(str(wav_path), always_2d=False)
    wav_arr = np.asarray(wav, dtype=np.float32)
    if wav_arr.ndim == 2:
        wav_arr = wav_arr.mean(axis=1)
    return list(backend.transcribe(wav_arr, int(sr), SessionConfig()))


def _estimate_audio_alignment(
    events: list[AudioEvent],
    gold: list[TabEvent],
    *,
    tolerance_s: float = 0.12,
    max_abs_shift: int = 3,
) -> tuple[int, float, dict[tuple[int, float], int]]:
    """Estimate whole-semitone and time-origin offsets from near matches."""

    if not events or not gold:
        return 0, 0.0, {(0, 0.0): 0}

    scores: dict[tuple[int, float], int] = {}
    for shift in range(-max_abs_shift, max_abs_shift + 1):
        for offset_step in range(-150, 151):
            time_shift = offset_step / 50.0
            matches = 0
            for event in events:
                aligned_onset = event.onset_s + time_shift
                nearest = min(gold, key=lambda g: abs(g.onset_s - aligned_onset))
                if abs(nearest.onset_s - aligned_onset) <= tolerance_s:
                    matches += int(nearest.pitch_midi == event.pitch_midi + shift)
            scores[(shift, time_shift)] = matches
    best_shift, best_time_shift = max(
        scores,
        key=lambda s: (scores[s], -abs(s[0]), -abs(s[1])),
    )
    return best_shift, best_time_shift, scores


def _shift_audio_events(
    events: list[AudioEvent],
    pitch_shift: int,
    time_shift_s: float = 0.0,
) -> list[AudioEvent]:
    if pitch_shift == 0 and time_shift_s == 0.0:
        return events
    return [
        replace(
            event,
            onset_s=event.onset_s + time_shift_s,
            offset_s=event.offset_s + time_shift_s,
            pitch_midi=event.pitch_midi + pitch_shift,
        )
        for event in events
    ]


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
        int(k.split("_")[1].split(".")[0]): v for k, v in ts.items() if k.split("_")[0] == clip_id
    }
    out: list[FrameFingering] = []
    empty = np.zeros((len(FRETTING_FINGERS), cfg.n_strings, cfg.max_fret + 1))
    for onset in onsets:
        idx = min(frame_times, key=lambda i: abs(frame_times[i] - onset)) if frame_times else None
        p = frames_dir / f"{clip_id}_{idx}.png" if idx is not None else None
        if p is None or not p.exists():
            out.append(
                FrameFingering(
                    t=onset,
                    finger_pos_logits=empty.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        frame = cv2.imread(str(p))
        homography = fb.detect(frame, None)
        if homography.confidence <= 0.0:
            out.append(
                FrameFingering(
                    t=onset,
                    finger_pos_logits=empty.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.hand_landmarks:
            out.append(
                FrameFingering(
                    t=onset,
                    finger_pos_logits=empty.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        h, w = frame.shape[:2]
        hands = [
            _build_hand_sample(lm, hd, frame_width=w, frame_height=h)
            for lm, hd in zip(res.hand_landmarks, res.handedness, strict=False)
        ]
        h_inv = np.linalg.inv(homography.H)
        hand = _select_fretting_hand_geometric(hands, h_inv)
        if hand is None:
            out.append(
                FrameFingering(
                    t=onset,
                    finger_pos_logits=empty.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        ff = compute_fingering(hand, homography, cfg)
        logits = ff.finger_pos_logits
        # Orientation correction for this rig: keypoint.py's nut/body + high-E/low-E
        # heuristics assume iPhone lap framing; the Kaggle rig is inverted on both axes
        # (nut at high-X, strings reversed). See v1_1_chunk2_cv_chain report.
        if FLIP_STRING:
            logits = logits[:, ::-1, :]
        if FLIP_FRET:
            logits = logits[:, :, ::-1]
        out.append(
            FrameFingering(
                t=onset,
                finger_pos_logits=logits,
                homography_confidence=ff.homography_confidence,
            )
        )
    return out


FLIP_FRET = False
FLIP_STRING = False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--clips", default=None, help="comma clip ids; default all")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--audio-source", choices=["gold", "highres"], default="gold")
    ap.add_argument(
        "--pitch-shift",
        type=int,
        default=None,
        help="manual semitone shift for highres events",
    )
    ap.add_argument(
        "--time-shift-s",
        type=float,
        default=None,
        help="manual seconds added to highres event onsets/offsets",
    )
    ap.add_argument("--orientation", choices=["auto", *ORIENTATION_BY_NAME.keys()], default="auto")
    ap.add_argument(
        "--no-gate",
        action="store_true",
        help="disable audio-compatible confidence gating",
    )
    ap.add_argument("--vote-frames", type=int, default=5)
    ap.add_argument("--vote-window-s", type=float, default=0.08)
    ap.add_argument("--min-homography-conf", type=float, default=0.1)
    ap.add_argument("--min-candidate-support", type=float, default=0.02)
    ap.add_argument("--min-best-ratio", type=float, default=1.2)
    ap.add_argument("--min-clip-coverage", type=float, default=0.71)
    ap.add_argument(
        "--flip-fret",
        action="store_true",
        help="invert canonical fret axis (nut/body)",
    )
    ap.add_argument("--flip-string", action="store_true", help="invert canonical string axis (E/e)")
    args = ap.parse_args(argv)

    global FLIP_FRET, FLIP_STRING
    FLIP_FRET = args.flip_fret
    FLIP_STRING = args.flip_string
    orientation: Orientation | None
    if args.flip_fret or args.flip_string:
        if args.flip_fret and args.flip_string:
            orientation = ORIENTATION_BY_NAME["flip-both"]
        elif args.flip_fret:
            orientation = ORIENTATION_BY_NAME["flip-fret"]
        else:
            orientation = ORIENTATION_BY_NAME["flip-string"]
    elif args.orientation == "auto":
        orientation = None
    else:
        orientation = ORIENTATION_BY_NAME[args.orientation]

    cfg = GuitarConfig()
    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    fb = KeypointFretboardBackend(backend=yolo)
    highres_backend = None
    if args.audio_source == "highres":
        from tabvision.audio.highres import HighResBackend

        highres_backend = HighResBackend()

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

    print(
        f"{'clip':>5} {'notes':>6} {'audio':>8} {'+real':>8} {'+oracle':>8} "
        f"{'real-d':>8} {'orient':>11} {'kept':>7} {'p-sh':>5} {'t-sh':>6}"
    )
    rows = []
    for cid in clip_ids:
        gold = parse_clip(cid, args.root, ts, cfg)
        if not gold:
            continue
        if args.audio_source == "highres":
            assert highres_backend is not None
            ev = _load_highres_audio_events(cid, args.root, highres_backend)
            auto_pitch_shift, auto_time_shift, _alignment_scores = _estimate_audio_alignment(
                ev, gold
            )
            pitch_shift = args.pitch_shift if args.pitch_shift is not None else auto_pitch_shift
            time_shift = args.time_shift_s if args.time_shift_s is not None else auto_time_shift
            ev = _shift_audio_events(ev, pitch_shift, time_shift)
        else:
            ev = _events_from_gold(gold)
            pitch_shift = 0
            time_shift = 0.0
        fa = tab_f1(fuse(ev, [], cfg), gold).f1
        real, chosen_orientation, _orientation_scores, kept = _robust_real_fingerings_for_clip(
            cid,
            args.root,
            ev,
            cfg,
            fb,
            landmarker,
            orientation=orientation,
            gate=not args.no_gate,
            vote_frames=args.vote_frames,
            vote_window_s=args.vote_window_s,
            min_homography_confidence=args.min_homography_conf,
            min_candidate_support=args.min_candidate_support,
            min_best_ratio=args.min_best_ratio,
            min_clip_coverage=args.min_clip_coverage,
        )
        fr = tab_f1(fuse(ev, real, cfg), gold).f1
        fo = tab_f1(fuse(ev, _oracle_fingerings(gold, cfg), cfg), gold).f1
        rows.append((cid, len(gold), fa, fr, fo, kept, pitch_shift, time_shift))
        print(
            f"{cid:>5} {len(gold):>6} {fa:>8.4f} {fr:>8.4f} {fo:>8.4f} "
            f"{fr - fa:>+8.4f} {chosen_orientation.name:>11} {kept:>7} "
            f"{pitch_shift:>+5} {time_shift:>+6.2f}"
        )

    if rows:
        n = len(rows)
        ma = sum(r[2] for r in rows) / n
        mr = sum(r[3] for r in rows) / n
        mo = sum(r[4] for r in rows) / n
        kept_total = sum(r[5] for r in rows)
        print(
            f"{'ALL':>5} {sum(r[1] for r in rows):>6} {ma:>8.4f} {mr:>8.4f} "
            f"{mo:>8.4f} {mr - ma:>+8.4f} {'':>11} {kept_total:>7} "
            f"{'':>5} {'':>6}  ({n} clips)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
