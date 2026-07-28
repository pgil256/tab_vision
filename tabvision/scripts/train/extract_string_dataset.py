"""Extract a string-resolution training set from GAPS train clips — v1.1 WS4.

Builds the supervised dataset for the learned string-resolution model
(``docs/plans/2026-06-25-v1.1-ws4-learned-string-model-design.md``). Per GAPS
``train`` clip with a downloaded video:

  1. parse gold tab (``string_idx`` / ``fret`` / ``pitch_midi`` per note),
  2. recover the audio<->video crop offset (xcorr, cached),
  3. detect a per-clip **neck crop rectangle** — the camera is static within a
     clip, so the median YOLO ``neck`` box over a handful of sampled frames
     serves every note (turns ~140K YOLO calls into ~clips×samples),
  4. for each gold note grab the onset frame, crop the neck region, resize, and
     save a JPEG + a manifest row.

Output (under ``--out-dir``): ``crops/<stem>/<note>.jpg`` + ``manifest.jsonl``
(one row per note). Incremental + resumable: clips already in the manifest are
skipped. Eval/training-only NC data — never committed or redistributed.

Usage::

    cd tabvision
    export TABVISION_DATA_ROOT=~/.tabvision/data
    export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
    python -m scripts.train.extract_string_dataset --clips train \
        --checkpoint ~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import estimate_offset, read_split_stems
from tabvision.demux import _frame_iterator, _probe_metadata
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.video.fretboard.keypoint import _obb_to_corners

DEFAULT_VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video"
DEFAULT_OUT = Path.home() / ".tabvision" / "cache" / "gaps_string_dataset"


def neck_crop_rect(
    corner_sets: list[np.ndarray],
    frame_w: int,
    frame_h: int,
    *,
    pad_frac: float = 0.35,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned crop rect around the median neck box, padded + clamped.

    Args:
        corner_sets: list of ``(4, 2)`` neck-OBB corner arrays (image px), one
            per sampled frame. The median per-corner position is used so a few
            misdetections don't move the crop.
        frame_w, frame_h: frame size for clamping.
        pad_frac: fraction of the box's width/height to pad on each side — the
            fretting hand wraps above/around the neck, so the crop must include
            a margin beyond the bare fretboard box.

    Returns:
        ``(x0, y0, x1, y1)`` integer pixel rect, or ``None`` if no boxes.
    """
    if not corner_sets:
        return None
    stacked = np.stack(corner_sets, axis=0)  # (N, 4, 2)
    med = np.median(stacked, axis=0)  # (4, 2)
    x0, y0 = med[:, 0].min(), med[:, 1].min()
    x1, y1 = med[:, 0].max(), med[:, 1].max()
    pad_x = (x1 - x0) * pad_frac
    pad_y = (y1 - y0) * pad_frac
    x0 = int(max(0, np.floor(x0 - pad_x)))
    y0 = int(max(0, np.floor(y0 - pad_y)))
    x1 = int(min(frame_w, np.ceil(x1 + pad_x)))
    y1 = int(min(frame_h, np.ceil(y1 + pad_y)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def _sample_neck_rect(
    video_path: Path,
    yolo,  # noqa: ANN001 - YoloOBBBackend
    *,
    n_samples: int,
    pad_frac: float,
) -> tuple[int, int, int, int] | None:
    """Median neck crop rect from ``n_samples`` frames via fast cv2 seeks."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            return None
        idxs = np.linspace(0, max(0, total - 1), num=min(n_samples, total), dtype=int)
        corner_sets: list[np.ndarray] = []
        fw = fh = 0
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fh, fw = frame.shape[:2]
            neck = yolo.predict_all(frame).best_neck()
            if neck is not None:
                corner_sets.append(_obb_to_corners(neck))
        if not corner_sets:
            return None
        return neck_crop_rect(corner_sets, fw, fh, pad_frac=pad_frac)
    finally:
        cap.release()


def _offset_s(stem: str, wav: Path, vid: Path, cache_dir: Path) -> tuple[float, float]:
    """(offset_s, peak_ratio), cached as ``{stem}.offset.pkl`` (shared w/ the probe)."""
    cache = cache_dir / f"{stem}.offset.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            res = pickle.load(fh)
        return float(res.offset_s), float(res.peak_ratio)
    res = estimate_offset(wav, vid)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(res, fh)
    return float(res.offset_s), float(res.peak_ratio)


def sustain_frame_index(
    onset_s: float,
    next_onset_s: float | None,
    offset_s: float,
    fps: float,
    *,
    lead_s: float = 0.080,
    max_s: float = 0.400,
    tail_guard_s: float = 0.040,
) -> int:
    """Frame index inside a note's *sustain*, not on its onset (Phase D).

    The WS4 negative was diagnosed as whole-neck crops starving the model, but a
    second contributor is label noise at the onset frame: the fretting hand is
    still arriving, so the frame labelled with a note's (string, fret) often does
    not yet show that shape. Sampling from
    ``[onset + lead_s, min(onset + max_s, next_onset - tail_guard_s)]`` picks a
    frame where the shape is held.

    The window is clamped so it never crosses into the next note, and never
    lands before the onset itself when notes are very close together.
    """
    start = onset_s + lead_s
    end = onset_s + max_s
    if next_onset_s is not None:
        end = min(end, next_onset_s - tail_guard_s)
    if end < start:
        # Notes closer than lead+guard: fall back toward the onset rather than
        # borrowing a frame that belongs to the following note.
        target = max(onset_s, min(start, next_onset_s or start) - tail_guard_s)
    else:
        target = 0.5 * (start + end)
    return int(round((target + offset_s) * fps))


def hand_tight_rect(
    hand,  # noqa: ANN001 - HandSample
    frame_w: int,
    frame_h: int,
    *,
    pad_mult: float = 1.6,
    min_extent_px: int = 160,
) -> tuple[int, int, int, int] | None:
    """Square crop around the fretting hand's landmark span (Phase D).

    The banked WS4 root cause was that a whole-neck crop squished into 224x224
    leaves the hand only a few pixels tall — "the whole-neck crop starves the
    model". This centres a square crop on the hand instead, padded by
    ``pad_mult`` and floored at ``min_extent_px`` so a distant hand still yields
    enough context to resolve which string is being pressed.
    """
    coords = [hand.wrist_xy, *(f.tip_xy for f in hand.fingers.values())]
    pts = np.asarray(coords, dtype=np.float64)
    if pts.size == 0 or not np.isfinite(pts).all():
        return None
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    half = max(extent * pad_mult, float(min_extent_px)) / 2.0
    x0 = int(max(0.0, np.floor(cx - half)))
    y0 = int(max(0.0, np.floor(cy - half)))
    x1 = int(min(float(frame_w), np.ceil(cx + half)))
    y1 = int(min(float(frame_h), np.ceil(cy + half)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


def _hand_rect_for_frame(
    frame,  # noqa: ANN001 - BGR ndarray
    landmarker,  # noqa: ANN001 - mediapipe HandLandmarker
    yolo,  # noqa: ANN001 - YoloOBBBackend, for neck-relative fretting-hand choice
) -> tuple[int, int, int, int] | None:
    """Locate the fretting hand in one frame and return its square crop rect."""
    if landmarker is None:
        return None
    import cv2
    import mediapipe as mp

    from scripts.eval.v1_1_real_chain_probe import _select_fretting_hand_geometric
    from tabvision.video.fretboard.keypoint import predictions_to_homography
    from tabvision.video.hand.mediapipe_backend import _build_hand_sample

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    h, w = frame.shape[:2]
    hands = [
        _build_hand_sample(lm, hd, frame_width=w, frame_height=h)
        for lm, hd in zip(res.hand_landmarks, res.handedness, strict=False)
    ]
    if len(hands) == 1:
        hand = hands[0]
    else:
        homography = predictions_to_homography(yolo.predict_all(frame))
        if homography.confidence <= 0.0:
            return None
        hand = _select_fretting_hand_geometric(hands, np.linalg.inv(homography.H))
    if hand is None:
        return None
    return hand_tight_rect(hand, w, h)


def extract_clip(
    stem: str,
    data_root: Path,
    video_cache: Path,
    out_dir: Path,
    yolo,  # noqa: ANN001
    *,
    crop_size: int,
    n_samples: int,
    pad_frac: float,
    offset_cache: Path,
    sustain: bool = False,
    hand_tight: bool = False,
    landmarker=None,  # noqa: ANN001 - mediapipe HandLandmarker, required by hand_tight
) -> list[dict]:
    """Extract all gold-note crops for one clip; returns manifest rows."""
    import cv2

    gaps = data_root / "gaps"
    xml = gaps / "musicxml" / f"{stem}.xml"
    wav = gaps / "audio" / f"{stem}.wav"
    vid = video_cache / f"{stem}.mp4"
    if not (xml.exists() and wav.exists() and vid.exists()):
        return []
    gold = parse_gaps(xml)
    if not gold:
        return []

    offset, peak_ratio = _offset_s(stem, wav, vid, offset_cache)
    _dur, fps = _probe_metadata(vid)
    rect = _sample_neck_rect(vid, yolo, n_samples=n_samples, pad_frac=pad_frac)
    if rect is None:
        return []
    x0, y0, x1, y1 = rect

    # Map each note to a video frame index; crop in a single streamed pass.
    want: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
        if sustain:
            nxt = gold[i + 1].onset_s if i + 1 < len(gold) else None
            fi = sustain_frame_index(g.onset_s, nxt, offset, fps)
        else:
            fi = int(round((g.onset_s + offset) * fps))
        if fi >= 0:
            want.setdefault(fi, []).append(i)
    if not want:
        return []
    max_fi = max(want)

    clip_dir = out_dir / "crops" / stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for fi, (_t, frame) in enumerate(_frame_iterator(vid, fps)):
        if fi > max_fi:
            break
        if fi not in want:
            continue
        cx0, cy0, cx1, cy1 = x0, y0, x1, y1
        if hand_tight:
            # Per-frame hand crop; fall back to the clip neck rect when the hand
            # is missing so a dropout costs one crop's quality, not the note.
            hand_rect = _hand_rect_for_frame(frame, landmarker, yolo)
            if hand_rect is not None:
                cx0, cy0, cx1, cy1 = hand_rect
        crop = frame[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        for i in want[fi]:
            g = gold[i]
            rel = f"crops/{stem}/{i:05d}.jpg"
            cv2.imwrite(str(out_dir / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append(
                {
                    "stem": stem,
                    "note": i,
                    "jpg": rel,
                    "string_idx": int(g.string_idx),
                    "fret": int(g.fret),
                    "pitch_midi": int(g.pitch_midi),
                    "onset_s": float(g.onset_s),
                    "offset_s": float(offset),
                    "peak_ratio": float(peak_ratio),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--clips", default="train", help="'train'/'test'/'all' or comma-separated stems"
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="YOLO-OBB checkpoint")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--neck-samples", type=int, default=20)
    ap.add_argument(
        "--hand-tight",
        action="store_true",
        help="Phase D: crop around the fretting hand per frame instead of the "
        "clip-wide neck rect (the banked WS4 root cause was that the whole-neck "
        "crop starves the model). Requires MediaPipe.",
    )
    ap.add_argument(
        "--sustain",
        action="store_true",
        help="Phase D: sample a frame inside the note's sustain rather than at "
        "its onset, where the fretting hand is still arriving.",
    )
    ap.add_argument("--pad-frac", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    csv_path = args.data_root / "gaps" / "gaps_metadata_with_splits.csv"
    if args.clips in ("train", "test", "all"):
        stems = read_split_stems(csv_path, args.clips)
    else:
        stems = tuple(s.strip() for s in args.clips.split(",") if s.strip())
    if args.limit is not None:
        stems = stems[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    offset_cache = Path.home() / ".tabvision" / "cache" / "gaps_video_chain"
    manifest_path = args.out_dir / "manifest.jsonl"
    done = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["stem"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"clips={len(stems)} already-extracted={len(done)} out={args.out_dir}", flush=True)

    from tabvision.video.guitar.yolo_backend import YoloOBBBackend

    ckpt = args.checkpoint or os.environ.get("TABVISION_GUITAR_YOLO_CHECKPOINT")
    yolo = YoloOBBBackend(checkpoint_path=ckpt, conf=args.conf, device="cpu")
    landmarker = None
    if args.hand_tight:
        from scripts.eval.v1_1_gaps_video_chain_probe import _build_landmarker

        landmarker = _build_landmarker()

    total_rows = 0
    with open(manifest_path, "a", encoding="utf-8") as out:
        for i, stem in enumerate(stems):
            if stem in done:
                continue
            try:
                rows = extract_clip(
                    stem,
                    args.data_root,
                    args.video_cache,
                    args.out_dir,
                    yolo,
                    crop_size=args.crop_size,
                    n_samples=args.neck_samples,
                    pad_frac=args.pad_frac,
                    offset_cache=offset_cache,
                    sustain=args.sustain,
                    hand_tight=args.hand_tight,
                    landmarker=landmarker,
                )
            except Exception as exc:  # noqa: BLE001 — keep the long batch alive
                print(
                    f"[FAIL {i + 1}/{len(stems)}] {stem}: {type(exc).__name__}: {exc}", flush=True
                )
                continue
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            total_rows += len(rows)
            print(
                f"[ok {i + 1}/{len(stems)}] {stem}: {len(rows)} crops (cum {total_rows})",
                flush=True,
            )

    print(f"\nextraction done: {total_rows} new crops -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
