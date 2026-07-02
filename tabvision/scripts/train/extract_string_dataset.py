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

    # Map each note onset to a video frame index; crop in a single streamed pass.
    want: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
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
        crop = frame[y0:y1, x0:x1]
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
