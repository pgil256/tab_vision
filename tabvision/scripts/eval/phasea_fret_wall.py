"""Phase A: the fret-detection wall, measured from the rich CV cache.

The chunk-6 WS3 analysis found ~68% of ambiguous notes sit on clips where the
full-frame YOLO pass detects ~0 fret OBBs at conf 0.25 — the uniform-partition
fallback then runs and the WS1 rule-of-18 lever cannot fire. This cache-only
diagnostic reports, per clip: cached-frame fret-OBB counts (median / mean /
zero-share / >=4-wire share, 4 being ``calibrate``'s minimum), the clip's
ambiguous-note count, and the aggregate **share of ambiguous notes on
zero-median-fret clips** — the WS3 statistic — so 360p full-frame and 720p
crop-then-detect caches can be compared like for like.

Reproduce::

    cd tabvision
    python -m scripts.eval.phasea_fret_wall                       # 360p baseline
    python -m scripts.eval.phasea_fret_wall \\
        --cache-dir ~/.tabvision/cache/gaps_video_chain_720 --cache-suffix .crop
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import RawFrameCV, rawcv_cache_path
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import GuitarConfig


@dataclass
class ClipFretWall:
    stem: str
    n_frames: int  # cached frames with a usable CV record
    median_frets: float
    mean_frets: float
    zero_share: float  # share of cached frames with 0 fret OBBs
    calib_share: float  # share of cached frames with >=4 fret OBBs (calibrate's minimum)
    n_ambiguous: int


def _clip_stats(
    stem: str,
    cache_dir: Path,
    conf: float,
    suffix: str,
    data_root: Path,
    cfg: GuitarConfig,
) -> ClipFretWall | None:
    cache_path = rawcv_cache_path(cache_dir, stem, conf, suffix=suffix)
    if not cache_path.exists():
        print(f"  [skip] {stem}: no cache ({cache_path.name})")
        return None
    with open(cache_path, "rb") as fh:
        cache: dict[int, RawFrameCV | None] = pickle.load(fh)
    counts = np.array([len(rec.preds.frets) for rec in cache.values() if rec is not None])
    if counts.size == 0:
        print(f"  [skip] {stem}: cache has no usable frames")
        return None
    xml = data_root / "gaps" / "musicxml" / f"{stem}.xml"
    n_ambiguous = 0
    if xml.exists():
        gold = parse_gaps(xml)
        n_ambiguous = sum(1 for g in gold if len(candidate_positions(g.pitch_midi, cfg)) >= 2)
    return ClipFretWall(
        stem=stem,
        n_frames=int(counts.size),
        median_frets=float(np.median(counts)),
        mean_frets=float(counts.mean()),
        zero_share=float((counts == 0).mean()),
        calib_share=float((counts >= 4).mean()),
        n_ambiguous=n_ambiguous,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument(
        "--cache-dir", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO conf (cache key)")
    ap.add_argument("--cache-suffix", default="", help="rich-cache suffix (e.g. '.crop')")
    ap.add_argument("--clips", default="clean12", help="'clean12' or comma-separated stems")
    args = ap.parse_args(argv)

    clips = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )
    cfg = GuitarConfig()
    print(
        f"{'clip':>12} {'frames':>7} {'med':>5} {'mean':>6} {'zero%':>6} {'>=4%':>6} {'ambig':>6}"
    )
    rows: list[ClipFretWall] = []
    for stem in clips:
        row = _clip_stats(stem, args.cache_dir, args.conf, args.cache_suffix, args.data_root, cfg)
        if row is None:
            continue
        rows.append(row)
        print(
            f"{row.stem:>12} {row.n_frames:>7} {row.median_frets:>5.1f} {row.mean_frets:>6.2f} "
            f"{row.zero_share:>6.1%} {row.calib_share:>6.1%} {row.n_ambiguous:>6}"
        )
    if not rows:
        print("no clips measured")
        return 1

    total_ambig = sum(r.n_ambiguous for r in rows)
    wall_ambig = sum(r.n_ambiguous for r in rows if r.median_frets == 0.0)
    wall_clips = [r.stem for r in rows if r.median_frets == 0.0]
    share = wall_ambig / total_ambig if total_ambig else 0.0
    print(
        f"\nzero-median-fret clips: {len(wall_clips)}/{len(rows)}"
        + (f" ({', '.join(wall_clips)})" if wall_clips else "")
    )
    print(
        f"WS3 statistic — ambiguous notes on zero-median-fret clips: "
        f"{wall_ambig}/{total_ambig} = {share:.3f}  (chunk-6 360p baseline ~0.68)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
