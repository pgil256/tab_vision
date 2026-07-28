"""Why is 118_VD1wc pathological? (2026-07-28)

Three independent measurements single this clip out: Phase A's largest per-clip
regression (-0.150), the only clip where fret calibration *hurts*
(-0.129 -- and uncalibrated it is the best clip in clean-12 at 0.895), and the
clip where the E2 keypoint model detects almost nothing (~0.02 instances/frame).

That combination is the clue: the hand->string mapping is excellent under a
*uniform* fret partition and gets worse when a fitted map is applied, so the
fitted map is not merely weak but systematically wrong. This dumps the geometry
per frame so the failure mode is identifiable rather than guessed at.

Reproduce::

    cd tabvision
    python -m scripts.eval.diag_118_pathology
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import rawcv_cache_path
from tabvision.types import GuitarConfig
from tabvision.video.fretboard.calibrate import (
    calibrate_fret_xs,
    nut_at_high_canonical_x,
    project_to_canonical,
)

DEFAULT_CV_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain_720"


def uniform_fret_xs(cfg: GuitarConfig) -> np.ndarray:
    """The pre-WS1 uniform partition's cell centres, for comparison."""
    n = cfg.max_fret + 1
    edges = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-cache", type=Path, default=DEFAULT_CV_CACHE)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cache-suffix", default=".crop")
    ap.add_argument("--stems", default="118_VD1wc", help="comma-separated, or 'clean12'")
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    stems = CLEAN_12 if args.stems == "clean12" else tuple(args.stems.split(","))
    uni = uniform_fret_xs(cfg)

    print(f"{'clip':<12}{'frames':>7}{'hconf':>8}{'nfret':>7}{'nut?':>6}{'nuthigh':>9}{'fire':>7}")
    per_clip = {}
    for stem in stems:
        path = rawcv_cache_path(args.cv_cache, stem, args.conf, suffix=args.cache_suffix)
        if not path.exists():
            print(f"{stem:<12} no cache")
            continue
        with open(path, "rb") as fh:
            raw = pickle.load(fh)
        hconf, nfret, has_nut, nut_high, fired = [], [], [], [], []
        maps = []
        for rec in raw.values():
            if rec is None:
                continue
            hconf.append(rec.homography.confidence)
            nfret.append(len(rec.preds.frets))
            has_nut.append(rec.preds.best_nut() is not None)
            if rec.preds.frets:
                wc = np.array([[d.cx, d.cy] for d in rec.preds.frets], dtype=np.float64)
                wx = project_to_canonical(rec.homography, wc)[:, 0]
                wx = np.sort(wx[(wx > -0.15) & (wx < 1.25)])
                if wx.size >= 2:
                    nut_high.append(nut_at_high_canonical_x(wx))
            xs = calibrate_fret_xs(rec.preds, rec.homography, cfg)
            fired.append(xs is not None)
            if xs is not None:
                maps.append(xs)
        n = len(hconf)
        print(
            f"{stem:<12}{n:7d}{np.mean(hconf):8.3f}{np.mean(nfret):7.2f}"
            f"{np.mean(has_nut):6.2f}{(np.mean(nut_high) if nut_high else float('nan')):9.2f}"
            f"{np.mean(fired):7.3f}"
        )
        per_clip[stem] = maps

    for stem, maps in per_clip.items():
        if not maps:
            print(f"\n{stem}: calibration never fired — nothing to compare")
            continue
        med = np.median(np.stack(maps), axis=0)
        print(f"\n{stem}: median fitted fret-cell centres vs the uniform partition")
        print(f"  {'fret':>5}{'fitted':>9}{'uniform':>9}{'delta':>9}")
        for k in range(0, min(len(med), len(uni)), 2):
            print(f"  {k:5d}{med[k]:9.4f}{uni[k]:9.4f}{med[k] - uni[k]:+9.4f}")
        d = np.diff(med)
        print(
            f"  monotonic: {bool(np.all(d > 0) or np.all(d < 0))}  "
            f"span: {med.min():.3f}..{med.max():.3f}  "
            f"first-gap/last-gap: {abs(d[0]) / abs(d[-1]):.2f} (rule-of-18 expects > 1)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
