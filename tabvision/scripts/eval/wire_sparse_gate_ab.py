"""Does per-clip calibration fire rate predict when calibrating hurts?

Runs the experiment pre-registered in
``docs/plans/2026-07-28-wire-sparse-calibration-gate-preregistration.md``.
Read that first — in particular §1, which explains why the *obvious* version of
this experiment is circular and must be rejected.

The lever: `calibrate_fret_xs` is net-harmful on clips where it fires rarely
(E2 report §6). A per-clip gate would fall back to the uniform partition below
some fire-rate threshold. The question is not whether that helps on the clips
where the harm was *observed* — it trivially does, by construction — but whether
fire rate **transfers**, i.e. predicts out of sample.

Primary test is therefore leave-one-clip-out: the threshold applied to each
held-out clip is fitted only on the other eleven.

Reproduce::

    cd tabvision
    python -m scripts.eval.wire_sparse_gate_ab
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.e2_fret_registration_ab import (
    DEFAULT_CV_CACHE,
    DEFAULT_KPT_CACHE,
    DEFAULT_VIDEO_CACHE,
    _run_clip,
)
from tabvision.types import GuitarConfig

# Pre-registered search grid and tie-break (preregistration §5).
T_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95
PROCEED_GAIN = 0.010


@dataclass
class ClipRow:
    stem: str
    n: int  # ambiguous notes with CV evidence
    fire: float  # per-clip calibration fire rate
    correct_cal: int
    correct_uni: int

    @property
    def a_cal(self) -> float:
        return self.correct_cal / self.n if self.n else 0.0

    @property
    def a_uni(self) -> float:
        return self.correct_uni / self.n if self.n else 0.0

    @property
    def d(self) -> float:
        """Per-clip benefit of calibrating (positive = calibration helps)."""
        return self.a_cal - self.a_uni


def gated_correct(row: ClipRow, t: float) -> int:
    """Correct count for this clip under a gate at threshold ``t``."""
    return row.correct_uni if row.fire < t else row.correct_cal


def best_threshold(rows: list[ClipRow]) -> float:
    """T maximising pooled gated accuracy over ``rows``.

    Ties break toward the LARGER T (preregistration §5) — more gating, i.e. the
    more conservative fallback to the uniform partition.
    """
    best_t = T_GRID[0]
    best_correct = -1
    for t in T_GRID:
        total = sum(gated_correct(r, t) for r in rows)
        if total >= best_correct:  # >= walks the tie up to the largest T
            best_correct = total
            best_t = t
    return best_t


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation; no scipy, since CI installs only ``.[dev]``."""
    if len(x) < 3:
        return float("nan")

    def rank(v: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(v, dtype=float), kind="stable")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        return r

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    ap.add_argument("--cv-cache", type=Path, default=DEFAULT_CV_CACHE)
    ap.add_argument("--kpt-cache", type=Path, default=DEFAULT_KPT_CACHE)
    ap.add_argument("--clips", default="clean12")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--det-conf", type=float, default=0.10)
    ap.add_argument("--cache-suffix", default=".crop")
    ap.add_argument("--window-s", type=float, default=0.06)
    ap.add_argument("--vote-frames", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )

    rows: list[ClipRow] = []
    print(f"wire-sparse calibration gate — {len(stems)} clips\n")
    for stem in stems:
        res = _run_clip(
            stem,
            data_root=args.data_root,
            video_cache=args.video_cache,
            cv_cache=args.cv_cache,
            kpt_cache=args.kpt_cache,
            cfg=cfg,
            conf=args.conf,
            det_conf=args.det_conf,
            cache_suffix=args.cache_suffix,
            window_s=args.window_s,
            max_frames=args.vote_frames,
            # Geometry arms only — this experiment must not depend on E2's model.
            arms=("uniform", "obb"),
        )
        if res is None:
            continue
        obb, uni = res["obb"], res["uniform"]
        rows.append(
            ClipRow(
                stem=stem,
                n=obb.diag.have_cv,
                fire=obb.fire / obb.usable if obb.usable else 0.0,
                correct_cal=obb.diag.correct,
                correct_uni=uni.diag.correct,
            )
        )

    if len(rows) < 3:
        print("too few clips ran to test anything")
        return 2

    print(f"{'clip':<12}{'n':>6}{'fire':>8}{'a_uni':>8}{'a_cal':>8}{'d':>8}")
    for r in sorted(rows, key=lambda r: r.fire):
        print(f"{r.stem:<12}{r.n:6d}{r.fire:8.3f}{r.a_uni:8.3f}{r.a_cal:8.3f}{r.d:+8.3f}")

    total_n = sum(r.n for r in rows)
    ungated = sum(r.correct_cal for r in rows)
    uniform_all = sum(r.correct_uni for r in rows)

    # --- primary: leave-one-clip-out -------------------------------------
    loo_correct = 0
    picks: list[tuple[str, float, bool]] = []
    for i, r in enumerate(rows):
        others = rows[:i] + rows[i + 1 :]
        t = best_threshold(others)
        gated = r.fire < t
        loo_correct += gated_correct(r, t)
        picks.append((r.stem, t, gated))

    loo_acc = loo_correct / total_n
    ungated_acc = ungated / total_n
    gain = loo_acc - ungated_acc

    print(f"\n{'=' * 66}")
    print("LOO threshold chosen per held-out clip (fitted on the other 11):")
    for stem, t, gated in picks:
        print(f"  {stem:<12} T={t:.2f}  -> {'GATED (uniform)' if gated else 'calibrated'}")

    print(f"\nPOOLED over {len(rows)} clips, {total_n} ambiguous notes")
    print(f"  uniform everywhere      {uniform_all / total_n:.4f}  ({uniform_all}/{total_n})")
    print(f"  ungated (current)       {ungated_acc:.4f}  ({ungated}/{total_n})")
    print(f"  LOO-gated               {loo_acc:.4f}  ({loo_correct}/{total_n})")
    print(f"  gain vs ungated         {gain:+.4f}")

    # --- mechanism (descriptive only) ------------------------------------
    rho = spearman([r.fire for r in rows], [r.d for r in rows])
    lo = sorted(rows, key=lambda r: r.fire)[: len(rows) // 2]
    hi = sorted(rows, key=lambda r: r.fire)[len(rows) // 2 :]

    def wmean(rs: list[ClipRow]) -> float:
        n = sum(r.n for r in rs)
        return sum(r.d * r.n for r in rs) / n if n else float("nan")

    print("\nMECHANISM (descriptive — explains an effect, does not establish one)")
    print(f"  Spearman(fire, d)       {rho:+.3f}")
    print(f"  note-weighted mean d, low-fire half   {wmean(lo):+.4f}")
    print(f"  note-weighted mean d, high-fire half  {wmean(hi):+.4f}")

    # --- pre-registered verdict ------------------------------------------
    if gain >= PROCEED_GAIN:
        verdict = f"PASS — proceed to the source-disjoint-10 confirmation (gain >= {PROCEED_GAIN})"
    elif gain > 0:
        verdict = "DIRECTIONAL — positive but < 0.010; bank, do not spend on confirmation"
    else:
        verdict = "FAIL — fire rate does not transfer out of sample; bank the negative"
    print(f"\nPRE-REGISTERED VERDICT: {verdict}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
