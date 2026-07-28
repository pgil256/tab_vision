"""Phase A: emit the 360p-vs-720p-crop comparison tables as markdown.

Reuses the two cache-only diagnostics rather than re-deriving their math:
:mod:`scripts.eval.v1_1_gaps_string_diag` for the WS1 leading indicator (the A5
decision variable) and :mod:`scripts.eval.phasea_fret_wall` for the detection
statistics. Both arms are read from their own rich caches, so this is seconds,
not a CV re-run.

Reproduce::

    cd tabvision
    python -m scripts.eval.phasea_report > /tmp/phasea_tables.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.gaps_cv_cache import make_fret_xs_calibrator
from scripts.eval.phasea_fret_wall import ClipFretWall, _clip_stats
from scripts.eval.v1_1_gaps_string_diag import _diagnose_clip
from tabvision.types import GuitarConfig


@dataclass
class Arm:
    label: str
    video_cache: Path
    cache_dir: Path
    suffix: str


@dataclass
class ClipRow:
    stem: str
    wall: ClipFretWall | None
    have_cv: int
    uniform_correct: int
    ws1_correct: int

    @property
    def uniform(self) -> float:
        return self.uniform_correct / self.have_cv if self.have_cv else 0.0

    @property
    def ws1(self) -> float:
        return self.ws1_correct / self.have_cv if self.have_cv else 0.0


def _measure(arm: Arm, stems: tuple[str, ...], data_root: Path, cfg: GuitarConfig) -> list[ClipRow]:
    calibrator = make_fret_xs_calibrator(cfg)
    rows: list[ClipRow] = []
    for stem in stems:
        uniform = _diagnose_clip(
            stem,
            data_root,
            arm.video_cache,
            arm.cache_dir,
            cfg,
            conf=0.25,
            window_s=0.06,
            max_frames=1,
            calibrate=None,
            cache_suffix=arm.suffix,
        )
        if uniform is None:
            continue
        ws1 = _diagnose_clip(
            stem,
            data_root,
            arm.video_cache,
            arm.cache_dir,
            cfg,
            conf=0.25,
            window_s=0.06,
            max_frames=1,
            calibrate=calibrator,
            cache_suffix=arm.suffix,
        )
        rows.append(
            ClipRow(
                stem=stem,
                wall=_clip_stats(stem, arm.cache_dir, 0.25, arm.suffix, data_root, cfg),
                have_cv=uniform.have_cv,
                uniform_correct=uniform.correct,
                ws1_correct=ws1.correct if ws1 else 0,
            )
        )
    return rows


def _agg(rows: list[ClipRow]) -> tuple[int, int, int]:
    return (
        sum(r.uniform_correct for r in rows),
        sum(r.ws1_correct for r in rows),
        sum(r.have_cv for r in rows),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision" / "data")
    ap.add_argument("--cache-root", type=Path, default=Path.home() / ".tabvision" / "cache")
    ap.add_argument("--clips", default="clean12")
    args = ap.parse_args(argv)

    stems = (
        CLEAN_12
        if args.clips == "clean12"
        else tuple(s.strip() for s in args.clips.split(",") if s.strip())
    )
    cfg = GuitarConfig()
    base = Arm("360p", args.cache_root / "gaps_video", args.cache_root / "gaps_video_chain", "")
    crop = Arm(
        "720p-crop",
        args.cache_root / "gaps_video_720",
        args.cache_root / "gaps_video_chain_720",
        ".crop",
    )
    a_rows = {r.stem: r for r in _measure(base, stems, args.data_root, cfg)}
    b_rows = {r.stem: r for r in _measure(crop, stems, args.data_root, cfg)}
    both = [s for s in stems if s in a_rows and s in b_rows]
    if not both:
        print("no clip measured in both arms yet")
        return 1

    print("### Leading indicator — ambiguous-note string accuracy (best orientation)\n")
    print("| clip | haveCV | 360p uniform | 360p WS1 | 720p-crop uniform | 720p-crop WS1 | Δ WS1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for stem in both:
        a, b = a_rows[stem], b_rows[stem]
        print(
            f"| {stem} | {a.have_cv} | {a.uniform:.3f} | {a.ws1:.3f} | "
            f"{b.uniform:.3f} | **{b.ws1:.3f}** | {b.ws1 - a.ws1:+.3f} |"
        )
    au, aw, an = _agg([a_rows[s] for s in both])
    bu, bw, bn = _agg([b_rows[s] for s in both])
    print(
        f"| **AGG** | {an}/{bn} | **{au / an:.3f}** | **{aw / an:.3f}** | "
        f"**{bu / bn:.3f}** | **{bw / bn:.3f}** | **{bw / bn - aw / an:+.3f}** |"
    )
    print(
        f"\n({au}/{an}, {aw}/{an} vs {bu}/{bn}, {bw}/{bn}; "
        f"banked context: uniform 0.544 / WS1 0.574; audio prior 0.778)\n"
    )

    print("### Fret-detection wall\n")
    print(
        "| clip | 360p med frets | 720p-crop med frets | 360p ≥4-wire frames | "
        "720p-crop ≥4-wire frames | 360p Hconf | 720p-crop Hconf |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for stem in both:
        aw_, bw_ = a_rows[stem].wall, b_rows[stem].wall
        if aw_ is None or bw_ is None:
            continue
        print(
            f"| {stem} | {aw_.median_frets:.1f} | {bw_.median_frets:.1f} | "
            f"{aw_.calib_share:.1%} | {bw_.calib_share:.1%} | "
            f"{aw_.mean_hconf:.3f} | {bw_.mean_hconf:.3f} |"
        )
    for label, rows in (("360p", a_rows), ("720p-crop", b_rows)):
        walls = [rows[s].wall for s in both if rows[s].wall is not None]
        ambig = sum(w.n_ambiguous for w in walls)
        wall_ambig = sum(w.n_ambiguous for w in walls if w.median_frets == 0.0)
        zero_clips = [w.stem for w in walls if w.median_frets == 0.0]
        print(
            f"\n**{label} WS3 statistic** — ambiguous notes on zero-median-fret clips: "
            f"{wall_ambig}/{ambig} = {wall_ambig / ambig if ambig else 0:.3f}"
            + (f" (clips: {', '.join(zero_clips)})" if zero_clips else " (none)")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
