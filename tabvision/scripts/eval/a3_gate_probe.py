"""A3 gate probe — accept/reject a single fusion-constant override on one manifest.

Confirms a sweep candidate against the measurement discipline: per-tier
lower-95 CI (GuitarSet 60-clip player-05) and per-clip no-regression (GAPS
clean-12). The override is fusion-only, so raw AudioEvents are transcribed once
(cache-shared with the sweep) and re-fused for baseline vs override — the two
scorings differ only in the rebound module global.

Reproduce (the two gate legs for `OPEN_STRING_BONUS=0.0`)::

    cd tabvision   # env from tabvision-eval-env
    # leg 1 — GuitarSet 60-clip player-05 (accepted config), per-tier lo-95:
    python -m scripts.eval.a3_gate_probe --manifest data/eval/local_guitarset.toml \
        --splits validation --backend highres --position-prior guitarset-v1 \
        --set OPEN_STRING_BONUS=0.0 --output ../docs/EVAL_REPORTS/a3_gate_open0_gs60_2026-07-06.md
    # leg 2 — GAPS clean-12 (accepted config), per-clip no-regression:
    python -m scripts.eval.a3_gate_probe --manifest data/eval/gaps.toml --splits test \
        --backend highres --position-prior none --clean12 \
        --set OPEN_STRING_BONUS=0.0 --output ../docs/EVAL_REPORTS/a3_gate_open0_gaps12_2026-07-06.md
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.a3_fusion_sweep import ClipData, _raw_events_cached, make_shared_audio_backend
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.composite import _resolve_path, _session_from_clip
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers import get_parser
from tabvision.fusion import chord_shapes, playability
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.viterbi import fuse
from tabvision.types import GuitarConfig

_CACHE_DIR = Path.home() / ".tabvision/cache/a3_fusion_sweep"  # shared with the sweep


def _const_module(name: str) -> object:
    """The fusion module that defines constant ``name``.

    Emission/transition weights live in ``playability``; the A5 chord-shape
    bonus in ``chord_shapes``. Lets ``--set`` gate any sweepable axis, not just
    ``playability`` ones.
    """
    for mod in (playability, chord_shapes):
        if hasattr(mod, name):
            return mod
    raise SystemExit(f"unknown fusion constant: {name!r}")


def gate_passed(lower95_held: bool, n_regressions: int, strict_per_clip: bool) -> bool:
    """Decide the gate verdict from the two measurement bars.

    - ``lower95_held``: no tier's lower-95 CI regressed. This bar ALWAYS gates.
    - per-clip no-regression (``n_regressions == 0``): the hard bar only for the
      GAPS clean-12 cross-domain leg (``strict_per_clip=True``). On the in-domain
      GuitarSet 60-clip confirm it is informational — a few clips regressing
      while the aggregate lower-95 holds is expected and not a FAIL.
    """
    if not lower95_held:
        return False
    if strict_per_clip and n_regressions > 0:
        return False
    return True


def _score(
    clips: list[ClipData], prior: object, cfg: GuitarConfig
) -> dict[str, list[tuple[str, float]]]:
    """Per-tier list of (clip_id, tab_f1) under the current global config."""
    by_tier: dict[str, list[tuple[str, float]]] = {}
    for clip in clips:
        events = clip.raw_events
        if prior is not None:
            events = apply_pitch_position_prior(events, prior)  # type: ignore[arg-type]
        f1 = tab_f1(fuse(events, [], cfg), clip.gold).f1
        by_tier.setdefault(clip.tier, []).append((clip.clip_id, f1))
    return by_tier


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--backend", default="highres")
    ap.add_argument("--position-prior", default="guitarset-v1")
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--set", dest="override", required=True, help="e.g. OPEN_STRING_BONUS=0.0")
    ap.add_argument(
        "--clean12", action="store_true", help="filter GAPS clips to the CLEAN_12 stems"
    )
    ap.add_argument(
        "--strict-per-clip",
        action="store_true",
        help=(
            "make per-clip no-regression a HARD gate (the GAPS clean-12 "
            "cross-domain bar). Omit for the in-domain GuitarSet 60-clip "
            "confirm, whose bar is per-tier lower-95 — there a few individual "
            "clips regressing while the aggregate lower-95 holds is expected "
            "and reported informationally, not a FAIL."
        ),
    )
    ap.add_argument("--media-root", type=Path, default=None)
    ap.add_argument("--annotation-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    name, _, raw_val = args.override.partition("=")
    name, override_val = name.strip(), float(raw_val)
    mod = _const_module(name)
    baseline_val = getattr(mod, name)

    prior_name: str | None = args.position_prior
    if prior_name and prior_name.lower() == "none":
        prior_name = None

    cfg = GuitarConfig()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    payload = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    prior = load_pitch_position_prior(prior_name, cfg=cfg) if prior_name else None

    print("transcribing / loading raw audio events...")
    shared_backend = make_shared_audio_backend(args.backend)
    clips: list[ClipData] = []
    for clip in payload.get("clips") or []:
        if clip["split"] not in splits:
            continue
        if args.clean12 and not any(clip["id"].endswith(stem) for stem in CLEAN_12):
            continue
        media = _resolve_path(clip["media_path"], args.media_root)
        annotation = _resolve_path(clip["annotation_path"], args.annotation_root)
        gold = get_parser(clip["annotation_format"])(annotation, cfg)
        raw = _raw_events_cached(
            media, _session_from_clip(clip), args.backend, _CACHE_DIR, shared_backend
        )
        clips.append(ClipData(clip["id"], clip["tier"], raw, gold))
    print(f"{len(clips)} clips ready\n")

    setattr(mod, name, baseline_val)
    base = _score(clips, prior, cfg)
    setattr(mod, name, override_val)
    over = _score(clips, prior, cfg)
    setattr(mod, name, baseline_val)  # restore

    lines = [f"# A3 gate — `{name}` {baseline_val} → {override_val} ({args.manifest.name})", ""]
    lines.append(
        f"Config: `{args.backend}` + `{prior_name or 'none'}`, splits `{args.splits}`"
        f"{', CLEAN_12 subset' if args.clean12 else ''}. Fusion-only A/B (raw events shared)."
    )
    lines.append("")

    # Per-tier lower-95 (GuitarSet acceptance protocol).
    lines.append("## Per-tier Tab F1 (mean / lower-95, bootstrap N=10k)")
    lines.append("")
    lines.append(
        "| tier | clips | baseline mean (lo-95) | override mean (lo-95) | Δ mean | Δ lo-95 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    verdict_ok = True
    for tier in sorted(base):
        b = [f for _, f in base[tier]]
        o = [f for _, f in over[tier]]
        bc = bootstrap_ci(b, n_bootstrap=10_000, seed=42)
        oc = bootstrap_ci(o, n_bootstrap=10_000, seed=42)
        dmean = oc.statistic - bc.statistic
        dlo = oc.lower - bc.lower
        lines.append(
            f"| {tier} | {len(b)} | {bc.statistic:.4f} ({bc.lower:.4f}) | "
            f"{oc.statistic:.4f} ({oc.lower:.4f}) | {dmean:+.4f} | {dlo:+.4f} |"
        )
        # The acceptance bar: NO tier may regress its lower-95 CI.
        if dlo < -1e-4:
            verdict_ok = False

    # Per-clip no-regression (GAPS discipline).
    lines.append("")
    lines.append("## Per-clip deltas (no-regression check)")
    lines.append("")
    all_deltas = []
    regressions = []
    for tier in sorted(base):
        bd = dict(base[tier])
        for cid, of1 in over[tier]:
            d = of1 - bd[cid]
            all_deltas.append(d)
            if d < -1e-4:
                regressions.append((cid, d))
    improved = sum(1 for d in all_deltas if d > 1e-4)
    lines.append(
        f"- {len(all_deltas)} clips: {improved} improved, {len(regressions)} regressed, "
        f"{len(all_deltas) - improved - len(regressions)} unchanged."
    )
    if regressions:
        regressions.sort(key=lambda x: x[1])
        worst = ", ".join(f"{c} ({d:+.3f})" for c, d in regressions[:8])
        lines.append(f"- Worst regressions: {worst}")
    lines.append("")

    per_clip_ok = len(regressions) == 0
    lines.append("## Verdict")
    lines.append("")
    passed = gate_passed(verdict_ok, len(regressions), args.strict_per_clip)
    lo95_msg = "all tiers' lower-95 held" if verdict_ok else "a tier's lower-95 REGRESSED"
    if args.strict_per_clip:
        per_clip_msg = (
            "no per-clip regressions"
            if per_clip_ok
            else f"{len(regressions)} per-clip regression(s) [HARD gate]"
        )
    else:
        per_clip_msg = (
            f"{len(regressions)} per-clip regression(s) [informational — "
            "lower-95 is the bar for the in-domain confirm]"
        )
    bar = "per-clip no-regression + lower-95" if args.strict_per_clip else "per-tier lower-95"
    lines.append(f"**{'PASS' if passed else 'FAIL'}** — {lo95_msg}; {per_clip_msg}. (bar: {bar})")
    lines.append("")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
