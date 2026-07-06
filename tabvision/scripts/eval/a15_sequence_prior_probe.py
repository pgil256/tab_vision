"""A15 step 1 — GuitarSet fingering-sequence prior probe (oracle audio).

Measures whether *sequence* statistics (anchor-to-anchor transition
n-grams learned from GuitarSet train players) carry string-resolution
signal beyond the shipped unigram pitch-position prior, in-domain, with
zero new data.

The probe decodes **gold pitches** (synthetic ``AudioEvent``s built from
the validation gold with confidence 1.0) through the real cluster
Viterbi, so the only thing measured is the fusion decode's position
choice — audio errors are out of the loop. That makes the run fast
(seconds per config, no transcription) and the comparison clean:

- ``handcoded``     — transition terms as shipped, no unigram prior.
- ``unigram``       — the accepted ``guitarset-v1`` config (baseline).
- ``seq-only``      — learned transition prior alone (no unigram).
- ``unigram+seq``   — unigram emission + learned transition, weight-swept.

Usage (from the ``tabvision/`` package dir, venv active)::

    python scripts/eval/a15_sequence_prior_probe.py \
        --manifest data/eval/local_gs_val24.toml \
        --data-home C:/Users/patri/.tabvision/data/guitarset \
        --output ../docs/EVAL_REPORTS/a15_guitarset_sequence_probe_2026-07-02.md
"""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tabvision.eval.composite import _resolve_path
from tabvision.eval.guitarset_audio import list_guitarset_track_ids, parse_guitarset_jams
from tabvision.eval.metrics import TabF1Result, chord_instance_accuracy, tab_f1
from tabvision.eval.parsers import get_parser
from tabvision.fusion import fuse, playability
from tabvision.fusion.position_prior import (
    apply_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.transition_prior import (
    TransitionPrior,
    extract_transitions,
    learn_transition_prior,
    load_transition_prior,
)
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent


@dataclass(frozen=True)
class Clip:
    clip_id: str
    tier: str
    annotation_path: Path
    annotation_format: str


@dataclass(frozen=True)
class ConfigResult:
    label: str
    per_tier: dict[str, TabF1Result]
    overall: TabF1Result
    chord_acc_strummed: float


def load_manifest_clips(manifest_path: Path, splits: tuple[str, ...]) -> list[Clip]:
    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    clips = []
    for row in payload.get("clips", []):
        if splits and str(row.get("split", "")) not in splits:
            continue
        clips.append(
            Clip(
                clip_id=str(row["id"]),
                tier=str(row["tier"]),
                annotation_path=_resolve_path(str(row["annotation_path"]), None),
                annotation_format=str(row["annotation_format"]),
            )
        )
    if not clips:
        raise SystemExit(f"no clips for splits {splits} in {manifest_path}")
    return clips


def synthetic_audio_events(gold: list[TabEvent]) -> list[AudioEvent]:
    return [
        AudioEvent(
            onset_s=g.onset_s,
            offset_s=g.onset_s + g.duration_s,
            pitch_midi=g.pitch_midi,
            velocity=0.8,
            confidence=1.0,
        )
        for g in gold
    ]


def micro_tab_f1(results: list[TabF1Result]) -> TabF1Result:
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def run_config(
    label: str,
    clips: list[Clip],
    golds: dict[str, list[TabEvent]],
    *,
    unigram_enabled: bool,
    seq_prior: TransitionPrior | None,
    seq_weight: float,
    unigram_prior,
    cfg: GuitarConfig,
) -> ConfigResult:
    playability.set_transition_prior(seq_prior, seq_weight)
    try:
        per_tier_results: dict[str, list[TabF1Result]] = {}
        all_results: list[TabF1Result] = []
        chord_matched = 0
        chord_total = 0
        for clip in clips:
            gold = golds[clip.clip_id]
            events = synthetic_audio_events(gold)
            if unigram_enabled:
                events = apply_pitch_position_prior(events, unigram_prior)
            decoded = fuse(events, [], cfg, SessionConfig(), lambda_vision=0.0)
            score = tab_f1(decoded, gold)
            per_tier_results.setdefault(clip.tier, []).append(score)
            all_results.append(score)
            if clip.tier == "clean_acoustic_strummed":
                chord = chord_instance_accuracy(decoded, gold)
                chord_matched += chord.matched_chords
                chord_total += chord.total_chords
    finally:
        playability.set_transition_prior(None)

    return ConfigResult(
        label=label,
        per_tier={tier: micro_tab_f1(rs) for tier, rs in per_tier_results.items()},
        overall=micro_tab_f1(all_results),
        chord_acc_strummed=(chord_matched / chord_total) if chord_total else 0.0,
    )


def transition_stats_markdown(train_tracks: list[list[TabEvent]]) -> list[str]:
    """Descriptive stats: how predictable is Δstring given Δpitch?"""
    from collections import Counter

    joint: Counter[tuple[int, int]] = Counter()
    for track in train_tracks:
        for dp, ds, _prev_fret in extract_transitions(track):
            joint[(dp, ds)] += 1
    total = sum(joint.values())
    if not total:
        return ["*(no transitions extracted)*"]

    marg_ds: Counter[int] = Counter()
    marg_dp: Counter[int] = Counter()
    for (dp, ds), n in joint.items():
        marg_ds[ds] += n
        marg_dp[dp] += n

    def entropy(counts: dict[int, int], denom: int) -> float:
        return -sum((n / denom) * math.log2(n / denom) for n in counts.values() if n)

    h_ds = entropy(marg_ds, total)
    h_cond = 0.0
    for dp, n_dp in marg_dp.items():
        row = {ds: n for (d, ds), n in joint.items() if d == dp}
        h_cond += (n_dp / total) * entropy(row, n_dp)

    same_string = marg_ds.get(0, 0) / total
    lines = [
        f"- train transitions: **{total:,}** (cluster-anchor to anchor)",
        f"- marginal H(Δstring) = **{h_ds:.3f} bits**; "
        f"conditional H(Δstring | Δpitch) = **{h_cond:.3f} bits** "
        f"(information gain **{h_ds - h_cond:.3f} bits**)",
        f"- same-string transitions overall: **{same_string:.1%}**",
        "",
        "| Δpitch | n | P(same string) | argmax Δstring |",
        "|---|---|---|---|",
    ]
    for dp in sorted(marg_dp, key=lambda d: -marg_dp[d])[:10]:
        n_dp = marg_dp[dp]
        row = {ds: n for (d, ds), n in joint.items() if d == dp}
        top_ds = max(row, key=lambda k: row[k])
        lines.append(f"| {dp:+d} | {n_dp:,} | {row.get(0, 0) / n_dp:.2f} | {top_ds:+d} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--data-home",
        type=Path,
        default=None,
        help="GuitarSet root (train JAMS); required unless --train-source manifest",
    )
    ap.add_argument(
        "--train-source",
        choices=("guitarset", "manifest"),
        default="guitarset",
        help="learn transition stats from GuitarSet train players (default) or from"
        " the eval manifest's own train split (in-domain check)",
    )
    ap.add_argument(
        "--seq-without-unigram",
        action="store_true",
        help="also sweep seq-prior configs without the unigram prior (for corpora"
        " whose accepted config is --position-prior none, e.g. GAPS)",
    )
    ap.add_argument(
        "--artifact-priors",
        default="",
        help="comma-separated named/path transition-prior artifacts to sweep"
        " (e.g. guitarset-seq-v1,pdmx-seq-v1) — loaded via load_transition_prior",
    )
    ap.add_argument(
        "--skip-learned",
        action="store_true",
        help="skip the learned-from-train-tracks sweep (artifact comparison only;"
        " no train parsing, no transition-stats section)",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--backoff-kappa", type=float, default=8.0)
    ap.add_argument(
        "--weights",
        default="0.5,1.0,2.0,4.0",
        help="comma-separated sequence-prior weights to sweep",
    )
    ap.add_argument("--splits", default="validation,test")
    args = ap.parse_args(argv)

    cfg = GuitarConfig()
    weights = [float(w) for w in args.weights.split(",") if w.strip()]

    artifact_names = [n.strip() for n in args.artifact_priors.split(",") if n.strip()]
    if args.skip_learned and not artifact_names:
        raise SystemExit("--skip-learned requires --artifact-priors")

    if args.skip_learned:
        train_tracks = []
        train_label = "*(learned sweep skipped — artifact comparison only)*"
    elif args.train_source == "manifest":
        train_clips = load_manifest_clips(args.manifest, ("train",))
        print(f"parsing {len(train_clips)} manifest train clips…", flush=True)
        train_tracks = []
        skipped = 0
        for c in train_clips:
            try:
                train_tracks.append(list(get_parser(c.annotation_format)(c.annotation_path, cfg)))
            except (KeyError, ValueError) as exc:  # malformed gold: skip for count stats
                skipped += 1
                print(f"  [skip] {c.clip_id}: {exc!r}", flush=True)
        train_label = (
            f"`{args.manifest.name}` train split "
            f"({len(train_tracks)} clips parsed, {skipped} skipped malformed)"
        )
    else:
        if args.data_home is None:
            raise SystemExit("--data-home is required with --train-source guitarset")
        train_ids = list_guitarset_track_ids(args.data_home, split="train")
        if not train_ids:
            raise SystemExit(f"no GuitarSet train tracks under {args.data_home}")
        print(f"parsing {len(train_ids)} train tracks…", flush=True)
        train_tracks = [
            parse_guitarset_jams(args.data_home / "annotation" / f"{tid}.jams") for tid in train_ids
        ]
        train_label = f"GuitarSet players != 05 ({len(train_ids)} tracks)"

    priors = (
        {}
        if args.skip_learned
        else {
            (scheme, singles): learn_transition_prior(
                train_tracks,
                scheme=scheme,
                alpha=args.alpha,
                backoff_kappa=args.backoff_kappa,
                singleton_only=singles,
            )
            for scheme in ("delta", "delta_fret")
            for singles in (False, True)
        }
    )
    unigram = load_pitch_position_prior("guitarset-v1", cfg=cfg)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    clips = load_manifest_clips(args.manifest, splits)
    print(f"scoring {len(clips)} clips (splits: {', '.join(splits)})…", flush=True)
    golds = {
        c.clip_id: list(get_parser(c.annotation_format)(c.annotation_path, cfg)) for c in clips
    }

    configs: list[ConfigResult] = []

    def run(label: str, *, unigram_enabled: bool, seq: TransitionPrior | None, w: float = 1.0):
        result = run_config(
            label,
            clips,
            golds,
            unigram_enabled=unigram_enabled,
            seq_prior=seq,
            seq_weight=w,
            unigram_prior=unigram,
            cfg=cfg,
        )
        configs.append(result)
        print(f"  {label}: overall {result.overall.f1:.4f}", flush=True)

    run("handcoded (no unigram, no seq)", unigram_enabled=False, seq=None)
    run("unigram guitarset-v1 (baseline)", unigram_enabled=True, seq=None)
    for (scheme, singles), prior in priors.items():
        stats_label = "stats-singles" if singles else "stats-all"
        for w in weights:
            run(
                f"unigram + seq {scheme} {stats_label} w={w}",
                unigram_enabled=True,
                seq=prior,
                w=w,
            )
        if args.seq_without_unigram:
            for w in weights:
                run(
                    f"handcoded + seq {scheme} {stats_label} w={w} (no unigram)",
                    unigram_enabled=False,
                    seq=prior,
                    w=w,
                )

    for name in artifact_names:
        artifact = load_transition_prior(name)
        for w in weights:
            run(f"unigram + artifact {name} w={w}", unigram_enabled=True, seq=artifact, w=w)
        if args.seq_without_unigram:
            for w in weights:
                run(
                    f"handcoded + artifact {name} w={w} (no unigram)",
                    unigram_enabled=False,
                    seq=artifact,
                    w=w,
                )

    tiers = sorted({c.tier for c in clips})
    lines = [
        "# A15 step 1 — fingering-sequence prior probe (oracle audio)",
        "",
        f"- manifest: `{args.manifest}` ({len(clips)} clips, splits: {', '.join(splits)})",
        f"- prior training data: {train_label}",
        "- decode: gold pitches -> synthetic AudioEvents -> `fuse()` "
        "(audio errors excluded; scores are position-resolution only)",
        "- sequence prior gated to singleton→singleton cluster moves in the decode "
        "(chord transitions stay hand-coded — A5 territory)",
        f"- smoothing: alpha={args.alpha}, backoff_kappa={args.backoff_kappa}",
        "",
        "## Transition statistics (train split)",
        "",
        *transition_stats_markdown(train_tracks),
        "",
        "## Oracle-audio Tab F1 by config",
        "",
        "| config | "
        + " | ".join(t.replace("clean_acoustic_", "") for t in tiers)
        + " | overall | chord acc (strummed) |",
        "|---|" + "---|" * (len(tiers) + 2),
    ]
    for r in configs:
        tier_cells = " | ".join(
            f"{r.per_tier[t].f1:.4f}" if t in r.per_tier else "—" for t in tiers
        )
        lines.append(
            f"| {r.label} | {tier_cells} | {r.overall.f1:.4f} | {r.chord_acc_strummed:.3f} |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
