"""Program S Phase S1a — SynthTab prior probe on GuitarSet dev players.

Gold-pitch oracle probe (A15 step-1 instrument): decodes synthetic
``AudioEvent`` streams built from development gold (players 00-04 only;
the frozen player-05 confirmation set is never touched) through the real
cluster Viterbi, so the only thing measured is the decode's position
choice. Conditions swap the pitch-position and transition priors:

- ``baseline``        — registered ``guitarset-v1`` + ``guitarset-seq-v1``.
- ``st-acoustic``     — SynthTab acoustic-variant position + sequence.
- ``st-all``          — SynthTab all-guitar-variant position + sequence.
- ``st-acoustic-pos`` — SynthTab acoustic position, registered sequence.
- ``st-acoustic-seq`` — registered position, SynthTab acoustic sequence.

Sequence weight is the production ``SEQUENCE_PRIOR_WEIGHT`` (4.0). Paired
clip-stratified bootstrap (10k resamples, strata = player|mode, fixed
seed) gives the delta CI vs baseline. Gate per the S1a plan: aggregate
CI lower bound > 0 with comp non-inferiority (mean delta >= -0.005 and
one-sided 95% lower bound > -0.01).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.eval.metrics import TabF1Result, tab_f1
from tabvision.fusion import fuse, playability
from tabvision.fusion.position_prior import (
    apply_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.transition_prior import load_transition_prior
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_720
COMP_MEAN_FLOOR = -0.005
COMP_ONESIDED_FLOOR = -0.01


@dataclass(frozen=True)
class Clip:
    track_id: str
    player: str
    mode: str  # "solo" | "comp"


def _list_dev_clips(data_home: Path, cap: int) -> list[Clip]:
    annotation_dir = data_home / "annotation"
    clips = [
        Clip(path.stem, path.stem[:2], path.stem.rsplit("_", 1)[-1])
        for path in sorted(annotation_dir.glob("*.jams"))
        if path.stem[:2] in DEV_PLAYERS
    ]
    if not clips:
        raise SystemExit(f"no dev annotations under {annotation_dir}")
    return clips[:cap] if cap else clips


def _oracle_events(gold: list[TabEvent]) -> list[AudioEvent]:
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


def _macro(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _micro(results: list[TabF1Result]) -> float:
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bootstrap(
    clips: list[Clip],
    base: dict[str, float],
    cand: dict[str, float],
    *,
    mode: str | None = None,
) -> tuple[float, float, float, float]:
    """Paired stratified bootstrap: (mean delta, lo95, hi95, one-sided lo95)."""
    strata: dict[str, list[str]] = {}
    for clip in clips:
        if mode is not None and clip.mode != mode:
            continue
        strata.setdefault(f"{clip.player}|{clip.mode}", []).append(clip.track_id)
    deltas = {
        track_id: cand[track_id] - base[track_id] for ids in strata.values() for track_id in ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stratum_arrays = [np.array([deltas[track_id] for track_id in ids]) for ids in strata.values()]
    total = sum(len(a) for a in stratum_arrays)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        acc = 0.0
        for arr in stratum_arrays:
            acc += arr[rng.integers(0, len(arr), len(arr))].sum()
        means[i] = acc / total
    observed = sum(a.sum() for a in stratum_arrays) / total
    lo, hi = np.percentile(means, [2.5, 97.5])
    onesided = float(np.percentile(means, 5.0))
    return float(observed), float(lo), float(hi), onesided


def _run_condition(
    label: str,
    clips: list[Clip],
    golds: dict[str, list[TabEvent]],
    *,
    position_prior,
    sequence_prior,
    cfg: GuitarConfig,
) -> dict[str, object]:
    playability.set_transition_prior(sequence_prior, SEQUENCE_PRIOR_WEIGHT)
    per_clip: dict[str, float] = {}
    results: list[TabF1Result] = []
    started = time.perf_counter()
    try:
        for clip in clips:
            gold = golds[clip.track_id]
            events = apply_pitch_position_prior(_oracle_events(gold), position_prior)
            decoded = fuse(events, [], cfg, SessionConfig(), lambda_vision=0.0)
            score = tab_f1(decoded, gold)
            per_clip[clip.track_id] = score.f1
            results.append(score)
    finally:
        playability.set_transition_prior(None)
    elapsed = time.perf_counter() - started
    solo = [per_clip[c.track_id] for c in clips if c.mode == "solo"]
    comp = [per_clip[c.track_id] for c in clips if c.mode == "comp"]
    print(
        f"{label}: aggregate={_macro(list(per_clip.values())):.4f} "
        f"solo={_macro(solo):.4f} comp={_macro(comp):.4f} ({elapsed:.0f}s)",
        flush=True,
    )
    return {
        "label": label,
        "per_clip": per_clip,
        "macro_aggregate": _macro(list(per_clip.values())),
        "macro_solo": _macro(solo),
        "macro_comp": _macro(comp),
        "micro": _micro(results),
        "seconds": round(elapsed, 1),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--priors-dir", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=0, help="Cap for smoke runs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        default=None,
        metavar="LABEL=POS,SEQ",
        help=(
            "Replace the default condition list (baseline stays first). POS/SEQ "
            "are 'gs' (registered guitarset pair) or a filename in --priors-dir."
        ),
    )
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    data_home = args.data_home or (Path(data_root) / "guitarset")
    priors_dir = args.priors_dir or (Path(data_root) / "models" / "synthtab_priors")

    cfg = GuitarConfig()
    gs_pos = load_pitch_position_prior("guitarset-v1", cfg=cfg)
    gs_seq = load_transition_prior("guitarset-seq-v1")
    artifact_paths: dict[str, Path] = {}

    def _pos(spec: str):
        if spec == "gs":
            return gs_pos
        path = priors_dir / spec
        artifact_paths[spec] = path
        return load_pitch_position_prior(path, cfg=cfg)

    def _seq(spec: str):
        if spec == "gs":
            return gs_seq
        path = priors_dir / spec
        artifact_paths[spec] = path
        return load_transition_prior(path)

    if args.conditions:
        conditions = [("baseline", gs_pos, gs_seq)]
        for spec in args.conditions:
            label, _, pair = spec.partition("=")
            pos_spec, _, seq_spec = pair.partition(",")
            conditions.append((label, _pos(pos_spec), _seq(seq_spec)))
    else:
        st_pos_ac = _pos("synthtab_v1_acoustic.json")
        st_seq_ac = _seq("synthtab_seq_v1_acoustic.json")
        conditions = [
            ("baseline", gs_pos, gs_seq),
            ("st-acoustic", st_pos_ac, st_seq_ac),
            ("st-all", _pos("synthtab_v1_all.json"), _seq("synthtab_seq_v1_all.json")),
            ("st-acoustic-pos", st_pos_ac, gs_seq),
            ("st-acoustic-seq", gs_pos, st_seq_ac),
        ]

    clips = _list_dev_clips(data_home, args.clips)
    golds = {
        clip.track_id: parse_guitarset_jams(data_home / "annotation" / f"{clip.track_id}.jams")
        for clip in clips
    }
    print(f"clips: {len(clips)} (players {sorted({c.player for c in clips})})")
    rows = [
        _run_condition(label, clips, golds, position_prior=pos, sequence_prior=seq, cfg=cfg)
        for label, pos, seq in conditions
    ]

    base_scores = rows[0]["per_clip"]
    for row in rows[1:]:
        observed, lo, hi, _ = _bootstrap(clips, base_scores, row["per_clip"])
        comp_observed, _, _, comp_onesided = _bootstrap(
            clips, base_scores, row["per_clip"], mode="comp"
        )
        gate = lo > 0.0 and comp_observed >= COMP_MEAN_FLOOR and comp_onesided > COMP_ONESIDED_FLOOR
        row["delta"] = {
            "aggregate_mean": observed,
            "aggregate_ci95": [lo, hi],
            "comp_mean": comp_observed,
            "comp_onesided_lo95": comp_onesided,
            "gate_pass": gate,
        }

    payload = {
        "clips": len(clips),
        "players": sorted({c.player for c in clips}),
        "sequence_weight": SEQUENCE_PRIOR_WEIGHT,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "artifacts": {
            name: {"path": str(p), "sha256": _sha256(p)}
            for name, p in sorted(artifact_paths.items())
        },
        "conditions": [{k: v for k, v in row.items() if k != "per_clip"} for row in rows],
        "per_clip": {row["label"]: row["per_clip"] for row in rows},
    }
    args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# S1a SynthTab prior probe — oracle-pitch decode, GuitarSet dev players 00-04",
        "",
        f"Clips: {len(clips)} | sequence weight {SEQUENCE_PRIOR_WEIGHT} | "
        f"paired stratified bootstrap {BOOTSTRAP_RESAMPLES} resamples, "
        f"seed {BOOTSTRAP_SEED}",
        "",
        "| condition | aggregate | solo | comp | micro "
        "| Δagg [95% CI] | Δcomp (1-sided lo95) | gate |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        if "delta" in row:
            delta = row["delta"]
            delta_cell = (
                f"{delta['aggregate_mean']:+.4f} "
                f"[{delta['aggregate_ci95'][0]:+.4f}, {delta['aggregate_ci95'][1]:+.4f}]"
            )
            comp_cell = f"{delta['comp_mean']:+.4f} ({delta['comp_onesided_lo95']:+.4f})"
            gate_cell = "**PASS**" if delta["gate_pass"] else "fail"
        else:
            delta_cell = comp_cell = "—"
            gate_cell = "baseline"
        lines.append(
            f"| {row['label']} | {row['macro_aggregate']:.4f} | {row['macro_solo']:.4f} "
            f"| {row['macro_comp']:.4f} | {row['micro']:.4f} | {delta_cell} "
            f"| {comp_cell} | {gate_cell} |"
        )
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
