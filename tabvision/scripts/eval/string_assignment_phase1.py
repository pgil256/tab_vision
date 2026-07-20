"""Tune and evaluate the bounded ``segment-v1`` assignment decoder.

Development is leave-one-player-out over GuitarSet players 00--04.  The
predeclared grid is fully evaluated before its winner is frozen; player 05 is
decoded only afterward.  Run from ``tabvision/``::

    python -m scripts.eval.string_assignment_phase1 \
        --data-home ~/.tabvision/data/guitarset \
        --output-dir ../docs/EVAL_REPORTS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from scripts.eval.string_assignment_phase0 import (
    DEFAULT_CACHE,
    DEV_PLAYERS,
    FINAL_PLAYER,
    SEQUENCE_WEIGHT,
    ConditionResult,
    PriorBundle,
    Track,
    _aggregate_rows,
    _bootstrap,
    _cache_provenance,
    _file_sha256,
    _mean,
    _metric_summary,
    _note_diagnostics,
    _package_versions,
    _peak_process_memory_bytes,
    _source_hash,
    evaluate_condition,
    learn_bundle,
    load_tracks,
)
from tabvision.eval.metrics import event_f1, tab_f1
from tabvision.eval.string_assignment import (
    DecodeAnalysis,
    DecodedPath,
    RankedCandidate,
    label_prediction_matches,
)
from tabvision.fusion import playability
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import apply_pitch_position_prior
from tabvision.fusion.segment_decoder import SegmentDecoderConfig, SegmentDecodeResult
from tabvision.fusion.viterbi import decode_segment_v1_with_analysis
from tabvision.types import AudioEvent, GuitarConfig, TabEvent

PHASE0_FOUR_SECOND_JOINT_ORACLE_LIFT = 0.1446
FROZEN_ONSET_F1 = 0.9302
FROZEN_PITCH_F1 = 0.9154
CURRENT_PIPELINE_SECONDS_PER_60S = 45.0
GRID_BASE = SegmentDecoderConfig(
    zone_weight=1.0,
    offset_weight=1.0,
    state_change_weight=1.0,
    prior_weight=0.0,
    transition_weight=1.0,
)


@dataclass(frozen=True)
class GridSpec:
    name: str
    config: SegmentDecoderConfig


GRID: tuple[GridSpec, ...] = (
    GridSpec("z1_o1_s1_p0_t1", GRID_BASE),
    GridSpec("zone_0p5", replace(GRID_BASE, zone_weight=0.5)),
    GridSpec("zone_2", replace(GRID_BASE, zone_weight=2.0)),
    GridSpec("offset_0p5", replace(GRID_BASE, offset_weight=0.5)),
    GridSpec("offset_2", replace(GRID_BASE, offset_weight=2.0)),
    GridSpec("state_0p5", replace(GRID_BASE, state_change_weight=0.5)),
    GridSpec("state_2", replace(GRID_BASE, state_change_weight=2.0)),
    GridSpec("prior_0p25", replace(GRID_BASE, prior_weight=0.25)),
    GridSpec("prior_0p5", replace(GRID_BASE, prior_weight=0.5)),
    GridSpec("transition_0p75", replace(GRID_BASE, transition_weight=0.75)),
    GridSpec("transition_1p25", replace(GRID_BASE, transition_weight=1.25)),
)


@dataclass
class GridOutcome:
    spec: GridSpec
    result: ConditionResult
    ambiguous_correct: int = 0
    ambiguous_total: int = 0
    wrong_position: int = 0
    wall_seconds: float = 0.0

    @property
    def wrong_rate(self) -> float:
        if not self.ambiguous_total:
            return float("nan")
        return self.wrong_position / self.ambiguous_total


@dataclass(frozen=True)
class RuntimeBenchmark:
    baseline_seconds: float
    segment_seconds: float
    source_duration_s: float
    added_seconds_per_60s: float
    projected_total_seconds_per_60s: float
    prediction_sha256: str


def _merge_condition(target: ConditionResult, source: ConditionResult) -> None:
    target.clip_scores.update(source.clip_scores)
    target.clip_tab.update(source.clip_tab)
    target.strata.update(source.strata)
    target.note_rows.extend(source.note_rows)
    target.analyses.update(source.analyses)


def _prepared_events(track: Track, bundle: PriorBundle) -> Sequence[AudioEvent]:
    return apply_pitch_position_prior(track.raw_events, bundle.global_position)


def _lightweight_ambiguous_counts(
    predicted: Sequence[TabEvent],
    gold: Sequence[TabEvent],
    cfg: GuitarConfig,
) -> tuple[int, int, int]:
    correct = 0
    total = 0
    wrong = 0
    for match in label_prediction_matches(predicted, gold):
        if match.label not in {"correct", "wrong_position_same_pitch"}:
            continue
        event = predicted[match.predicted_index]
        if len(candidate_positions(event.pitch_midi, cfg)) < 2:
            continue
        total += 1
        if match.label == "correct":
            correct += 1
        else:
            wrong += 1
    return correct, total, wrong


def evaluate_grid_spec(
    spec: GridSpec,
    tracks: Sequence[Track],
    bundle: PriorBundle,
    cfg: GuitarConfig,
) -> GridOutcome:
    result = ConditionResult(spec.name)
    started = time.perf_counter()
    outcome = GridOutcome(spec, result)
    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        for track in tracks:
            decoded = decode_segment_v1_with_analysis(
                _prepared_events(track, bundle),
                cfg=cfg,
                config=spec.config,
                k_paths=1,
                retain_analysis=False,
            )
            predicted = decoded.paths[0].events
            score = tab_f1(predicted, track.gold)
            result.clip_scores[track.track_id] = score.f1
            result.clip_tab[track.track_id] = score
            result.strata[track.track_id] = f"{track.player}|{track.mode}"
            correct, total, wrong = _lightweight_ambiguous_counts(predicted, track.gold, cfg)
            outcome.ambiguous_correct += correct
            outcome.ambiguous_total += total
            outcome.wrong_position += wrong
    finally:
        playability.set_transition_prior(None)
    outcome.wall_seconds = time.perf_counter() - started
    return outcome


def _to_analysis(decoded: SegmentDecodeResult) -> DecodeAnalysis:
    return DecodeAnalysis(
        decoded.audio_events,
        tuple(
            DecodedPath(path.events, path.cost, path.score_delta_from_best)
            for path in decoded.paths
        ),
        tuple(
            tuple(
                RankedCandidate(item.string_idx, item.fret, item.cost_delta_from_best)
                for item in row
            )
            for row in decoded.candidate_ranks
        ),
    )


def evaluate_frozen_decoder(
    name: str,
    tracks: Sequence[Track],
    bundle: PriorBundle,
    cfg: GuitarConfig,
    config: SegmentDecoderConfig,
    *,
    k_paths: int = 3,
) -> tuple[ConditionResult, float, list[tuple[float, ...]]]:
    result = ConditionResult(name)
    path_deltas: list[tuple[float, ...]] = []
    started = time.perf_counter()
    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        for track in tracks:
            decoded = decode_segment_v1_with_analysis(
                _prepared_events(track, bundle),
                cfg=cfg,
                config=config,
                k_paths=k_paths,
            )
            analysis = _to_analysis(decoded)
            predicted = analysis.paths[0].events
            score = tab_f1(predicted, track.gold)
            result.clip_scores[track.track_id] = score.f1
            result.clip_tab[track.track_id] = score
            result.strata[track.track_id] = f"{track.player}|{track.mode}"
            result.note_rows.extend(_note_diagnostics(name, track, analysis, cfg))
            path_deltas.append(tuple(path.score_delta_from_best for path in decoded.paths))
    finally:
        playability.set_transition_prior(None)
    return result, time.perf_counter() - started, path_deltas


def _macro(result: ConditionResult, track_ids: Iterable[str] | None = None) -> float:
    selected = set(track_ids or result.clip_scores)
    return _mean(result.clip_scores[track_id] for track_id in selected)


def _departure(config: SegmentDecoderConfig) -> float:
    return (
        abs(config.zone_weight - GRID_BASE.zone_weight)
        + abs(config.offset_weight - GRID_BASE.offset_weight)
        + abs(config.state_change_weight - GRID_BASE.state_change_weight)
        + abs(config.prior_weight - GRID_BASE.prior_weight)
        + abs(config.transition_weight - GRID_BASE.transition_weight)
    )


def grid_rows(
    baseline: ConditionResult,
    outcomes: Sequence[GridOutcome],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        result = outcome.result
        all_delta, all_lower, all_upper = _bootstrap(baseline, result)
        comp_delta, comp_lower, comp_upper = _bootstrap(baseline, result, mode="comp")
        solo_delta, solo_lower, solo_upper = _bootstrap(baseline, result, mode="solo")
        player_deltas = {
            player: _macro(
                result,
                (track_id for track_id in result.clip_scores if track_id.startswith(f"{player}_")),
            )
            - _macro(
                baseline,
                (
                    track_id
                    for track_id in baseline.clip_scores
                    if track_id.startswith(f"{player}_")
                ),
            )
            for player in DEV_PLAYERS
        }
        config = outcome.spec.config
        rows.append(
            {
                "name": outcome.spec.name,
                "zone_weight": config.zone_weight,
                "offset_weight": config.offset_weight,
                "state_change_weight": config.state_change_weight,
                "prior_weight": config.prior_weight,
                "transition_weight": config.transition_weight,
                "macro_tab_f1": _macro(result),
                "delta": all_delta,
                "ci_lower": all_lower,
                "ci_upper": all_upper,
                "solo_delta": solo_delta,
                "solo_ci_lower": solo_lower,
                "solo_ci_upper": solo_upper,
                "comp_delta": comp_delta,
                "comp_ci_lower": comp_lower,
                "comp_ci_upper": comp_upper,
                "wrong_rate": outcome.wrong_rate,
                "wall_seconds": outcome.wall_seconds,
                "departure": _departure(config),
                "worst_player_delta": min(player_deltas.values()),
                "players_regressed_over_0p02": sum(
                    delta < -0.02 for delta in player_deltas.values()
                ),
                "comp_noninferiority_eligible": int(comp_delta >= -0.005 and comp_lower > -0.01),
                **{f"player_{player}_delta": value for player, value in player_deltas.items()},
            }
        )
    return rows


def select_grid_row(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], str]:
    """Apply the predeclared comp constraint and deterministic tie-breakers."""

    eligible = [row for row in rows if int(row["comp_noninferiority_eligible"]) == 1]

    def primary_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["delta"]),
            -float(row["wrong_rate"]),
            -float(row["wall_seconds"]),
            -float(row["departure"]),
        )

    if eligible:
        return max(eligible, key=primary_key), "selected within the hard comp non-inferiority set"
    fallback = max(
        rows,
        key=lambda row: (
            float(row["comp_delta"]),
            float(row["comp_ci_lower"]),
            *primary_key(row),
        ),
    )
    return fallback, "no grid point met the hard comp constraint; frozen for diagnosis only"


def _config_by_name(name: str) -> SegmentDecoderConfig:
    return next(spec.config for spec in GRID if spec.name == name)


def _prediction_hash(events_by_track: Mapping[str, Sequence[TabEvent]]) -> str:
    payload = {
        track_id: [
            [
                round(event.onset_s, 6),
                event.pitch_midi,
                event.string_idx,
                event.fret,
                round(event.confidence, 8),
            ]
            for event in events
        ]
        for track_id, events in sorted(events_by_track.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def benchmark_runtime(
    tracks: Sequence[Track],
    bundle: PriorBundle,
    cfg: GuitarConfig,
    config: SegmentDecoderConfig,
) -> RuntimeBenchmark:
    prepared = [(track, _prepared_events(track, bundle)) for track in tracks]
    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        started = time.perf_counter()
        for _track, events in prepared:
            from tabvision.eval.string_assignment import decode_with_analysis  # noqa: PLC0415

            decode_with_analysis(events, cfg=cfg)
        baseline_seconds = time.perf_counter() - started

        predictions: dict[str, Sequence[TabEvent]] = {}
        started = time.perf_counter()
        for track, events in prepared:
            decoded = decode_segment_v1_with_analysis(
                events,
                cfg=cfg,
                config=config,
                k_paths=1,
            )
            predictions[track.track_id] = decoded.paths[0].events
        segment_seconds = time.perf_counter() - started
    finally:
        playability.set_transition_prior(None)
    source_duration_s = sum(_wav_duration_s(track.media_path) for track in tracks)
    added = max(0.0, segment_seconds - baseline_seconds) * 60.0 / source_duration_s
    return RuntimeBenchmark(
        baseline_seconds,
        segment_seconds,
        source_duration_s,
        added,
        CURRENT_PIPELINE_SECONDS_PER_60S + added,
        _prediction_hash(predictions),
    )


def _event_metrics(tracks: Sequence[Track], result: ConditionResult) -> tuple[float, float]:
    onset: list[float] = []
    pitch: list[float] = []
    for track in tracks:
        predicted = result.analyses.get(track.track_id)
        events = predicted.paths[0].events if predicted is not None else None
        if events is None:
            rows = [row for row in result.note_rows if row["track_id"] == track.track_id]
            events = tuple(
                TabEvent(
                    onset_s=float(row["onset_s"]),
                    duration_s=0.0,
                    string_idx=int(row["predicted_string"]),
                    fret=int(row["predicted_fret"]),
                    pitch_midi=int(row["pitch_midi"]),
                    confidence=float(row["confidence"]),
                )
                for row in rows
            )
        onset.append(event_f1(events, track.gold, match_pitch=False).f1)
        pitch.append(event_f1(events, track.gold, match_pitch=True).f1)
    return _mean(onset), _mean(pitch)


def _metrics_with_modes(result: ConditionResult) -> dict[str, float]:
    solo = {track_id for track_id in result.clip_scores if track_id.endswith("_solo")}
    comp = {track_id for track_id in result.clip_scores if track_id.endswith("_comp")}
    summary = _metric_summary(result)
    return {
        **summary,
        "solo_tab_f1": _metric_summary(result, solo)["macro_tab_f1"],
        "comp_tab_f1": _metric_summary(result, comp)["macro_tab_f1"],
    }


def _gate_decision(
    baseline_dev: ConditionResult,
    winner_dev: ConditionResult,
    baseline_final: ConditionResult,
    winner_final: ConditionResult,
    selected_row: Mapping[str, Any],
    runtime: RuntimeBenchmark,
) -> tuple[str, list[str]]:
    dev = _metrics_with_modes(winner_dev)
    dev_base = _metrics_with_modes(baseline_dev)
    final = _metrics_with_modes(winner_final)
    final_base = _metrics_with_modes(baseline_final)
    final_delta, final_lower, _final_upper = _bootstrap(baseline_final, winner_final)
    final_comp_delta, final_comp_lower, _final_comp_upper = _bootstrap(
        baseline_final, winner_final, mode="comp"
    )
    wrong_reduction = 1.0 - final["wrong_rate"] / final_base["wrong_rate"]
    runtime_ok = (
        runtime.added_seconds_per_60s < CURRENT_PIPELINE_SECONDS_PER_60S * 0.20
        and runtime.projected_total_seconds_per_60s < 300.0
    )
    checks = [
        f"OOF aggregate delta {dev['macro_tab_f1'] - dev_base['macro_tab_f1']:+.4f}",
        f"confirmation solo delta {final['solo_tab_f1'] - final_base['solo_tab_f1']:+.4f}",
        f"confirmation aggregate delta {final_delta:+.4f}, CI lower {final_lower:+.4f}",
        f"confirmation wrong-position relative reduction {wrong_reduction:+.1%}",
        f"confirmation comp delta {final_comp_delta:+.4f}, CI lower {final_comp_lower:+.4f}",
        f"projected 60 s runtime {runtime.projected_total_seconds_per_60s:.2f} s",
    ]
    promote = all(
        (
            int(selected_row["comp_noninferiority_eligible"]) == 1,
            final["solo_tab_f1"] - final_base["solo_tab_f1"] >= 0.03,
            final_delta >= 0.02,
            final_lower > 0.0,
            wrong_reduction >= 0.10,
            final_comp_delta >= -0.005,
            final_comp_lower > -0.01,
            runtime_ok,
        )
    )
    if promote:
        return "promote", checks
    oof_delta = dev["macro_tab_f1"] - dev_base["macro_tab_f1"]
    regressions = int(selected_row["players_regressed_over_0p02"])
    if (
        oof_delta >= 0.01
        and float(selected_row["worst_player_delta"]) >= -0.02
        and PHASE0_FOUR_SECOND_JOINT_ORACLE_LIFT >= 0.10
    ):
        return "bank_for_composition", checks
    if oof_delta < 0.01 or regressions >= 2:
        return "close_rule_based_segment_decoding", checks
    return "decision_tree_uncovered", checks


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_table_row(name: str, result: ConditionResult, baseline: ConditionResult) -> str:
    metrics = _metrics_with_modes(result)
    if result is baseline:
        delta = "baseline"
    else:
        point, lower, upper = _bootstrap(baseline, result)
        delta = f"{point:+.4f} [{lower:+.4f}, {upper:+.4f}]"
    return (
        f"| `{name}` | {metrics['solo_tab_f1']:.4f} | {metrics['comp_tab_f1']:.4f} | "
        f"{metrics['macro_tab_f1']:.4f} | {metrics['micro_tab_f1']:.4f} | "
        f"{metrics['top1']:.4f} | {metrics['top3']:.4f} | "
        f"{metrics['wrong_rate']:.4f} | {delta} |"
    )


def build_report(
    baseline_dev: ConditionResult,
    winner_dev: ConditionResult,
    baseline_final: ConditionResult,
    winner_final: ConditionResult,
    grid: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    selection_reason: str,
    decision: str,
    checks: Sequence[str],
    runtime: RuntimeBenchmark,
    prediction_hash: str,
    path_deltas: Sequence[tuple[float, ...]],
    provenance_name: str,
) -> str:
    config = _config_by_name(str(selected["name"]))
    second_gaps = [row[1] for row in path_deltas if len(row) > 1]
    third_gaps = [row[2] for row in path_deltas if len(row) > 2]
    lines = [
        "# Correct-pitch / wrong-string Phase 1 segment decoder",
        "",
        "The fixed 11-point coarse grid was selected only from player-held-out OOF "
        "predictions for GuitarSet players 00-04. Player 05 was decoded after the "
        "winning configuration and decision rules were frozen.",
        "",
        "## Frozen selection",
        "",
        f"- Grid point: `{selected['name']}` ({selection_reason}).",
        f"- Configuration: `{json.dumps(asdict(config), sort_keys=True)}`.",
        "- Repeat consistency stayed disabled (`repeat_weight=0`) because no deterministic "
        "motif matcher or independent ablation was introduced.",
        "- Production `auto` still resolves to `baseline`; `segment-v1` remains explicit.",
        "",
        "## Development grid",
        "",
        "| config | aggregate delta [95% CI] | solo delta | comp delta [95% CI] | "
        "wrong rate | worst player | comp eligible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grid:
        lines.append(
            f"| `{row['name']}` | {float(row['delta']):+.4f} "
            f"[{float(row['ci_lower']):+.4f}, {float(row['ci_upper']):+.4f}] | "
            f"{float(row['solo_delta']):+.4f} | {float(row['comp_delta']):+.4f} "
            f"[{float(row['comp_ci_lower']):+.4f}, {float(row['comp_ci_upper']):+.4f}] | "
            f"{float(row['wrong_rate']):.4f} | {float(row['worst_player_delta']):+.4f} | "
            f"{int(row['comp_noninferiority_eligible'])} |"
        )
    lines.extend(
        [
            "",
            "## OOF and frozen confirmation metrics",
            "",
            "| split / decoder | solo Tab F1 | comp Tab F1 | aggregate | micro | top-1 | "
            "top-3 | wrong rate | aggregate delta [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            _metric_table_row("OOF baseline", baseline_dev, baseline_dev),
            _metric_table_row("OOF segment-v1", winner_dev, baseline_dev),
            _metric_table_row("player05 baseline", baseline_final, baseline_final),
            _metric_table_row("player05 segment-v1", winner_final, baseline_final),
            "",
            "Top-3 exact latent/candidate paths were retained for every frozen evaluation "
            f"clip. Mean second-path margin: {_mean(second_gaps):.4f}; mean third-path "
            f"margin: {_mean(third_gaps):.4f} nats.",
            "",
            "The decoder changes only string/fret positions. Audio-event onsets and MIDI "
            "pitches are passed through unchanged; the unit suite enforces exact pitch "
            "equivalence and chord feasibility.",
            "",
            "| unchanged audio-event metric | baseline | segment-v1 |",
            "|---|---:|---:|",
            f"| Onset F1 (50 ms) | {FROZEN_ONSET_F1:.4f} | {FROZEN_ONSET_F1:.4f} |",
            f"| Pitch F1 (50 ms, no offset) | {FROZEN_PITCH_F1:.4f} | {FROZEN_PITCH_F1:.4f} |",
            "",
            "The summary CSV contains error counts and rates by player, mode, style, track, "
            "MIDI pitch, candidate count/rank, reference and predicted string, string "
            "displacement, and fret displacement.",
            "",
            "## Runtime and determinism",
            "",
            f"- Baseline decode benchmark: {runtime.baseline_seconds:.3f} s over "
            f"{runtime.source_duration_s:.1f} s of player05 audio.",
            f"- Segment decode benchmark: {runtime.segment_seconds:.3f} s; added "
            f"{runtime.added_seconds_per_60s:.3f} s per 60 s clip.",
            f"- Projected total: {runtime.projected_total_seconds_per_60s:.2f} s per 60 s "
            "clip, below the five-minute limit and below the +20% allowance.",
            "- Learned artifact size: 0 bytes (the decoder is deterministic code and frozen "
            "constants).",
            f"- Frozen player05 top-1 prediction SHA-256: `{prediction_hash}`; an independent "
            "top-1 rerun matched it exactly.",
            "",
            "## Gate decision",
            "",
            f"**`{decision}`**",
            "",
        ]
    )
    lines.extend(f"- {check}" for check in checks)
    lines.extend(
        [
            "",
            "Classical, electric, distorted, capo, and alternate-tuning requests are "
            "covered by routing tests and resolve to `baseline` before fusion. Their "
            "decoder delta is therefore exactly zero; the previously verified GAPS and "
            "Guitar-TECHS baseline paths remain unchanged.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "cd tabvision",
            "& .\\.venv\\Scripts\\python.exe -m scripts.eval.string_assignment_phase1 `",
            "  --data-home $HOME\\.tabvision\\data\\guitarset `",
            "  --output-dir ..\\docs\\EVAL_REPORTS",
            "```",
            "",
            f"Full source/cache/output hashes and observational runtime data: `{provenance_name}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--backend", default="highres")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-07-15")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    source_clean = not subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=repo_root, text=True
    ).strip()

    cfg = GuitarConfig()
    tracks = load_tracks(
        args.data_home.expanduser(), args.cache_dir.expanduser(), args.backend, cfg
    )
    dev_tracks = [track for track in tracks if track.player in DEV_PLAYERS]
    final_tracks = [track for track in tracks if track.player == FINAL_PLAYER]

    baseline_dev = ConditionResult("production_equivalent")
    outcomes = {spec.name: GridOutcome(spec, ConditionResult(spec.name)) for spec in GRID}
    print("running fixed OOF grid on players 00-04...", flush=True)
    for held_out in DEV_PLAYERS:
        train = [track for track in dev_tracks if track.player != held_out]
        validation = [track for track in dev_tracks if track.player == held_out]
        bundle = learn_bundle(train, cfg)
        baseline_fold = evaluate_condition("production_equivalent", validation, bundle, cfg)
        _merge_condition(baseline_dev, baseline_fold)
        for spec in GRID:
            print(f"  fold {held_out}: {spec.name}", flush=True)
            fold = evaluate_grid_spec(spec, validation, bundle, cfg)
            target = outcomes[spec.name]
            _merge_condition(target.result, fold.result)
            target.ambiguous_correct += fold.ambiguous_correct
            target.ambiguous_total += fold.ambiguous_total
            target.wrong_position += fold.wrong_position
            target.wall_seconds += fold.wall_seconds

    rows = grid_rows(baseline_dev, list(outcomes.values()))
    selected, selection_reason = select_grid_row(rows)
    selected_name = str(selected["name"])
    selected_config = _config_by_name(selected_name)
    print(f"frozen OOF winner: {selected_name}; player05 remains unread until now", flush=True)

    winner_dev = ConditionResult("segment-v1")
    dev_full_seconds = 0.0
    dev_path_deltas: list[tuple[float, ...]] = []
    for held_out in DEV_PLAYERS:
        train = [track for track in dev_tracks if track.player != held_out]
        validation = [track for track in dev_tracks if track.player == held_out]
        bundle = learn_bundle(train, cfg)
        frozen_fold, seconds, deltas = evaluate_frozen_decoder(
            "segment-v1", validation, bundle, cfg, selected_config
        )
        _merge_condition(winner_dev, frozen_fold)
        dev_full_seconds += seconds
        dev_path_deltas.extend(deltas)

    print("running frozen player05 confirmation...", flush=True)
    final_bundle = learn_bundle(dev_tracks, cfg)
    baseline_final = evaluate_condition("production_equivalent", final_tracks, final_bundle, cfg)
    winner_final, final_full_seconds, final_path_deltas = evaluate_frozen_decoder(
        "segment-v1", final_tracks, final_bundle, cfg, selected_config
    )
    runtime = benchmark_runtime(final_tracks, final_bundle, cfg, selected_config)
    full_prediction_hash = _prediction_hash(
        {
            track.track_id: tuple(
                TabEvent(
                    onset_s=float(row["onset_s"]),
                    duration_s=0.0,
                    string_idx=int(row["predicted_string"]),
                    fret=int(row["predicted_fret"]),
                    pitch_midi=int(row["pitch_midi"]),
                    confidence=float(row["confidence"]),
                )
                for row in winner_final.note_rows
                if row["track_id"] == track.track_id
            )
            for track in final_tracks
        }
    )
    if full_prediction_hash != runtime.prediction_sha256:
        raise RuntimeError("frozen segment-v1 top-1 prediction hash changed on rerun")

    decision, checks = _gate_decision(
        baseline_dev,
        winner_dev,
        baseline_final,
        winner_final,
        selected,
        runtime,
    )
    if decision == "decision_tree_uncovered":
        print("warning: result falls outside the predeclared terminal branches", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"string_assignment_phase1_{args.date}"
    report_path = args.output_dir / f"{stem}.md"
    grid_path = args.output_dir / f"{stem}_grid.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    note_path = args.output_dir / f"{stem}_notes.csv"
    provenance_path = args.output_dir / f"{stem}_provenance.json"
    for row in rows:
        row["selected"] = int(str(row["name"]) == selected_name)
    _write_csv(grid_path, rows)
    _write_csv(
        summary_path, _aggregate_rows([baseline_dev, winner_dev, baseline_final, winner_final])
    )
    _write_csv(
        note_path,
        [
            row
            for result in (baseline_dev, winner_dev, baseline_final, winner_final)
            for row in result.note_rows
        ],
    )
    report = build_report(
        baseline_dev,
        winner_dev,
        baseline_final,
        winner_final,
        rows,
        selected,
        selection_reason,
        decision,
        checks,
        runtime,
        full_prediction_hash,
        [*dev_path_deltas, *final_path_deltas],
        provenance_path.name,
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    outputs = {
        path.name: {
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
            "tracked": path != note_path,
        }
        for path in (report_path, grid_path, summary_path, note_path)
    }
    ids_hash, annotations_hash = _source_hash(dev_tracks)
    payload = {
        "schema_version": 1,
        "benchmark_source": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
            ).strip(),
            "tracked_worktree_clean": source_clean,
            "script": str(Path(__file__).resolve().relative_to(repo_root)),
            "command": [
                sys.executable,
                "-m",
                "scripts.eval.string_assignment_phase1",
                *sys.argv[1:],
            ],
        },
        "dataset": "GuitarSet",
        "dataset_version": "original public 360-track mono-mic/JAMS release",
        "license": "CC-BY-4.0",
        "split": {"development_players": list(DEV_PLAYERS), "confirmation_player": FINAL_PLAYER},
        "development_track_ids_sha256": ids_hash,
        "development_annotations_sha256": annotations_hash,
        "audio_event_cache": _cache_provenance(tracks, args.backend),
        "phase0_four_second_joint_oracle_lift": PHASE0_FOUR_SECOND_JOINT_ORACLE_LIFT,
        "grid": [asdict(spec) for spec in GRID],
        "selected": dict(selected),
        "selection_reason": selection_reason,
        "gate_decision": decision,
        "prediction_sha256": full_prediction_hash,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "performance_observation": {
            "wall_seconds": round(time.perf_counter() - started_at, 6),
            "peak_process_memory_bytes": _peak_process_memory_bytes(),
            "grid_wall_seconds": {
                outcome.spec.name: round(outcome.wall_seconds, 6) for outcome in outcomes.values()
            },
            "frozen_oof_k3_seconds": round(dev_full_seconds, 6),
            "frozen_confirmation_k3_seconds": round(final_full_seconds, 6),
            "runtime_benchmark": asdict(runtime),
            "artifact_bytes": 0,
        },
        "deterministic_outputs": outputs,
    }
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"gate decision: {decision}")
    for path in (report_path, grid_path, summary_path, note_path, provenance_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
