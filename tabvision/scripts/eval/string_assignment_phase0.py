"""Run the leakage-free correct-pitch/wrong-string Phase 0 benchmark.

The script transcribes GuitarSet once, then evaluates every fusion condition
against identical cached ``AudioEvent`` objects. Development results are
leave-one-player-out over players 00-04; player 05 is decoded only after all
development folds have completed.

Run from ``tabvision/``::

    python -m scripts.eval.string_assignment_phase0 \
        --data-home ~/.tabvision/data/guitarset \
        --output-dir ../docs/EVAL_REPORTS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.eval.a3_fusion_sweep import _raw_events_cached, make_shared_audio_backend
from scripts.eval.build_guitarset_seq_v1_prior import build_payload as build_sequence_payload
from scripts.eval.build_guitarset_v1_prior import build_payload as build_position_payload
from scripts.eval.string_assignment_oracles import (
    OracleApplication,
    apply_gold_oracle,
    fixed_window_groups,
    track_groups,
)
from tabvision.eval.guitarset_audio import list_guitarset_track_ids, parse_guitarset_jams
from tabvision.eval.metrics import TabF1Result, event_f1, tab_f1
from tabvision.eval.string_assignment import (
    DecodeAnalysis,
    decode_with_analysis,
    expected_calibration_error,
    label_prediction_matches,
    paired_stratified_bootstrap,
    phrase_windows,
    wrong_below_correct_auc,
)
from tabvision.fusion import chord, playability
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.transition_prior import (
    TransitionPrior,
    learn_transition_prior,
    load_transition_prior,
)
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
FINAL_PLAYER = "05"
CONDITIONS = ("none", "position_only", "production_equivalent", "mode_specific")
SEQUENCE_WEIGHT = 4.0
DEFAULT_CACHE = Path.home() / ".tabvision/cache/a3_fusion_sweep"
WINDOW_SIZES = (1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class Track:
    track_id: str
    player: str
    mode: str
    style: str
    media_path: Path
    annotation_path: Path
    audio_cache_key: str
    audio_cache_path: Path
    gold: tuple[TabEvent, ...]
    raw_events: tuple[AudioEvent, ...]


@dataclass
class ConditionResult:
    name: str
    clip_scores: dict[str, float] = field(default_factory=dict)
    clip_tab: dict[str, TabF1Result] = field(default_factory=dict)
    strata: dict[str, str] = field(default_factory=dict)
    note_rows: list[dict[str, Any]] = field(default_factory=list)
    analyses: dict[str, DecodeAnalysis] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorBundle:
    global_position: PitchPositionPrior
    global_sequence: TransitionPrior
    mode_position: Mapping[str, PitchPositionPrior]
    mode_sequence: Mapping[str, TransitionPrior]


@dataclass
class OracleAggregate:
    """One held-out oracle row accumulated across all clips."""

    name: str
    ambiguous_correct: int = 0
    ambiguous_total: int = 0
    dropped_impossible: int = 0
    clip_tab: dict[str, TabF1Result] = field(default_factory=dict)
    state_counts: Counter[str] = field(default_factory=Counter)

    @property
    def ambiguous_accuracy(self) -> float:
        return self.ambiguous_correct / self.ambiguous_total


def _track_metadata(track_id: str) -> tuple[str, str, str]:
    player, body = track_id.split("_", 1)
    mode = body.rsplit("_", 1)[1]
    style = body.split("-", 1)[0]
    if player not in (*DEV_PLAYERS, FINAL_PLAYER) or mode not in ("solo", "comp"):
        raise ValueError(f"unexpected GuitarSet track id: {track_id}")
    return player, mode, style


def _session_for_mode(mode: str) -> SessionConfig:
    return SessionConfig(style="fingerstyle" if mode == "solo" else "strumming")


def _audio_cache_identity(
    media_path: Path,
    backend_name: str,
    cache_dir: Path,
) -> tuple[str, Path]:
    key = json.dumps(
        {
            "media": str(media_path.resolve()),
            "backend": backend_name,
            "mtime": media_path.stat().st_mtime_ns,
        }
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return digest, cache_dir / f"{media_path.stem}.{digest}.json"


def load_tracks(
    data_home: Path,
    cache_dir: Path,
    backend_name: str,
    cfg: GuitarConfig,
) -> list[Track]:
    track_ids = list_guitarset_track_ids(data_home, split="all")
    expected = 60 * 6
    if len(track_ids) != expected:
        raise RuntimeError(f"expected {expected} GuitarSet tracks, found {len(track_ids)}")

    backend = make_shared_audio_backend(backend_name)
    tracks: list[Track] = []
    for index, track_id in enumerate(track_ids, start=1):
        player, mode, style = _track_metadata(track_id)
        media = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        annotation = data_home / "annotation" / f"{track_id}.jams"
        cache_key, cache_path = _audio_cache_identity(media, backend_name, cache_dir)
        raw = _raw_events_cached(
            media,
            _session_for_mode(mode),
            backend_name,
            cache_dir,
            backend,
        )
        gold = parse_guitarset_jams(annotation, cfg)
        tracks.append(
            Track(
                track_id,
                player,
                mode,
                style,
                media,
                annotation,
                cache_key,
                cache_path,
                tuple(gold),
                tuple(raw),
            )
        )
        if index % 20 == 0:
            print(f"  cache ready: {index}/{len(track_ids)} tracks", flush=True)
    return tracks


def learn_bundle(tracks: Sequence[Track], cfg: GuitarConfig) -> PriorBundle:
    if not tracks:
        raise ValueError("cannot learn priors without tracks")

    def position(subset: Sequence[Track]) -> PitchPositionPrior:
        examples = [event for track in subset for event in track.gold]
        return learn_pitch_position_prior(examples, cfg=cfg, alpha=1.0, power=2.0)

    def sequence(subset: Sequence[Track]) -> TransitionPrior:
        return learn_transition_prior(
            (track.gold for track in subset),
            scheme="delta_fret",
            alpha=0.5,
            backoff_kappa=8.0,
            singleton_only=True,
        )

    modes = {mode: [track for track in tracks if track.mode == mode] for mode in ("solo", "comp")}
    return PriorBundle(
        global_position=position(tracks),
        global_sequence=sequence(tracks),
        mode_position={mode: position(subset) for mode, subset in modes.items()},
        mode_sequence={mode: sequence(subset) for mode, subset in modes.items()},
    )


def _condition_priors(
    condition: str,
    track: Track,
    bundle: PriorBundle,
) -> tuple[PitchPositionPrior | None, TransitionPrior | None]:
    if condition == "none":
        return None, None
    if condition == "position_only":
        return bundle.global_position, None
    if condition == "production_equivalent":
        return bundle.global_position, bundle.global_sequence
    if condition == "mode_specific":
        return bundle.mode_position[track.mode], bundle.mode_sequence[track.mode]
    raise ValueError(f"unknown condition: {condition}")


def evaluate_condition(
    name: str,
    tracks: Sequence[Track],
    bundle: PriorBundle,
    cfg: GuitarConfig,
    *,
    keep_analyses: bool = False,
) -> ConditionResult:
    result = ConditionResult(name)
    try:
        for track in tracks:
            position, sequence = _condition_priors(name, track, bundle)
            playability.set_transition_prior(sequence, SEQUENCE_WEIGHT)
            events: Sequence[AudioEvent] = track.raw_events
            if position is not None:
                events = apply_pitch_position_prior(events, position)
            analysis = decode_with_analysis(events, cfg=cfg)
            if not analysis.paths:
                raise RuntimeError(f"decode produced no path for {track.track_id}")
            predicted = analysis.paths[0].events
            score = tab_f1(predicted, track.gold)
            result.clip_scores[track.track_id] = score.f1
            result.clip_tab[track.track_id] = score
            result.strata[track.track_id] = f"{track.player}|{track.mode}"
            if keep_analyses:
                result.analyses[track.track_id] = analysis
            result.note_rows.extend(_note_diagnostics(name, track, analysis, cfg))
    finally:
        playability.set_transition_prior(None)
    return result


def _note_diagnostics(
    condition: str,
    track: Track,
    analysis: DecodeAnalysis,
    cfg: GuitarConfig,
) -> list[dict[str, Any]]:
    predicted = analysis.paths[0].events
    matches = label_prediction_matches(predicted, track.gold)
    cluster_positions: dict[int, tuple[int, int]] = {}
    flat_index = 0
    clusters = chord.cluster_events(list(analysis.audio_events))
    for cluster_index, cluster_events in enumerate(clusters):
        for cluster_event_index in range(len(cluster_events)):
            cluster_positions[flat_index] = (cluster_index, cluster_event_index)
            flat_index += 1
    if flat_index != len(predicted):
        raise AssertionError("cluster indexing drifted from the decoded event order")

    session = _session_for_mode(track.mode)
    rows: list[dict[str, Any]] = []
    for match in matches:
        event = predicted[match.predicted_index]
        ranks = analysis.candidate_ranks[match.predicted_index]
        cluster_index, cluster_event_index = cluster_positions[match.predicted_index]
        gold = track.gold[match.gold_index] if match.gold_index is not None else None
        gold_rank: int | str = ""
        if gold is not None:
            gold_rank = next(
                (
                    rank_idx
                    for rank_idx, candidate in enumerate(ranks, start=1)
                    if (candidate.string_idx, candidate.fret) == (gold.string_idx, gold.fret)
                ),
                "",
            )
        pitch_matched = match.label in ("correct", "wrong_position_same_pitch")
        ambiguous = pitch_matched and len(ranks) >= 2
        candidate_path = ";".join(
            f"{candidate.string_idx}:{candidate.fret}:{candidate.cost_delta_from_best:.8f}"
            for candidate in ranks
        )
        rows.append(
            {
                "condition": condition,
                "evaluation_split": (
                    "development_oof" if track.player in DEV_PLAYERS else "held_out_05"
                ),
                "track_id": track.track_id,
                "player": track.player,
                "mode": track.mode,
                "style": track.style,
                "session_instrument": session.instrument,
                "session_tone": session.tone,
                "session_style": session.style,
                "tuning_midi": ",".join(str(pitch) for pitch in cfg.tuning_midi),
                "capo": cfg.capo,
                "max_fret": cfg.max_fret,
                "audio_cache_key": track.audio_cache_key,
                "event_id": f"{track.track_id}:{match.predicted_index:06d}",
                "event_index": match.predicted_index,
                "cluster_index": cluster_index,
                "cluster_event_index": cluster_event_index,
                "gold_index": "" if match.gold_index is None else match.gold_index,
                "onset_s": f"{event.onset_s:.6f}",
                "pitch_midi": event.pitch_midi,
                "candidate_count": len(ranks),
                "candidate_path": candidate_path,
                "predicted_string": event.string_idx,
                "predicted_fret": event.fret,
                "reference_string": "" if gold is None else gold.string_idx,
                "reference_fret": "" if gold is None else gold.fret,
                "predicted_minus_reference_string": (
                    "" if gold is None else event.string_idx - gold.string_idx
                ),
                "fret_displacement": "" if gold is None else event.fret - gold.fret,
                "confidence": f"{event.confidence:.8f}",
                "reference_rank": gold_rank,
                "candidate_top1": int(ambiguous and gold_rank == 1),
                "candidate_top3": int(ambiguous and isinstance(gold_rank, int) and gold_rank <= 3),
                "ambiguous_pitch_match": int(ambiguous),
                "label": match.label,
            }
        )
    return rows


def _sum_tab(scores: Iterable[TabF1Result]) -> TabF1Result:
    items = list(scores)
    tp = sum(item.true_positives for item in items)
    fp = sum(item.false_positives for item in items)
    fn = sum(item.false_negatives for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


def _metric_summary(result: ConditionResult, track_ids: set[str] | None = None) -> dict[str, float]:
    selected = track_ids or set(result.clip_scores)
    scores = [result.clip_scores[track_id] for track_id in selected]
    micro = _sum_tab(result.clip_tab[track_id] for track_id in selected)
    rows = [row for row in result.note_rows if str(row["track_id"]) in selected]
    ambiguous = [row for row in rows if row["ambiguous_pitch_match"] == 1]
    confidence = [(float(row["confidence"]), row["label"] == "correct") for row in ambiguous]
    wrong = sum(row["label"] == "wrong_position_same_pitch" for row in ambiguous)
    return {
        "macro_tab_f1": sum(scores) / len(scores),
        "micro_tab_f1": micro.f1,
        "ambiguous_n": float(len(ambiguous)),
        "top1": _mean(float(row["candidate_top1"]) for row in ambiguous),
        "top3": _mean(float(row["candidate_top3"]) for row in ambiguous),
        "wrong_rate": wrong / len(ambiguous) if ambiguous else float("nan"),
        "auc": wrong_below_correct_auc(confidence),
        "ece": expected_calibration_error(confidence),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def _bootstrap(
    baseline: ConditionResult,
    candidate: ConditionResult,
    *,
    mode: str | None = None,
) -> tuple[float, float, float]:
    ids = {
        track_id
        for track_id in baseline.clip_scores
        if mode is None or track_id.endswith(f"_{mode}")
    }
    boot = paired_stratified_bootstrap(
        {key: baseline.clip_scores[key] for key in ids},
        {key: candidate.clip_scores[key] for key in ids},
        {key: baseline.strata[key] for key in ids},
        n_resamples=10_000,
        seed=42,
    )
    return boot.mean_delta, boot.lower, boot.upper


def _oracle_probe(
    final_tracks: Sequence[Track],
    baseline: ConditionResult,
    bundle: PriorBundle,
    cfg: GuitarConfig,
) -> dict[str, float]:
    baseline_correct = anchored_correct = best3_correct = total = phrases = 0
    infeasible_phrases = 0
    baseline_tab_scores: list[float] = []
    anchored_tab_scores: list[float] = []
    best3_tab_scores: list[float] = []
    try:
        for track in final_tracks:
            analysis = baseline.analyses[track.track_id]
            predicted = analysis.paths[0].events
            anchored_predicted = list(predicted)
            best3_predicted = list(predicted)
            matches = label_prediction_matches(predicted, track.gold)
            gold_by_index = {
                match.predicted_index: track.gold[match.gold_index]
                for match in matches
                if match.gold_index is not None
                and match.label in ("correct", "wrong_position_same_pitch")
                and len(analysis.candidate_ranks[match.predicted_index]) >= 2
            }
            windows = phrase_windows(predicted, set(gold_by_index))
            playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
            for window in windows:
                anchor_gold = gold_by_index[window.anchor_index]
                constraint = Candidate(anchor_gold.string_idx, anchor_gold.fret)
                left = (
                    Candidate(
                        predicted[window.start_index - 1].string_idx,
                        predicted[window.start_index - 1].fret,
                    )
                    if window.start_index > 0
                    else None
                )
                right = (
                    Candidate(
                        predicted[window.end_index].string_idx, predicted[window.end_index].fret
                    )
                    if window.end_index < len(predicted)
                    else None
                )
                local_gold = {
                    index - window.start_index: gold
                    for index, gold in gold_by_index.items()
                    if window.start_index <= index < window.end_index
                }
                if not local_gold:
                    continue
                phrase = decode_with_analysis(
                    analysis.audio_events[window.start_index : window.end_index],
                    cfg=cfg,
                    constraints={window.anchor_index - window.start_index: constraint},
                    k_paths=3,
                    left_boundary=left,
                    right_boundary=right,
                )
                if not phrase.paths:
                    # This is the refinement API's honest 422 case: a valid
                    # pitch-preserving correction can still be incompatible
                    # with other detected pitches in the same chord state.
                    # Count it as no oracle improvement instead of silently
                    # dropping a hard phrase from the denominator.
                    unchanged_correct = sum(
                        (
                            predicted[window.start_index + index].string_idx,
                            predicted[window.start_index + index].fret,
                        )
                        == (gold.string_idx, gold.fret)
                        for index, gold in local_gold.items()
                    )
                    baseline_correct += unchanged_correct
                    anchored_correct += unchanged_correct
                    best3_correct += unchanged_correct
                    total += len(local_gold)
                    phrases += 1
                    infeasible_phrases += 1
                    continue
                baseline_correct += sum(
                    (
                        predicted[window.start_index + index].string_idx,
                        predicted[window.start_index + index].fret,
                    )
                    == (gold.string_idx, gold.fret)
                    for index, gold in local_gold.items()
                )
                path_correct = [
                    sum(
                        (path.events[index].string_idx, path.events[index].fret)
                        == (gold.string_idx, gold.fret)
                        for index, gold in local_gold.items()
                    )
                    for path in phrase.paths
                ]
                best_path_index = max(range(len(path_correct)), key=path_correct.__getitem__)
                anchored_predicted[window.start_index : window.end_index] = phrase.paths[0].events
                best3_predicted[window.start_index : window.end_index] = phrase.paths[
                    best_path_index
                ].events
                anchored_correct += path_correct[0]
                best3_correct += max(path_correct)
                total += len(local_gold)
                phrases += 1
            baseline_tab_scores.append(tab_f1(predicted, track.gold).f1)
            anchored_tab_scores.append(tab_f1(anchored_predicted, track.gold).f1)
            best3_tab_scores.append(tab_f1(best3_predicted, track.gold).f1)
    finally:
        playability.set_transition_prior(None)
    if not total:
        raise RuntimeError("oracle probe found no ambiguous pitch-matched notes")
    baseline_accuracy = baseline_correct / total
    anchored_accuracy = anchored_correct / total
    best3_accuracy = best3_correct / total
    return {
        "phrases": float(phrases),
        "infeasible_phrases": float(infeasible_phrases),
        "notes": float(total),
        "baseline": baseline_accuracy,
        "anchored_top1": anchored_accuracy,
        "best_of_three": best3_accuracy,
        "anchor_lift": anchored_accuracy - baseline_accuracy,
        "top3_lift": best3_accuracy - anchored_accuracy,
        "baseline_tab_f1": _mean(baseline_tab_scores),
        "anchored_top1_tab_f1": _mean(anchored_tab_scores),
        "best_of_three_tab_f1": _mean(best3_tab_scores),
    }


def _segment_oracles(
    final_tracks: Sequence[Track],
    baseline: ConditionResult,
    cfg: GuitarConfig,
) -> list[OracleAggregate]:
    names = ["baseline"]
    for strategy in ("offset", "fret_zone", "joint"):
        names.append(f"{strategy}_track")
        names.extend(f"{strategy}_{window:g}s" for window in WINDOW_SIZES)
    aggregates = {name: OracleAggregate(name) for name in names}

    for track in final_tracks:
        analysis = baseline.analyses[track.track_id]
        predicted = analysis.paths[0].events
        matches = label_prediction_matches(predicted, track.gold)
        gold_by_index = {
            match.predicted_index: track.gold[match.gold_index]
            for match in matches
            if match.gold_index is not None
            and match.label in ("correct", "wrong_position_same_pitch")
            and len(analysis.candidate_ranks[match.predicted_index]) >= 2
        }
        baseline_correct = sum(
            (predicted[index].string_idx, predicted[index].fret) == (gold.string_idx, gold.fret)
            for index, gold in gold_by_index.items()
        )
        baseline_row = aggregates["baseline"]
        baseline_row.ambiguous_correct += baseline_correct
        baseline_row.ambiguous_total += len(gold_by_index)
        baseline_row.clip_tab[track.track_id] = tab_f1(predicted, track.gold)

        groupings = {"track": track_groups(set(gold_by_index))}
        groupings.update(
            {
                f"{window:g}s": fixed_window_groups(
                    predicted,
                    set(gold_by_index),
                    window,
                )
                for window in WINDOW_SIZES
            }
        )
        for strategy in ("offset", "fret_zone", "joint"):
            for grouping_name, groups in groupings.items():
                application = apply_gold_oracle(
                    predicted,
                    analysis.candidate_ranks,
                    gold_by_index,
                    groups,
                    strategy=strategy,
                    cfg=cfg,
                )
                _add_oracle_application(
                    aggregates[f"{strategy}_{grouping_name}"],
                    track,
                    application,
                )

    baseline_accuracy = aggregates["baseline"].ambiguous_accuracy
    baseline_tab = _oracle_tab_summary(aggregates["baseline"])["macro_tab_f1"]
    for aggregate in aggregates.values():
        if aggregate.ambiguous_accuracy + 1e-12 < baseline_accuracy:
            raise AssertionError(f"{aggregate.name} ambiguous accuracy regressed below baseline")
        if _oracle_tab_summary(aggregate)["macro_tab_f1"] + 1e-12 < baseline_tab:
            raise AssertionError(f"{aggregate.name} Tab F1 regressed below baseline")
    return [aggregates[name] for name in names]


def _add_oracle_application(
    aggregate: OracleAggregate,
    track: Track,
    application: OracleApplication,
) -> None:
    aggregate.ambiguous_correct += application.ambiguous_correct
    aggregate.ambiguous_total += application.ambiguous_total
    aggregate.dropped_impossible += application.dropped_impossible
    aggregate.state_counts.update(dict(application.state_counts))
    aggregate.clip_tab[track.track_id] = tab_f1(application.events, track.gold)


def _oracle_tab_summary(aggregate: OracleAggregate) -> dict[str, float]:
    micro = _sum_tab(aggregate.clip_tab.values())
    return {
        "macro_tab_f1": _mean(result.f1 for result in aggregate.clip_tab.values()),
        "micro_tab_f1": micro.f1,
    }


def _baseline_event_metrics(
    final_tracks: Sequence[Track],
    baseline: ConditionResult,
) -> dict[str, float]:
    onset_scores: list[float] = []
    pitch_scores: list[float] = []
    for track in final_tracks:
        predicted = baseline.analyses[track.track_id].paths[0].events
        onset_scores.append(event_f1(predicted, track.gold, match_pitch=False).f1)
        pitch_scores.append(event_f1(predicted, track.gold, match_pitch=True).f1)
    return {"onset_f1": _mean(onset_scores), "pitch_f1": _mean(pitch_scores)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(results: Sequence[ConditionResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    dimensions = (
        ("player", lambda row: str(row["player"])),
        ("mode", lambda row: str(row["mode"])),
        ("style", lambda row: str(row["style"])),
        ("track_id", lambda row: str(row["track_id"])),
        ("pitch_midi", lambda row: str(row["pitch_midi"])),
        ("candidate_count", lambda row: str(row["candidate_count"])),
        ("candidate_rank", lambda row: str(row["reference_rank"])),
        ("predicted_string", lambda row: str(row["predicted_string"])),
        ("reference_string", lambda row: str(row["reference_string"])),
        (
            "predicted_minus_reference_string",
            lambda row: str(row["predicted_minus_reference_string"]),
        ),
        ("fret_displacement", lambda row: str(row["fret_displacement"])),
        (
            "candidate_rank_by_player_mode",
            lambda row: f"{row['player']}|{row['mode']}|{row['reference_rank']}",
        ),
        (
            "string_fret_displacement",
            lambda row: f"{row['predicted_minus_reference_string']}|{row['fret_displacement']}",
        ),
    )
    for result in results:
        evaluation_split = str(result.note_rows[0]["evaluation_split"])
        for dimension, value_for in dimensions:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in result.note_rows:
                buckets.setdefault(value_for(row), []).append(row)
            for value in sorted(buckets):
                rows = buckets[value]
                ambiguous = [row for row in rows if row["ambiguous_pitch_match"] == 1]
                out.append(
                    {
                        "condition": result.name,
                        "evaluation_split": evaluation_split,
                        "dimension": dimension,
                        "value": value,
                        "predicted_notes": len(rows),
                        "ambiguous_pitch_matches": len(ambiguous),
                        "correct_notes": sum(row["label"] == "correct" for row in rows),
                        "wrong_position_same_pitch": sum(
                            row["label"] == "wrong_position_same_pitch" for row in rows
                        ),
                        "position_accuracy": _mean(
                            float(row["label"] == "correct") for row in ambiguous
                        ),
                        "wrong_position_rate": _mean(
                            float(row["label"] == "wrong_position_same_pitch") for row in ambiguous
                        ),
                        "candidate_top1": _mean(float(row["candidate_top1"]) for row in ambiguous),
                        "candidate_top3": _mean(float(row["candidate_top3"]) for row in ambiguous),
                    }
                )
    return out


def _confusion_markdown(result: ConditionResult, player: str) -> list[str]:
    rows = [
        row
        for row in result.note_rows
        if row["player"] == player and row["ambiguous_pitch_match"] == 1
    ]
    counts = Counter((int(row["reference_string"]), int(row["predicted_string"])) for row in rows)
    lines = [
        "| reference \\ predicted | 0 | 1 | 2 | 3 | 4 | 5 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for reference in range(6):
        cells = " | ".join(str(counts[(reference, predicted)]) for predicted in range(6))
        lines.append(f"| {reference} | {cells} |")
    return lines


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _report_metric_row(
    name: str,
    solo: float,
    comp: float,
    summary: Mapping[str, float],
    delta: str,
) -> str:
    return (
        f"| `{name}` | {solo:.4f} | {comp:.4f} | "
        f"{summary['macro_tab_f1']:.4f} | {summary['micro_tab_f1']:.4f} | "
        f"{summary['top1']:.4f} | {summary['top3']:.4f} | "
        f"{summary['wrong_rate']:.4f} | {_fmt(summary['auc'])} | "
        f"{_fmt(summary['ece'])} | {delta} |"
    )


def _report(
    dev: Sequence[ConditionResult],
    final: Sequence[ConditionResult],
    oracle: Mapping[str, float],
    segment_oracles: Sequence[OracleAggregate],
    event_metrics: Mapping[str, float],
    provenance_path: Path,
) -> str:
    dev_by = {result.name: result for result in dev}
    final_by = {result.name: result for result in final}
    baseline_dev = dev_by["production_equivalent"]
    baseline_final = final_by["production_equivalent"]
    baseline_final_macro = _metric_summary(baseline_final)["macro_tab_f1"]
    segment_by = {result.name: result for result in segment_oracles}
    segment_baseline = segment_by["baseline"]
    segment_baseline_tab = _oracle_tab_summary(segment_baseline)
    joint_four = segment_by["joint_4s"]
    joint_lift = joint_four.ambiguous_accuracy - segment_baseline.ambiguous_accuracy
    lines = [
        "# Correct-pitch / wrong-string Phase 0 benchmark",
        "",
        (
            "High-resolution audio events are cached once. Development numbers are "
            "out-of-fold: each of players 00-04 is decoded with priors trained on the "
            "other four. Player 05 is the untouched final confirmation. Video and the "
            "retired melodic prior are disabled."
        ),
        "",
        (
            "Primary Tab F1 is the mean of standard per-clip Tab F1; micro Tab F1 is "
            "included as a cross-check. Confidence intervals are paired, "
            "clip-stratified 10,000-resample bootstraps."
        ),
        "",
        "## Development: leave-one-player-out players 00-04",
        "",
        (
            "| condition | solo Tab F1 | comp Tab F1 | all Tab F1 | micro | "
            "ambiguous top-1 | top-3 | same-pitch wrong rate | AUC | ECE | "
            "delta vs production [95% CI] |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CONDITIONS:
        result = dev_by[name]
        solo_ids = {key for key in result.clip_scores if key.endswith("_solo")}
        comp_ids = {key for key in result.clip_scores if key.endswith("_comp")}
        summary = _metric_summary(result)
        solo = _metric_summary(result, solo_ids)["macro_tab_f1"]
        comp = _metric_summary(result, comp_ids)["macro_tab_f1"]
        if name == baseline_dev.name:
            delta = "baseline"
        else:
            point, lower, upper = _bootstrap(baseline_dev, result)
            delta = f"{point:+.4f} [{lower:+.4f}, {upper:+.4f}]"
        lines.append(_report_metric_row(name, solo, comp, summary, delta))

    lines.extend(
        [
            "",
            "## Final confirmation: held-out player 05",
            "",
            (
                "| condition | solo Tab F1 | comp Tab F1 | all Tab F1 | micro | "
                "ambiguous top-1 | top-3 | same-pitch wrong rate | AUC | ECE | "
                "delta vs production [95% CI] |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (*CONDITIONS, "checked_in_production"):
        result = final_by[name]
        solo_ids = {key for key in result.clip_scores if key.endswith("_solo")}
        comp_ids = {key for key in result.clip_scores if key.endswith("_comp")}
        summary = _metric_summary(result)
        solo = _metric_summary(result, solo_ids)["macro_tab_f1"]
        comp = _metric_summary(result, comp_ids)["macro_tab_f1"]
        if name == baseline_final.name:
            delta = "baseline"
        else:
            point, lower, upper = _bootstrap(baseline_final, result)
            delta = f"{point:+.4f} [{lower:+.4f}, {upper:+.4f}]"
        lines.append(_report_metric_row(name, solo, comp, summary, delta))

    mode_specific_dev = dev_by["mode_specific"]
    mode_specific_final = final_by["mode_specific"]
    lines.extend(
        [
            "",
            "### Mode-specific prior deltas by target mode",
            "",
            "| split | mode | mean delta | paired 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for split, baseline, candidate in (
        ("development OOF", baseline_dev, mode_specific_dev),
        ("held-out 05 context", baseline_final, mode_specific_final),
    ):
        for mode in ("solo", "comp"):
            point, lower, upper = _bootstrap(baseline, candidate, mode=mode)
            lines.append(f"| {split} | {mode} | {point:+.4f} | [{lower:+.4f}, {upper:+.4f}] |")
    lines.extend(
        [
            "",
            (
                "Promotion is development-gated. The held-out per-mode rows are reported "
                "for confirmation context only and are not used to select a prior."
            ),
        ]
    )

    lines.extend(
        [
            "",
            (
                "Phase 0 baseline reproduction gate (`abs(delta) <= 1e-4`): "
                f"**{'PASS' if abs(baseline_final_macro - 0.6126) <= 1e-4 else 'FAIL'}** "
                f"(expected `0.6126`, observed "
                f"`{baseline_final_macro:.4f}`)."
            ),
            "",
            "### Held-out production string confusion matrix",
            "",
            *_confusion_markdown(baseline_final, FINAL_PLAYER),
            "",
            "### Held-out audio-event metrics",
            "",
            "| onset F1 | pitch F1 |",
            "|---:|---:|",
            f"| {event_metrics['onset_f1']:.4f} | {event_metrics['pitch_f1']:.4f} |",
            "",
            (
                "The segment/oracle transforms below operate only on string/fret "
                "assignment. The frozen baseline audio events are unchanged."
            ),
            "",
            "## Segment and fret-zone diagnostic ceilings",
            "",
            (
                "Each row chooses one gold state per track or fixed window. Fixed "
                "windows are cluster-safe; every selected position is a real candidate "
                "for the unchanged MIDI pitch. Impossible shifted candidates are dropped."
            ),
            "",
            (
                "| oracle | ambiguous accuracy | lift | macro Tab F1 | lift | micro "
                "Tab F1 | dropped impossible |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in segment_oracles:
        tab = _oracle_tab_summary(result)
        lines.append(
            f"| `{result.name}` | {result.ambiguous_accuracy:.4f} | "
            f"{result.ambiguous_accuracy - segment_baseline.ambiguous_accuracy:+.4f} | "
            f"{tab['macro_tab_f1']:.4f} | "
            f"{tab['macro_tab_f1'] - segment_baseline_tab['macro_tab_f1']:+.4f} | "
            f"{tab['micro_tab_f1']:.4f} | {result.dropped_impossible} |"
        )
    lines.extend(
        [
            "",
            (
                "Phase 0 segment-signal gate (four-second joint lift `>= +0.10`): "
                f"**{'PASS' if joint_lift >= 0.10 else 'FAIL'}** "
                f"({segment_baseline.ambiguous_accuracy:.4f} -> "
                f"{joint_four.ambiguous_accuracy:.4f}, {joint_lift:+.4f})."
            ),
            "",
            "## Phrase oracle",
            "",
            (
                f"Phrases: **{int(oracle['phrases'])}**; infeasible gold-anchor "
                f"phrases: **{int(oracle['infeasible_phrases'])}**; ambiguous "
                f"pitch-matched notes: **{int(oracle['notes'])}**. Infeasible "
                "phrases count as no improvement rather than being dropped."
            ),
            "",
            (
                "| baseline ambiguous | one gold anchor, top-1 | lift | best of 3 | "
                "lift over anchored top-1 | baseline Tab F1 | anchored Tab F1 | "
                "best-of-3 Tab F1 |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {oracle['baseline']:.4f} | {oracle['anchored_top1']:.4f} | "
                f"{oracle['anchor_lift']:+.4f} | {oracle['best_of_three']:.4f} | "
                f"{oracle['top3_lift']:+.4f} | {oracle['baseline_tab_f1']:.4f} | "
                f"{oracle['anchored_top1_tab_f1']:.4f} | "
                f"{oracle['best_of_three_tab_f1']:.4f} |"
            ),
            "",
            (
                "Refinement build gate (`>= +0.10`): "
                f"**{'PASS' if oracle['anchor_lift'] >= 0.10 else 'FAIL'}**. "
                "Multiple-alternative gate (`>= +0.05` over anchored top-1): "
                f"**{'PASS' if oracle['top3_lift'] >= 0.05 else 'FAIL'}**."
            ),
            "",
            "## Reproduction and provenance",
            "",
            "```powershell",
            "cd tabvision",
            (
                "& .\\.venv\\Scripts\\python.exe -m "
                "scripts.eval.string_assignment_phase0 --data-home "
                "$HOME\\.tabvision\\data\\guitarset --output-dir "
                "..\\docs\\EVAL_REPORTS"
            ),
            "```",
            "",
            (
                f"Artifact and dataset provenance: `{provenance_path.name}`. The grouped "
                "diagnostic summary includes candidate-rank, displacement, player/mode, "
                "pitch, style, and clip slices. The raw stable note table is generated "
                "locally and git-ignored because it is reproducible and approximately "
                "30 MB. Runtime/peak-memory observations, exact command/environment, "
                "cache identities, and deterministic output hashes live in provenance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = (json.dumps(payload, indent=2) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def _source_hash(tracks: Sequence[Track]) -> tuple[str, str]:
    ids = sorted(track.track_id for track in tracks)
    ids_hash = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    annotations = hashlib.sha256()
    for track in sorted(tracks, key=lambda item: item.track_id):
        annotations.update(track.track_id.encode())
        annotations.update(b"\0")
        annotations.update(hashlib.sha256(track.annotation_path.read_bytes()).digest())
    return ids_hash, annotations.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "mir_eval", "hf-midi-transcription", "pytest", "ruff", "mypy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _cache_provenance(tracks: Sequence[Track], backend_name: str) -> dict[str, Any]:
    records: list[str] = []
    total_bytes = 0
    for track in sorted(tracks, key=lambda item: item.track_id):
        if not track.audio_cache_path.exists():
            raise RuntimeError(f"missing audio-event cache after load: {track.audio_cache_path}")
        size = track.audio_cache_path.stat().st_size
        total_bytes += size
        records.append(
            f"{track.track_id}\0{track.audio_cache_key}\0"
            f"{_file_sha256(track.audio_cache_path)}\0{size}"
        )
    manifest = "\n".join(records) + "\n"
    return {
        "backend": backend_name,
        "key_schema": "sha1-16(canonical absolute media path, backend, media mtime_ns)",
        "entries": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
    }


def _write_provenance(
    path: Path,
    training_tracks: Sequence[Track],
    all_tracks: Sequence[Track],
    *,
    backend_name: str,
    command: Sequence[str],
    outputs: Mapping[str, Mapping[str, Any]],
    runtime_seconds: float,
    peak_memory_bytes: int,
    source_worktree_clean: bool,
) -> None:
    priors_dir = Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors"
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    ids_hash, annotations_hash = _source_hash(training_tracks)
    all_ids_hash, all_annotations_hash = _source_hash(all_tracks)
    data_home = training_tracks[0].annotation_path.parents[1]
    rebuilt_payloads = {
        "guitarset_v1.json": build_position_payload(
            data_home=data_home,
            validation_player=FINAL_PLAYER,
        ),
        "guitarset_seq_v1.json": build_sequence_payload(
            data_home=data_home,
            validation_player=FINAL_PLAYER,
            scheme="delta_fret",
            alpha=0.5,
            backoff_kappa=8.0,
            singleton_only=True,
        ),
    }
    artifacts = []
    for filename, source_commit, construction_command, constants in (
        (
            "guitarset_v1.json",
            "936a5ccf2b4ecbb9d79bddb38c2d0115d471fc7b",
            (
                "python -m scripts.eval.build_guitarset_v1_prior --data-home "
                "<GUITARSET> --validation-player 05 --output <OUTPUT>"
            ),
            {"alpha": 1.0, "power": 2.0},
        ),
        (
            "guitarset_seq_v1.json",
            "e6244c5e132cbc8b2816da67b8d70a73df6a8e5d",
            (
                "python -m scripts.eval.build_guitarset_seq_v1_prior --data-home "
                "<GUITARSET> --validation-player 05 --scheme delta_fret --alpha 0.5 "
                "--backoff-kappa 8.0 --output <OUTPUT>"
            ),
            {"scheme": "delta_fret", "alpha": 0.5, "backoff_kappa": 8.0, "singleton_only": True},
        ),
    ):
        artifact = priors_dir / filename
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload != rebuilt_payloads[filename]:
            raise RuntimeError(f"checked-in {filename} does not reproduce from players 00-04")
        artifacts.append(
            {
                "artifact": filename,
                "canonical_json_sha256": _canonical_json_hash(artifact),
                "source_commit": source_commit,
                "construction_command": construction_command,
                "constants": constants,
                "declared_validation_player": payload["validation_player"],
                "declared_training_tracks": payload["training_tracks"],
                "rebuild_status": "semantic and canonical-LF byte match in this run",
            }
        )
    payload = {
        "schema_version": 1,
        "benchmark_source": {
            "commit": benchmark_source_commit,
            "tracked_worktree_clean": source_worktree_clean,
            "script": str(Path(__file__).resolve().relative_to(repo_root)),
            "command": list(command),
        },
        "dataset": "GuitarSet",
        "dataset_version": "original public 360-track mono-mic/JAMS release",
        "dataset_reference": "Xi et al., ISMIR 2018",
        "license": "CC-BY-4.0",
        "split": {"training_players": list(DEV_PLAYERS), "held_out_player": FINAL_PLAYER},
        "training_tracks": len(training_tracks),
        "training_track_ids_sha256": ids_hash,
        "training_annotations_sha256": annotations_hash,
        "all_tracks": len(all_tracks),
        "all_track_ids_sha256": all_ids_hash,
        "all_annotations_sha256": all_annotations_hash,
        "audio_event_cache": _cache_provenance(all_tracks, backend_name),
        "artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "performance_observation": {
            "wall_seconds": round(runtime_seconds, 6),
            "peak_process_memory_bytes": peak_memory_bytes,
            "artifact_bytes": sum(int(item["bytes"]) for item in outputs.values()),
            "note": "observational metadata; excluded from deterministic report/CSV hashes",
        },
        "deterministic_outputs": dict(outputs),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _peak_process_memory_bytes() -> int:
    if os.name == "nt":
        output = subprocess.check_output(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64",
            ],
            text=True,
        )
        return int(output.strip())

    import resource  # noqa: PLC0415

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _checked_in_bundle(cfg: GuitarConfig) -> PriorBundle:
    position = load_pitch_position_prior("guitarset-v1", cfg=cfg)
    sequence = load_transition_prior("guitarset-seq-v1")
    return PriorBundle(
        position,
        sequence,
        {"solo": position, "comp": position},
        {"solo": sequence, "comp": sequence},
    )


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
    source_worktree_clean = not subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    ).strip()

    cfg = GuitarConfig()
    tracks = load_tracks(
        args.data_home.expanduser(), args.cache_dir.expanduser(), args.backend, cfg
    )
    dev_tracks = [track for track in tracks if track.player in DEV_PLAYERS]
    final_tracks = [track for track in tracks if track.player == FINAL_PLAYER]

    print("running leave-one-player-out development benchmark...", flush=True)
    dev_results = {name: ConditionResult(name) for name in CONDITIONS}
    for held_out in DEV_PLAYERS:
        train = [track for track in dev_tracks if track.player != held_out]
        validation = [track for track in dev_tracks if track.player == held_out]
        bundle = learn_bundle(train, cfg)
        print(f"  fold {held_out}: train={len(train)} validation={len(validation)}", flush=True)
        for name in CONDITIONS:
            fold = evaluate_condition(name, validation, bundle, cfg)
            target = dev_results[name]
            target.clip_scores.update(fold.clip_scores)
            target.clip_tab.update(fold.clip_tab)
            target.strata.update(fold.strata)
            target.note_rows.extend(fold.note_rows)

    print("freezing development choices; running player-05 confirmation...", flush=True)
    final_bundle = learn_bundle(dev_tracks, cfg)
    final_results = {
        name: evaluate_condition(
            name,
            final_tracks,
            final_bundle,
            cfg,
            keep_analyses=name == "production_equivalent",
        )
        for name in CONDITIONS
    }
    checked = evaluate_condition(
        "production_equivalent",
        final_tracks,
        _checked_in_bundle(cfg),
        cfg,
    )
    checked.name = "checked_in_production"
    for row in checked.note_rows:
        row["condition"] = checked.name
    final_results[checked.name] = checked

    oracle = _oracle_probe(
        final_tracks,
        final_results["production_equivalent"],
        final_bundle,
        cfg,
    )
    segment_oracles = _segment_oracles(
        final_tracks,
        final_results["production_equivalent"],
        cfg,
    )
    event_metrics = _baseline_event_metrics(
        final_tracks,
        final_results["production_equivalent"],
    )
    reproduced = _metric_summary(final_results["production_equivalent"])["macro_tab_f1"]
    if abs(reproduced - 0.6126) > 1e-4:
        raise RuntimeError(
            f"production baseline mismatch: expected 0.6126 within 1e-4, got {reproduced:.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"string_assignment_phase0_{args.date}"
    note_path = args.output_dir / f"{stem}_notes.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    provenance_path = args.output_dir / f"{stem}_provenance.json"
    report_path = args.output_dir / f"{stem}.md"
    all_results = [*dev_results.values(), *final_results.values()]
    _write_csv(note_path, [row for result in all_results for row in result.note_rows])
    _write_csv(summary_path, _aggregate_rows(all_results))
    report = _report(
        list(dev_results.values()),
        list(final_results.values()),
        oracle,
        segment_oracles,
        event_metrics,
        provenance_path,
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    outputs = {
        output.name: {
            "sha256": _file_sha256(output),
            "bytes": output.stat().st_size,
            "tracked": output != note_path,
        }
        for output in (report_path, summary_path, note_path)
    }
    command = [sys.executable, "-m", "scripts.eval.string_assignment_phase0", *sys.argv[1:]]
    _write_provenance(
        provenance_path,
        dev_tracks,
        tracks,
        backend_name=args.backend,
        command=command,
        outputs=outputs,
        runtime_seconds=time.perf_counter() - started_at,
        peak_memory_bytes=_peak_process_memory_bytes(),
        source_worktree_clean=source_worktree_clean,
    )
    print(f"wrote {report_path}")
    print(f"wrote {note_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
