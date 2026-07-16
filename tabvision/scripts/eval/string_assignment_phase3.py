"""Evaluate raw high-resolution uncertainty and a fixed checkpoint ensemble.

Development is five-fold leave-one-player-out over GuitarSet players 00--04.
The production GAPS stream is immutable: posterior extraction is side-channel
only, and player 05 is opened only when a development gate passes.

Run from ``tabvision/``::

    python -m scripts.eval.string_assignment_phase3 \
        --data-home ~/.tabvision/data/guitarset \
        --output-dir ../docs/EVAL_REPORTS
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from scripts.eval.string_assignment_phase0 import (
    DEFAULT_CACHE,
    DEV_PLAYERS,
    FINAL_PLAYER,
    SEQUENCE_WEIGHT,
    ConditionResult,
    Track,
    _bootstrap,
    _file_sha256,
    learn_bundle,
    load_tracks,
)
from tabvision.audio.checkpoint_ensemble import (
    CheckpointAlignment,
    align_checkpoints,
    intersection_events,
    select_events,
    union_events,
)
from tabvision.audio.highres import (
    DEFAULT_HF_REPO,
    HIGHRES_BEGIN_NOTE,
    HIGHRES_FRAMES_PER_SECOND,
    HighResBackend,
    HighResPosteriors,
)
from tabvision.audio.highres_cache import (
    HighResCacheError,
    HighResCacheRecord,
    read_highres_cache,
    write_highres_cache,
)
from tabvision.demux import demux
from tabvision.eval.metrics import TabF1Result, event_f1, tab_f1
from tabvision.eval.string_assignment import decode_with_analysis, label_prediction_matches
from tabvision.fusion import playability
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import apply_pitch_position_prior
from tabvision.types import AudioEvent, GuitarConfig, TabEvent

CHECKPOINTS = ("guitar_gaps", "guitar_fl")
CONDITIONS = (
    "baseline",
    "gaps",
    "fl",
    "union",
    "intersection",
    "confidence_winner",
    "logistic",
)
ENSEMBLE_CANDIDATES = ("fl", "union", "intersection", "confidence_winner", "logistic")
CHECKPOINT_FILES = {"guitar_gaps": "guitar-gaps.pth", "guitar_fl": "guitar-fl.pth"}
DEFAULT_POSTERIOR_CACHE = Path.home() / ".tabvision/cache/string_assignment_phase3"
MATCH_TOLERANCE_S = 0.05
SELECTOR_THRESHOLD = 0.5
LOGISTIC_L2 = 1.0
LOGISTIC_ITERATIONS = 50
ALT_MIN_PROBABILITY = 0.20
ALT_MIN_RATIO = 0.50
ONSET_SUBTHRESHOLD_FLOOR = 0.10
FRAME_SUBTHRESHOLD_FLOOR = 0.05
CURRENT_PIPELINE_SECONDS_PER_60S = 45.0
FROZEN_ENSEMBLE_WINNER = "confidence_winner"

Source = Literal["gaps", "fl"]


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def probability(self, features: np.ndarray) -> float:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        value = float(self.weights[0] + standardized @ self.weights[1:])
        return _sigmoid(value)


@dataclass(frozen=True)
class FoldModels:
    gaps_calibrator: LogisticModel
    fl_calibrator: LogisticModel
    combiner: LogisticModel


@dataclass(frozen=True)
class Phase3TrackData:
    track: Track
    gaps_events: tuple[AudioEvent, ...]
    fl_events: tuple[AudioEvent, ...]
    alignment: CheckpointAlignment
    gaps_features: tuple[np.ndarray, ...]
    fl_features: tuple[np.ndarray, ...]
    gaps_labels: tuple[bool, ...]
    fl_labels: tuple[bool, ...]


@dataclass
class OracleAggregate:
    current_errors: int = 0
    pitch_off_errors: int = 0
    missed_errors: int = 0
    top2_recoverable: int = 0
    top3_recoverable: int = 0
    lattice_recoverable: int = 0
    correct_events: int = 0
    extra_false_candidates: int = 0
    missed_subthreshold_onset: int = 0
    missed_subthreshold_frame: int = 0
    pitch_entropy_rows: list[tuple[float, bool]] = field(default_factory=list)
    string_entropy_rows: list[tuple[float, bool]] = field(default_factory=list)

    def merge(self, other: OracleAggregate) -> None:
        for key in (
            "current_errors",
            "pitch_off_errors",
            "missed_errors",
            "top2_recoverable",
            "top3_recoverable",
            "lattice_recoverable",
            "correct_events",
            "extra_false_candidates",
            "missed_subthreshold_onset",
            "missed_subthreshold_frame",
        ):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.pitch_entropy_rows.extend(other.pitch_entropy_rows)
        self.string_entropy_rows.extend(other.string_entropy_rows)


@dataclass(frozen=True)
class CacheStats:
    checkpoint: str
    tracks: int
    source_duration_s: float
    inference_wall_s: float
    new_tracks: int
    cache_bytes: int

    @property
    def seconds_per_60s(self) -> float:
        if not self.source_duration_s:
            return float("nan")
        return self.inference_wall_s * 60.0 / self.source_duration_s


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fit_logistic(features: Sequence[np.ndarray], labels: Sequence[bool]) -> LogisticModel:
    """Fit the one fixed L2 logistic model with deterministic Newton steps."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("logistic training data must be a non-empty feature matrix")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1.0e-8] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = float(np.clip(np.mean(y), 1.0e-5, 1.0 - 1.0e-5))
    weights[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(design.shape[1], dtype=np.float64) * LOGISTIC_L2
    penalty[0, 0] = 0.0
    for _ in range(LOGISTIC_ITERATIONS):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        variance = np.maximum(probabilities * (1.0 - probabilities), 1.0e-6)
        gradient = design.T @ (probabilities - y) + penalty @ weights
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1.0e-8:
            break
    return LogisticModel(mean, scale, weights)


def _event_matches(
    predicted: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
) -> tuple[tuple[int | None, ...], tuple[bool, ...]]:
    gold_used = [False] * len(gold)
    matched: list[int | None] = []
    order = sorted(range(len(predicted)), key=lambda index: predicted[index].onset_s)
    matches_by_index: list[int | None] = [None] * len(predicted)
    for index in order:
        event = predicted[index]
        candidates = [
            (abs(event.onset_s - target.onset_s), gold_index)
            for gold_index, target in enumerate(gold)
            if not gold_used[gold_index]
            and target.pitch_midi == event.pitch_midi
            and abs(event.onset_s - target.onset_s) <= MATCH_TOLERANCE_S + 1.0e-9
        ]
        if candidates:
            _delta, gold_index = min(candidates)
            gold_used[gold_index] = True
            matches_by_index[index] = gold_index
    matched.extend(matches_by_index)
    return tuple(matched), tuple(gold_used)


def _posterior_window(posteriors: HighResPosteriors, onset_s: float) -> np.ndarray:
    center = min(
        posteriors.frame_count - 1,
        int(round(max(0.0, onset_s) * posteriors.frames_per_second)),
    )
    start = max(0, center - 1)
    stop = min(posteriors.frame_count, center + 2)
    return np.max(posteriors.reg_onset_output[start:stop], axis=0)


def _entropy_and_margin(probabilities: np.ndarray) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        entropy = 0.0
    else:
        normalized = np.clip(values / total, 1.0e-12, 1.0)
        entropy = float(-np.sum(normalized * np.log(normalized)) / math.log(len(values)))
    ordered = np.sort(values)
    margin = float(ordered[-1] - ordered[-2])
    return entropy, margin


def _feature_rows(
    source: Source,
    events: Sequence[AudioEvent],
    posteriors: HighResPosteriors,
    alignment: CheckpointAlignment,
) -> tuple[np.ndarray, ...]:
    agreements = {
        match.gaps_index if source == "gaps" else match.fl_index for match in alignment.agreements
    }
    disagreements = {
        match.gaps_index if source == "gaps" else match.fl_index
        for match in alignment.disagreements
    }
    rows: list[np.ndarray] = []
    for index, event in enumerate(events):
        onset_probability, frame_probability = posteriors.event_scores(
            event.onset_s, event.pitch_midi
        )
        probabilities = _posterior_window(posteriors, event.onset_s)
        entropy, margin = _entropy_and_margin(probabilities)
        rows.append(
            np.asarray(
                (
                    onset_probability,
                    frame_probability,
                    min(max(event.offset_s - event.onset_s, 0.0), 2.0) / 2.0,
                    event.velocity,
                    event.pitch_midi / 127.0,
                    float(source == "fl"),
                    float(index in agreements),
                    float(index in disagreements),
                    entropy,
                    margin,
                    float(np.max(probabilities)),
                ),
                dtype=np.float64,
            )
        )
    return tuple(rows)


def _stream_fields(event: AudioEvent) -> tuple[Any, ...]:
    return (
        event.onset_s,
        event.offset_s,
        event.pitch_midi,
        event.velocity,
        event.confidence,
        event.tags,
    )


def _track_data(
    track: Track,
    gaps: HighResCacheRecord,
    fl: HighResCacheRecord,
) -> tuple[Phase3TrackData, OracleAggregate]:
    if gaps.posteriors is None or fl.posteriors is None:
        raise RuntimeError(f"posterior cache missing tensors for {track.track_id}")
    baseline_fields = tuple(_stream_fields(event) for event in track.raw_events)
    gaps_fields = tuple(_stream_fields(event) for event in gaps.events)
    if baseline_fields != gaps_fields:
        raise RuntimeError(f"GAPS posterior pass changed the frozen stream for {track.track_id}")
    alignment = align_checkpoints(gaps.events, fl.events)
    gaps_matches, _gold_used = _event_matches(gaps.events, track.gold)
    fl_matches, _ = _event_matches(fl.events, track.gold)
    data = Phase3TrackData(
        track=track,
        gaps_events=gaps.events,
        fl_events=fl.events,
        alignment=alignment,
        gaps_features=_feature_rows("gaps", gaps.events, gaps.posteriors, alignment),
        fl_features=_feature_rows("fl", fl.events, fl.posteriors, alignment),
        gaps_labels=tuple(index is not None for index in gaps_matches),
        fl_labels=tuple(index is not None for index in fl_matches),
    )
    return data, _oracle_for_track(track, gaps, gaps_matches)


def _oracle_for_track(
    track: Track,
    gaps: HighResCacheRecord,
    prediction_matches: Sequence[int | None],
) -> OracleAggregate:
    assert gaps.posteriors is not None
    aggregate = OracleAggregate()
    matched_gold = {index for index in prediction_matches if index is not None}
    for predicted_index, gold_index in enumerate(prediction_matches):
        probabilities = _posterior_window(gaps.posteriors, gaps.events[predicted_index].onset_s)
        entropy, _margin = _entropy_and_margin(probabilities)
        aggregate.pitch_entropy_rows.append((entropy, gold_index is None))
        if gold_index is None:
            continue
        aggregate.correct_events += 1
        ranks = np.argsort(probabilities)[::-1]
        emitted_class = gaps.events[predicted_index].pitch_midi - HIGHRES_BEGIN_NOTE
        alternate_index = next(int(index) for index in ranks if int(index) != emitted_class)
        alternate_probability = float(probabilities[alternate_index])
        best_probability = max(float(probabilities[int(ranks[0])]), 1.0e-9)
        alternate_pitch = alternate_index + HIGHRES_BEGIN_NOTE
        if (
            alternate_pitch != track.gold[gold_index].pitch_midi
            and alternate_probability >= ALT_MIN_PROBABILITY
            and alternate_probability / best_probability >= ALT_MIN_RATIO
        ):
            aggregate.extra_false_candidates += 1
    for gold_index, gold_event in enumerate(track.gold):
        if gold_index in matched_gold:
            continue
        aggregate.current_errors += 1
        nearby_wrong_pitch = any(
            abs(event.onset_s - gold_event.onset_s) <= MATCH_TOLERANCE_S + 1.0e-9
            for event in gaps.events
        )
        if nearby_wrong_pitch:
            aggregate.pitch_off_errors += 1
        else:
            aggregate.missed_errors += 1
        probabilities = _posterior_window(gaps.posteriors, gold_event.onset_s)
        ranks = np.argsort(probabilities)[::-1]
        gold_class = gold_event.pitch_midi - HIGHRES_BEGIN_NOTE
        if not 0 <= gold_class < len(probabilities):
            continue
        top2 = {int(index) for index in ranks[:2]}
        top3 = {int(index) for index in ranks[:3]}
        aggregate.top2_recoverable += int(gold_class in top2)
        aggregate.top3_recoverable += int(gold_class in top3)
        second_probability = float(probabilities[int(ranks[1])])
        best_probability = max(float(probabilities[int(ranks[0])]), 1.0e-9)
        aggregate.lattice_recoverable += int(
            gold_class in top2
            and second_probability >= ALT_MIN_PROBABILITY
            and second_probability / best_probability >= ALT_MIN_RATIO
        )
        if not nearby_wrong_pitch:
            onset_value = float(probabilities[gold_class])
            _onset, frame_value = gaps.posteriors.event_scores(
                gold_event.onset_s, gold_event.pitch_midi
            )
            aggregate.missed_subthreshold_onset += int(
                ONSET_SUBTHRESHOLD_FLOOR <= onset_value < 0.3
            )
            aggregate.missed_subthreshold_frame += int(
                FRAME_SUBTHRESHOLD_FLOOR <= frame_value < 0.1
            )
    return aggregate


def _fit_fold_models(data: Sequence[Phase3TrackData]) -> FoldModels:
    source_features: dict[Source, list[np.ndarray]] = {"gaps": [], "fl": []}
    source_labels: dict[Source, list[bool]] = {"gaps": [], "fl": []}
    combined_features: list[np.ndarray] = []
    combined_labels: list[bool] = []
    for item in data:
        agreed_gaps = {match.gaps_index for match in item.alignment.agreements}
        agreed_fl = {match.fl_index for match in item.alignment.agreements}
        sources: tuple[Source, Source] = ("gaps", "fl")
        for source in sources:
            if source == "gaps":
                features, labels, agreed = item.gaps_features, item.gaps_labels, agreed_gaps
            else:
                features, labels, agreed = item.fl_features, item.fl_labels, agreed_fl
            for index, (row, label) in enumerate(zip(features, labels, strict=True)):
                if index in agreed:
                    continue
                source_features[source].append(row[:1])
                source_labels[source].append(label)
                combined_features.append(row)
                combined_labels.append(label)
    return FoldModels(
        gaps_calibrator=fit_logistic(source_features["gaps"], source_labels["gaps"]),
        fl_calibrator=fit_logistic(source_features["fl"], source_labels["fl"]),
        combiner=fit_logistic(combined_features, combined_labels),
    )


def _variants(item: Phase3TrackData, models: FoldModels) -> dict[str, tuple[AudioEvent, ...]]:
    confidence_models = {"gaps": models.gaps_calibrator, "fl": models.fl_calibrator}
    feature_maps = {"gaps": item.gaps_features, "fl": item.fl_features}

    def confidence_score(source: Source, index: int, _event: AudioEvent) -> float:
        return confidence_models[source].probability(feature_maps[source][index][:1])

    def logistic_score(source: Source, index: int, _event: AudioEvent) -> float:
        return models.combiner.probability(feature_maps[source][index])

    return {
        "baseline": item.track.raw_events,
        "gaps": item.gaps_events,
        "fl": item.fl_events,
        "union": union_events(item.gaps_events, item.fl_events),
        "intersection": intersection_events(item.gaps_events, item.fl_events),
        "confidence_winner": select_events(
            item.gaps_events,
            item.fl_events,
            score=confidence_score,
            threshold=SELECTOR_THRESHOLD,
        ),
        "logistic": select_events(
            item.gaps_events,
            item.fl_events,
            score=logistic_score,
            threshold=SELECTOR_THRESHOLD,
        ),
    }


def _merge_result(target: ConditionResult, source: ConditionResult) -> None:
    target.clip_scores.update(source.clip_scores)
    target.clip_tab.update(source.clip_tab)
    target.strata.update(source.strata)
    target.note_rows.extend(source.note_rows)


def _diagnostic_rows(
    condition: str,
    item: Phase3TrackData,
    predicted: Sequence[TabEvent],
    cfg: GuitarConfig,
) -> list[dict[str, Any]]:
    rows = []
    for match in label_prediction_matches(predicted, item.track.gold):
        event = predicted[match.predicted_index]
        gold = item.track.gold[match.gold_index] if match.gold_index is not None else None
        candidates = candidate_positions(event.pitch_midi, cfg)
        reference_rank: int | str = ""
        if gold is not None and event.pitch_midi == gold.pitch_midi:
            reference_rank = next(
                (
                    index
                    for index, candidate in enumerate(candidates, start=1)
                    if (candidate.string_idx, candidate.fret) == (gold.string_idx, gold.fret)
                ),
                "",
            )
        rows.append(
            {
                "condition": condition,
                "track_id": item.track.track_id,
                "player": item.track.player,
                "mode": item.track.mode,
                "style": item.track.style,
                "pitch_midi": event.pitch_midi,
                "candidate_count": len(candidates),
                "reference_candidate_rank": reference_rank,
                "predicted_string": event.string_idx,
                "predicted_fret": event.fret,
                "reference_string": "" if gold is None else gold.string_idx,
                "reference_fret": "" if gold is None else gold.fret,
                "string_displacement": ("" if gold is None else event.string_idx - gold.string_idx),
                "fret_displacement": "" if gold is None else event.fret - gold.fret,
                "label": match.label,
            }
        )
    return rows


def _evaluate_fold(
    held_out: Sequence[Phase3TrackData],
    train: Sequence[Phase3TrackData],
    cfg: GuitarConfig,
    oracle: OracleAggregate,
) -> tuple[dict[str, ConditionResult], FoldModels]:
    models = _fit_fold_models(train)
    bundle = learn_bundle([item.track for item in train], cfg)
    results = {name: ConditionResult(name) for name in CONDITIONS}
    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        for item in held_out:
            variants = _variants(item, models)
            for name, raw_events in variants.items():
                events = apply_pitch_position_prior(raw_events, bundle.global_position)
                analysis = decode_with_analysis(events, cfg=cfg)
                predicted = analysis.paths[0].events
                score = tab_f1(predicted, item.track.gold)
                result = results[name]
                result.clip_scores[item.track.track_id] = score.f1
                result.clip_tab[item.track.track_id] = score
                result.strata[item.track.track_id] = f"{item.track.player}|{item.track.mode}"
                result.note_rows.extend(_diagnostic_rows(name, item, predicted, cfg))
                if name == "baseline":
                    labels = label_prediction_matches(predicted, item.track.gold)
                    for match in labels:
                        if match.label not in {"correct", "wrong_position_same_pitch"}:
                            continue
                        entropy = float(item.gaps_features[match.predicted_index][8])
                        oracle.string_entropy_rows.append(
                            (entropy, match.label == "wrong_position_same_pitch")
                        )
    finally:
        playability.set_transition_prior(None)
    return results, models


def _evaluate_confirmation(
    confirmation: Sequence[Phase3TrackData],
    development: Sequence[Phase3TrackData],
    cfg: GuitarConfig,
) -> tuple[
    dict[str, ConditionResult],
    dict[str, dict[str, Sequence[AudioEvent]]],
    FoldModels,
]:
    models = _fit_fold_models(development)
    bundle = learn_bundle([item.track for item in development], cfg)
    names = ("baseline", FROZEN_ENSEMBLE_WINNER)
    results = {name: ConditionResult(name) for name in names}
    raw_events: dict[str, dict[str, Sequence[AudioEvent]]] = {name: {} for name in names}
    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        for item in confirmation:
            variants = _variants(item, models)
            for name in names:
                raw = variants[name]
                predicted = (
                    decode_with_analysis(
                        apply_pitch_position_prior(raw, bundle.global_position),
                        cfg=cfg,
                    )
                    .paths[0]
                    .events
                )
                score = tab_f1(predicted, item.track.gold)
                result = results[name]
                result.clip_scores[item.track.track_id] = score.f1
                result.clip_tab[item.track.track_id] = score
                result.strata[item.track.track_id] = f"{item.track.player}|{item.track.mode}"
                result.note_rows.extend(_diagnostic_rows(name, item, predicted, cfg))
                raw_events[name][item.track.track_id] = raw
    finally:
        playability.set_transition_prior(None)
    return results, raw_events, models


def _sum_tab(results: Iterable[TabF1Result]) -> TabF1Result:
    items = list(results)
    tp = sum(item.true_positives for item in items)
    fp = sum(item.false_positives for item in items)
    fn = sum(item.false_negatives for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def _condition_rows(
    results: Mapping[str, ConditionResult],
    raw_events: Mapping[str, Mapping[str, Sequence[AudioEvent]]],
    tracks: Mapping[str, Track],
    *,
    names: Sequence[str] = CONDITIONS,
) -> list[dict[str, Any]]:
    baseline = results["baseline"]
    rows = []
    for name in names:
        result = results[name]
        tab_micro = _sum_tab(result.clip_tab.values())
        onset_f1 = _mean(
            event_f1(
                _audio_as_tab(raw_events[name][track_id]),
                tracks[track_id].gold,
                match_pitch=False,
            ).f1
            for track_id in result.clip_scores
        )
        pitch_f1 = _mean(
            event_f1(
                _audio_as_tab(raw_events[name][track_id]),
                tracks[track_id].gold,
                match_pitch=True,
            ).f1
            for track_id in result.clip_scores
        )
        delta, lower, upper = _bootstrap(baseline, result)
        player_deltas = {
            player: _mean(
                result.clip_scores[track_id] - baseline.clip_scores[track_id]
                for track_id in result.clip_scores
                if track_id.startswith(f"{player}_")
            )
            for player in DEV_PLAYERS
            if any(track_id.startswith(f"{player}_") for track_id in result.clip_scores)
        }
        solo_ids = [track_id for track_id in result.clip_scores if track_id.endswith("_solo")]
        comp_ids = [track_id for track_id in result.clip_scores if track_id.endswith("_comp")]
        ambiguous = [
            row
            for row in result.note_rows
            if row["label"] in {"correct", "wrong_position_same_pitch"}
            and int(row["candidate_count"]) > 1
        ]
        correct = sum(row["label"] == "correct" for row in ambiguous)
        top3 = sum(
            isinstance(row["reference_candidate_rank"], int)
            and int(row["reference_candidate_rank"]) <= 3
            for row in ambiguous
        )
        wrong_position = sum(row["label"] == "wrong_position_same_pitch" for row in ambiguous)
        rows.append(
            {
                "condition": name,
                "macro_tab_f1": _mean(result.clip_scores.values()),
                "solo_tab_f1": _mean(result.clip_scores[track_id] for track_id in solo_ids),
                "comp_tab_f1": _mean(result.clip_scores[track_id] for track_id in comp_ids),
                "micro_tab_f1": tab_micro.f1,
                "onset_f1": onset_f1,
                "pitch_f1": pitch_f1,
                "ambiguous_top1": correct / len(ambiguous) if ambiguous else float("nan"),
                "ambiguous_top3": top3 / len(ambiguous) if ambiguous else float("nan"),
                "same_pitch_wrong": wrong_position,
                "same_pitch_total": len(ambiguous),
                "same_pitch_wrong_rate": (
                    wrong_position / len(ambiguous) if ambiguous else float("nan")
                ),
                "delta": delta,
                "ci_lower": lower,
                "ci_upper": upper,
                "worst_player_delta": min(player_deltas.values()) if player_deltas else delta,
                **{f"player_{player}_delta": value for player, value in player_deltas.items()},
            }
        )
    return rows


def _audio_as_tab(events: Sequence[AudioEvent]) -> tuple[TabEvent, ...]:
    return tuple(
        TabEvent(
            onset_s=event.onset_s,
            duration_s=max(0.0, event.offset_s - event.onset_s),
            string_idx=0,
            fret=0,
            pitch_midi=event.pitch_midi,
            confidence=event.confidence,
        )
        for event in events
    )


def _error_summary_rows(
    results: Mapping[str, ConditionResult],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    dimensions = (
        "player",
        "mode",
        "style",
        "pitch_midi",
        "candidate_count",
        "reference_string",
        "string_displacement",
        "fret_displacement",
    )
    rows = []
    for name in names:
        diagnostics = results[name].note_rows
        for dimension in dimensions:
            values: dict[tuple[str, str], int] = {}
            for row in diagnostics:
                if row["label"] == "correct":
                    continue
                key = (str(row[dimension]), str(row["label"]))
                values[key] = values.get(key, 0) + 1
            for (value, label), count in sorted(values.items()):
                rows.append(
                    {
                        "condition": name,
                        "dimension": dimension,
                        "value": value,
                        "label": label,
                        "count": count,
                    }
                )
    return rows


def _auc(rows: Sequence[tuple[float, bool]]) -> float:
    positives = [score for score, label in rows if label]
    negatives = [score for score, label in rows if not label]
    if not positives or not negatives:
        return float("nan")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            favorable += float(positive > negative) + 0.5 * float(positive == negative)
    return favorable / (len(positives) * len(negatives))


def _source_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _cache_path(cache_dir: Path, checkpoint: str, track: Track) -> Path:
    return cache_dir / checkpoint / f"{track.track_id}.npz"


def _checkpoint_path(checkpoint: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DEFAULT_HF_REPO, CHECKPOINT_FILES[checkpoint]))


def _base_provenance(
    track: Track,
    checkpoint: str,
    *,
    checkpoint_sha256: str,
    demux_sha256: str,
    ffmpeg_path: Path,
    ffmpeg_sha256: str,
) -> dict[str, str]:
    media = track.media_path.resolve()
    return {
        "track_id": track.track_id,
        "source_path": str(media),
        "source_size": str(media.stat().st_size),
        "source_mtime_ns": str(media.stat().st_mtime_ns),
        "source_sha256": _file_sha256(media),
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "demux_sha256": demux_sha256,
        "ffmpeg_path": str(ffmpeg_path),
        "ffmpeg_sha256": ffmpeg_sha256,
        "hf_midi_transcription": importlib.metadata.version("hf-midi-transcription"),
        "frames_per_second": str(HIGHRES_FRAMES_PER_SECOND),
        "begin_note": str(HIGHRES_BEGIN_NOTE),
    }


def ensure_caches(
    tracks: Sequence[Track],
    checkpoint: str,
    cache_dir: Path,
) -> CacheStats:
    ffmpeg_path = _resolve_ffmpeg()
    checkpoint_file = _checkpoint_path(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint_file)
    demux_sha256 = _file_sha256(Path(sys.modules[demux.__module__].__file__ or ""))
    backend: HighResBackend | None = None
    total_duration = 0.0
    total_wall = 0.0
    new_tracks = 0
    cache_bytes = 0
    for index, track in enumerate(tracks, start=1):
        expected = _base_provenance(
            track,
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            demux_sha256=demux_sha256,
            ffmpeg_path=ffmpeg_path,
            ffmpeg_sha256=_file_sha256(ffmpeg_path),
        )
        path = _cache_path(cache_dir, checkpoint, track)
        record: HighResCacheRecord | None = None
        try:
            record = read_highres_cache(path, expected_provenance=expected)
        except HighResCacheError:
            pass
        duration_s = _source_duration_s(track.media_path)
        if record is None:
            if backend is None:
                backend = HighResBackend(checkpoint=checkpoint)
                backend._load_model()
            demuxed = demux(str(track.media_path))
            started = time.perf_counter()
            result = backend.transcribe_with_posteriors(
                demuxed.wav,
                demuxed.sample_rate,
                _session_for_track(track),
            )
            wall_s = time.perf_counter() - started
            provenance = {
                **expected,
                "source_duration_s": f"{duration_s:.9f}",
                "inference_wall_s": f"{wall_s:.9f}",
            }
            write_highres_cache(path, result, provenance=provenance)
            record = read_highres_cache(path, expected_provenance=expected)
            new_tracks += 1
            print(
                f"  {checkpoint}: {index}/{len(tracks)} {track.track_id} "
                f"{len(record.events)} events in {wall_s:.1f}s",
                flush=True,
            )
        total_duration += float(record.provenance.get("source_duration_s", duration_s))
        total_wall += float(record.provenance.get("inference_wall_s", "nan"))
        cache_bytes += path.stat().st_size
        if index % 20 == 0 and not new_tracks:
            print(f"  {checkpoint}: validated {index}/{len(tracks)} caches", flush=True)
    return CacheStats(checkpoint, len(tracks), total_duration, total_wall, new_tracks, cache_bytes)


def _resolve_ffmpeg() -> Path:
    configured = shutil.which("ffmpeg")
    if configured is not None:
        return Path(configured).resolve()
    managed = Path.home() / ".tabvision/tools/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    if not managed.is_file():
        raise RuntimeError("ffmpeg is neither on PATH nor installed under ~/.tabvision/tools")
    os.environ["PATH"] = str(managed.parent) + os.pathsep + os.environ.get("PATH", "")
    return managed.resolve()


def _session_for_track(track: Track):  # type: ignore[no-untyped-def]
    from tabvision.types import SessionConfig

    return SessionConfig(style="fingerstyle" if track.mode == "solo" else "strumming")


def _load_phase3_data(
    tracks: Sequence[Track],
    cache_dir: Path,
) -> tuple[list[Phase3TrackData], OracleAggregate]:
    data = []
    oracle = OracleAggregate()
    for index, track in enumerate(tracks, start=1):
        gaps = read_highres_cache(_cache_path(cache_dir, "guitar_gaps", track))
        fl = read_highres_cache(_cache_path(cache_dir, "guitar_fl", track))
        item, track_oracle = _track_data(track, gaps, fl)
        data.append(item)
        oracle.merge(track_oracle)
        if index % 20 == 0:
            print(f"  posterior features: {index}/{len(tracks)}", flush=True)
    return data, oracle


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(
    rows: Sequence[Mapping[str, Any]],
    oracle: OracleAggregate,
    winner: Mapping[str, Any],
    posterior_gate: bool,
    ensemble_gate: bool,
    projected_runtime: float,
) -> str:
    lines = [
        "# String assignment Phase 3: high-resolution uncertainty and checkpoint ensemble",
        "",
        "## Development OOF results",
        "",
        "| condition | solo | comp | macro Tab F1 | delta [95% CI] | onset F1 | "
        "pitch F1 | ambiguous top-1 | top-3 | wrong rate | worst player |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['solo_tab_f1']:.4f} | "
            f"{row['comp_tab_f1']:.4f} | {row['macro_tab_f1']:.4f} | "
            f"{row['delta']:+.4f} [{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] | "
            f"{row['onset_f1']:.4f} | {row['pitch_f1']:.4f} | "
            f"{row['ambiguous_top1']:.4f} | {row['ambiguous_top3']:.4f} | "
            f"{row['same_pitch_wrong_rate']:.4f} | {row['worst_player_delta']:+.4f} |"
        )
    error_denominator = max(oracle.current_errors, 1)
    correct_denominator = max(oracle.correct_events, 1)
    lines.extend(
        [
            "",
            "## Posterior oracle and calibration",
            "",
            f"- Current pitch-off or missed gold events: `{oracle.current_errors}` "
            f"(pitch-off `{oracle.pitch_off_errors}`, missed `{oracle.missed_errors}`).",
            f"- Gold pitch in raw top 2: `{oracle.top2_recoverable / error_denominator:.1%}`; "
            f"top 3: `{oracle.top3_recoverable / error_denominator:.1%}`.",
            f"- Fixed two-hypothesis eligibility recovers "
            f"`{oracle.lattice_recoverable / error_denominator:.1%}` and adds "
            f"`{10.0 * oracle.extra_false_candidates / correct_denominator:.2f}` false "
            "candidates per ten correct events.",
            f"- Missed notes with subthreshold onset evidence: "
            f"`{oracle.missed_subthreshold_onset}/{oracle.missed_errors}`; frame evidence: "
            f"`{oracle.missed_subthreshold_frame}/{oracle.missed_errors}`.",
            f"- Posterior-entropy error AUC: pitch `{_auc(oracle.pitch_entropy_rows):.4f}`, "
            f"string `{_auc(oracle.string_entropy_rows):.4f}`.",
            "",
            "## Gate decision",
            "",
            f"- Posterior/lattice gate: **{'pass' if posterior_gate else 'fail'}**.",
            f"- Best checkpoint/ensemble condition: `{winner['condition']}` at "
            f"`{winner['delta']:+.4f}` with lower bound `{winner['ci_lower']:+.4f}`.",
            f"- Ensemble gate: **{'pass' if ensemble_gate else 'fail'}**.",
            f"- Projected two-checkpoint CPU pipeline runtime: `{projected_runtime:.2f}` "
            "seconds per 60-second clip.",
            "",
            (
                "No automatic event-stream change is authorized by this result."
                if not ensemble_gate
                else "The development ensemble gate passed; player-05 confirmation is required."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _confirmation_report(
    rows: Sequence[Mapping[str, Any]],
    safety_pass: bool,
    automatic_guardrails_pass: bool,
) -> str:
    baseline = next(row for row in rows if row["condition"] == "baseline")
    winner = next(row for row in rows if row["condition"] == FROZEN_ENSEMBLE_WINNER)
    wrong_reduction = 1.0 - float(winner["same_pitch_wrong_rate"]) / float(
        baseline["same_pitch_wrong_rate"]
    )
    return "\n".join(
        (
            "## Frozen player 05 confirmation",
            "",
            "Player 05 was opened only after `confidence_winner`, its features, and its "
            "0.5 threshold were frozen from players 00–04. No confirmation result was used "
            "for retuning.",
            "",
            "| condition | solo | comp | aggregate | micro | onset F1 | pitch F1 | "
            "ambiguous top-1 | top-3 | wrong rate | delta [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| `baseline` | {baseline['solo_tab_f1']:.4f} | {baseline['comp_tab_f1']:.4f} | "
            f"{baseline['macro_tab_f1']:.4f} | {baseline['micro_tab_f1']:.4f} | "
            f"{baseline['onset_f1']:.4f} | {baseline['pitch_f1']:.4f} | "
            f"{baseline['ambiguous_top1']:.4f} | {baseline['ambiguous_top3']:.4f} | "
            f"{baseline['same_pitch_wrong_rate']:.4f} | baseline |",
            f"| `{FROZEN_ENSEMBLE_WINNER}` | {winner['solo_tab_f1']:.4f} | "
            f"{winner['comp_tab_f1']:.4f} | {winner['macro_tab_f1']:.4f} | "
            f"{winner['micro_tab_f1']:.4f} | {winner['onset_f1']:.4f} | "
            f"{winner['pitch_f1']:.4f} | {winner['ambiguous_top1']:.4f} | "
            f"{winner['ambiguous_top3']:.4f} | {winner['same_pitch_wrong_rate']:.4f} | "
            f"{winner['delta']:+.4f} [{winner['ci_lower']:+.4f}, {winner['ci_upper']:+.4f}] |",
            "",
            f"Same-pitch wrong-position relative reduction: `{wrong_reduction:+.1%}`.",
            f"Phase 3 confirmation safety check: **{'pass' if safety_pass else 'fail'}**.",
            "Cumulative automatic promotion guardrails: "
            f"**{'pass' if automatic_guardrails_pass else 'fail'}**.",
            "The artifact is registered for explicit clean-acoustic evaluation; automatic "
            "routing remains on the production GAPS backend pending the integrated Phase 7 gate.",
            "",
        )
    )


def _logistic_payload(model: LogisticModel) -> dict[str, list[float]]:
    return {
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "weights": model.weights.tolist(),
    }


def _run_confirmation(
    args: argparse.Namespace,
    all_tracks: Sequence[Track],
    cfg: GuitarConfig,
) -> int:
    output_dir = args.output_dir.resolve()
    stem = "string_assignment_phase3_2026-07-15"
    provenance_path = output_dir / f"{stem}_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError("run the full development OOF evaluation before confirmation")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not provenance.get("ensemble_gate"):
        raise RuntimeError("development ensemble gate did not pass")
    if provenance.get("winner", {}).get("condition") != FROZEN_ENSEMBLE_WINNER:
        raise RuntimeError("frozen confirmation winner does not match development selection")
    development_tracks = [track for track in all_tracks if track.player in DEV_PLAYERS]
    confirmation_tracks = [track for track in all_tracks if track.player == FINAL_PLAYER]
    cache_stats = [
        ensure_caches(confirmation_tracks, checkpoint, args.posterior_cache_dir)
        for checkpoint in CHECKPOINTS
    ]
    development, _ = _load_phase3_data(development_tracks, args.posterior_cache_dir)
    confirmation, _ = _load_phase3_data(confirmation_tracks, args.posterior_cache_dir)
    results, raw_events, models = _evaluate_confirmation(confirmation, development, cfg)
    names = ("baseline", FROZEN_ENSEMBLE_WINNER)
    rows = _condition_rows(
        results,
        raw_events,
        {track.track_id: track for track in confirmation_tracks},
        names=names,
    )
    baseline = next(row for row in rows if row["condition"] == "baseline")
    winner = next(row for row in rows if row["condition"] == FROZEN_ENSEMBLE_WINNER)
    safety_pass = all(
        (
            float(winner["delta"]) > 0.0,
            float(winner["ci_lower"]) >= 0.0,
            float(winner["onset_f1"]) >= float(baseline["onset_f1"]),
            float(winner["pitch_f1"]) >= float(baseline["pitch_f1"]),
        )
    )
    wrong_reduction = 1.0 - float(winner["same_pitch_wrong_rate"]) / float(
        baseline["same_pitch_wrong_rate"]
    )
    automatic_guardrails_pass = all(
        (
            float(winner["solo_tab_f1"]) - float(baseline["solo_tab_f1"]) >= 0.03,
            float(winner["delta"]) >= 0.02,
            float(winner["ci_lower"]) > 0.0,
            wrong_reduction >= 0.10,
            float(winner["comp_tab_f1"]) - float(baseline["comp_tab_f1"]) >= -0.005,
            float(winner["onset_f1"]) >= float(baseline["onset_f1"]),
            float(winner["pitch_f1"]) >= float(baseline["pitch_f1"]),
        )
    )
    _write_csv(output_dir / f"{stem}_player05_conditions.csv", rows)
    _write_csv(
        output_dir / f"{stem}_player05_errors.csv",
        _error_summary_rows(results, names),
    )
    report_path = output_dir / f"{stem}.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.split("## Frozen player 05 confirmation", 1)[0].rstrip() + "\n\n"
    report += _confirmation_report(rows, safety_pass, automatic_guardrails_pass)
    report_path.write_text(report, encoding="utf-8")
    artifact_path = Path(__file__).resolve().parents[2] / "tabvision/audio/ensemble_v1.json"
    artifact = {
        "schema_version": 1,
        "registered": True,
        "automatic_activation": False,
        "winner": FROZEN_ENSEMBLE_WINNER,
        "threshold": SELECTOR_THRESHOLD,
        "match_tolerance_s": MATCH_TOLERANCE_S,
        "checkpoints": {"gaps": "guitar_gaps", "fl": "guitar_fl"},
        "calibrators": {
            "gaps": _logistic_payload(models.gaps_calibrator),
            "fl": _logistic_payload(models.fl_calibrator),
        },
        "training_split": "GuitarSet players 00-04",
        "confirmation_split": "GuitarSet player 05; not used for fitting",
        "validated_domain": {
            "instrument": "acoustic",
            "tone": "clean",
            "tuning_midi": [40, 45, 50, 55, 59, 64],
            "capo": 0,
        },
        "development_macro_tab_f1": provenance["winner"]["macro_tab_f1"],
        "development_delta": provenance["winner"]["delta"],
        "development_ci_lower": provenance["winner"]["ci_lower"],
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    provenance.update(
        {
            "player05_opened": True,
            "confirmation_cache_stats": [asdict(item) for item in cache_stats],
            "confirmation_rows": rows,
            "confirmation_safety_pass": safety_pass,
            "automatic_guardrails_pass": automatic_guardrails_pass,
            "confirmation_models": {
                "gaps_calibrator": _logistic_payload(models.gaps_calibrator),
                "fl_calibrator": _logistic_payload(models.fl_calibrator),
                "combiner": _logistic_payload(models.combiner),
            },
            "ensemble_artifact": str(artifact_path),
            "ensemble_artifact_sha256": _file_sha256(artifact_path),
            "phase3_decision": (
                "promote_confidence_winner" if safety_pass else "block_on_confirmation_safety"
            ),
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(_confirmation_report(rows, safety_pass, automatic_guardrails_pass), flush=True)
    return 0 if safety_pass else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", type=Path, default=Path.home() / ".tabvision/data/guitarset")
    parser.add_argument("--legacy-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--posterior-cache-dir", type=Path, default=DEFAULT_POSTERIOR_CACHE)
    parser.add_argument("--output-dir", type=Path, default=Path("../docs/EVAL_REPORTS"))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--confirm-player05", action="store_true")
    parser.add_argument("--max-tracks", type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = GuitarConfig()
    print("loading frozen GuitarSet event cache...", flush=True)
    all_tracks = load_tracks(args.data_home, args.legacy_cache_dir, "highres", cfg)
    if args.confirm_player05:
        return _run_confirmation(args, all_tracks, cfg)
    dev_tracks = [track for track in all_tracks if track.player in DEV_PLAYERS]
    if args.max_tracks is not None:
        dev_tracks = dev_tracks[: args.max_tracks]
    cache_stats = [
        ensure_caches(dev_tracks, checkpoint, args.posterior_cache_dir)
        for checkpoint in CHECKPOINTS
    ]
    for stats in cache_stats:
        print(
            f"{stats.checkpoint}: {stats.tracks} caches, {stats.cache_bytes} bytes, "
            f"{stats.seconds_per_60s:.2f}s inference/60s",
            flush=True,
        )
    if args.cache_only:
        return 0
    if len(dev_tracks) != 300:
        raise RuntimeError("full OOF evaluation requires all 300 development tracks")
    data, oracle = _load_phase3_data(dev_tracks, args.posterior_cache_dir)
    oof = {name: ConditionResult(name) for name in CONDITIONS}
    raw_events: dict[str, dict[str, Sequence[AudioEvent]]] = {name: {} for name in CONDITIONS}
    fold_models: dict[str, Any] = {}
    for player in DEV_PLAYERS:
        print(f"fold {player}: fitting fixed development-only calibrators", flush=True)
        train = [item for item in data if item.track.player != player]
        held = [item for item in data if item.track.player == player]
        fold_results, models = _evaluate_fold(held, train, cfg, oracle)
        fold_models[player] = {
            "gaps_calibrator": models.gaps_calibrator.weights.tolist(),
            "fl_calibrator": models.fl_calibrator.weights.tolist(),
            "combiner": models.combiner.weights.tolist(),
        }
        for name, result in fold_results.items():
            _merge_result(oof[name], result)
        for item in held:
            variants = _variants(item, models)
            for name, events in variants.items():
                raw_events[name][item.track.track_id] = events
    tracks_by_id = {track.track_id: track for track in dev_tracks}
    rows = _condition_rows(oof, raw_events, tracks_by_id)
    row_by_name = {str(row["condition"]): row for row in rows}
    winner = max(
        (row_by_name[name] for name in ENSEMBLE_CANDIDATES),
        key=lambda row: (
            float(row["macro_tab_f1"]),
            -ENSEMBLE_CANDIDATES.index(str(row["condition"])),
        ),
    )
    baseline = row_by_name["baseline"]
    false_per_ten = 10.0 * oracle.extra_false_candidates / max(oracle.correct_events, 1)
    posterior_gate = (
        oracle.lattice_recoverable / max(oracle.current_errors, 1) >= 0.25 and false_per_ten <= 1.0
    )
    fl_runtime = next(
        item.seconds_per_60s for item in cache_stats if item.checkpoint == "guitar_fl"
    )
    projected_runtime = CURRENT_PIPELINE_SECONDS_PER_60S + fl_runtime
    ensemble_gate = all(
        (
            float(winner["delta"]) >= 0.01,
            float(winner["ci_lower"]) > 0.0,
            float(winner["onset_f1"]) >= float(baseline["onset_f1"]),
            float(winner["pitch_f1"]) >= float(baseline["pitch_f1"]),
            projected_runtime < 300.0,
        )
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "string_assignment_phase3_2026-07-15"
    _write_csv(output_dir / f"{stem}_conditions.csv", rows)
    _write_csv(
        output_dir / f"{stem}_errors.csv",
        _error_summary_rows(oof, ("baseline", FROZEN_ENSEMBLE_WINNER)),
    )
    report = _report(rows, oracle, winner, posterior_gate, ensemble_gate, projected_runtime)
    (output_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    provenance = {
        "command": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "fixed_constants": {
            "checkpoints": CHECKPOINTS,
            "conditions": CONDITIONS,
            "match_tolerance_s": MATCH_TOLERANCE_S,
            "selector_threshold": SELECTOR_THRESHOLD,
            "logistic_l2": LOGISTIC_L2,
            "logistic_iterations": LOGISTIC_ITERATIONS,
            "alt_min_probability": ALT_MIN_PROBABILITY,
            "alt_min_ratio": ALT_MIN_RATIO,
        },
        "cache_stats": [asdict(item) for item in cache_stats],
        "fold_model_weights": fold_models,
        "oracle": {
            key: value for key, value in asdict(oracle).items() if not key.endswith("_rows")
        },
        "condition_rows": rows,
        "winner": winner,
        "posterior_gate": posterior_gate,
        "ensemble_gate": ensemble_gate,
        "projected_runtime_seconds_per_60s": projected_runtime,
        "player05_opened": False,
        "wall_seconds": time.perf_counter() - started,
    }
    provenance_path = output_dir / f"{stem}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(report, flush=True)
    if ensemble_gate:
        print(
            "development gate passed; player 05 remains unopened in this run and requires "
            "the frozen confirmation path",
            flush=True,
        )
        return 2
    if posterior_gate:
        print(
            "posterior gate passed; bounded two-hypothesis decoder evaluation is required",
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
