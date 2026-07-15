"""Train and evaluate the constrained Phase 2 contextual candidate reranker.

The experiment is fixed before execution: five player-held-out folds, one
masked linear control, one two-layer 64-wide Transformer, and four decoder
compositions.  Player 05 is evaluated only after the development decision and
final epoch counts are frozen.

Run from ``tabvision/`` with the project virtual environment::

    .venv/Scripts/python -m scripts.eval.string_assignment_phase2 \
        --data-home ~/.tabvision/data/guitarset \
        --output-dir ../docs/EVAL_REPORTS
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import wave
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from scripts.eval.string_assignment_phase0 import (
    DEFAULT_CACHE,
    DEV_PLAYERS,
    FINAL_PLAYER,
    SEQUENCE_WEIGHT,
    PriorBundle,
    Track,
    _cache_provenance,
    _file_sha256,
    _package_versions,
    _peak_process_memory_bytes,
    _source_hash,
    learn_bundle,
    load_tracks,
)
from tabvision.eval.metrics import TabF1Result, tab_f1
from tabvision.eval.string_assignment import (
    label_prediction_matches,
    paired_stratified_bootstrap,
)
from tabvision.fusion import chord, playability
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.context_reranker import (
    CANDIDATE_FEATURE_DIM,
    CONTEXT_OVERLAP_EVENTS,
    EVENT_FEATURE_DIM,
    MAX_CANDIDATES,
    MAX_CONTEXT_EVENTS,
    ContextFeatures,
    SegmentHint,
    apply_context_probabilities,
    build_context_features,
    context_windows,
    make_context_model,
    make_masked_linear_model,
    masked_softmax,
    merge_window_logits,
    parameter_count,
)
from tabvision.fusion.position_prior import apply_pitch_position_prior
from tabvision.fusion.segment_decoder import DEFAULT_SEGMENT_CONFIG, SegmentDecodeResult
from tabvision.fusion.viterbi import decode_segment_v1_with_analysis
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

CONTROL_MAX_EPOCHS = 12
CONTEXT_MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 16
CONTROL_LEARNING_RATE = 1.0e-2
CONTEXT_LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
TRANSITION_CONSISTENCY_WEIGHT = 0.05
BASE_SEED = 2715
CURRENT_PIPELINE_SECONDS_PER_60S = 45.0
FROZEN_ONSET_F1 = 0.9302
FROZEN_PITCH_F1 = 0.9154
ARTIFACT_NAME = "context-v1"


@dataclass(frozen=True)
class PreparedTrack:
    track: Track
    bundle: PriorBundle
    events: tuple[AudioEvent, ...]
    features: ContextFeatures
    labels: np.ndarray
    gold_strings: np.ndarray
    gold_frets: np.ndarray
    baseline: tuple[TabEvent, ...]
    segment: tuple[TabEvent, ...]
    baseline_top3: np.ndarray


@dataclass
class EvaluationResult:
    name: str
    clip_scores: dict[str, float] = field(default_factory=dict)
    clip_tab: dict[str, TabF1Result] = field(default_factory=dict)
    strata: dict[str, str] = field(default_factory=dict)
    ambiguous_correct: int = 0
    ambiguous_top3: int = 0
    ambiguous_total: int = 0
    error_rows: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: list[tuple[str, float, int, int, int]] = field(default_factory=list)

    def merge(self, other: EvaluationResult) -> None:
        self.clip_scores.update(other.clip_scores)
        self.clip_tab.update(other.clip_tab)
        self.strata.update(other.strata)
        self.ambiguous_correct += other.ambiguous_correct
        self.ambiguous_top3 += other.ambiguous_top3
        self.ambiguous_total += other.ambiguous_total
        self.error_rows.extend(other.error_rows)
        self.prediction_rows.extend(other.prediction_rows)


@dataclass(frozen=True)
class TrainOutcome:
    model: Any
    best_epoch: int
    best_validation_tab_f1: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class RuntimeResult:
    context_seconds: float
    source_duration_s: float
    added_seconds_per_60s: float
    projected_total_seconds_per_60s: float
    peak_memory_bytes: int
    prediction_sha256: str
    deterministic_rerun_sha256: str


def player_folds(
    tracks: Sequence[Track],
) -> tuple[tuple[str, tuple[Track, ...], tuple[Track, ...]], ...]:
    """Return the fixed folds and fail closed on any player leakage."""

    if {track.player for track in tracks} != set(DEV_PLAYERS):
        raise ValueError("development folds require exactly players 00-04")
    folds = []
    for held_out in DEV_PLAYERS:
        train = tuple(track for track in tracks if track.player != held_out)
        validation = tuple(track for track in tracks if track.player == held_out)
        train_players = {track.player for track in train}
        validation_players = {track.player for track in validation}
        if train_players & validation_players or validation_players != {held_out}:
            raise AssertionError("player-held-out split leaked")
        folds.append((held_out, train, validation))
    return tuple(folds)


def _session(track: Track) -> SessionConfig:
    return SessionConfig(style="fingerstyle" if track.mode == "solo" else "strumming")


def _segment_hints(decoded: SegmentDecodeResult) -> tuple[SegmentHint, ...]:
    if not decoded.paths:
        return ()
    clusters = chord.cluster_events(decoded.audio_events)
    hints: list[SegmentHint] = []
    states = decoded.paths[0].latent_states
    if len(states) != len(decoded.segments):
        raise AssertionError("segment state count drifted from segment boundaries")
    for boundary, state in zip(decoded.segments, states, strict=True):
        for cluster_index in range(boundary.start_cluster, boundary.end_cluster):
            hints.extend(
                SegmentHint(state.string_offset, state.zone_center) for _ in clusters[cluster_index]
            )
    if len(hints) != len(decoded.audio_events):
        raise AssertionError("segment hints drifted from decoded events")
    return tuple(hints)


def prepare_track(track: Track, bundle: PriorBundle, cfg: GuitarConfig) -> PreparedTrack:
    """Prepare one track using only the fold's training-derived priors."""

    from tabvision.eval.string_assignment import decode_with_analysis

    playability.set_transition_prior(bundle.global_sequence, SEQUENCE_WEIGHT)
    try:
        prior_events = apply_pitch_position_prior(track.raw_events, bundle.global_position)
        baseline_analysis = decode_with_analysis(prior_events, cfg=cfg)
        if not baseline_analysis.paths:
            raise RuntimeError(f"baseline decode failed for {track.track_id}")
        baseline = baseline_analysis.paths[0].events
        segment_decoded = decode_segment_v1_with_analysis(
            baseline_analysis.audio_events,
            cfg=cfg,
            config=DEFAULT_SEGMENT_CONFIG,
            k_paths=1,
            retain_analysis=False,
        )
        if not segment_decoded.paths:
            raise RuntimeError(f"segment decode failed for {track.track_id}")
        segment = segment_decoded.paths[0].events
        hints = _segment_hints(segment_decoded)
        features = build_context_features(
            baseline_analysis.audio_events,
            cfg=cfg,
            session=_session(track),
            baseline=baseline,
            segment_hints=hints,
        )
    finally:
        playability.set_transition_prior(None)

    labels = np.full(len(baseline), -100, dtype=np.int64)
    gold_strings = np.full(len(baseline), -1, dtype=np.int64)
    gold_frets = np.full(len(baseline), -1, dtype=np.int64)
    top3 = np.zeros(len(baseline), dtype=np.bool_)
    for match in label_prediction_matches(baseline, track.gold):
        if match.gold_index is None or match.label not in {"correct", "wrong_position_same_pitch"}:
            continue
        gold = track.gold[match.gold_index]
        candidates = features.candidates[match.predicted_index]
        label = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if (candidate.string_idx, candidate.fret) == (gold.string_idx, gold.fret)
            ),
            None,
        )
        if label is None:
            continue
        labels[match.predicted_index] = label
        gold_strings[match.predicted_index] = gold.string_idx
        gold_frets[match.predicted_index] = gold.fret
        ranks = baseline_analysis.candidate_ranks[match.predicted_index]
        top3[match.predicted_index] = any(
            (item.string_idx, item.fret) == (gold.string_idx, gold.fret) for item in ranks[:3]
        )
    return PreparedTrack(
        track,
        bundle,
        tuple(baseline_analysis.audio_events),
        features,
        labels,
        gold_strings,
        gold_frets,
        tuple(baseline),
        tuple(segment),
        top3,
    )


def prepare_tracks(
    tracks: Sequence[Track], bundle: PriorBundle, cfg: GuitarConfig, *, label: str
) -> list[PreparedTrack]:
    out = []
    for index, track in enumerate(tracks, start=1):
        out.append(prepare_track(track, bundle, cfg))
        if index % 30 == 0:
            print(f"  {label}: prepared {index}/{len(tracks)}", flush=True)
    return out


def _frequency_counts(tracks: Sequence[PreparedTrack]) -> Counter[tuple[str, int, int]]:
    counts: Counter[tuple[str, int, int]] = Counter()
    for prepared in tracks:
        candidate_counts = prepared.features.candidate_mask.sum(axis=1)
        for index in np.flatnonzero(prepared.labels >= 0):
            counts[
                (
                    prepared.track.player,
                    int(prepared.gold_strings[index]),
                    int(candidate_counts[index]),
                )
            ] += 1
    return counts


def _sample_weights(prepared: PreparedTrack, counts: Counter[tuple[str, int, int]]) -> np.ndarray:
    weights = np.zeros(len(prepared.labels), dtype=np.float32)
    candidate_counts = prepared.features.candidate_mask.sum(axis=1)
    for index in np.flatnonzero(prepared.labels >= 0):
        key = (
            prepared.track.player,
            int(prepared.gold_strings[index]),
            int(candidate_counts[index]),
        )
        weights[index] = 1.0 / counts[key]
    return weights


def _training_windows(
    tracks: Sequence[PreparedTrack],
) -> tuple[tuple[PreparedTrack, tuple[int, ...]], ...]:
    return tuple(
        (prepared, window)
        for prepared in tracks
        for window in context_windows(
            prepared.features.cluster_ids,
            max_events=MAX_CONTEXT_EVENTS,
            overlap_events=0,
        )
        if np.any(prepared.labels[list(window)] >= 0)
    )


def _batch(
    examples: Sequence[tuple[PreparedTrack, tuple[int, ...]]],
    counts: Counter[tuple[str, int, int]],
) -> tuple[torch.Tensor, ...]:
    batch_size = len(examples)
    max_length = max(len(indices) for _prepared, indices in examples)
    event = np.zeros((batch_size, max_length, EVENT_FEATURE_DIM), dtype=np.float32)
    candidate = np.zeros(
        (batch_size, max_length, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=np.float32
    )
    candidate_mask = np.zeros((batch_size, max_length, MAX_CANDIDATES), dtype=np.bool_)
    padding_mask = np.ones((batch_size, max_length), dtype=np.bool_)
    labels = np.full((batch_size, max_length), -100, dtype=np.int64)
    weights = np.zeros((batch_size, max_length), dtype=np.float32)
    gold_strings = np.full((batch_size, max_length), -1, dtype=np.float32)
    gold_frets = np.full((batch_size, max_length), -1, dtype=np.float32)
    for batch_index, (prepared, indices) in enumerate(examples):
        selected = np.asarray(indices, dtype=np.int64)
        length = len(selected)
        event[batch_index, :length] = prepared.features.event_features[selected]
        candidate[batch_index, :length] = prepared.features.candidate_features[selected]
        candidate_mask[batch_index, :length] = prepared.features.candidate_mask[selected]
        padding_mask[batch_index, :length] = False
        labels[batch_index, :length] = prepared.labels[selected]
        weights[batch_index, :length] = _sample_weights(prepared, counts)[selected]
        gold_strings[batch_index, :length] = prepared.gold_strings[selected]
        gold_frets[batch_index, :length] = prepared.gold_frets[selected]
    positive = weights > 0
    weights[positive] /= float(weights[positive].mean())
    return (
        torch.from_numpy(event),
        torch.from_numpy(candidate),
        torch.from_numpy(candidate_mask),
        torch.from_numpy(padding_mask),
        torch.from_numpy(labels),
        torch.from_numpy(weights),
        torch.from_numpy(gold_strings),
        torch.from_numpy(gold_frets),
    )


def _loss(
    logits: torch.Tensor,
    candidate_features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    gold_strings: torch.Tensor,
    gold_frets: torch.Tensor,
    *,
    transition_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_loss = functional.cross_entropy(
        logits.reshape(-1, MAX_CANDIDATES), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(labels)
    ce = (flat_loss * weights).sum() / weights.sum().clamp_min(1.0)
    transition = logits.new_zeros(())
    pair_mask = (labels[:, 1:] >= 0) & (labels[:, :-1] >= 0)
    if transition_weight > 0.0 and bool(pair_mask.any()):
        probabilities = torch.softmax(logits, dim=-1)
        string_numbers = torch.arange(6, dtype=logits.dtype, device=logits.device)
        candidate_strings = (candidate_features[..., :6] * string_numbers).sum(dim=-1)
        candidate_frets = candidate_features[..., 6] * 24.0
        expected_strings = (probabilities * candidate_strings).sum(dim=-1)
        expected_frets = (probabilities * candidate_frets).sum(dim=-1)
        predicted_delta_string = expected_strings[:, 1:] - expected_strings[:, :-1]
        predicted_delta_fret = expected_frets[:, 1:] - expected_frets[:, :-1]
        gold_delta_string = gold_strings[:, 1:] - gold_strings[:, :-1]
        gold_delta_fret = gold_frets[:, 1:] - gold_frets[:, :-1]
        pair_weights = torch.minimum(weights[:, 1:], weights[:, :-1]) * pair_mask
        pair_error = (
            torch.abs(predicted_delta_string - gold_delta_string) / 5.0
            + torch.abs(predicted_delta_fret - gold_delta_fret) / 24.0
        )
        transition = (pair_error * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
    return ce + transition_weight * transition, ce, transition


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(
    kind: str,
    train_tracks: Sequence[PreparedTrack],
    validation_tracks: Sequence[PreparedTrack] | None,
    *,
    seed: int,
    fixed_epochs: int | None = None,
) -> TrainOutcome:
    """Train one predeclared model and early-stop only on validation Tab F1."""

    _seed_everything(seed)
    if kind == "control":
        model = make_masked_linear_model()
        max_epochs = CONTROL_MAX_EPOCHS
        learning_rate = CONTROL_LEARNING_RATE
        transition_weight = 0.0
    elif kind == "context":
        model = make_context_model()
        max_epochs = CONTEXT_MAX_EPOCHS
        learning_rate = CONTEXT_LEARNING_RATE
        transition_weight = TRANSITION_CONSISTENCY_WEIGHT
    else:
        raise ValueError(f"unknown model kind: {kind}")
    if parameter_count(model) >= 500_000:
        raise AssertionError("context model exceeds the predeclared parameter cap")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    counts = _frequency_counts(train_tracks)
    windows = _training_windows(train_tracks)
    rng = np.random.default_rng(seed)
    epochs = fixed_epochs or max_epochs
    best_score = -math.inf
    best_epoch = epochs
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        ce_losses: list[float] = []
        transition_losses: list[float] = []
        order = rng.permutation(len(windows))
        for start in range(0, len(order), BATCH_SIZE):
            examples = [windows[int(index)] for index in order[start : start + BATCH_SIZE]]
            tensors = _batch(examples, counts)
            event, candidate, mask, padding, labels, weights, gold_strings, gold_frets = tensors
            optimizer.zero_grad(set_to_none=True)
            logits = model(event, candidate, mask, padding)
            loss, ce, transition = _loss(
                logits,
                candidate,
                labels,
                weights,
                gold_strings,
                gold_frets,
                transition_weight=transition_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            ce_losses.append(float(ce.detach()))
            transition_losses.append(float(transition.detach()))
        if validation_tracks is None:
            score = float("nan")
        else:
            score = summarize(
                _evaluate_context(model, validation_tracks, decoder="baseline")
            ).macro_tab_f1
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "candidate_ce": float(np.mean(ce_losses)),
                "transition_loss": float(np.mean(transition_losses)),
                "validation_tab_f1": score,
            }
        )
        print(
            f"    {kind} epoch {epoch:02d}: loss={np.mean(losses):.5f} "
            f"validation_tab_f1={score:.5f}",
            flush=True,
        )
        if validation_tracks is None:
            continue
        if score > best_score + 1.0e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOPPING_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    if validation_tracks is None:
        best_score = float("nan")
        best_epoch = epochs
    return TrainOutcome(model, best_epoch, best_score, tuple(history))


@torch.inference_mode()
def predict_probabilities(model: Any, prepared: PreparedTrack) -> np.ndarray:
    features = prepared.features
    windows = context_windows(
        features.cluster_ids,
        max_events=MAX_CONTEXT_EVENTS,
        overlap_events=CONTEXT_OVERLAP_EVENTS,
    )
    window_logits = []
    for indices in windows:
        selected = np.asarray(indices, dtype=np.int64)
        event = torch.from_numpy(features.event_features[selected]).unsqueeze(0)
        candidate = torch.from_numpy(features.candidate_features[selected]).unsqueeze(0)
        mask = torch.from_numpy(features.candidate_mask[selected]).unsqueeze(0)
        padding = torch.zeros((1, len(indices)), dtype=torch.bool)
        window_logits.append(model(event, candidate, mask, padding).squeeze(0).cpu().numpy())
    logits = merge_window_logits(len(prepared.events), windows, window_logits)
    return masked_softmax(logits, features.candidate_mask)


def _prediction_rank(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ranks = np.full(len(labels), 99, dtype=np.int64)
    for index in np.flatnonzero(labels >= 0):
        order = np.argsort(-probabilities[index], kind="stable")
        ranks[index] = int(np.flatnonzero(order == labels[index])[0]) + 1
    return ranks


def _add_track_result(
    result: EvaluationResult,
    prepared: PreparedTrack,
    predicted: Sequence[TabEvent],
    top3: np.ndarray,
) -> None:
    score = tab_f1(predicted, prepared.track.gold)
    track_id = prepared.track.track_id
    result.clip_scores[track_id] = score.f1
    result.clip_tab[track_id] = score
    result.strata[track_id] = f"{prepared.track.player}|{prepared.track.mode}"
    matches = label_prediction_matches(predicted, prepared.track.gold)
    for match in matches:
        event = predicted[match.predicted_index]
        result.prediction_rows.append(
            (track_id, event.onset_s, event.pitch_midi, event.string_idx, event.fret)
        )
        gold = prepared.track.gold[match.gold_index] if match.gold_index is not None else None
        candidate_count = len(candidate_positions(event.pitch_midi))
        ambiguous = match.label in {"correct", "wrong_position_same_pitch"} and candidate_count >= 2
        if ambiguous:
            result.ambiguous_total += 1
            result.ambiguous_correct += int(match.label == "correct")
            result.ambiguous_top3 += int(top3[match.predicted_index])
        if gold is not None:
            result.error_rows.append(
                {
                    "condition": result.name,
                    "track_id": track_id,
                    "player": prepared.track.player,
                    "mode": prepared.track.mode,
                    "style": prepared.track.style,
                    "pitch_midi": event.pitch_midi,
                    "candidate_count": candidate_count,
                    "reference_string": gold.string_idx,
                    "predicted_string": event.string_idx,
                    "predicted_minus_reference_string": event.string_idx - gold.string_idx,
                    "fret_displacement": event.fret - gold.fret,
                    "label": match.label,
                }
            )


def _evaluate_existing(tracks: Sequence[PreparedTrack], *, decoder: str) -> EvaluationResult:
    result = EvaluationResult(decoder)
    for prepared in tracks:
        predicted = prepared.baseline if decoder == "baseline" else prepared.segment
        _add_track_result(result, prepared, predicted, prepared.baseline_top3)
    return result


def _evaluate_context(
    model: Any, tracks: Sequence[PreparedTrack], *, decoder: str
) -> EvaluationResult:
    from tabvision.eval.string_assignment import decode_with_analysis

    result = EvaluationResult(f"context_{decoder}")
    for prepared in tracks:
        probabilities = predict_probabilities(model, prepared)
        rank = _prediction_rank(probabilities, prepared.labels)
        top3 = rank <= 3
        reranked = apply_context_probabilities(
            prepared.events,
            prepared.features,
            probabilities,
        )
        playability.set_transition_prior(prepared.bundle.global_sequence, SEQUENCE_WEIGHT)
        try:
            if decoder == "baseline":
                decoded = decode_with_analysis(reranked)
                predicted = decoded.paths[0].events
            elif decoder == "segment":
                decoded_segment = decode_segment_v1_with_analysis(
                    reranked,
                    config=DEFAULT_SEGMENT_CONFIG,
                    k_paths=1,
                    retain_analysis=False,
                )
                predicted = decoded_segment.paths[0].events
            else:
                raise ValueError(f"unknown decoder: {decoder}")
        finally:
            playability.set_transition_prior(None)
        _add_track_result(result, prepared, predicted, top3)
    return result


def _sum_tab(items: Iterable[TabF1Result]) -> TabF1Result:
    scores = list(items)
    tp = sum(item.true_positives for item in scores)
    fp = sum(item.false_positives for item in scores)
    fn = sum(item.false_negatives for item in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


@dataclass(frozen=True)
class MetricSummary:
    macro_tab_f1: float
    micro_tab_f1: float
    ambiguous_top1: float
    ambiguous_top3: float
    wrong_position_rate: float
    clips: int


def summarize(result: EvaluationResult, *, mode: str | None = None) -> MetricSummary:
    ids = [
        track_id for track_id in result.clip_scores if mode is None or track_id.endswith(f"_{mode}")
    ]
    if mode is None:
        correct = result.ambiguous_correct
        top3 = result.ambiguous_top3
        total = result.ambiguous_total
    else:
        rows = [row for row in result.error_rows if row["mode"] == mode]
        ambiguous = [
            row
            for row in rows
            if row["label"] in {"correct", "wrong_position_same_pitch"}
            and row["candidate_count"] >= 2
        ]
        correct = sum(row["label"] == "correct" for row in ambiguous)
        total = len(ambiguous)
        # Top-3 is decoder-independent and reported aggregate; keep mode fields
        # explicit rather than inventing a per-mode value from error rows.
        top3 = correct
    micro = _sum_tab(result.clip_tab[track_id] for track_id in ids)
    return MetricSummary(
        macro_tab_f1=float(np.mean([result.clip_scores[track_id] for track_id in ids])),
        micro_tab_f1=micro.f1,
        ambiguous_top1=correct / total if total else float("nan"),
        ambiguous_top3=top3 / total if total else float("nan"),
        wrong_position_rate=(total - correct) / total if total else float("nan"),
        clips=len(ids),
    )


def _bootstrap(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    mode: str | None = None,
) -> tuple[float, float, float]:
    ids = {
        track_id
        for track_id in baseline.clip_scores
        if mode is None or track_id.endswith(f"_{mode}")
    }
    outcome = paired_stratified_bootstrap(
        {key: baseline.clip_scores[key] for key in ids},
        {key: candidate.clip_scores[key] for key in ids},
        {key: baseline.strata[key] for key in ids},
        n_resamples=10_000,
        seed=42,
    )
    return outcome.mean_delta, outcome.lower, outcome.upper


def _prediction_hash(result: EvaluationResult) -> str:
    payload = json.dumps(sorted(result.prediction_rows), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_duration(tracks: Sequence[Track]) -> float:
    duration = 0.0
    for track in tracks:
        with wave.open(str(track.media_path), "rb") as handle:
            duration += handle.getnframes() / handle.getframerate()
    return duration


def _metric_row(result: EvaluationResult) -> dict[str, Any]:
    aggregate = summarize(result)
    solo = summarize(result, mode="solo")
    comp = summarize(result, mode="comp")
    return {
        "condition": result.name,
        **asdict(aggregate),
        "solo_macro_tab_f1": solo.macro_tab_f1,
        "comp_macro_tab_f1": comp.macro_tab_f1,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_error_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse row-level diagnostics into the required error-count dimensions."""

    dimensions = (
        "reference_string",
        "predicted_minus_reference_string",
        "fret_displacement",
        "pitch_midi",
        "candidate_count",
        "style",
        "player",
    )
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in rows:
        for dimension in dimensions:
            counts[
                (
                    str(row["condition"]),
                    dimension,
                    str(row[dimension]),
                    str(row["label"]),
                )
            ] += 1
    return [
        {
            "condition": condition,
            "dimension": dimension,
            "value": value,
            "label": label,
            "count": count,
        }
        for (condition, dimension, value, label), count in sorted(counts.items())
    ]


def _manifest_payload(
    artifact_path: Path,
    *,
    registered: bool,
    decision: str,
    metrics: dict[str, Any],
    fold_epochs: dict[str, list[int]],
    final_epochs: int,
    source_hash: str,
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "name": ARTIFACT_NAME,
        "artifact_kind": "assignment_context",
        "artifact_version": "guitarset-context-v1",
        "artifact_file": artifact_path.name,
        "artifact_sha256": _file_sha256(artifact_path),
        "registered": registered,
        "mode": "all",
        "compatible_position_prior": "guitarset-v1",
        "compatible_sequence_prior": "guitarset-seq-v1",
        "compatible_decoder_versions": ["baseline", "segment-v1"],
        "architecture": {
            "encoder_layers": 2,
            "d_model": 64,
            "attention_heads": 4,
            "feed_forward": 128,
            "dropout": 0.1,
            "max_events": MAX_CONTEXT_EVENTS,
            "overlap_events": CONTEXT_OVERLAP_EVENTS,
            "event_feature_dim": EVENT_FEATURE_DIM,
            "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
            "max_candidates": MAX_CANDIDATES,
        },
        "training": {
            "development_players": list(DEV_PLAYERS),
            "confirmation_player": FINAL_PLAYER,
            "fold_seeds": [BASE_SEED + index for index in range(len(DEV_PLAYERS))],
            "fold_best_epochs": fold_epochs,
            "final_epochs": final_epochs,
            "transition_consistency_weight": TRANSITION_CONSISTENCY_WEIGHT,
            "early_stopping_metric": "macro per-clip Tab F1",
        },
        "data": {
            "corpus": "GuitarSet",
            "license": "CC BY 4.0",
            "source_sha256": source_hash,
            "symbolic_pretraining": "not_run",
        },
        "metrics": metrics,
        "gate": {"decision": decision},
        "code": {
            "base_commit": _git_commit(),
            "context_reranker_sha256": _file_sha256(
                package_root / "tabvision/fusion/context_reranker.py"
            ),
            "evaluation_script_sha256": _file_sha256(Path(__file__).resolve()),
        },
    }


def _report(
    *,
    oof_results: dict[str, EvaluationResult],
    final_results: dict[str, EvaluationResult],
    selected_name: str,
    decision: str,
    control_gain: float,
    context_gain: float,
    bootstrap: tuple[float, float, float],
    comp_bootstrap: tuple[float, float, float],
    player_deltas: dict[str, float],
    runtime: RuntimeResult,
    artifact_path: Path,
    manifest_path: Path,
    artifact_registered: bool,
    fold_epochs: dict[str, list[int]],
    pdmx_reason: str,
) -> str:
    repository_root = Path(__file__).resolve().parents[3]
    artifact_label = artifact_path.relative_to(repository_root).as_posix()
    manifest_label = manifest_path.relative_to(repository_root).as_posix()
    lines = [
        "# String assignment Phase 2: constrained contextual candidate reranker",
        "",
        "## Decision",
        "",
        f"**{decision}.** The fixed contextual architecture was evaluated once with "
        "player-held-out OOF predictions; no post-result model or hyperparameter search was run.",
        "",
        f"Selected predeclared composition: `{selected_name}`. The artifact is "
        f"`{'registered' if artifact_registered else 'unregistered'}` and automatic routing "
        "therefore remains on the last gate-passed baseline unless all promotion criteria passed.",
        "",
        "## Development OOF results",
        "",
        "| condition | aggregate macro | micro | solo macro | comp macro | "
        "ambiguous top-1 | top-3 | wrong rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "segment", "control_baseline", "context_baseline", "context_segment"):
        result = oof_results[name]
        row = _metric_row(result)
        lines.append(
            f"| {name} | {row['macro_tab_f1']:.4f} | {row['micro_tab_f1']:.4f} | "
            f"{row['solo_macro_tab_f1']:.4f} | {row['comp_macro_tab_f1']:.4f} | "
            f"{row['ambiguous_top1']:.4f} | {row['ambiguous_top3']:.4f} | "
            f"{row['wrong_position_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Linear-control ambiguous top-1 gain: `{control_gain:+.4f}`.",
            f"- Selected contextual ambiguous top-1 gain: `{context_gain:+.4f}`.",
            f"- Aggregate Tab F1 delta / paired 95% CI: `{bootstrap[0]:+.4f}` "
            f"`[{bootstrap[1]:+.4f}, {bootstrap[2]:+.4f}]`.",
            f"- Comp delta / paired 95% CI: `{comp_bootstrap[0]:+.4f}` "
            f"`[{comp_bootstrap[1]:+.4f}, {comp_bootstrap[2]:+.4f}]`.",
            "- Player-fold deltas: "
            + ", ".join(f"`{player} {delta:+.4f}`" for player, delta in player_deltas.items())
            + ".",
            f"- Fold best epochs: `{json.dumps(fold_epochs, sort_keys=True)}`.",
            "- Ambiguous-note rows mean events with at least two physically playable "
            "pitch-preserving candidates; baseline and candidates use the same pool.",
            f"- Optional PDMX pretraining: **not run** — {pdmx_reason}",
            "",
            "Onset and pitch events are byte-for-byte unchanged by construction; their frozen "
            f"benchmark values remain onset F1 `{FROZEN_ONSET_F1:.4f}` and pitch F1 "
            f"`{FROZEN_PITCH_F1:.4f}`.",
            "",
            "## Frozen player 05 confirmation",
            "",
            "| condition | aggregate macro | solo macro | comp macro | "
            "ambiguous top-1 | wrong rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("baseline", "segment", selected_name):
        row = _metric_row(final_results[name])
        lines.append(
            f"| {name} | {row['macro_tab_f1']:.4f} | {row['solo_macro_tab_f1']:.4f} | "
            f"{row['comp_macro_tab_f1']:.4f} | {row['ambiguous_top1']:.4f} | "
            f"{row['wrong_position_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifact, runtime, and safety",
            "",
            f"- TorchScript artifact: `{artifact_label}` "
            f"(`{artifact_path.stat().st_size}` bytes, SHA-256 `{_file_sha256(artifact_path)}`).",
            f"- Manifest: `{manifest_label}`; architecture is below 500,000 parameters.",
            f"- Added context inference: `{runtime.added_seconds_per_60s:.3f}` s per 60 s; "
            f"projected pipeline total `{runtime.projected_total_seconds_per_60s:.2f}` s.",
            f"- Peak process memory: `{runtime.peak_memory_bytes}` bytes.",
            f"- Frozen prediction hash: `{runtime.prediction_sha256}`; deterministic rerun: "
            f"`{runtime.deterministic_rerun_sha256}`.",
            "- Context routing is restricted to clean acoustic, standard tuning, capo 0. "
            "Classical/electric/out-of-domain requests fall back to baseline.",
            "- Missing, corrupt, incompatible, and unregistered artifacts fall back to baseline.",
            "",
            "## Reproducibility",
            "",
            "Training used CPU PyTorch from `tabvision/.venv`, deterministic algorithms, fixed "
            "seeds, inverse joint-frequency weighting, and early stopping on held-out macro Tab "
            "F1. Error rows are written beside this report for grouping by string, offset, fret, "
            "pitch, candidate count, style, and player.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-home", type=Path, default=Path.home() / ".tabvision/data/guitarset")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--backend", default="highres")
    parser.add_argument("--output-dir", type=Path, default=Path("../docs/EVAL_REPORTS"))
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    cfg = GuitarConfig()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("loading fixed GuitarSet cache...", flush=True)
    tracks = load_tracks(args.data_home, args.cache_dir, args.backend, cfg)
    dev_tracks = [track for track in tracks if track.player in DEV_PLAYERS]
    final_tracks = [track for track in tracks if track.player == FINAL_PLAYER]

    oof_results = {
        name: EvaluationResult(name)
        for name in (
            "baseline",
            "segment",
            "control_baseline",
            "context_baseline",
            "context_segment",
        )
    }
    fold_epochs: dict[str, list[int]] = {"control": [], "context": []}
    histories: dict[str, Any] = {}
    print("running five fixed player-held-out folds...", flush=True)
    for fold_index, (held_out, train, validation) in enumerate(player_folds(dev_tracks)):
        print(f"  fold {held_out}: learning fold-only priors", flush=True)
        bundle = learn_bundle(train, cfg)
        prepared_train = prepare_tracks(train, bundle, cfg, label=f"fold {held_out} train")
        prepared_validation = prepare_tracks(
            validation, bundle, cfg, label=f"fold {held_out} validation"
        )
        oof_results["baseline"].merge(_evaluate_existing(prepared_validation, decoder="baseline"))
        oof_results["segment"].merge(_evaluate_existing(prepared_validation, decoder="segment"))
        control = train_model(
            "control", prepared_train, prepared_validation, seed=BASE_SEED + fold_index
        )
        context = train_model(
            "context", prepared_train, prepared_validation, seed=BASE_SEED + fold_index
        )
        fold_epochs["control"].append(control.best_epoch)
        fold_epochs["context"].append(context.best_epoch)
        histories[held_out] = {
            "control": list(control.history),
            "context": list(context.history),
        }
        control_result = _evaluate_context(control.model, prepared_validation, decoder="baseline")
        control_result.name = "control_baseline"
        oof_results["control_baseline"].merge(control_result)
        context_baseline = _evaluate_context(context.model, prepared_validation, decoder="baseline")
        context_baseline.name = "context_baseline"
        oof_results["context_baseline"].merge(context_baseline)
        context_segment = _evaluate_context(context.model, prepared_validation, decoder="segment")
        context_segment.name = "context_segment"
        oof_results["context_segment"].merge(context_segment)

    baseline_summary = summarize(oof_results["baseline"])
    control_summary = summarize(oof_results["control_baseline"])
    selected_name = max(
        ("context_baseline", "context_segment"),
        key=lambda name: summarize(oof_results[name]).macro_tab_f1,
    )
    selected = oof_results[selected_name]
    selected_summary = summarize(selected)
    control_gain = control_summary.ambiguous_top1 - baseline_summary.ambiguous_top1
    context_gain = selected_summary.ambiguous_top1 - baseline_summary.ambiguous_top1
    boot = _bootstrap(oof_results["baseline"], selected)
    comp_boot = _bootstrap(oof_results["baseline"], selected, mode="comp")
    solo_delta = (
        summarize(selected, mode="solo").macro_tab_f1
        - summarize(oof_results["baseline"], mode="solo").macro_tab_f1
    )
    aggregate_delta = selected_summary.macro_tab_f1 - baseline_summary.macro_tab_f1
    wrong_reduction = (
        baseline_summary.wrong_position_rate - selected_summary.wrong_position_rate
    ) / baseline_summary.wrong_position_rate
    player_deltas = {
        player: float(
            np.mean(
                [
                    selected.clip_scores[track_id] - oof_results["baseline"].clip_scores[track_id]
                    for track_id in selected.clip_scores
                    if track_id.startswith(f"{player}_")
                ]
            )
        )
        for player in DEV_PLAYERS
    }
    promoted = (
        solo_delta >= 0.03
        and aggregate_delta >= 0.02
        and boot[1] > 0.0
        and wrong_reduction >= 0.10
        and comp_boot[0] >= -0.005
        and comp_boot[1] > -0.01
        and sum(delta > 0.0 for delta in player_deltas.values()) >= 4
    )
    if promoted:
        decision = "Promote context-v1"
    elif context_gain >= 0.03:
        decision = "Useful contextual diagnostic; do not ship"
    elif context_gain < 0.02:
        decision = "Close symbolic-context expansion"
    else:
        decision = "Context evidence is inconclusive; do not ship"
    pdmx_trigger = aggregate_delta >= 0.015
    fold_range = max(player_deltas.values()) - min(player_deltas.values())
    pdmx_reason = (
        "the GuitarSet-only aggregate gain was below +0.015"
        if not pdmx_trigger
        else (
            "fold variance was not data-limited under the fixed diagnostic"
            if fold_range < 0.02
            else "trigger met; a separately approved license-safe pretraining run is required"
        )
    )
    if pdmx_trigger and fold_range >= 0.02:
        raise RuntimeError(
            "optional PDMX pretraining trigger met; stop for the plan's license/cost checkpoint"
        )

    development_checkpoint = {
        "decision": decision,
        "selected_composition": selected_name,
        "promoted_before_runtime_gate": promoted,
        "control_ambiguous_top1_gain": control_gain,
        "context_ambiguous_top1_gain": context_gain,
        "aggregate_bootstrap": boot,
        "comp_bootstrap": comp_boot,
        "player_deltas": player_deltas,
        "fold_epochs": fold_epochs,
        "metrics": {name: _metric_row(result) for name, result in oof_results.items()},
        "clip_scores": {name: result.clip_scores for name, result in oof_results.items()},
    }
    (output_dir / "string_assignment_phase2_2026-07-15_oof_checkpoint.json").write_text(
        json.dumps(development_checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_context_epochs = max(1, round(statistics.median(fold_epochs["context"])))
    final_control_epochs = max(1, round(statistics.median(fold_epochs["control"])))
    print(
        f"frozen OOF decision: {decision}; composition={selected_name}; "
        f"player05 remains unevaluated; final epochs context={final_context_epochs}",
        flush=True,
    )
    final_bundle = learn_bundle(dev_tracks, cfg)
    prepared_dev = prepare_tracks(dev_tracks, final_bundle, cfg, label="final train")
    final_model = train_model(
        "context",
        prepared_dev,
        None,
        seed=BASE_SEED + 100,
        fixed_epochs=final_context_epochs,
    )
    # Train the frozen control epoch count as a reproducibility check, without
    # using its output to alter the already-selected contextual composition.
    train_model(
        "control",
        prepared_dev,
        None,
        seed=BASE_SEED + 100,
        fixed_epochs=final_control_epochs,
    )

    print("running frozen player05 confirmation...", flush=True)
    prepared_final = prepare_tracks(final_tracks, final_bundle, cfg, label="player05")
    final_results = {
        "baseline": _evaluate_existing(prepared_final, decoder="baseline"),
        "segment": _evaluate_existing(prepared_final, decoder="segment"),
    }
    started = time.perf_counter()
    final_results["context_baseline"] = _evaluate_context(
        final_model.model, prepared_final, decoder="baseline"
    )
    final_results["context_segment"] = _evaluate_context(
        final_model.model, prepared_final, decoder="segment"
    )
    context_seconds = time.perf_counter() - started
    rerun = _evaluate_context(
        final_model.model,
        prepared_final,
        decoder="segment" if selected_name == "context_segment" else "baseline",
    )
    rerun.name = selected_name
    source_duration = _source_duration(final_tracks)
    selected_hash = _prediction_hash(final_results[selected_name])
    rerun_hash = _prediction_hash(rerun)
    if selected_hash != rerun_hash:
        raise AssertionError("fixed-seed player05 prediction hash changed on rerun")
    added_per_60 = context_seconds * 60.0 / source_duration
    runtime = RuntimeResult(
        context_seconds,
        source_duration,
        added_per_60,
        CURRENT_PIPELINE_SECONDS_PER_60S + added_per_60,
        _peak_process_memory_bytes(),
        selected_hash,
        rerun_hash,
    )
    if runtime.added_seconds_per_60s >= 0.2 * CURRENT_PIPELINE_SECONDS_PER_60S:
        promoted = False
        decision = "Runtime gate failed; do not ship"

    artifact_dir = Path(__file__).resolve().parents[2] / "tabvision/fusion/priors"
    artifact_path = artifact_dir / "context_v1.pt"
    scripted = torch.jit.script(final_model.model)
    scripted.save(str(artifact_path))
    reloaded = torch.jit.load(str(artifact_path))
    example = prepared_final[0]
    sample_window = context_windows(example.features.cluster_ids)[0]
    selected_indices = np.asarray(sample_window, dtype=np.int64)
    sample_args = (
        torch.from_numpy(example.features.event_features[selected_indices]).unsqueeze(0),
        torch.from_numpy(example.features.candidate_features[selected_indices]).unsqueeze(0),
        torch.from_numpy(example.features.candidate_mask[selected_indices]).unsqueeze(0),
        torch.zeros((1, len(sample_window)), dtype=torch.bool),
    )
    torch.testing.assert_close(final_model.model(*sample_args), reloaded(*sample_args))

    source_digest, _source_kind = _source_hash(dev_tracks)
    manifest_path = artifact_dir / "context_v1.manifest.json"
    manifest_metrics = {
        "selected_composition": selected_name,
        "oof_baseline_macro_tab_f1": baseline_summary.macro_tab_f1,
        "oof_context_macro_tab_f1": selected_summary.macro_tab_f1,
        "oof_aggregate_delta": aggregate_delta,
        "oof_ambiguous_top1_gain": context_gain,
        "oof_wrong_position_relative_reduction": wrong_reduction,
        "player_fold_deltas": player_deltas,
    }
    manifest = _manifest_payload(
        artifact_path,
        registered=promoted,
        decision=decision,
        metrics=manifest_metrics,
        fold_epochs=fold_epochs,
        final_epochs=final_context_epochs,
        source_hash=source_digest,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    errors_path = output_dir / "string_assignment_phase2_2026-07-15_errors.csv"
    _write_csv(
        errors_path,
        aggregate_error_counts(oof_results["baseline"].error_rows + selected.error_rows),
    )
    summary_path = output_dir / "string_assignment_phase2_2026-07-15_metrics.csv"
    _write_csv(summary_path, [_metric_row(result) for result in oof_results.values()])
    history_path = output_dir / "string_assignment_phase2_2026-07-15_training.json"
    history_path.write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "string_assignment_phase2_2026-07-15.md"
    report_path.write_text(
        _report(
            oof_results=oof_results,
            final_results=final_results,
            selected_name=selected_name,
            decision=decision,
            control_gain=control_gain,
            context_gain=context_gain,
            bootstrap=boot,
            comp_bootstrap=comp_boot,
            player_deltas=player_deltas,
            runtime=runtime,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
            artifact_registered=promoted,
            fold_epochs=fold_epochs,
            pdmx_reason=pdmx_reason,
        ),
        encoding="utf-8",
    )
    run_payload = {
        "decision": decision,
        "selected_composition": selected_name,
        "promoted": promoted,
        "metrics": manifest_metrics,
        "runtime": asdict(runtime),
        "fold_epochs": fold_epochs,
        "final_epochs": final_context_epochs,
        "source": _cache_provenance(tracks, args.backend),
        "packages": _package_versions(),
        "artifact_sha256": _file_sha256(artifact_path),
        "manifest_sha256": _file_sha256(manifest_path),
    }
    (output_dir / "string_assignment_phase2_2026-07-15_run.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"report: {report_path}", flush=True)
    print(f"decision: {decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
