"""Pitch-preserving contextual candidate reranking for string assignment.

The feature builder regenerates every playable ``(string, fret)`` from the
detected MIDI pitch.  Models therefore score a closed candidate set and can
never change pitch.  PyTorch is imported lazily so the default fusion path
does not acquire a runtime dependency on the training extra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabvision.fusion import chord, playability
from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.fusion.evidence import combine_candidate_evidence
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

MAX_CANDIDATES = 6
EVENT_FEATURE_DIM = 33
CANDIDATE_FEATURE_DIM = 16
MAX_CONTEXT_EVENTS = 128
CONTEXT_OVERLAP_EVENTS = 32


@dataclass(frozen=True)
class SegmentHint:
    """Phase 1 latent state aligned to one audio event."""

    string_offset: int = 0
    zone_center: int | None = None
    baseline_string_idx: int | None = None


@dataclass(frozen=True)
class ContextFeatures:
    """Dense model inputs plus the pitch-derived candidate identities."""

    event_features: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    candidates: tuple[tuple[Candidate, ...], ...]
    cluster_ids: np.ndarray

    def __post_init__(self) -> None:
        n_events = len(self.candidates)
        expected_event = (n_events, EVENT_FEATURE_DIM)
        expected_candidate = (n_events, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM)
        expected_mask = (n_events, MAX_CANDIDATES)
        if self.event_features.shape != expected_event:
            raise ValueError(
                f"event_features must have shape {expected_event}, got {self.event_features.shape}"
            )
        if self.candidate_features.shape != expected_candidate:
            raise ValueError(
                "candidate_features must have shape "
                f"{expected_candidate}, got {self.candidate_features.shape}"
            )
        if self.candidate_mask.shape != expected_mask:
            raise ValueError(
                f"candidate_mask must have shape {expected_mask}, got {self.candidate_mask.shape}"
            )
        if self.cluster_ids.shape != (n_events,):
            raise ValueError("cluster_ids must contain one value per event")


def build_context_features(
    events: Sequence[AudioEvent],
    *,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    baseline: Sequence[TabEvent] | None = None,
    segment_hints: Sequence[SegmentHint] | None = None,
) -> ContextFeatures:
    """Build deterministic inference-available event and candidate features.

    ``events`` must already carry the fold-specific corpus prior in
    ``AudioEvent.fret_prior``.  The same values supply the corpus probability
    and current emission terms without consulting annotations.
    """

    cfg = cfg or GuitarConfig()
    session = session or SessionConfig()
    ordered = sorted(
        (event for event in events if candidate_positions(event.pitch_midi, cfg)),
        key=lambda event: event.onset_s,
    )
    n_events = len(ordered)
    if baseline is not None and len(baseline) != n_events:
        raise ValueError("baseline must contain one position per playable event")
    if segment_hints is not None and len(segment_hints) != n_events:
        raise ValueError("segment_hints must contain one hint per playable event")

    baseline_candidates = (
        tuple(Candidate(event.string_idx, event.fret) for event in baseline)
        if baseline is not None
        else tuple(_lowest_cost_candidate(event, cfg) for event in ordered)
    )
    raw_hints = tuple(segment_hints or (SegmentHint(),) * n_events)
    hints = tuple(
        SegmentHint(
            hint.string_offset,
            hint.zone_center,
            baseline_candidates[index].string_idx,
        )
        for index, hint in enumerate(raw_hints)
    )
    per_event = tuple(tuple(candidate_positions(event.pitch_midi, cfg)) for event in ordered)

    event_matrix = np.zeros((n_events, EVENT_FEATURE_DIM), dtype=np.float32)
    candidate_matrix = np.zeros((n_events, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((n_events, MAX_CANDIDATES), dtype=np.bool_)
    cluster_ids = _cluster_ids(ordered)
    cluster_members: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids.tolist()):
        cluster_members.setdefault(cluster_id, []).append(index)

    for index, (event, candidates, hint) in enumerate(zip(ordered, per_event, hints, strict=True)):
        members = cluster_members[int(cluster_ids[index])]
        pitches = [ordered[item].pitch_midi for item in members]
        pitch_rank = sorted(pitches).index(event.pitch_midi)
        event_matrix[index] = _event_features(
            ordered,
            index,
            len(members),
            pitch_rank,
            event.pitch_midi == min(pitches),
            len(candidates),
            hint,
            session,
        )
        prev_position = baseline_candidates[index - 1] if index else None
        next_position = baseline_candidates[index + 1] if index + 1 < n_events else None
        prev_gap = event.onset_s - ordered[index - 1].onset_s if index else None
        next_gap = ordered[index + 1].onset_s - event.onset_s if index + 1 < n_events else None
        for candidate_index, candidate in enumerate(candidates):
            mask[index, candidate_index] = True
            candidate_matrix[index, candidate_index] = _candidate_features(
                event,
                candidate,
                len(candidates),
                prev_position,
                next_position,
                prev_gap,
                next_gap,
                hint,
                cfg,
            )

    return ContextFeatures(event_matrix, candidate_matrix, mask, per_event, cluster_ids)


def _event_features(
    events: Sequence[AudioEvent],
    index: int,
    chord_size: int,
    note_rank: int,
    bass_note: bool,
    candidate_count: int,
    hint: SegmentHint,
    session: SessionConfig,
) -> np.ndarray:
    event = events[index]
    previous = events[index - 1] if index else None
    following = events[index + 1] if index + 1 < len(events) else None
    values: list[float] = [event.pitch_midi / 127.0]
    values.extend(float(item == event.pitch_midi % 12) for item in range(12))
    values.extend(
        (
            _clip_interval(event.pitch_midi - previous.pitch_midi) if previous else 0.0,
            _clip_interval(following.pitch_midi - event.pitch_midi) if following else 0.0,
            math.log1p(max(0.0, event.offset_s - event.onset_s)),
            math.log1p(max(0.0, event.onset_s - previous.onset_s)) if previous else 0.0,
            math.log1p(max(0.0, following.onset_s - event.onset_s)) if following else 0.0,
            min(chord_size, 6) / 6.0,
            note_rank / max(chord_size - 1, 1),
            float(bass_note),
            candidate_count / MAX_CANDIDATES,
            hint.string_offset / 5.0,
            0.0 if hint.zone_center is None else hint.zone_center / 24.0,
            float(hint.zone_center is None),
        )
    )
    values.extend(
        float(session.instrument == item) for item in ("acoustic", "classical", "electric")
    )
    values.extend(float(session.tone == item) for item in ("clean", "distorted"))
    values.extend(float(session.style == item) for item in ("fingerstyle", "strumming", "mixed"))
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (EVENT_FEATURE_DIM,):
        raise AssertionError(f"event feature schema drift: {array.shape}")
    return array


def _candidate_features(
    event: AudioEvent,
    candidate: Candidate,
    candidate_count: int,
    previous: Candidate | None,
    following: Candidate | None,
    previous_gap_s: float | None,
    next_gap_s: float | None,
    hint: SegmentHint,
    cfg: GuitarConfig,
) -> np.ndarray:
    corpus_probability = _prior_probability(event, candidate)
    emission = playability.emission_cost(candidate, event, None, cfg, lambda_vision=0.0)
    prev_transition = (
        playability.transition_cost(previous, candidate, cfg, gap_s=previous_gap_s)
        if previous is not None
        else 0.0
    )
    next_transition = (
        playability.transition_cost(candidate, following, cfg, gap_s=next_gap_s)
        if following is not None
        else 0.0
    )
    values = [float(candidate.string_idx == item) for item in range(6)]
    values.extend(
        (
            candidate.fret / max(cfg.max_fret, 1),
            float(candidate.fret == cfg.capo),
            candidate_count / MAX_CANDIDATES,
            max(-20.0, math.log(max(corpus_probability, 1e-9))) / 20.0,
            min(emission, 20.0) / 20.0,
            min(prev_transition, 20.0) / 20.0,
            min(next_transition, 20.0) / 20.0,
            (
                candidate.string_idx
                - (
                    hint.baseline_string_idx
                    if hint.baseline_string_idx is not None
                    else candidate.string_idx
                )
                - hint.string_offset
            )
            / 5.0,
            (
                0.0
                if hint.zone_center is None or candidate.fret == cfg.capo
                else abs(candidate.fret - hint.zone_center) / max(cfg.max_fret, 1)
            ),
            float(hint.zone_center is None),
        )
    )
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (CANDIDATE_FEATURE_DIM,):
        raise AssertionError(f"candidate feature schema drift: {array.shape}")
    return array


def _clip_interval(value: int) -> float:
    return max(-24, min(24, value)) / 24.0


def _prior_probability(event: AudioEvent, candidate: Candidate) -> float:
    prior = event.fret_prior
    if prior is None:
        return 1.0
    try:
        if prior.ndim == 2:
            return float(prior[candidate.string_idx, candidate.fret])
        if prior.ndim == 1:
            return float(prior[candidate.fret])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.0
    return 0.0


def _lowest_cost_candidate(event: AudioEvent, cfg: GuitarConfig) -> Candidate:
    return min(
        candidate_positions(event.pitch_midi, cfg),
        key=lambda candidate: playability.emission_cost(
            candidate, event, None, cfg, lambda_vision=0.0
        ),
    )


def _cluster_ids(events: Sequence[AudioEvent]) -> np.ndarray:
    ids = np.empty(len(events), dtype=np.int64)
    offset = 0
    for cluster_id, members in enumerate(chord.cluster_events(events)):
        ids[offset : offset + len(members)] = cluster_id
        offset += len(members)
    return ids


def context_windows(
    cluster_ids: Sequence[int] | np.ndarray,
    *,
    max_events: int = MAX_CONTEXT_EVENTS,
    overlap_events: int = CONTEXT_OVERLAP_EVENTS,
) -> tuple[tuple[int, ...], ...]:
    """Return cluster-safe overlapping event windows covering each event."""

    if max_events < 1:
        raise ValueError("max_events must be positive")
    if not 0 <= overlap_events < max_events:
        raise ValueError("overlap_events must be in [0, max_events)")
    ids = [int(value) for value in cluster_ids]
    if not ids:
        return ()
    clusters: list[list[int]] = []
    for index, cluster_id in enumerate(ids):
        if not clusters or ids[clusters[-1][0]] != cluster_id:
            clusters.append([])
        clusters[-1].append(index)
    if any(len(item) > max_events for item in clusters):
        raise ValueError("one onset cluster exceeds the context window limit")

    windows: list[tuple[int, ...]] = []
    start_cluster = 0
    while start_cluster < len(clusters):
        end_cluster = start_cluster
        event_count = 0
        while end_cluster < len(clusters):
            next_count = event_count + len(clusters[end_cluster])
            if next_count > max_events:
                break
            event_count = next_count
            end_cluster += 1
        windows.append(
            tuple(index for group in clusters[start_cluster:end_cluster] for index in group)
        )
        if end_cluster == len(clusters):
            break
        retained = 0
        next_start = end_cluster
        while next_start > start_cluster and retained < overlap_events:
            next_start -= 1
            retained += len(clusters[next_start])
        start_cluster = max(start_cluster + 1, next_start)
    return tuple(windows)


def merge_window_logits(
    n_events: int,
    windows: Sequence[Sequence[int]],
    logits: Sequence[np.ndarray],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> np.ndarray:
    """Average overlapping logits into exactly one score row per event."""

    if len(windows) != len(logits):
        raise ValueError("windows and logits must have the same length")
    totals = np.zeros((n_events, max_candidates), dtype=np.float64)
    counts = np.zeros(n_events, dtype=np.int64)
    for indices, values in zip(windows, logits, strict=True):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(indices), max_candidates):
            raise ValueError("window logits have an incompatible shape")
        for local_index, event_index in enumerate(indices):
            totals[event_index] += array[local_index]
            counts[event_index] += 1
    if n_events and np.any(counts == 0):
        raise ValueError("windows do not cover every event")
    return (totals / np.maximum(counts[:, None], 1)).astype(np.float32)


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Numerically stable candidate softmax with exact zero for masked cells."""

    scores = np.asarray(logits, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    if scores.shape != valid.shape:
        raise ValueError("logits and mask must have identical shapes")
    if scores.ndim != 2 or np.any(valid.sum(axis=1) == 0):
        raise ValueError("each event must expose at least one playable candidate")
    masked = np.where(valid, scores, -np.inf)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponent = np.where(valid, np.exp(masked - maximum), 0.0)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def apply_context_probabilities(
    events: Sequence[AudioEvent],
    features: ContextFeatures,
    probabilities: np.ndarray,
    *,
    cfg: GuitarConfig | None = None,
    evidence_weight: float = 1.0,
) -> list[AudioEvent]:
    """Combine pitch-safe context evidence with each event's existing prior."""

    cfg = cfg or GuitarConfig()
    if probabilities.shape != features.candidate_mask.shape:
        raise ValueError("probabilities must align with the candidate mask")
    playable_events = [event for event in events if candidate_positions(event.pitch_midi, cfg)]
    if len(playable_events) != len(features.candidates):
        raise ValueError("features do not align with the playable events")
    out: list[AudioEvent] = []
    feature_index = 0
    for event in sorted(events, key=lambda item: item.onset_s):
        candidates = candidate_positions(event.pitch_midi, cfg)
        if not candidates:
            out.append(event)
            continue
        evidence = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
        for candidate_index, candidate in enumerate(features.candidates[feature_index]):
            evidence[candidate.string_idx, candidate.fret] = probabilities[
                feature_index, candidate_index
            ]
        combined = combine_candidate_evidence(
            event.pitch_midi,
            cfg,
            {
                "existing": (event.fret_prior, 1.0),
                "context": (evidence, evidence_weight),
            },
        )
        out.append(
            AudioEvent(
                onset_s=event.onset_s,
                offset_s=event.offset_s,
                pitch_midi=event.pitch_midi,
                velocity=event.velocity,
                confidence=event.confidence,
                pitch_logits=event.pitch_logits,
                fret_prior=combined,
                tags=event.tags,
            )
        )
        feature_index += 1
    return out


def make_masked_linear_model() -> Any:
    """Construct the Phase 2A control without importing torch at package load."""

    import torch

    class MaskedLinearCandidateModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scorer = torch.nn.Linear(EVENT_FEATURE_DIM + CANDIDATE_FEATURE_DIM, 1)

        def forward(  # type: ignore[no-untyped-def]
            self, event_features, candidate_features, candidate_mask, padding_mask
        ):
            del padding_mask
            expanded = event_features.unsqueeze(2).expand(-1, -1, candidate_features.shape[2], -1)
            logits = self.scorer(torch.cat((expanded, candidate_features), dim=-1)).squeeze(-1)
            return logits.masked_fill(~candidate_mask, -1.0e9)

    return MaskedLinearCandidateModel()


def make_context_model(*, dropout: float = 0.1) -> Any:
    """Construct the frozen two-layer, 64-wide Phase 2B architecture."""

    import torch

    class ContextCandidateTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.event_projection = torch.nn.Linear(EVENT_FEATURE_DIM, 64)
            self.position_embedding = torch.nn.Embedding(MAX_CONTEXT_EVENTS, 64)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
            self.candidate_head = torch.nn.Sequential(
                torch.nn.Linear(64 + CANDIDATE_FEATURE_DIM, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, 1),
            )

        def forward(  # type: ignore[no-untyped-def]
            self, event_features, candidate_features, candidate_mask, padding_mask
        ):
            length = event_features.shape[1]
            positions = torch.arange(length, device=event_features.device).unsqueeze(0)
            encoded = self.encoder(
                self.event_projection(event_features) + self.position_embedding(positions),
                src_key_padding_mask=padding_mask,
            )
            expanded = encoded.unsqueeze(2).expand(-1, -1, candidate_features.shape[2], -1)
            logits = self.candidate_head(torch.cat((expanded, candidate_features), dim=-1)).squeeze(
                -1
            )
            return logits.masked_fill(~candidate_mask, -1.0e9)

    return ContextCandidateTransformer()


def parameter_count(model: Any) -> int:
    """Return the number of trainable parameters for a torch model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "CANDIDATE_FEATURE_DIM",
    "CONTEXT_OVERLAP_EVENTS",
    "ContextFeatures",
    "EVENT_FEATURE_DIM",
    "MAX_CANDIDATES",
    "MAX_CONTEXT_EVENTS",
    "SegmentHint",
    "apply_context_probabilities",
    "build_context_features",
    "context_windows",
    "make_context_model",
    "make_masked_linear_model",
    "masked_softmax",
    "merge_window_logits",
    "parameter_count",
]
