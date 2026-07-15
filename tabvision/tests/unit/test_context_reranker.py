from __future__ import annotations

import numpy as np
import pytest

from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.context_reranker import (
    CANDIDATE_FEATURE_DIM,
    EVENT_FEATURE_DIM,
    MAX_CANDIDATES,
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
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent


def _event(onset: float, pitch: int, prior: np.ndarray | None = None) -> AudioEvent:
    return AudioEvent(onset, onset + 0.25, pitch, 0.8, 0.9, fret_prior=prior)


def _prior(cfg: GuitarConfig, pitch: int, values: tuple[float, ...]) -> np.ndarray:
    matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    for candidate, value in zip(candidate_positions(pitch, cfg), values, strict=True):
        matrix[candidate.string_idx, candidate.fret] = value
    return matrix / matrix.sum()


def test_candidate_mask_and_application_preserve_pitch() -> None:
    cfg = GuitarConfig()
    events = [_event(0.0, 64), _event(0.5, 67), _event(1.0, 40)]
    baseline = [
        TabEvent(event.onset_s, 0.25, candidate.string_idx, candidate.fret, event.pitch_midi, 1.0)
        for event in events
        for candidate in candidate_positions(event.pitch_midi, cfg)[:1]
    ]
    features = build_context_features(
        events,
        cfg=cfg,
        session=SessionConfig(style="fingerstyle"),
        baseline=baseline,
        segment_hints=(SegmentHint(1, 5), SegmentHint(1, 5), SegmentHint()),
    )

    assert features.event_features.shape == (3, EVENT_FEATURE_DIM)
    assert features.candidate_features.shape == (3, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM)
    assert features.candidate_mask.sum(axis=1).tolist() == [6, 5, 1]
    logits = np.tile(np.arange(MAX_CANDIDATES, dtype=np.float64), (3, 1))
    probabilities = masked_softmax(logits, features.candidate_mask)
    reranked = apply_context_probabilities(events, features, probabilities, cfg=cfg)

    assert [event.pitch_midi for event in reranked] == [64, 67, 40]
    for event in reranked:
        candidates = candidate_positions(event.pitch_midi, cfg)
        if len(candidates) == 1:
            assert event.fret_prior is None
            continue
        nonzero = np.argwhere(event.fret_prior > 0)
        positions = {(int(string), int(fret)) for string, fret in nonzero}
        assert positions == {(candidate.string_idx, candidate.fret) for candidate in candidates}
        assert all(cfg.tuning_midi[string] + fret == event.pitch_midi for string, fret in positions)


def test_uniform_context_evidence_is_neutral() -> None:
    cfg = GuitarConfig()
    prior = _prior(cfg, 64, (0.05, 0.1, 0.15, 0.2, 0.2, 0.3))
    events = [_event(0.0, 64, prior)]
    features = build_context_features(events, cfg=cfg)
    probabilities = masked_softmax(np.zeros((1, MAX_CANDIDATES)), features.candidate_mask)

    reranked = apply_context_probabilities(events, features, probabilities, cfg=cfg)

    np.testing.assert_allclose(reranked[0].fret_prior, prior, atol=1e-12)


def test_cluster_safe_windows_overlap_and_merge_once_per_event() -> None:
    cluster_ids = np.repeat(np.arange(8), 2)
    windows = context_windows(cluster_ids, max_events=8, overlap_events=2)

    assert windows == (
        tuple(range(0, 8)),
        tuple(range(6, 14)),
        tuple(range(12, 16)),
    )
    for window in windows:
        assert len(window) <= 8
        assert all(
            not ((index in window) ^ (index + 1 in window))
            for index in range(0, len(cluster_ids), 2)
        )
    logits = [
        np.full((len(window), MAX_CANDIDATES), float(index), dtype=np.float32)
        for index, window in enumerate(windows, start=1)
    ]
    merged = merge_window_logits(len(cluster_ids), windows, logits)
    assert merged.shape == (len(cluster_ids), MAX_CANDIDATES)
    np.testing.assert_allclose(merged[6:8], 1.5)
    np.testing.assert_allclose(merged[12:14], 2.5)


def test_masked_models_are_small_scriptable_and_mask_invalid_candidates() -> None:
    torch = pytest.importorskip("torch")
    event = torch.zeros((1, 4, EVENT_FEATURE_DIM))
    candidates = torch.zeros((1, 4, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM))
    mask = torch.tensor([[[True, True, False, False, False, False]] * 4])
    padding = torch.zeros((1, 4), dtype=torch.bool)

    for model in (make_masked_linear_model(), make_context_model(dropout=0.0)):
        model.eval()
        assert parameter_count(model) < 500_000
        eager = model(event, candidates, mask, padding)
        assert torch.all(eager[..., 2:] == -1.0e9)
        scripted = torch.jit.script(model)
        actual = scripted(event, candidates, mask, padding)
        torch.testing.assert_close(actual, eager)

    context = make_context_model(dropout=0.0).eval()
    padded_event = torch.zeros((1, 6, EVENT_FEATURE_DIM))
    padded_candidates = torch.zeros((1, 6, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM))
    padded_mask = torch.tensor([[[True, True, False, False, False, False]] * 6])
    padded_padding = torch.tensor([[False, False, False, False, True, True]])
    unpadded = context(event, candidates, mask, padding)
    padded = context(padded_event, padded_candidates, padded_mask, padded_padding)
    torch.testing.assert_close(unpadded, padded[:, :4], atol=1e-6, rtol=1e-6)


def test_feature_builder_rejects_misaligned_segment_hints() -> None:
    with pytest.raises(ValueError, match="one hint per playable event"):
        build_context_features([_event(0.0, 64)], segment_hints=())
