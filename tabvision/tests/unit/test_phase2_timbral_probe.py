from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from scripts.eval.phase2_timbral_probe import (  # noqa: E402
    FEATURE_DIM,
    MAX_CANDIDATES,
    AudioCandidateRanker,
    _candidate_features,
    _combined_predictions,
)


def test_candidate_features_have_fixed_shape_and_mask() -> None:
    arrays = {
        "labels": np.asarray([1], dtype=np.int8),
        "strings": np.asarray([[1, 2, -1, -1, -1, -1]], dtype=np.int8),
        "frets": np.asarray([[7, 2, -1, -1, -1, -1]], dtype=np.int8),
        "pitches": np.asarray([52], dtype=np.int16),
        "modes": np.asarray([0], dtype=np.int8),
        "rms": np.asarray([0.1], dtype=np.float32),
    }
    features, mask = _candidate_features(arrays)
    assert features.shape == (1, MAX_CANDIDATES, FEATURE_DIM)
    assert mask.tolist() == [[True, True, False, False, False, False]]
    assert features[0, 0, 1] == 1.0
    assert features[0, 0, 11] == 1.0  # fingerstyle


def test_audio_ranker_is_under_parameter_cap_and_masks_candidates() -> None:
    model = AudioCandidateRanker().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) < 250_000
    waveform = torch.zeros((2, 8192), dtype=torch.float32)
    features = torch.zeros((2, MAX_CANDIDATES, FEATURE_DIM), dtype=torch.float32)
    mask = torch.tensor(
        [[True, True, False, False, False, False], [True, True, True, False, False, False]]
    )
    with torch.inference_mode():
        logits = model(waveform, features, mask)
    assert logits.shape == (2, MAX_CANDIDATES)
    assert torch.all(logits[~mask] < -1e8)


def test_combiner_never_selects_masked_candidate() -> None:
    prior = np.asarray([[0.7, 0.3, 0, 0, 0, 0]], dtype=np.float32)
    logits = np.asarray([[0, 1, 100, 100, 100, 100]], dtype=np.float32)
    mask = np.asarray([[True, True, False, False, False, False]])
    predicted = _combined_predictions(prior, logits, mask, temperature=1.0, weight=1.0)
    assert predicted.tolist() == [1]
