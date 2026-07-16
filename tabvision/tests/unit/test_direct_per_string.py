from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tabvision.eval.direct_per_string import (  # noqa: E402
    MEL_BANDS,
    PITCH_CLASSES,
    STRINGS,
    WINDOW_SAMPLES,
    DirectOutputs,
    DirectPerStringNet,
    extract_window,
    gold_pitch_string_scores,
    log_mel_batch,
    mel_filterbank,
    multitask_loss,
    parameter_count,
    string_scores_to_fret_prior,
)


def test_model_has_all_heads_and_stays_below_parameter_cap() -> None:
    model = DirectPerStringNet()

    outputs = model(torch.zeros((2, MEL_BANDS, 65)))

    assert outputs.onset_logits.shape == (2, STRINGS, PITCH_CLASSES)
    assert outputs.frame_logits.shape == (2, STRINGS, PITCH_CLASSES)
    assert outputs.global_pitch_logits.shape == (2, PITCH_CLASSES)
    assert outputs.occupancy_logits.shape == (2, STRINGS)
    assert parameter_count(model) < 5_000_000


def test_window_and_log_mel_are_fixed_and_deterministic() -> None:
    signal = np.linspace(-1.0, 1.0, 16_000, dtype=np.float32)
    window = extract_window(signal, 0.0)

    first = log_mel_batch(torch.from_numpy(window[None, :]))
    second = log_mel_batch(torch.from_numpy(window[None, :]))

    assert window.shape == (WINDOW_SAMPLES,)
    assert np.all(window[:1024] == 0.0)
    assert first.shape == (1, MEL_BANDS, 65)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_mel_filterbank_is_finite_nonnegative_and_normalized() -> None:
    filters = mel_filterbank()

    assert filters.shape == (MEL_BANDS, 257)
    assert np.all(np.isfinite(filters))
    assert np.all(filters >= 0.0)
    np.testing.assert_allclose(filters.sum(axis=1), 1.0, atol=1.0e-6)


def test_gold_pitch_scores_gather_matching_pitch_from_six_heads() -> None:
    onset = torch.zeros((2, STRINGS, PITCH_CLASSES))
    frame = torch.zeros_like(onset)
    onset[0, 3, 64 - 21] = 2.0
    frame[0, 3, 64 - 21] = 4.0
    outputs = DirectOutputs(onset, frame, torch.zeros((2, PITCH_CLASSES)), torch.zeros((2, 6)))

    scores = gold_pitch_string_scores(outputs, torch.tensor([64, 69]))

    assert scores.shape == (2, 6)
    assert scores[0, 3] == pytest.approx(4.0)


def test_duplicate_inhibition_penalizes_excess_string_count() -> None:
    onset = torch.zeros((1, STRINGS, PITCH_CLASSES))
    frame_low = torch.full_like(onset, -10.0)
    frame_high = torch.full_like(onset, -10.0)
    frame_high[:, :, 43] = 10.0
    targets = torch.zeros_like(onset)
    targets[:, 2, 43] = 1.0
    global_targets = torch.zeros((1, PITCH_CLASSES))
    global_targets[:, 43] = 1.0
    occupancy = torch.zeros((1, STRINGS))
    occupancy[:, 2] = 1.0

    low = multitask_loss(
        DirectOutputs(onset, frame_low, global_targets, occupancy),
        targets,
        targets,
        global_targets,
        occupancy,
    )
    high = multitask_loss(
        DirectOutputs(onset, frame_high, global_targets, occupancy),
        targets,
        targets,
        global_targets,
        occupancy,
    )

    assert high.duplicate_inhibition > low.duplicate_inhibition + 0.2


def test_string_scores_map_only_to_playable_fret_prior_cells() -> None:
    matrix = string_scores_to_fret_prior(64, np.asarray([-3, -2, -1, 0, 1, 2]))

    assert matrix.shape == (6, 25)
    assert matrix.sum() == pytest.approx(1.0)
    nonzero = {tuple(index) for index in np.argwhere(matrix > 0.0)}
    assert nonzero == {(0, 24), (1, 19), (2, 14), (3, 9), (4, 5), (5, 0)}
