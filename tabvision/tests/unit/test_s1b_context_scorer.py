"""Tests for the Q2 contextual model's inference adapter.

``ContextScorer`` pads a track up to a multiple of the window, reshapes into
windows, and flattens the logits back. A reshape or trim that is off by one
would silently attribute note *i*'s distribution to note *j* — the rescoring
gate would still produce a plausible-looking number, so the alignment is
asserted directly against a model whose output is a known function of the
input.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.eval.s1b_rescore_lattice import LatticeNote, Track
from scripts.eval.s1b_train_context import (
    PITCH_LOW,
    WINDOW,
    ContextScorer,
    ContextStringModel,
    gap_bucket,
)


class _EchoModel(torch.nn.Module):
    """Logits encode the input pitch token, so alignment is observable."""

    def forward(self, pitch: torch.Tensor, _gap: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*pitch.shape, 6)
        logits[..., 0] = pitch.float()
        return logits

    def eval(self) -> _EchoModel:  # ContextScorer calls this
        return self


def _track(pitches: list[int]) -> Track:
    return Track(
        track_id="t",
        mode="solo",
        notes=[
            LatticeNote(
                track_id="t",
                mode="solo",
                event_index=index,
                cluster_index=index,
                onset_s=index * 0.25,
                pitch_midi=pitch,
                candidates=((0, pitch - PITCH_LOW, 0.0),),
                gold_string=0,
                gold_fret=pitch - PITCH_LOW,
                ambiguous=False,
            )
            for index, pitch in enumerate(pitches)
        ],
    )


def test_scorer_rows_stay_aligned_across_window_boundaries() -> None:
    # Longer than one window and not a multiple of it — the padded case.
    pitches = [PITCH_LOW + (index % 40) for index in range(WINDOW + 17)]
    scorer = ContextScorer(_EchoModel())  # type: ignore[arg-type]
    rows = scorer.log_probs(_track(pitches))

    assert rows.shape == (len(pitches), 6)
    # The echo model is elementwise, so each note's row is fully determined
    # by its own pitch token. Compare exactly: any padding trim or reshape
    # slip shows up as a shifted row rather than a subtly wrong ranking.
    tokens = torch.tensor([[float(pitch - PITCH_LOW + 1)] for pitch in pitches])
    expected_logits = torch.cat([tokens, torch.zeros(len(pitches), 5)], dim=1)
    expected = torch.log_softmax(expected_logits, dim=-1).numpy()
    np.testing.assert_allclose(rows, expected, rtol=1e-6)


def test_scorer_returns_normalized_log_probabilities() -> None:
    torch.manual_seed(0)
    scorer = ContextScorer(ContextStringModel(d_model=32, layers=1, heads=2))
    rows = scorer.log_probs(_track([PITCH_LOW + index % 24 for index in range(70)]))
    assert rows.shape == (70, 6)
    np.testing.assert_allclose(np.exp(rows).sum(axis=1), 1.0, rtol=1e-5)


def test_scorer_handles_empty_track() -> None:
    scorer = ContextScorer(ContextStringModel(d_model=32, layers=1, heads=2))
    assert scorer.log_probs(Track(track_id="t", mode="solo", notes=[])).shape == (0, 6)


def test_gap_buckets_separate_cluster_from_phrase_timing() -> None:
    buckets = gap_bucket(np.asarray([0, 5, 30, 90, 5000]))
    assert buckets[0] == buckets[1]  # both inside the 10 ms cluster bucket
    assert buckets[1] < buckets[2] < buckets[3] < buckets[4]
    assert buckets.min() >= 1  # 0 is reserved for padding
