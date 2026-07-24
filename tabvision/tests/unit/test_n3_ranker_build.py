"""Tests for the N3 ranker-build primitives.

The replay's wrong-reduction is the metric the whole comparison reports, and
the physics feature must be a no-op on abstained rows — both are asserted
directly, since a subtle error in either silently changes the headline delta.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.n3_ranker_build import Row, build_features, replay


def _row(event_id: str, track: str, *, wrong: bool, top3: bool, idx: int) -> Row:
    return Row(
        event_id=event_id,
        track_id=track,
        player=track[:2],
        mode="solo",
        event_index=idx,
        onset_ms=idx * 500,
        pitch=64,
        predicted_string=1,
        candidates=((1, 14, 0.0), (2, 9, 0.4), (3, 4, 1.1)),
        reference_in_top3=top3,
        wrong=wrong,
        cluster_size=1,
    )


def test_replay_reduction_counts_correctable_selected_wrong_notes() -> None:
    # One track, 3 wrong notes; risk surfaces them in order. At 10 s the budget
    # is 5 reviews (10/2), so all are selectable; reduction is the share of
    # wrong notes that are correctable (gold in top-3).
    rows = [
        _row("a", "00_x", wrong=True, top3=True, idx=0),
        _row("b", "00_x", wrong=True, top3=True, idx=1),
        _row("c", "00_x", wrong=True, top3=False, idx=2),  # not correctable
        _row("d", "00_x", wrong=False, top3=True, idx=3),
    ]
    risk = np.array([0.9, 0.8, 0.7, 0.1])
    out = replay(rows, risk)
    # 3 wrong total, 2 correctable and both selected within budget -> 2/3.
    assert out[10] == 2 / 3


def test_replay_budget_limits_reviews() -> None:
    # 6 wrong correctable notes on one track; at 2 s/note a 2 s budget reviews
    # only the single highest-risk note.
    rows = [_row(str(i), "00_x", wrong=True, top3=True, idx=i) for i in range(6)]
    risk = np.arange(6, dtype=float)[::-1]  # row 0 highest risk
    out = replay(rows, risk)
    # budget 10 -> 5 reviews of 6 wrong -> 5/6.
    assert out[10] == 5 / 6


def test_physics_feature_is_zero_when_abstained() -> None:
    rows = [_row("a", "00_x", wrong=True, top3=True, idx=0)]
    # No physics entry for "a" -> the three physics columns stay zero.
    feats = build_features(rows, physics={}, with_physics=True)
    assert feats.shape == (1, 7)
    assert list(feats[0, 4:]) == [0.0, 0.0, 0.0]


def test_physics_feature_encodes_doubt() -> None:
    rows = [_row("a", "00_x", wrong=True, top3=True, idx=0)]
    # prob_decoder low -> doubt feature (1 - prob) high.
    feats = build_features(
        rows, physics={"a": {"prob_decoder": 0.1, "r2": 0.9, "fired": 1.0}}, with_physics=True
    )
    assert feats[0, 4] == 0.9  # 1 - 0.1
    assert feats[0, 5] == 0.9  # r2
    assert feats[0, 6] == 1.0  # fired


def test_decoder_arm_has_no_physics_columns() -> None:
    rows = [_row("a", "00_x", wrong=True, top3=True, idx=0)]
    assert build_features(rows, physics={}, with_physics=False).shape == (1, 4)
