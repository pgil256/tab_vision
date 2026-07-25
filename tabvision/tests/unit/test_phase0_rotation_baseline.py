"""Tests for the Phase 0 rotation / baseline / decomposition runner.

Two properties carry this run's validity, and both are cheap to assert.

The **split** must be a genuine partition: the sealed player cannot appear in
development, or the confirmation set is scored by a configuration that trained
on it. And the **leave-one-out priors** must actually hold their player out — a
fold that silently trained on everybody would inflate every number in the report
while looking completely normal, which is the failure mode this whole rotation
exists to prevent.

The third property — that the harness measures the configuration that *ships* —
cannot be asserted from a unit test, because it depends on real audio. It is
enforced at runtime by the reproduction check against player 05's published
numbers. That check exists because an earlier pass of the script measured the
strict-isolation arm while reporting itself as shipped.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.phase0_rotation_baseline import (
    ALL_PLAYERS,
    BUCKETS,
    BURNED_PLAYER,
    DEV_PLAYERS,
    REPRODUCTION,
    SEALED_PLAYER,
    build_loo_priors,
    loss_shares,
    sum_decompositions,
)
from tabvision.eval.error_decomposition import ErrorDecomposition
from tabvision.types import GuitarConfig, TabEvent


def _event(pitch: int, string_idx: int, fret: int, onset: float) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.25,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def test_split_is_a_partition_and_the_burned_player_is_back_in_dev() -> None:
    assert SEALED_PLAYER not in DEV_PLAYERS
    assert set(DEV_PLAYERS) | {SEALED_PLAYER} == set(ALL_PLAYERS)
    assert len(DEV_PLAYERS) == len(ALL_PLAYERS) - 1
    # Player 05 was the old sealed set; it is spent, so it must be a dev player
    # now. If it were still held out the rotation would not have happened.
    assert BURNED_PLAYER in DEV_PLAYERS
    assert BURNED_PLAYER != SEALED_PLAYER


def test_loo_priors_actually_hold_their_player_out() -> None:
    """A fold must not have seen the held-out player's fingerings.

    Player ``A`` plays pitch 64 only at string 1 / fret 0; player ``B`` plays the
    same pitch only at string 2 / fret 5. Each fold trains on the *other* one, so
    the mass must sit entirely on the other player's position.
    """
    cfg = GuitarConfig()
    gold = {
        "A": {"A_track": [_event(64, 1, 0, 0.0), _event(64, 1, 0, 1.0)]},
        "B": {"B_track": [_event(64, 2, 5, 0.0), _event(64, 2, 5, 1.0)]},
    }
    import scripts.eval.phase0_rotation_baseline as module

    original = module.ALL_PLAYERS
    module.ALL_PLAYERS = ("A", "B")
    try:
        positions, sequences = build_loo_priors(gold, cfg)
    finally:
        module.ALL_PLAYERS = original

    assert set(positions) == {"A", "B"}
    assert set(sequences) == {"A", "B"}

    # A's fold saw only B, so B's position must dominate, and vice versa.
    a_matrix = positions["A"].matrix_for_pitch(64)
    b_matrix = positions["B"].matrix_for_pitch(64)
    assert a_matrix is not None and b_matrix is not None
    assert a_matrix[2, 5] > a_matrix[1, 0]
    assert b_matrix[1, 0] > b_matrix[2, 5]


def test_sum_decompositions_totals_every_bucket() -> None:
    items = [
        ErrorDecomposition(
            correct=3,
            wrong_position_same_pitch=2,
            pitch_off=1,
            timing_only=0,
            missed_onset=4,
            extra_detection=1,
        ),
        ErrorDecomposition(
            correct=1,
            wrong_position_same_pitch=1,
            pitch_off=0,
            timing_only=2,
            missed_onset=0,
            extra_detection=3,
        ),
    ]
    totals = sum_decompositions(items)
    assert totals == {
        "correct": 4,
        "wrong_position_same_pitch": 3,
        "pitch_off": 1,
        "timing_only": 2,
        "missed_onset": 4,
        "extra_detection": 4,
    }
    assert set(totals) == set(BUCKETS)


def test_loss_shares_exclude_correct_and_sum_to_one() -> None:
    totals = {
        "correct": 100,
        "wrong_position_same_pitch": 5,
        "pitch_off": 2,
        "timing_only": 1,
        "missed_onset": 1,
        "extra_detection": 1,
    }
    shares = loss_shares(totals)
    assert "correct" not in shares
    assert np.isclose(sum(shares.values()), 1.0)
    # 5 of the 10 loss events, not 5 of the 110 total.
    assert np.isclose(shares["wrong_position_same_pitch"], 0.5)


def test_loss_shares_are_zero_when_nothing_was_lost() -> None:
    totals = {name: 0 for name in BUCKETS}
    totals["correct"] = 42
    assert set(loss_shares(totals).values()) == {0.0}


def test_reproduction_targets_are_the_published_shipped_numbers() -> None:
    """Pins the check to the *shipped* arm, not the strict one.

    0.7346 is `raw-partial` from player05_batched_confirm_2026-07-24; 0.7119 is
    `raw-strict`. An earlier pass of the runner measured the latter while
    labelling itself shipped, so this assertion is the regression guard.
    """
    assert REPRODUCTION["baseline"] == 0.6340
    assert REPRODUCTION["shipped"] == 0.7346
    assert REPRODUCTION["shipped"] != 0.7119
