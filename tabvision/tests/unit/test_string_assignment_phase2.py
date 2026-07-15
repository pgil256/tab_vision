from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts.eval.string_assignment_phase0 import DEV_PLAYERS
from scripts.eval.string_assignment_phase2 import (
    PreparedTrack,
    aggregate_error_counts,
    player_folds,
    train_model,
)
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.context_reranker import (
    CANDIDATE_FEATURE_DIM,
    EVENT_FEATURE_DIM,
    MAX_CANDIDATES,
    ContextFeatures,
)
from tabvision.types import AudioEvent, TabEvent


def test_player_folds_never_mix_the_held_out_player() -> None:
    tracks = [
        SimpleNamespace(player=player, track_id=f"{player}_{index}")
        for player in DEV_PLAYERS
        for index in range(2)
    ]

    folds = player_folds(cast(Any, tracks))

    assert [held_out for held_out, _train, _validation in folds] == list(DEV_PLAYERS)
    for held_out, train, validation in folds:
        assert {track.player for track in validation} == {held_out}
        assert held_out not in {track.player for track in train}


def _tiny_prepared_track() -> PreparedTrack:
    rng = np.random.default_rng(8)
    n_events = 4
    event_features = rng.normal(size=(n_events, EVENT_FEATURE_DIM)).astype(np.float32)
    candidate_features = rng.normal(size=(n_events, MAX_CANDIDATES, CANDIDATE_FEATURE_DIM)).astype(
        np.float32
    )
    candidate_mask = np.zeros((n_events, MAX_CANDIDATES), dtype=np.bool_)
    candidate_mask[:, :2] = True
    features = ContextFeatures(
        event_features,
        candidate_features,
        candidate_mask,
        tuple(((Candidate(0, 0), Candidate(1, 1))) for _ in range(n_events)),
        np.arange(n_events, dtype=np.int64),
    )
    events = tuple(
        AudioEvent(float(index), float(index) + 0.2, 40, 0.8, 0.9) for index in range(n_events)
    )
    tabs = tuple(TabEvent(float(index), 0.2, 0, 0, 40, 1.0) for index in range(n_events))
    track = SimpleNamespace(player="00", track_id="00_tiny", gold=tabs)
    return PreparedTrack(
        cast(Any, track),
        cast(Any, None),
        events,
        features,
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        tabs,
        tabs,
        np.ones(n_events, dtype=np.bool_),
    )


def test_fixed_seed_tiny_training_is_deterministic() -> None:
    pytest.importorskip("torch")
    prepared = _tiny_prepared_track()

    first = train_model("control", [prepared], None, seed=19, fixed_epochs=2)
    second = train_model("control", [prepared], None, seed=19, fixed_epochs=2)

    assert [row["training_loss"] for row in first.history] == [
        row["training_loss"] for row in second.history
    ]
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        np.testing.assert_array_equal(
            first_parameter.detach().numpy(), second_parameter.detach().numpy()
        )


def test_error_diagnostics_are_stored_as_counts_not_note_rows() -> None:
    row = {
        "condition": "baseline",
        "reference_string": 2,
        "predicted_minus_reference_string": 1,
        "fret_displacement": -5,
        "pitch_midi": 64,
        "candidate_count": 4,
        "style": "bn",
        "player": "00",
        "label": "wrong_position_same_pitch",
    }

    counts = aggregate_error_counts([row, row])

    assert len(counts) == 7
    assert {item["count"] for item in counts} == {2}
