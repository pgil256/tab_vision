"""Tests for the Q2 lattice rescoring gate.

The load-bearing invariant is that ``lambda = 0`` reproduces the banked
decoder ranking *exactly*. If it does not, every reported delta is measuring
the harness rather than the scorer — the same failure mode the Q1 merge
pilot guarded with its identity-baseline test.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from scripts.eval.s1b_rescore_lattice import (
    LatticeNote,
    Track,
    UniformScorer,
    load_lattice,
    rescore,
)

FIELDNAMES = [
    "condition",
    "evaluation_split",
    "track_id",
    "mode",
    "event_index",
    "cluster_index",
    "onset_s",
    "pitch_midi",
    "candidate_path",
    "reference_string",
    "reference_fret",
    "ambiguous_pitch_match",
]


class _StringPreferenceScorer:
    """Puts all its mass on one string, so its effect is predictable."""

    name = "test"

    def __init__(self, preferred: int) -> None:
        self.preferred = preferred

    def log_probs(self, track: Track) -> np.ndarray:
        rows = np.full((len(track.notes), 6), math.log(0.01), dtype=np.float64)
        rows[:, self.preferred] = math.log(0.95)
        return rows


def _note(
    *,
    candidates: tuple[tuple[int, int, float], ...],
    gold: tuple[int, int],
    index: int = 0,
    mode: str = "solo",
) -> LatticeNote:
    return LatticeNote(
        track_id="t",
        mode=mode,
        event_index=index,
        cluster_index=index,
        onset_s=float(index),
        pitch_midi=64,
        candidates=candidates,
        gold_string=gold[0],
        gold_fret=gold[1],
        ambiguous=True,
    )


def test_lambda_zero_reproduces_the_decoder_ranking() -> None:
    # Gold sits at rank 2, so the decoder is wrong and must stay wrong.
    track = Track(
        track_id="t",
        mode="solo",
        notes=[_note(candidates=((2, 9, 0.0), (1, 14, 0.4)), gold=(1, 14))],
    )
    summary = rescore([track], UniformScorer(), lambdas=(0.0,))
    assert summary["sweep"][0]["top1"] == 0.0
    assert summary["sweep"][0]["ambiguous_notes"] == 1


def test_uniform_scorer_never_changes_the_ranking() -> None:
    notes = [
        _note(candidates=((2, 9, 0.0), (1, 14, 0.4)), gold=(2, 9), index=0),
        _note(candidates=((3, 4, 0.0), (4, 9, 0.2)), gold=(4, 9), index=1),
    ]
    track = Track(track_id="t", mode="solo", notes=notes)
    summary = rescore([track], UniformScorer(), lambdas=(0.0, 1.0, 8.0, 1e9))
    # One of two gold answers is at rank 1; a constant offset cannot move it.
    assert {row["top1"] for row in summary["sweep"]} == {0.5}


def test_scorer_can_flip_a_rank_two_note_and_is_counted() -> None:
    track = Track(
        track_id="t",
        mode="solo",
        notes=[_note(candidates=((2, 9, 0.0), (1, 14, 0.4)), gold=(1, 14))],
    )
    # Preferring string 1 must overcome the 0.4 cost gap at high lambda.
    summary = rescore([track], _StringPreferenceScorer(1), lambdas=(0.0, 8.0))
    by_lambda = {row["lambda"]: row for row in summary["sweep"]}
    assert by_lambda[0.0]["top1"] == 0.0
    assert by_lambda[8.0]["top1"] == 1.0
    assert by_lambda[8.0]["rank2_notes"] == 1
    assert by_lambda[8.0]["rank2_flip_rate"] == 1.0


def test_solo_and_comp_are_reported_separately() -> None:
    notes = [
        _note(candidates=((2, 9, 0.0), (1, 14, 0.4)), gold=(2, 9), index=0, mode="solo"),
        _note(candidates=((3, 4, 0.0), (4, 9, 0.2)), gold=(4, 9), index=1, mode="comp"),
    ]
    summary = rescore(
        [Track(track_id="t", mode="solo", notes=notes)], UniformScorer(), lambdas=(0.0,)
    )
    row = summary["sweep"][0]
    assert row["solo_top1"] == 1.0
    assert row["comp_top1"] == 0.0


def test_load_lattice_filters_split_and_orders_by_event_index(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        common = {
            "condition": "production_equivalent",
            "track_id": "00_x_solo",
            "mode": "solo",
            "cluster_index": "0",
            "onset_s": "0.5",
            "pitch_midi": "64",
            "candidate_path": "2:9:0.00000000;1:14:0.40000000",
            "reference_string": "1",
            "reference_fret": "14",
            "ambiguous_pitch_match": "1",
        }
        # Deliberately out of order, plus a held-out row that must be excluded.
        writer.writerow({**common, "evaluation_split": "development_oof", "event_index": "1"})
        writer.writerow({**common, "evaluation_split": "development_oof", "event_index": "0"})
        writer.writerow({**common, "evaluation_split": "held_out_05", "event_index": "2"})

    tracks = load_lattice(path, split="development_oof")
    assert len(tracks) == 1
    assert [note.event_index for note in tracks[0].notes] == [0, 1]
    assert tracks[0].notes[0].candidates == ((2, 9, 0.0), (1, 14, 0.4))
    assert tracks[0].notes[0].gold_string == 1


def test_rows_without_gold_are_skipped(tmp_path: Path) -> None:
    unmatched = LatticeNote(
        track_id="t",
        mode="solo",
        event_index=0,
        cluster_index=0,
        onset_s=0.0,
        pitch_midi=64,
        candidates=((2, 9, 0.0), (1, 14, 0.4)),
        gold_string=None,
        gold_fret=None,
        ambiguous=True,
    )
    summary = rescore(
        [Track(track_id="t", mode="solo", notes=[unmatched])], UniformScorer(), lambdas=(0.0,)
    )
    assert summary["sweep"][0]["ambiguous_notes"] == 0
