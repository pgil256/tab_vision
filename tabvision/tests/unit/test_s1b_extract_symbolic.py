"""Tests for the Q2 SynthTab symbolic corpus extraction.

Two things have to hold for the corpus to be usable as pretraining
substrate: the flat-array-plus-offsets layout must round-trip per-track
sequences exactly (a trainer slices windows by those offsets, so an
off-by-one silently mixes two songs into one phrase), and
``ambiguous_note_share`` must count the same notion of ambiguity the Phase 0
lattice scores — a pitch playable at more than one position.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.s1b_extract_symbolic import characterize


def _corpus(tracks: list[list[tuple[int, int, int, int]]]) -> dict[str, np.ndarray]:
    """Build the flat layout from per-track (onset_ms, pitch, string, fret)."""
    onset: list[int] = []
    pitch: list[int] = []
    string: list[int] = []
    fret: list[int] = []
    lengths: list[int] = []
    for notes in tracks:
        lengths.append(len(notes))
        for note in notes:
            onset.append(note[0])
            pitch.append(note[1])
            string.append(note[2])
            fret.append(note[3])
    return {
        "onset_ms": np.asarray(onset, dtype=np.int32),
        "pitch": np.asarray(pitch, dtype=np.int16),
        "string": np.asarray(string, dtype=np.int8),
        "fret": np.asarray(fret, dtype=np.int8),
        "track_offset": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "track_program": np.zeros(len(tracks), dtype=np.int16),
    }


def test_offsets_slice_tracks_without_bleeding() -> None:
    arrays = _corpus([[(0, 40, 0, 0), (500, 45, 1, 0)], [(0, 64, 5, 0)]])
    offsets = arrays["track_offset"]
    assert offsets.tolist() == [0, 2, 3]
    first = arrays["pitch"][offsets[0] : offsets[1]]
    second = arrays["pitch"][offsets[1] : offsets[2]]
    assert first.tolist() == [40, 45]
    assert second.tolist() == [64]


def test_ambiguous_share_matches_playable_positions() -> None:
    # MIDI 40 (low open E) is playable at exactly one position; MIDI 64 is
    # playable on several strings, which is what makes it a decoder problem.
    arrays = _corpus([[(0, 40, 0, 0), (500, 64, 0, 24), (1000, 64, 5, 0)]])
    stats = characterize(arrays)
    assert stats["ambiguous_note_share"] == 2 / 3
    assert stats["mean_positions_per_note"] > 1.0


def test_clusters_use_the_eighty_millisecond_decode_grouping() -> None:
    # Three simultaneous notes, then an isolated one 500 ms later.
    arrays = _corpus([[(0, 40, 0, 0), (30, 47, 1, 2), (60, 52, 2, 2), (560, 64, 5, 0)]])
    stats = characterize(arrays)
    assert stats["clusters"] == 2
    assert stats["mean_cluster_size"] == 2.0
    assert stats["polyphonic_cluster_share"] == 0.5


def test_cluster_grouping_chains_within_gap() -> None:
    # 0/70/140 ms: each gap is under 80 ms, so all three chain into one
    # cluster even though the span exceeds 80 ms — matching the decode's
    # successive-gap rule rather than a fixed window.
    arrays = _corpus([[(0, 40, 0, 0), (70, 47, 1, 2), (140, 52, 2, 2)]])
    stats = characterize(arrays)
    assert stats["clusters"] == 1
    assert stats["mean_cluster_size"] == 3.0
