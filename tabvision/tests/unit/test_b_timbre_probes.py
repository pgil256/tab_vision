"""Tests for the Track B complement and separability probes.

The probes' conclusions rest on two things being right: the oracle really is an
oracle (all mass on the gold string, so nothing achievable can beat it), and the
AUC really is chance-centred (so "0.5" means what the report says it means). Both
are cheap to pin and neither is obvious from reading the code.
"""

from __future__ import annotations

import numpy as np

from scripts.eval.b_timbre_complement import ambiguous, gold_string_for, oracle_prior
from scripts.eval.b_timbre_separability import pair_auc
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


def _audio_event(pitch: int, onset: float) -> AudioEvent:
    return AudioEvent(
        onset_s=onset,
        offset_s=onset + 0.4,
        pitch_midi=pitch,
        velocity=0.8,
        confidence=0.9,
    )


def _tab_event(pitch: int, string_idx: int, fret: int, onset: float) -> TabEvent:
    return TabEvent(
        onset_s=onset,
        duration_s=0.4,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch,
        confidence=1.0,
    )


def test_oracle_puts_all_mass_on_the_gold_string() -> None:
    cfg = GuitarConfig()
    event = _audio_event(64, 0.0)  # E4 — playable on several strings
    candidates = candidate_positions(64, cfg)
    target = candidates[-1].string_idx
    matrix = oracle_prior(event, target, cfg)
    assert matrix is not None
    # Every non-zero cell is on the gold string, and at least one exists.
    strings = {int(s) for s in np.argwhere(matrix > 0)[:, 0]}
    assert strings == {target}


def test_oracle_returns_none_when_the_gold_string_cannot_play_the_pitch() -> None:
    cfg = GuitarConfig()
    event = _audio_event(40, 0.0)  # low E — only the lowest string
    playable = {c.string_idx for c in candidate_positions(40, cfg)}
    impossible = next(s for s in range(cfg.n_strings) if s not in playable)
    assert oracle_prior(event, impossible, cfg) is None


def test_ambiguous_matches_the_candidate_count() -> None:
    cfg = GuitarConfig()
    for pitch in (40, 64, 67):
        expected = len(candidate_positions(pitch, cfg)) > 1
        assert ambiguous(_audio_event(pitch, 0.0), cfg) is expected


def test_gold_string_matches_on_pitch_within_tolerance() -> None:
    gold = [_tab_event(64, 2, 9, 1.000), _tab_event(64, 1, 14, 5.000)]
    # Within tolerance of the first.
    assert gold_string_for(_audio_event(64, 1.020), gold) == 2
    # Nearer the second.
    assert gold_string_for(_audio_event(64, 4.990), gold) == 1
    # Outside tolerance of both.
    assert gold_string_for(_audio_event(64, 3.000), gold) is None
    # Right time, wrong pitch.
    assert gold_string_for(_audio_event(65, 1.000), gold) is None


def test_pair_auc_is_chance_centred_and_orientation_correct() -> None:
    # Perfectly separated, positives higher.
    assert pair_auc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0
    # Perfectly separated the other way.
    assert pair_auc(np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])) == 0.0
    # Identical distributions land on chance.
    same = np.array([1.0, 2.0, 3.0])
    assert pair_auc(same, same.copy()) == 0.5


def test_pair_auc_handles_ties_without_bias() -> None:
    """All-tied scores must read as chance, not as a win for either side."""
    assert pair_auc(np.ones(5), np.ones(5)) == 0.5
