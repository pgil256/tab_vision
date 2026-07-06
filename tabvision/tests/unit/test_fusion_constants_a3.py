"""A3/A4 — env-overridable fusion constants + time-scaled transitions.

The sweep harness rebinds these module globals at runtime, so the cost
functions must read them per call. A4's gap-decay must be a no-op at the
default ``TRANSITION_GAP_TAU = inf`` and shrink the hand-continuity terms as the
inter-onset gap grows when TAU is finite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tabvision.fusion.chord as chord
import tabvision.fusion.playability as play
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.playability import emission_cost, transition_cost
from tabvision.types import AudioEvent, GuitarConfig


def _ev(midi: int, *, fret_prior: np.ndarray | None = None) -> AudioEvent:
    return AudioEvent(
        onset_s=0.0,
        offset_s=0.2,
        pitch_midi=midi,
        velocity=0.8,
        confidence=0.8,
        fret_prior=fret_prior,
    )


# --- A3: FRET_PRIOR_WEIGHT -------------------------------------------------------


def test_fret_prior_weight_zero_removes_prior_term(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GuitarConfig()
    arr = np.full((cfg.n_strings, cfg.max_fret + 1), 1e-3)
    arr[3, 9] = 1.0
    cand = Candidate(string_idx=3, fret=9)
    monkeypatch.setattr(play, "FRET_PRIOR_WEIGHT", 0.0)
    with_prior = emission_cost(cand, _ev(64, fret_prior=arr), None, cfg)
    without_prior = emission_cost(cand, _ev(64), None, cfg)
    assert with_prior == pytest.approx(without_prior)


def test_fret_prior_weight_scales_prior_term(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GuitarConfig()
    arr = np.full((cfg.n_strings, cfg.max_fret + 1), 1e-3)
    arr[3, 9] = 0.5  # a sub-1 prob so -log is positive and weight-scaling is visible
    cand = Candidate(string_idx=3, fret=9)
    monkeypatch.setattr(play, "FRET_PRIOR_WEIGHT", 1.0)
    base = emission_cost(cand, _ev(64, fret_prior=arr), None, cfg)
    monkeypatch.setattr(play, "FRET_PRIOR_WEIGHT", 2.0)
    doubled = emission_cost(cand, _ev(64, fret_prior=arr), None, cfg)
    # The extra cost over the no-prior term doubles.
    no_prior = emission_cost(cand, _ev(64), None, cfg)
    assert (doubled - no_prior) == pytest.approx(2 * (base - no_prior))


# --- A4: TRANSITION_GAP_TAU -----------------------------------------------------


def test_gap_decay_is_noop_at_default_tau() -> None:
    cfg = GuitarConfig()
    a, b = Candidate(string_idx=0, fret=0), Candidate(string_idx=0, fret=3)
    # Default TAU = inf: any gap_s must give the gap-blind cost.
    assert math.isinf(play.TRANSITION_GAP_TAU)
    blind = transition_cost(a, b, cfg, use_sequence_prior=False)
    assert transition_cost(a, b, cfg, use_sequence_prior=False, gap_s=5.0) == blind


def test_gap_decay_shrinks_continuity_with_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GuitarConfig()
    a, b = Candidate(string_idx=0, fret=0), Candidate(string_idx=0, fret=3)
    blind = transition_cost(a, b, cfg, use_sequence_prior=False)
    monkeypatch.setattr(play, "TRANSITION_GAP_TAU", 1.0)
    # gap_s=None still means no decay; a finite gap decays toward zero.
    assert transition_cost(a, b, cfg, use_sequence_prior=False, gap_s=None) == pytest.approx(blind)
    near = transition_cost(a, b, cfg, use_sequence_prior=False, gap_s=0.1)
    far = transition_cost(a, b, cfg, use_sequence_prior=False, gap_s=3.0)
    assert abs(far) < abs(near) < abs(blind) or abs(far) < abs(blind)
    # The decayed cost equals exp(-gap/TAU) * the gap-blind cost (linear terms).
    assert far == pytest.approx(math.exp(-3.0) * blind)


# --- A3: runtime rebinding reaches the cost functions ---------------------------


def test_rebinding_open_string_bonus_changes_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = GuitarConfig()
    cand = Candidate(string_idx=5, fret=0)  # open high E
    monkeypatch.setattr(play, "OPEN_STRING_BONUS", 0.0)
    no_bonus = emission_cost(cand, _ev(64), None, cfg)
    monkeypatch.setattr(play, "OPEN_STRING_BONUS", 1.0)
    big_bonus = emission_cost(cand, _ev(64), None, cfg)
    assert big_bonus == pytest.approx(no_bonus - 1.0)


def test_cluster_events_reads_module_gap_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    evs = [_ev(40), AudioEvent(0.05, 0.25, 45, 0.8, 0.8), AudioEvent(0.5, 0.7, 50, 0.8, 0.8)]
    monkeypatch.setattr(chord, "CHORD_MAX_GAP_S", 0.08)
    # 0.0 and 0.05 cluster (gap 0.05 <= 0.08); 0.5 is separate.
    assert [len(c) for c in chord.cluster_events(evs)] == [2, 1]
    monkeypatch.setattr(chord, "CHORD_MAX_GAP_S", 0.01)
    # Now nothing clusters.
    assert [len(c) for c in chord.cluster_events(evs)] == [1, 1, 1]
