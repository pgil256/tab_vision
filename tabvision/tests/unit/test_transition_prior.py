"""Unit tests for the A15 learned transition prior + playability wiring."""

from __future__ import annotations

import math

import pytest

from tabvision.fusion import playability
from tabvision.fusion.candidates import Candidate
from tabvision.fusion.transition_prior import (
    TransitionPrior,
    extract_transitions,
    learn_transition_prior,
    load_transition_prior,
)
from tabvision.types import GuitarConfig, TabEvent


def _note(onset_s: float, string_idx: int, fret: int, cfg: GuitarConfig) -> TabEvent:
    return TabEvent(
        onset_s=onset_s,
        duration_s=0.2,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=cfg.tuning_midi[string_idx] + fret,
        confidence=1.0,
    )


@pytest.fixture(autouse=True)
def _clean_prior_state():
    """Never leak an installed prior (module-global) into other tests."""
    yield
    playability.set_transition_prior(None, 1.0)


def test_extract_transitions_single_line():
    cfg = GuitarConfig()
    track = [
        _note(0.0, 2, 2, cfg),  # D string fret 2 (E3, 52)
        _note(0.5, 2, 4, cfg),  # D string fret 4 (F#3, 54)
        _note(1.0, 3, 2, cfg),  # G string fret 2 (A3, 57)
    ]
    transitions = extract_transitions(track)
    assert transitions == [(2, 0, 2), (3, 1, 4)]


def test_extract_transitions_chord_cluster_uses_anchor():
    cfg = GuitarConfig()
    # Cluster at t=0 within the 80 ms chain: anchor = lowest pressed fret.
    track = [
        _note(0.00, 0, 3, cfg),
        _note(0.05, 1, 2, cfg),  # anchor (lowest fret pressed, fret 2)
        _note(0.09, 2, 0, cfg),  # open string: not an anchor candidate
        _note(1.00, 1, 5, cfg),
    ]
    transitions = extract_transitions(track)
    assert len(transitions) == 1
    dp, ds, prev_fret = transitions[0]
    assert prev_fret == 2
    assert ds == 0  # string 1 -> string 1
    assert dp == 3  # fret 2 -> fret 5 on the same string


def test_learned_delta_prior_prefers_observed_pattern():
    cfg = GuitarConfig()
    # Corpus: ascending fourths (+5 semitones) always cross up one string.
    track = []
    t = 0.0
    for _ in range(30):
        track.extend([_note(t, 1, 5, cfg), _note(t + 0.5, 2, 5, cfg)])
        t += 2.0
    prior = learn_transition_prior([track], scheme="delta", alpha=0.5)

    prev = Candidate(string_idx=1, fret=5)
    up_one_string = Candidate(string_idx=2, fret=5)
    same_string = Candidate(string_idx=1, fret=10)
    assert prior.prob(prev, up_one_string, cfg) > prior.prob(prev, same_string, cfg)
    assert prior.cost(prev, up_one_string, cfg) < prior.cost(prev, same_string, cfg)


def test_unseen_delta_pitch_is_ranking_neutral():
    cfg = GuitarConfig()
    prior = TransitionPrior(scheme="delta", delta_table={})
    prev = Candidate(string_idx=2, fret=5)
    costs = {
        ds: prior.cost(prev, Candidate(string_idx=2 + ds, fret=5), cfg) for ds in (-1, 0, 1, 2)
    }
    assert len({round(c, 12) for c in costs.values()}) == 1


def test_delta_fret_backoff_tracks_delta_when_region_unseen():
    cfg = GuitarConfig()
    track = []
    t = 0.0
    for _ in range(30):
        track.extend([_note(t, 1, 5, cfg), _note(t + 0.5, 2, 5, cfg)])
        t += 2.0
    prior = learn_transition_prior([track], scheme="delta_fret", alpha=0.5, backoff_kappa=8.0)

    # Prev fret 12 (region 10+) was never observed -> falls back to delta table.
    prev = Candidate(string_idx=1, fret=12)
    curr = Candidate(string_idx=2, fret=12)
    delta_prior = learn_transition_prior([track], scheme="delta", alpha=0.5)
    assert prior.prob(prev, curr, cfg) == pytest.approx(delta_prior.prob(prev, curr, cfg))


def test_extract_transitions_singleton_only_skips_chord_moves():
    cfg = GuitarConfig()
    track = [
        _note(0.00, 1, 2, cfg),  # singleton
        _note(0.50, 1, 4, cfg),  # singleton  -> kept (singleton->singleton)
        _note(1.00, 0, 3, cfg),  # chord (two notes within 80 ms)
        _note(1.05, 1, 5, cfg),
        _note(2.00, 2, 2, cfg),  # singleton -> chord->singleton is skipped
    ]
    all_moves = extract_transitions(track)
    singles = extract_transitions(track, singleton_only=True)
    assert len(all_moves) == 3
    assert len(singles) == 1
    assert singles[0] == (2, 0, 2)  # fret 2 -> fret 4 on string 1


@pytest.mark.parametrize("name", ["guitarset-seq-v1", "pdmx-seq-v1", "guitarset-pdmx-seq-v1"])
def test_named_artifact_loads_and_matches_musician_conventions(name):
    cfg = GuitarConfig()
    prior = load_transition_prior(name)
    assert prior.scheme == "delta_fret"

    # Unison repeat (Δpitch 0) overwhelmingly stays on the same string.
    prev = Candidate(string_idx=2, fret=5)
    stay = Candidate(string_idx=2, fret=5)
    hop = Candidate(string_idx=3, fret=0)
    assert prior.prob(prev, stay, cfg) > prior.prob(prev, hop, cfg)


def test_env_var_installs_named_prior(monkeypatch):
    monkeypatch.setenv("TABVISION_TRANSITION_PRIOR", "guitarset-seq-v1")
    playability._TRANSITION_PRIOR = None
    playability._TRANSITION_PRIOR_ENV_READ = False
    prior = playability.active_transition_prior()
    assert prior is not None
    assert prior.scheme == "delta_fret"


def test_transition_cost_gate_flag_skips_prior():
    cfg = GuitarConfig()
    prev = Candidate(string_idx=1, fret=5)
    curr = Candidate(string_idx=2, fret=5)

    playability.set_transition_prior(None, 1.0)
    base = playability.transition_cost(prev, curr, cfg)

    prior = TransitionPrior(scheme="delta", delta_table={})
    playability.set_transition_prior(prior, 3.0)
    gated_off = playability.transition_cost(prev, curr, cfg, use_sequence_prior=False)
    assert gated_off == pytest.approx(base)
    gated_on = playability.transition_cost(prev, curr, cfg, use_sequence_prior=True)
    assert gated_on > base


def test_transition_cost_default_off_and_install():
    cfg = GuitarConfig()
    prev = Candidate(string_idx=1, fret=5)
    curr = Candidate(string_idx=2, fret=5)

    playability.set_transition_prior(None, 1.0)
    base = playability.transition_cost(prev, curr, cfg)

    table = {0: tuple([0.9] + [0.1 / 10] * 10)}  # arbitrary non-uniform row for dp=0
    prior = TransitionPrior(scheme="delta", delta_table=table)
    playability.set_transition_prior(prior, 2.0)
    with_prior = playability.transition_cost(prev, curr, cfg)
    # dp for this pair: (tuning[2]+5) - (tuning[1]+5) = 5 -> unseen -> uniform 1/11.
    assert with_prior == pytest.approx(base + 2.0 * -math.log(1.0 / 11.0))

    playability.set_transition_prior(None)
    assert playability.transition_cost(prev, curr, cfg) == pytest.approx(base)
