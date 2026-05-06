"""Unit tests for ``tabvision.fusion.playability``.

Covers:
- emission cost: audio-only ranking matches the legacy greedy decoder's
  preferences (lower fret + open-string bonus).
- emission cost: vision evidence pulls a candidate that audio is
  indifferent on.
- emission cost: open-string bonus correctly recovers fret 0 when the
  vision marginal is uniform.
- transition cost: same-string is cheaper than string-jump.
- transition cost: hand-span barrier triggers only past ``MAX_HAND_SPAN``.
- ``find_fingering_at`` picks the nearest non-empty fingering.
"""

from __future__ import annotations

import numpy as np

from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.fusion.playability import (
    HAND_SPAN_BARRIER,
    MAX_HAND_SPAN,
    OPEN_STRING_BONUS,
    SAME_STRING_BONUS,
    emission_cost,
    find_fingering_at,
    transition_cost,
)
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig

# ---------- helpers ----------


def _ev(midi: int, t: float = 0.0, confidence: float = 0.8) -> AudioEvent:
    return AudioEvent(
        onset_s=t,
        offset_s=t + 0.25,
        pitch_midi=midi,
        velocity=0.8,
        confidence=confidence,
    )


def _peaked_fingering(
    t: float,
    target_string: int,
    target_fret: int,
    n_strings: int = 6,
    max_fret: int = 24,
) -> FrameFingering:
    """Marginal sharply peaked at ``(target_string, target_fret)``."""
    logits = np.zeros((4, n_strings, max_fret + 1), dtype=np.float64)
    logits[0, target_string, target_fret] = 10.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


def _uniform_fingering(t: float, n_strings: int = 6, max_fret: int = 24) -> FrameFingering:
    """Marginal ≈ uniform across (string, fret) cells."""
    logits = np.ones((4, n_strings, max_fret + 1), dtype=np.float64)
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=0.9)


# ---------- emission ----------


def test_emission_audio_only_prefers_lower_fret():
    """Without vision evidence, lowest-fret candidate has lowest emission cost.

    A4 (MIDI 69) candidates: s5f5 (high E, fret 5) and s4f9 (B, fret 9), among
    others. The plain low-fret bias should pick s5f5.
    """
    cfg = GuitarConfig()
    ev = _ev(69)
    cands = candidate_positions(69, cfg)
    costs = [(c, emission_cost(c, ev, None, cfg)) for c in cands]
    best = min(costs, key=lambda kv: kv[1])[0]
    assert best.fret == 5
    assert best.string_idx == 5  # high E


def test_emission_open_string_bonus_recovers_fret_zero():
    """For a pitch with a fret-0 option, the open-string bonus puts it on top.

    E2 (MIDI 40) has only one candidate: s0f0 — the bonus should make its
    emission cost lower than any fingered alternative would have been.
    """
    cfg = GuitarConfig()
    ev = _ev(40)
    cands = candidate_positions(40, cfg)
    assert len(cands) == 1 and cands[0].fret == 0
    open_cost = emission_cost(cands[0], ev, None, cfg)

    # Compare against a synthetic fret-1 candidate's would-be cost: same
    # pitch contribution, but no bonus and one tick of low-fret bias.
    fake = Candidate(string_idx=0, fret=1)
    # Construct a fake AudioEvent with the same confidence so the per-event
    # constant cancels out.
    fake_cost = emission_cost(fake, ev, None, cfg)
    assert open_cost < fake_cost
    assert (fake_cost - open_cost) >= OPEN_STRING_BONUS - 1e-9


def test_emission_vision_pulls_pick_off_lowest_fret():
    """Vision evidence should override the lowest-fret default.

    A4 (MIDI 69) audio-only picks s5f5 (high E, fret 5). With a fingering
    peaked at s2f14 (G string, fret 14 — also a valid A4 position), the
    emission cost there should be lower despite the higher fret.
    """
    cfg = GuitarConfig()
    ev = _ev(69, t=1.0)
    fing = _peaked_fingering(t=1.0, target_string=2, target_fret=14)

    audio_pick = Candidate(string_idx=5, fret=5)
    vision_pick = Candidate(string_idx=2, fret=14)

    audio_cost = emission_cost(audio_pick, ev, fing, cfg, lambda_vision=1.0)
    vision_cost = emission_cost(vision_pick, ev, fing, cfg, lambda_vision=1.0)
    assert vision_cost < audio_cost


def test_emission_uniform_vision_does_not_change_ranking():
    """A uniform fingering should not flip the audio-only preference."""
    cfg = GuitarConfig()
    ev = _ev(69)
    fing = _uniform_fingering(t=0.0)
    cands = candidate_positions(69, cfg)
    pure_audio = sorted(cands, key=lambda c: emission_cost(c, ev, None, cfg))
    with_uniform = sorted(
        cands,
        key=lambda c: emission_cost(c, ev, fing, cfg, lambda_vision=1.0),
    )
    assert [c for c in pure_audio] == [c for c in with_uniform]


# ---------- transition ----------


def test_transition_same_string_is_cheaper_than_string_jump():
    """Same-string continuity bonus beats a one-fret string jump."""
    cfg = GuitarConfig()
    prev = Candidate(string_idx=5, fret=5)
    same_string = Candidate(string_idx=5, fret=7)  # 2 frets up, same string
    string_jump = Candidate(string_idx=4, fret=5)  # different string, same fret
    assert transition_cost(prev, same_string, cfg) < transition_cost(prev, string_jump, cfg)


def test_transition_hand_span_barrier_only_past_threshold():
    """Costs are mild within ``MAX_HAND_SPAN`` and steep beyond it."""
    cfg = GuitarConfig()
    prev = Candidate(string_idx=5, fret=5)
    within = Candidate(string_idx=5, fret=5 + MAX_HAND_SPAN)  # at threshold
    beyond = Candidate(string_idx=5, fret=5 + MAX_HAND_SPAN + 1)  # one past

    cost_within = transition_cost(prev, within, cfg)
    cost_beyond = transition_cost(prev, beyond, cfg)

    # The barrier kicks in for `beyond`, so the gap should be ≥ HAND_SPAN_BARRIER
    # (modulo the small extra position-shift cost of one more fret).
    assert (cost_beyond - cost_within) >= HAND_SPAN_BARRIER - 1e-6


def test_transition_zero_when_unchanged():
    """No-op transition (same string, same fret) yields the bare continuity bonus."""
    cfg = GuitarConfig()
    p = Candidate(string_idx=3, fret=7)
    cost = transition_cost(p, p, cfg)
    # 0 position shift + same-string bonus → -SAME_STRING_BONUS exactly.
    assert cost == -SAME_STRING_BONUS


# ---------- find_fingering_at ----------


def test_find_fingering_at_picks_closest_non_empty():
    fings = [
        _peaked_fingering(t=0.0, target_string=0, target_fret=0),
        _peaked_fingering(t=1.0, target_string=5, target_fret=5),
        _peaked_fingering(t=2.0, target_string=3, target_fret=3),
    ]
    chosen = find_fingering_at(1.1, fings)
    assert chosen is not None
    assert chosen.t == 1.0


def test_find_fingering_at_skips_empty_logits():
    """All-zero logits = no evidence; should be skipped."""
    empty = FrameFingering(
        t=0.5,
        finger_pos_logits=np.zeros((4, 6, 25)),
        homography_confidence=0.0,
    )
    real = _peaked_fingering(t=2.0, target_string=2, target_fret=7)
    chosen = find_fingering_at(0.6, [empty, real])
    assert chosen is real


def test_find_fingering_at_returns_none_when_all_empty():
    empty = FrameFingering(
        t=0.5,
        finger_pos_logits=np.zeros((4, 6, 25)),
        homography_confidence=0.0,
    )
    assert find_fingering_at(0.6, [empty]) is None


def test_find_fingering_at_returns_none_for_empty_input():
    assert find_fingering_at(0.6, []) is None
