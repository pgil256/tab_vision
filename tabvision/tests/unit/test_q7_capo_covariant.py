"""Tests for the Q7 capo-covariant prior transform.

The one load-bearing claim is the index arithmetic: a capo-``C`` session must
read the capo-0 prior ``C`` frets and ``C`` semitones lower. Get that wrong
and the probe's whole comparison is meaningless, so it is asserted directly
against a prior with a known, asymmetric shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.eval.q7_capo_covariant_probe import _covariant_score, _naive_score
from tabvision.fusion.position_prior import PitchPositionPrior
from tabvision.types import GuitarConfig, SessionConfig

CFG = GuitarConfig()


def _prior_with_known_cell() -> PitchPositionPrior:
    """A prior where pitch P puts all its mass on (string 2, fret 5)."""
    by_pitch = {}
    for pitch in range(40, 90):
        matrix = np.full((CFG.n_strings, CFG.max_fret + 1), 1e-6, dtype=np.float64)
        matrix[2, 5] = 1.0
        by_pitch[pitch] = matrix / matrix.sum()
    return PitchPositionPrior(by_pitch=by_pitch)


def test_capo_zero_is_a_no_op() -> None:
    prior = _prior_with_known_cell()
    for string in range(6):
        for fret in range(6):
            cov = _covariant_score(prior, 64, string, fret, 0)
            naive = _naive_score(prior, 64, string, fret)
            assert cov == naive


def test_covariant_reads_the_shifted_cell() -> None:
    prior = _prior_with_known_cell()
    # Under capo 3, a note sounding pitch 67 on (string 2, fret 8) has
    # relative position (2, 5) at the capo-0 pitch 64 — the loaded cell.
    hot = _covariant_score(prior, 67, 2, 8, 3)
    cold = _covariant_score(prior, 67, 2, 7, 3)  # relative fret 4, not the peak
    assert hot is not None and cold is not None
    assert hot > cold


def test_naive_ignores_the_capo_and_reads_the_wrong_cell() -> None:
    prior = _prior_with_known_cell()
    # Same note as above, but naive looks up (pitch 67, fret 8) directly. The
    # prior's mass for pitch 67 is at fret 5, so fret 8 is not the peak —
    # the capo-ignorant lookup misses.
    peak = _naive_score(prior, 67, 2, 5)
    off = _naive_score(prior, 67, 2, 8)
    assert peak is not None and off is not None
    assert peak > off


def test_out_of_range_shift_returns_none() -> None:
    prior = _prior_with_known_cell()
    # fret - capo negative is unreachable, and a pitch below the learned
    # range has no matrix — both must decline rather than fabricate a score.
    assert _covariant_score(prior, 41, 0, 0, 5) is None
    assert _covariant_score(prior, 30, 0, 10, 2) is None


# --- routing (2026-07-24 decision: capo>0 uses the covariant prior) ---


def _capo_policy(capo: int, **kwargs: object):
    from tabvision.fusion.inference_policy import resolve_inference_policy

    return resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=GuitarConfig(capo=capo),
        session=SessionConfig(),
        audio_backend_name="highres-ensemble",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("capo", [1, 2, 4, 7])
def test_capo_sessions_get_the_position_prior(capo: int) -> None:
    """Capo>0 used to route to priors=none, a measured collapse.

    Q7: 0.2956 Tab F1 at capo 2 vs a 0.6773 capo-0 control, string wrong on
    two thirds of notes, because the no-prior fallback prefers low frets and
    under a capo every candidate is at fret >= capo.
    """
    assert _capo_policy(capo).resolved_position_prior == "guitarset-v1"


@pytest.mark.parametrize("capo", [1, 2, 4, 7])
def test_capo_suppresses_the_sequence_prior(capo: int) -> None:
    """The sequence artifact is not capo-invariant, so it stays off.

    ``delta_fret`` conditions on the absolute previous-fret region. Q7
    measured the pairing rather than assuming: covariant+seq 0.6766 vs
    covariant 0.6827 at capo 2.
    """
    policy = _capo_policy(capo)
    assert policy.resolved_sequence_prior == "none"
    assert "not capo-invariant" in policy.resolution_reason


@pytest.mark.parametrize("capo", [1, 2, 4, 7])
def test_capo_still_abstains_from_the_physics_channel(capo: int) -> None:
    """B0 describes the *open* string; a capo moves length and tension."""
    assert _capo_policy(capo).resolved_string_evidence == "none"


def test_capo_zero_is_unchanged() -> None:
    policy = _capo_policy(0)
    assert policy.resolved_position_prior == "guitarset-v1"
    assert policy.resolved_sequence_prior == "guitarset-seq-v1"
    assert policy.resolved_string_evidence == "acoustic-physics-v1"


def test_capo_routing_does_not_leak_to_other_instruments() -> None:
    from tabvision.fusion.inference_policy import resolve_inference_policy

    for instrument in ("classical", "electric"):
        policy = resolve_inference_policy(
            requested_position_prior="auto",
            requested_sequence_prior="auto",
            requested_string_evidence="auto",
            cfg=GuitarConfig(capo=3),
            session=SessionConfig(instrument=instrument),
            audio_backend_name="highres",
        )
        # Unmeasured under capo; keeps the acoustic change from silently
        # widening. Classical+capo remains a known gap.
        assert policy.resolved_position_prior == "none"


def test_alternate_tuning_with_capo_still_abstains() -> None:
    from tabvision.fusion.inference_policy import resolve_inference_policy

    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=GuitarConfig(capo=2, tuning_midi=(38, 45, 50, 55, 59, 64)),
        session=SessionConfig(),
        audio_backend_name="highres",
    )
    assert policy.resolved_position_prior == "none"
