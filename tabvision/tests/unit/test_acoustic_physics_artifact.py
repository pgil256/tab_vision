"""Tests for the registered acoustic-physics-v1 string-evidence artifact.

Three things matter here. The artifact must carry the *gate-passed* decode
config rather than inheriting module defaults — otherwise editing a default
would silently change what a registered, hash-verified artifact does. `auto`
must resolve to it inside the gated domain and abstain outside (default-on
since 2026-07-24, player-05 +0.1006 [+0.0615, +0.1416]). And the table must
stay the untouched derivation: a level correction that helped in-distribution
was refuted on held-out data, so the tests pin the shipped numbers against
re-introduction.
"""

from __future__ import annotations

import pytest

from tabvision.errors import ConfigurationError
from tabvision.fusion.artifact_registry import load_artifact_manifest
from tabvision.fusion.inference_policy import resolve_inference_policy
from tabvision.fusion.string_physics import (
    REGISTERED_TABLE,
    load_string_evidence,
    reference_stiffness_model,
)
from tabvision.types import GuitarConfig, SessionConfig


def _acoustic() -> tuple[GuitarConfig, SessionConfig]:
    return GuitarConfig(), SessionConfig()


def test_artifact_is_registered_and_hash_verified() -> None:
    # load_artifact_manifest recomputes the sha256 and raises on mismatch,
    # so a clean load is the integrity check.
    manifest = load_artifact_manifest(REGISTERED_TABLE, expected_kind="string_evidence")
    assert manifest.registered
    assert manifest.kind == "string_evidence"
    assert manifest.sha256


def test_artifact_table_matches_the_physics_module() -> None:
    loaded = load_string_evidence().model
    computed = reference_stiffness_model()
    assert loaded.log_b0.keys() == computed.log_b0.keys()
    for string, value in computed.log_b0.items():
        assert loaded.log_b0[string] == pytest.approx(value)
    assert loaded.fret_exponent == pytest.approx(computed.fret_exponent)


def test_artifact_is_not_level_corrected() -> None:
    """A +0.60 log-B level correction was tried and reverted on 2026-07-24.

    It gained +0.0160 [+0.0088, +0.0233] on GuitarSet dev and measured -0.0066
    [-0.0224, +0.0079] on sealed player-05 — non-overlapping intervals. The
    level error is physically real (N4 measured it directly on hex pickups)
    but instrument-specific: per-player offsets ran +0.514 to +1.092, wider
    than the correction itself. This asserts the table is the untouched
    derivation, so a future session cannot silently reintroduce a shift that
    only helps in-distribution.
    """
    loaded = load_string_evidence().model
    derived = reference_stiffness_model()
    for string, value in derived.log_b0.items():
        assert loaded.log_b0[string] == pytest.approx(value, abs=1e-12)


def test_artifact_carries_the_gate_passed_decode_config() -> None:
    config = load_string_evidence()
    # These are the values the full-dev and player-05 runs froze.
    assert config.weight == pytest.approx(1.0)
    assert config.min_r2 == pytest.approx(0.50)
    assert config.sigma > 0.0
    # N1, confirmed on sealed player-05 2026-07-24: +0.0226 [+0.0022, +0.0446]
    # over strict, lifting applied notes 834 -> 2227 of 8709.
    assert config.isolation == "partial_aware"


@pytest.mark.parametrize("backend", ["highres", "highres-ensemble"])
def test_auto_resolves_to_the_artifact_in_the_gated_domain(backend: str) -> None:
    """Default-on, 2026-07-24: player-05 +0.1006 [+0.0615, +0.1416].

    ``highres-ensemble`` is parametrized deliberately — it is the clean-acoustic
    ``auto`` backend and the one every Q6 gate was measured on, but the domain
    guard used to require the literal ``"highres"`` and silently excluded it.
    """
    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=cfg,
        session=session,
        audio_backend_name=backend,
    )
    assert policy.resolved_string_evidence == REGISTERED_TABLE
    assert any(a.name == REGISTERED_TABLE for a in policy.artifacts)


def test_auto_matches_the_pipelines_own_default_backend() -> None:
    """Guards the seam that broke: routing and the guard must agree.

    ``audio_backend_for_session`` picks the backend, ``_automatic_timbre_domain``
    accepts it. They were written apart and drifted; this pins them together so
    a future backend rename cannot silently disable the channel.
    """
    from tabvision.pipeline import audio_backend_for_session

    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=cfg,
        session=session,
        audio_backend_name=audio_backend_for_session(session),
    )
    assert policy.resolved_string_evidence == REGISTERED_TABLE


@pytest.mark.parametrize(
    "session,cfg",
    [
        (SessionConfig(instrument="classical"), GuitarConfig()),
        (SessionConfig(instrument="electric"), GuitarConfig()),
        (SessionConfig(), GuitarConfig(capo=3)),
    ],
)
def test_auto_abstains_outside_the_gated_domain(session: SessionConfig, cfg: GuitarConfig) -> None:
    # Abstention keeps the GAPS classical no-regression result true by
    # construction rather than by measurement.
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=cfg,
        session=session,
        audio_backend_name="highres",
    )
    assert policy.resolved_string_evidence == "none"


def test_explicit_none_still_disables_the_channel() -> None:
    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="none",
        cfg=cfg,
        session=session,
        audio_backend_name="highres",
    )
    assert policy.resolved_string_evidence == "none"


@pytest.mark.parametrize("backend", ["highres", "highres-ensemble"])
def test_explicit_selection_resolves_to_the_artifact(backend: str) -> None:
    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence=REGISTERED_TABLE,
        cfg=cfg,
        session=session,
        audio_backend_name=backend,
    )
    assert policy.resolved_string_evidence == REGISTERED_TABLE


@pytest.mark.parametrize(
    "session,cfg",
    [
        (SessionConfig(instrument="classical"), GuitarConfig()),
        (SessionConfig(instrument="electric"), GuitarConfig()),
        (SessionConfig(), GuitarConfig(capo=3)),
    ],
)
def test_explicit_selection_is_refused_outside_the_validated_domain(
    session: SessionConfig, cfg: GuitarConfig
) -> None:
    # A second guard independent of the module's own abstention: the policy
    # layer refuses rather than silently applying a table that does not
    # describe the instrument.
    with pytest.raises(ConfigurationError):
        resolve_inference_policy(
            requested_position_prior="auto",
            requested_sequence_prior="auto",
            requested_string_evidence=REGISTERED_TABLE,
            cfg=cfg,
            session=session,
            audio_backend_name="highres",
        )
