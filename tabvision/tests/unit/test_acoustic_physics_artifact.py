"""Tests for the registered acoustic-physics-v1 string-evidence artifact.

Two things matter here. The artifact must carry the *gate-passed* decode
config rather than inheriting module defaults — otherwise editing a default
would silently change what a registered, hash-verified artifact does. And
registering it must not alter `auto`: promotion into the default path is a
separate decision, and this asserts the code has not pre-empted it.
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


def test_auto_still_resolves_to_none() -> None:
    # Registering the artifact must NOT promote it into the default path.
    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        cfg=cfg,
        session=session,
        audio_backend_name="highres",
    )
    assert policy.resolved_string_evidence == "none"


def test_explicit_selection_resolves_to_the_artifact() -> None:
    cfg, session = _acoustic()
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence=REGISTERED_TABLE,
        cfg=cfg,
        session=session,
        audio_backend_name="highres",
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
