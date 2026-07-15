from __future__ import annotations

import pytest

from tabvision.errors import ConfigurationError
from tabvision.fusion.inference_policy import resolve_inference_policy
from tabvision.types import GuitarConfig, SessionConfig


def _resolve(
    *,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    position: str = "auto",
    sequence: str = "auto",
    timbre: str = "auto",
    decoder: str | None = "auto",
):
    return resolve_inference_policy(
        requested_position_prior=position,
        requested_sequence_prior=sequence,
        requested_string_evidence=timbre,
        requested_assignment_decoder=decoder,
        cfg=cfg or GuitarConfig(),
        session=session or SessionConfig(),
        audio_backend_name="highres",
    )


@pytest.mark.parametrize("style", ["fingerstyle", "strumming", "mixed"])
def test_auto_uses_global_pair_for_supported_acoustic_styles(style: str) -> None:
    policy = _resolve(session=SessionConfig(style=style))
    assert policy.resolved_position_prior == "guitarset-v1"
    assert policy.resolved_sequence_prior == "guitarset-seq-v1"
    assert policy.resolved_string_evidence == "none"
    assert policy.resolved_assignment_decoder == "baseline"
    assert {item.name for item in policy.artifacts} == {
        "guitarset-v1",
        "guitarset-seq-v1",
    }


@pytest.mark.parametrize(
    ("cfg", "session"),
    [
        (GuitarConfig(capo=2), SessionConfig()),
        (GuitarConfig(tuning_midi=(38, 45, 50, 55, 59, 64)), SessionConfig()),
        (GuitarConfig(), SessionConfig(instrument="classical")),
        (GuitarConfig(), SessionConfig(instrument="electric")),
        (GuitarConfig(), SessionConfig(tone="distorted")),
    ],
)
def test_auto_is_neutral_outside_validated_domain(
    cfg: GuitarConfig, session: SessionConfig
) -> None:
    policy = _resolve(cfg=cfg, session=session)
    assert policy.resolved_position_prior == "none"
    assert policy.resolved_sequence_prior == "none"
    assert policy.resolved_assignment_decoder == "baseline"


def test_explicit_registered_pair_is_reproducible_outside_auto_domain() -> None:
    policy = _resolve(
        session=SessionConfig(instrument="electric"),
        position="guitarset-v1",
        sequence="guitarset-seq-v1",
    )
    assert policy.resolved_position_prior == "guitarset-v1"
    assert policy.resolved_sequence_prior == "guitarset-seq-v1"


def test_explicit_sequence_requires_its_paired_position_prior() -> None:
    with pytest.raises(ConfigurationError, match="requires position prior"):
        _resolve(position="none", sequence="guitarset-seq-v1")


def test_legacy_none_values_remain_explicitly_disabled() -> None:
    policy = resolve_inference_policy(
        requested_position_prior=None,
        requested_sequence_prior=None,
        requested_string_evidence=None,
        cfg=GuitarConfig(),
        session=SessionConfig(),
        audio_backend_name="highres",
    )
    assert policy.resolved_position_prior == "none"
    assert policy.resolved_sequence_prior == "none"
    assert policy.resolved_string_evidence == "none"


def test_explicit_unregistered_timbre_artifact_fails_clearly() -> None:
    with pytest.raises(ConfigurationError, match="unknown learned artifact"):
        _resolve(timbre="guitarset-timbre-v1")


@pytest.mark.parametrize(
    ("cfg", "session"),
    [
        (GuitarConfig(capo=2), SessionConfig()),
        (GuitarConfig(tuning_midi=(38, 45, 50, 55, 59, 64)), SessionConfig()),
        (GuitarConfig(), SessionConfig(instrument="classical")),
        (GuitarConfig(), SessionConfig(instrument="electric")),
        (GuitarConfig(), SessionConfig(tone="distorted")),
    ],
)
def test_explicit_segment_decoder_is_available_only_in_validated_domain(
    cfg: GuitarConfig,
    session: SessionConfig,
) -> None:
    acoustic = _resolve(decoder="segment-v1")
    unsupported = _resolve(decoder="segment-v1", cfg=cfg, session=session)

    assert acoustic.resolved_assignment_decoder == "segment-v1"
    assert unsupported.resolved_assignment_decoder == "baseline"
    assert "restricted" in unsupported.assignment_decoder_reason


def test_unregistered_context_decoder_falls_back_to_baseline() -> None:
    policy = _resolve(decoder="context-v1")

    assert policy.requested_assignment_decoder == "context-v1"
    assert policy.resolved_assignment_decoder == "baseline"
    assert "not registered" in policy.assignment_decoder_reason


def test_context_decoder_is_never_activated_outside_its_domain() -> None:
    policy = _resolve(
        decoder="context-v1",
        session=SessionConfig(instrument="electric"),
    )

    assert policy.resolved_assignment_decoder == "baseline"
    assert "restricted" in policy.assignment_decoder_reason


def test_assignment_decoder_environment_is_honored_when_request_is_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABVISION_ASSIGNMENT_DECODER", "segment-v1")
    policy = _resolve(decoder=None)
    assert policy.requested_assignment_decoder == "segment-v1"
    assert policy.resolved_assignment_decoder == "segment-v1"


def test_auto_falls_back_to_neutral_when_registered_artifact_is_invalid(monkeypatch) -> None:
    def fail(name: str, **kwargs):
        raise ConfigurationError("artifact hash mismatch")

    monkeypatch.setattr("tabvision.fusion.inference_policy.load_artifact_manifest", fail)
    policy = _resolve()
    assert policy.resolved_position_prior == "none"
    assert policy.resolved_sequence_prior == "none"
    assert "hash mismatch" in policy.resolution_reason
