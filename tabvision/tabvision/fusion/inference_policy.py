"""Session-aware resolution of position, sequence, and timbral evidence."""

from __future__ import annotations

from dataclasses import dataclass

from tabvision.errors import ConfigurationError
from tabvision.fusion.artifact_registry import ArtifactManifest, load_artifact_manifest
from tabvision.types import DEFAULT_TUNING_MIDI, GuitarConfig, SessionConfig


@dataclass(frozen=True)
class ArtifactIdentity:
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class ResolvedInferencePolicy:
    requested_position_prior: str
    resolved_position_prior: str
    requested_sequence_prior: str
    resolved_sequence_prior: str
    requested_string_evidence: str
    resolved_string_evidence: str
    artifacts: tuple[ArtifactIdentity, ...]
    resolution_reason: str


def resolve_inference_policy(
    *,
    requested_position_prior: str | None,
    requested_sequence_prior: str | None,
    requested_string_evidence: str | None,
    cfg: GuitarConfig,
    session: SessionConfig,
    audio_backend_name: str,
) -> ResolvedInferencePolicy:
    """Resolve requested policy without loading model weights.

    Automatic corpus priors are restricted to the validated clean acoustic,
    standard-tuning, capo-zero domain. Explicit registered corpus artifacts are
    useful for reproducible evaluation and rollback, so they bypass automatic
    domain routing; their sequence artifact must still be the registered pair.
    """

    # ``None`` was the pre-policy API's explicit off switch. Public entrypoints
    # now default to the string ``"auto"``, so keep None backward-compatible.
    requested_position = _choice(requested_position_prior, default="none")
    requested_sequence = _choice(requested_sequence_prior, default="none")
    requested_timbre = _choice(requested_string_evidence, default="none")
    reasons: list[str] = []

    position_manifest: ArtifactManifest | None = None
    if requested_position == "auto":
        if _automatic_acoustic_domain(cfg, session):
            try:
                position_manifest = load_artifact_manifest("guitarset-v1", expected_kind="position")
            except ConfigurationError as exc:
                reasons.append(f"automatic position prior unavailable: {exc}")
        else:
            reasons.append("session is outside the validated acoustic prior domain")
    elif requested_position != "none":
        position_manifest = load_artifact_manifest(requested_position, expected_kind="position")
    resolved_position = position_manifest.name if position_manifest else "none"

    sequence_manifest: ArtifactManifest | None = None
    if requested_sequence == "auto":
        paired = position_manifest.compatible_sequence_prior if position_manifest else None
        if paired:
            try:
                sequence_manifest = load_artifact_manifest(paired, expected_kind="sequence")
            except ConfigurationError as exc:
                reasons.append(f"automatic sequence prior unavailable: {exc}")
    elif requested_sequence != "none":
        sequence_manifest = load_artifact_manifest(requested_sequence, expected_kind="sequence")
        expected_position = sequence_manifest.compatible_position_prior
        if expected_position != resolved_position:
            raise ConfigurationError(
                f"sequence prior {sequence_manifest.name!r} requires position prior "
                f"{expected_position!r}, resolved {resolved_position!r}"
            )
    resolved_sequence = sequence_manifest.name if sequence_manifest else "none"

    # Phase 2 registers the timbral model here. Until then auto is a neutral
    # fallback, while an explicit unavailable request is a clear error.
    resolved_timbre = "none"
    timbre_manifest: ArtifactManifest | None = None
    if requested_timbre not in {"auto", "none"}:
        timbre_manifest = load_artifact_manifest(requested_timbre, expected_kind="string_evidence")
        if not _automatic_timbre_domain(cfg, session, audio_backend_name):
            raise ConfigurationError(
                f"string evidence {requested_timbre!r} requires clean acoustic, "
                "standard tuning, capo 0, and the highres backend"
            )
        resolved_timbre = timbre_manifest.name
    elif requested_timbre == "auto":
        reasons.append("no gate-passed timbral artifact is registered")

    manifests = [position_manifest, sequence_manifest, timbre_manifest]
    identities = tuple(
        ArtifactIdentity(item.name, item.version, item.sha256)
        for item in manifests
        if item is not None
    )
    return ResolvedInferencePolicy(
        requested_position_prior=requested_position,
        resolved_position_prior=resolved_position,
        requested_sequence_prior=requested_sequence,
        resolved_sequence_prior=resolved_sequence,
        requested_string_evidence=requested_timbre,
        resolved_string_evidence=resolved_timbre,
        artifacts=identities,
        resolution_reason="; ".join(reasons) or "explicit registered policy",
    )


def _choice(value: str | None, *, default: str) -> str:
    text = (value or default).strip().lower()
    return text or default


def _automatic_acoustic_domain(cfg: GuitarConfig, session: SessionConfig) -> bool:
    return (
        session.instrument == "acoustic"
        and session.tone == "clean"
        and cfg.tuning_midi == DEFAULT_TUNING_MIDI
        and cfg.capo == 0
    )


def _automatic_timbre_domain(
    cfg: GuitarConfig,
    session: SessionConfig,
    audio_backend_name: str,
) -> bool:
    return _automatic_acoustic_domain(cfg, session) and audio_backend_name == "highres"


__all__ = [
    "ArtifactIdentity",
    "ResolvedInferencePolicy",
    "resolve_inference_policy",
]
