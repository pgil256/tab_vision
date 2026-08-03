"""Session-aware resolution of position, sequence, and timbral evidence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from tabvision.errors import ConfigurationError
from tabvision.fusion.artifact_registry import ArtifactManifest, load_artifact_manifest
from tabvision.types import DEFAULT_TUNING_MIDI, GuitarConfig, SessionConfig

DEFAULT_STRING_EVIDENCE = "acoustic-physics-v1"
"""Artifact ``auto`` resolves to inside the gated clean-acoustic domain."""


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
    requested_assignment_decoder: str
    resolved_assignment_decoder: str
    assignment_decoder_reason: str
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
    requested_assignment_decoder: str | None = None,
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

    # SPEC §1.5 carve-out (2026-08-02): an explicit position prior may be a
    # local personal artifact file rather than a registered name. Detected on
    # the *raw* argument — ``_choice`` lowercases, which would corrupt a path.
    personal_position = _personal_position_prior_path(requested_position_prior)

    position_manifest: ArtifactManifest | None = None
    if personal_position is not None:
        requested_position = str(personal_position)
        reasons.append("explicit personal position-prior artifact (local file)")
    elif requested_position == "auto":
        if _automatic_position_domain(cfg, session):
            try:
                position_manifest = load_artifact_manifest("guitarset-v1", expected_kind="position")
                if cfg.capo > 0:
                    reasons.append(f"position prior re-indexed for capo {cfg.capo}")
            except ConfigurationError as exc:
                reasons.append(f"automatic position prior unavailable: {exc}")
        elif _automatic_classical_domain(cfg, session):
            # 2026-07-20 personal-posture program: classical sessions get the
            # GAPS-trained pair instead of neutral. Unregistered/missing
            # artifacts degrade to none, never to the acoustic prior (the
            # banked -13.8pp cross-domain regression).
            try:
                position_manifest = load_artifact_manifest("gaps-v1", expected_kind="position")
            except ConfigurationError as exc:
                reasons.append(f"automatic classical position prior unavailable: {exc}")
        else:
            reasons.append("session is outside the validated acoustic prior domain")
    elif requested_position != "none":
        position_manifest = load_artifact_manifest(requested_position, expected_kind="position")
    if personal_position is not None:
        resolved_position = str(personal_position)
    else:
        resolved_position = position_manifest.name if position_manifest else "none"

    sequence_manifest: ArtifactManifest | None = None
    if requested_sequence == "auto":
        paired = position_manifest.compatible_sequence_prior if position_manifest else None
        if personal_position is not None:
            # The registered sequence artifacts were gated *coupled to* their
            # named position priors; a personal prior has no validated pairing.
            reasons.append(
                "personal position prior has no validated sequence pairing; sequence prior off"
            )
            paired = None
        if paired and cfg.capo > 0:
            # The registered sequence artifact uses the ``delta_fret`` scheme,
            # which conditions on the *absolute* previous-fret region and is
            # therefore NOT capo-invariant — unlike the position prior, which
            # re-indexes exactly. Q7 measured the pairing under capo rather
            # than assuming: covariant+seq 0.6766 vs covariant 0.6827 at capo
            # 2, i.e. it costs a little and gains nothing. Position prior on,
            # sequence prior off, is the arm that was gated.
            reasons.append(
                "sequence prior is not capo-invariant (delta_fret conditions on "
                "absolute fret region); suppressed under capo"
            )
            paired = None
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

    # The physics channel is on by default in its gated domain (2026-07-24
    # user decision). It is specification-derived rather than fitted, and it
    # abstains per note whenever the partial structure is unreadable, so the
    # failure mode of enabling it is "no evidence", not "wrong evidence".
    # Held-out player-05: +0.1006 [+0.0615, +0.1416] vs no channel.
    resolved_timbre = "none"
    timbre_manifest: ArtifactManifest | None = None
    if requested_timbre not in {"auto", "none"}:
        timbre_manifest = load_artifact_manifest(requested_timbre, expected_kind="string_evidence")
        if not _automatic_timbre_domain(cfg, session, audio_backend_name):
            raise ConfigurationError(
                f"string evidence {requested_timbre!r} requires clean acoustic, "
                f"standard tuning, capo 0, and one of {sorted(TIMBRE_BACKENDS)}"
            )
        resolved_timbre = timbre_manifest.name
    elif requested_timbre == "auto":
        if _automatic_timbre_domain(cfg, session, audio_backend_name):
            try:
                timbre_manifest = load_artifact_manifest(
                    DEFAULT_STRING_EVIDENCE, expected_kind="string_evidence"
                )
                resolved_timbre = timbre_manifest.name
            except ConfigurationError as exc:
                reasons.append(f"automatic string evidence unavailable: {exc}")
        else:
            reasons.append("session is outside the validated string-evidence domain")

    requested_decoder, resolved_decoder, decoder_reason = resolve_assignment_decoder(
        requested_assignment_decoder,
        cfg=cfg,
        session=session,
    )

    manifests = [position_manifest, sequence_manifest, timbre_manifest]
    identities = tuple(
        ArtifactIdentity(item.name, item.version, item.sha256)
        for item in manifests
        if item is not None
    )
    if personal_position is not None:
        identities = (
            ArtifactIdentity(
                name=f"personal:{personal_position.name}",
                version="local",
                sha256=hashlib.sha256(personal_position.read_bytes()).hexdigest(),
            ),
            *identities,
        )
    return ResolvedInferencePolicy(
        requested_position_prior=requested_position,
        resolved_position_prior=resolved_position,
        requested_sequence_prior=requested_sequence,
        resolved_sequence_prior=resolved_sequence,
        requested_string_evidence=requested_timbre,
        resolved_string_evidence=resolved_timbre,
        requested_assignment_decoder=requested_decoder,
        resolved_assignment_decoder=resolved_decoder,
        assignment_decoder_reason=decoder_reason,
        artifacts=identities,
        resolution_reason="; ".join(reasons) or "explicit registered policy",
    )


def _choice(value: str | None, *, default: str) -> str:
    text = (value or default).strip().lower()
    return text or default


def _personal_position_prior_path(requested: str | None) -> Path | None:
    """Return the path when an explicit position prior is a local artifact.

    Personal artifacts (SPEC §1.5 carve-out, 2026-08-02;
    ``scripts/train/build_personal_prior.py``) are addressed by filesystem
    path, distinguished from registered names by the ``.json`` suffix — no
    registered artifact name carries one. The file is validated *here*, at
    policy time, so a bad path fails before minutes of audio inference
    rather than after. Structural validation of the counts stays in
    :func:`tabvision.fusion.position_prior.load_pitch_position_prior`.
    """
    if not requested:
        return None
    text = requested.strip()
    if not text.lower().endswith(".json"):
        return None
    path = Path(text)
    if not path.is_file():
        raise ConfigurationError(f"personal position-prior file not found: {text}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"personal position-prior file is unreadable: {text}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("counts"), list)
    ):
        raise ConfigurationError(f"personal position-prior file has an unsupported schema: {text}")
    return path


def _automatic_acoustic_domain(cfg: GuitarConfig, session: SessionConfig) -> bool:
    """Clean acoustic, standard tuning, **capo 0**.

    The strict domain. Gates the physics channel and the additive decoders,
    both of which genuinely break under a capo: ``B0`` describes the *open*
    string, and a capo moves the speaking length and tension.
    """
    return _automatic_position_domain(cfg, session) and cfg.capo == 0


def _automatic_position_domain(cfg: GuitarConfig, session: SessionConfig) -> bool:
    """Clean acoustic, standard tuning, **any capo**.

    The position prior is the one artifact that survives a capo, because a
    capo shifts the whole fretboard uniformly: a note at absolute fret ``f``
    under capo ``C`` is the shape ``f-C`` without one, so re-indexing both
    axes makes the capo-0 prior apply exactly (``capo_covariant_prior``).

    Routing capo sessions to ``priors=none`` was not merely forgoing a bonus.
    Without a prior the decoder falls back to a low-fret preference, and under
    a capo every candidate sits at fret >= C, so that heuristic mispicks
    systematically: Q7 measured **0.2956** Tab F1 at capo 2 against a 0.6773
    capo-0 control, with the string wrong on two thirds of notes. The
    covariant prior restores **0.6827** (+0.3870 [+0.2818, +0.4906]).
    """
    return (
        session.instrument == "acoustic"
        and session.tone == "clean"
        and cfg.tuning_midi == DEFAULT_TUNING_MIDI
    )


def _automatic_classical_domain(cfg: GuitarConfig, session: SessionConfig) -> bool:
    """Clean classical, standard tuning, capo 0 — the gaps-v1 validated domain."""
    return (
        session.instrument == "classical"
        and session.tone == "clean"
        and cfg.tuning_midi == DEFAULT_TUNING_MIDI
        and cfg.capo == 0
    )


TIMBRE_BACKENDS = frozenset({"highres", "highres-ensemble"})
"""Backends the physics channel is gated on.

``highres-ensemble`` is not optional here: it has been the clean-acoustic
``auto`` audio backend since the 2026-07-20 personal-posture decision, and
**every Q6 gate was measured on it** — the full-dev OOF run and both player-05
confirmations all instantiate ``HighResEnsembleBackend``. This guard formerly
required the literal ``"highres"``, which excluded the exact backend the gate
used: an explicit ``--string-evidence acoustic-physics-v1`` raised
``ConfigurationError`` on the default path, and no test caught it because the
policy tests all passed ``"highres"``. Both members are clean-acoustic
checkpoints and the caller has already been narrowed to the acoustic domain.
"""


def _automatic_timbre_domain(
    cfg: GuitarConfig,
    session: SessionConfig,
    audio_backend_name: str,
) -> bool:
    return _automatic_acoustic_domain(cfg, session) and audio_backend_name in TIMBRE_BACKENDS


def resolve_assignment_decoder(
    requested: str | None,
    *,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> tuple[str, str, str]:
    """Resolve the additive string-assignment decoder policy.

    During Phase 1, ``auto`` deliberately remains the legacy baseline.  An
    explicit ``segment-v1`` request is available only in the validation domain;
    unsupported instruments/configurations fall back safely to ``baseline``.
    """

    value = requested
    if value is None:
        value = os.environ.get("TABVISION_ASSIGNMENT_DECODER", "auto")
    requested_decoder = _choice(value, default="auto")
    choices = {"auto", "baseline", "segment-v1", "context-v1"}
    if requested_decoder not in choices:
        raise ConfigurationError(
            "assignment decoder must be one of auto, baseline, segment-v1, context-v1; "
            f"got {requested_decoder!r}"
        )
    if requested_decoder == "baseline":
        return requested_decoder, "baseline", "explicit rollback decoder"
    if requested_decoder == "auto":
        return (
            requested_decoder,
            "baseline",
            "segment-v1 has not passed the automatic promotion gate",
        )
    if requested_decoder == "context-v1":
        if not _automatic_acoustic_domain(cfg, session):
            return (
                requested_decoder,
                "baseline",
                "context-v1 is restricted to clean acoustic, standard tuning, capo 0",
            )
        try:
            load_artifact_manifest("context-v1", expected_kind="assignment_context")
        except ConfigurationError as exc:
            return (
                requested_decoder,
                "baseline",
                f"context-v1 is unavailable; falling back to baseline: {exc}",
            )
        # A gate-passed artifact must be integrated as candidate evidence
        # before this branch can become reachable. Keeping the final guard
        # fail-safe prevents a manifest edit alone from enabling an incomplete
        # runtime path.
        return (
            requested_decoder,
            "baseline",
            "context-v1 runtime activation requires an integrated gate-passed artifact",
        )
    if not _automatic_acoustic_domain(cfg, session):
        return (
            requested_decoder,
            "baseline",
            "segment-v1 is restricted to clean acoustic, standard tuning, capo 0",
        )
    return requested_decoder, "segment-v1", "explicit validated-domain request"


__all__ = [
    "ArtifactIdentity",
    "ResolvedInferencePolicy",
    "resolve_assignment_decoder",
    "resolve_inference_policy",
]
