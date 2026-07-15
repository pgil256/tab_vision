"""Hash-verified registry for learned string-assignment artifacts.

Only artifacts whose development gate passed are runtime-loadable. Candidate
artifacts may live beside them with ``registered=false`` manifests so failed
experiments remain reproducible without becoming selectable production policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast

from tabvision.errors import ConfigurationError

ArtifactKind = Literal["position", "sequence", "string_evidence", "assignment_context"]

_ARTIFACT_DIR = Path(__file__).with_name("priors")
_MANIFEST_FILES: dict[str, str] = {
    "guitarset-v1": "guitarset_v1.manifest.json",
    "guitarset-seq-v1": "guitarset_seq_v1.manifest.json",
    "guitarset-solo-v1": "guitarset_solo_v1.manifest.json",
    "guitarset-solo-seq-v1": "guitarset_solo_seq_v1.manifest.json",
    "guitarset-comp-v1": "guitarset_comp_v1.manifest.json",
    "guitarset-comp-seq-v1": "guitarset_comp_seq_v1.manifest.json",
    "context-v1": "context_v1.manifest.json",
}


@dataclass(frozen=True)
class ArtifactManifest:
    name: str
    kind: ArtifactKind
    version: str
    artifact_path: Path
    sha256: str
    registered: bool
    compatible_position_prior: str | None
    compatible_sequence_prior: str | None
    mode: str
    payload: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def load_artifact_manifest(
    name: str,
    *,
    expected_kind: ArtifactKind | None = None,
    require_registered: bool = True,
) -> ArtifactManifest:
    """Load and verify one manifest and its artifact bytes."""

    manifest_name = _MANIFEST_FILES.get(name)
    if manifest_name is None:
        known = ", ".join(sorted(_MANIFEST_FILES))
        raise ConfigurationError(f"unknown learned artifact {name!r}; known: {known}")
    manifest_path = _ARTIFACT_DIR / manifest_name
    if not manifest_path.is_file():
        raise ConfigurationError(f"artifact manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"artifact manifest is invalid: {manifest_path}") from exc
    if payload.get("schema_version") != 1:
        raise ConfigurationError(f"unsupported artifact manifest schema: {manifest_path}")

    kind = str(payload.get("artifact_kind", ""))
    if kind not in {"position", "sequence", "string_evidence", "assignment_context"}:
        raise ConfigurationError(f"invalid artifact kind in {manifest_path}: {kind!r}")
    if expected_kind is not None and kind != expected_kind:
        raise ConfigurationError(f"artifact {name!r} is {kind}, expected {expected_kind}")
    registered = bool(payload.get("registered", False))
    if require_registered and not registered:
        reason = str(payload.get("gate", {}).get("decision", "development gate failed"))
        raise ConfigurationError(f"artifact {name!r} is not registered: {reason}")

    artifact_file = str(payload.get("artifact_file", ""))
    artifact_path = _ARTIFACT_DIR / artifact_file
    if not artifact_file or not artifact_path.is_file():
        raise ConfigurationError(f"artifact file is missing for {name!r}: {artifact_path}")
    expected_sha = str(payload.get("artifact_sha256", "")).lower()
    actual_sha = _sha256(artifact_path)
    if not expected_sha or actual_sha != expected_sha:
        raise ConfigurationError(
            f"artifact hash mismatch for {name!r}: expected {expected_sha}, got {actual_sha}"
        )

    return ArtifactManifest(
        name=name,
        kind=cast(ArtifactKind, kind),
        version=str(payload.get("artifact_version", name)),
        artifact_path=artifact_path,
        sha256=actual_sha,
        registered=registered,
        compatible_position_prior=_optional_string(payload.get("compatible_position_prior")),
        compatible_sequence_prior=_optional_string(payload.get("compatible_sequence_prior")),
        mode=str(payload.get("mode", "all")),
        payload=payload,
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def registered_artifact_names(kind: ArtifactKind) -> tuple[str, ...]:
    out = []
    for name in sorted(_MANIFEST_FILES):
        try:
            manifest = load_artifact_manifest(name, require_registered=False)
        except ConfigurationError:
            continue
        if manifest.kind == kind and manifest.registered:
            out.append(name)
    return tuple(out)


__all__ = [
    "ArtifactManifest",
    "load_artifact_manifest",
    "registered_artifact_names",
]
