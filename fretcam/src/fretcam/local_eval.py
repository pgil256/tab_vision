"""Manage an opt-in, local-only FretCam finger-position evaluation set.

This module deliberately has no camera or frame-extraction code.  It registers
public-licensed or reproducibly synthetic media already placed under a
machine-local dataset root and stores timestamped annotations beside it.
Private and user recordings are rejected in every role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
SCHEMA_ID = "fretcam-local-eval-manifest-v1"
MANIFEST_NAME = "manifest.json"

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
POSITIONS = frozenset(range(1, 13))
TECHNIQUES = frozenset({"note", "chord", "barre", "stretch", "slide"})
PHASES = frozenset({"stable", "shifting", "occluded", "invalid"})
FRAMINGS = frozenset({"close", "full_neck"})
HANDEDNESS = frozenset({"left", "right"})
LIGHTING = frozenset({"bright", "dim", "warm", "cool", "uneven", "mixed"})
SKIN_TONES = frozenset(
    {"not_recorded", "very_light", "light", "medium", "deep", "very_deep"}
)
APPEARANCE_BASES = frozenset(
    {
        "not_recorded",
        "source_declared",
        "licensed_dataset_label",
        "synthetic_specification",
    }
)
GUITARS = frozenset(
    {
        "acoustic_steel",
        "acoustic_classical",
        "electric_solid",
        "electric_hollow",
        "other",
    }
)
SLEEVES = frozenset({"bare_arm", "short_sleeve", "long_sleeve", "glove", "other"})
BACKGROUNDS = frozenset({"plain", "cluttered", "outdoor", "studio", "other"})
FACE_VISIBILITY = frozenset({"none", "partial", "full"})
PROVENANCE_KINDS = frozenset({"public_licensed", "synthetic"})
FINGER_VISIBILITY = frozenset(
    {"visible", "partially_visible", "occluded", "out_of_frame"}
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_MEDIA_STEM = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_LICENSE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,63}$")
_MEDIA_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})

_POLICY = {
    "purpose": "fretcam_local_development_evaluation",
    "storage_scope": "local_only",
    "capture_mode": "manual_opt_in_only",
    "automatic_recording": False,
    "automatic_frame_extraction": False,
    "source_scope": "public_or_synthetic_only",
    "private_or_user_media_allowed": False,
    "model_training_allowed": False,
    "release_gate_allowed": False,
    "media_retention": "user_managed",
}


class LocalEvalError(ValueError):
    """Raised when a local evaluation dataset violates its safety contract."""


def default_dataset_root() -> Path:
    """Return the machine-local default; this path is never in the repository."""
    return Path.home() / ".tabvision" / "cache" / "fretcam_local_eval"


def schema_path() -> Path:
    """Return the packaged JSON Schema for editor and external-tool use."""
    return Path(__file__).with_name("schemas") / "local_eval_manifest_v1.schema.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise LocalEvalError(f"{context} missing keys: {', '.join(sorted(missing))}")
    if extra:
        raise LocalEvalError(f"{context} has unknown keys: {', '.join(sorted(extra))}")


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LocalEvalError(f"{context} must be an object")
    return value


def _require_list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise LocalEvalError(f"{context} must be an array")
    return value


def _require_enum(value: Any, choices: frozenset[Any], *, context: str) -> None:
    if value not in choices:
        rendered = ", ".join(str(choice) for choice in sorted(choices))
        raise LocalEvalError(f"{context} must be one of: {rendered}")


def _require_bool(value: Any, *, context: str) -> None:
    if not isinstance(value, bool):
        raise LocalEvalError(f"{context} must be true or false")


def _require_nullable_bool(value: Any, *, context: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise LocalEvalError(f"{context} must be true, false, or null")


def _validate_slug(value: Any, *, context: str) -> None:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise LocalEvalError(
            f"{context} must be 2-64 lowercase letters, digits, '_' or '-'"
        )


def _validate_timestamp(value: Any, *, context: str) -> None:
    if not isinstance(value, str):
        raise LocalEvalError(f"{context} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LocalEvalError(f"{context} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise LocalEvalError(f"{context} must include a timezone")


def _git_ancestor(path: Path) -> Path | None:
    resolved = path.resolve(strict=False)
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def validate_dataset_root(root: Path) -> Path:
    """Resolve a dataset root and reject every location inside a Git worktree."""
    resolved = root.expanduser().resolve(strict=False)
    git_root = _git_ancestor(resolved)
    if git_root is not None:
        raise LocalEvalError(
            f"dataset root must not be inside a Git repository: {git_root}"
        )
    return resolved


def _manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    target = _manifest_path(root)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def init_dataset(root: Path, *, dataset_id: str) -> dict[str, Any]:
    """Create an empty manifest without opening a camera or creating frames."""
    resolved = validate_dataset_root(root)
    _validate_slug(dataset_id, context="dataset_id")
    target = _manifest_path(resolved)
    if target.exists():
        raise LocalEvalError(f"dataset already initialized: {target}")
    if resolved.exists() and any(resolved.iterdir()):
        raise LocalEvalError(
            "dataset root is not empty; choose an empty local directory"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "media").mkdir()
    created_at = _now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_id": SCHEMA_ID,
        "dataset_id": dataset_id,
        "created_at": created_at,
        "updated_at": created_at,
        "policy": dict(_POLICY),
        "clips": [],
    }
    _write_manifest(resolved, manifest)
    return manifest


def _safe_media_path(root: Path, value: Any, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise LocalEvalError("media.path must be a relative POSIX path")
    pure_path = PurePosixPath(value)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise LocalEvalError("media.path must stay inside the dataset root")
    if len(pure_path.parts) != 2 or pure_path.parts[0] != "media":
        raise LocalEvalError("media.path must have the form media/<safe-filename>")
    filename = pure_path.name
    suffix = Path(filename).suffix.lower()
    if (
        suffix not in _MEDIA_SUFFIXES
        or _MEDIA_STEM.fullmatch(Path(filename).stem) is None
    ):
        raise LocalEvalError(
            "media filename must use a safe lowercase stem and a supported video suffix"
        )
    path = (root / Path(*pure_path.parts)).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LocalEvalError("media.path escapes the dataset root") from exc
    if must_exist and (not path.is_file() or path.is_symlink()):
        raise LocalEvalError(f"media file is missing or is a symlink: {value}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_policy(value: Any) -> None:
    policy = _require_mapping(value, context="policy")
    _exact_keys(policy, set(_POLICY), context="policy")
    for key, expected in _POLICY.items():
        if policy[key] != expected:
            raise LocalEvalError(f"policy.{key} is immutable and must be {expected!r}")


def _validate_provenance(value: Any, *, capture: Mapping[str, Any]) -> None:
    provenance = _require_mapping(value, context="provenance")
    _exact_keys(
        provenance,
        {
            "kind",
            "source_uri",
            "license",
            "rights_verified",
            "appearance_metadata_rights_verified",
            "redistribution_allowed",
        },
        context="provenance",
    )
    kind = provenance["kind"]
    _require_enum(kind, PROVENANCE_KINDS, context="provenance.kind")
    source_uri = provenance["source_uri"]
    if not isinstance(source_uri, str) or not source_uri.startswith(
        ("https://", "http://")
    ):
        raise LocalEvalError("provenance.source_uri must be an HTTP(S) URL")
    license_name = provenance["license"]
    if not isinstance(license_name, str) or _LICENSE.fullmatch(license_name) is None:
        raise LocalEvalError("provenance.license must be a short SPDX-style identifier")
    _require_bool(provenance["rights_verified"], context="provenance.rights_verified")
    _require_bool(
        provenance["appearance_metadata_rights_verified"],
        context="provenance.appearance_metadata_rights_verified",
    )
    _require_bool(
        provenance["redistribution_allowed"],
        context="provenance.redistribution_allowed",
    )

    if not provenance["rights_verified"]:
        raise LocalEvalError(
            f"{kind} clips require a source URL, license, and verified use rights"
        )
    if license_name in {"private-local-only", "unknown", "unlicensed"}:
        raise LocalEvalError(
            f"{kind} clips require their actual public or synthetic license"
        )

    appearance_recorded = capture["skin_tone"] != "not_recorded"
    appearance_basis = capture["appearance_basis"]
    if appearance_recorded:
        if appearance_basis == "not_recorded":
            raise LocalEvalError(
                "recorded skin-tone metadata requires an appearance basis"
            )
        if not provenance["appearance_metadata_rights_verified"]:
            raise LocalEvalError(
                "appearance metadata requires verified labeling rights"
            )
        if kind == "public_licensed" and appearance_basis not in {
            "source_declared",
            "licensed_dataset_label",
        }:
            raise LocalEvalError(
                "public appearance metadata must be source-declared or a "
                "licensed dataset label"
            )
        if kind == "synthetic" and appearance_basis != "synthetic_specification":
            raise LocalEvalError(
                "synthetic appearance metadata must come from its specification"
            )
    elif appearance_basis != "not_recorded":
        raise LocalEvalError(
            "appearance_basis must be not_recorded when skin_tone is not_recorded"
        )
    elif provenance["appearance_metadata_rights_verified"]:
        raise LocalEvalError(
            "appearance metadata rights must be false when no appearance label is stored"
        )


def _validate_capture(value: Any) -> Mapping[str, Any]:
    capture = _require_mapping(value, context="capture")
    _exact_keys(
        capture,
        {
            "handedness",
            "framing",
            "lighting",
            "skin_tone",
            "appearance_basis",
            "guitar",
            "sleeve",
            "background",
            "face_visibility",
            "minors_present",
        },
        context="capture",
    )
    _require_enum(capture["handedness"], HANDEDNESS, context="capture.handedness")
    _require_enum(capture["framing"], FRAMINGS, context="capture.framing")
    _require_enum(capture["lighting"], LIGHTING, context="capture.lighting")
    _require_enum(capture["skin_tone"], SKIN_TONES, context="capture.skin_tone")
    _require_enum(
        capture["appearance_basis"],
        APPEARANCE_BASES,
        context="capture.appearance_basis",
    )
    _require_enum(capture["guitar"], GUITARS, context="capture.guitar")
    _require_enum(capture["sleeve"], SLEEVES, context="capture.sleeve")
    _require_enum(capture["background"], BACKGROUNDS, context="capture.background")
    _require_enum(
        capture["face_visibility"],
        FACE_VISIBILITY,
        context="capture.face_visibility",
    )
    _require_bool(capture["minors_present"], context="capture.minors_present")
    if capture["minors_present"]:
        raise LocalEvalError("clips containing minors are not accepted")
    return capture


def _validate_finger(value: Any, *, context: str) -> None:
    finger = _require_mapping(value, context=context)
    _exact_keys(finger, {"visibility", "pressing", "contacts"}, context=context)
    visibility = finger["visibility"]
    _require_enum(visibility, FINGER_VISIBILITY, context=f"{context}.visibility")
    pressing = finger["pressing"]
    _require_nullable_bool(pressing, context=f"{context}.pressing")
    contacts = _require_list(finger["contacts"], context=f"{context}.contacts")

    if visibility in {"occluded", "out_of_frame"} and pressing is not None:
        raise LocalEvalError(
            f"{context}.pressing must be null when the finger is not observable"
        )
    if visibility == "visible" and pressing is None:
        raise LocalEvalError(
            f"{context}.pressing must be labeled when the finger is visible"
        )
    if pressing is True and not contacts:
        raise LocalEvalError(f"{context} pressing=true requires at least one contact")
    if pressing is not True and contacts:
        raise LocalEvalError(f"{context} contacts require pressing=true")

    seen_strings: set[int] = set()
    for index, raw_contact in enumerate(contacts):
        contact_context = f"{context}.contacts[{index}]"
        contact = _require_mapping(raw_contact, context=contact_context)
        _exact_keys(contact, {"string", "fret"}, context=contact_context)
        string = contact["string"]
        fret = contact["fret"]
        if (
            isinstance(string, bool)
            or not isinstance(string, int)
            or not 1 <= string <= 6
        ):
            raise LocalEvalError(f"{contact_context}.string must be 1-6")
        if isinstance(fret, bool) or not isinstance(fret, int) or not 0 <= fret <= 24:
            raise LocalEvalError(f"{contact_context}.fret must be 0-24")
        if string in seen_strings:
            raise LocalEvalError(f"{context} repeats string {string}")
        seen_strings.add(string)


def _validate_annotation(value: Any, *, duration_ms: int, context: str) -> None:
    annotation = _require_mapping(value, context=context)
    _exact_keys(
        annotation,
        {
            "annotation_id",
            "timestamp_ms",
            "phase",
            "position",
            "technique",
            "fingers",
        },
        context=context,
    )
    _validate_slug(annotation["annotation_id"], context=f"{context}.annotation_id")
    timestamp_ms = annotation["timestamp_ms"]
    if (
        isinstance(timestamp_ms, bool)
        or not isinstance(timestamp_ms, int)
        or not 0 <= timestamp_ms <= duration_ms
    ):
        raise LocalEvalError(f"{context}.timestamp_ms must be within the clip duration")
    phase = annotation["phase"]
    _require_enum(phase, PHASES, context=f"{context}.phase")
    position = annotation["position"]
    if phase == "stable":
        if position not in POSITIONS:
            raise LocalEvalError(
                f"{context}.position must be 1-12 for a stable annotation"
            )
    elif phase in {"shifting", "invalid"} and position is not None:
        raise LocalEvalError(f"{context}.position must be null while {phase}")
    elif position is not None and position not in POSITIONS:
        raise LocalEvalError(f"{context}.position must be 1-12 or null")
    technique = annotation["technique"]
    _require_enum(technique, TECHNIQUES, context=f"{context}.technique")
    fingers = _require_mapping(annotation["fingers"], context=f"{context}.fingers")
    _exact_keys(fingers, set(FINGERS), context=f"{context}.fingers")
    for finger_name in FINGERS:
        _validate_finger(
            fingers[finger_name],
            context=f"{context}.fingers.{finger_name}",
        )
    if technique == "barre":
        widest_contact = max(len(fingers[finger]["contacts"]) for finger in FINGERS)
        if widest_contact < 2:
            raise LocalEvalError(
                f"{context} barre annotations require one multi-string contact"
            )


def _validate_clip(
    value: Any,
    *,
    root: Path,
    verify_media: bool,
    context: str,
) -> None:
    clip = _require_mapping(value, context=context)
    _exact_keys(
        clip,
        {"clip_id", "media", "provenance", "capture", "annotations"},
        context=context,
    )
    _validate_slug(clip["clip_id"], context=f"{context}.clip_id")
    media = _require_mapping(clip["media"], context=f"{context}.media")
    _exact_keys(
        media,
        {"path", "sha256", "byte_size", "duration_ms"},
        context=f"{context}.media",
    )
    path = _safe_media_path(root, media["path"], must_exist=verify_media)
    sha256 = media["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise LocalEvalError(f"{context}.media.sha256 must be lowercase SHA-256")
    byte_size = media["byte_size"]
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size <= 0:
        raise LocalEvalError(f"{context}.media.byte_size must be positive")
    duration_ms = media["duration_ms"]
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms <= 0
    ):
        raise LocalEvalError(f"{context}.media.duration_ms must be positive")
    if verify_media:
        if path.stat().st_size != byte_size:
            raise LocalEvalError(f"{context}.media.byte_size does not match the file")
        if _sha256(path) != sha256:
            raise LocalEvalError(f"{context}.media.sha256 does not match the file")

    capture = _validate_capture(clip["capture"])
    _validate_provenance(clip["provenance"], capture=capture)
    annotations = _require_list(clip["annotations"], context=f"{context}.annotations")
    annotation_ids: set[str] = set()
    timestamps: set[int] = set()
    for index, annotation in enumerate(annotations):
        annotation_context = f"{context}.annotations[{index}]"
        _validate_annotation(
            annotation,
            duration_ms=duration_ms,
            context=annotation_context,
        )
        annotation_id = annotation["annotation_id"]
        timestamp_ms = annotation["timestamp_ms"]
        if annotation_id in annotation_ids:
            raise LocalEvalError(f"{context} repeats annotation_id {annotation_id}")
        if timestamp_ms in timestamps:
            raise LocalEvalError(f"{context} repeats timestamp_ms {timestamp_ms}")
        annotation_ids.add(annotation_id)
        timestamps.add(timestamp_ms)


def validate_manifest(
    manifest: Any,
    *,
    root: Path,
    verify_media: bool = True,
) -> None:
    """Validate schema, public/synthetic provenance, policy, and media integrity."""
    resolved = validate_dataset_root(root)
    document = _require_mapping(manifest, context="manifest")
    _exact_keys(
        document,
        {
            "schema_version",
            "schema_id",
            "dataset_id",
            "created_at",
            "updated_at",
            "policy",
            "clips",
        },
        context="manifest",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        raise LocalEvalError(f"schema_version must be {SCHEMA_VERSION}")
    if document["schema_id"] != SCHEMA_ID:
        raise LocalEvalError(f"schema_id must be {SCHEMA_ID}")
    _validate_slug(document["dataset_id"], context="dataset_id")
    _validate_timestamp(document["created_at"], context="created_at")
    _validate_timestamp(document["updated_at"], context="updated_at")
    _validate_policy(document["policy"])
    clips = _require_list(document["clips"], context="clips")
    clip_ids: set[str] = set()
    media_paths: set[str] = set()
    hashes: set[str] = set()
    for index, clip in enumerate(clips):
        context = f"clips[{index}]"
        _validate_clip(
            clip,
            root=resolved,
            verify_media=verify_media,
            context=context,
        )
        clip_id = clip["clip_id"]
        media_path = clip["media"]["path"]
        digest = clip["media"]["sha256"]
        if clip_id in clip_ids:
            raise LocalEvalError(f"duplicate clip_id: {clip_id}")
        if media_path in media_paths:
            raise LocalEvalError(f"duplicate media path: {media_path}")
        if digest in hashes:
            raise LocalEvalError(
                f"duplicate media content registered by clip {clip_id}"
            )
        clip_ids.add(clip_id)
        media_paths.add(media_path)
        hashes.add(digest)


def load_dataset(root: Path, *, verify_media: bool = True) -> dict[str, Any]:
    """Load and structurally validate a local manifest."""
    resolved = validate_dataset_root(root)
    path = _manifest_path(resolved)
    if not path.is_file():
        raise LocalEvalError(f"dataset is not initialized: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalEvalError(f"could not read manifest: {path}") from exc
    if not isinstance(value, dict):
        raise LocalEvalError("manifest must be a JSON object")
    validate_manifest(value, root=resolved, verify_media=verify_media)
    return value


def add_clip(
    root: Path,
    *,
    clip_id: str,
    media_path: str,
    duration_ms: int,
    provenance_kind: str,
    source_uri: str,
    license_name: str,
    rights_verified: bool,
    appearance_metadata_rights_verified: bool,
    redistribution_allowed: bool,
    handedness: str,
    framing: str,
    lighting: str,
    skin_tone: str,
    appearance_basis: str,
    guitar: str,
    sleeve: str,
    background: str,
    face_visibility: str,
    minors_present: bool,
) -> dict[str, Any]:
    """Register an existing local video; this function never copies or records it."""
    resolved = validate_dataset_root(root)
    manifest = load_dataset(resolved)
    path = _safe_media_path(resolved, media_path, must_exist=True)
    relative_path = path.relative_to(resolved).as_posix()
    clip = {
        "clip_id": clip_id,
        "media": {
            "path": relative_path,
            "sha256": _sha256(path),
            "byte_size": path.stat().st_size,
            "duration_ms": duration_ms,
        },
        "provenance": {
            "kind": provenance_kind,
            "source_uri": source_uri,
            "license": license_name,
            "rights_verified": rights_verified,
            "appearance_metadata_rights_verified": (
                appearance_metadata_rights_verified
            ),
            "redistribution_allowed": redistribution_allowed,
        },
        "capture": {
            "handedness": handedness,
            "framing": framing,
            "lighting": lighting,
            "skin_tone": skin_tone,
            "appearance_basis": appearance_basis,
            "guitar": guitar,
            "sleeve": sleeve,
            "background": background,
            "face_visibility": face_visibility,
            "minors_present": minors_present,
        },
        "annotations": [],
    }
    manifest["clips"].append(clip)
    validate_manifest(manifest, root=resolved)
    _write_manifest(resolved, manifest)
    return clip


def parse_finger_spec(value: str) -> dict[str, Any]:
    """Parse ``visibility:pressing-state:string@fret,...`` from the CLI."""
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise LocalEvalError(
            "finger labels use visibility:pressing|hovering|unknown:string@fret,..."
        )
    visibility, state, contact_text = parts
    _require_enum(visibility, FINGER_VISIBILITY, context="finger visibility")
    pressing_values = {"pressing": True, "hovering": False, "unknown": None}
    if state not in pressing_values:
        raise LocalEvalError("finger state must be pressing, hovering, or unknown")
    contacts: list[dict[str, int]] = []
    if contact_text not in {"", "-"}:
        for item in contact_text.split(","):
            try:
                string_text, fret_text = item.split("@", 1)
                contacts.append({"string": int(string_text), "fret": int(fret_text)})
            except (TypeError, ValueError) as exc:
                raise LocalEvalError(
                    "contacts use string@fret separated by commas"
                ) from exc
    result = {
        "visibility": visibility,
        "pressing": pressing_values[state],
        "contacts": contacts,
    }
    _validate_finger(result, context="finger")
    return result


def add_annotation(
    root: Path,
    *,
    clip_id: str,
    annotation_id: str,
    timestamp_ms: int,
    phase: str,
    position: int | None,
    technique: str,
    fingers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Add one timestamped, fully labeled five-finger observation."""
    resolved = validate_dataset_root(root)
    manifest = load_dataset(resolved)
    clip = next(
        (item for item in manifest["clips"] if item["clip_id"] == clip_id),
        None,
    )
    if clip is None:
        raise LocalEvalError(f"unknown clip_id: {clip_id}")
    finger_mapping = _require_mapping(fingers, context="fingers")
    _exact_keys(finger_mapping, set(FINGERS), context="fingers")
    annotation = {
        "annotation_id": annotation_id,
        "timestamp_ms": timestamp_ms,
        "phase": phase,
        "position": position,
        "technique": technique,
        "fingers": {finger: dict(finger_mapping[finger]) for finger in FINGERS},
    }
    clip["annotations"].append(annotation)
    clip["annotations"].sort(key=lambda item: item["timestamp_ms"])
    validate_manifest(manifest, root=resolved)
    _write_manifest(resolved, manifest)
    return annotation


def coverage_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Report whether the opt-in set covers the requested diversity axes."""
    clips = manifest["clips"]
    annotations = [annotation for clip in clips for annotation in clip["annotations"]]
    observed_positions = {
        annotation["position"]
        for annotation in annotations
        if annotation["phase"] == "stable" and annotation["position"] is not None
    }
    observed_techniques = {annotation["technique"] for annotation in annotations}
    observed_framings = {clip["capture"]["framing"] for clip in clips}
    observed_handedness = {clip["capture"]["handedness"] for clip in clips}
    observed_lighting = {clip["capture"]["lighting"] for clip in clips}
    observed_skin_tones = {
        clip["capture"]["skin_tone"]
        for clip in clips
        if clip["capture"]["skin_tone"] != "not_recorded"
    }
    observed_guitars = {clip["capture"]["guitar"] for clip in clips}
    observed_sleeves = {clip["capture"]["sleeve"] for clip in clips}
    observed_backgrounds = {clip["capture"]["background"] for clip in clips}
    pressing_fingers = {
        finger
        for annotation in annotations
        for finger, label in annotation["fingers"].items()
        if label["pressing"] is True
    }

    missing: list[str] = []
    for position in sorted(POSITIONS - observed_positions):
        missing.append(f"position:{position}")
    for technique in sorted(TECHNIQUES - observed_techniques):
        missing.append(f"technique:{technique}")
    for framing in sorted(FRAMINGS - observed_framings):
        missing.append(f"framing:{framing}")
    for hand in sorted(HANDEDNESS - observed_handedness):
        missing.append(f"handedness:{hand}")
    for lighting_value in sorted(
        {"bright", "dim", "warm", "cool", "uneven"} - observed_lighting
    ):
        missing.append(f"lighting:{lighting_value}")
    for finger in ("index", "middle", "ring", "pinky"):
        if finger not in pressing_fingers:
            missing.append(f"pressing_finger:{finger}")
    diversity = {
        "skin_tone": {"observed": len(observed_skin_tones), "required": 3},
        "guitar": {"observed": len(observed_guitars), "required": 2},
        "sleeve": {"observed": len(observed_sleeves), "required": 2},
        "background": {"observed": len(observed_backgrounds), "required": 2},
    }
    for name, counts in diversity.items():
        if counts["observed"] < counts["required"]:
            missing.append(
                f"diversity:{name}:{counts['observed']}/{counts['required']}"
            )

    return {
        "complete": not missing,
        "clip_count": len(clips),
        "annotation_count": len(annotations),
        "observed": {
            "positions": sorted(observed_positions),
            "techniques": sorted(observed_techniques),
            "framings": sorted(observed_framings),
            "handedness": sorted(observed_handedness),
            "lighting": sorted(observed_lighting),
            "skin_tones": sorted(observed_skin_tones),
            "guitars": sorted(observed_guitars),
            "sleeves": sorted(observed_sleeves),
            "backgrounds": sorted(observed_backgrounds),
            "pressing_fingers": sorted(pressing_fingers),
        },
        "diversity": diversity,
        "missing": missing,
    }


def validate_dataset(
    root: Path,
    *,
    require_coverage: bool = True,
) -> dict[str, Any]:
    """Validate integrity and optionally require all requested coverage axes."""
    resolved = validate_dataset_root(root)
    manifest = load_dataset(resolved)
    coverage = coverage_report(manifest)
    if require_coverage and not coverage["complete"]:
        raise LocalEvalError(
            "dataset is structurally valid but coverage is incomplete: "
            + ", ".join(coverage["missing"])
        )
    return coverage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_dataset_root(),
        help="local dataset root (default: ~/.tabvision/cache/fretcam_local_eval)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize an empty dataset")
    init_parser.add_argument("--dataset-id", required=True)

    clip_parser = subparsers.add_parser(
        "add-clip",
        help="register existing media and its capture/provenance metadata",
    )
    clip_parser.add_argument("--clip-id", required=True)
    clip_parser.add_argument(
        "--media",
        required=True,
        help="relative path under media/, for example media/session-01.mp4",
    )
    clip_parser.add_argument("--duration-ms", required=True, type=int)
    clip_parser.add_argument(
        "--provenance", choices=sorted(PROVENANCE_KINDS), required=True
    )
    clip_parser.add_argument("--source-uri", required=True)
    clip_parser.add_argument("--license", required=True)
    clip_parser.add_argument("--confirm-rights", action="store_true")
    clip_parser.add_argument(
        "--confirm-appearance-metadata-rights",
        action="store_true",
    )
    clip_parser.add_argument("--allow-redistribution", action="store_true")
    clip_parser.add_argument("--handedness", choices=sorted(HANDEDNESS), required=True)
    clip_parser.add_argument("--framing", choices=sorted(FRAMINGS), required=True)
    clip_parser.add_argument("--lighting", choices=sorted(LIGHTING), required=True)
    clip_parser.add_argument(
        "--skin-tone", choices=sorted(SKIN_TONES), default="not_recorded"
    )
    clip_parser.add_argument(
        "--appearance-basis",
        choices=sorted(APPEARANCE_BASES),
        default="not_recorded",
    )
    clip_parser.add_argument("--guitar", choices=sorted(GUITARS), required=True)
    clip_parser.add_argument("--sleeve", choices=sorted(SLEEVES), required=True)
    clip_parser.add_argument("--background", choices=sorted(BACKGROUNDS), required=True)
    clip_parser.add_argument(
        "--face-visibility",
        choices=sorted(FACE_VISIBILITY),
        default="none",
    )
    clip_parser.add_argument("--minors-present", action="store_true")

    annotation_parser = subparsers.add_parser(
        "add-annotation",
        help="add one timestamped five-finger annotation",
    )
    annotation_parser.add_argument("--clip-id", required=True)
    annotation_parser.add_argument("--annotation-id", required=True)
    annotation_parser.add_argument("--timestamp-ms", type=int, required=True)
    annotation_parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    annotation_parser.add_argument("--position", type=int)
    annotation_parser.add_argument(
        "--technique", choices=sorted(TECHNIQUES), required=True
    )
    for finger in FINGERS:
        annotation_parser.add_argument(
            f"--{finger}",
            required=True,
            help="visibility:pressing|hovering|unknown:string@fret,...",
        )

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate schema, privacy/provenance, media hashes, and coverage",
    )
    validate_parser.add_argument(
        "--schema-only",
        action="store_true",
        help="allow incomplete coverage while building the set",
    )
    subparsers.add_parser("coverage", help="print current coverage gaps")
    return parser


def _json_output(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            manifest = init_dataset(args.root, dataset_id=args.dataset_id)
            _json_output(
                {
                    "dataset_id": manifest["dataset_id"],
                    "manifest": str(_manifest_path(validate_dataset_root(args.root))),
                    "automatic_recording": False,
                }
            )
        elif args.command == "add-clip":
            clip = add_clip(
                args.root,
                clip_id=args.clip_id,
                media_path=args.media,
                duration_ms=args.duration_ms,
                provenance_kind=args.provenance,
                source_uri=args.source_uri,
                license_name=args.license,
                rights_verified=args.confirm_rights,
                appearance_metadata_rights_verified=(
                    args.confirm_appearance_metadata_rights
                ),
                redistribution_allowed=args.allow_redistribution,
                handedness=args.handedness,
                framing=args.framing,
                lighting=args.lighting,
                skin_tone=args.skin_tone,
                appearance_basis=args.appearance_basis,
                guitar=args.guitar,
                sleeve=args.sleeve,
                background=args.background,
                face_visibility=args.face_visibility,
                minors_present=args.minors_present,
            )
            _json_output(
                {
                    "clip_id": clip["clip_id"],
                    "media": clip["media"],
                    "automatic_recording": False,
                }
            )
        elif args.command == "add-annotation":
            annotation = add_annotation(
                args.root,
                clip_id=args.clip_id,
                annotation_id=args.annotation_id,
                timestamp_ms=args.timestamp_ms,
                phase=args.phase,
                position=args.position,
                technique=args.technique,
                fingers={
                    finger: parse_finger_spec(getattr(args, finger))
                    for finger in FINGERS
                },
            )
            _json_output(
                {
                    "clip_id": args.clip_id,
                    "annotation_id": annotation["annotation_id"],
                    "timestamp_ms": annotation["timestamp_ms"],
                }
            )
        elif args.command == "validate":
            coverage = validate_dataset(
                args.root,
                require_coverage=not args.schema_only,
            )
            _json_output({"valid": True, "coverage": coverage})
        else:
            manifest = load_dataset(args.root)
            _json_output(coverage_report(manifest))
        return 0
    except LocalEvalError as exc:
        print(f"fretcam-local-eval: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
