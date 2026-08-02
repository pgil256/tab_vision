"""Content identity and byte-compatible helpers for high-resolution event banks.

This module deliberately excludes experiment scoring and reporting code.  A
bank identity changes only when the event-producing implementation, artifacts,
execution contract, or runtime evidence changes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tabvision.audio.highres_ensemble import DEFAULT_ENSEMBLE_ARTIFACT, HighResEnsembleBackend
from tabvision.types import AudioEvent, SessionConfig

BANK_IDENTITY_FORMAT_VERSION = 2
BANK_SOURCE_FORMAT_VERSION = 1
HIGHRES_BANK_EXECUTION: Mapping[str, Any] = {
    "device": "cpu",
    "batch_size": 8,
    "onset_threshold": 0.3,
    "offset_threshold": 0.3,
    "frame_threshold": 0.1,
    "offline": True,
}
_RUNTIME_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "soundfile",
    "hf-midi-transcription",
    "piano-transcription-inference",
    "pretty-midi",
    "mido",
)
_CHECKPOINT_FILENAMES = ("guitar-gaps.pth", "guitar-fl.pth")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without embedding the file location."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bank_source_paths() -> tuple[tuple[str, Path], ...]:
    package_root = Path(__file__).resolve().parents[1]
    # ``HighResBackend`` is constructed with ``filter_config=False`` below, so
    # ``audio/filters.py`` is not on the successful bank-output path and is
    # deliberately excluded from this source identity.
    return (
        ("tabvision/audio/checkpoint_ensemble.py", package_root / "audio/checkpoint_ensemble.py"),
        ("tabvision/audio/highres.py", package_root / "audio/highres.py"),
        ("tabvision/audio/highres_ensemble.py", package_root / "audio/highres_ensemble.py"),
        ("tabvision/eval/highres_event_bank.py", Path(__file__).resolve()),
        ("tabvision/types.py", package_root / "types.py"),
    )


def bank_source_revision() -> dict[str, Any]:
    """Identify only repository bytes that can change banked event output."""

    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for logical_path, path in sorted(_bank_source_paths()):
        raw = path.read_bytes()
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        records.append(
            {
                "path": logical_path,
                "sha256": _sha256_bytes(raw),
                "size_bytes": len(raw),
            }
        )
    return {
        "format_version": BANK_SOURCE_FORMAT_VERSION,
        "source_sha256": digest.hexdigest(),
        "files": records,
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _direct_url_evidence(raw: str | None) -> dict[str, Any] | None:
    """Retain source provenance without hashing installation paths or URLs."""

    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"present": True, "valid_json": False}
    if not isinstance(payload, Mapping):
        return {"present": True, "valid_json": False}

    evidence: dict[str, Any] = {"present": True, "valid_json": True}
    vcs = payload.get("vcs_info")
    if isinstance(vcs, Mapping):
        evidence["vcs_info"] = {
            key: vcs[key]
            for key in ("vcs", "commit_id", "requested_revision")
            if isinstance(vcs.get(key), str)
        }
    archive = payload.get("archive_info")
    if isinstance(archive, Mapping):
        hashes = archive.get("hashes")
        if isinstance(hashes, Mapping):
            evidence["archive_hashes"] = {
                str(key): str(value) for key, value in sorted(hashes.items())
            }
        elif isinstance(archive.get("hash"), str):
            evidence["archive_hash"] = archive["hash"]
    directory = payload.get("dir_info")
    if isinstance(directory, Mapping):
        evidence["editable"] = directory.get("editable") is True
    return evidence


def _distribution_evidence(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "version": "not-installed",
            "record_sha256": None,
            "direct_url": None,
        }

    record = distribution.read_text("RECORD")
    direct_url = distribution.read_text("direct_url.json")
    return {
        "version": distribution.version,
        "record_sha256": (_sha256_bytes(record.encode("utf-8")) if record is not None else None),
        "direct_url": _direct_url_evidence(direct_url),
    }


def _bank_runtime_identity() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "cpu": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "model": _cpu_model(),
        },
        "packages": {name: _distribution_evidence(name) for name in _RUNTIME_DISTRIBUTIONS},
    }


def _highres_checkpoint_snapshot_root() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--xavriley--midi-transcription-models"
        / "snapshots"
    )


def _highres_checkpoint_records() -> tuple[list[dict[str, Any]], dict[str, str]]:
    root = _highres_checkpoint_snapshot_root()
    records: list[dict[str, Any]] = []
    locations: dict[str, str] = {}
    for filename in _CHECKPOINT_FILENAMES:
        matches = sorted(path for path in root.glob(f"*/{filename}") if path.is_file())
        identities = {
            (sha256_file(path), path.stat().st_size, path.parent.name): path for path in matches
        }
        if len(identities) != 1:
            raise RuntimeError(
                f"expected one exact cached highres identity for {filename}, "
                f"found {len(identities)}"
            )
        (digest, size_bytes, revision), path = next(iter(identities.items()))
        records.append(
            {
                "filename": filename,
                "sha256": digest,
                "size_bytes": size_bytes,
                "huggingface_revision": revision,
            }
        )
        locations[filename] = str(path.resolve())
    return records, locations


def _ensemble_artifact_record() -> tuple[dict[str, Any], str]:
    path = Path(DEFAULT_ENSEMBLE_ARTIFACT)
    return (
        {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        str(path.resolve()),
    )


def highres_bank_backend_identity() -> dict[str, Any]:
    """Build a path- and Git-independent identity for future event banks."""

    checkpoints, checkpoint_locations = _highres_checkpoint_records()
    ensemble_artifact, ensemble_location = _ensemble_artifact_record()
    identity_material = {
        "format_version": BANK_IDENTITY_FORMAT_VERSION,
        "backend": "highres-ensemble",
        "bank_source_revision": bank_source_revision(),
        "checkpoints": checkpoints,
        "ensemble_artifact": ensemble_artifact,
        "session": asdict(SessionConfig()),
        "execution": dict(HIGHRES_BANK_EXECUTION),
        "runtime": _bank_runtime_identity(),
    }
    return {
        **identity_material,
        "identity_sha256": _sha256_bytes(_canonical_json(identity_material).encode("utf-8")),
        "locations": {
            "checkpoints": checkpoint_locations,
            "ensemble_artifact": ensemble_location,
        },
    }


def new_highres_bank_backend() -> HighResEnsembleBackend:
    """Construct the frozen local-only CPU backend used to produce bank events."""

    _highres_checkpoint_records()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return HighResEnsembleBackend(
        device="cpu",
        batch_size=int(HIGHRES_BANK_EXECUTION["batch_size"]),
        onset_threshold=float(HIGHRES_BANK_EXECUTION["onset_threshold"]),
        offset_threshold=float(HIGHRES_BANK_EXECUTION["offset_threshold"]),
        frame_threshold=float(HIGHRES_BANK_EXECUTION["frame_threshold"]),
    )


def load_mono_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32 while preserving its original sample rate."""

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency readiness path
        raise RuntimeError("soundfile is required to load GuitarSet WAV files") from exc

    wav, sample_rate = sf.read(str(audio_path), always_2d=False)
    array = np.asarray(wav, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(f"expected mono/stereo audio, got shape {array.shape}")
    return array, int(sample_rate)


def _event_from_json(payload: Mapping[str, Any]) -> AudioEvent:
    tags = payload.get("tags", ())
    return AudioEvent(
        onset_s=float(payload["onset_s"]),
        offset_s=float(payload["offset_s"]),
        pitch_midi=int(payload["pitch_midi"]),
        velocity=float(payload["velocity"]),
        confidence=float(payload["confidence"]),
        tags=tuple(str(item) for item in tags) if isinstance(tags, (list, tuple)) else (),
    )


def events_to_json(events: Sequence[AudioEvent]) -> list[dict[str, Any]]:
    """Return the exact event payload shape used by the existing runner."""

    return [
        {
            "onset_s": float(event.onset_s),
            "offset_s": float(event.offset_s),
            "pitch_midi": int(event.pitch_midi),
            "velocity": float(event.velocity),
            "confidence": float(event.confidence),
            "tags": list(event.tags),
        }
        for event in events
    ]


def read_banked_events(path: str | Path) -> list[AudioEvent]:
    """Read, validate, and onset-sort the existing bank JSON representation."""

    cache_path = Path(path)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid banked event cache {cache_path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"banked event cache must be a JSON list: {cache_path}")
    try:
        return sorted(
            (_event_from_json(item) for item in payload if isinstance(item, Mapping)),
            key=lambda event: event.onset_s,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid event row in {cache_path}: {exc}") from exc


__all__ = [
    "HIGHRES_BANK_EXECUTION",
    "bank_source_revision",
    "events_to_json",
    "highres_bank_backend_identity",
    "load_mono_audio",
    "new_highres_bank_backend",
    "read_banked_events",
    "sha256_file",
]
