"""Versioned cache for high-resolution events and raw posterior matrices."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tabvision.audio.highres import HighResPosteriors, HighResTranscription
from tabvision.types import AudioEvent

HIGHRES_CACHE_SCHEMA_VERSION = 1


class HighResCacheError(ValueError):
    """A high-resolution cache is corrupt or incompatible with the request."""


@dataclass(frozen=True)
class HighResCacheRecord:
    events: tuple[AudioEvent, ...]
    posteriors: HighResPosteriors | None
    provenance: dict[str, str]
    legacy: bool = False


def write_highres_cache(
    path: Path,
    result: HighResTranscription,
    *,
    provenance: Mapping[str, str],
) -> None:
    """Atomically write float16 posterior storage with validated metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": HIGHRES_CACHE_SCHEMA_VERSION,
        "backend": "highres",
        "checkpoint": result.posteriors.checkpoint,
        "frames_per_second": result.posteriors.frames_per_second,
        "begin_note": result.posteriors.begin_note,
        "provenance": {str(key): str(value) for key, value in provenance.items()},
        "event_tags": [list(event.tags) for event in result.events],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "event_onset_s": np.asarray([event.onset_s for event in result.events], dtype=np.float64),
        "event_offset_s": np.asarray([event.offset_s for event in result.events], dtype=np.float64),
        "event_pitch_midi": np.asarray(
            [event.pitch_midi for event in result.events], dtype=np.int16
        ),
        "event_velocity": np.asarray([event.velocity for event in result.events], dtype=np.float64),
        "event_confidence": np.asarray(
            [event.confidence for event in result.events], dtype=np.float64
        ),
        "reg_onset_output": result.posteriors.reg_onset_output.astype(np.float16),
        "reg_offset_output": result.posteriors.reg_offset_output.astype(np.float16),
        "frame_output": result.posteriors.frame_output.astype(np.float16),
        "velocity_output": result.posteriors.velocity_output.astype(np.float16),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **arrays)  # type: ignore[arg-type]
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_highres_cache(
    path: Path,
    *,
    expected_provenance: Mapping[str, str] | None = None,
    legacy_json_path: Path | None = None,
) -> HighResCacheRecord:
    """Read a validated cache, optionally falling back to old event-only JSON."""

    path = Path(path)
    if not path.is_file():
        if legacy_json_path is not None and Path(legacy_json_path).is_file():
            return _read_legacy_json(Path(legacy_json_path))
        raise HighResCacheError(f"highres cache is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = _metadata(payload)
            _validate_metadata(metadata, expected_provenance)
            posteriors = HighResPosteriors(
                checkpoint=str(metadata["checkpoint"]),
                frames_per_second=int(metadata["frames_per_second"]),
                begin_note=int(metadata["begin_note"]),
                reg_onset_output=_posterior_array(payload, "reg_onset_output"),
                reg_offset_output=_posterior_array(payload, "reg_offset_output"),
                frame_output=_posterior_array(payload, "frame_output"),
                velocity_output=_posterior_array(payload, "velocity_output"),
            )
            events = _events(payload, metadata, posteriors)
    except HighResCacheError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HighResCacheError(f"highres cache is corrupt: {path}: {exc}") from exc
    provenance = {str(key): str(value) for key, value in metadata["provenance"].items()}
    return HighResCacheRecord(events, posteriors, provenance)


def _metadata(payload: Any) -> dict[str, Any]:
    if "metadata" not in payload:
        raise HighResCacheError("highres cache metadata is missing")
    raw = payload["metadata"]
    if raw.shape != ():
        raise HighResCacheError("highres cache metadata must be scalar JSON")
    metadata = json.loads(str(raw.item()))
    if not isinstance(metadata, dict):
        raise HighResCacheError("highres cache metadata must be an object")
    return metadata


def _validate_metadata(
    metadata: Mapping[str, Any],
    expected_provenance: Mapping[str, str] | None,
) -> None:
    if metadata.get("schema_version") != HIGHRES_CACHE_SCHEMA_VERSION:
        raise HighResCacheError(
            f"unsupported highres cache schema: {metadata.get('schema_version')!r}"
        )
    if metadata.get("backend") != "highres":
        raise HighResCacheError("cache was not produced by the highres backend")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise HighResCacheError("highres cache provenance is missing")
    for key, expected in (expected_provenance or {}).items():
        actual = provenance.get(key)
        if str(actual) != str(expected):
            raise HighResCacheError(
                f"highres cache provenance mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )


def _posterior_array(payload: Any, key: str) -> np.ndarray:
    if key not in payload:
        raise HighResCacheError(f"highres cache posterior is missing: {key}")
    return np.asarray(payload[key], dtype=np.float32)


def _events(
    payload: Any,
    metadata: Mapping[str, Any],
    posteriors: HighResPosteriors,
) -> tuple[AudioEvent, ...]:
    keys = (
        "event_onset_s",
        "event_offset_s",
        "event_pitch_midi",
        "event_velocity",
        "event_confidence",
    )
    if any(key not in payload for key in keys):
        raise HighResCacheError("highres cache event arrays are incomplete")
    arrays = [np.asarray(payload[key]) for key in keys]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise HighResCacheError("highres cache event arrays have different lengths")
    tags = metadata.get("event_tags")
    if not isinstance(tags, list) or len(tags) != len(arrays[0]):
        raise HighResCacheError("highres cache event tags do not align")
    events = []
    for index in range(len(arrays[0])):
        onset = float(arrays[0][index])
        offset = float(arrays[1][index])
        pitch = int(arrays[2][index])
        velocity = float(arrays[3][index])
        confidence = float(arrays[4][index])
        values = (onset, offset, velocity, confidence)
        if any(not np.isfinite(value) for value in values):
            raise HighResCacheError("highres cache event values must be finite")
        if onset < 0.0 or offset < onset or not 0 <= pitch <= 127:
            raise HighResCacheError("highres cache contains an invalid event")
        raw_tags = tags[index]
        if not isinstance(raw_tags, list):
            raise HighResCacheError("highres cache event tags must be lists")
        events.append(
            AudioEvent(
                onset_s=onset,
                offset_s=offset,
                pitch_midi=pitch,
                velocity=velocity,
                confidence=confidence,
                pitch_logits=posteriors.pitch_logits_at(onset),
                tags=tuple(str(item) for item in raw_tags),
            )
        )
    return tuple(events)


def _read_legacy_json(path: Path) -> HighResCacheRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HighResCacheError(f"legacy highres cache is corrupt: {path}") from exc
    if not isinstance(payload, list):
        raise HighResCacheError("legacy highres cache must contain an event list")
    events = []
    for row in payload:
        if not isinstance(row, dict):
            raise HighResCacheError("legacy highres cache contains a non-object event")
        try:
            events.append(
                AudioEvent(
                    onset_s=float(row["onset_s"]),
                    offset_s=float(row["offset_s"]),
                    pitch_midi=int(row["pitch_midi"]),
                    velocity=float(row["velocity"]),
                    confidence=float(row["confidence"]),
                    tags=tuple(str(item) for item in row.get("tags", ())),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HighResCacheError(f"invalid legacy highres cache event: {exc}") from exc
    return HighResCacheRecord(tuple(events), None, {}, legacy=True)


__all__ = [
    "HIGHRES_CACHE_SCHEMA_VERSION",
    "HighResCacheError",
    "HighResCacheRecord",
    "read_highres_cache",
    "write_highres_cache",
]
