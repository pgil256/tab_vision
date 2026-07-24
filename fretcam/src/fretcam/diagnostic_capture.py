"""Explicit, bounded, local-only image diagnostics for FretCam.

Normal FretCam sessions never instantiate storage.  This module retains exact
JPEG packets in memory only after an explicit opt-in and writes a package only
after a second explicit confirmation.  Saved packages are troubleshooting
artifacts, never datasets, labels, accuracy evidence, or training inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import MappingProxyType

SCHEMA_VERSION = 1
SCHEMA_ID = "fretcam-local-diagnostic-package-v1"
TRACE_PACKAGE_KIND = "live_trace"
FAILURE_PACKAGE_KIND = "failure"

FINGER_ORDER = ("index", "middle", "ring", "pinky")
_FINGERS = frozenset(FINGER_ORDER)
_SAFE_FRAME_PATH = re.compile(r"frames/[0-9]{6}\.jpg")
_MAX_NOTE_CHARACTERS = 240

DIAGNOSTICS_POLICY: Mapping[str, object] = MappingProxyType(
    {
        "purpose": "local_diagnostics_only",
        "private_user_media": True,
        "automatic_collection": False,
        "training_allowed": False,
        "evaluation_allowed": False,
        "threshold_tuning_allowed": False,
        "release_evidence_allowed": False,
        "redistribution_allowed": False,
    }
)


class DiagnosticCaptureError(ValueError):
    """Raised when a local diagnostic request is invalid or unsafe."""


@dataclass(frozen=True)
class BufferLimits:
    """Hard in-memory bounds for one capture channel."""

    duration_s: float
    max_frames: int
    max_bytes: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if self.max_frames < 1:
            raise ValueError("max_frames must be positive")
        if self.max_bytes < 4:
            raise ValueError("max_bytes must be at least four")


TRACE_LIMITS = BufferLimits(
    duration_s=10.0,
    max_frames=120,
    max_bytes=24 * 1024 * 1024,
)
FAILURE_LIMITS = BufferLimits(
    duration_s=2.0,
    max_frames=24,
    max_bytes=6 * 1024 * 1024,
)


@dataclass(frozen=True)
class FailureExpectation:
    """User-supplied reproduction hint, explicitly not an evaluation label."""

    position: int | None
    pressing_fingers: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        position = self.position
        if isinstance(position, bool) or (
            position is not None
            and (not isinstance(position, int) or not 1 <= position <= 12)
        ):
            raise DiagnosticCaptureError("position must be 1-12 or unknown")
        raw_fingers = self.pressing_fingers
        if isinstance(raw_fingers, str):
            raise DiagnosticCaptureError("pressing_fingers must be an array")
        invalid = {value for value in raw_fingers if value not in _FINGERS}
        if invalid:
            raise DiagnosticCaptureError(
                "pressing_fingers may contain only index, middle, ring, and pinky"
            )
        canonical = tuple(finger for finger in FINGER_ORDER if finger in raw_fingers)
        object.__setattr__(self, "pressing_fingers", canonical)
        note = self.note
        if note is not None:
            if not isinstance(note, str):
                raise DiagnosticCaptureError("note must be a string")
            note = " ".join(note.split())
            if not note:
                note = None
            elif len(note) > _MAX_NOTE_CHARACTERS:
                raise DiagnosticCaptureError(
                    f"note must be at most {_MAX_NOTE_CHARACTERS} characters"
                )
            object.__setattr__(self, "note", note)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FailureExpectation:
        """Validate an expectation object from a JSON control payload."""

        unknown_keys = set(value) - {"position", "pressing_fingers", "note"}
        if unknown_keys:
            raise DiagnosticCaptureError(
                f"unknown failure expectation fields: {sorted(unknown_keys)}"
            )
        position = value.get("position")
        if position == "unknown":
            position = None
        raw_fingers = value.get("pressing_fingers", ())
        if not isinstance(raw_fingers, (list, tuple, set, frozenset)):
            raise DiagnosticCaptureError("pressing_fingers must be an array")
        return cls(
            position=position,  # type: ignore[arg-type]
            pressing_fingers=tuple(raw_fingers),  # type: ignore[arg-type]
            note=value.get("note"),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CaptureStatus:
    trace_enabled: bool
    trace_frames: int
    trace_bytes: int
    failure_enabled: bool
    failure_frames: int
    failure_bytes: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _CapturedFrame:
    sequence: int
    observed_at_s: float
    processor_timestamp_s: float
    jpeg: bytes
    hud_response: dict[str, object]
    client_metadata: dict[str, object]
    server_metadata: dict[str, object]


class _BoundedFrameBuffer:
    def __init__(self, limits: BufferLimits, *, rolling: bool) -> None:
        self.limits = limits
        self.rolling = rolling
        self.frames: deque[_CapturedFrame] = deque()
        self.total_bytes = 0

    def append(self, frame: _CapturedFrame) -> bool:
        payload_bytes = len(frame.jpeg)
        if payload_bytes > self.limits.max_bytes:
            raise DiagnosticCaptureError(
                "one JPEG exceeds the configured diagnostic byte limit"
            )
        if (
            not self.rolling
            and self.frames
            and (
                len(self.frames) >= self.limits.max_frames
                or self.total_bytes + payload_bytes > self.limits.max_bytes
                or frame.observed_at_s - self.frames[0].observed_at_s
                > self.limits.duration_s
            )
        ):
            return False
        self.frames.append(frame)
        self.total_bytes += payload_bytes
        while self.frames and (
            len(self.frames) > self.limits.max_frames
            or self.total_bytes > self.limits.max_bytes
            or frame.observed_at_s - self.frames[0].observed_at_s
            > self.limits.duration_s
        ):
            removed = self.frames.popleft()
            self.total_bytes -= len(removed.jpeg)
        return True

    def attach_client_metadata(
        self,
        sequence: int,
        metadata: Mapping[str, object],
    ) -> bool:
        for index, frame in enumerate(self.frames):
            if frame.sequence != sequence:
                continue
            self.frames[index] = replace(
                frame,
                client_metadata=dict(metadata),
            )
            return True
        return False

    def clear(self) -> None:
        self.frames.clear()
        self.total_bytes = 0


def default_diagnostics_root() -> Path:
    """Return the generated local cache root without creating it."""

    return Path.home() / ".tabvision" / "cache" / "fretcam_diagnostics"


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_resolving(path)
    for candidate in reversed((absolute, *absolute.parents)):
        if candidate.exists() and candidate.is_symlink():
            raise DiagnosticCaptureError(
                f"diagnostic paths must not contain symlinks: {candidate}"
            )


def _git_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def validate_diagnostics_root(root: Path) -> Path:
    """Resolve a storage root and reject Git-contained or symlinked paths."""

    _reject_symlink_components(root)
    resolved = _absolute_without_resolving(root).resolve(strict=False)
    git_root = _git_ancestor(resolved)
    if git_root is not None:
        raise DiagnosticCaptureError(
            f"diagnostic root must not be inside a Git repository: {git_root}"
        )
    return resolved


def validate_diagnostics_policy(value: object) -> None:
    """Enforce the immutable non-training/non-evaluation policy."""

    if not isinstance(value, Mapping) or dict(value) != DIAGNOSTICS_POLICY:
        raise DiagnosticCaptureError("diagnostic package policy is invalid")


def _json_copy(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    try:
        encoded = json.dumps(value, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DiagnosticCaptureError(f"{field} must be JSON-compatible") from exc
    if not isinstance(decoded, dict):
        raise DiagnosticCaptureError(f"{field} must be an object")
    return decoded


def _finite_timestamp(value: float, *, field: str) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticCaptureError(f"{field} must be finite") from exc
    if not math.isfinite(timestamp):
        raise DiagnosticCaptureError(f"{field} must be finite")
    return timestamp


def _hud_timestamp(response: Mapping[str, object]) -> float | None:
    detection = response.get("detection")
    if not isinstance(detection, Mapping):
        return None
    value = detection.get("timestamp_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    timestamp = float(value)
    return timestamp if math.isfinite(timestamp) else None


def _validate_jpeg(payload: bytes) -> bytes:
    jpeg = bytes(payload)
    if (
        len(jpeg) < 4
        or not jpeg.startswith(b"\xff\xd8")
        or not jpeg.endswith(b"\xff\xd9")
    ):
        raise DiagnosticCaptureError("diagnostic payload is not an exact JPEG packet")
    return jpeg


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _package_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{kind}-{stamp}-{secrets.token_hex(4)}"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _frame_record(
    frame: _CapturedFrame,
    *,
    first_observed_at_s: float,
    frame_path: str,
) -> dict[str, object]:
    server = dict(frame.server_metadata)
    server.update(
        {
            "observed_at_s": frame.observed_at_s,
            "relative_timestamp_s": frame.observed_at_s - first_observed_at_s,
            "processor_timestamp_s": frame.processor_timestamp_s,
        }
    )
    return {
        "sequence": frame.sequence,
        "path": frame_path,
        "payload_bytes": len(frame.jpeg),
        "sha256": hashlib.sha256(frame.jpeg).hexdigest(),
        "client": dict(frame.client_metadata),
        "server": server,
        "hud": dict(frame.hud_response),
    }


def _validate_replay_controls(
    controls: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    validated: list[dict[str, object]] = []
    for control in controls:
        copied = _json_copy(control, field="replay control")
        if (
            set(copied) != {"type", "player_handedness"}
            or copied.get("type") != "settings"
        ):
            raise DiagnosticCaptureError(
                "only handedness settings may be stored as replay controls"
            )
        if copied.get("player_handedness") not in {"right", "left"}:
            raise DiagnosticCaptureError(
                "replay control handedness must be 'right' or 'left'"
            )
        validated.append(copied)
    return tuple(validated)


class LocalCaptureSession:
    """Per-WebSocket, opt-in capture state with no automatic persistence."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        trace_limits: BufferLimits = TRACE_LIMITS,
        failure_limits: BufferLimits = FAILURE_LIMITS,
    ) -> None:
        self.root = validate_diagnostics_root(root or default_diagnostics_root())
        self.trace_limits = trace_limits
        self.failure_limits = failure_limits
        # A parity trace preserves its clean-session prefix. Rolling it would
        # discard the processor warm-up state needed for an exact replay.
        self._trace = _BoundedFrameBuffer(trace_limits, rolling=False)
        self._failure = _BoundedFrameBuffer(failure_limits, rolling=True)
        self._trace_enabled = False
        self._failure_enabled = False
        self._trace_session: dict[str, object] = {}
        self._trace_replay_controls: tuple[dict[str, object], ...] = ()
        self._next_sequence = 1
        self._last_observed_at_s: float | None = None
        self._lock = RLock()

    def start_trace(
        self,
        *,
        session_metadata: Mapping[str, object] | None = None,
        replay_controls: Sequence[Mapping[str, object]] = (),
    ) -> CaptureStatus:
        """Opt in to a fresh bounded parity trace without touching disk."""

        with self._lock:
            self._trace.clear()
            self._trace_enabled = True
            self._trace_session = _json_copy(
                session_metadata or {},
                field="trace session metadata",
            )
            self._trace_replay_controls = _validate_replay_controls(replay_controls)
            if self._failure_enabled:
                self._failure.clear()
            return self.status()

    def cancel_trace(self) -> CaptureStatus:
        """Disable trace capture and irreversibly clear its in-memory packets."""

        with self._lock:
            self._trace_enabled = False
            self._trace.clear()
            self._trace_session = {}
            self._trace_replay_controls = ()
            return self.status()

    def set_failure_buffer(self, enabled: bool) -> CaptureStatus:
        """Enable a new rolling failure window, or disable and clear it."""

        if not isinstance(enabled, bool):
            raise DiagnosticCaptureError("failure buffer enabled must be boolean")
        with self._lock:
            self._failure.clear()
            self._failure_enabled = enabled
            return self.status()

    def record_frame(
        self,
        payload: bytes,
        hud_response: Mapping[str, object],
        *,
        observed_at_s: float,
        processor_timestamp_s: float | None = None,
        client_metadata: Mapping[str, object] | None = None,
        server_metadata: Mapping[str, object] | None = None,
    ) -> int | None:
        """Retain one successfully processed exact JPEG for enabled channels."""

        with self._lock:
            if not self._trace_enabled and not self._failure_enabled:
                return None
            observed = _finite_timestamp(observed_at_s, field="observed_at_s")
            if (
                self._last_observed_at_s is not None
                and observed < self._last_observed_at_s
            ):
                raise DiagnosticCaptureError(
                    "observed_at_s must not move backwards within a session"
                )
            processor_timestamp = (
                _hud_timestamp(hud_response)
                if processor_timestamp_s is None
                else _finite_timestamp(
                    processor_timestamp_s,
                    field="processor_timestamp_s",
                )
            )
            if processor_timestamp is None:
                processor_timestamp = observed
            jpeg = _validate_jpeg(payload)
            if self._trace_enabled and len(jpeg) > self.trace_limits.max_bytes:
                raise DiagnosticCaptureError(
                    "one JPEG exceeds the configured trace byte limit"
                )
            if self._failure_enabled and len(jpeg) > self.failure_limits.max_bytes:
                raise DiagnosticCaptureError(
                    "one JPEG exceeds the configured failure-buffer byte limit"
                )
            response = _json_copy(hud_response, field="HUD response")
            client = _json_copy(
                client_metadata or {},
                field="client metadata",
            )
            server = _json_copy(
                server_metadata or {},
                field="server metadata",
            )
            frame = _CapturedFrame(
                sequence=self._next_sequence,
                observed_at_s=observed,
                processor_timestamp_s=processor_timestamp,
                jpeg=jpeg,
                hud_response=response,
                client_metadata=client,
                server_metadata=server,
            )
            retained = False
            if self._trace_enabled:
                retained = self._trace.append(frame) or retained
            if self._failure_enabled:
                retained = self._failure.append(frame) or retained
            self._next_sequence += 1
            self._last_observed_at_s = observed
            return frame.sequence if retained else None

    def attach_client_metadata(
        self,
        sequence: int,
        metadata: Mapping[str, object],
    ) -> None:
        """Attach post-frame browser context without delaying frame processing."""

        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise DiagnosticCaptureError("capture sequence must be a positive integer")
        copied = _json_copy(metadata, field="client metadata")
        with self._lock:
            attached = False
            if self._trace_enabled:
                attached = self._trace.attach_client_metadata(sequence, copied)
            if self._failure_enabled:
                attached = (
                    self._failure.attach_client_metadata(sequence, copied) or attached
                )
            if not attached and sequence >= self._next_sequence:
                raise DiagnosticCaptureError(
                    "frame context does not match a retained capture frame"
                )

    def save_trace(self, *, confirm: bool = False) -> Path:
        """Persist the current trace only after explicit confirmation."""

        if confirm is not True:
            raise DiagnosticCaptureError("explicit trace save confirmation is required")
        with self._lock:
            if not self._trace_enabled:
                raise DiagnosticCaptureError("trace capture is not enabled")
            frames = tuple(self._trace.frames)
            if not frames:
                raise DiagnosticCaptureError("trace capture has no frames")
            package = self._write_package(
                kind=TRACE_PACKAGE_KIND,
                frames=frames,
                limits=self.trace_limits,
                session=self._trace_session,
                replay_controls=self._trace_replay_controls,
                expectation=None,
            )
            self._trace_enabled = False
            self._trace.clear()
            self._trace_session = {}
            self._trace_replay_controls = ()
            return package

    def mark_failure(
        self,
        expectation: FailureExpectation | Mapping[str, object],
        *,
        confirm: bool = False,
    ) -> Path:
        """Persist the rolling failure window with a debug-only expectation."""

        if confirm is not True:
            raise DiagnosticCaptureError(
                "explicit failure diagnostic confirmation is required"
            )
        expected = (
            expectation
            if isinstance(expectation, FailureExpectation)
            else FailureExpectation.from_mapping(expectation)
        )
        with self._lock:
            if not self._failure_enabled:
                raise DiagnosticCaptureError("failure buffer is not enabled")
            frames = tuple(self._failure.frames)
            if not frames:
                raise DiagnosticCaptureError("failure buffer has no frames")
            package = self._write_package(
                kind=FAILURE_PACKAGE_KIND,
                frames=frames,
                limits=self.failure_limits,
                session={},
                replay_controls=(),
                expectation=expected,
            )
            self._failure.clear()
            return package

    def disconnect(self) -> None:
        """Clear and disable every transient buffer at a socket boundary."""

        with self._lock:
            self._trace_enabled = False
            self._failure_enabled = False
            self._trace.clear()
            self._failure.clear()
            self._trace_session = {}
            self._trace_replay_controls = ()
            self._next_sequence = 1
            self._last_observed_at_s = None

    def status(self) -> CaptureStatus:
        with self._lock:
            return CaptureStatus(
                trace_enabled=self._trace_enabled,
                trace_frames=len(self._trace.frames),
                trace_bytes=self._trace.total_bytes,
                failure_enabled=self._failure_enabled,
                failure_frames=len(self._failure.frames),
                failure_bytes=self._failure.total_bytes,
            )

    def _write_package(
        self,
        *,
        kind: str,
        frames: tuple[_CapturedFrame, ...],
        limits: BufferLimits,
        session: Mapping[str, object],
        replay_controls: Sequence[Mapping[str, object]],
        expectation: FailureExpectation | None,
    ) -> Path:
        root = validate_diagnostics_root(self.root)
        category_name = "traces" if kind == TRACE_PACKAGE_KIND else "failures"
        category = root / category_name
        package_id = _package_id(kind)
        final = category / package_id
        staging = category / f".{package_id}.tmp"
        category.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(category)
        if category.resolve(strict=True).parent != root.resolve(strict=True):
            raise DiagnosticCaptureError("diagnostic category escapes its root")
        staging.mkdir()
        try:
            frames_directory = staging / "frames"
            frames_directory.mkdir()
            first_timestamp = frames[0].observed_at_s
            records: list[dict[str, object]] = []
            for frame in frames:
                relative_path = f"frames/{frame.sequence:06d}.jpg"
                if _SAFE_FRAME_PATH.fullmatch(relative_path) is None:
                    raise DiagnosticCaptureError("generated frame path is unsafe")
                target = staging / Path(*relative_path.split("/"))
                _write_bytes(target, frame.jpeg)
                records.append(
                    _frame_record(
                        frame,
                        first_observed_at_s=first_timestamp,
                        frame_path=relative_path,
                    )
                )
            manifest: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "schema_id": SCHEMA_ID,
                "package_kind": kind,
                "package_id": package_id,
                "created_at": _now(),
                "policy": dict(DIAGNOSTICS_POLICY),
                "limits": asdict(limits),
                "session": dict(session),
                "replay_controls": [dict(value) for value in replay_controls],
                "frame_count": len(records),
                "total_payload_bytes": sum(len(frame.jpeg) for frame in frames),
                "duration_s": frames[-1].observed_at_s - first_timestamp,
                "frames": records,
            }
            if expectation is not None:
                manifest["expectation_role"] = "debug_reproduction_only"
                manifest["expectation"] = expectation.as_dict()
            serialized = json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            _write_text(staging / "manifest.json", serialized + "\n")
            staging.replace(final)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return final


__all__ = [
    "BufferLimits",
    "CaptureStatus",
    "DIAGNOSTICS_POLICY",
    "DiagnosticCaptureError",
    "FAILURE_LIMITS",
    "FAILURE_PACKAGE_KIND",
    "FINGER_ORDER",
    "FailureExpectation",
    "LocalCaptureSession",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TRACE_LIMITS",
    "TRACE_PACKAGE_KIND",
    "default_diagnostics_root",
    "validate_diagnostics_policy",
    "validate_diagnostics_root",
]
