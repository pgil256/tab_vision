"""Load and compare exact FretCam live-path diagnostic traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import cv2
import numpy as np

from fretcam.diagnostic_capture import (
    DIAGNOSTICS_POLICY,
    FAILURE_PACKAGE_KIND,
    SCHEMA_ID,
    SCHEMA_VERSION,
    TRACE_LIMITS,
    TRACE_PACKAGE_KIND,
    DiagnosticCaptureError,
    validate_diagnostics_policy,
)

_FRAME_PATH = re.compile(r"frames/[0-9]{6}\.jpg")
_MISSING = object()
_BROWSER_CONTEXT_FIELDS = frozenset(
    {
        "sequence",
        "session_offset_ms",
        "source_width",
        "source_height",
        "inference_width",
        "inference_height",
        "jpeg_quality",
        "payload_bytes",
    }
)
_DEFAULT_COMPARE_FIELDS = (
    "frame.width",
    "frame.height",
    "position.state",
    "position.position",
    "position.reason",
    "position.raw_index_fret",
    "position.smoothed_index_fret",
    "position.confidence",
    "position.temporal_agreement",
    "position.observation_confidence",
    "position.stable_for_ms",
    "detection.confidence_factors.blockers",
    "detection.confidence_factors.board",
    "detection.confidence_factors.freshness",
    "detection.confidence_factors.stability",
    "detection.confidence_factors.landmark_quality",
    "detection.confidence_factors.on_neck",
    "detection.confidence_factors.finger_agreement",
    "detection.confidence_factors.coarse_agreement",
    "detection.confidence_factors.support_sufficiency",
    "detection.confidence_factors.combined",
    "detection.index_fret",
    "detection.index_fret_raw",
    "detection.position_fret",
    "detection.observation_confidence",
    "detection.finger_contacts",
    "detection.neck_locked",
    "detection.fret_map_locked",
    "detection.homography_confidence",
    "detection.geometry_status",
    "detection.homography_method",
    "detection.geometry_age_ms",
    "detection.detector_age_ms",
    "detection.geometry_stability",
    "detection.detector_ran",
    "detection.detector_result_consumed",
    "detection.detector_result_accepted",
    "detection.detector_requested",
    "detection.detector_pending",
    "detection.hand_detector_ran",
    "detection.hand_detector_calls",
    "detection.hand_source",
    "detection.hand_search_source",
    "detection.hand_search_attempts",
    "detection.hand_refresh_reason",
    "detection.hand_detector_interval_ms",
    "detection.hand_schedule_mode",
    "detection.hand_tracking_quality",
    "detection.hand_pose_quality",
    "detection.hand_pose_continuity",
    "detection.hand_pose_identity_score",
    "detection.hand_pose_residual_fraction",
    "detection.hand_pose_predicted",
)


class TraceError(DiagnosticCaptureError):
    """Raised when a trace is invalid or cannot be compared safely."""


class TraceProcessor(Protocol):
    def warmup(self) -> None: ...

    def reset(self) -> None: ...

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float,
    ) -> Mapping[str, object]: ...

    def handle_control(self, message: dict[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...


TraceProcessorFactory = Callable[[], TraceProcessor]


@dataclass(frozen=True)
class TraceFrame:
    sequence: int
    relative_timestamp_s: float
    processor_timestamp_s: float
    payload: bytes
    client_metadata: dict[str, object]
    server_metadata: dict[str, object]
    live_hud: dict[str, object]


@dataclass(frozen=True)
class LoadedTrace:
    package_path: Path
    package_id: str
    session: dict[str, object]
    replay_controls: tuple[dict[str, object], ...]
    frames: tuple[TraceFrame, ...]


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_resolving(path)
    for candidate in reversed((absolute, *absolute.parents)):
        if candidate.exists() and candidate.is_symlink():
            raise TraceError(f"trace paths must not contain symlinks: {candidate}")


def _git_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TraceError(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TraceError(f"{field} must be an array")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"{field} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TraceError(f"{field} must be finite")
    return parsed


def _json_object_copy(value: object, *, field: str) -> dict[str, object]:
    mapping = _mapping(value, field=field)
    try:
        encoded = json.dumps(mapping, allow_nan=False)
        copied = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TraceError(f"{field} must be JSON-compatible") from exc
    assert isinstance(copied, dict)
    return copied


def _safe_frame_file(package: Path, value: object) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise TraceError("frame path must be a relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or _FRAME_PATH.fullmatch(value) is None:
        raise TraceError("frame path is unsafe")
    unresolved = package / Path(*pure.parts)
    _reject_symlink_components(unresolved)
    target = unresolved.resolve(strict=False)
    try:
        target.relative_to(package)
    except ValueError as exc:
        raise TraceError("frame path escapes its trace package") from exc
    if not target.is_file():
        raise TraceError(f"trace frame is missing: {value}")
    return target


def _validate_replay_controls(value: object) -> tuple[dict[str, object], ...]:
    controls = _list(value, field="replay_controls")
    validated: list[dict[str, object]] = []
    for index, raw in enumerate(controls):
        control = _json_object_copy(raw, field=f"replay_controls[{index}]")
        if set(control) != {"type", "player_handedness"}:
            raise TraceError("trace replay controls contain unsupported fields")
        if control.get("type") != "settings" or control.get(
            "player_handedness"
        ) not in {"right", "left"}:
            raise TraceError("trace replay control is not a valid settings message")
        validated.append(control)
    return tuple(validated)


def load_trace(package_path: Path) -> LoadedTrace:
    """Load, contain, hash-check, and validate one exact-JPEG trace."""

    _reject_symlink_components(package_path)
    package = _absolute_without_resolving(package_path).resolve(strict=True)
    if not package.is_dir():
        raise TraceError("trace package must be a directory")
    git_root = _git_ancestor(package)
    if git_root is not None:
        raise TraceError(f"trace package must not be inside Git: {git_root}")
    manifest_path = package / "manifest.json"
    _reject_symlink_components(manifest_path)
    if not manifest_path.is_file():
        raise TraceError("trace manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError("trace manifest is not valid JSON") from exc
    manifest = _mapping(manifest, field="manifest")
    package_kind = manifest.get("package_kind")
    if package_kind == FAILURE_PACKAGE_KIND:
        raise TraceError("failure diagnostic packages cannot be replayed as traces")
    if package_kind != TRACE_PACKAGE_KIND:
        raise TraceError("package is not a FretCam live trace")
    if "expectation" in manifest or "expectation_role" in manifest:
        raise TraceError("failure expectations are forbidden in replay traces")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("schema_id") != SCHEMA_ID
    ):
        raise TraceError("trace schema is unsupported")
    validate_diagnostics_policy(manifest.get("policy"))
    package_id = manifest.get("package_id")
    if not isinstance(package_id, str) or not package_id:
        raise TraceError("trace package_id is invalid")
    frame_count = _integer(manifest.get("frame_count"), field="frame_count", minimum=1)
    if frame_count > TRACE_LIMITS.max_frames:
        raise TraceError("trace exceeds its frame limit")
    declared_total = _integer(
        manifest.get("total_payload_bytes"),
        field="total_payload_bytes",
    )
    if declared_total > TRACE_LIMITS.max_bytes:
        raise TraceError("trace exceeds its byte limit")
    duration_s = _finite(manifest.get("duration_s"), field="duration_s")
    if not 0.0 <= duration_s <= TRACE_LIMITS.duration_s + 1e-9:
        raise TraceError("trace exceeds its duration limit")
    session = _json_object_copy(manifest.get("session"), field="session")
    replay_controls = _validate_replay_controls(manifest.get("replay_controls"))
    raw_frames = _list(manifest.get("frames"), field="frames")
    if len(raw_frames) != frame_count:
        raise TraceError("trace frame_count does not match frames")

    frames: list[TraceFrame] = []
    sequences: set[int] = set()
    paths: set[str] = set()
    actual_total = 0
    previous_relative = -math.inf
    previous_processor = -math.inf
    previous_client_sequence = 0
    browser_trace = session.get("source") == "browser_live"
    for index, raw in enumerate(raw_frames):
        record = _mapping(raw, field=f"frames[{index}]")
        sequence = _integer(
            record.get("sequence"),
            field=f"frames[{index}].sequence",
            minimum=1,
        )
        if sequence in sequences:
            raise TraceError("trace frame sequences must be unique")
        sequences.add(sequence)
        relative_path = record.get("path")
        if not isinstance(relative_path, str) or relative_path in paths:
            raise TraceError("trace frame paths must be unique strings")
        paths.add(relative_path)
        frame_path = _safe_frame_file(package, relative_path)
        payload = frame_path.read_bytes()
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise TraceError(f"trace frame is not an exact JPEG: {relative_path}")
        payload_bytes = _integer(
            record.get("payload_bytes"),
            field=f"frames[{index}].payload_bytes",
            minimum=4,
        )
        if len(payload) != payload_bytes:
            raise TraceError(f"trace frame byte count differs: {relative_path}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or hashlib.sha256(payload).hexdigest() != digest:
            raise TraceError(f"trace frame hash differs: {relative_path}")
        actual_total += len(payload)
        server = _json_object_copy(
            record.get("server"),
            field=f"frames[{index}].server",
        )
        relative = _finite(
            server.get("relative_timestamp_s"),
            field=f"frames[{index}].server.relative_timestamp_s",
        )
        processor_timestamp = _finite(
            server.get("processor_timestamp_s"),
            field=f"frames[{index}].server.processor_timestamp_s",
        )
        if relative < previous_relative or processor_timestamp < previous_processor:
            raise TraceError("trace timestamps must be nondecreasing")
        previous_relative = relative
        previous_processor = processor_timestamp
        client_metadata = _json_object_copy(
            record.get("client"),
            field=f"frames[{index}].client",
        )
        live_hud = _json_object_copy(
            record.get("hud"),
            field=f"frames[{index}].hud",
        )
        if browser_trace:
            if set(client_metadata) != _BROWSER_CONTEXT_FIELDS:
                raise TraceError(
                    "browser trace frame context is incomplete or unsupported"
                )
            client_sequence = _integer(
                client_metadata.get("sequence"),
                field=f"frames[{index}].client.sequence",
                minimum=1,
            )
            if client_sequence <= previous_client_sequence:
                raise TraceError("browser frame-context sequences must increase")
            previous_client_sequence = client_sequence
            context_payload_bytes = _integer(
                client_metadata.get("payload_bytes"),
                field=f"frames[{index}].client.payload_bytes",
                minimum=4,
            )
            if context_payload_bytes != len(payload):
                raise TraceError("browser frame-context byte count differs")
            inference_width = _integer(
                client_metadata.get("inference_width"),
                field=f"frames[{index}].client.inference_width",
                minimum=1,
            )
            inference_height = _integer(
                client_metadata.get("inference_height"),
                field=f"frames[{index}].client.inference_height",
                minimum=1,
            )
            _integer(
                client_metadata.get("source_width"),
                field=f"frames[{index}].client.source_width",
                minimum=1,
            )
            _integer(
                client_metadata.get("source_height"),
                field=f"frames[{index}].client.source_height",
                minimum=1,
            )
            jpeg_quality = _finite(
                client_metadata.get("jpeg_quality"),
                field=f"frames[{index}].client.jpeg_quality",
            )
            session_offset_ms = _finite(
                client_metadata.get("session_offset_ms"),
                field=f"frames[{index}].client.session_offset_ms",
            )
            if not 0.0 <= jpeg_quality <= 1.0 or session_offset_ms < 0.0:
                raise TraceError("browser frame context contains an invalid value")
            decoded = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if decoded is None or decoded.shape[:2] != (
                inference_height,
                inference_width,
            ):
                raise TraceError("browser frame dimensions differ from exact JPEG")
            hud_frame = _mapping(
                live_hud.get("frame"),
                field=f"frames[{index}].hud.frame",
            )
            if (
                hud_frame.get("width") != inference_width
                or hud_frame.get("height") != inference_height
            ):
                raise TraceError("live HUD dimensions differ from browser context")
        frames.append(
            TraceFrame(
                sequence=sequence,
                relative_timestamp_s=relative,
                processor_timestamp_s=processor_timestamp,
                payload=payload,
                client_metadata=client_metadata,
                server_metadata=server,
                live_hud=live_hud,
            )
        )
    if actual_total != declared_total:
        raise TraceError("trace total_payload_bytes does not match frame files")
    if frames and not math.isclose(
        frames[-1].relative_timestamp_s,
        duration_s,
        abs_tol=1e-9,
    ):
        raise TraceError("trace duration_s does not match frame timestamps")
    return LoadedTrace(
        package_path=package,
        package_id=package_id,
        session=session,
        replay_controls=replay_controls,
        frames=tuple(frames),
    )


def _value_at_path(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    if path.endswith(".blockers") and isinstance(current, (list, tuple)):
        return tuple(sorted(str(item) for item in current))
    return current


def _equivalent(left: object, right: object) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
    return left == right


def _render_value(value: object) -> object:
    return "<missing>" if value is _MISSING else value


def _default_processor_factory() -> TraceProcessor:
    from fretcam.processing import HudFrameProcessor

    return HudFrameProcessor()  # type: ignore[return-value]


def compare_trace(
    trace: Path | LoadedTrace,
    *,
    processor_factory: TraceProcessorFactory | None = None,
    pace: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    compare_fields: Sequence[str] = _DEFAULT_COMPARE_FIELDS,
) -> dict[str, object]:
    """Replay exact packets and report selected live/replay differences per frame."""

    loaded = load_trace(trace) if isinstance(trace, Path) else trace
    if not loaded.frames:
        raise TraceError("trace has no frames")
    factory = processor_factory or _default_processor_factory
    processor = factory()
    frame_results: list[dict[str, object]] = []
    divergence_counts: Counter[str] = Counter()
    try:
        processor.warmup()
        for control in loaded.replay_controls:
            processor.handle_control(dict(control))
        processor.reset()
        started = clock()
        for frame in loaded.frames:
            if pace:
                remaining = frame.relative_timestamp_s - (clock() - started)
                if remaining > 0.0:
                    sleeper(remaining)
            try:
                replay_raw = processor.process_jpeg(
                    frame.payload,
                    timestamp_s=frame.processor_timestamp_s,
                )
            except TypeError as exc:
                raise TraceError(
                    "trace processor must accept process_jpeg(..., timestamp_s=...)"
                ) from exc
            replay = _json_object_copy(replay_raw, field="replay HUD response")
            differences: list[dict[str, object]] = []
            for field in compare_fields:
                live_value = _value_at_path(frame.live_hud, field)
                replay_value = _value_at_path(replay, field)
                if _equivalent(live_value, replay_value):
                    continue
                divergence_counts[field] += 1
                differences.append(
                    {
                        "field": field,
                        "live": _render_value(live_value),
                        "replay": _render_value(replay_value),
                    }
                )
            frame_results.append(
                {
                    "sequence": frame.sequence,
                    "relative_timestamp_s": frame.relative_timestamp_s,
                    "processor_timestamp_s": frame.processor_timestamp_s,
                    "matched": not differences,
                    "differences": differences,
                }
            )
    finally:
        processor.close()
    mismatched = sum(not bool(frame["matched"]) for frame in frame_results)
    return {
        "schema_version": 1,
        "comparison": "fretcam-live-trace-parity",
        "policy": dict(DIAGNOSTICS_POLICY),
        "trace_id": loaded.package_id,
        "paced": pace,
        "frames": len(frame_results),
        "matched_frames": len(frame_results) - mismatched,
        "mismatched_frames": mismatched,
        "divergence_counts": dict(sorted(divergence_counts.items())),
        "frame_results": frame_results,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--no-pace",
        action="store_true",
        help="replay immediately while preserving recorded inference timestamps",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    result = compare_trace(args.trace, pace=not args.no_pace)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()


__all__ = [
    "LoadedTrace",
    "TraceError",
    "TraceFrame",
    "TraceProcessor",
    "TraceProcessorFactory",
    "compare_trace",
    "load_trace",
    "main",
]
