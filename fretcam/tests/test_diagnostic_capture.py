from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fretcam.diagnostic_capture as capture_module
from fretcam.diagnostic_capture import (
    DIAGNOSTICS_POLICY,
    FAILURE_LIMITS,
    TRACE_LIMITS,
    BufferLimits,
    DiagnosticCaptureError,
    FailureExpectation,
    LocalCaptureSession,
    default_diagnostics_root,
    validate_diagnostics_root,
)


def _jpeg(marker: bytes = b"x", *, size: int | None = None) -> bytes:
    body = marker if size is None else marker * max(1, size - 4)
    return b"\xff\xd8" + body + b"\xff\xd9"


def _hud(timestamp_s: float, *, position: int = 5) -> dict[str, object]:
    return {
        "type": "hud",
        "detection": {
            "timestamp_s": timestamp_s,
            "geometry_status": "tracked",
            "detector_ran": False,
            "confidence_factors": {"blockers": []},
        },
        "position": {"state": "locked", "position": position},
        "server_ms": 12.5,
    }


def test_default_limits_and_root_match_the_explicit_local_policy() -> None:
    assert TRACE_LIMITS == BufferLimits(
        duration_s=10.0,
        max_frames=120,
        max_bytes=24 * 1024 * 1024,
    )
    assert FAILURE_LIMITS == BufferLimits(
        duration_s=2.0,
        max_frames=24,
        max_bytes=6 * 1024 * 1024,
    )
    assert default_diagnostics_root() == (
        Path.home() / ".tabvision" / "cache" / "fretcam_diagnostics"
    )
    assert DIAGNOSTICS_POLICY["automatic_collection"] is False
    assert DIAGNOSTICS_POLICY["training_allowed"] is False
    assert DIAGNOSTICS_POLICY["evaluation_allowed"] is False
    assert DIAGNOSTICS_POLICY["threshold_tuning_allowed"] is False
    assert DIAGNOSTICS_POLICY["release_evidence_allowed"] is False
    with pytest.raises(TypeError):
        DIAGNOSTICS_POLICY["evaluation_allowed"] = True  # type: ignore[index]


def test_disabled_and_cancelled_capture_never_create_storage(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    session = LocalCaptureSession(root=root)

    assert (
        session.record_frame(
            b"not-even-validated-while-disabled",
            {},
            observed_at_s=1.0,
        )
        is None
    )
    assert not root.exists()

    session.start_trace()
    session.record_frame(_jpeg(), _hud(2.0), observed_at_s=2.0)
    assert session.status().trace_frames == 1
    assert not root.exists()

    session.cancel_trace()
    session.set_failure_buffer(True)
    session.record_frame(_jpeg(b"f"), _hud(3.0), observed_at_s=3.0)
    session.disconnect()
    status = session.status()
    assert status.trace_enabled is False
    assert status.failure_enabled is False
    assert status.trace_frames == 0
    assert status.failure_frames == 0
    assert not root.exists()


def test_trace_and_failure_buffers_enforce_independent_bounds(tmp_path: Path) -> None:
    session = LocalCaptureSession(
        root=tmp_path / "diagnostics",
        trace_limits=BufferLimits(duration_s=1.0, max_frames=3, max_bytes=40),
        failure_limits=BufferLimits(duration_s=0.25, max_frames=2, max_bytes=20),
    )
    session.start_trace()
    session.set_failure_buffer(True)

    for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.6), start=1):
        session.record_frame(
            _jpeg(str(index).encode(), size=10),
            _hud(timestamp),
            observed_at_s=timestamp,
        )

    status = session.status()
    assert status.trace_frames == 3
    assert status.trace_bytes == 30
    assert status.failure_frames == 1
    assert status.failure_bytes == 10

    session.record_frame(
        _jpeg(b"z", size=10),
        _hud(1.3),
        observed_at_s=1.3,
    )
    status = session.status()
    # Exact traces retain the clean-session prefix needed for parity replay;
    # only the failure window rolls.
    assert status.trace_frames == 3
    assert status.failure_frames == 1


def test_frozen_trace_ignores_later_packets_and_remains_saveable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostics"
    session = LocalCaptureSession(
        root=root,
        trace_limits=BufferLimits(duration_s=10.0, max_frames=1, max_bytes=40),
    )
    session.start_trace()

    first_sequence = session.record_frame(
        _jpeg(b"first"),
        _hud(1.0),
        observed_at_s=1.0,
    )
    assert first_sequence == 1
    session.attach_client_metadata(first_sequence, {"sequence": 1})
    assert (
        session.record_frame(
            _jpeg(b"later"),
            _hud(1.1),
            observed_at_s=1.1,
        )
        is None
    )

    assert session.status().trace_enabled is True
    assert session.status().trace_frames == 1
    package = session.save_trace(confirm=True)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 1
    assert manifest["frames"][0]["client"]["sequence"] == 1


def test_trace_save_is_explicit_atomic_and_byte_exact(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    first = _jpeg(b"first")
    second = _jpeg(b"second")
    session = LocalCaptureSession(root=root)
    session.start_trace(
        session_metadata={"reason": "live-offline-parity"},
        replay_controls=({"type": "settings", "player_handedness": "left"},),
    )
    session.record_frame(
        first,
        _hud(10.0),
        observed_at_s=20.0,
        client_metadata={
            "sequence": 1,
            "source": {"width": 1280, "height": 720},
            "inference": {"width": 640, "height": 360, "jpeg_quality": 0.72},
        },
        server_metadata={"received_ns": 20_000_000_000},
    )
    session.record_frame(
        second,
        _hud(10.1),
        observed_at_s=20.1,
        client_metadata={"sequence": 2},
    )

    with pytest.raises(DiagnosticCaptureError, match="confirmation"):
        session.save_trace()
    assert not root.exists()

    package = session.save_trace(confirm=True)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_kind"] == "live_trace"
    assert manifest["policy"] == DIAGNOSTICS_POLICY
    assert manifest["frame_count"] == 2
    assert manifest["total_payload_bytes"] == len(first) + len(second)
    assert manifest["session"] == {"reason": "live-offline-parity"}
    assert manifest["replay_controls"] == [
        {"type": "settings", "player_handedness": "left"}
    ]
    assert not any(path.name.endswith(".tmp") for path in package.parent.iterdir())

    for expected, record in zip((first, second), manifest["frames"], strict=True):
        frame_path = package / Path(*record["path"].split("/"))
        assert frame_path.read_bytes() == expected
        assert record["sha256"] == hashlib.sha256(expected).hexdigest()
        assert not Path(record["path"]).is_absolute()
    assert manifest["frames"][0]["hud"]["position"]["position"] == 5
    assert manifest["frames"][0]["client"]["source"]["width"] == 1280
    assert manifest["frames"][0]["server"]["processor_timestamp_s"] == 10.0
    assert session.status().trace_enabled is False
    assert session.status().trace_frames == 0


@pytest.mark.parametrize(
    "value",
    [
        {"position": 0, "pressing_fingers": []},
        {"position": 13, "pressing_fingers": []},
        {"position": True, "pressing_fingers": []},
        {"position": 5, "pressing_fingers": ["thumb"]},
        {"position": 5, "pressing_fingers": "index"},
        {"position": 5, "pressing_fingers": [], "unexpected": True},
        {"position": 5, "pressing_fingers": [], "note": "x" * 241},
    ],
)
def test_failure_expectation_rejects_invalid_values(value: dict[str, object]) -> None:
    with pytest.raises(DiagnosticCaptureError):
        FailureExpectation.from_mapping(value)


def test_failure_mark_requires_confirmation_and_remains_debug_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostics"
    session = LocalCaptureSession(root=root)
    session.set_failure_buffer(True)
    session.record_frame(
        _jpeg(b"failure"),
        _hud(4.0, position=6),
        observed_at_s=4.0,
    )
    expectation = FailureExpectation.from_mapping(
        {
            "position": "unknown",
            "pressing_fingers": ["pinky", "index", "index"],
            "note": "  wrong lock after a slide  ",
        }
    )
    assert expectation.position is None
    assert expectation.pressing_fingers == ("index", "pinky")
    assert expectation.note == "wrong lock after a slide"

    with pytest.raises(DiagnosticCaptureError, match="confirmation"):
        session.mark_failure(expectation)
    assert not root.exists()

    package = session.mark_failure(expectation, confirm=True)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_kind"] == "failure"
    assert manifest["expectation_role"] == "debug_reproduction_only"
    assert manifest["expectation"] == {
        "position": None,
        "pressing_fingers": ["index", "pinky"],
        "note": "wrong lock after a slide",
    }
    assert manifest["policy"] == DIAGNOSTICS_POLICY
    assert session.status().failure_enabled is True
    assert session.status().failure_frames == 0


def test_git_roots_and_existing_symlinks_are_rejected(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(DiagnosticCaptureError, match="Git repository"):
        validate_diagnostics_root(repository / "local-private-diagnostics")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-root"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this Windows account cannot create symlinks")
    with pytest.raises(DiagnosticCaptureError, match="symlinks"):
        validate_diagnostics_root(link / "capture")


def test_failed_atomic_write_cleans_staging_and_keeps_memory_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostics"
    session = LocalCaptureSession(root=root)
    session.start_trace()
    session.record_frame(_jpeg(), _hud(1.0), observed_at_s=1.0)

    def fail_write(_path: Path, _payload: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(capture_module, "_write_bytes", fail_write)
    with pytest.raises(OSError, match="injected"):
        session.save_trace(confirm=True)

    trace_root = root / "traces"
    assert trace_root.is_dir()
    assert list(trace_root.iterdir()) == []
    assert session.status().trace_enabled is True
    assert session.status().trace_frames == 1
