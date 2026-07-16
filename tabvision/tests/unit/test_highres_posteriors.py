from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tabvision.audio.highres import (
    HIGHRES_BEGIN_NOTE,
    HIGHRES_CLASSES,
    HighResBackend,
    HighResPosteriors,
    HighResTranscription,
    _posteriors_from_output,
)
from tabvision.audio.highres_cache import (
    HIGHRES_CACHE_SCHEMA_VERSION,
    HighResCacheError,
    read_highres_cache,
    write_highres_cache,
)
from tabvision.errors import BackendError
from tabvision.types import AudioEvent, SessionConfig


def _posterior_fixture(frames: int = 12) -> HighResPosteriors:
    onset = np.full((frames, HIGHRES_CLASSES), 0.01, dtype=np.float32)
    offset = np.full_like(onset, 0.02)
    frame = np.full_like(onset, 0.03)
    velocity = np.full_like(onset, 0.04)
    onset[4, 64 - HIGHRES_BEGIN_NOTE] = 0.8
    frame[4:7, 64 - HIGHRES_BEGIN_NOTE] = (0.6, 0.7, 0.5)
    return HighResPosteriors(
        "guitar_gaps",
        100,
        HIGHRES_BEGIN_NOTE,
        onset,
        offset,
        frame,
        velocity,
    )


def test_fixed_backend_output_extracts_real_pitch_logits() -> None:
    fixture = _posterior_fixture()
    output = {
        "reg_onset_output": fixture.reg_onset_output,
        "reg_offset_output": fixture.reg_offset_output,
        "frame_output": fixture.frame_output,
        "velocity_output": fixture.velocity_output,
    }

    extracted = _posteriors_from_output(output, checkpoint="guitar_gaps")
    logits = extracted.pitch_logits_at(0.04, radius_frames=0)

    assert logits.shape == (128,)
    assert logits.dtype == np.float32
    assert logits[64] == pytest.approx(np.log(0.8 / 0.2), rel=1e-6)
    assert int(np.argmax(logits)) == 64
    assert extracted.event_scores(0.04, 64) == pytest.approx((0.8, 0.7))


def test_posterior_output_is_trimmed_to_source_duration() -> None:
    padded = np.zeros((1001, HIGHRES_CLASSES), dtype=np.float32)
    output = {
        key: padded
        for key in (
            "reg_onset_output",
            "reg_offset_output",
            "frame_output",
            "velocity_output",
        )
    }

    extracted = _posteriors_from_output(
        output,
        checkpoint="guitar_gaps",
        duration_s=5.0,
    )

    assert extracted.frame_count == 501


def test_exact_segment_boundary_accepts_nonduplicated_endpoint_frame() -> None:
    exact_boundary = np.zeros((6000, HIGHRES_CLASSES), dtype=np.float32)
    output = {
        key: exact_boundary
        for key in (
            "reg_onset_output",
            "reg_offset_output",
            "frame_output",
            "velocity_output",
        )
    }

    extracted = _posteriors_from_output(
        output,
        checkpoint="guitar_gaps",
        duration_s=60.0,
    )

    assert extracted.frame_count == 6000


def test_posterior_output_shorter_than_full_frame_coverage_is_rejected() -> None:
    too_short = np.zeros((5999, HIGHRES_CLASSES), dtype=np.float32)
    output = {
        key: too_short
        for key in (
            "reg_onset_output",
            "reg_offset_output",
            "frame_output",
            "velocity_output",
        )
    }

    with pytest.raises(BackendError, match="shorter than the source duration"):
        _posteriors_from_output(output, checkpoint="guitar_gaps", duration_s=60.0)


def test_invalid_posterior_shape_is_rejected() -> None:
    wrong = np.zeros((10, 87), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(frames, 88\)"):
        HighResPosteriors("guitar", 100, 21, wrong, wrong, wrong, wrong)


def test_default_transcribe_path_removes_logits_without_changing_events(monkeypatch) -> None:
    fixture = _posterior_fixture()
    event = AudioEvent(0.04, 0.3, 64, 0.75, 0.75, pitch_logits=fixture.pitch_logits_at(0.04))
    backend = HighResBackend(include_pitch_logits=False)
    monkeypatch.setattr(
        backend,
        "transcribe_with_posteriors",
        lambda *_args: HighResTranscription((event,), fixture),
    )

    actual = backend.transcribe(np.zeros(8, dtype=np.float32), 16_000, SessionConfig())

    assert len(actual) == 1
    assert actual[0].pitch_logits is None
    assert (
        actual[0].onset_s,
        actual[0].offset_s,
        actual[0].pitch_midi,
        actual[0].velocity,
        actual[0].confidence,
    ) == (0.04, 0.3, 64, 0.75, 0.75)


def test_posterior_cache_round_trip_and_provenance(tmp_path: Path) -> None:
    fixture = _posterior_fixture()
    event = AudioEvent(
        0.04,
        0.3,
        64,
        0.75,
        0.75,
        pitch_logits=fixture.pitch_logits_at(0.04),
        tags=("fixture",),
    )
    path = tmp_path / "track.npz"
    provenance = {"source_sha256": "abc", "checkpoint_sha256": "def"}
    write_highres_cache(
        path,
        HighResTranscription((event,), fixture),
        provenance=provenance,
    )

    restored = read_highres_cache(path, expected_provenance=provenance)

    assert restored.legacy is False
    assert restored.provenance == provenance
    assert restored.posteriors is not None
    np.testing.assert_allclose(
        restored.posteriors.reg_onset_output,
        fixture.reg_onset_output,
        atol=5e-4,
    )
    assert restored.events[0].tags == ("fixture",)
    assert int(np.argmax(restored.events[0].pitch_logits)) == 64


def test_posterior_cache_rejects_provenance_mismatch(tmp_path: Path) -> None:
    fixture = _posterior_fixture()
    path = tmp_path / "track.npz"
    write_highres_cache(
        path,
        HighResTranscription((), fixture),
        provenance={"checkpoint_sha256": "expected"},
    )

    with pytest.raises(HighResCacheError, match="provenance mismatch"):
        read_highres_cache(path, expected_provenance={"checkpoint_sha256": "other"})


def test_posterior_cache_rejects_corrupt_schema_and_shape(tmp_path: Path) -> None:
    path = tmp_path / "broken.npz"
    metadata = {
        "schema_version": HIGHRES_CACHE_SCHEMA_VERSION + 1,
        "backend": "highres",
        "checkpoint": "guitar",
        "frames_per_second": 100,
        "begin_note": 21,
        "provenance": {},
        "event_tags": [],
    }
    np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(HighResCacheError, match="unsupported highres cache schema"):
        read_highres_cache(path)

    metadata["schema_version"] = HIGHRES_CACHE_SCHEMA_VERSION
    wrong = np.zeros((3, 87), dtype=np.float16)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        event_onset_s=np.asarray([]),
        event_offset_s=np.asarray([]),
        event_pitch_midi=np.asarray([]),
        event_velocity=np.asarray([]),
        event_confidence=np.asarray([]),
        reg_onset_output=wrong,
        reg_offset_output=wrong,
        frame_output=wrong,
        velocity_output=wrong,
    )
    with pytest.raises(HighResCacheError, match=r"shape \(frames, 88\)"):
        read_highres_cache(path)


def test_old_event_only_cache_falls_back_without_fabricated_logits(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            [
                {
                    "onset_s": 0.1,
                    "offset_s": 0.4,
                    "pitch_midi": 64,
                    "velocity": 0.75,
                    "confidence": 0.75,
                    "tags": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    restored = read_highres_cache(tmp_path / "missing.npz", legacy_json_path=legacy)

    assert restored.legacy is True
    assert restored.posteriors is None
    assert restored.events[0].pitch_logits is None
