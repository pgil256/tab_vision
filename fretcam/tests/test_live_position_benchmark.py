from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

import fretcam.live_position_benchmark as live_benchmark
from fretcam.benchmark import SYNTHETIC_JPEG
from fretcam.detection import ConfidenceFactors
from fretcam.live_position_benchmark import (
    LiveCondition,
    _PreparedFrame,
    _apply_camera_motion,
    _apply_lighting,
    _apply_temporary_occlusion,
    _prepare_frame,
    build_conditions,
    run_live_benchmark,
    summarize_results,
)
from fretcam.position_benchmark import (
    BenchmarkManifest,
    BenchmarkSequence,
    PositionLabel,
    _sample_times,
)


def _label(start_s: float, end_s: float) -> PositionLabel:
    return PositionLabel(
        start_s=start_s,
        end_s=end_s,
        state="stable",
        position=5,
        technique="note",
        visibility="full_neck",
        lighting="bright",
        verification="frame_reviewed",
        notes="unit fixture",
    )


def _sequence(
    sequence_id: str,
    source: str,
    split: str,
    *,
    end_s: float,
) -> BenchmarkSequence:
    return BenchmarkSequence(
        sequence_id=sequence_id,
        source=source,
        split=split,  # type: ignore[arg-type]
        start_s=0.0,
        end_s=end_s,
        labels=(_label(0.0, end_s),),
    )


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        version=1,
        name="live-unit",
        corpus="GAPS",
        corpus_license="CC-BY-NC-SA-4.0",
        public_only=True,
        sample_fps=10.0,
        annotation_policy="unit fixture",
        sequences=(
            _sequence("dev", "public-dev", "dev", end_s=0.2),
            _sequence("test", "public-test", "test", end_s=0.1),
        ),
    )


class FakeLiveProcessor:
    def __init__(self) -> None:
        self.warmed = False
        self.closed = False
        self.reset_count = 0
        self.controls: list[dict[str, object]] = []
        self.timestamps: list[float | None] = []
        self.factors = ConfidenceFactors(
            board=0.9,
            freshness=1.0,
            stability=0.8,
            landmark_quality=0.85,
            on_neck=1.0,
            finger_agreement=0.8,
            coarse_agreement=0.75,
            support_sufficiency=0.7,
            combined=0.72,
            chord_compatibility=0.6,
            blockers=(),
        )

    def warmup(self) -> None:
        self.warmed = True

    def reset(self) -> None:
        self.reset_count += 1

    def process_jpeg(
        self,
        payload: bytes,
        *,
        timestamp_s: float | None = None,
    ) -> dict[str, object]:
        assert payload.startswith(b"\xff\xd8")
        self.timestamps.append(timestamp_s)
        return {
            "type": "hud",
            "version": 2,
            "detection": {
                "confidence_factors": asdict(self.factors),
                "geometry_status": "tracked",
                "geometry_age_ms": 12.0,
            },
            "position": {
                "state": "locked",
                "position": 5,
                "confidence": 0.8,
                "raw_index_fret": 5.0,
            },
            "server_ms": 1.25,
        }

    def handle_control(self, message: dict[str, object]) -> dict[str, object]:
        self.controls.append(message)
        return {
            "type": "control",
            "status": "settings_applied",
            "player_handedness": message.get("player_handedness"),
        }

    def close(self) -> None:
        self.closed = True


def test_coverage_matrix_exercises_every_requested_axis_without_cartesian_cost() -> (
    None
):
    conditions = build_conditions()

    assert len(conditions) == 15
    assert {condition.sample_fps for condition in conditions} == {
        2.0,
        5.0,
        10.0,
        20.0,
    }
    assert {condition.jpeg_quality for condition in conditions} == {50, 72, 90}
    assert {condition.inference_size_px for condition in conditions} == {
        320,
        480,
        640,
    }
    assert {condition.lighting for condition in conditions} == {
        "native",
        "bright",
        "dim",
        "warm",
        "cool",
        "uneven",
    }
    assert {condition.perturbation for condition in conditions} == {
        "none",
        "occlusion",
        "camera_motion",
    }
    assert len({condition.condition_id for condition in conditions}) == 15

    cartesian = build_conditions(
        fps_values=(2.0, 10.0),
        jpeg_qualities=(50, 72),
        inference_sizes=(320,),
        lighting_values=("native", "dim"),
        perturbations=("none", "occlusion"),
        matrix="cartesian",
    )
    assert len(cartesian) == 16


def test_lighting_occlusion_and_motion_transforms_are_deterministic() -> None:
    x = np.linspace(0, 255, 80, dtype=np.uint8)
    frame = np.repeat(x[None, :, None], 40, axis=0)
    frame = np.repeat(frame, 3, axis=2)

    for lighting in ("bright", "dim", "warm", "cool", "uneven"):
        first = _apply_lighting(frame, lighting)  # type: ignore[arg-type]
        second = _apply_lighting(frame, lighting)  # type: ignore[arg-type]
        assert np.array_equal(first, second)
        assert first.shape == frame.shape
        assert not np.array_equal(first, frame)

    before = _apply_temporary_occlusion(frame, 0.2)
    during = _apply_temporary_occlusion(frame, 0.5)
    assert np.array_equal(before, frame)
    assert np.count_nonzero(during == 0) > np.count_nonzero(frame == 0)
    assert np.array_equal(
        _apply_camera_motion(frame, 0.37),
        _apply_camera_motion(frame, 0.37),
    )
    assert np.array_equal(frame[:, :, 0], np.repeat(x[None, :], 40, axis=0))


def test_prepared_browser_jpeg_honours_inference_size() -> None:
    frame = np.full((360, 640, 3), 127, dtype=np.uint8)
    payload = _prepare_frame(
        frame,
        label=None,
        condition=LiveCondition(
            sample_fps=10.0,
            jpeg_quality=50,
            inference_size_px=320,
            lighting="dim",
            perturbation="camera_motion",
        ),
        progress=0.5,
    )

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (180, 320)
    assert payload.startswith(b"\xff\xd8")


def test_live_runner_uses_real_websocket_and_preserves_response_factors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_payloads(
        sequence: BenchmarkSequence,
        condition: LiveCondition,
        _video_cache: Path,
    ) -> list[_PreparedFrame]:
        return [
            _PreparedFrame(timestamp_s=timestamp, payload=SYNTHETIC_JPEG)
            for timestamp in _sample_times(
                sequence.start_s,
                sequence.end_s,
                condition.sample_fps,
            )
        ]

    monkeypatch.setattr(
        live_benchmark,
        "_prepare_sequence_payloads",
        fake_payloads,
    )
    processor = FakeLiveProcessor()
    payload = run_live_benchmark(
        _manifest(),
        video_cache=tmp_path,
        conditions=(
            LiveCondition(
                sample_fps=10.0,
                jpeg_quality=72,
                inference_size_px=640,
                lighting="native",
                perturbation="none",
            ),
        ),
        splits={"dev"},
        processor_factory=lambda: processor,
        startup_timeout_s=10.0,
        pace=False,
    )

    result = payload["conditions"][0]
    predictions = result["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["confidence_factors"]["board"] == pytest.approx(0.9)
    assert predictions[0]["confidence_factors"]["chord_compatibility"] == (
        pytest.approx(0.6)
    )
    assert predictions[0]["geometry_status"] == "tracked"
    assert result["metrics"]["overall"]["displayed_position_precision"]["value"] == 1.0
    assert result["transport"]["frames"] == 2
    assert processor.warmed
    assert processor.reset_count == 1
    assert processor.controls == [{"type": "settings", "player_handedness": "right"}]
    assert len(processor.timestamps) == 2
    assert all(timestamp is not None for timestamp in processor.timestamps)
    assert processor.timestamps == sorted(processor.timestamps)
    assert processor.closed

    summary = summarize_results(payload)
    assert summary["conditions"][0]["blocker_counts"] == {}
    assert summary["conditions"][0]["displayed_position_precision"]["value"] == 1.0


def test_runner_rejects_non_public_or_non_gaps_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    private = BenchmarkManifest(
        **{
            **asdict(manifest),
            "corpus": "private",
            "public_only": False,
            "sequences": manifest.sequences,
        }
    )

    with pytest.raises(ValueError, match="public GAPS"):
        run_live_benchmark(
            private,
            video_cache=tmp_path,
            conditions=(build_conditions()[0],),
        )


def test_list_conditions_does_not_require_models_or_cached_media(
    capsys: pytest.CaptureFixture[str],
) -> None:
    live_benchmark.main(["--list-conditions"])

    listed = __import__("json").loads(capsys.readouterr().out)
    assert listed["count"] == 15
    assert listed["conditions"][0]["condition_id"].startswith("fps-10")
