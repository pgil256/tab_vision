"""Unit tests for ``tabvision.pipeline.run_pipeline``.

Uses recording fake backends + a monkeypatched ``demux`` so no real
video file or model weights are touched.
"""

from __future__ import annotations

import numpy as np
import pytest

import tabvision.pipeline as pipeline
from tabvision.types import (
    AudioEvent,
    DemuxResult,
    FrameFingering,
    GuitarBBox,
    Homography,
)
from tabvision.video.hand.neck_anchor import HandNeckAnchor

# ---------- fakes ----------


def _make_demux_result(
    n_frames: int = 30, fps: float = 30.0, sr: int = 22050, audio_seconds: float = 1.0
) -> DemuxResult:
    n_audio = int(sr * audio_seconds)
    frames = ((i / fps, np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(n_frames))
    return DemuxResult(
        wav=np.zeros(n_audio, dtype=np.float32),
        sample_rate=sr,
        duration_s=n_frames / fps,
        fps=fps,
        frame_iterator=frames,
    )


class _FakeAudioBackend:
    name = "fake_audio"

    def __init__(self, events=None):
        self._events = list(events) if events is not None else []
        self.calls: list[tuple[int, int]] = []

    def transcribe(self, wav, sr, session):
        self.calls.append((wav.size, sr))
        return list(self._events)


class _FakeGuitarBackend:
    name = "fake_guitar"

    def __init__(self, *, return_none: bool = False):
        self._return_none = return_none
        self.calls: list[np.ndarray] = []

    def detect(self, frame):
        self.calls.append(frame)
        if self._return_none:
            return None
        return GuitarBBox(x=0.0, y=0.0, w=10.0, h=10.0, confidence=0.9)


class _FakeFretboardBackend:
    name = "fake_fretboard"

    def __init__(self, *, zero_conf: bool = False):
        self._zero = zero_conf
        self.calls = []

    def detect(self, frame, guitar_box):
        self.calls.append((frame, guitar_box))
        return Homography(
            H=np.eye(3),
            confidence=0.0 if self._zero else 0.9,
            method="fake",
        )


class _FakeHandBackend:
    name = "fake_hand"

    def __init__(self, *, anchor: HandNeckAnchor | None = None):
        self.calls = []
        self.anchor_calls = []
        self.anchor = anchor

    def detect(self, frame, H, cfg):  # noqa: N803 — math name
        self.calls.append((frame, H, cfg))
        return FrameFingering(
            t=0.0,
            finger_pos_logits=np.ones((4, 6, 25), dtype=np.float64),
            homography_confidence=0.9,
        )

    def detect_anchor(self, frame, H, cfg):  # noqa: N803 — math name
        self.anchor_calls.append((frame, H, cfg))
        return self.anchor


# ---------- tests ----------


def test_run_pipeline_audio_only_when_video_disabled(monkeypatch):
    """video_enabled=False bypasses video and goes audio-only."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )
    out = pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=audio,
        video_enabled=False,
    )
    assert len(out) == 1
    assert out[0].pitch_midi == 69
    assert audio.calls, "audio backend should have been invoked"


def test_run_pipeline_invokes_video_backends(monkeypatch):
    """With video enabled, all three video backends are called per sampled frame."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=9))
    audio = _FakeAudioBackend(events=[])
    guitar = _FakeGuitarBackend()
    fretboard = _FakeFretboardBackend()
    hand = _FakeHandBackend()

    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=audio,
        guitar_backend=guitar,
        fretboard_backend=fretboard,
        hand_backend=hand,
        video_stride=3,  # 9 frames / stride 3 = 3 sampled frames
    )

    assert len(guitar.calls) == 3
    assert len(fretboard.calls) == 3
    assert len(hand.calls) == 3


def test_run_pipeline_stride_one_runs_every_frame(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=5))
    guitar = _FakeGuitarBackend()
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        guitar_backend=guitar,
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=_FakeHandBackend(),
        video_stride=1,
    )
    assert len(guitar.calls) == 5


def test_run_pipeline_stride_must_be_positive(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    with pytest.raises(ValueError, match="video_stride"):
        pipeline.run_pipeline(
            "ignored.mp4",
            audio_backend=_FakeAudioBackend(),
            guitar_backend=_FakeGuitarBackend(),
            fretboard_backend=_FakeFretboardBackend(),
            hand_backend=_FakeHandBackend(),
            video_stride=0,
        )


def test_run_pipeline_skips_fretboard_when_no_guitar(monkeypatch):
    """If guitar backend returns None, fretboard + hand backends are skipped."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=3))
    guitar = _FakeGuitarBackend(return_none=True)
    fretboard = _FakeFretboardBackend()
    hand = _FakeHandBackend()
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        guitar_backend=guitar,
        fretboard_backend=fretboard,
        hand_backend=hand,
        video_stride=1,
    )
    assert len(guitar.calls) == 3
    assert len(fretboard.calls) == 0  # skipped — no guitar bbox
    assert len(hand.calls) == 0


def test_run_pipeline_skips_hand_when_no_fretboard(monkeypatch):
    """If fretboard returns zero-confidence H, hand backend is skipped."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=3))
    guitar = _FakeGuitarBackend()
    fretboard = _FakeFretboardBackend(zero_conf=True)
    hand = _FakeHandBackend()
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        guitar_backend=guitar,
        fretboard_backend=fretboard,
        hand_backend=hand,
        video_stride=1,
    )
    assert len(guitar.calls) == 3
    assert len(fretboard.calls) == 3  # called — guitar bbox was present
    assert len(hand.calls) == 0  # skipped — fretboard had zero conf


def test_run_pipeline_propagates_lambda_vision(monkeypatch):
    """lambda_vision arg threads through to fuse."""
    captured: dict = {}

    def fake_fuse(events, fings, cfg, session, *, lambda_vision=1.0):
        captured["lambda_vision"] = lambda_vision
        return []

    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    monkeypatch.setattr(pipeline, "fuse", fake_fuse)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        guitar_backend=_FakeGuitarBackend(),
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=_FakeHandBackend(),
        lambda_vision=2.5,
    )
    assert captured["lambda_vision"] == 2.5


def test_run_pipeline_attaches_neck_anchor_prior_before_fusion(monkeypatch):
    """Coarse hand-neck anchors become ``AudioEvent.fret_prior`` evidence."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=3))
    captured: dict = {}

    def fake_fuse(events, fings, cfg, session, *, lambda_vision=1.0):
        captured["events"] = list(events)
        return []

    monkeypatch.setattr(pipeline, "fuse", fake_fuse)
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )
    hand = _FakeHandBackend(anchor=HandNeckAnchor(10.0, 9.0, 11.0, 0.9))

    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=audio,
        guitar_backend=_FakeGuitarBackend(),
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=hand,
        video_stride=1,
    )

    event = captured["events"][0]
    assert event.fret_prior is not None
    assert int(event.fret_prior.sum(axis=0).argmax()) == 10
    assert len(hand.anchor_calls) == 3


def test_run_pipeline_skips_neck_anchor_prior_when_vision_weight_zero(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    captured: dict = {}

    def fake_fuse(events, fings, cfg, session, *, lambda_vision=1.0):
        captured["events"] = list(events)
        return []

    monkeypatch.setattr(pipeline, "fuse", fake_fuse)
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )

    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=audio,
        guitar_backend=_FakeGuitarBackend(),
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=_FakeHandBackend(anchor=HandNeckAnchor(10.0, 9.0, 11.0, 0.9)),
        lambda_vision=0.0,
    )

    assert captured["events"][0].fret_prior is None


def test_run_pipeline_falls_back_to_audio_only_on_video_import_failure(monkeypatch, caplog):
    """Soft import failure of any video backend → audio-only with a warning."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())

    def boom():
        raise pipeline._VideoImportError("simulated mediapipe missing")

    monkeypatch.setattr(pipeline, "_make_guitar_backend", boom)

    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )
    with caplog.at_level("WARNING", logger="tabvision.pipeline"):
        out = pipeline.run_pipeline("ignored.mp4", audio_backend=audio)

    assert any("falling back to audio-only" in rec.message for rec in caplog.records)
    assert len(out) == 1


def test_run_pipeline_constructs_audio_backend_by_name_when_not_provided(monkeypatch):
    """When audio_backend is None, audio_backend_name flows to the factory."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    captured: dict = {}

    def fake_factory(name):
        captured["name"] = name
        return _FakeAudioBackend()

    monkeypatch.setattr(pipeline, "_make_audio_backend", fake_factory)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend_name="basicpitch",
        guitar_backend=_FakeGuitarBackend(),
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=_FakeHandBackend(),
    )
    assert captured["name"] == "basicpitch"
