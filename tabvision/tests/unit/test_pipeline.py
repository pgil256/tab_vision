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


def test_run_pipeline_reports_stages_audio_only(monkeypatch):
    """progress_callback sees demux → model_load → audio_inference → decode
    (no video_analysis when the video stack is disabled)."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    stages: list[str] = []
    out = pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(
            events=[
                AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)
            ]
        ),
        video_enabled=False,
        progress_callback=stages.append,
    )
    assert stages == ["demux", "model_load", "audio_inference", "decode"]
    assert len(out) == 1


def test_run_pipeline_reports_video_stage_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=3))
    stages: list[str] = []
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        guitar_backend=_FakeGuitarBackend(),
        fretboard_backend=_FakeFretboardBackend(),
        hand_backend=_FakeHandBackend(),
        video_stride=1,
        progress_callback=stages.append,
    )
    assert stages == ["demux", "model_load", "audio_inference", "video_analysis", "decode"]


def test_run_pipeline_progress_callback_errors_are_swallowed(monkeypatch):
    """A broken progress callback must never break the transcription itself."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())

    def broken(_stage: str) -> None:
        raise RuntimeError("progress sink went away")

    out = pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(
            events=[
                AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)
            ]
        ),
        video_enabled=False,
        progress_callback=broken,
    )
    assert len(out) == 1


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


def test_run_pipeline_default_does_not_attach_pitch_position_prior(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    captured: dict = {}

    def fake_fuse(events, fings, cfg, session, *, lambda_vision=1.0):
        captured["events"] = list(events)
        return []

    monkeypatch.setattr(pipeline, "fuse", fake_fuse)
    monkeypatch.setattr(
        pipeline,
        "load_pitch_position_prior",
        lambda _name, *, cfg=None: pytest.fail("position prior should be explicit"),
        raising=False,
    )
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )

    pipeline.run_pipeline("ignored.mp4", audio_backend=audio, video_enabled=False)

    assert captured["events"][0].fret_prior is None


def test_run_pipeline_attaches_named_pitch_position_prior_when_explicit(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    captured: dict = {}
    prior_matrix = np.ones((6, 25), dtype=np.float64) / 150.0

    def fake_fuse(events, fings, cfg, session, *, lambda_vision=1.0):
        captured["events"] = list(events)
        return []

    class _FakePrior:
        def matrix_for_pitch(self, pitch_midi):
            return prior_matrix if pitch_midi == 69 else None

    monkeypatch.setattr(pipeline, "fuse", fake_fuse)
    monkeypatch.setattr(
        pipeline,
        "load_pitch_position_prior",
        lambda name, *, cfg=None: _FakePrior(),
        raising=False,
    )
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )

    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=audio,
        video_enabled=False,
        position_prior="guitarset-v1",
    )

    assert captured["events"][0].fret_prior is prior_matrix


# ---------- sequence prior (A15) coupling ----------


@pytest.fixture
def seq_prior_env(monkeypatch):
    """Clear the sweep env knobs and record set/load calls."""
    monkeypatch.delenv("TABVISION_TRANSITION_PRIOR", raising=False)
    monkeypatch.delenv("TABVISION_TRANSITION_PRIOR_WEIGHT", raising=False)
    calls: dict = {"sentinel": object()}

    def fake_load(name):
        calls["loaded"] = name
        return calls["sentinel"]

    def fake_set(prior, weight=None):
        calls.update(prior=prior, weight=weight, set_called=True)

    monkeypatch.setattr(pipeline, "load_transition_prior", fake_load)
    monkeypatch.setattr(pipeline, "set_transition_prior", fake_set)
    return calls


def test_install_sequence_prior_auto_couples_on_when_position_prior_active(seq_prior_env):
    pipeline._install_sequence_prior("auto", position_prior_active=True)
    assert seq_prior_env["loaded"] == pipeline.SEQUENCE_PRIOR_DEFAULT
    assert seq_prior_env["prior"] is seq_prior_env["sentinel"]
    assert seq_prior_env["weight"] == pipeline.SEQUENCE_PRIOR_WEIGHT


def test_install_sequence_prior_auto_off_when_position_prior_off(seq_prior_env):
    """The coupling is load-bearing: uncoupled seq prior is a banked GAPS
    regression (0.647→0.593, DECISIONS.md 2026-07-02). 'auto' without the
    position prior must clear any previously installed prior."""
    pipeline._install_sequence_prior("auto", position_prior_active=False)
    assert seq_prior_env["set_called"] is True
    assert seq_prior_env["prior"] is None
    assert "loaded" not in seq_prior_env


def test_install_sequence_prior_none_clears_even_when_position_prior_active(seq_prior_env):
    pipeline._install_sequence_prior("none", position_prior_active=True)
    assert seq_prior_env["prior"] is None


def test_install_sequence_prior_explicit_name_overrides_coupling(seq_prior_env):
    """An explicit artifact name is a deliberate uncoupled run (ablations)."""
    pipeline._install_sequence_prior("guitarset-seq-v1", position_prior_active=False)
    assert seq_prior_env["loaded"] == "guitarset-seq-v1"
    assert seq_prior_env["prior"] is seq_prior_env["sentinel"]


def test_install_sequence_prior_env_var_wins_over_argument(seq_prior_env, monkeypatch):
    """TABVISION_TRANSITION_PRIOR is the sweep knob — when set, the
    pipeline must not clobber playability's env-driven lazy install."""
    monkeypatch.setenv("TABVISION_TRANSITION_PRIOR", "guitarset-seq-v1")
    pipeline._install_sequence_prior("auto", position_prior_active=True)
    assert "set_called" not in seq_prior_env


def test_install_sequence_prior_env_weight_respected(seq_prior_env, monkeypatch):
    """An explicit weight env var keeps playability's env-derived weight
    instead of the gate-accepted default."""
    monkeypatch.setenv("TABVISION_TRANSITION_PRIOR_WEIGHT", "2.0")
    pipeline._install_sequence_prior("auto", position_prior_active=True)
    assert seq_prior_env["weight"] is None


def test_run_pipeline_default_installs_sequence_prior_with_position_prior(
    seq_prior_env, monkeypatch
):
    """Default-on flip: the accepted config (position prior on) carries the
    sequence prior at the gate-accepted weight without any flag."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    monkeypatch.setattr(
        pipeline, "load_pitch_position_prior", lambda name, *, cfg=None: _NoopPositionPrior()
    )
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )

    pipeline.run_pipeline(
        "ignored.mp4", audio_backend=audio, video_enabled=False, position_prior="guitarset-v1"
    )

    assert seq_prior_env["loaded"] == pipeline.SEQUENCE_PRIOR_DEFAULT
    assert seq_prior_env["weight"] == pipeline.SEQUENCE_PRIOR_WEIGHT


def test_run_pipeline_no_position_prior_keeps_sequence_prior_off(seq_prior_env, monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    audio = _FakeAudioBackend(
        events=[AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)]
    )

    pipeline.run_pipeline("ignored.mp4", audio_backend=audio, video_enabled=False)

    assert seq_prior_env["prior"] is None
    assert "loaded" not in seq_prior_env


class _NoopPositionPrior:
    def matrix_for_pitch(self, pitch_midi):
        return None


def test_run_pipeline_keeps_melodic_prior_disabled_by_default(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))

    def fail_if_called(events, cfg):
        pytest.fail("melodic prior should be explicit")

    monkeypatch.setattr(pipeline, "apply_melodic_segment_prior", fail_if_called)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        video_enabled=False,
    )


def test_run_pipeline_can_attach_melodic_prior_when_explicit(monkeypatch):
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result(n_frames=1))
    captured = {}

    def fake_prior(events, cfg):
        captured["called"] = True
        return events

    monkeypatch.setattr(pipeline, "apply_melodic_segment_prior", fake_prior)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(),
        video_enabled=False,
        melodic_prior_enabled=True,
    )

    assert captured["called"] is True


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

    def fake_factory(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
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


def test_run_pipeline_keeps_backend_default_filters_when_audio_filters_none(monkeypatch):
    """audio_filters=None (default) ⇒ no filter_config kwarg ⇒ backend default kept."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    captured: dict = {}

    def fake_factory(name, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeAudioBackend()

    monkeypatch.setattr(pipeline, "_make_audio_backend", fake_factory)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend_name="highres",
        video_enabled=False,
    )
    assert captured["kwargs"] == {}


@pytest.mark.parametrize("value", [True, False])
def test_run_pipeline_forwards_audio_filters_override_to_factory(monkeypatch, value):
    """A bool audio_filters override is forwarded as filter_config to the factory."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())
    captured: dict = {}

    def fake_factory(name, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeAudioBackend()

    monkeypatch.setattr(pipeline, "_make_audio_backend", fake_factory)
    pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend_name="highres",
        video_enabled=False,
        audio_filters=value,
    )
    assert captured["kwargs"] == {"filter_config": value}


def test_run_pipeline_ignores_audio_filters_when_backend_injected(monkeypatch):
    """An injected backend carries its own filter config; audio_filters is ignored."""
    monkeypatch.setattr(pipeline, "demux", lambda _p: _make_demux_result())

    def fail_if_called(name, **kwargs):  # pragma: no cover - must not run
        pytest.fail("factory should not be called when a backend is injected")

    monkeypatch.setattr(pipeline, "_make_audio_backend", fail_if_called)
    out = pipeline.run_pipeline(
        "ignored.mp4",
        audio_backend=_FakeAudioBackend(
            events=[
                AudioEvent(onset_s=0.0, offset_s=0.25, pitch_midi=69, velocity=0.8, confidence=0.8)
            ]
        ),
        video_enabled=False,
        audio_filters=True,
    )
    assert len(out) == 1
