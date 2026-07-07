"""A3 sweep harness — the shared-backend fix.

``_raw_events_cached`` must reuse ONE pre-built backend across every clip
rather than rebuilding it per call: rebuilding reloads the ~0.5 GB highres
checkpoint per clip and accumulates memory (a fresh torchlibrosa STFT init)
until a 60+-clip run OOMs partway through. This test only checks the *reuse*
contract with a fake backend/demux (no torch/highres needed): the same
backend instance's ``.transcribe`` is called on every cache miss, and never
rebuilt.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.eval.a3_fusion_sweep import _raw_events_cached
from tabvision.types import AudioEvent, DemuxResult, SessionConfig


class _FakeBackend:
    def __init__(self) -> None:
        self.transcribe_calls = 0

    def transcribe(self, wav: object, sample_rate: int, session: SessionConfig) -> list[AudioEvent]:
        self.transcribe_calls += 1
        return [AudioEvent(onset_s=0.0, offset_s=0.2, pitch_midi=40, velocity=0.8, confidence=0.8)]


def _fake_demux(_path: str) -> DemuxResult:
    return DemuxResult(wav=b"", sample_rate=22050, duration_s=1.0, fps=30.0)


def test_raw_events_cached_reuses_the_passed_backend_instance(tmp_path: Path) -> None:
    """Two different clips (cache misses) must call .transcribe on the SAME
    backend object -- never construct a second one internally."""
    backend = _FakeBackend()
    cache_dir = tmp_path / "cache"
    clip_a = tmp_path / "a.wav"
    clip_b = tmp_path / "b.wav"
    clip_a.write_bytes(b"x")
    clip_b.write_bytes(b"y")

    with patch("tabvision.demux.demux", side_effect=_fake_demux):
        _raw_events_cached(clip_a, SessionConfig(), "highres", cache_dir, backend)
        _raw_events_cached(clip_b, SessionConfig(), "highres", cache_dir, backend)

    assert backend.transcribe_calls == 2  # one per clip, same instance both times


def test_raw_events_cached_skips_transcribe_on_cache_hit(tmp_path: Path) -> None:
    """A second call for the SAME clip must hit the cache, not re-transcribe."""
    backend = _FakeBackend()
    cache_dir = tmp_path / "cache"
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x")

    with patch("tabvision.demux.demux", side_effect=_fake_demux):
        first = _raw_events_cached(clip, SessionConfig(), "highres", cache_dir, backend)
        second = _raw_events_cached(clip, SessionConfig(), "highres", cache_dir, backend)

    assert backend.transcribe_calls == 1  # second call was a cache hit
    assert [e.pitch_midi for e in first] == [e.pitch_midi for e in second]


def test_raw_events_cached_requires_backend_argument() -> None:
    """The signature has no default for ``backend`` -- callers must build one
    via make_shared_audio_backend() and pass it explicitly (no silent
    per-clip fallback that would reintroduce the OOM)."""
    import inspect

    sig = inspect.signature(_raw_events_cached)
    assert "backend" in sig.parameters
    assert sig.parameters["backend"].default is inspect.Parameter.empty


@pytest.mark.parametrize("backend_name", ["highres"])
def test_make_shared_audio_backend_dispatches_to_registry(backend_name: str) -> None:
    """make_shared_audio_backend is a thin wrapper over the backend registry
    (not a from-scratch construction) -- this pins that contract without
    importing the heavy highres extras."""
    from scripts.eval import a3_fusion_sweep

    with patch("tabvision.audio.backend.make") as mock_make:
        mock_make.return_value = "sentinel-backend"
        result = a3_fusion_sweep.make_shared_audio_backend(backend_name)

    mock_make.assert_called_once_with(backend_name)
    assert result == "sentinel-backend"
