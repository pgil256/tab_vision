"""``HighResBackend``'s onset/offset/frame threshold kwargs must actually reach
the underlying transcriptor.

``hf_midi_transcription.MidiTranscriptionModel.__init__`` accepts
``onset_threshold``/``offset_threshold``/``frame_threshold`` but drops them when
building its internal ``piano_transcription_inference.PianoTranscription`` (only
``instrument`` is forwarded to ``_init_transcriptor``) — confirmed by an eval
probe (2026-06-30) where changing ``HighResBackend(onset_threshold=...)`` /
``frame_threshold=...`` produced bit-identical Tab F1 and error-decomposition
output to the defaults. ``PianoTranscription`` does read these as plain
instance attributes fresh on every ``transcribe()`` call, so ``_load_model``
sets them directly on ``self._model.transcriptor`` after construction. These
tests use a fake ``MidiTranscriptionModel`` that reproduces the real library's
broken constructor wiring (stores kwargs but doesn't forward them to its
``transcriptor``), so they fail if the post-construction patch is removed.
"""

from __future__ import annotations

import sys
import types

import pytest

from tabvision.audio.highres import HighResBackend


class _FakeTranscriptor:
    """Mimics piano_transcription_inference.PianoTranscription's real
    (functional) attributes — defaults match the upstream library exactly."""

    def __init__(self) -> None:
        self.onset_threshold = 0.3
        self.offset_threshod = 0.3  # upstream typo, not ours
        self.frame_threshold = 0.1


class _FakeMidiTranscriptionModel:
    """Mimics the real hf_midi_transcription bug: constructor kwargs are
    accepted and ignored — self.transcriptor always gets the library's
    own hard-coded defaults, never the values passed in here."""

    def __init__(self, **kwargs: object) -> None:
        self.config = dict(kwargs)
        self.transcriptor = _FakeTranscriptor()


@pytest.fixture
def fake_hf_midi_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("hf_midi_transcription")
    fake_module.MidiTranscriptionModel = _FakeMidiTranscriptionModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hf_midi_transcription", fake_module)


def test_custom_onset_threshold_reaches_transcriptor(fake_hf_midi_transcription: None) -> None:
    backend = HighResBackend(onset_threshold=0.15)
    model = backend._load_model()
    assert model.transcriptor.onset_threshold == 0.15


def test_custom_frame_threshold_reaches_transcriptor(fake_hf_midi_transcription: None) -> None:
    backend = HighResBackend(frame_threshold=0.25)
    model = backend._load_model()
    assert model.transcriptor.frame_threshold == 0.25


def test_custom_offset_threshold_reaches_transcriptor(fake_hf_midi_transcription: None) -> None:
    backend = HighResBackend(offset_threshold=0.45)
    model = backend._load_model()
    # NB: upstream PianoTranscription's own attribute name has a typo
    # ("offset_threshod") — that's the real attribute RegressionPostProcessor
    # reads, so this is what must be set, not "offset_threshold".
    assert model.transcriptor.offset_threshod == 0.45


def test_default_thresholds_match_library_defaults(fake_hf_midi_transcription: None) -> None:
    """Sanity check: HighResBackend's own defaults equal the (buggy) library's
    hard-coded defaults, which is why this bug has been behaviorally invisible
    in production so far — the patch is currently a no-op for default usage."""
    backend = HighResBackend()
    model = backend._load_model()
    assert model.transcriptor.onset_threshold == backend.onset_threshold == 0.3
    assert model.transcriptor.offset_threshod == backend.offset_threshold == 0.3
    assert model.transcriptor.frame_threshold == backend.frame_threshold == 0.1
