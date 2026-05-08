"""Unit tests for the audio backend registry."""

import pytest

from tabvision.audio import backend as backend_module
from tabvision.audio.backend import available_backends, make
from tabvision.errors import InvalidInputError


def test_available_backends_includes_phase_1_and_2():
    names = available_backends()
    assert "basicpitch" in names
    assert "highres" in names
    assert "highres-fl" in names


def test_make_unknown_backend_raises():
    with pytest.raises(InvalidInputError):
        make("does-not-exist")


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):
        backend_module.register("basicpitch", lambda **k: None)  # type: ignore[arg-type]


def test_make_basicpitch_does_not_load_model():
    """Construction should be cheap; model loading is lazy."""
    b = make("basicpitch")
    assert b.name == "basicpitch"


def test_make_highres_does_not_download_weights():
    """Construction should not hit the network — only ``transcribe`` does."""
    b = make("highres")
    assert b.name == "highres"
    # No model loaded yet.
    assert getattr(b, "_model", None) is None


def test_highres_fl_factory_uses_fl_checkpoint():
    b = make("highres-fl")
    assert b.checkpoint == "guitar_fl"  # type: ignore[attr-defined]


def test_highres_rejects_unknown_checkpoint():
    from tabvision.audio.highres import HighResBackend

    with pytest.raises(InvalidInputError):
        HighResBackend(checkpoint="banjo")
