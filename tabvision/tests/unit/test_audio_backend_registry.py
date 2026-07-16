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
    assert "highres-ensemble" in names


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


def test_highres_ensemble_factory_loads_registered_artifact_without_weights():
    b = make("highres-ensemble")

    assert b.name == "highres-ensemble"
    assert b.artifact.threshold == 0.5  # type: ignore[attr-defined]


def test_highres_rejects_unknown_checkpoint():
    from tabvision.audio.highres import HighResBackend

    with pytest.raises(InvalidInputError):
        HighResBackend(checkpoint="banjo")


def test_make_forwards_filter_config_to_backend():
    """``make(name, filter_config=...)`` reaches the backend's filter wiring.

    basicpitch defaults filters on; ``filter_config=False`` disables them.
    highres defaults filters off; ``filter_config=True`` enables the
    conservative config. This is the CLI/pipeline → backend plumbing path.
    """
    bp_default = make("basicpitch")
    assert bp_default.filter_config is not None  # type: ignore[attr-defined]
    bp_off = make("basicpitch", filter_config=False)
    assert bp_off.filter_config is None  # type: ignore[attr-defined]

    hr_default = make("highres")
    assert hr_default.filter_config is None  # type: ignore[attr-defined]
    hr_on = make("highres", filter_config=True)
    assert hr_on.filter_config is not None  # type: ignore[attr-defined]


def test_make_forwards_explicit_filter_config_instance():
    from tabvision.audio.filters import AudioFilterConfig

    cfg = AudioFilterConfig(min_confidence=0.42)
    b = make("highres", filter_config=cfg)
    assert b.filter_config is cfg  # type: ignore[attr-defined]
