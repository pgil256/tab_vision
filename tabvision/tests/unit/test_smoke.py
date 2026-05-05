"""Phase 0 smoke test — verifies the package imports and CI is wired.

Replaced with real unit tests in Phase 1+.
"""

import tabvision
from tabvision import errors, types


def test_version_string():
    assert isinstance(tabvision.__version__, str)
    assert len(tabvision.__version__) > 0


def test_errors_hierarchy():
    assert issubclass(errors.InvalidInputError, errors.TabVisionError)
    assert issubclass(errors.BackendError, errors.TabVisionError)
    assert issubclass(errors.FusionError, errors.TabVisionError)


def test_default_tuning_is_standard():
    # Low E to high E in MIDI: 40, 45, 50, 55, 59, 64.
    assert types.DEFAULT_TUNING_MIDI == (40, 45, 50, 55, 59, 64)


def test_guitar_config_defaults():
    cfg = types.GuitarConfig()
    assert cfg.capo == 0
    assert cfg.n_strings == 6
    assert cfg.max_fret == 24


def test_session_config_defaults():
    cfg = types.SessionConfig()
    assert cfg.instrument == "acoustic"
    assert cfg.tone == "clean"
    assert cfg.style == "mixed"
