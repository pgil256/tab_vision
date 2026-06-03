"""Tone-routing toggle: the session's declared instrument selects the backbone.

Acoustic/classical → acoustic ``highres``; electric → the separately fine-tuned
``highres-electric`` (a v2 checkpoint). Until that checkpoint is configured,
selecting it must fail fast with a clear, actionable message.

Runnable two ways:
  - ``pytest tabvision/tests/unit/test_audio_routing.py``
  - ``python tabvision/tests/unit/test_audio_routing.py``  (no pytest dep)
"""

from __future__ import annotations

from tabvision.audio.backend import make
from tabvision.errors import BackendError
from tabvision.pipeline import audio_backend_for_session
from tabvision.types import SessionConfig


def test_routes_electric_to_electric_backend() -> None:
    assert (
        audio_backend_for_session(SessionConfig(instrument="electric"))
        == "highres-electric"
    )


def test_routes_acoustic_and_classical_to_highres() -> None:
    assert audio_backend_for_session(SessionConfig(instrument="acoustic")) == "highres"
    assert audio_backend_for_session(SessionConfig(instrument="classical")) == "highres"


def _assert_electric_guard() -> None:
    import os

    os.environ.pop("TABVISION_HIGHRES_ELECTRIC_CKPT", None)
    backend = make("highres-electric")
    try:
        backend._load_model()
    except BackendError as exc:
        assert "not trained yet" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("expected BackendError for unconfigured electric backbone")


def test_electric_backend_guard_without_checkpoint() -> None:
    _assert_electric_guard()


if __name__ == "__main__":
    test_routes_electric_to_electric_backend()
    test_routes_acoustic_and_classical_to_highres()
    _assert_electric_guard()
    print("PASS: audio routing + electric guard")
