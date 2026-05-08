"""AudioBackend protocol + registry — see SPEC.md §8 and §7 Phase 2.

Backends register a factory function under a string name. The CLI uses
the registry to map ``--audio-backend NAME`` to the right class without
hard-importing every backend up front (some have heavy deps like
TensorFlow or PyTorch).
"""

from __future__ import annotations

from typing import Callable

from tabvision.errors import InvalidInputError
from tabvision.types import AudioBackend

# Registry: name → factory(**kwargs) → AudioBackend.
# Factories are deferred-import to avoid pulling all backend deps at startup.
_REGISTRY: dict[str, Callable[..., AudioBackend]] = {}


def register(name: str, factory: Callable[..., AudioBackend]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"audio backend already registered: {name!r}")
    _REGISTRY[name] = factory


def make(name: str, **kwargs) -> AudioBackend:  # type: ignore[no-untyped-def]
    if name not in _REGISTRY:
        raise InvalidInputError(
            f"unknown audio backend: {name!r}; "
            f"available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def available_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


# ----- Built-in registrations (deferred-import factories) -----


def _basicpitch_factory(**kwargs):  # type: ignore[no-untyped-def]
    from tabvision.audio.basicpitch import BasicPitchBackend

    return BasicPitchBackend(**kwargs)


def _highres_factory(**kwargs):  # type: ignore[no-untyped-def]
    from tabvision.audio.highres import HighResBackend

    return HighResBackend(**kwargs)


def _highres_fl_factory(**kwargs):  # type: ignore[no-untyped-def]
    from tabvision.audio.highres import HighResBackend

    kwargs.setdefault("checkpoint", "guitar_fl")
    return HighResBackend(**kwargs)


register("basicpitch", _basicpitch_factory)
register("highres", _highres_factory)
register("highres-fl", _highres_fl_factory)


__all__ = ["AudioBackend", "register", "make", "available_backends"]
