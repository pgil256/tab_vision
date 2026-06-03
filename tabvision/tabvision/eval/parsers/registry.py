"""Annotation-parser registry.

Each annotation source (GuitarSet JAMS, Guitar-TECHS 6-track MIDI, EGDB
GuitarPro, etc.) gets a parser module that registers itself here on
import. Composite-eval dispatch then routes by
``Manifest.clip.annotation_format`` to the registered parser.

This file is import-side-effect free: the registry is empty at first
import. Built-in parsers are registered by ``parsers/__init__.py``
importing their submodules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tabvision.types import GuitarConfig, TabEvent

ParserFn = Callable[[str | Path, GuitarConfig | None], list[TabEvent]]
"""``(annotation_path, cfg) -> list[TabEvent]``. ``cfg`` may be ``None``."""


_PARSERS: dict[str, ParserFn] = {}


def register_parser(format_name: str, fn: ParserFn) -> None:
    """Register ``fn`` as the parser for ``format_name``.

    Raises ``ValueError`` if ``format_name`` is already registered.
    """
    if format_name in _PARSERS:
        raise ValueError(
            f"Parser already registered for format {format_name!r}; "
            f"call clear_parsers() first if this is intentional."
        )
    _PARSERS[format_name] = fn


def get_parser(format_name: str) -> ParserFn:
    """Look up the parser for ``format_name``.

    Raises ``KeyError`` with the list of known formats if not registered.
    """
    if format_name not in _PARSERS:
        known = ", ".join(sorted(_PARSERS)) or "(none registered)"
        raise KeyError(f"Unknown annotation format: {format_name!r}. Known: {known}.")
    return _PARSERS[format_name]


def list_parsers() -> list[str]:
    """Return the sorted list of registered format names."""
    return sorted(_PARSERS)


def clear_parsers() -> None:
    """Remove all registered parsers. For tests only."""
    _PARSERS.clear()


__all__ = [
    "ParserFn",
    "clear_parsers",
    "get_parser",
    "list_parsers",
    "register_parser",
]
