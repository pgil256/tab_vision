"""Annotation parsers — uniform interface for source-specific tab labels.

Each parser module exposes:

- ``FORMAT_NAME``: the string key that appears in
  ``Manifest.clip.annotation_format`` (added in Phase 0 to support
  multi-source composite eval).
- ``parse(annotation_path, cfg) -> list[TabEvent]``: pure function;
  no I/O outside the file at ``annotation_path``.

Submodule imports below trigger registration in
:mod:`tabvision.eval.parsers.registry`.
"""

# Built-in parsers — importing them registers their FORMAT_NAME.
from tabvision.eval.parsers import (  # noqa: F401
    guitar_techs_midi,
    guitarset_jams,
    utaustin_tablature_npy,
)
from tabvision.eval.parsers.registry import (
    ParserFn,
    clear_parsers,
    get_parser,
    list_parsers,
    register_parser,
)

__all__ = [
    "ParserFn",
    "clear_parsers",
    "get_parser",
    "list_parsers",
    "register_parser",
]
