"""GuitarSet JAMS annotation parser.

Wraps the existing :func:`tabvision.eval.guitarset_audio.parse_guitarset_jams`
under the uniform parser interface so composite-eval dispatch can route
``annotation_format = "guitarset_jams"`` clips here.
"""

from __future__ import annotations

from tabvision.eval.guitarset_audio import parse_guitarset_jams as parse
from tabvision.eval.parsers.registry import register_parser

FORMAT_NAME = "guitarset_jams"

register_parser(FORMAT_NAME, parse)


__all__ = ["FORMAT_NAME", "parse"]
