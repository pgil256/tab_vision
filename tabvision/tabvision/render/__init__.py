"""Render — see SPEC.md §3.3, §8.

Public entrypoint: ``render(events, fmt, cfg) -> bytes``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from tabvision.types import GuitarConfig, TabEvent

RenderFormat = Literal["ascii", "gp5", "musicxml", "midi"]


def render(events: Sequence[TabEvent], fmt: RenderFormat, cfg: GuitarConfig) -> bytes:
    """Render tab events to the requested output format."""
    if fmt == "ascii":
        from tabvision.render.ascii import render as render_ascii

        return render_ascii(events, cfg).encode("utf-8")
    if fmt == "gp5":
        from tabvision.render.gp5 import render as render_gp5

        return render_gp5(events, cfg)
    if fmt == "musicxml":
        from tabvision.render.musicxml import render as render_musicxml

        return render_musicxml(events, cfg)
    if fmt == "midi":
        from tabvision.render.midi import render as render_midi

        return render_midi(events, cfg)
    raise ValueError(f"unsupported render format: {fmt}")


__all__ = ["RenderFormat", "render"]
