"""ASCII tab renderer — Phase 1 deliverable.

Produces a 6-line standard tab. High E on top (``string_idx=5``), low E on
bottom (``string_idx=0``). Low-confidence notes (< 0.5) are marked with
``?`` after the fret number.

This is the simplest possible renderer: one column per event in onset
order, no fixed time-grid. Phase 9 polish adds proper rhythm rendering.
"""

from __future__ import annotations

from typing import Sequence

from tabvision.types import GuitarConfig, TabEvent

LOW_CONFIDENCE_THRESHOLD = 0.5
ROW_WIDTH = 80
STRING_NAMES = ("E", "A", "D", "G", "B", "e")  # 0=low E .. 5=high E
DASH = "-"
BAR = "|"


def render(
    events: Sequence[TabEvent],
    cfg: GuitarConfig | None = None,
) -> str:
    """Render TabEvents to ASCII tab.

    Args:
        events: TabEvents in onset order. (Re-sorted defensively.)
        cfg: Guitar configuration; affects the header.

    Returns:
        ASCII tab as a single string with newline separators.
    """
    if cfg is None:
        cfg = GuitarConfig()

    sorted_events = sorted(events, key=lambda e: e.onset_s)
    columns = [_event_column(ev) for ev in sorted_events]

    body = _columns_to_lines(columns, cfg.n_strings)
    header = _header(cfg, n_events=len(sorted_events))
    return header + "\n" + body


def _event_column(ev: TabEvent) -> tuple[int, str]:
    """Single-event column: returns (string_idx, cell_text)."""
    cell = str(ev.fret)
    if ev.confidence < LOW_CONFIDENCE_THRESHOLD:
        cell += "?"
    return ev.string_idx, cell


def _columns_to_lines(columns: list[tuple[int, str]], n_strings: int) -> str:
    """Stack columns into 6 string-lines, wrapping at ROW_WIDTH."""
    if not columns:
        return _empty_tab(n_strings)

    # Compute per-column width (uniform = max cell + 1 dash padding).
    cell_width = max(len(c) for _, c in columns) + 1

    lines: list[str] = []
    for s in reversed(range(n_strings)):
        line = STRING_NAMES[s] + BAR
        for string_idx, cell in columns:
            if string_idx == s:
                line += cell + DASH * (cell_width - len(cell))
            else:
                line += DASH * cell_width
        line += BAR
        lines.append(line)

    return _wrap_rows(lines, ROW_WIDTH)


def _empty_tab(n_strings: int) -> str:
    return "\n".join(
        f"{STRING_NAMES[s]}{BAR}{DASH * 16}{BAR}"
        for s in reversed(range(n_strings))
    )


def _wrap_rows(lines: list[str], width: int) -> str:
    """Wrap parallel lines to ``width`` columns, preserving alignment."""
    if not lines:
        return ""

    line_len = len(lines[0])
    if line_len <= width:
        return "\n".join(lines)

    chunks: list[str] = []
    pos = 2  # Skip the leading "X|" prefix.
    while pos < line_len:
        end = min(pos + width - 4, line_len)
        for s_idx, line in enumerate(lines):
            prefix = line[:2]
            chunks.append(prefix + line[pos:end] + BAR)
            if s_idx < len(lines) - 1:
                chunks.append("")  # group separator handled below
        chunks.append("")  # blank line between row groups
        pos = end

    # Re-stitch into a single block with grouped lines.
    return _stitch_groups(chunks, n_strings=len(lines))


def _stitch_groups(chunks: list[str], n_strings: int) -> str:
    """Rebuild text from per-group chunks emitted by ``_wrap_rows``."""
    out: list[str] = []
    n_per_group = n_strings + 1  # 6 strings + separator
    for i, chunk in enumerate(chunks):
        if i % n_per_group < n_strings:
            out.append(chunk)
        elif chunk == "" and out and out[-1] != "":
            out.append("")
    return "\n".join(line for line in out if line is not None).rstrip() + "\n"


def _header(cfg: GuitarConfig, n_events: int) -> str:
    # Tuning conventionally written low-to-high.
    tuning_names = " ".join(STRING_NAMES[i].upper() for i in range(cfg.n_strings))
    capo = f"Capo: {cfg.capo}" if cfg.capo > 0 else "Capo: none"
    return (
        "TabVision ASCII tab\n"
        f"Tuning: {tuning_names}   {capo}   Notes: {n_events}\n"
        "Low-confidence notes marked with '?'.\n"
    )


__all__ = ["render"]
