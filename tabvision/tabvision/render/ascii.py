"""ASCII tab renderer — Phase 1 deliverable.

Produces a 6-line standard tab. High E on top (``string_idx=5``), low E on
bottom (``string_idx=0``). Low-confidence notes (< 0.5) are marked with
``?`` after the fret number.

This is the simplest possible renderer: one column per event in onset
order, no fixed time-grid.

Phase 9 polish adds an opt-in ``color`` mode (SPEC §7.3, "color-graded if
terminal supports"): each fret is wrapped in an ANSI colour by confidence
band — green (high) / yellow (medium) / red (low). Colour is strictly
additive: ``color=False`` (the default, and every file/dispatch path) emits
byte-identical plain output, so the ``?`` marker remains the dependency-free
signal.
"""

from __future__ import annotations

from collections.abc import Sequence

from tabvision.types import GuitarConfig, TabEvent

LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
ROW_WIDTH = 80
STRING_NAMES = ("E", "A", "D", "G", "B", "e")  # 0=low E .. 5=high E
DASH = "-"
BAR = "|"

# ANSI SGR colour codes for the opt-in confidence gradient.
_ANSI_RESET = "\x1b[0m"
_ANSI_HIGH = "\x1b[32m"  # green  — confidence >= HIGH_CONFIDENCE_THRESHOLD
_ANSI_MEDIUM = "\x1b[33m"  # yellow — LOW <= confidence < HIGH
_ANSI_LOW = "\x1b[31m"  # red    — confidence < LOW_CONFIDENCE_THRESHOLD


def render(
    events: Sequence[TabEvent],
    cfg: GuitarConfig | None = None,
    *,
    color: bool = False,
) -> str:
    """Render TabEvents to ASCII tab.

    Args:
        events: TabEvents in onset order. (Re-sorted defensively.)
        cfg: Guitar configuration; affects the header.
        color: When ``True``, wrap each fret in an ANSI confidence colour
            (green/yellow/red). Off by default so file output and the
            format dispatch stay plain and byte-stable. The caller (CLI)
            enables it only for an interactive TTY — see
            ``tabvision.cli._should_color``.

    Returns:
        ASCII tab as a single string with newline separators.
    """
    if cfg is None:
        cfg = GuitarConfig()

    sorted_events = sorted(events, key=lambda e: e.onset_s)
    for ev in sorted_events:
        _validate_event(ev, cfg)

    if color:
        body = _columns_to_lines_color(sorted_events, cfg.n_strings)
    else:
        columns = [_event_column(ev) for ev in sorted_events]
        body = _columns_to_lines(columns, cfg.n_strings)
    header = _header(cfg, n_events=len(sorted_events), color=color)
    return header + "\n" + body


def _validate_event(event: TabEvent, cfg: GuitarConfig) -> None:
    """Mirror the range checks in gp5/midi/musicxml (see those renderers'
    ``_validate_event``/``_validate_string``). Without this, a malformed
    ``string_idx`` (e.g. from an upstream fusion bug) would silently fail to
    match any row in ``_columns_to_lines`` and the note would vanish from the
    rendered tab instead of erroring — the worst outcome for the default
    output format.
    """
    if not 0 <= event.string_idx < cfg.n_strings:
        raise ValueError(f"string_idx out of range: {event.string_idx}")
    if not 0 <= event.fret <= cfg.max_fret:
        raise ValueError(f"fret out of range: {event.fret}")


def _event_column(ev: TabEvent) -> tuple[int, str]:
    """Single-event column: returns (string_idx, cell_text)."""
    cell = str(ev.fret)
    if ev.confidence < LOW_CONFIDENCE_THRESHOLD:
        cell += "?"
    return ev.string_idx, cell


def _confidence_color(confidence: float) -> str:
    """ANSI colour for a confidence band (green/yellow/red)."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return _ANSI_HIGH
    if confidence >= LOW_CONFIDENCE_THRESHOLD:
        return _ANSI_MEDIUM
    return _ANSI_LOW


def _columns_to_lines_color(events: Sequence[TabEvent], n_strings: int) -> str:
    """Colour-graded variant of ``_columns_to_lines``.

    Uses column-based wrapping — it never character-slices an assembled line,
    so ANSI escapes can be injected without corrupting alignment. (The plain
    path's ``_wrap_rows`` slices by character offset and is *not* colour-safe,
    which is why colour has its own layout pass rather than a post-filter.)
    Cell widths are computed from the visible fret text only.
    """
    if not events:
        return _empty_tab(n_strings)

    plain_cells: list[str] = []
    for ev in events:
        cell = str(ev.fret)
        if ev.confidence < LOW_CONFIDENCE_THRESHOLD:
            cell += "?"
        plain_cells.append(cell)
    cell_width = max(len(c) for c in plain_cells) + 1

    usable = ROW_WIDTH - 3  # leading "e|" + trailing "|"
    cols_per_row = max(1, usable // cell_width)

    blocks: list[str] = []
    for start in range(0, len(events), cols_per_row):
        group = list(
            zip(
                events[start : start + cols_per_row],
                plain_cells[start : start + cols_per_row],
                strict=True,
            )
        )
        lines: list[str] = []
        for s in reversed(range(n_strings)):
            line = STRING_NAMES[s] + BAR
            for ev, cell in group:
                if ev.string_idx == s:
                    colored = _confidence_color(ev.confidence) + cell + _ANSI_RESET
                    line += colored + DASH * (cell_width - len(cell))
                else:
                    line += DASH * cell_width
            line += BAR
            lines.append(line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


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
    return "\n".join(f"{STRING_NAMES[s]}{BAR}{DASH * 16}{BAR}" for s in reversed(range(n_strings)))


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


def _header(cfg: GuitarConfig, n_events: int, *, color: bool = False) -> str:
    # Tuning conventionally written low-to-high.
    tuning_names = " ".join(STRING_NAMES[i].upper() for i in range(cfg.n_strings))
    capo = f"Capo: {cfg.capo}" if cfg.capo > 0 else "Capo: none"
    if color:
        legend = (
            f"Confidence: {_ANSI_HIGH}high{_ANSI_RESET} "
            f"{_ANSI_MEDIUM}medium{_ANSI_RESET} "
            f"{_ANSI_LOW}low{_ANSI_RESET}  (low frets also marked '?').\n"
        )
    else:
        legend = "Low-confidence notes marked with '?'.\n"
    return f"TabVision ASCII tab\nTuning: {tuning_names}   {capo}   Notes: {n_events}\n" + legend


__all__ = ["render"]
