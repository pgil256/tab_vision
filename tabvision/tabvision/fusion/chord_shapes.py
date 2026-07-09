"""Chord-shape dictionary + per-cluster shape bonus — roadmap A5.

Ports the voicing shapes from v0's ``tabvision-server/app/chord_shapes.py``
into a Phase-5 emission term. A *chord cluster* (see :mod:`tabvision.fusion.chord`)
decodes to an ordered tuple of ``(string, fret)`` candidates; among the
pitch-equivalent assignments the Viterbi enumerates, this term rewards the one
whose positions coincide with a **recognised guitar voicing** (open chords,
E/A-shape barres, power chords). That biases strummed clusters toward
shape-consistent fingerings — the ``wrong_position`` lever called out in the
roadmap A5 entry.

Only the voicing *shapes* are ported. v0's scale-box patterns, ``GuitarPosition``
ranges, and ``PlayingStyle`` weighting are **not** — they play no part in an
emission bonus and would be dead code here.

String-index convention
------------------------
v0 numbers strings ``1 = high E .. 6 = low E``; v1 uses ``Candidate.string_idx``
with ``0 = low E .. 5 = high E`` (SPEC §8 / :mod:`tabvision.fusion.candidates`).
The single conversion is ``string_idx = 6 - v0_string``, applied once in
:func:`_positions`. Shapes assume **standard tuning, no capo** (v1's default
``GuitarConfig`` and the acoustic eval corpora); under a non-standard tuning or
capo the geometric positions no longer correspond to the produced pitches, so
the bonus is best left at its default (off) there.

Default magnitude (A5-gated)
----------------------------
:data:`CHORD_SHAPE_BONUS` defaults to ``0.1`` — the value that cleared the full
A3 gate on 2026-07-07 (in-domain 60-clip lower-95 + GAPS clean-12 strict
no-regression; strummed +0.005, single-line +0.000). Set it to ``0.0`` (or
``TABVISION_CHORD_SHAPE_BONUS=0.0``) to disable the term — then
:func:`chord_shape_cost` returns ``0.0`` and the decode is **bit-identical** to a
fusion without it. The bonus is env-overridable and runtime-rebindable —
:mod:`tabvision.fusion.viterbi` reads it live per call, so the A3 sweep
(``scripts.eval.a3_fusion_sweep``) can grid over it. Because a match needs
``>= CHORD_SHAPE_MIN_NOTES`` (default 3) overlapping positions, a singleton
(single-line) or dyad cluster can never trigger it — **single-line Tab F1 is
invariant to this term at any magnitude** (empirically exact on GuitarSet).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tabvision.fusion.candidates import Candidate

CHORD_SHAPE_BONUS = float(os.environ.get("TABVISION_CHORD_SHAPE_BONUS", "0.1"))
"""Reward (nats) **per on-shape note** subtracted from a cluster's state
emission when its decoded positions overlap a known voicing by at least
:data:`CHORD_SHAPE_MIN_NOTES`. Default ``0.1`` — the A5-gated shipped value
(2026-07-07: clears both gate legs, strummed +0.005; ``0.25+`` over-biases and
regresses). Set to ``0.0`` to disable. Env-overridable
(``TABVISION_CHORD_SHAPE_BONUS``)."""

CHORD_SHAPE_MIN_NOTES = int(float(os.environ.get("TABVISION_CHORD_SHAPE_MIN_NOTES", "3")))
"""Minimum number of a state's positions that must land on one canonical
voicing for the bonus to apply. Default ``3`` (a triad) — dyads and single
notes are excluded, keeping the single-line tier untouched. Env-overridable
(``TABVISION_CHORD_SHAPE_MIN_NOTES``)."""


@dataclass(frozen=True)
class Voicing:
    """A recognised chord shape as a set of ``(string_idx, fret)`` positions."""

    name: str
    positions: frozenset[tuple[int, int]]


def _positions(strings: Mapping[int, int | None]) -> frozenset[tuple[int, int]]:
    """Convert a v0 ``{v0_string: fret|None}`` shape to v1 ``(string_idx, fret)``.

    ``None`` frets (muted / not played) are dropped; ``string_idx = 6 - v0_string``.
    """
    return frozenset((6 - s, f) for s, f in strings.items() if f is not None)


# --- Open / first-position voicings (ported verbatim from v0 OPEN_CHORDS) ---
_OPEN_SHAPES: list[tuple[str, dict[int, int | None]]] = [
    ("C major open", {6: None, 5: 3, 4: 2, 3: 0, 2: 1, 1: 0}),
    ("D major open", {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 2}),
    ("D minor open", {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 1}),
    ("E major open", {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}),
    ("E minor open", {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0}),
    ("G major open", {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3}),
    ("G major open alt", {6: 3, 5: 2, 4: 0, 3: 0, 2: 3, 1: 3}),
    ("A major open", {6: None, 5: 0, 4: 2, 3: 2, 2: 2, 1: 0}),
    ("A minor open", {6: None, 5: 0, 4: 2, 3: 2, 2: 1, 1: 0}),
    ("F major (partial)", {6: None, 5: None, 4: 3, 3: 2, 2: 1, 1: 1}),
    ("F major barre", {6: 1, 5: 3, 4: 3, 3: 2, 2: 1, 1: 1}),
    ("Bm barre", {6: None, 5: 2, 4: 4, 3: 4, 2: 3, 1: 2}),
    ("D7 open", {6: None, 5: None, 4: 0, 3: 2, 2: 1, 1: 2}),
    ("A7 open", {6: None, 5: 0, 4: 2, 3: 0, 2: 2, 1: 0}),
    ("E7 open", {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0}),
    ("G7 open", {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 1}),
    ("C7 open", {6: None, 5: 3, 4: 2, 3: 3, 2: 1, 1: 0}),
    ("Cadd9", {6: None, 5: 3, 4: 2, 3: 0, 2: 3, 1: 0}),
    ("Dsus2", {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 0}),
    ("Dsus4", {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 3}),
    ("Asus2", {6: None, 5: 0, 4: 2, 3: 2, 2: 0, 1: 0}),
    ("Asus4", {6: None, 5: 0, 4: 2, 3: 2, 2: 3, 1: 0}),
]

# --- Movable barre templates (v0 ``_E_*``/``_A_*`` shapes at open position) ---
_E_MAJOR = {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}
_E_MINOR = {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0}
_E_DOM7 = {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0}
_A_MAJOR: dict[int, int | None] = {6: None, 5: 0, 4: 2, 3: 2, 2: 2, 1: 0}
_A_MINOR: dict[int, int | None] = {6: None, 5: 0, 4: 2, 3: 2, 2: 1, 1: 0}
_A_DOM7: dict[int, int | None] = {6: None, 5: 0, 4: 2, 3: 0, 2: 2, 1: 0}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# v1 open-string MIDI, indexed by v0 string number (6 = low E .. 1 = high E).
_V0_OPEN_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}


def _shifted(shape: Mapping[int, int | None], root_fret: int) -> dict[int, int | None]:
    """Move a barre template up the neck by ``root_fret`` frets."""
    return {s: (None if f is None else f + root_fret) for s, f in shape.items()}


def _barre_voicings() -> list[Voicing]:
    """E-shape (6th-string root) and A-shape (5th-string root) barres, frets 1–12."""
    out: list[Voicing] = []
    for fret in range(1, 13):
        e_root = _NOTE_NAMES[(_V0_OPEN_MIDI[6] + fret) % 12]
        out.append(
            Voicing(f"{e_root} major (E-shape @{fret})", _positions(_shifted(_E_MAJOR, fret)))
        )
        out.append(Voicing(f"{e_root}m (E-shape @{fret})", _positions(_shifted(_E_MINOR, fret))))
        out.append(Voicing(f"{e_root}7 (E-shape @{fret})", _positions(_shifted(_E_DOM7, fret))))
        a_root = _NOTE_NAMES[(_V0_OPEN_MIDI[5] + fret) % 12]
        out.append(
            Voicing(f"{a_root} major (A-shape @{fret})", _positions(_shifted(_A_MAJOR, fret)))
        )
        out.append(Voicing(f"{a_root}m (A-shape @{fret})", _positions(_shifted(_A_MINOR, fret))))
        out.append(Voicing(f"{a_root}7 (A-shape @{fret})", _positions(_shifted(_A_DOM7, fret))))
    return out


def _power_voicings() -> list[Voicing]:
    """Root-fifth(-octave) power chords on the 6th and 5th strings, frets 0–12."""
    out: list[Voicing] = []
    for fret in range(0, 13):
        e_root = _NOTE_NAMES[(_V0_OPEN_MIDI[6] + fret) % 12]
        out.append(
            Voicing(
                f"{e_root}5 (6th, 3-note @{fret})", _positions({6: fret, 5: fret + 2, 4: fret + 2})
            )
        )
        out.append(Voicing(f"{e_root}5 (6-5 dyad @{fret})", _positions({6: fret, 5: fret + 2})))
        a_root = _NOTE_NAMES[(_V0_OPEN_MIDI[5] + fret) % 12]
        out.append(
            Voicing(
                f"{a_root}5 (5th, 3-note @{fret})", _positions({5: fret, 4: fret + 2, 3: fret + 2})
            )
        )
    return out


VOICINGS: list[Voicing] = (
    [Voicing(name, _positions(shape)) for name, shape in _OPEN_SHAPES]
    + _barre_voicings()
    + _power_voicings()
)
"""Every recognised voicing (open + barre + power), positions in v1 convention."""

# Inverted index: (string_idx, fret) -> voicing indices containing it. Lets
# :func:`best_shape_overlap` score a state without scanning all voicings.
_VOICINGS_BY_POSITION: dict[tuple[int, int], list[int]] = {}
for _vid, _v in enumerate(VOICINGS):
    for _pos in _v.positions:
        _VOICINGS_BY_POSITION.setdefault(_pos, []).append(_vid)


def best_shape_overlap(positions: Iterable[tuple[int, int]]) -> int:
    """Largest number of ``positions`` that coincide with a single voicing.

    ``|P ∩ V|`` maximised over all voicings ``V``. ``0`` when no voicing shares
    a position (or ``positions`` is empty).
    """
    counts: dict[int, int] = {}
    for pos in positions:
        for vid in _VOICINGS_BY_POSITION.get(pos, ()):
            counts[vid] = counts.get(vid, 0) + 1
    return max(counts.values(), default=0)


def chord_shape_cost(state: Sequence[Candidate]) -> float:
    """Emission adjustment (nats, ``<= 0``) rewarding a shape-consistent state.

    Returns ``-CHORD_SHAPE_BONUS * overlap`` when the state's positions overlap
    some canonical voicing by at least :data:`CHORD_SHAPE_MIN_NOTES`, else
    ``0.0``. With the default ``CHORD_SHAPE_BONUS == 0.0`` this is a fast,
    exact ``0.0`` — no voicing scan, bit-identical to a fusion without the term.
    """
    if CHORD_SHAPE_BONUS == 0.0:
        return 0.0
    overlap = best_shape_overlap((c.string_idx, c.fret) for c in state)
    if overlap < CHORD_SHAPE_MIN_NOTES:
        return 0.0
    return -CHORD_SHAPE_BONUS * overlap


__all__ = [
    "CHORD_SHAPE_BONUS",
    "CHORD_SHAPE_MIN_NOTES",
    "Voicing",
    "VOICINGS",
    "best_shape_overlap",
    "chord_shape_cost",
]
