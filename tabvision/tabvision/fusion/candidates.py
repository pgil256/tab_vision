"""Candidate (string, fret) generator — Phase 5 deliverable, used in Phase 1.

Given a MIDI pitch and a ``GuitarConfig`` (tuning + capo + max_fret),
returns every playable (string_idx, fret) on the instrument.

Port target: ``tabvision-server/app/guitar_mapping.py`` (with string-index
convention swapped: spec uses 0 = low E .. 5 = high E).
"""

from __future__ import annotations

from dataclasses import dataclass

from tabvision.types import GuitarConfig


@dataclass(frozen=True)
class Candidate:
    """A single playable position for a target pitch."""

    string_idx: int  # 0 = low E, 5 = high E (matches SPEC §8 TabEvent)
    fret: int  # 0 = open (or capo), max_fret inclusive


def candidate_positions(pitch_midi: int, cfg: GuitarConfig | None = None) -> list[Candidate]:
    """All valid positions for ``pitch_midi`` under ``cfg``.

    Capo handling: open strings effectively start at ``cfg.capo``. A pitch
    that would require a fret below the capo is unplayable on that string
    and is skipped.

    Returns the list sorted by (fret, string_idx) ascending — the first
    element is the "lowest fret" pick used in Phase 1's audio-only fusion.
    """
    if cfg is None:
        cfg = GuitarConfig()

    candidates: list[Candidate] = []
    for s, open_midi in enumerate(cfg.tuning_midi):
        fret = pitch_midi - open_midi
        if cfg.capo <= fret <= cfg.max_fret:
            candidates.append(Candidate(string_idx=s, fret=fret))

    candidates.sort(key=lambda c: (c.fret, c.string_idx))
    return candidates


__all__ = ["Candidate", "candidate_positions"]
