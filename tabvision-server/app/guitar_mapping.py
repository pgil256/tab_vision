"""Guitar MIDI note to fret/string position mapping."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """A fret/string position on the guitar."""
    string: int  # 1-6 (1=high E, 6=low E)
    fret: int    # 0-24


# Standard tuning: MIDI note numbers for open strings
STANDARD_TUNING = {
    6: 40,  # Low E (E2)
    5: 45,  # A (A2)
    4: 50,  # D (D3)
    3: 55,  # G (G3)
    2: 59,  # B (B3)
    1: 64,  # High E (E4)
}

MAX_FRET = 24


def get_candidate_positions(midi_note: int, capo_fret: int = 0) -> list[Position]:
    """Return all valid fret/string positions for a MIDI note.

    Args:
        midi_note: MIDI note number (e.g., 69 for A4)
        capo_fret: Fret where capo is placed (0 = no capo)

    Returns:
        List of Position objects, sorted by fret ascending
    """
    positions = []

    for string, open_midi in STANDARD_TUNING.items():
        fret = midi_note - open_midi
        if capo_fret <= fret <= MAX_FRET:
            positions.append(Position(string=string, fret=fret))

    # Sort by fret ascending
    positions.sort(key=lambda p: p.fret)
    return positions


def pick_lowest_fret(candidates: list[Position]) -> Position | None:
    """Select the position with the lowest fret number.

    Args:
        candidates: List of possible positions

    Returns:
        Position with lowest fret, or None if list is empty
    """
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.fret)
