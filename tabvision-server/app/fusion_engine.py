"""Fusion engine for combining audio analysis into tab notes."""
from dataclasses import dataclass
from uuid import uuid4
from app.audio_pipeline import DetectedNote
from app.guitar_mapping import get_candidate_positions, pick_lowest_fret


@dataclass
class TabNote:
    """A note in the guitar tablature."""
    id: str
    timestamp: float        # seconds
    string: int             # 1-6
    fret: int               # 0-24
    confidence: float       # 0.0-1.0
    confidence_level: str   # "high", "medium", "low"
    midi_note: int          # Original MIDI note for debugging


def get_confidence_level(confidence: float) -> str:
    """Map confidence score to level.

    Args:
        confidence: Score from 0.0 to 1.0

    Returns:
        "high" (>0.8), "medium" (0.5-0.8), or "low" (<0.5)
    """
    if confidence > 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def fuse_audio_only(
    detected_notes: list[DetectedNote],
    capo_fret: int = 0
) -> list[TabNote]:
    """Convert detected audio notes to TabNotes using lowest-fret heuristic.

    Args:
        detected_notes: Notes detected from audio analysis
        capo_fret: Fret where capo is placed (0 = no capo)

    Returns:
        List of TabNote objects
    """
    tab_notes = []

    for note in detected_notes:
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue  # Skip notes outside guitar range

        position = pick_lowest_fret(candidates)
        if position is None:
            continue

        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=note.start_time,
            string=position.string,
            fret=position.fret,
            confidence=note.confidence,
            confidence_level=get_confidence_level(note.confidence),
            midi_note=note.midi_note,
        ))

    return tab_notes
