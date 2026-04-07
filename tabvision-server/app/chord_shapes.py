"""Chord shape templates, scale box patterns, and positional awareness for tablature.

Provides fretboard-shape-aware heuristics that go beyond interval-based scoring:
- Common chord voicing templates (open shapes, barre patterns)
- Scale box patterns (pentatonic, major scale fingerings)
- Positional pattern awareness (fret ranges per position)
- Context-aware chord shape recognition
- Genre/style heuristic weights
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.guitar_mapping import Position, STANDARD_TUNING, MAX_FRET


# ---------------------------------------------------------------------------
# Chord voicing templates
# ---------------------------------------------------------------------------
# Each voicing is a dict: string_number -> fret (None = not played, 0 = open)
# String numbering: 1=high E, 2=B, 3=G, 4=D, 5=A, 6=low E

@dataclass(frozen=True)
class ChordVoicing:
    """A specific chord voicing on the fretboard."""
    name: str           # e.g. "C major open"
    chord_type: str     # e.g. "major", "minor", "power", "dom7"
    root_note: int      # MIDI note of root (e.g. 48 for C3)
    strings: dict[int, Optional[int]]  # string -> fret (None = muted/not played)
    is_barre: bool = False
    barre_fret: int = 0  # which fret is barred (0 = no barre)
    style_tags: tuple[str, ...] = ()  # e.g. ("open", "rock", "jazz")

    @property
    def played_positions(self) -> list[Position]:
        """Get all played positions in this voicing."""
        return [
            Position(string=s, fret=f)
            for s, f in self.strings.items()
            if f is not None
        ]

    @property
    def fret_span(self) -> int:
        """Fret span of fretted notes (excluding open strings)."""
        fretted = [f for f in self.strings.values() if f is not None and f > 0]
        if not fretted:
            return 0
        return max(fretted) - min(fretted)

    @property
    def midi_notes(self) -> list[int]:
        """Get MIDI notes produced by this voicing."""
        notes = []
        for string, fret in self.strings.items():
            if fret is not None:
                notes.append(STANDARD_TUNING[string] + fret)
        return sorted(notes)


def _barre_shape(name: str, chord_type: str, root_midi: int,
                 shape: dict[int, Optional[int]], root_fret: int,
                 barre_string_range: tuple[int, int],
                 style_tags: tuple[str, ...] = ("barre",)) -> ChordVoicing:
    """Create a barre chord voicing by shifting a shape to a root fret."""
    strings = {}
    for s, f in shape.items():
        if f is None:
            strings[s] = None
        else:
            strings[s] = f + root_fret
    return ChordVoicing(
        name=name, chord_type=chord_type, root_note=root_midi,
        strings=strings, is_barre=True, barre_fret=root_fret,
        style_tags=style_tags,
    )


# --- Open chord voicings ---
OPEN_CHORDS: list[ChordVoicing] = [
    # C major open
    ChordVoicing("C major open", "major", 48,
                 {6: None, 5: 3, 4: 2, 3: 0, 2: 1, 1: 0},
                 style_tags=("open", "classical", "folk")),
    # D major open
    ChordVoicing("D major open", "major", 50,
                 {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 2},
                 style_tags=("open", "folk")),
    # D minor open
    ChordVoicing("D minor open", "minor", 50,
                 {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 1},
                 style_tags=("open", "folk")),
    # E major open
    ChordVoicing("E major open", "major", 40,
                 {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0},
                 style_tags=("open", "rock", "folk")),
    # E minor open
    ChordVoicing("Em open", "minor", 40,
                 {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0},
                 style_tags=("open", "rock", "folk")),
    # G major open
    ChordVoicing("G major open", "major", 43,
                 {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 3},
                 style_tags=("open", "folk", "rock")),
    # G major open (alt - 3rd fret B string)
    ChordVoicing("G major open alt", "major", 43,
                 {6: 3, 5: 2, 4: 0, 3: 0, 2: 3, 1: 3},
                 style_tags=("open", "folk")),
    # A major open
    ChordVoicing("A major open", "major", 45,
                 {6: None, 5: 0, 4: 2, 3: 2, 2: 2, 1: 0},
                 style_tags=("open", "rock", "folk")),
    # A minor open
    ChordVoicing("Am open", "minor", 45,
                 {6: None, 5: 0, 4: 2, 3: 2, 2: 1, 1: 0},
                 style_tags=("open", "classical", "folk")),
    # F major (partial barre)
    ChordVoicing("F major", "major", 41,
                 {6: None, 5: None, 4: 3, 3: 2, 2: 1, 1: 1},
                 is_barre=True, barre_fret=1,
                 style_tags=("open", "folk")),
    # F major full barre
    ChordVoicing("F major barre", "major", 41,
                 {6: 1, 5: 3, 4: 3, 3: 2, 2: 1, 1: 1},
                 is_barre=True, barre_fret=1,
                 style_tags=("barre",)),
    # B minor barre
    ChordVoicing("Bm barre", "minor", 47,
                 {6: None, 5: 2, 4: 4, 3: 4, 2: 3, 1: 2},
                 is_barre=True, barre_fret=2,
                 style_tags=("barre",)),
    # D7 open
    ChordVoicing("D7 open", "dom7", 50,
                 {6: None, 5: None, 4: 0, 3: 2, 2: 1, 1: 2},
                 style_tags=("open", "blues")),
    # A7 open
    ChordVoicing("A7 open", "dom7", 45,
                 {6: None, 5: 0, 4: 2, 3: 0, 2: 2, 1: 0},
                 style_tags=("open", "blues")),
    # E7 open
    ChordVoicing("E7 open", "dom7", 40,
                 {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0},
                 style_tags=("open", "blues")),
    # G7 open
    ChordVoicing("G7 open", "dom7", 43,
                 {6: 3, 5: 2, 4: 0, 3: 0, 2: 0, 1: 1},
                 style_tags=("open", "blues")),
    # C7 open
    ChordVoicing("C7 open", "dom7", 48,
                 {6: None, 5: 3, 4: 2, 3: 3, 2: 1, 1: 0},
                 style_tags=("open", "blues")),
    # Cadd9
    ChordVoicing("Cadd9", "add9", 48,
                 {6: None, 5: 3, 4: 2, 3: 0, 2: 3, 1: 0},
                 style_tags=("open", "folk")),
    # Dsus2
    ChordVoicing("Dsus2", "sus2", 50,
                 {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 0},
                 style_tags=("open",)),
    # Dsus4
    ChordVoicing("Dsus4", "sus4", 50,
                 {6: None, 5: None, 4: 0, 3: 2, 2: 3, 1: 3},
                 style_tags=("open",)),
    # Asus2
    ChordVoicing("Asus2", "sus2", 45,
                 {6: None, 5: 0, 4: 2, 3: 2, 2: 0, 1: 0},
                 style_tags=("open",)),
    # Asus4
    ChordVoicing("Asus4", "sus4", 45,
                 {6: None, 5: 0, 4: 2, 3: 2, 2: 3, 1: 0},
                 style_tags=("open",)),
]


# --- E-shape barre chord templates (root on 6th string) ---
_E_MAJOR_SHAPE = {6: 0, 5: 2, 4: 2, 3: 1, 2: 0, 1: 0}
_E_MINOR_SHAPE = {6: 0, 5: 2, 4: 2, 3: 0, 2: 0, 1: 0}
_E_DOM7_SHAPE = {6: 0, 5: 2, 4: 0, 3: 1, 2: 0, 1: 0}

# --- A-shape barre chord templates (root on 5th string) ---
_A_MAJOR_SHAPE = {6: None, 5: 0, 4: 2, 3: 2, 2: 2, 1: 0}
_A_MINOR_SHAPE = {6: None, 5: 0, 4: 2, 3: 2, 2: 1, 1: 0}
_A_DOM7_SHAPE = {6: None, 5: 0, 4: 2, 3: 0, 2: 2, 1: 0}


def _generate_barre_chords() -> list[ChordVoicing]:
    """Generate barre chords from E-shape and A-shape templates across the neck."""
    chords = []
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # E-shape barres (root on 6th string, frets 1-12)
    for fret in range(1, 13):
        root_midi = STANDARD_TUNING[6] + fret  # 6th string open = E2 (40)
        root_name = note_names[(root_midi - 40) % 12]

        chords.append(_barre_shape(
            f"{root_name} major E-shape", "major", root_midi,
            _E_MAJOR_SHAPE, fret, (1, 6), ("barre", "rock"),
        ))
        chords.append(_barre_shape(
            f"{root_name}m E-shape", "minor", root_midi,
            _E_MINOR_SHAPE, fret, (1, 6), ("barre", "rock"),
        ))
        chords.append(_barre_shape(
            f"{root_name}7 E-shape", "dom7", root_midi,
            _E_DOM7_SHAPE, fret, (1, 6), ("barre", "blues"),
        ))

    # A-shape barres (root on 5th string, frets 1-12)
    for fret in range(1, 13):
        root_midi = STANDARD_TUNING[5] + fret  # 5th string open = A2 (45)
        root_name = note_names[(root_midi - 45) % 12]

        chords.append(_barre_shape(
            f"{root_name} major A-shape", "major", root_midi,
            _A_MAJOR_SHAPE, fret, (1, 5), ("barre",),
        ))
        chords.append(_barre_shape(
            f"{root_name}m A-shape", "minor", root_midi,
            _A_MINOR_SHAPE, fret, (1, 5), ("barre",),
        ))
        chords.append(_barre_shape(
            f"{root_name}7 A-shape", "dom7", root_midi,
            _A_DOM7_SHAPE, fret, (1, 5), ("barre", "blues"),
        ))

    return chords


# --- Power chords ---
def _generate_power_chords() -> list[ChordVoicing]:
    """Generate power chord voicings across the neck."""
    chords = []
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # 6th string root power chords
    for fret in range(0, 13):
        root_midi = STANDARD_TUNING[6] + fret
        root_name = note_names[(root_midi - 40) % 12]
        chords.append(ChordVoicing(
            f"{root_name}5 (6th)", "power", root_midi,
            {6: fret, 5: fret + 2, 4: fret + 2, 3: None, 2: None, 1: None},
            style_tags=("rock", "punk", "metal"),
        ))
        # Two-string power chord
        chords.append(ChordVoicing(
            f"{root_name}5 (6-5)", "power", root_midi,
            {6: fret, 5: fret + 2, 4: None, 3: None, 2: None, 1: None},
            style_tags=("rock", "punk", "metal"),
        ))

    # 5th string root power chords
    for fret in range(0, 13):
        root_midi = STANDARD_TUNING[5] + fret
        root_name = note_names[(root_midi - 45) % 12]
        chords.append(ChordVoicing(
            f"{root_name}5 (5th)", "power", root_midi,
            {6: None, 5: fret, 4: fret + 2, 3: fret + 2, 2: None, 1: None},
            style_tags=("rock", "punk", "metal"),
        ))

    return chords


# Build the complete voicing database
CHORD_VOICINGS: list[ChordVoicing] = (
    OPEN_CHORDS
    + _generate_barre_chords()
    + _generate_power_chords()
)

# Index by pitch-class set for fast lookup
_VOICING_BY_PITCH_CLASSES: dict[frozenset[int], list[ChordVoicing]] = {}
for _v in CHORD_VOICINGS:
    _pc = frozenset(m % 12 for m in _v.midi_notes)
    _VOICING_BY_PITCH_CLASSES.setdefault(_pc, []).append(_v)


# ---------------------------------------------------------------------------
# Scale box patterns
# ---------------------------------------------------------------------------
# Each pattern defines fret offsets per string relative to a root fret position.
# Offsets are relative to the position root fret (e.g., position V root = fret 5).
# Format: string -> list of fret offsets from position root

@dataclass(frozen=True)
class ScalePattern:
    """A scale fingering pattern on the fretboard."""
    name: str
    pattern_type: str     # "pentatonic_minor", "pentatonic_major", "major", "minor"
    # Fret offsets per string relative to position root.
    # string -> tuple of fret offsets (0 = position root fret)
    offsets: dict[int, tuple[int, ...]]
    style_tags: tuple[str, ...] = ()

    def get_frets_at_position(self, position_fret: int) -> dict[int, tuple[int, ...]]:
        """Get actual frets for this pattern at a given position."""
        result = {}
        for string, offs in self.offsets.items():
            frets = tuple(position_fret + o for o in offs if 0 <= position_fret + o <= MAX_FRET)
            if frets:
                result[string] = frets
        return result


# Minor pentatonic boxes (the 5 CAGED positions)
PENTATONIC_MINOR_PATTERNS: list[ScalePattern] = [
    ScalePattern("Minor Pentatonic Box 1", "pentatonic_minor",
                 {6: (0, 3), 5: (0, 3), 4: (0, 2), 3: (0, 2), 2: (0, 3), 1: (0, 3)},
                 style_tags=("rock", "blues")),
    ScalePattern("Minor Pentatonic Box 2", "pentatonic_minor",
                 {6: (0, 2), 5: (0, 2), 4: (-1, 2), 3: (-1, 2), 2: (0, 2), 1: (0, 2)},
                 style_tags=("rock", "blues")),
    ScalePattern("Minor Pentatonic Box 3", "pentatonic_minor",
                 {6: (0, 2), 5: (0, 2), 4: (0, 2), 3: (0, 2), 2: (0, 3), 1: (0, 2)},
                 style_tags=("rock", "blues")),
    ScalePattern("Minor Pentatonic Box 4", "pentatonic_minor",
                 {6: (0, 2), 5: (0, 3), 4: (0, 2), 3: (0, 2), 2: (0, 2), 1: (0, 2)},
                 style_tags=("rock", "blues")),
    ScalePattern("Minor Pentatonic Box 5", "pentatonic_minor",
                 {6: (0, 3), 5: (0, 2), 4: (0, 2), 3: (0, 2), 2: (0, 3), 1: (0, 3)},
                 style_tags=("rock", "blues")),
]

# Major scale patterns (3-note-per-string system)
MAJOR_SCALE_PATTERNS: list[ScalePattern] = [
    ScalePattern("Major Scale Pos 1 (3nps)", "major",
                 {6: (0, 2, 4), 5: (0, 2, 4), 4: (1, 2, 4), 3: (1, 2, 4), 2: (1, 3, 4), 1: (0, 2, 4)},
                 style_tags=("classical", "jazz")),
    ScalePattern("Major Scale Pos 2 (3nps)", "major",
                 {6: (0, 2, 4), 5: (0, 2, 3), 4: (0, 2, 4), 3: (0, 2, 4), 2: (0, 2, 3), 1: (0, 2, 4)},
                 style_tags=("classical", "jazz")),
    ScalePattern("Major Scale Pos 3 (3nps)", "major",
                 {6: (0, 2, 3), 5: (0, 2, 4), 4: (0, 2, 4), 3: (0, 1, 4), 2: (0, 2, 4), 1: (0, 2, 3)},
                 style_tags=("classical", "jazz")),
]

# Natural minor scale patterns
MINOR_SCALE_PATTERNS: list[ScalePattern] = [
    ScalePattern("Natural Minor Pos 1 (3nps)", "minor",
                 {6: (0, 2, 3), 5: (0, 2, 3), 4: (0, 2, 3), 3: (0, 2, 4), 2: (0, 1, 3), 1: (0, 2, 3)},
                 style_tags=("classical",)),
]

ALL_SCALE_PATTERNS: list[ScalePattern] = (
    PENTATONIC_MINOR_PATTERNS
    + MAJOR_SCALE_PATTERNS
    + MINOR_SCALE_PATTERNS
)


# ---------------------------------------------------------------------------
# Positional pattern awareness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuitarPosition:
    """A hand position on the guitar neck."""
    name: str           # e.g. "Position I", "Position V"
    root_fret: int      # the index finger fret
    fret_range: tuple[int, int]  # (min_fret, max_fret) reachable

    def contains_fret(self, fret: int) -> bool:
        """Check if a fret is within this position's range."""
        return self.fret_range[0] <= fret <= self.fret_range[1]


# Standard guitar positions (classical numbering)
GUITAR_POSITIONS: list[GuitarPosition] = [
    GuitarPosition("Open", 0, (0, 4)),
    GuitarPosition("Position I", 1, (1, 4)),
    GuitarPosition("Position II", 2, (2, 5)),
    GuitarPosition("Position III", 3, (3, 6)),
    GuitarPosition("Position IV", 4, (4, 7)),
    GuitarPosition("Position V", 5, (5, 8)),
    GuitarPosition("Position VI", 6, (6, 9)),
    GuitarPosition("Position VII", 7, (7, 10)),
    GuitarPosition("Position VIII", 8, (8, 11)),
    GuitarPosition("Position IX", 9, (9, 12)),
    GuitarPosition("Position X", 10, (10, 13)),
    GuitarPosition("Position XI", 11, (11, 14)),
    GuitarPosition("Position XII", 12, (12, 15)),
]


def get_position_for_fret(fret: int) -> Optional[GuitarPosition]:
    """Get the guitar position that best matches a fret."""
    if fret == 0:
        return GUITAR_POSITIONS[0]  # Open position
    for pos in GUITAR_POSITIONS:
        if pos.root_fret == fret:
            return pos
    # Find closest
    return min(GUITAR_POSITIONS, key=lambda p: abs(p.root_fret - fret))


# ---------------------------------------------------------------------------
# Genre / style heuristics
# ---------------------------------------------------------------------------

class PlayingStyle(Enum):
    """Playing style that affects position selection preferences."""
    DEFAULT = "default"
    ROCK = "rock"
    BLUES = "blues"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    FOLK = "folk"
    METAL = "metal"
    PUNK = "punk"


@dataclass
class StyleWeights:
    """Scoring weight adjustments for a playing style."""
    # Chord voicing preferences
    open_chord_bonus: float = 0.0     # bonus for open chord voicings
    barre_chord_bonus: float = 0.0    # bonus for barre chord voicings
    power_chord_bonus: float = 0.0    # bonus for power chord voicings

    # Position preferences
    lower_fret_weight: float = 0.05   # penalty per fret for lower-fret preference
    position_stay_weight: float = 0.35  # penalty per fret for position changes

    # Scale pattern preferences
    pentatonic_bonus: float = 0.0     # bonus for notes in pentatonic pattern
    scale_pattern_bonus: float = 0.0  # bonus for notes in major/minor scale pattern

    # Voicing match bonus (when chord notes match a known voicing)
    voicing_match_bonus: float = 0.3  # bonus for matching a known chord voicing


STYLE_WEIGHTS: dict[PlayingStyle, StyleWeights] = {
    PlayingStyle.DEFAULT: StyleWeights(),
    PlayingStyle.ROCK: StyleWeights(
        open_chord_bonus=0.1,
        power_chord_bonus=0.3,
        pentatonic_bonus=0.15,
        lower_fret_weight=0.03,
        voicing_match_bonus=0.35,
    ),
    PlayingStyle.BLUES: StyleWeights(
        open_chord_bonus=0.15,
        pentatonic_bonus=0.2,
        voicing_match_bonus=0.3,
    ),
    PlayingStyle.CLASSICAL: StyleWeights(
        open_chord_bonus=0.2,
        scale_pattern_bonus=0.15,
        lower_fret_weight=0.04,
        position_stay_weight=0.4,
        voicing_match_bonus=0.25,
    ),
    PlayingStyle.JAZZ: StyleWeights(
        barre_chord_bonus=0.15,
        scale_pattern_bonus=0.1,
        lower_fret_weight=0.02,
        position_stay_weight=0.3,
        voicing_match_bonus=0.2,
    ),
    PlayingStyle.FOLK: StyleWeights(
        open_chord_bonus=0.25,
        lower_fret_weight=0.05,
        voicing_match_bonus=0.35,
    ),
    PlayingStyle.METAL: StyleWeights(
        power_chord_bonus=0.35,
        pentatonic_bonus=0.1,
        lower_fret_weight=0.02,
        voicing_match_bonus=0.3,
    ),
    PlayingStyle.PUNK: StyleWeights(
        power_chord_bonus=0.4,
        lower_fret_weight=0.03,
        voicing_match_bonus=0.3,
    ),
}


# ---------------------------------------------------------------------------
# Matching / scoring functions
# ---------------------------------------------------------------------------

@dataclass
class ChordShapeConfig:
    """Configuration for chord shape constraints."""
    enabled: bool = True
    # Weight for chord shape score in position optimization
    shape_score_weight: float = 0.3
    # Minimum notes to apply chord shape scoring
    min_chord_notes: int = 2
    # Penalty for notes that don't fit any voicing
    unvoiced_penalty: float = 0.15
    # Only suppress notes below this confidence threshold
    suppress_confidence_threshold: float = 0.5
    # Playing style
    style: PlayingStyle = PlayingStyle.DEFAULT
    # Scale pattern matching
    scale_pattern_enabled: bool = True
    # Position awareness
    position_awareness_enabled: bool = True


def find_matching_voicings(
    midi_notes: list[int],
    style: PlayingStyle = PlayingStyle.DEFAULT,
) -> list[tuple[ChordVoicing, float]]:
    """Find chord voicings that match the given MIDI notes.

    Returns voicings sorted by match score (best first).
    Match considers:
    - Pitch class match (notes modulo octave)
    - Exact octave/position match
    - Style preference

    Args:
        midi_notes: MIDI notes in the chord
        style: Playing style for preference scoring

    Returns:
        List of (voicing, score) tuples, sorted by score descending
    """
    if len(midi_notes) < 2:
        return []

    pitch_classes = frozenset(m % 12 for m in midi_notes)
    weights = STYLE_WEIGHTS.get(style, STYLE_WEIGHTS[PlayingStyle.DEFAULT])
    results = []

    # Look up voicings with matching pitch classes
    # Also check subsets (partial voicings) and supersets (the chord notes
    # are a subset of the voicing)
    for pc_set, voicings in _VOICING_BY_PITCH_CLASSES.items():
        # Require significant overlap
        overlap = pitch_classes & pc_set
        if len(overlap) < min(2, len(pitch_classes)):
            continue

        for voicing in voicings:
            voicing_pcs = frozenset(m % 12 for m in voicing.midi_notes)
            overlap_ratio = len(overlap) / max(len(pitch_classes), len(voicing_pcs))
            if overlap_ratio < 0.5:
                continue

            score = overlap_ratio

            # Exact pitch-class match bonus
            if pitch_classes == voicing_pcs:
                score += 0.5
            elif pitch_classes.issubset(voicing_pcs):
                score += 0.3  # Playing a subset of the voicing

            # Style tag bonus
            for tag in voicing.style_tags:
                if tag == "open" and weights.open_chord_bonus > 0:
                    score += weights.open_chord_bonus
                elif tag == "barre" and weights.barre_chord_bonus > 0:
                    score += weights.barre_chord_bonus
                elif tag == "rock" and style == PlayingStyle.ROCK:
                    score += 0.1
                elif tag == "blues" and style == PlayingStyle.BLUES:
                    score += 0.1
                elif tag == "jazz" and style == PlayingStyle.JAZZ:
                    score += 0.1

            # Power chord bonus
            if voicing.chord_type == "power" and weights.power_chord_bonus > 0:
                score += weights.power_chord_bonus

            results.append((voicing, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def score_position_against_voicing(
    positions: list[Position],
    voicing: ChordVoicing,
) -> float:
    """Score how well a set of positions matches a chord voicing.

    Args:
        positions: Proposed positions for chord notes
        voicing: Target chord voicing

    Returns:
        Score from 0.0 to 1.0 (1.0 = exact match)
    """
    if not positions:
        return 0.0

    voicing_positions = set(voicing.played_positions)
    matches = sum(1 for p in positions if p in voicing_positions)
    return matches / max(len(positions), len(voicing_positions))


def score_positions_against_scale(
    positions: list[Position],
    position_fret: int,
    style: PlayingStyle = PlayingStyle.DEFAULT,
) -> float:
    """Score how well positions fit within recognized scale patterns.

    Args:
        positions: Note positions to evaluate
        position_fret: Current hand position root fret
        style: Playing style

    Returns:
        Score from 0.0 to 1.0
    """
    if not positions:
        return 0.0

    weights = STYLE_WEIGHTS.get(style, STYLE_WEIGHTS[PlayingStyle.DEFAULT])
    best_score = 0.0

    for pattern in ALL_SCALE_PATTERNS:
        frets_at_pos = pattern.get_frets_at_position(position_fret)
        if not frets_at_pos:
            continue

        matches = 0
        for pos in positions:
            if pos.string in frets_at_pos and pos.fret in frets_at_pos[pos.string]:
                matches += 1

        if len(positions) > 0:
            match_ratio = matches / len(positions)
        else:
            match_ratio = 0.0

        # Apply style-specific bonuses
        pattern_bonus = 0.0
        if "pentatonic" in pattern.pattern_type:
            pattern_bonus = weights.pentatonic_bonus
        else:
            pattern_bonus = weights.scale_pattern_bonus

        score = match_ratio + pattern_bonus * match_ratio
        best_score = max(best_score, score)

    return best_score


def find_best_voicing_for_chord(
    midi_notes: list[int],
    hand_position_fret: Optional[float] = None,
    style: PlayingStyle = PlayingStyle.DEFAULT,
) -> Optional[ChordVoicing]:
    """Find the best chord voicing for a set of MIDI notes.

    Considers hand position proximity and style preferences.

    Args:
        midi_notes: MIDI notes in the chord
        hand_position_fret: Current hand position (None = no preference)
        style: Playing style

    Returns:
        Best matching voicing, or None if no good match
    """
    matches = find_matching_voicings(midi_notes, style)
    if not matches:
        return None

    best_voicing = None
    best_score = float('-inf')

    for voicing, base_score in matches:
        score = base_score

        # Penalize voicings far from hand position
        if hand_position_fret is not None:
            voicing_frets = [f for f in voicing.strings.values() if f is not None and f > 0]
            if voicing_frets:
                voicing_center = sum(voicing_frets) / len(voicing_frets)
                distance = abs(voicing_center - hand_position_fret)
                score -= distance * 0.05

        if score > best_score:
            best_score = score
            best_voicing = voicing

    return best_voicing


def get_voicing_positions(
    voicing: ChordVoicing,
    target_midi_notes: list[int],
) -> dict[int, Position]:
    """Get positions from a voicing that match target MIDI notes.

    Maps each target MIDI note to the corresponding position in the voicing.

    Args:
        voicing: Chord voicing to extract positions from
        target_midi_notes: MIDI notes we need positions for

    Returns:
        Dict mapping MIDI note -> Position
    """
    result = {}
    voicing_note_to_pos = {}
    for string, fret in voicing.strings.items():
        if fret is not None:
            midi = STANDARD_TUNING[string] + fret
            voicing_note_to_pos[midi] = Position(string=string, fret=fret)

    for midi in target_midi_notes:
        if midi in voicing_note_to_pos:
            result[midi] = voicing_note_to_pos[midi]
        else:
            # Try octave equivalence
            for v_midi, pos in voicing_note_to_pos.items():
                if v_midi % 12 == midi % 12:
                    # Check if this string can actually produce this exact note
                    needed_fret = midi - STANDARD_TUNING[pos.string]
                    if 0 <= needed_fret <= MAX_FRET:
                        result[midi] = Position(string=pos.string, fret=needed_fret)
                        break

    return result


def score_chord_voicing(midi_notes: list[int]) -> float:
    """Score how well a set of MIDI notes fits known chord voicings.

    Args:
        midi_notes: List of MIDI note numbers in the chord

    Returns:
        Score from -1.0 to 1.0. Positive = good voicing, negative = unlikely.
    """
    if len(midi_notes) < 2:
        return 0.0

    matches = find_matching_voicings(midi_notes)
    if matches:
        best_score = matches[0][1]
        if best_score >= 1.0:
            return 1.0
        elif best_score >= 0.7:
            return 0.8
        elif best_score >= 0.5:
            return 0.5
        else:
            return 0.3

    return -0.5


def filter_non_fitting_notes(
    chord_midi_notes: list[int],
    chord_positions: list[tuple[int, Position]],
    chord_confidences: list[float],
    config: ChordShapeConfig,
) -> list[int]:
    """Identify note indices that don't fit any chord voicing and have low confidence.

    Args:
        chord_midi_notes: MIDI notes in the chord
        chord_positions: List of (index, Position) pairs
        chord_confidences: Confidence for each note
        config: Chord shape configuration

    Returns:
        List of indices to suppress (remove from output)
    """
    if len(chord_midi_notes) < config.min_chord_notes + 1:
        return []

    base_score = score_chord_voicing(chord_midi_notes)
    if base_score >= 0.5:
        return []

    suppress_indices = []
    for i in range(len(chord_midi_notes)):
        if chord_confidences[i] >= config.suppress_confidence_threshold:
            continue

        remaining = [m for j, m in enumerate(chord_midi_notes) if j != i]
        if len(remaining) < config.min_chord_notes:
            continue

        new_score = score_chord_voicing(remaining)
        if new_score > base_score + 0.3:
            suppress_indices.append(i)

    return suppress_indices
