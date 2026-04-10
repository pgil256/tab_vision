"""Temporal interpolation for filling gaps in repeating patterns.

Detects repeating note/chord patterns and interpolates missing notes
at expected positions in the pattern.
"""
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4
from collections import Counter


logger = logging.getLogger(__name__)


@dataclass
class TemporalInterpolationConfig:
    """Configuration for temporal interpolation."""
    enabled: bool = False
    # Minimum repetitions to establish a pattern
    min_repetitions: int = 3
    # Timing tolerance for pattern matching (fraction of IOI)
    timing_tolerance: float = 0.15
    # Confidence assigned to interpolated notes
    interpolated_confidence: float = 0.65
    # Maximum gap size to interpolate (in pattern units)
    max_gap_units: int = 2


@dataclass
class PatternTemplate:
    """A detected repeating pattern."""
    # The inter-onset interval (IOI) of the pattern
    ioi: float
    # List of (relative_time, string, fret, midi_note) for each note in one repetition
    notes: list[tuple[float, int, int, int]]
    # Timestamps where this pattern was observed
    observed_at: list[float]
    # Number of confirmed repetitions
    repetitions: int


def detect_patterns(tab_notes: list) -> list[PatternTemplate]:
    """Detect repeating note/chord patterns in tab notes.

    Uses IOI (inter-onset interval) histogram to find the dominant
    rhythmic interval, then looks for note sequences that repeat
    at that interval.

    Args:
        tab_notes: List of TabNote objects

    Returns:
        List of detected patterns
    """
    if len(tab_notes) < 4:
        return []

    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)

    # Compute IOI histogram for single notes
    timestamps = sorted(set(n.timestamp for n in sorted_notes))
    if len(timestamps) < 3:
        return []

    iois = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if not iois:
        return []

    # Quantize IOIs to find dominant intervals
    # Round to nearest 0.05s
    quantized = [round(ioi * 20) / 20 for ioi in iois if 0.1 <= ioi <= 2.0]
    if not quantized:
        return []

    ioi_counts = Counter(quantized)
    dominant_iois = [ioi for ioi, count in ioi_counts.most_common(3) if count >= 2]

    if not dominant_iois:
        return []

    patterns = []

    for dominant_ioi in dominant_iois:
        if dominant_ioi <= 0:
            continue

        # Group notes by their position in the pattern grid
        # Each note gets a grid position = round(timestamp / dominant_ioi)
        grid: dict[int, list] = {}
        for note in sorted_notes:
            grid_pos = round(note.timestamp / dominant_ioi)
            grid.setdefault(grid_pos, []).append(note)

        # Find grid positions that have the same note content
        # A "pattern unit" is the set of (string, fret, midi) at a grid position
        def _note_signature(notes_at_pos):
            return frozenset((n.string, n.fret, n.midi_note) for n in notes_at_pos)

        sig_to_positions: dict[frozenset, list[int]] = {}
        for pos, notes_at in grid.items():
            sig = _note_signature(notes_at)
            sig_to_positions.setdefault(sig, []).append(pos)

        # Find signatures that repeat enough times
        for sig, positions in sig_to_positions.items():
            if len(positions) < 3:  # Need at least 3 repetitions
                continue

            # Check if positions are roughly evenly spaced
            sorted_pos = sorted(positions)
            gaps = [sorted_pos[i + 1] - sorted_pos[i] for i in range(len(sorted_pos) - 1)]
            if not gaps:
                continue

            # Most gaps should be 1 (consecutive grid positions)
            gap_counts = Counter(gaps)
            dominant_gap = gap_counts.most_common(1)[0][0]

            if dominant_gap < 1 or dominant_gap > 3:
                continue

            # Build the pattern template
            sample_notes = grid[sorted_pos[0]]
            pattern_notes = [
                (0.0, n.string, n.fret if isinstance(n.fret, int) else 0, n.midi_note)
                for n in sample_notes
            ]

            observed_times = [pos * dominant_ioi for pos in sorted_pos]

            patterns.append(PatternTemplate(
                ioi=dominant_ioi * dominant_gap,
                notes=pattern_notes,
                observed_at=observed_times,
                repetitions=len(sorted_pos),
            ))

    logger.info(f"Temporal interpolation: detected {len(patterns)} patterns")
    return patterns


def interpolate_missing(
    tab_notes: list,
    patterns: list[PatternTemplate],
    config: TemporalInterpolationConfig,
) -> list:
    """Fill gaps where a pattern is established but notes are missing.

    For each pattern, finds grid positions where the pattern should appear
    but no notes were detected, and interpolates the missing notes.

    Args:
        tab_notes: Existing tab notes
        patterns: Detected patterns
        config: Interpolation configuration

    Returns:
        Tab notes augmented with interpolated notes
    """
    from app.fusion_engine import TabNote, get_confidence_level

    if not patterns:
        return tab_notes

    existing_timestamps = set(round(n.timestamp, 2) for n in tab_notes)
    added = []

    for pattern in patterns:
        if pattern.repetitions < config.min_repetitions:
            continue

        if not pattern.observed_at or pattern.ioi <= 0:
            continue

        # Find expected timestamps based on pattern
        min_t = min(pattern.observed_at)
        max_t = max(pattern.observed_at)

        # Check each expected position between first and last observed
        t = min_t
        while t <= max_t + pattern.ioi * 0.5:
            rounded_t = round(t, 2)

            # Is this a gap? (no note near this timestamp)
            has_note = any(
                abs(rounded_t - et) < pattern.ioi * config.timing_tolerance
                for et in existing_timestamps
            )

            if not has_note:
                # Check gap size (how many consecutive positions are missing)
                gap_size = 0
                check_t = t
                while check_t <= max_t:
                    check_rounded = round(check_t, 2)
                    if any(abs(check_rounded - et) < pattern.ioi * config.timing_tolerance
                           for et in existing_timestamps):
                        break
                    gap_size += 1
                    check_t += pattern.ioi

                if gap_size <= config.max_gap_units:
                    # Interpolate the pattern notes at this timestamp
                    for rel_time, string, fret, midi_note in pattern.notes:
                        new_note = TabNote(
                            id=str(uuid4()),
                            timestamp=t + rel_time,
                            string=string,
                            fret=fret,
                            confidence=config.interpolated_confidence,
                            confidence_level=get_confidence_level(config.interpolated_confidence),
                            midi_note=midi_note,
                            is_part_of_chord=len(pattern.notes) > 1,
                            chord_id=str(uuid4()) if len(pattern.notes) > 1 else None,
                        )
                        added.append(new_note)

            t += pattern.ioi

    if added:
        logger.info(f"Temporal interpolation: added {len(added)} notes")

    return tab_notes + added
