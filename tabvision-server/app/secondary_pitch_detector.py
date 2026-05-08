"""Secondary pitch detector using pYIN for ensemble detection.

Complements Basic Pitch by detecting notes that the primary detector
misses, particularly in polyphonic contexts where Basic Pitch struggles
with overlapping harmonics.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from app.audio_pipeline import DetectedNote, GUITAR_MIDI_MIN, GUITAR_MIDI_MAX, group_notes_into_chords

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for multi-model ensemble detection."""
    enabled: bool = True
    # Which secondary model to use
    secondary_model: str = "pyin"
    # Minimum confidence from secondary detector to consider
    secondary_min_confidence: float = 0.4
    # Time tolerance for merging (avoid duplicates)
    merge_tolerance: float = 0.03
    # Primary detector takes precedence on conflicts
    primary_takes_precedence: bool = True
    # Only add secondary notes at timestamps with chord deficits
    deficit_only: bool = True
    # Minimum expected chord size to detect deficit
    min_expected_chord_size: int = 2


def _detect_with_pyin(audio_path: str) -> list[DetectedNote]:
    """Run pYIN pitch detection on audio file.

    pYIN is better at monophonic/sparse polyphonic detection and can catch
    notes that Basic Pitch misses in dense polyphonic sections.

    Args:
        audio_path: Path to WAV audio file

    Returns:
        List of DetectedNote objects from pYIN
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not installed, skipping pYIN detection")
        return []

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Run pYIN for pitch detection
    # pYIN returns frame-level f0 estimates
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, sr=sr,
        fmin=librosa.midi_to_hz(GUITAR_MIDI_MIN),
        fmax=librosa.midi_to_hz(GUITAR_MIDI_MAX),
        frame_length=2048,
        hop_length=512,
    )

    if f0 is None:
        return []

    # Convert frame-level f0 to note events
    hop_time = 512 / sr
    notes = []
    current_note_start = None
    current_midi = None
    current_probs = []

    for i in range(len(f0)):
        if voiced_flag[i] and not np.isnan(f0[i]):
            midi = int(round(librosa.hz_to_midi(f0[i])))
            if GUITAR_MIDI_MIN <= midi <= GUITAR_MIDI_MAX:
                if current_midi == midi:
                    # Continue current note
                    current_probs.append(voiced_probs[i])
                else:
                    # New note - save previous if exists
                    if current_midi is not None and current_note_start is not None:
                        end_time = i * hop_time
                        avg_prob = sum(current_probs) / len(current_probs)
                        if end_time - current_note_start >= 0.03:  # min duration
                            notes.append(DetectedNote(
                                start_time=current_note_start,
                                end_time=end_time,
                                midi_note=current_midi,
                                confidence=float(avg_prob),
                                amplitude=float(avg_prob),
                            ))
                    current_note_start = i * hop_time
                    current_midi = midi
                    current_probs = [voiced_probs[i]]
            else:
                # Out of range - close current note
                if current_midi is not None and current_note_start is not None:
                    end_time = i * hop_time
                    avg_prob = sum(current_probs) / len(current_probs)
                    if end_time - current_note_start >= 0.03:
                        notes.append(DetectedNote(
                            start_time=current_note_start,
                            end_time=end_time,
                            midi_note=current_midi,
                            confidence=float(avg_prob),
                            amplitude=float(avg_prob),
                        ))
                current_midi = None
                current_note_start = None
                current_probs = []
        else:
            # Unvoiced - close current note
            if current_midi is not None and current_note_start is not None:
                end_time = i * hop_time
                avg_prob = sum(current_probs) / len(current_probs)
                if end_time - current_note_start >= 0.03:
                    notes.append(DetectedNote(
                        start_time=current_note_start,
                        end_time=end_time,
                        midi_note=current_midi,
                        confidence=float(avg_prob),
                        amplitude=float(avg_prob),
                    ))
            current_midi = None
            current_note_start = None
            current_probs = []

    # Close last note
    if current_midi is not None and current_note_start is not None:
        end_time = len(f0) * hop_time
        avg_prob = sum(current_probs) / len(current_probs)
        if end_time - current_note_start >= 0.03:
            notes.append(DetectedNote(
                start_time=current_note_start,
                end_time=end_time,
                midi_note=current_midi,
                confidence=float(avg_prob),
                amplitude=float(avg_prob),
            ))

    logger.info(f"pYIN detected {len(notes)} notes")
    return notes


def _find_chord_deficits(
    primary_notes: list[DetectedNote],
    min_expected: int = 2
) -> list[tuple[float, int]]:
    """Find timestamps where fewer notes than expected are detected.

    Uses the mode (most common) chord size across the entire piece as the
    expected size, rather than a local median. This prevents the detector
    from adapting to consistently missed notes in a section.

    Args:
        primary_notes: Notes from primary detector
        min_expected: Minimum expected notes per chord event

    Returns:
        List of (timestamp, deficit_count) pairs
    """
    from collections import Counter

    chords = group_notes_into_chords(primary_notes, tolerance=0.08)

    # Calculate typical chord sizes in the piece
    chord_sizes = [len(c) for c in chords]
    if not chord_sizes:
        return []

    # Use mode (most common chord size) across entire piece for multi-note chords
    multi_note_sizes = [s for s in chord_sizes if s >= min_expected]
    if multi_note_sizes:
        size_counts = Counter(multi_note_sizes)
        typical_size = size_counts.most_common(1)[0][0]
    else:
        typical_size = min_expected

    # Look for timestamps where chord size is below the typical size
    deficits = []
    for i, chord in enumerate(chords):
        if len(chord) < typical_size and len(chord) >= 1:
            deficit = typical_size - len(chord)
            timestamp = chord[0].start_time
            deficits.append((timestamp, deficit))

    return deficits


def _is_duplicate(
    note: DetectedNote,
    existing: list[DetectedNote],
    tolerance: float
) -> bool:
    """Check if a note duplicates an existing detection."""
    for other in existing:
        if other.midi_note == note.midi_note:
            if abs(other.start_time - note.start_time) < tolerance:
                return True
    return False


def detect_with_ensemble(
    audio_path: str,
    primary_notes: list[DetectedNote],
    config: EnsembleConfig,
) -> list[DetectedNote]:
    """Run secondary detector and merge results with primary notes.

    Only adds notes from the secondary detector at timestamps where
    the primary detector appears to have missed notes (chord deficits).

    Args:
        audio_path: Path to WAV audio file
        primary_notes: Notes from primary detector (Basic Pitch)
        config: Ensemble configuration

    Returns:
        Merged note list (primary + new secondary detections)
    """
    if not config.enabled:
        return primary_notes

    # Run secondary detector
    if config.secondary_model == "pyin":
        secondary_notes = _detect_with_pyin(audio_path)
    else:
        logger.warning(f"Unknown secondary model: {config.secondary_model}")
        return primary_notes

    if not secondary_notes:
        return primary_notes

    # Filter by confidence
    secondary_notes = [n for n in secondary_notes if n.confidence >= config.secondary_min_confidence]

    if config.deficit_only:
        # Only add at deficit timestamps
        deficits = _find_chord_deficits(primary_notes, config.min_expected_chord_size)
        if not deficits:
            logger.info("No chord deficits found, skipping ensemble merge")
            return primary_notes

        deficit_timestamps = [t for t, _ in deficits]

        # Filter secondary notes to deficit timestamps
        added = []
        for note in secondary_notes:
            # Check if this note is near a deficit timestamp
            near_deficit = any(
                abs(note.start_time - dt) < 0.3
                for dt in deficit_timestamps
            )
            if not near_deficit:
                continue

            # Skip duplicates
            if _is_duplicate(note, primary_notes, config.merge_tolerance):
                continue

            added.append(note)

        logger.info(
            f"Ensemble: {len(deficits)} deficits found, "
            f"adding {len(added)} notes from {config.secondary_model}"
        )
        return primary_notes + added
    else:
        # Add all non-duplicate secondary notes
        added = []
        for note in secondary_notes:
            if not _is_duplicate(note, primary_notes, config.merge_tolerance):
                added.append(note)

        logger.info(f"Ensemble: adding {len(added)} notes from {config.secondary_model}")
        return primary_notes + added
