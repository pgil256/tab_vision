"""Spectral residual analysis for detecting missed polyphonic notes.

Computes a CQT spectrogram, subtracts the energy from already-detected
notes, and looks for residual peaks at timestamps where the primary
detector appears to have missed notes.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from app.audio_pipeline import DetectedNote, GUITAR_MIDI_MIN, GUITAR_MIDI_MAX, group_notes_into_chords

logger = logging.getLogger(__name__)


@dataclass
class SpectralResidualConfig:
    """Configuration for spectral residual analysis."""
    enabled: bool = False
    # CQT resolution (higher = finer pitch resolution but slower)
    bins_per_octave: int = 36
    # Minimum residual energy to consider as a note (relative to max)
    residual_threshold: float = 0.15
    # Window around onset times to search for residual peaks
    onset_window: float = 0.1
    # Minimum confidence for residual-detected notes
    min_residual_confidence: float = 0.45
    # Only search at deficit timestamps (like ensemble)
    deficit_only: bool = True


def analyze_spectral_residual(
    audio_path: str,
    detected_notes: list[DetectedNote],
    config: SpectralResidualConfig,
) -> list[DetectedNote]:
    """Analyze spectral residual to find missed notes.

    Computes a CQT, subtracts harmonics of detected notes, and looks for
    remaining peaks that might be missed polyphonic notes.

    Args:
        audio_path: Path to WAV audio file
        detected_notes: Already-detected notes
        config: Spectral residual configuration

    Returns:
        Original notes plus any newly detected residual notes
    """
    if not config.enabled:
        return detected_notes

    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not installed, skipping spectral residual analysis")
        return detected_notes

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Compute CQT
    n_bins = config.bins_per_octave * 4  # ~4 octaves for guitar range
    fmin = librosa.midi_to_hz(GUITAR_MIDI_MIN)

    C = np.abs(librosa.cqt(
        y, sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=config.bins_per_octave,
        hop_length=512,
    ))

    hop_time = 512 / sr
    n_frames = C.shape[1]

    # Build a mask of energy to subtract (from detected notes)
    mask = np.zeros_like(C)
    for note in detected_notes:
        # Convert MIDI to CQT bin
        midi_offset = note.midi_note - GUITAR_MIDI_MIN
        bin_idx = int(round(midi_offset * config.bins_per_octave / 12))
        if bin_idx < 0 or bin_idx >= n_bins:
            continue

        start_frame = int(note.start_time / hop_time)
        end_frame = int(note.end_time / hop_time)
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames - 1))

        # Mark the fundamental and first few harmonics
        for harmonic in range(1, 4):
            h_bin = bin_idx + int(round(config.bins_per_octave * np.log2(harmonic)))
            if 0 <= h_bin < n_bins:
                # Spread across a few bins for tolerance
                for offset in range(-1, 2):
                    b = h_bin + offset
                    if 0 <= b < n_bins:
                        mask[b, start_frame:end_frame + 1] = 1.0

    # Compute residual
    residual = C * (1.0 - mask)

    # Find deficit timestamps (same approach as ensemble)
    if config.deficit_only:
        chords = group_notes_into_chords(detected_notes, tolerance=0.05)
        chord_sizes = [len(c) for c in chords]
        deficit_timestamps = []

        for i, chord in enumerate(chords):
            window = chord_sizes[max(0, i - 3):i + 4]
            if not window:
                continue
            typical_size = sorted(window)[len(window) // 2]
            if len(chord) < typical_size:
                deficit_timestamps.append(chord[0].start_time)

        if not deficit_timestamps:
            logger.info("Spectral residual: no deficit timestamps found")
            return detected_notes
    else:
        # Search all onset times
        deficit_timestamps = list(set(n.start_time for n in detected_notes))

    # Search for residual peaks at deficit timestamps
    new_notes = []
    max_residual = residual.max() if residual.max() > 0 else 1.0

    for timestamp in deficit_timestamps:
        center_frame = int(timestamp / hop_time)
        start_frame = max(0, center_frame - int(config.onset_window / hop_time))
        end_frame = min(n_frames - 1, center_frame + int(config.onset_window / hop_time))

        if start_frame >= end_frame:
            continue

        # Get residual energy in this window
        window_residual = residual[:, start_frame:end_frame + 1]
        avg_residual = window_residual.mean(axis=1)

        # Find peaks above threshold
        threshold = config.residual_threshold * max_residual
        for bin_idx in range(n_bins):
            if avg_residual[bin_idx] > threshold:
                # Convert bin to MIDI note
                midi_note = GUITAR_MIDI_MIN + int(round(bin_idx * 12 / config.bins_per_octave))

                if midi_note < GUITAR_MIDI_MIN or midi_note > GUITAR_MIDI_MAX:
                    continue

                # Check if this note is already detected
                is_duplicate = False
                for existing in detected_notes:
                    if (existing.midi_note == midi_note and
                            abs(existing.start_time - timestamp) < config.onset_window):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Also check against already-found residual notes
                    for rn in new_notes:
                        if (rn.midi_note == midi_note and
                                abs(rn.start_time - timestamp) < config.onset_window):
                            is_duplicate = True
                            break

                if not is_duplicate:
                    # Estimate note duration from residual decay
                    note_end_frame = end_frame
                    for f in range(end_frame + 1, min(n_frames, end_frame + int(0.5 / hop_time))):
                        if residual[bin_idx, f] < threshold * 0.5:
                            note_end_frame = f
                            break
                        note_end_frame = f

                    confidence = min(1.0, float(avg_residual[bin_idx] / max_residual))
                    if confidence >= config.min_residual_confidence:
                        new_notes.append(DetectedNote(
                            start_time=timestamp,
                            end_time=note_end_frame * hop_time,
                            midi_note=midi_note,
                            confidence=confidence,
                            amplitude=confidence,
                        ))

    logger.info(f"Spectral residual: found {len(new_notes)} additional notes")
    return detected_notes + new_notes
