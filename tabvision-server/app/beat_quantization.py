"""Beat quantization for snapping note timestamps to a rhythmic grid.

Uses librosa beat tracking to detect tempo, then snaps note timestamps
to the nearest subdivision (default: 16th note) for consistent timing.
High-confidence notes anchor the grid; lower-confidence notes snap harder.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from app.fusion_engine import TabNote

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for beat quantization."""
    enabled: bool = True
    # Subdivision: 4=quarter, 8=eighth, 16=sixteenth
    subdivision: int = 16
    # Maximum snap distance (seconds) — notes farther than this won't be snapped
    max_snap_distance: float = 0.08
    # Confidence-weighted snapping: high-confidence notes snap less
    # snap_strength ranges from 0 (no snap) to 1 (full snap)
    snap_strength_low: float = 1.0   # strength for low confidence notes
    snap_strength_high: float = 0.3  # strength for high confidence notes


def detect_tempo(audio_path: str) -> tuple[float, list[float]]:
    """Detect tempo and beat positions from audio.

    Args:
        audio_path: Path to WAV audio file

    Returns:
        Tuple of (tempo_bpm, beat_times) where beat_times is a list of
        beat positions in seconds
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=22050)

    # Use librosa beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Convert frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # tempo may be an ndarray with one element
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        tempo = float(tempo)

    logger.info(f"Detected tempo: {tempo:.1f} BPM, {len(beat_times)} beats")
    return tempo, list(beat_times)


def build_grid(
    tempo: float,
    beat_times: list[float],
    duration: float,
    subdivision: int = 16
) -> list[float]:
    """Build a quantization grid from detected beats.

    Creates evenly spaced grid points between beats at the specified
    subdivision level.

    Args:
        tempo: Detected tempo in BPM
        beat_times: Beat positions in seconds
        duration: Total audio duration in seconds
        subdivision: Grid subdivision (4, 8, or 16)

    Returns:
        Sorted list of grid positions in seconds
    """
    if not beat_times or len(beat_times) < 2:
        # Fallback: use tempo to generate uniform grid
        if tempo <= 0:
            tempo = 120.0
        beat_interval = 60.0 / tempo
        sub_interval = beat_interval / (subdivision / 4)
        grid = []
        t = 0.0
        while t <= duration:
            grid.append(t)
            t += sub_interval
        return grid

    # Number of subdivisions per beat (subdivision/4 since 4=quarter note=1 beat)
    subs_per_beat = subdivision // 4

    grid = set()

    # Add subdivisions between each pair of consecutive beats
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i + 1]
        for j in range(subs_per_beat):
            t = start + (end - start) * j / subs_per_beat
            grid.add(round(t, 6))

    # Add the last beat itself (loop above only adds up to, not including, the last beat)
    grid.add(round(beat_times[-1], 6))

    # Extend grid before first beat and after last beat
    if len(beat_times) >= 2:
        avg_interval = (beat_times[-1] - beat_times[0]) / (len(beat_times) - 1)
        sub_interval = avg_interval / subs_per_beat

        # Before first beat
        t = beat_times[0] - sub_interval
        while t >= 0:
            grid.add(round(t, 6))
            t -= sub_interval

        # After last beat
        t = beat_times[-1] + sub_interval
        while t <= duration + sub_interval:
            grid.add(round(t, 6))
            t += sub_interval

    return sorted(grid)


def quantize_notes(
    tab_notes: list[TabNote],
    audio_path: str,
    config: Optional[QuantizationConfig] = None,
) -> list[TabNote]:
    """Quantize note timestamps to the nearest beat subdivision.

    High-confidence notes snap less aggressively than low-confidence notes,
    so confident detections anchor the timing.

    Args:
        tab_notes: List of TabNote objects to quantize
        audio_path: Path to audio file for tempo detection
        config: Quantization configuration

    Returns:
        New list of TabNote objects with quantized timestamps
    """
    if not tab_notes:
        return tab_notes

    if config is None:
        config = QuantizationConfig()

    if not config.enabled:
        return tab_notes

    # Detect tempo and beats
    try:
        tempo, beat_times = detect_tempo(audio_path)
    except Exception as e:
        logger.warning(f"Beat detection failed: {e}, skipping quantization")
        return tab_notes

    # Build grid
    max_time = max(
        (n.end_time if n.end_time else n.timestamp for n in tab_notes),
        default=0
    )
    grid = build_grid(tempo, beat_times, max_time + 1.0, config.subdivision)

    if not grid:
        return tab_notes

    # Snap each note to nearest grid point
    quantized = []
    for note in tab_notes:
        # Find nearest grid point
        nearest_grid = min(grid, key=lambda g: abs(g - note.timestamp))
        distance = abs(nearest_grid - note.timestamp)

        if distance > config.max_snap_distance:
            # Too far from grid — don't snap
            quantized.append(note)
            continue

        # Confidence-weighted snap strength
        if note.confidence_level == "high":
            strength = config.snap_strength_high
        elif note.confidence_level == "low":
            strength = config.snap_strength_low
        else:
            strength = (config.snap_strength_high + config.snap_strength_low) / 2

        # Interpolate between original and grid position
        new_timestamp = note.timestamp + (nearest_grid - note.timestamp) * strength

        # Compute new end_time maintaining duration
        new_end_time = note.end_time
        if note.end_time is not None:
            duration = note.end_time - note.timestamp
            new_end_time = new_timestamp + duration

        # Create new TabNote with quantized timestamp
        quantized.append(TabNote(
            id=note.id,
            timestamp=round(new_timestamp, 6),
            string=note.string,
            fret=note.fret,
            confidence=note.confidence,
            confidence_level=note.confidence_level,
            midi_note=note.midi_note,
            end_time=round(new_end_time, 6) if new_end_time is not None else None,
            technique=note.technique,
            is_part_of_chord=note.is_part_of_chord,
            chord_id=note.chord_id,
            video_matched=note.video_matched,
            audio_confidence=note.audio_confidence,
            video_confidence=note.video_confidence,
            pitch_bend=note.pitch_bend,
        ))

    snapped_count = sum(
        1 for orig, quant in zip(tab_notes, quantized)
        if orig.timestamp != quant.timestamp
    )
    logger.info(
        f"Quantized {snapped_count}/{len(tab_notes)} notes "
        f"(tempo={tempo:.0f} BPM, grid={config.subdivision}th notes)"
    )

    return quantized
