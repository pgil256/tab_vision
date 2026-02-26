"""Audio pipeline for extracting audio and detecting pitch from video."""
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectedNote:
    """A note detected by pitch analysis."""
    start_time: float   # seconds
    end_time: float     # seconds
    midi_note: int      # MIDI note number (e.g., 69 = A4)
    confidence: float   # 0.0-1.0 from pitch detector
    amplitude: float = 0.0  # Raw amplitude from pitch detector
    pitch_bend: float = 0.0  # Pitch bend/vibrato amount

    @property
    def duration(self) -> float:
        """Note duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class AudioAnalysisConfig:
    """Configuration for audio analysis parameters."""
    # Minimum confidence threshold for note detection
    min_confidence: float = 0.3
    # Minimum note duration in seconds (filter out spurious detections)
    min_note_duration: float = 0.03
    # Maximum gap between notes to merge (for legato playing)
    merge_gap_threshold: float = 0.02
    # Minimum amplitude threshold
    min_amplitude: float = 0.1
    # Whether to filter harmonics/overtones
    filter_harmonics: bool = True
    # Guitar frequency range (E2 to ~E6 with some headroom)
    min_frequency_hz: float = 80.0   # ~E2
    max_frequency_hz: float = 1400.0  # ~F6 (high frets on high E)
    # Onset detection refinement window (seconds)
    onset_refinement_window: float = 0.05
    # Sustain re-detection filtering
    sustain_redetection_window: float = 0.6  # seconds
    sustain_amplitude_ratio: float = 0.95    # remove if amp < this * original
    # Harmonics filtering (improved)
    harmonic_time_tolerance: float = 0.15    # seconds (wider than old 0.05)
    harmonic_amplitude_ratio: float = 0.7    # use amplitude, not confidence
    filter_sub_harmonics: bool = True        # also remove octave-below artifacts


# Guitar MIDI note range (E2 = 40 to about E6 = 88 for 24 frets on high E)
GUITAR_MIDI_MIN = 40  # E2 (low E string open)
GUITAR_MIDI_MAX = 88  # E6 (24th fret on high E string)


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio track from video as WAV file.

    Args:
        video_path: Path to input video file
        output_path: Path for output WAV file (mono, 22050 Hz)

    Returns:
        Path to the created WAV file

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Use ffmpeg to extract audio as mono WAV at 22050 Hz (what Basic Pitch expects)
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "22050",  # Sample rate
        "-ac", "1",  # Mono
        output_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg did not create output file")

    return output_path


def analyze_pitch(
    audio_path: str,
    config: Optional[AudioAnalysisConfig] = None
) -> list[DetectedNote]:
    """Analyze audio file for pitch content using Basic Pitch.

    Args:
        audio_path: Path to WAV audio file
        config: Optional analysis configuration

    Returns:
        List of DetectedNote objects, filtered and refined

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ImportError: If basic_pitch is not installed
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if config is None:
        config = AudioAnalysisConfig()

    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ImportError(
            "basic_pitch is not installed. "
            "Install with: pip install basic-pitch"
        )

    # Run Basic Pitch inference with optimized parameters for guitar
    # Using lower onset/frame thresholds to catch quieter notes
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=0.4,  # Lower for better onset detection
        frame_threshold=0.25,  # Lower to catch sustaining notes
        minimum_note_length=config.min_note_duration * 1000,  # Convert to ms
        minimum_frequency=config.min_frequency_hz,
        maximum_frequency=config.max_frequency_hz,
    )

    # Convert note_events to DetectedNote objects
    # note_events is a list of (start_time, end_time, pitch, amplitude, pitch_bends)
    raw_notes = []

    # First pass: collect amplitudes for normalization
    amplitudes = [event[3] for event in note_events if GUITAR_MIDI_MIN <= int(event[2]) <= GUITAR_MIDI_MAX]
    max_amplitude = max(amplitudes) if amplitudes else 1.0

    for event in note_events:
        start_time, end_time, midi_note, amplitude, pitch_bends = event
        midi_note = int(midi_note)

        # Skip notes outside guitar range
        if midi_note < GUITAR_MIDI_MIN or midi_note > GUITAR_MIDI_MAX:
            continue

        # Calculate pitch bend amount (for detecting bends/vibrato)
        pitch_bend = 0.0
        if pitch_bends is not None and len(pitch_bends) > 0:
            pitch_bend = float(max(abs(pb) for pb in pitch_bends))

        # Normalize amplitude to confidence score (0.0-1.0)
        # Basic Pitch amplitudes are typically 0.1-0.8 range, normalize to 0.3-1.0
        # This ensures even quieter but clear notes get reasonable confidence
        normalized_confidence = 0.3 + (float(amplitude) / max_amplitude) * 0.7

        raw_notes.append(DetectedNote(
            start_time=start_time,
            end_time=end_time,
            midi_note=midi_note,
            confidence=normalized_confidence,
            amplitude=float(amplitude),
            pitch_bend=pitch_bend,
        ))

    # Apply filtering pipeline
    filtered_notes = _filter_notes(raw_notes, config)

    # Merge closely spaced notes of the same pitch
    merged_notes = _merge_consecutive_notes(filtered_notes, config)

    # Filter sustain re-detections (same pitch repeated with lower amplitude)
    merged_notes = _filter_sustain_redetections(merged_notes, config)

    # Filter harmonics if enabled
    if config.filter_harmonics:
        merged_notes = _filter_harmonics(merged_notes, config)

    # Sort by start time
    merged_notes.sort(key=lambda n: n.start_time)

    return merged_notes


def _filter_notes(
    notes: list[DetectedNote],
    config: AudioAnalysisConfig
) -> list[DetectedNote]:
    """Filter notes based on confidence, duration, and amplitude thresholds.

    Args:
        notes: Raw detected notes
        config: Analysis configuration

    Returns:
        Filtered list of notes
    """
    filtered = []
    for note in notes:
        # Skip low confidence notes
        if note.confidence < config.min_confidence:
            continue

        # Skip very short notes (likely artifacts)
        if note.duration < config.min_note_duration:
            continue

        # Skip very quiet notes
        if note.amplitude < config.min_amplitude:
            continue

        filtered.append(note)

    return filtered


def _merge_consecutive_notes(
    notes: list[DetectedNote],
    config: AudioAnalysisConfig
) -> list[DetectedNote]:
    """Merge consecutive notes of the same pitch that are close together.

    This handles legato playing and minor detection gaps.

    Args:
        notes: Filtered notes
        config: Analysis configuration

    Returns:
        List with consecutive same-pitch notes merged
    """
    if not notes:
        return []

    # Sort by start time then pitch
    sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.midi_note))
    merged = []
    current = sorted_notes[0]

    for note in sorted_notes[1:]:
        # Check if same pitch and close enough to merge
        if (note.midi_note == current.midi_note and
            note.start_time - current.end_time <= config.merge_gap_threshold):
            # Merge: extend current note, average confidence
            avg_confidence = (current.confidence + note.confidence) / 2
            max_amplitude = max(current.amplitude, note.amplitude)
            max_bend = max(current.pitch_bend, note.pitch_bend)
            current = DetectedNote(
                start_time=current.start_time,
                end_time=note.end_time,
                midi_note=current.midi_note,
                confidence=avg_confidence,
                amplitude=max_amplitude,
                pitch_bend=max_bend,
            )
        else:
            merged.append(current)
            current = note

    merged.append(current)
    return merged


def _filter_sustain_redetections(
    notes: list[DetectedNote],
    config: AudioAnalysisConfig
) -> list[DetectedNote]:
    """Filter out sustain re-detections of the same pitch.

    When a note rings out, Basic Pitch may detect it as a new note onset.
    These re-detections have the same MIDI pitch but lower amplitude.

    Args:
        notes: Detected notes (sorted by start_time)
        config: Analysis configuration

    Returns:
        Notes with sustain re-detections removed
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    keep = []

    for i, note in enumerate(sorted_notes):
        is_redetection = False

        # Look back at previous notes with the same pitch
        for j in range(i - 1, -1, -1):
            prev = sorted_notes[j]
            time_gap = note.start_time - prev.start_time
            if time_gap > config.sustain_redetection_window:
                break
            if prev.midi_note != note.midi_note:
                continue
            # Same pitch, within window - check if this is a re-detection
            # The previous note should still be ringing (end_time near or past current start)
            if prev.end_time >= note.start_time - 0.1:
                if note.amplitude < prev.amplitude * config.sustain_amplitude_ratio:
                    is_redetection = True
                    break

        if not is_redetection:
            keep.append(note)

    return keep


def _filter_harmonics(notes: list[DetectedNote], config: Optional[AudioAnalysisConfig] = None) -> list[DetectedNote]:
    """Filter out likely harmonic overtones and sub-harmonic artifacts.

    When a guitar string is plucked, it produces harmonics at integer multiples
    of the fundamental frequency. Basic Pitch may detect these as separate notes.
    Also filters sub-harmonics (e.g., octave below a louder note) which are
    low-frequency artifacts from the pitch detector.

    Args:
        notes: Detected notes
        config: Analysis configuration (uses defaults if None)

    Returns:
        Notes with likely harmonics removed
    """
    if not notes:
        return []

    time_tolerance = config.harmonic_time_tolerance if config else 0.15
    amp_ratio = config.harmonic_amplitude_ratio if config else 0.7
    check_sub = config.filter_sub_harmonics if config else True

    # Harmonic intervals in semitones (octave, octave+fifth, 2 octaves)
    harmonic_intervals = [12, 19, 24, 28, 31]  # 1st through 5th harmonics

    filtered = []

    for note in notes:
        is_harmonic = False

        for other in notes:
            if other is note:
                continue

            # Check if notes are roughly simultaneous
            if abs(note.start_time - other.start_time) > time_tolerance:
                continue

            # Check if this note is a harmonic ABOVE the other
            interval = note.midi_note - other.midi_note
            if interval in harmonic_intervals:
                if note.amplitude < other.amplitude * amp_ratio:
                    is_harmonic = True
                    break

            # Check if this note is a sub-harmonic BELOW the other
            if check_sub:
                interval = other.midi_note - note.midi_note
                if interval in harmonic_intervals:
                    if note.amplitude < other.amplitude * amp_ratio:
                        is_harmonic = True
                        break

        if not is_harmonic:
            filtered.append(note)

    return filtered


def detect_note_onsets(
    notes: list[DetectedNote],
    audio_path: Optional[str] = None
) -> list[float]:
    """Extract onset timestamps from detected notes.

    Args:
        notes: Detected notes
        audio_path: Optional path to audio for onset refinement

    Returns:
        List of onset timestamps in seconds
    """
    # Group notes into chords (notes starting within 50ms)
    CHORD_TOLERANCE = 0.05
    onsets = []

    if not notes:
        return onsets

    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    current_chord_time = sorted_notes[0].start_time
    chord_notes = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        if note.start_time - current_chord_time <= CHORD_TOLERANCE:
            # Part of current chord
            chord_notes.append(note)
        else:
            # New chord - record onset of previous chord
            # Use confidence-weighted average of onset times
            total_conf = sum(n.confidence for n in chord_notes)
            if total_conf > 0:
                weighted_onset = sum(
                    n.start_time * n.confidence for n in chord_notes
                ) / total_conf
            else:
                weighted_onset = chord_notes[0].start_time
            onsets.append(weighted_onset)

            # Start new chord
            current_chord_time = note.start_time
            chord_notes = [note]

    # Don't forget the last chord
    if chord_notes:
        total_conf = sum(n.confidence for n in chord_notes)
        if total_conf > 0:
            weighted_onset = sum(
                n.start_time * n.confidence for n in chord_notes
            ) / total_conf
        else:
            weighted_onset = chord_notes[0].start_time
        onsets.append(weighted_onset)

    return onsets


def group_notes_into_chords(
    notes: list[DetectedNote],
    tolerance: float = 0.05
) -> list[list[DetectedNote]]:
    """Group simultaneous notes into chords.

    Args:
        notes: List of detected notes
        tolerance: Maximum time difference for notes to be in same chord

    Returns:
        List of chord groups (each group is a list of notes)
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    chords = []
    current_chord = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        if note.start_time - current_chord[0].start_time <= tolerance:
            current_chord.append(note)
        else:
            chords.append(current_chord)
            current_chord = [note]

    if current_chord:
        chords.append(current_chord)

    return chords
