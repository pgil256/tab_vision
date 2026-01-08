"""Audio pipeline for extracting audio and detecting pitch from video."""
import os
import subprocess
from dataclasses import dataclass


@dataclass
class DetectedNote:
    """A note detected by pitch analysis."""
    start_time: float   # seconds
    end_time: float     # seconds
    midi_note: int      # MIDI note number (e.g., 69 = A4)
    confidence: float   # 0.0-1.0 from pitch detector


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


def analyze_pitch(audio_path: str) -> list[DetectedNote]:
    """Analyze audio file for pitch content using Basic Pitch.

    Args:
        audio_path: Path to WAV audio file

    Returns:
        List of DetectedNote objects

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ImportError: If basic_pitch is not installed
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ImportError(
            "basic_pitch is not installed. "
            "Install with: pip install basic-pitch"
        )

    # Run Basic Pitch inference
    model_output, midi_data, note_events = predict(audio_path)

    # Convert note_events to DetectedNote objects
    # note_events is a list of (start_time, end_time, pitch, amplitude, pitch_bends)
    detected_notes = []
    for event in note_events:
        start_time, end_time, midi_note, amplitude, _ = event
        detected_notes.append(DetectedNote(
            start_time=start_time,
            end_time=end_time,
            midi_note=int(midi_note),
            confidence=float(amplitude),  # Use amplitude as confidence proxy
        ))

    return detected_notes
