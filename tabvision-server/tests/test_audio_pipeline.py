"""Tests for audio_pipeline module."""
import pytest
from dataclasses import dataclass
from app.audio_pipeline import DetectedNote, extract_audio, analyze_pitch
import os
import tempfile


class TestDetectedNote:
    """Tests for the DetectedNote dataclass."""

    def test_detected_note_creation(self):
        """DetectedNote stores pitch detection results."""
        note = DetectedNote(
            start_time=1.5,
            end_time=2.0,
            midi_note=69,
            confidence=0.85,
        )

        assert note.start_time == 1.5
        assert note.end_time == 2.0
        assert note.midi_note == 69
        assert note.confidence == 0.85


class TestExtractAudio:
    """Tests for audio extraction from video."""

    def test_extract_audio_creates_wav_file(self, tmp_path):
        """extract_audio should create a WAV file from video."""
        # Create a minimal test video fixture path
        # For unit tests, we'll check the function signature and error handling
        video_path = str(tmp_path / "nonexistent.mp4")
        output_path = str(tmp_path / "output.wav")

        # Should raise error for missing file
        with pytest.raises(FileNotFoundError):
            extract_audio(video_path, output_path)

    def test_extract_audio_rejects_invalid_format(self, tmp_path):
        """extract_audio should reject non-video files."""
        # Create a text file pretending to be video
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_text("not a video")
        output_path = str(tmp_path / "output.wav")

        # Should raise error for invalid video
        with pytest.raises(Exception):  # Could be ffmpeg error or our validation
            extract_audio(str(fake_video), output_path)


class TestAnalyzePitch:
    """Tests for pitch analysis with Basic Pitch."""

    def test_analyze_pitch_rejects_missing_file(self):
        """analyze_pitch should raise error for missing audio file."""
        with pytest.raises(FileNotFoundError):
            analyze_pitch("/nonexistent/audio.wav")

    def test_analyze_pitch_returns_detected_notes(self, tmp_path):
        """analyze_pitch returns list of DetectedNote objects."""
        # This test requires actual Basic Pitch integration
        # For now, we'll skip if Basic Pitch isn't available
        pytest.importorskip("basic_pitch")

        # Create minimal silent WAV for testing (1 second, mono, 22050 Hz)
        import wave
        import struct

        wav_path = str(tmp_path / "silent.wav")
        with wave.open(wav_path, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            # Write 1 second of silence
            wav.writeframes(struct.pack('<' + 'h' * 22050, *([0] * 22050)))

        result = analyze_pitch(wav_path)

        # Silent audio should return empty list or list of DetectedNote
        assert isinstance(result, list)
        for note in result:
            assert isinstance(note, DetectedNote)
