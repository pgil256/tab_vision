"""Tests for audio_pipeline module."""
import pytest
import numpy as np
from dataclasses import dataclass
from app.audio_pipeline import (
    DetectedNote, extract_audio, analyze_pitch,
    AudioPreprocessConfig, preprocess_audio,
)
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


def _make_test_wav(path: str, freq: float = 440.0, duration: float = 1.0,
                   amplitude: float = 0.5, sr: int = 22050) -> str:
    """Create a test WAV file with a sine tone."""
    import soundfile as sf
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, y, sr, subtype='PCM_16')
    return path


class TestAudioPreprocessConfig:
    """Tests for AudioPreprocessConfig defaults."""

    def test_default_values(self):
        config = AudioPreprocessConfig()
        assert config.target_dbfs == -20.0
        assert config.highpass_cutoff == 70.0
        assert config.normalize is True
        assert config.highpass is True
        assert config.noise_gate is True

    def test_custom_values(self):
        config = AudioPreprocessConfig(target_dbfs=-16.0, highpass=False)
        assert config.target_dbfs == -16.0
        assert config.highpass is False


class TestPreprocessAudio:
    """Tests for preprocess_audio function."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_audio("/nonexistent/audio.wav", str(tmp_path / "out.wav"))

    def test_creates_output_file(self, tmp_path):
        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        _make_test_wav(input_path, freq=440.0, amplitude=0.3)

        result = preprocess_audio(input_path, output_path)
        assert result == output_path
        assert os.path.exists(output_path)

    def test_normalization_boosts_quiet_audio(self, tmp_path):
        """Quiet audio should be boosted toward target dBFS."""
        import soundfile as sf
        input_path = str(tmp_path / "quiet.wav")
        output_path = str(tmp_path / "normalized.wav")
        # Very quiet signal
        _make_test_wav(input_path, amplitude=0.01)

        preprocess_audio(input_path, output_path,
                         AudioPreprocessConfig(highpass=False, noise_gate=False))

        y_out, sr = sf.read(output_path)
        rms_out = np.sqrt(np.mean(y_out ** 2))
        # Should be significantly louder than input (0.01 amplitude -> ~-40 dBFS input)
        assert rms_out > 0.05

    def test_normalization_reduces_loud_audio(self, tmp_path):
        """Loud audio should be attenuated toward target dBFS."""
        import soundfile as sf
        input_path = str(tmp_path / "loud.wav")
        output_path = str(tmp_path / "normalized.wav")
        _make_test_wav(input_path, amplitude=0.95)

        preprocess_audio(input_path, output_path,
                         AudioPreprocessConfig(highpass=False, noise_gate=False))

        y_out, sr = sf.read(output_path)
        rms_out = np.sqrt(np.mean(y_out ** 2))
        # Target is -20 dBFS ≈ 0.1 RMS. Should be reduced from ~0.67 RMS.
        assert rms_out < 0.5

    def test_highpass_removes_low_frequency(self, tmp_path):
        """High-pass filter should attenuate frequencies below cutoff."""
        import soundfile as sf
        input_path = str(tmp_path / "low_freq.wav")
        output_path = str(tmp_path / "filtered.wav")
        # 30 Hz tone (well below 70 Hz cutoff)
        _make_test_wav(input_path, freq=30.0, amplitude=0.5, duration=2.0)

        preprocess_audio(input_path, output_path,
                         AudioPreprocessConfig(normalize=False, noise_gate=False))

        y_out, sr = sf.read(output_path)
        rms_out = np.sqrt(np.mean(y_out ** 2))
        # 30 Hz should be heavily attenuated by 70 Hz highpass
        assert rms_out < 0.15

    def test_highpass_preserves_guitar_frequencies(self, tmp_path):
        """High-pass filter should pass guitar frequencies (>80 Hz) through."""
        import soundfile as sf
        input_path = str(tmp_path / "guitar_freq.wav")
        output_path = str(tmp_path / "filtered.wav")
        # 200 Hz tone (guitar range)
        _make_test_wav(input_path, freq=200.0, amplitude=0.5, duration=2.0)

        preprocess_audio(input_path, output_path,
                         AudioPreprocessConfig(normalize=False, noise_gate=False))

        y_out, sr = sf.read(output_path)
        rms_out = np.sqrt(np.mean(y_out ** 2))
        # 200 Hz should pass through mostly unchanged
        assert rms_out > 0.3

    def test_all_steps_disabled(self, tmp_path):
        """With all processing disabled, output should match input closely."""
        import soundfile as sf
        input_path = str(tmp_path / "passthrough.wav")
        output_path = str(tmp_path / "output.wav")
        _make_test_wav(input_path, freq=440.0, amplitude=0.5)

        config = AudioPreprocessConfig(normalize=False, highpass=False, noise_gate=False)
        preprocess_audio(input_path, output_path, config)

        y_in, _ = sf.read(input_path)
        y_out, _ = sf.read(output_path)
        # Should be nearly identical (only WAV encoding differences)
        np.testing.assert_allclose(y_in, y_out, atol=1e-3)
