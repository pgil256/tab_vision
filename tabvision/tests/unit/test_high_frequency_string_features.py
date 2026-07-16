from __future__ import annotations

import numpy as np
import pytest

from tabvision.eval.high_frequency_string import (
    FEATURE_NAMES,
    extract_high_frequency_features,
    extract_native_window,
)


def _tone(sample_rate: int, frequency_hz: float, duration_s: float = 1.0) -> np.ndarray:
    times = np.arange(round(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency_hz * times).astype(np.float32)


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_native_window_preserves_supported_rate_and_zero_pads_boundary(sample_rate: int) -> None:
    wav = np.ones(round(0.2 * sample_rate), dtype=np.float32)

    window = extract_native_window(wav, sample_rate, 0.0)

    assert len(window) == round(0.512 * sample_rate)
    assert np.all(window[: round(0.064 * sample_rate)] == 0.0)
    assert np.all(window[round(0.064 * sample_rate) : round(0.264 * sample_rate)] == 1.0)


def test_native_window_rejects_16khz_backend_signal() -> None:
    with pytest.raises(ValueError, match="44.1/48 kHz"):
        extract_native_window(np.zeros(16_000, dtype=np.float32), 16_000, 0.5)


def test_feature_vector_is_finite_fixed_and_deterministic() -> None:
    wav = _tone(44_100, 329.63)

    first = extract_high_frequency_features(wav, 44_100, 0.2, 64)
    second = extract_high_frequency_features(wav, 44_100, 0.2, 64)

    assert first.shape == (len(FEATURE_NAMES),)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_pick_noise_descriptor_responds_to_native_high_frequency_energy() -> None:
    sample_rate = 44_100
    low = _tone(sample_rate, 440.0)
    high = low.copy()
    start = round(0.2 * sample_rate)
    stop = start + round(0.064 * sample_rate)
    times = np.arange(stop - start, dtype=np.float64) / sample_rate
    high[start:stop] += (0.5 * np.sin(2.0 * np.pi * 8_000.0 * times)).astype(np.float32)
    index = FEATURE_NAMES.index("onset_pick_noise_ratio")

    low_features = extract_high_frequency_features(low, sample_rate, 0.2, 69)
    high_features = extract_high_frequency_features(high, sample_rate, 0.2, 69)

    assert high_features[index] > low_features[index] + 0.05


def test_raw_rms_is_separate_from_normalized_harmonic_envelope() -> None:
    wav = _tone(44_100, 220.0)
    harmonic = FEATURE_NAMES.index("harmonic_share_attack_1_1")
    rms = FEATURE_NAMES.index("raw_log_rms_attack")

    quiet = extract_high_frequency_features(0.1 * wav, 44_100, 0.2, 57)
    loud = extract_high_frequency_features(0.8 * wav, 44_100, 0.2, 57)

    assert quiet[harmonic] == pytest.approx(loud[harmonic], abs=1.0e-6)
    assert loud[rms] > quiet[rms]
