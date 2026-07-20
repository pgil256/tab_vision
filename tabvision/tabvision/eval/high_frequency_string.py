"""Deterministic native-rate descriptors for adjacent-string timbre probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SUPPORTED_SAMPLE_RATES = (44_100, 48_000)
WINDOW_START_S = -0.064
WINDOW_END_S = 0.448
WINDOW_DURATION_S = WINDOW_END_S - WINDOW_START_S

HARMONIC_BANDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (3, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 256),
)
DECAY_BANDS: tuple[tuple[str, int, int], ...] = (
    ("low", 1, 4),
    ("mid", 5, 16),
    ("high", 17, 256),
)
SEGMENT_FFT_SIZES = {
    "pre": 4096,
    "attack": 4096,
    "short": 8192,
    "long": 16384,
}


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for segment in ("attack", "short", "long"):
        names.extend(f"harmonic_share_{segment}_{lower}_{upper}" for lower, upper in HARMONIC_BANDS)
    names.extend(f"harmonic_available_{lower}_{upper}" for lower, upper in HARMONIC_BANDS)
    for band, _lower, _upper in DECAY_BANDS:
        names.extend((f"decay_short_over_attack_{band}", f"decay_long_over_short_{band}"))
    names.extend(f"inharmonicity_{segment}" for segment in ("attack", "short", "long"))
    names.extend(
        (
            "onset_spectral_centroid",
            "onset_rolloff_85",
            "onset_rolloff_95",
            "onset_pick_noise_ratio",
            "onset_pick_noise_flux",
        )
    )
    names.extend(f"raw_log_rms_{segment}" for segment in ("pre", "attack", "short", "long"))
    names.extend(
        f"raw_spectral_slope_db_per_octave_{segment}"
        for segment in ("pre", "attack", "short", "long")
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()


@dataclass(frozen=True)
class _Spectrum:
    frequencies: np.ndarray
    magnitude: np.ndarray
    power: np.ndarray


def extract_native_window(wav: np.ndarray, sample_rate: int, onset_s: float) -> np.ndarray:
    """Extract the fixed -64/+448 ms window without resampling."""

    signal = np.asarray(wav, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"expected mono waveform, got shape {signal.shape}")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"native string-timbre features require 44.1/48 kHz audio, got {sample_rate}"
        )
    if not math.isfinite(onset_s):
        raise ValueError("onset_s must be finite")
    length = int(round(WINDOW_DURATION_S * sample_rate))
    start = int(round((onset_s + WINDOW_START_S) * sample_rate))
    stop = start + length
    output = np.zeros(length, dtype=np.float32)
    source_start = max(0, start)
    source_stop = min(len(signal), stop)
    if source_stop > source_start:
        destination_start = source_start - start
        output[destination_start : destination_start + source_stop - source_start] = signal[
            source_start:source_stop
        ]
    return output


def extract_high_frequency_features(
    wav: np.ndarray,
    sample_rate: int,
    onset_s: float,
    pitch_midi: int,
) -> np.ndarray:
    """Extract the fixed interpretable Phase 4 descriptor vector.

    Harmonic shares are amplitude-normalized within each time resolution.
    RMS and spectral slopes are intentionally appended in raw units so they
    remain distinct from the normalized harmonic envelope.
    """

    if not 0 <= int(pitch_midi) <= 127:
        raise ValueError("pitch_midi must be in [0, 127]")
    window = extract_native_window(wav, sample_rate, onset_s)
    segments = _segments(window, sample_rate)
    spectra = {
        name: _spectrum(segment, sample_rate, SEGMENT_FFT_SIZES[name])
        for name, segment in segments.items()
    }
    fundamental_hz = 440.0 * 2.0 ** ((int(pitch_midi) - 69) / 12.0)
    harmonic = {
        name: _harmonic_measurements(spectra[name], fundamental_hz)
        for name in ("attack", "short", "long")
    }

    values: list[float] = []
    for name in ("attack", "short", "long"):
        amplitudes, _deviations = harmonic[name]
        total = float(np.sum(amplitudes)) + 1.0e-12
        values.extend(
            float(np.sum(amplitudes[lower - 1 : upper])) / total for lower, upper in HARMONIC_BANDS
        )

    maximum_harmonic = min(256, int((sample_rate / 2.0 * 0.98) // fundamental_hz))
    values.extend(
        max(0, min(maximum_harmonic, upper) - lower + 1) / (upper - lower + 1)
        for lower, upper in HARMONIC_BANDS
    )

    for _band, lower, upper in DECAY_BANDS:
        attack_total = _harmonic_band_total(harmonic["attack"][0], lower, upper)
        short_total = _harmonic_band_total(harmonic["short"][0], lower, upper)
        long_total = _harmonic_band_total(harmonic["long"][0], lower, upper)
        values.extend(
            (
                _bounded_log_ratio(short_total, attack_total),
                _bounded_log_ratio(long_total, short_total),
            )
        )

    for name in ("attack", "short", "long"):
        amplitudes, deviations = harmonic[name]
        limit = min(32, len(amplitudes))
        weights = amplitudes[:limit]
        denominator = float(np.sum(weights))
        values.append(
            float(np.sum(weights * deviations[:limit]) / denominator) if denominator else 0.0
        )

    attack_spectrum = spectra["attack"]
    pre_spectrum = spectra["pre"]
    values.extend(
        (
            _spectral_centroid(attack_spectrum),
            _spectral_rolloff(attack_spectrum, 0.85),
            _spectral_rolloff(attack_spectrum, 0.95),
            _band_power_ratio(attack_spectrum, 6000.0, min(18_000.0, sample_rate / 2.0)),
            _bounded_log_ratio(
                _band_power(attack_spectrum, 6000.0, min(18_000.0, sample_rate / 2.0)),
                _band_power(pre_spectrum, 6000.0, min(18_000.0, sample_rate / 2.0)),
            ),
        )
    )

    values.extend(_log_rms(segments[name]) for name in ("pre", "attack", "short", "long"))
    values.extend(_spectral_slope(spectra[name]) for name in ("pre", "attack", "short", "long"))
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(FEATURE_NAMES),):
        raise AssertionError(
            f"feature shape drift: expected {(len(FEATURE_NAMES),)}, got {output.shape}"
        )
    if np.any(~np.isfinite(output)):
        raise ValueError("high-frequency feature extraction produced non-finite values")
    return output


def _segments(window: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    boundaries_s = (0.0, 0.064, 0.128, 0.256, 0.512)
    boundaries = [int(round(value * sample_rate)) for value in boundaries_s]
    return {
        "pre": window[boundaries[0] : boundaries[1]],
        "attack": window[boundaries[1] : boundaries[2]],
        "short": window[boundaries[2] : boundaries[3]],
        "long": window[boundaries[3] : boundaries[4]],
    }


def _spectrum(segment: np.ndarray, sample_rate: int, n_fft: int) -> _Spectrum:
    taper = np.hanning(len(segment)).astype(np.float64)
    scale = max(float(np.sum(taper)), 1.0)
    magnitude = np.abs(np.fft.rfft(np.asarray(segment, dtype=np.float64) * taper, n=n_fft)) / scale
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return _Spectrum(frequencies, magnitude, np.square(magnitude))


def _harmonic_measurements(
    spectrum: _Spectrum,
    fundamental_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = min(256, int((float(spectrum.frequencies[-1]) * 0.98) // fundamental_hz))
    amplitudes = np.zeros(256, dtype=np.float64)
    deviations = np.zeros(256, dtype=np.float64)
    bin_width = float(spectrum.frequencies[1] - spectrum.frequencies[0])
    for harmonic in range(1, maximum + 1):
        expected = harmonic * fundamental_hz
        radius = max(bin_width, 0.04 * fundamental_hz)
        selected = np.flatnonzero(
            (spectrum.frequencies >= expected - radius)
            & (spectrum.frequencies <= expected + radius)
        )
        if not len(selected):
            continue
        local_index = int(selected[np.argmax(spectrum.magnitude[selected])])
        amplitudes[harmonic - 1] = float(spectrum.magnitude[local_index])
        deviations[harmonic - 1] = abs(float(spectrum.frequencies[local_index]) - expected) / max(
            expected, 1.0
        )
    return amplitudes, deviations


def _harmonic_band_total(amplitudes: np.ndarray, lower: int, upper: int) -> float:
    return float(np.sum(amplitudes[lower - 1 : upper]))


def _bounded_log_ratio(numerator: float, denominator: float) -> float:
    return float(np.clip(math.log((numerator + 1.0e-12) / (denominator + 1.0e-12)), -12, 12))


def _spectral_centroid(spectrum: _Spectrum) -> float:
    total = float(np.sum(spectrum.power[1:]))
    if not total:
        return 0.0
    value = float(np.sum(spectrum.frequencies[1:] * spectrum.power[1:]) / total)
    return value / float(spectrum.frequencies[-1])


def _spectral_rolloff(spectrum: _Spectrum, proportion: float) -> float:
    cumulative = np.cumsum(spectrum.power)
    if not len(cumulative) or cumulative[-1] <= 0.0:
        return 0.0
    index = min(len(cumulative) - 1, int(np.searchsorted(cumulative, proportion * cumulative[-1])))
    return float(spectrum.frequencies[index] / spectrum.frequencies[-1])


def _band_power(spectrum: _Spectrum, lower_hz: float, upper_hz: float) -> float:
    selected = (spectrum.frequencies >= lower_hz) & (spectrum.frequencies <= upper_hz)
    return float(np.sum(spectrum.power[selected]))


def _band_power_ratio(spectrum: _Spectrum, lower_hz: float, upper_hz: float) -> float:
    numerator = _band_power(spectrum, lower_hz, upper_hz)
    denominator = _band_power(spectrum, 80.0, float(spectrum.frequencies[-1]))
    return numerator / (denominator + 1.0e-12)


def _log_rms(segment: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(np.square(segment, dtype=np.float64))))
    return math.log10(max(rms, 1.0e-8))


def _spectral_slope(spectrum: _Spectrum) -> float:
    selected = (
        (spectrum.frequencies >= 80.0)
        & (spectrum.frequencies <= min(18_000.0, float(spectrum.frequencies[-1])))
        & (spectrum.magnitude > 1.0e-12)
    )
    if np.sum(selected) < 2:
        return 0.0
    x = np.log2(spectrum.frequencies[selected] / 440.0)
    y = 20.0 * np.log10(spectrum.magnitude[selected])
    centered = x - np.mean(x)
    denominator = float(np.sum(np.square(centered)))
    if not denominator:
        return 0.0
    return float(np.clip(np.sum(centered * (y - np.mean(y))) / denominator, -100, 100))


__all__ = [
    "FEATURE_NAMES",
    "HARMONIC_BANDS",
    "SUPPORTED_SAMPLE_RATES",
    "WINDOW_DURATION_S",
    "WINDOW_END_S",
    "WINDOW_START_S",
    "extract_high_frequency_features",
    "extract_native_window",
]
