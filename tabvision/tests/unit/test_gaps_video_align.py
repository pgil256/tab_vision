"""Unit tests for the GAPS video<->audio offset aligner core.

Exercises the pure cross-correlation lag estimator
(:func:`scripts.acquire.gaps_video.onset_envelope_lag`) with synthetic
onset-strength envelopes at known shifts, pinning down the sign convention
``video_time = audio_time + lag`` without needing network or media files.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.acquire.gaps_video import onset_envelope_lag


def _spiky_envelope(n: int = 400, n_spikes: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    env = rng.normal(0.0, 0.1, size=n)
    env[rng.choice(n, n_spikes, replace=False)] += rng.uniform(2.0, 5.0, size=n_spikes)
    return env


def _delay(env: np.ndarray, d: int) -> np.ndarray:
    """Return a copy of ``env`` delayed by ``d`` frames (``out[n] = env[n-d]``)."""
    out = np.zeros_like(env)
    if d >= 0:
        out[d:] = env[: env.size - d] if d < env.size else 0.0
    else:
        out[: env.size + d] = env[-d:]
    return out


@pytest.mark.parametrize("delay", [0, 1, 7, 25, -1, -12])
def test_recovers_known_delay(delay: int) -> None:
    audio = _spiky_envelope()
    video = _delay(audio, delay)  # video is audio delayed by `delay` frames
    lag, _ratio = onset_envelope_lag(video, audio)
    assert lag == delay


def test_sign_convention_video_delayed_is_positive() -> None:
    # A positive lag must mean the video lags the audio: a feature at audio
    # frame n appears at video frame n + lag.
    audio = _spiky_envelope(seed=3)
    video = _delay(audio, 9)
    lag, _ = onset_envelope_lag(video, audio)
    assert lag == 9


def test_peak_ratio_sharp_for_clean_shift() -> None:
    audio = _spiky_envelope(seed=1)
    video = _delay(audio, 5)
    _lag, ratio = onset_envelope_lag(video, audio)
    # An exact shifted copy yields a far sharper peak than self-similar noise.
    assert ratio > 2.0


def test_peak_ratio_low_for_uncorrelated() -> None:
    audio = _spiky_envelope(seed=1)
    video = _spiky_envelope(seed=999)  # unrelated
    _lag, ratio = onset_envelope_lag(video, audio)
    assert ratio < 2.0


def test_empty_envelope_raises() -> None:
    with pytest.raises(ValueError):
        onset_envelope_lag(np.array([]), np.array([1.0, 2.0]))
