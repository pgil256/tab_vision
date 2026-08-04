"""Fast local take review and deterministic FFmpeg cleanup."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tabvision.errors import BackendError, InvalidInputError

REVIEW_SAMPLE_RATE = 22_050
NORMALIZE_PEAK = 0.95


def analyze_take(path: str | Path, *, bins: int = 600) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise InvalidInputError(f"recording not found: {source}")
    if bins < 16:
        raise InvalidInputError("waveform bins must be at least 16")
    wav = _decode(source)
    duration = len(wav) / REVIEW_SAMPLE_RATE
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    clipped = np.abs(wav) >= 0.995
    clipped_samples = int(np.count_nonzero(clipped))
    clipped_runs = int(np.count_nonzero(clipped & ~np.r_[False, clipped[:-1]]))
    trim_start, trim_end = _auto_trim(wav, peak)
    tuning_cents, tuning_confidence, voiced_frames = _estimate_tuning(wav)
    minimums, maximums = _waveform_peaks(wav, bins)
    return {
        "duration": duration,
        "sample_rate": REVIEW_SAMPLE_RATE,
        "peak": peak,
        "clipped_samples": clipped_samples,
        "clipped_runs": clipped_runs,
        "auto_trim_start": trim_start,
        "auto_trim_end": trim_end,
        "tuning_cents": tuning_cents,
        "tuning_confidence": tuning_confidence,
        "voiced_frames": voiced_frames,
        "waveform_min": minimums,
        "waveform_max": maximums,
    }


def clean_take(
    source_path: str | Path,
    output_path: str | Path,
    *,
    trim_start: float = 0,
    trim_end: float | None = None,
    gain_db: float = 0,
    normalize: bool = False,
    highpass_hz: int = 0,
) -> Path:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise InvalidInputError(f"recording not found: {source}")
    if not math.isfinite(trim_start) or trim_start < 0:
        raise InvalidInputError("trim start must be finite and non-negative")
    if trim_end is not None and (
        not math.isfinite(trim_end) or trim_end <= trim_start + 0.25
    ):
        raise InvalidInputError("trim end must keep at least 0.25 seconds")
    if not math.isfinite(gain_db) or not -24 <= gain_db <= 24:
        raise InvalidInputError("gain must be between -24 and +24 dB")
    if highpass_hz not in {0, 60, 80, 100, 120}:
        raise InvalidInputError("high-pass must be 0, 60, 80, 100, or 120 Hz")
    if not shutil.which("ffmpeg"):
        raise BackendError("ffmpeg not on PATH; required for take cleanup")

    filters = [f"atrim=start={trim_start:.6f}" + (f":end={trim_end:.6f}" if trim_end else "")]
    filters.append("asetpts=N/SR/TB")
    if highpass_hz:
        filters.append(f"highpass=f={highpass_hz}")
    gain = 10 ** (gain_db / 20)
    if normalize:
        wav = _decode(source)
        start_index = min(len(wav), max(0, int(trim_start * REVIEW_SAMPLE_RATE)))
        end_index = (
            len(wav)
            if trim_end is None
            else min(len(wav), max(start_index + 1, int(trim_end * REVIEW_SAMPLE_RATE)))
        )
        kept_peak = (
            float(np.max(np.abs(wav[start_index:end_index])))
            if end_index > start_index
            else 0
        )
        if kept_peak > 0:
            gain *= NORMALIZE_PEAK / (kept_peak * gain)
    if abs(gain - 1) > 1e-5:
        filters.append(f"volume={gain:.8f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise BackendError(f"take cleanup failed: {result.stderr.decode(errors='replace').strip()}")
    if not output.is_file() or output.stat().st_size == 0:
        raise BackendError("take cleanup produced no audio")
    return output


def _decode(path: Path) -> np.ndarray:
    from tabvision.demux import _extract_audio

    return np.asarray(_extract_audio(path, REVIEW_SAMPLE_RATE), dtype=np.float32)


def _auto_trim(wav: np.ndarray, peak: float) -> tuple[float, float]:
    if wav.size == 0:
        return 0.0, 0.0
    threshold = max(0.004, peak * 0.025)
    active = np.flatnonzero(np.abs(wav) >= threshold)
    duration = len(wav) / REVIEW_SAMPLE_RATE
    if active.size == 0:
        return 0.0, duration
    padding = int(0.12 * REVIEW_SAMPLE_RATE)
    start = max(0, int(active[0]) - padding) / REVIEW_SAMPLE_RATE
    end = min(len(wav), int(active[-1]) + padding + 1) / REVIEW_SAMPLE_RATE
    if end - start < 0.25:
        return 0.0, duration
    return float(start), float(end)


def _waveform_peaks(wav: np.ndarray, bins: int) -> tuple[list[float], list[float]]:
    if wav.size == 0:
        return [0.0] * bins, [0.0] * bins
    edges = np.linspace(0, len(wav), bins + 1, dtype=int)
    minimums: list[float] = []
    maximums: list[float] = []
    for index in range(bins):
        chunk = wav[edges[index] : max(edges[index] + 1, edges[index + 1])]
        minimums.append(float(np.min(chunk)))
        maximums.append(float(np.max(chunk)))
    return minimums, maximums


def _estimate_tuning(wav: np.ndarray) -> tuple[float | None, float, int]:
    frame_size = 4096
    hop = 2048
    cents: list[float] = []
    candidate_frames = 0
    for start in range(0, max(0, len(wav) - frame_size + 1), hop):
        frame = wav[start : start + frame_size].astype(np.float64)
        frame -= np.mean(frame)
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < 0.008:
            continue
        candidate_frames += 1
        spectrum = np.fft.rfft(frame, n=frame_size * 2)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum))[:frame_size]
        min_lag = max(2, REVIEW_SAMPLE_RATE // 1100)
        max_lag = min(frame_size - 1, REVIEW_SAMPLE_RATE // 65)
        if max_lag <= min_lag or correlation[0] <= 0:
            continue
        lag = int(np.argmax(correlation[min_lag : max_lag + 1])) + min_lag
        confidence = float(correlation[lag] / correlation[0])
        if confidence < 0.55:
            continue
        frequency = REVIEW_SAMPLE_RATE / lag
        midi = 69 + 12 * math.log2(frequency / 440)
        cents.append((midi - round(midi)) * 100)
    if not cents:
        return None, 0.0, 0
    return float(np.median(cents)), len(cents) / max(1, candidate_frames), len(cents)


__all__ = ["NORMALIZE_PEAK", "analyze_take", "clean_take"]
