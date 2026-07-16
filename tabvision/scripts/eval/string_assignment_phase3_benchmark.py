"""Reproducible 60-second benchmark for the registered Phase 3 ensemble."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.eval.string_assignment_phase3 import _resolve_ffmpeg
from tabvision.audio.highres_ensemble import DEFAULT_ENSEMBLE_ARTIFACT
from tabvision.pipeline import run_pipeline_with_artifacts
from tabvision.types import AudioEvent, SessionConfig, TabEvent


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _rss_bytes() -> int:
    if os.name == "nt":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        return int(counters.working_set_size)

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    maximum_rss = int(usage.ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def _event_hash(audio_events: tuple[AudioEvent, ...], tab_events: tuple[TabEvent, ...]) -> str:
    digest = hashlib.sha256()
    for audio_event in audio_events:
        digest.update(b"audio\0")
        digest.update(
            np.asarray(
                (
                    audio_event.onset_s,
                    audio_event.offset_s,
                    audio_event.velocity,
                    audio_event.confidence,
                ),
                dtype="<f8",
            ).tobytes()
        )
        digest.update(int(audio_event.pitch_midi).to_bytes(2, "little", signed=True))
        digest.update("\0".join(audio_event.tags).encode("utf-8"))
        if audio_event.pitch_logits is not None:
            digest.update(np.asarray(audio_event.pitch_logits, dtype="<f4").tobytes())
    for tab_event in tab_events:
        digest.update(b"tab\0")
        digest.update(
            np.asarray(
                (tab_event.onset_s, tab_event.duration_s, tab_event.confidence), dtype="<f8"
            ).tobytes()
        )
        digest.update(int(tab_event.string_idx).to_bytes(2, "little", signed=True))
        digest.update(int(tab_event.fret).to_bytes(2, "little", signed=True))
        digest.update(int(tab_event.pitch_midi).to_bytes(2, "little", signed=True))
    return digest.hexdigest()


def _exact_duration(wav: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    target = round(sample_rate * duration_s)
    if len(wav) < 1:
        raise ValueError("benchmark input is empty")
    repeats = (target + len(wav) - 1) // len(wav)
    return np.tile(wav, repeats)[:target].astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--duration-s", type=float, default=60.0)
    args = parser.parse_args()
    _resolve_ffmpeg()

    wav, sample_rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1, dtype=np.float32)
    benchmark_wav = _exact_duration(wav, sample_rate, args.duration_s)

    baseline_rss = _rss_bytes()
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop.wait(0.05):
            peak_rss = max(peak_rss, _rss_bytes())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            media_path = Path(temporary_directory) / "phase3-benchmark.wav"
            sf.write(media_path, benchmark_wav, sample_rate, subtype="PCM_16")
            started = time.perf_counter()
            artifacts = run_pipeline_with_artifacts(
                media_path,
                audio_backend_name="highres-ensemble",
                video_enabled=False,
                session=SessionConfig(instrument="acoustic", tone="clean"),
            )
            wall_seconds = time.perf_counter() - started
    finally:
        stop.set()
        sampler.join()
        peak_rss = max(peak_rss, _rss_bytes())

    payload = {
        "source": str(args.wav.resolve()),
        "duration_s": args.duration_s,
        "sample_rate": sample_rate,
        "audio_event_count": len(artifacts.audio_events),
        "tab_event_count": len(artifacts.tab_events),
        "wall_seconds": wall_seconds,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": peak_rss,
        "incremental_peak_rss_bytes": peak_rss - baseline_rss,
        "artifact_bytes": DEFAULT_ENSEMBLE_ARTIFACT.stat().st_size,
        "artifact_sha256": hashlib.sha256(DEFAULT_ENSEMBLE_ARTIFACT.read_bytes()).hexdigest(),
        "output_sha256": _event_hash(artifacts.audio_events, artifacts.tab_events),
        "audio_backend": "highres-ensemble",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
