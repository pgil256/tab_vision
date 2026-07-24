"""Accuracy-loop N1 — where does the physics channel lose coverage?

The inharmonicity channel is 0.92 accurate on the notes it fires on but only
reaches 8-10% of detections, so coverage is the binding constraint on its
value. The full-dev run showed 52,741 events -> 9,927 isolated -> 4,407
fitted, i.e. isolation removes 81% and the fit removes a further 56% of what
survives. "The fit fails" is too coarse to act on.

This counts every rejection reason separately, and records the distribution of
the quantities the thresholds test, so the next change targets a measured loss
rather than a guessed one. Pure diagnosis: it changes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import DEV_PLAYERS, _event_from_json
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import (
    MAX_WINDOW_S,
    MIN_PARTIALS,
    MIN_WINDOW_S,
    SKIP_ATTACK_S,
    _find_partials,
    _fit,
    _isolated_flags,
)
from tabvision.fusion.string_physics import load_string_evidence
from tabvision.types import GuitarConfig

ZERO_PAD = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=60)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_full_dev_cache")

    cfg = GuitarConfig()
    evidence = load_string_evidence()
    tracks = sorted(
        p.stem for p in (data_home / "annotation").glob("*.jams") if p.stem[:2] in DEV_PLAYERS
    )[: args.clips]

    reasons: Counter[str] = Counter()
    durations_short: list[float] = []
    partial_counts: list[int] = []
    r2_values: list[float] = []
    r2_of_rejected: list[float] = []

    for track_id in tracks:
        cache = cache_dir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            continue
        events = sorted(
            (_event_from_json(x) for x in json.loads(cache.read_text("utf-8"))),
            key=lambda e: e.onset_s,
        )
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        audio = np.asarray(wav, dtype=np.float64)
        isolated = _isolated_flags(events)

        for event, is_iso in zip(events, isolated, strict=True):
            reasons["total"] += 1
            if len(candidate_positions(event.pitch_midi, cfg)) < 2:
                reasons["unambiguous_pitch"] += 1
                continue
            if not is_iso:
                reasons["not_isolated"] += 1
                continue
            duration = event.offset_s - event.onset_s
            if duration < MIN_WINDOW_S + SKIP_ATTACK_S:
                reasons["too_short"] += 1
                durations_short.append(duration)
                continue
            start = int((event.onset_s + SKIP_ATTACK_S) * sr)
            stop = start + int(min(MAX_WINDOW_S, duration - SKIP_ATTACK_S) * sr)
            if start < 0 or stop > audio.size:
                reasons["out_of_bounds"] += 1
                continue

            segment = audio[start:stop]
            if not np.any(np.abs(segment) > 0.0):
                reasons["silent"] += 1
                continue
            windowed = segment * np.hanning(segment.size)
            n_fft = int(2 ** math.ceil(math.log2(segment.size * ZERO_PAD)))
            spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
            peak = float(spectrum.max())
            if peak <= 0.0:
                reasons["silent"] += 1
                continue
            freqs_per_bin = sr / n_fft
            noise_floor = max(float(np.median(spectrum)) * 4.0, peak * 1e-4)
            nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)

            ks, measured = _find_partials(spectrum, freqs_per_bin, nominal, 0.0, sr, noise_floor)
            partial_counts.append(len(ks))
            if len(ks) < MIN_PARTIALS:
                reasons["too_few_partials"] += 1
                continue
            fitted = _fit(ks, measured)
            if fitted is None:
                reasons["fit_failed"] += 1
                continue
            r2 = fitted[2]
            r2_values.append(r2)
            if r2 < evidence.min_r2:
                reasons["low_r2"] += 1
                r2_of_rejected.append(r2)
                continue
            reasons["applied"] += 1

    total = reasons["total"] or 1
    summary: dict[str, Any] = {
        "clips": len(tracks),
        "min_r2": evidence.min_r2,
        "reasons": dict(reasons),
        "share_of_total": {k: v / total for k, v in reasons.items() if k != "total"},
        "partial_count_histogram": dict(sorted(Counter(partial_counts).items())),
        "r2_quantiles_accepted": {
            str(q): float(np.quantile(r2_values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        if r2_values
        else {},
        "r2_quantiles_rejected": {
            str(q): float(np.quantile(r2_of_rejected, q)) for q in (0.25, 0.5, 0.75, 0.9)
        }
        if r2_of_rejected
        else {},
        "short_note_duration_quantiles": {
            str(q): float(np.quantile(durations_short, q)) for q in (0.25, 0.5, 0.75, 0.9)
        }
        if durations_short
        else {},
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{len(tracks)} clips, {reasons['total']} events\n")
    order = [
        "unambiguous_pitch",
        "not_isolated",
        "too_short",
        "out_of_bounds",
        "silent",
        "too_few_partials",
        "fit_failed",
        "low_r2",
        "applied",
    ]
    for key in order:
        count = reasons.get(key, 0)
        print(f"  {key:>20}: {count:6d}  ({count / total:6.2%})")
    print(f"\n  partial counts found: {summary['partial_count_histogram']}")
    print(f"  r2 of accepted: {summary['r2_quantiles_accepted']}")
    print(f"  r2 of low_r2 rejects: {summary['r2_quantiles_rejected']}")
    print(f"  duration of too_short: {summary['short_note_duration_quantiles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
