"""Accuracy-loop Q6 — the physics channel on *detected* notes.

Gates A and B classified strings on **gold** notes: true onsets, true
pitches, isolation computed from the reference. That is the right way to ask
"does the physics work", and it passed (0.9200 mono-mic at 66.6% coverage
against a 0.65 count-prior control). It is not what an integration would see.

At inference the channel gets the ensemble's *detected* stream: onsets up to
the 50 ms tolerance off, pitches sometimes wrong, and isolation decidable
only from other detections. Every one of those degrades a B estimate — the
analysis window may start late or early, and a note believed isolated may
have an undetected neighbour ringing through it.

This measures the drop, offline, on the 20-clip banked ensemble cache. It is
the step the A14 video lever skipped: per-note evidence that looked strong in
isolation and then failed to lift the decoder.

Calibration stays honest: per-string ``B0`` comes from **gold** measurements
of the *other four* players (the Gate B leave-one-player-out folds), never
from the clip being scored.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from scripts.eval.n2_muscriptor_merge import _event_from_json, select_clips
from scripts.eval.q6_gate_a import (
    LOG2,
    MAX_WINDOW_S,
    MIN_WINDOW_S,
    SKIP_ATTACK_S,
    candidates_for_pitch,
    collect_measurements,
    estimate_inharmonicity,
)
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.types import GuitarConfig

MATCH_TOLERANCE_S = 0.05
DEV_PLAYERS = ("00", "01", "02", "03", "04")


def calibrate(rows: list[dict[str, Any]], min_r2: float) -> dict[str, dict[int, float]]:
    """Per-held-out-player B0 tables from the other players' gold notes."""
    usable = [row for row in rows if row["r2"] >= min_r2]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_player[row["player"]].append(row)
    tables: dict[str, dict[int, float]] = {}
    for held_out in DEV_PLAYERS:
        train = [row for player, items in by_player.items() if player != held_out for row in items]
        table: dict[int, float] = {}
        for string in range(6):
            values = [
                row["log_b"] - (row["fret"] / 6.0) * LOG2
                for row in train
                if row["string"] == string
            ]
            if values:
                table[string] = float(np.median(values))
        tables[held_out] = table
    return tables


def popular_strings(rows: list[dict[str, Any]], held_out: str) -> dict[int, int]:
    """Count-prior control, fitted on the same folds."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        if row["player"] != held_out:
            counts[row["pitch"]][row["string"]] += 1
    return {pitch: counter.most_common(1)[0][0] for pitch, counter in counts.items()}


def probe(
    clips: list[str],
    *,
    data_home: Path,
    workdir: Path,
    tables: dict[str, dict[int, float]],
    gold_rows: list[dict[str, Any]],
    min_r2: float,
) -> dict[str, Any]:
    cfg = GuitarConfig()
    detected_total = eligible = fitted_ok = 0
    scored = correct = control_correct = 0
    per_mode: dict[str, list[int]] = {"solo": [0, 0], "comp": [0, 0]}

    for track_id in clips:
        cache = workdir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            continue
        events = sorted(
            (_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))),
            key=lambda e: e.onset_s,
        )
        gold = sorted(
            parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg),
            key=lambda e: e.onset_s,
        )
        audio, sr = sf.read(
            data_home / "audio_mono-mic" / f"{track_id}_mic.wav", dtype="float32", always_2d=True
        )
        player = track_id[:2]
        table = tables.get(player, {})
        control = popular_strings(gold_rows, player)
        mode = "solo" if track_id.endswith("_solo") else "comp"

        for event in events:
            detected_total += 1
            duration = event.offset_s - event.onset_s
            if duration < MIN_WINDOW_S + SKIP_ATTACK_S:
                continue
            window_start = event.onset_s + SKIP_ATTACK_S
            window_end = window_start + min(MAX_WINDOW_S, duration - SKIP_ATTACK_S)
            # Isolation decided from *detections* — the only thing available
            # at inference. An undetected neighbour will silently violate it.
            if any(
                other is not event and other.onset_s < window_end and other.offset_s > window_start
                for other in events
            ):
                continue
            if len(candidates_for_pitch(event.pitch_midi)) < 2:
                continue
            eligible += 1

            start = int(window_start * sr)
            stop = int(window_end * sr)
            if stop > audio.shape[0]:
                continue
            segment = np.asarray(audio[start:stop, 0], dtype=np.float64)
            nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
            estimate = estimate_inharmonicity(segment, int(sr), nominal)
            if estimate is None or estimate[1] <= 0.0 or estimate[3] < min_r2:
                continue
            fitted_ok += 1

            # Score only where the detection is a real note: a false positive
            # has no gold string, and grading the channel on it would measure
            # the backend rather than the physics.
            reference = next(
                (
                    g
                    for g in gold
                    if g.pitch_midi == event.pitch_midi
                    and abs(g.onset_s - event.onset_s) <= MATCH_TOLERANCE_S
                ),
                None,
            )
            if reference is None:
                continue

            log_b = math.log(estimate[1])
            options = [
                (abs(log_b - (table[s] + (f / 6.0) * LOG2)), s)
                for s, f in candidates_for_pitch(event.pitch_midi)
                if s in table
            ]
            if not options:
                continue
            predicted = min(options)[1]
            hit = int(predicted == reference.string_idx)
            scored += 1
            correct += hit
            control_correct += int(control.get(event.pitch_midi, -1) == reference.string_idx)
            bucket = per_mode[mode]
            bucket[0] += hit
            bucket[1] += 1

    return {
        "min_r2": min_r2,
        "detected_events": detected_total,
        "eligible_isolated_ambiguous": eligible,
        "fit_succeeded": fitted_ok,
        "scored_vs_gold": scored,
        "share_of_detected_scored": scored / detected_total if detected_total else 0.0,
        "accuracy": correct / scored if scored else float("nan"),
        "count_prior_control": control_correct / scored if scored else float("nan"),
        "per_mode": {
            mode: {"n": c[1], "accuracy": c[0] / c[1] if c[1] else float("nan")}
            for mode, c in per_mode.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    print("calibrating B0 from gold mono-mic measurements (LOPO)...", flush=True)
    gold_rows = collect_measurements(args.data_home, DEV_PLAYERS, 0, "mono")
    print(f"  {len(gold_rows)} gold measurements", flush=True)

    clips = select_clips(args.data_home, "comp", 10) + select_clips(args.data_home, "solo", 10)
    results = []
    for min_r2 in (0.0, 0.50, 0.70, 0.90):
        tables = calibrate(gold_rows, min_r2=0.70)
        results.append(
            probe(
                clips,
                data_home=args.data_home,
                workdir=args.workdir,
                tables=tables,
                gold_rows=gold_rows,
                min_r2=min_r2,
            )
        )
        row = results[-1]
        print(
            f"  min_r2={min_r2:.2f}: scored={row['scored_vs_gold']:5d} "
            f"({row['share_of_detected_scored']:5.1%} of detections) "
            f"acc={row['accuracy']:.4f} control={row['count_prior_control']:.4f} "
            f"solo={row['per_mode']['solo']['accuracy']:.4f}",
            flush=True,
        )

    summary = {"clips": clips, "gold_measurements": len(gold_rows), "sweeps": results}
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
