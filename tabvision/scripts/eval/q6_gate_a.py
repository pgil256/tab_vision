"""Accuracy-loop Q6 (ROI deep-dive §4.1) — Gate A: B-estimator on hex stems.

Gate A asks the narrow question the two-gate split exists to isolate: **does
inharmonicity identify the string at all**, given the cleanest possible
signal? GuitarSet's hex-debleeded pickup gives one channel per string, so
every note arrives isolated with its string known — estimator quality is the
only variable. Gate B (mono-mic) then asks whether it survives bleed.

Estimator. For a stiff string ``f_k = k*f0*sqrt(1 + B*k^2)``. Squaring and
dividing by ``k`` linearises it:

    (f_k / k)^2 = f0^2 + (f0^2 * B) * k^2

so a straight line through ``((k^2), (f_k/k)^2)`` yields ``f0 = sqrt(intercept)``
and ``B = slope / intercept`` — no nonlinear fit, and the residual doubles as
a quality signal.

Classification is leave-one-player-out. Per fold, each string's open
coefficient ``B0_s`` is the median of ``log B - (fret/6)*log 2`` over the
training players (the ``B ∝ 2^(n/6)`` law from the separability precursor).
A test note is assigned to whichever candidate position's predicted ``log B``
is closest to its measured one.

Gate A passes at string accuracy >= 0.85 on **ambiguous** notes — notes whose
pitch is playable at more than one position, which is the only slice where
string identity is actually in question.
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

from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.types import GuitarConfig

DEV_PLAYERS = ("00", "01", "02", "03", "04")
OPEN_MIDI = (40, 45, 50, 55, 59, 64)
MAX_FRET = 24
LOG2 = math.log(2.0)

SKIP_ATTACK_S = 0.030
MAX_WINDOW_S = 0.400
MIN_WINDOW_S = 0.120
ZERO_PAD = 4
MAX_PARTIALS = 10
REL_TOLERANCE = 2 ** (60.0 / 1200.0) - 1.0  # +/- 60 cents, capped at 0.4*f0
MIN_PARTIALS = 4
GATE_A = 0.85
GATE_B = 0.70
MIN_COVERAGE = 0.10


def _parabolic_peak(spectrum: np.ndarray, index: int) -> float:
    """Sub-bin peak location by parabolic interpolation on log magnitude."""
    if index <= 0 or index >= len(spectrum) - 1:
        return float(index)
    left, centre, right = (
        math.log(max(spectrum[index - 1], 1e-12)),
        math.log(max(spectrum[index], 1e-12)),
        math.log(max(spectrum[index + 1], 1e-12)),
    )
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    return index + 0.5 * (left - right) / denominator


def _find_partials(
    spectrum: np.ndarray,
    freqs_per_bin: float,
    f0_guess: float,
    b_guess: float,
    sr: int,
    noise_floor: float,
) -> tuple[list[float], list[float]]:
    """Locate partials around the stiff-string prediction for ``b_guess``.

    The search half-width is capped at 0.4*f0. Without that cap a relative
    tolerance widens faster than the partials separate, and by k~10 the
    window swallows its neighbour — the fit then locks onto the wrong peak
    and reports a confidently biased B.
    """
    ks: list[float] = []
    measured: list[float] = []
    for k in range(1, MAX_PARTIALS + 1):
        predicted = k * f0_guess * math.sqrt(1.0 + b_guess * k * k)
        if predicted > sr / 2.0 * 0.9:
            break
        tolerance = min(predicted * REL_TOLERANCE, 0.4 * f0_guess)
        low = int((predicted - tolerance) / freqs_per_bin)
        high = int((predicted + tolerance) / freqs_per_bin) + 1
        if low < 1 or high >= len(spectrum):
            break
        band = spectrum[low:high]
        peak = int(np.argmax(band))
        if band[peak] <= noise_floor:
            continue
        refined = _parabolic_peak(spectrum, low + peak) * freqs_per_bin
        ks.append(float(k))
        measured.append(refined)
    return ks, measured


def _fit(ks: list[float], measured: list[float]) -> tuple[float, float, float] | None:
    """Linearised stiff-string fit: (f_k/k)^2 = f0^2 + (f0^2 B) k^2."""
    if len(ks) < MIN_PARTIALS:
        return None
    k_arr = np.asarray(ks)
    f_arr = np.asarray(measured)
    x = k_arr**2
    y = (f_arr / k_arr) ** 2
    slope, intercept = np.polyfit(x, y, 1)
    if intercept <= 0.0:
        return None
    predicted_y = slope * x + intercept
    ss_res = float(np.sum((y - predicted_y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(math.sqrt(intercept)), float(slope / intercept), r2


def estimate_inharmonicity(
    segment: np.ndarray, sr: int, nominal_f0: float
) -> tuple[float, float, int, float] | None:
    """Return (f0, B, partials_used, r2) or None if the fit is unusable."""
    if segment.size < int(MIN_WINDOW_S * sr):
        return None
    if not np.any(np.abs(segment) > 0.0):
        return None
    windowed = segment * np.hanning(segment.size)
    n_fft = int(2 ** math.ceil(math.log2(segment.size * ZERO_PAD)))
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    peak_magnitude = float(spectrum.max())
    if peak_magnitude <= 0.0:
        return None
    freqs_per_bin = sr / n_fft
    noise_floor = max(float(np.median(spectrum)) * 4.0, peak_magnitude * 1e-4)

    # Pass 1 assumes a harmonic series; pass 2 re-centres the search on the
    # partials the fitted B actually predicts, which matters most at high k
    # where the stiffness shift exceeds the search window.
    guess = 0.0
    result: tuple[float, float, float] | None = None
    used = 0
    for _ in range(2):
        ks, measured = _find_partials(spectrum, freqs_per_bin, nominal_f0, guess, sr, noise_floor)
        fitted = _fit(ks, measured)
        if fitted is None:
            return None
        result = fitted
        used = len(ks)
        guess = max(fitted[1], 0.0)
    if result is None:
        return None
    return result[0], result[1], used, result[2]


def candidates_for_pitch(pitch: int) -> list[tuple[int, int]]:
    return [
        (string, pitch - open_midi)
        for string, open_midi in enumerate(OPEN_MIDI)
        if 0 <= pitch - open_midi <= MAX_FRET
    ]


def collect_measurements(
    data_home: Path, players: tuple[str, ...], max_tracks: int = 0, source: str = "hex"
) -> list[dict[str, Any]]:
    """Measure B for every usable isolated gold note.

    ``source='hex'`` reads the note's own debleeded pickup channel — Gate A,
    where string identity is known and only estimator quality varies.
    ``source='mono'`` reads the room mic, so the same notes now carry
    reverb, body resonance and neighbouring-string sympathetic ringing:
    Gate B asks whether the estimator survives that.
    """
    cfg = GuitarConfig()
    hex_dir = data_home / "audio_hex-pickup_debleeded"
    rows: list[dict[str, Any]] = []
    channel_energy_check: list[tuple[int, int]] = []

    tracks_done = 0
    for jams_path in sorted((data_home / "annotation").glob("*.jams")):
        track_id = jams_path.stem
        if track_id[:2] not in players:
            continue
        if max_tracks and tracks_done >= max_tracks:
            break
        if source == "hex":
            wav_path = hex_dir / f"{track_id}_hex_cln.wav"
        else:
            wav_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        if not wav_path.is_file():
            continue
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        if source == "hex" and audio.shape[1] < 6:
            continue
        gold = sorted(parse_guitarset_jams(jams_path, cfg), key=lambda e: e.onset_s)

        for event in gold:
            if event.duration_s < MIN_WINDOW_S + SKIP_ATTACK_S:
                continue
            # Gate A is the *isolated-note* regime. Restricting to notes with
            # no other string sounding in the analysis window is not just
            # scope discipline: with a chord ringing, the hex channel's
            # residual crosstalk carries partials from louder neighbours and
            # the fit is measuring the wrong string.
            window_start = event.onset_s + SKIP_ATTACK_S
            window_end = window_start + min(MAX_WINDOW_S, event.duration_s - SKIP_ATTACK_S)
            if any(
                other is not event
                and other.onset_s < window_end
                and (other.onset_s + other.duration_s) > window_start
                for other in gold
            ):
                continue
            start = int((event.onset_s + SKIP_ATTACK_S) * sr)
            stop = start + int(min(MAX_WINDOW_S, event.duration_s - SKIP_ATTACK_S) * sr)
            if stop > audio.shape[0]:
                continue
            channel = event.string_idx if source == "hex" else 0
            segment = audio[start:stop, channel]
            if not np.any(segment):
                continue
            nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
            fitted = estimate_inharmonicity(np.asarray(segment, dtype=np.float64), sr, nominal)
            if fitted is None:
                continue
            _f0, b_value, partials, r2 = fitted
            if b_value <= 0.0:
                continue

            if source == "hex":
                # Sanity: the annotated string's channel should carry the note.
                energies = np.sum(audio[start:stop, :6].astype(np.float64) ** 2, axis=0)
                channel_energy_check.append((int(np.argmax(energies)), event.string_idx))

            rows.append(
                {
                    "track_id": track_id,
                    "player": track_id[:2],
                    "mode": "solo" if track_id.endswith("_solo") else "comp",
                    "string": event.string_idx,
                    "fret": event.fret,
                    "pitch": event.pitch_midi,
                    "log_b": math.log(b_value),
                    "partials": partials,
                    "r2": r2,
                    "ambiguous": len(candidates_for_pitch(event.pitch_midi)) > 1,
                }
            )
        tracks_done += 1
        print(f"  {track_id}: {len(rows)} cumulative measurements", flush=True)

    agree = sum(1 for got, want in channel_energy_check if got == want)
    if channel_energy_check:
        share = agree / len(channel_energy_check)
        print(f"channel<->string mapping check: {share:.4f} agreement", flush=True)
        # On isolated notes the annotated string's channel should dominate
        # almost always (measured 0.98 on a 00_* solo sample). A reversed
        # channel order scores ~0.01 here, so this cleanly separates the two
        # conventions rather than merely nudging past chance.
        if share < 0.85:
            raise SystemExit(
                f"hex channel order does not match string_idx ({share:.3f}); "
                "the loader's convention needs revisiting before trusting Gate A"
            )
    return rows


def classify_leave_one_player_out(rows: list[dict[str, Any]], min_r2: float) -> dict[str, Any]:
    usable = [row for row in rows if row["r2"] >= min_r2]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_player[row["player"]].append(row)

    total = correct = amb_total = amb_correct = 0
    baseline_correct = 0
    per_mode: dict[str, list[int]] = {"solo": [0, 0], "comp": [0, 0]}

    for held_out in sorted(by_player):
        train = [row for player, items in by_player.items() if player != held_out for row in items]
        # B0_s from the B ∝ 2^(fret/6) law, median for robustness.
        b0: dict[int, float] = {}
        for string in range(6):
            values = [
                row["log_b"] - (row["fret"] / 6.0) * LOG2
                for row in train
                if row["string"] == string
            ]
            if values:
                b0[string] = float(np.median(values))
        # Control: the context-free count prior on the *same* notes. Without
        # it, a high B accuracy could just mean these isolated notes are easy.
        popular: dict[int, int] = {}
        counts: dict[int, Counter] = defaultdict(Counter)
        for row in train:
            counts[row["pitch"]][row["string"]] += 1
        for pitch, counter in counts.items():
            popular[pitch] = counter.most_common(1)[0][0]
        for row in by_player[held_out]:
            options = candidates_for_pitch(row["pitch"])
            scored = [
                (abs(row["log_b"] - (b0[s] + (f / 6.0) * LOG2)), s) for s, f in options if s in b0
            ]
            if not scored:
                continue
            predicted = min(scored)[1]
            hit = int(predicted == row["string"])
            total += 1
            correct += hit
            baseline_correct += int(popular.get(row["pitch"], -1) == row["string"])
            bucket = per_mode[row["mode"]]
            bucket[0] += hit
            bucket[1] += 1
            if row["ambiguous"]:
                amb_total += 1
                amb_correct += hit

    return {
        "min_r2": min_r2,
        "notes_scored": total,
        "coverage": len(usable) / len(rows) if rows else 0.0,
        "count_prior_baseline": baseline_correct / total if total else float("nan"),
        "accuracy_all": correct / total if total else float("nan"),
        "ambiguous_notes": amb_total,
        "accuracy_ambiguous": amb_correct / amb_total if amb_total else float("nan"),
        "per_mode": {
            mode: {"n": counts[1], "accuracy": counts[0] / counts[1] if counts[1] else float("nan")}
            for mode, counts in per_mode.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--source", choices=("hex", "mono"), default="hex")
    args = parser.parse_args()

    rows = collect_measurements(args.data_home, DEV_PLAYERS, args.max_tracks, args.source)
    if not rows:
        raise SystemExit("no usable measurements — is the hex partition present?")

    thresholds = (0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
    results = [classify_leave_one_player_out(rows, min_r2) for min_r2 in thresholds]
    # Require real coverage: an arm that keeps 1% of notes can post a
    # flattering number while abstaining on everything hard.
    eligible = [item for item in results if item["coverage"] >= MIN_COVERAGE]
    best = max(eligible or results, key=lambda item: item["accuracy_ambiguous"])
    summary = {
        "measurements": len(rows),
        "ambiguous_share": sum(1 for row in rows if row["ambiguous"]) / len(rows),
        "median_partials": float(np.median([row["partials"] for row in rows])),
        "median_r2": float(np.median([row["r2"] for row in rows])),
        "sweeps": results,
        "best": best,
        "source": args.source,
        "gate": GATE_A if args.source == "hex" else GATE_B,
        "gate_pass": bool(
            best["accuracy_ambiguous"] >= (GATE_A if args.source == "hex" else GATE_B)
        ),
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nmeasurements: {len(rows)} (median {summary['median_partials']:.0f} partials, "
        f"median r2 {summary['median_r2']:.4f})"
    )
    for result in results:
        print(
            f"  min_r2={result['min_r2']:.2f}: n={result['notes_scored']:6d} "
            f"cover={result['coverage']:6.1%} "
            f"ambiguous={result['accuracy_ambiguous']:.4f} "
            f"(count-prior control {result['count_prior_baseline']:.4f}) "
            f"solo={result['per_mode']['solo']['accuracy']:.4f}"
        )
    label = "Gate A (hex)" if args.source == "hex" else "Gate B (mono-mic)"
    print(
        f"\n{label}: ambiguous accuracy {best['accuracy_ambiguous']:.4f} "
        f"at {best['coverage']:.1%} coverage vs {summary['gate']} -> "
        f"{'PASS' if summary['gate_pass'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
