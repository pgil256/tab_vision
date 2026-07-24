"""Accuracy-loop Q5 (ROI deep-dive §4.2) — onset snapping prototype.

Tab F1 gates on a 50 ms onset match, so a note whose pitch and string are
both right still scores zero if its onset lands 51 ms out. The banked
decomposition puts 2.2% of loss in ``timing_only`` outright, with more hiding
in ``missed_onset`` as boundary cases. This refines onset times *before*
fusion against the audio's own spectral-flux structure, then re-scores.

Two admission shapes, both predeclared:

- ``snap-<W>`` — move each onset to the strongest flux peak within +/-W ms.
- ``strum-<W>`` — the same, then collapse each 80 ms cluster onto its
  members' median snapped onset. A strummed chord is one physical gesture
  spread 30-60 ms across the strings; snapping members independently can
  *widen* a cluster and fragment the voicing the 80 ms grouping is meant to
  hold together.

**This intentionally changes onsets**, so the report carries the full
onset/pitch battery, not just Tab F1 — the A15 bit-identity discipline does
not apply and its absence has to be visible.

Pure pre-fuse event surgery on the banked 20-clip ensemble cache: no backend
inference, no pipeline change, no contract change.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import (
    _event_from_json,
    _score,
    build_oof_priors,
    select_clips,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.fusion.transition_prior import CLUSTER_GAP_S
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
N_FFT = 1024
HOP = 256


@dataclass(frozen=True)
class SnapVariant:
    name: str
    window_s: float
    strum: bool


VARIANTS: tuple[SnapVariant, ...] = (
    SnapVariant("baseline", 0.0, False),
    SnapVariant("snap-10ms", 0.010, False),
    SnapVariant("snap-20ms", 0.020, False),
    SnapVariant("snap-30ms", 0.030, False),
    SnapVariant("snap-50ms", 0.050, False),
    SnapVariant("strum-20ms", 0.020, True),
    SnapVariant("strum-30ms", 0.030, True),
)


def onset_envelope(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-wave-rectified spectral flux and its frame times.

    Plain STFT flux rather than a library onset detector: the goal is a
    *local* strength curve to snap against, not a detection decision — the
    events already exist.
    """
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    window = np.hanning(N_FFT).astype(np.float32)
    frames = 1 + max(0, (len(wav) - N_FFT) // HOP)
    spectrum = np.empty((frames, N_FFT // 2 + 1), dtype=np.float32)
    for index in range(frames):
        start = index * HOP
        spectrum[index] = np.abs(np.fft.rfft(wav[start : start + N_FFT] * window))
    flux = np.diff(spectrum, axis=0, prepend=spectrum[:1])
    strength = np.maximum(flux, 0.0).sum(axis=1)
    times = np.arange(frames, dtype=np.float64) * HOP / sr
    return strength, times


def snap_onset(onset_s: float, strength: np.ndarray, times: np.ndarray, window_s: float) -> float:
    """Nearest strongest flux peak within the window, else the original."""
    if window_s <= 0.0 or times.size == 0:
        return onset_s
    low = np.searchsorted(times, onset_s - window_s, side="left")
    high = np.searchsorted(times, onset_s + window_s, side="right")
    if high <= low:
        return onset_s
    local = strength[low:high]
    if not np.any(local > 0.0):
        return onset_s
    return float(times[low + int(np.argmax(local))])


def apply_snapping(
    events: Sequence[AudioEvent],
    strength: np.ndarray,
    times: np.ndarray,
    variant: SnapVariant,
) -> tuple[list[AudioEvent], list[float]]:
    """Rewrite onsets (keeping durations) and report each note's shift."""
    if variant.window_s <= 0.0:
        return list(events), [0.0] * len(events)

    ordered = sorted(events, key=lambda event: event.onset_s)
    snapped = [snap_onset(event.onset_s, strength, times, variant.window_s) for event in ordered]

    if variant.strum:
        # Group on the *original* onsets — the 80 ms grouping the decode uses
        # is defined on what the backend emitted, and regrouping on snapped
        # times would let snapping silently redraw cluster boundaries.
        start = 0
        for index in range(1, len(ordered) + 1):
            boundary = (
                index == len(ordered)
                or ordered[index].onset_s - ordered[index - 1].onset_s > CLUSTER_GAP_S
            )
            if boundary:
                if index - start > 1:
                    shared = float(np.median(snapped[start:index]))
                    for position in range(start, index):
                        snapped[position] = shared
                start = index

    result: list[AudioEvent] = []
    shifts: list[float] = []
    for event, onset in zip(ordered, snapped, strict=True):
        duration = event.offset_s - event.onset_s
        result.append(replace(event, onset_s=onset, offset_s=onset + duration))
        # Paired before the final re-sort: snapping can reorder events, so
        # zipping the sorted lists afterwards would compare different notes.
        shifts.append(abs(onset - event.onset_s))
    result.sort(key=lambda event: (event.onset_s, event.pitch_midi))
    return result, shifts


def run(clips: Sequence[str], *, data_home: Path, workdir: Path) -> dict[str, Any]:
    import soundfile as sf

    cfg = GuitarConfig()
    session = SessionConfig()
    oof_priors = build_oof_priors(data_home, cfg)

    scores: dict[str, list[dict[str, float]]] = {v.name: [] for v in VARIANTS}
    decomps: dict[str, list[ErrorDecomposition]] = {v.name: [] for v in VARIANTS}
    displacements: dict[str, list[float]] = {v.name: [] for v in VARIANTS}
    per_clip: list[dict[str, Any]] = []

    for track_id in clips:
        cache = workdir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            raise SystemExit(f"missing ensemble cache for {track_id}")
        events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        wav, sr = sf.read(data_home / "audio_mono-mic" / f"{track_id}_mic.wav", dtype="float32")
        strength, times = onset_envelope(np.asarray(wav), int(sr))
        prior = oof_priors[track_id[:2]]

        row: dict[str, Any] = {"track_id": track_id, "mode": row_mode(track_id)}
        for variant in VARIANTS:
            moved, shifts = apply_snapping(events, strength, times, variant)
            displacements[variant.name].extend(shifts)
            metrics, decomposition = _score(moved, gold, cfg=cfg, session=session, prior=prior)
            scores[variant.name].append(metrics)
            decomps[variant.name].append(decomposition)
            row[variant.name] = metrics
        per_clip.append(row)
        print(
            f"{track_id}: base tab={row['baseline']['tab_f1']:.4f} "
            f"snap30={row['snap-30ms']['tab_f1']:.4f} "
            f"strum30={row['strum-30ms']['tab_f1']:.4f}",
            flush=True,
        )

    return {
        "clips": list(clips),
        "variants": summarize(scores, decomps, displacements),
        "per_clip": per_clip,
    }


def row_mode(track_id: str) -> str:
    return "solo" if track_id.endswith("_solo") else "comp"


def summarize(
    scores: dict[str, list[dict[str, float]]],
    decomps: dict[str, list[ErrorDecomposition]],
    displacements: dict[str, list[float]],
) -> dict[str, Any]:
    def column(name: str, metric: str) -> np.ndarray:
        return np.asarray([row[metric] for row in scores[name]], dtype=np.float64)

    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        entry: dict[str, Any] = {
            "window_s": variant.window_s,
            "strum": variant.strum,
            "mean_abs_shift_ms": (
                float(np.mean(displacements[variant.name]) * 1000.0)
                if displacements[variant.name]
                else 0.0
            ),
            "decomposition": aggregate_decompositions(decomps[variant.name]).to_dict(),
        }
        # Every gate this touches, not just Tab F1 — snapping moves onsets by
        # construction, so onset and pitch must be reported as first-class.
        for metric in ("tab_f1", "onset_f1", "pitch_f1"):
            values = column(variant.name, metric)
            delta = values - column("baseline", metric)
            ci = bootstrap_ci(delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            entry[metric] = float(values.mean())
            entry[f"{metric}_delta"] = float(delta.mean())
            entry[f"{metric}_lo95"] = ci.lower
            entry[f"{metric}_hi95"] = ci.upper
        summary[variant.name] = entry
    return summary


def write_report(summary: dict[str, Any], path: Path) -> None:
    variants = summary["variants"]
    lines = [
        "# Q5 onset snapping — pre-fuse onset refinement",
        "",
        f"{len(summary['clips'])} GuitarSet dev clips (10 comp + 10 solo), offline replay "
        "of the banked ensemble events; shipped clean-acoustic decode with the "
        "leave-one-player-out position prior + `guitarset-seq-v1` @ w=4.0. "
        "Snapping targets half-wave-rectified STFT spectral flux "
        f"(n_fft={N_FFT}, hop={HOP}).",
        "",
        "**This changes onsets by construction**, so onset and pitch F1 are "
        "reported as first-class gates alongside Tab F1.",
        "",
        "| variant | mean \\|shift\\| | Tab F1 | ΔTab [lo, hi] | onset F1 | Δonset [lo, hi] "
        "| pitch F1 | Δpitch [lo, hi] |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for variant in VARIANTS:
        row = variants[variant.name]
        lines.append(
            f"| `{variant.name}` | {row['mean_abs_shift_ms']:.1f} ms "
            f"| {row['tab_f1']:.4f} | {row['tab_f1_delta']:+.4f} "
            f"[{row['tab_f1_lo95']:+.4f}, {row['tab_f1_hi95']:+.4f}] "
            f"| {row['onset_f1']:.4f} | {row['onset_f1_delta']:+.4f} "
            f"[{row['onset_f1_lo95']:+.4f}, {row['onset_f1_hi95']:+.4f}] "
            f"| {row['pitch_f1']:.4f} | {row['pitch_f1_delta']:+.4f} "
            f"[{row['pitch_f1_lo95']:+.4f}, {row['pitch_f1_hi95']:+.4f}] |"
        )
    lines += [
        "",
        "## Six-bucket decomposition",
        "",
        "| variant | correct | wrong_position | pitch_off | timing_only | missed_onset "
        "| extra_detection |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        buckets = variants[variant.name]["decomposition"]
        lines.append(
            f"| `{variant.name}` | {buckets['correct']} "
            f"| {buckets['wrong_position_same_pitch']} | {buckets['pitch_off']} "
            f"| {buckets['timing_only']} | {buckets['missed_onset']} "
            f"| {buckets['extra_detection']} |"
        )
    lines += [
        "",
        f"Bootstrap: paired per-clip deltas, N={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}. "
        "Acceptance would need lo-95 > 0 on Tab F1 *and* no CI-significant "
        "regression on onset or pitch F1.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--comp-clips", type=int, default=10)
    parser.add_argument("--solo-clips", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    workdir = args.workdir or (data_root / "models" / "muscriptor_probe")

    clips = select_clips(data_home, "comp", args.comp_clips)
    clips += select_clips(data_home, "solo", args.solo_clips)
    summary = run(clips, data_home=data_home, workdir=workdir)

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output is not None:
        write_report(summary, args.output)

    for variant in VARIANTS:
        row = summary["variants"][variant.name]
        print(
            f"{variant.name:>12}: tab={row['tab_f1']:.4f} "
            f"({row['tab_f1_delta']:+.4f} [{row['tab_f1_lo95']:+.4f}, "
            f"{row['tab_f1_hi95']:+.4f}])  onset={row['onset_f1']:.4f} "
            f"({row['onset_f1_delta']:+.4f})"
        )
    best = max(
        (v for v in VARIANTS if v.name != "baseline"),
        key=lambda v: summary["variants"][v.name]["tab_f1_delta"],
    )
    row = summary["variants"][best.name]
    passes = (
        row["tab_f1_lo95"] > 0.0 and row["onset_f1_lo95"] >= 0.0 and row["pitch_f1_lo95"] >= 0.0
    )
    print(f"\nbest={best.name} → {'PASS' if passes else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
