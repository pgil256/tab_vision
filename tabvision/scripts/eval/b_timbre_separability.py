"""Track B — direct timbral separability, covered vs abstain, no model needed.

The complement probe establishes *where* a timbral model would have to work: the
~75% of ambiguous notes where the physics channel abstains. This asks whether
the signal is there at all, and deliberately asks it the cheapest way that can
still give a decisive answer.

Following Q6's separability precursor rather than Phase 2/Phase 4's model-first
approach: before training anything, ask whether same-pitch notes on *different
strings* are separable in timbral feature space. If they are not, no classifier
built on those features can succeed, and the question is closed without a
training run. Q6 used exactly this shape to justify the physics channel before
building it.

The comparison that matters is **covered vs abstain**, because the hypothesis
under test is mechanistic: the physics channel abstains when partials are
unreadable, which happens under simultaneity. If timbral separability collapses
on the same population for the same reason, then timbre and physics are not
complementary — they are limited by the same thing, and timbre cannot cover
physics' complement however it is modelled.

Features are deliberately plain and few: log-band energies, spectral centroid,
rolloff, flatness, and attack/decay slope. Phase 4 already ran a far richer set
(multi-resolution harmonic envelopes through Nyquist, pick noise, inharmonicity)
over 56,742 pairs and reached +0.0072 against a +0.05 gate. If plain features
show separability collapsing on the abstain population, that explains Phase 4's
result rather than merely repeating it. If plain features show healthy
separability there, the honest conclusion is that the question needs Phase 4's
features again — which those entries forbid re-running enlarged, so it would go
back to the user rather than proceed.

Separability is measured as leave-one-player-out AUC for same-pitch
string-vs-string pairs, which needs no calibration and no threshold, and is
directly comparable between the two populations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.a_physics_coverage import load_banked
from scripts.eval.b_timbre_complement import ambiguous, gold_string_for
from scripts.eval.n2_muscriptor_merge import _event_from_json
from scripts.eval.phase0_rotation_baseline import BURNED_PLAYER, DEV_PLAYERS, gold_by_player
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.string_physics import load_string_evidence
from tabvision.types import AudioEvent, GuitarConfig

SKIP_ATTACK_S = 0.02
WINDOW_S = 0.30
MIN_PAIRS = 40
BANDS = (200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0)


def features(event: AudioEvent, audio: np.ndarray, sr: int) -> np.ndarray | None:
    """A small, plain timbral descriptor for one note."""
    start = int((event.onset_s + SKIP_ATTACK_S) * sr)
    stop = start + int(WINDOW_S * sr)
    if start < 0 or stop > audio.size or stop - start < 512:
        return None
    frame = audio[start:stop]
    if not np.any(frame):
        return None
    window = np.hanning(frame.size)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(frame.size, 1.0 / sr)
    total = float(spectrum.sum())
    if total <= 0.0:
        return None

    values: list[float] = []
    lower = 0.0
    for edge in BANDS:
        mask = (freqs >= lower) & (freqs < edge)
        values.append(math.log(float(spectrum[mask].sum()) + 1e-9))
        lower = edge

    centroid = float((freqs * spectrum).sum() / total)
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    geometric = math.exp(float(np.mean(np.log(spectrum + 1e-12))))
    flatness = geometric / (total / spectrum.size + 1e-12)

    half = frame.size // 2
    early = float(np.sqrt(np.mean(frame[:half] ** 2)) + 1e-9)
    late = float(np.sqrt(np.mean(frame[half:] ** 2)) + 1e-9)
    decay = math.log(late / early)

    values.extend([math.log(centroid + 1e-9), math.log(rolloff + 1e-9), flatness, decay])
    vector = np.asarray(values, dtype=np.float64)
    return vector if np.all(np.isfinite(vector)) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — the Mann-Whitney convention.

    Plain ``argsort`` gives tied values arbitrary distinct ranks, which makes a
    tie count as a win for whichever side happens to sort first. With all scores
    tied that yields 1.0 or 0.0 instead of chance. Real LDA projections rarely
    tie exactly, so this is a small correction, but a statistic that cannot
    represent "no information" is the wrong statistic for a probe whose entire
    job is deciding whether information exists.
    """
    order = values.argsort(kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    return ranks


def pair_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUC between two score sets; 0.5 is chance, ties neutral."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = _average_ranks(combined)
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size)
    )


def fit_lda(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Fisher direction between two classes — the cheapest honest separator."""
    a = x[y == 1]
    b = x[y == 0]
    if a.shape[0] < 3 or b.shape[0] < 3:
        return None
    mean_diff = a.mean(axis=0) - b.mean(axis=0)
    pooled = np.cov(np.vstack([a - a.mean(axis=0), b - b.mean(axis=0)]).T)
    pooled = pooled + np.eye(pooled.shape[0]) * 1e-3
    try:
        return np.linalg.solve(pooled, mean_diff)
    except np.linalg.LinAlgError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--fit-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    fit_cache = args.fit_cache or (data_root / "models" / "a_partial_fit_cache")
    dev_cache = data_root / "models" / "q6_full_dev_cache"
    sealed_cache = data_root / "models" / "q6_player05_cache"

    cfg = GuitarConfig()
    evidence = load_string_evidence()
    isolation = evidence.isolation
    gold = gold_by_player(data_home, cfg)
    clips = sorted(t for p in DEV_PLAYERS for t in gold[p])
    if args.limit:
        clips = clips[: args.limit]

    # population -> pitch -> string -> list[(player, feature vector)]
    store: dict[str, dict[int, dict[int, list[tuple[str, np.ndarray]]]]] = {
        "covered": defaultdict(lambda: defaultdict(list)),
        "abstain": defaultdict(lambda: defaultdict(list)),
    }

    for index, track_id in enumerate(clips, start=1):
        player = track_id[:2]
        cache = sealed_cache if player == BURNED_PLAYER else dev_cache
        events = [
            _event_from_json(item)
            for item in json.loads((cache / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        ordered = sorted(events, key=lambda event: event.onset_s)
        banked = load_banked(fit_cache / f"{track_id}.{isolation}.json")
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        audio = np.asarray(wav, dtype=np.float64)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        clip_gold = gold[player][track_id]

        for i, event in enumerate(ordered):
            if not ambiguous(event, cfg):
                continue
            string_idx = gold_string_for(event, clip_gold)
            if string_idx is None:
                continue
            vector = features(event, audio, int(sr))
            if vector is None:
                continue
            fit = banked.get(i)
            population = "covered" if (fit is not None and fit.r2 >= evidence.min_r2) else "abstain"
            store[population][event.pitch_midi][string_idx].append((player, vector))

        if index % 50 == 0 or index == len(clips):
            print(f"  [{index}/{len(clips)}]", flush=True)

    results: dict[str, Any] = {"clips": len(clips), "populations": {}}
    print(f"\n{'population':<12}{'pairs':>8}{'notes':>9}{'mean AUC':>11}{'median':>9}")
    for population, by_pitch in store.items():
        aucs: list[float] = []
        pairs = 0
        notes = 0
        for _pitch, by_string in by_pitch.items():
            strings = sorted(by_string)
            notes += sum(len(v) for v in by_string.values())
            for a_i in range(len(strings)):
                for b_i in range(a_i + 1, len(strings)):
                    a_rows = by_string[strings[a_i]]
                    b_rows = by_string[strings[b_i]]
                    if len(a_rows) + len(b_rows) < MIN_PAIRS:
                        continue
                    players = {p for p, _ in a_rows} | {p for p, _ in b_rows}
                    if len(players) < 2:
                        continue
                    fold_aucs: list[float] = []
                    for held in sorted(players):
                        tr_x, tr_y, te_x, te_y = [], [], [], []
                        for label, rows in ((1, a_rows), (0, b_rows)):
                            for owner, vector in rows:
                                if owner == held:
                                    te_x.append(vector)
                                    te_y.append(label)
                                else:
                                    tr_x.append(vector)
                                    tr_y.append(label)
                        if len(te_x) < 4 or len(tr_x) < 10:
                            continue
                        direction = fit_lda(np.asarray(tr_x), np.asarray(tr_y))
                        if direction is None:
                            continue
                        scores = np.asarray(te_x) @ direction
                        te_y_arr = np.asarray(te_y)
                        auc = pair_auc(scores[te_y_arr == 1], scores[te_y_arr == 0])
                        if not math.isnan(auc):
                            fold_aucs.append(auc)
                    if fold_aucs:
                        aucs.append(float(np.mean(fold_aucs)))
                        pairs += 1
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        median_auc = float(np.median(aucs)) if aucs else float("nan")
        results["populations"][population] = {
            "string_pairs": pairs,
            "notes": notes,
            "mean_auc": mean_auc,
            "median_auc": median_auc,
            "aucs": aucs,
        }
        print(f"{population:<12}{pairs:>8}{notes:>9}{mean_auc:>11.4f}{median_auc:>9.4f}")

    print("\n(0.5 = chance. Leave-one-player-out; a pair needs >=2 players and >=40 notes.)")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
