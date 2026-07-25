"""Track D — decompose the detection buckets before building anything.

Phase 0 promoted this track: ``missed_onset`` + ``extra_detection`` are **33.3%**
of development loss and **39.4%** of sealed loss, against 47.6% / 38.6% for
wrong position. On the sealed player they are effectively tied with the bucket
that has absorbed nearly all of the project's accuracy effort. These two have
had almost none.

**Measurement first, and possibly measurement only.** The A10 precedent is the
model: decomposing ``pitch_off`` *closed* it — 52% "other", no dominant mode —
and saved the build that would otherwise have followed. The same discipline
applies here. This script does not fix anything. It asks whether there is a
dominant fixable mode, and a negative answer is a complete result.

What it asks
------------

**``extra_detection`` — is it harmonic leakage?** For each spurious prediction,
find the nearest gold note in time and classify the pitch interval between
them. A pipeline hallucinating octaves and fifths above real notes is
over-reading the harmonic series and is fixable at the detector. A flat
distribution is not. This deliberately mirrors ``classify_pitch_off_delta``,
because A10 already proved the shape of that answer is what decides the build.

Also asked: **is it ring-out?** A spurious detection whose pitch matches a
recently-ended gold note is the decay of that note being re-triggered, which is
a different fix (offset handling) from harmonic leakage.

**``missed_onset`` — is it masking?** For each missed gold note, count how many
other gold notes are sounding at its onset. Phase 0's 5:1 strummed:single-line
split points at notes buried inside dense voicings rather than a general recall
failure. Also recorded: the missed note's pitch register and its duration,
since short and low notes are separately plausible causes.

Everything replays banked events and gold. No audio, no inference, no tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.eval.n2_muscriptor_merge import _event_from_json
from scripts.eval.phase0_rotation_baseline import (
    BURNED_PLAYER,
    DEV_PLAYERS,
    build_loo_priors,
    gold_by_player,
)
from tabvision.eval.error_decomposition import Residuals, decompose_errors
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.types import GuitarConfig, SessionConfig, TabEvent

NEAR_S = 0.25
"""How close a gold note must be to a spurious detection to be its referent."""

RING_OUT_S = 1.0
"""How long after a gold note ends its decay can still be re-triggered."""


def interval_class(semitones: int) -> str:
    """Classify a spurious detection's interval from its nearest gold note.

    Mirrors ``classify_pitch_off_delta`` deliberately — A10 established that the
    *shape* of this distribution is what decides whether a bucket is worth
    building against, so the two must be readable side by side.
    """
    magnitude = abs(semitones)
    if magnitude == 0:
        return "unison"
    if magnitude % 12 == 0:
        return "octave"
    if magnitude % 12 in (5, 7):
        return "fifth_fourth"
    if magnitude <= 2:
        return "semitone"
    return "other"


def concurrency_at(onset: float, events: list[TabEvent], exclude: TabEvent) -> int:
    """How many other gold notes are sounding at this onset."""
    return sum(
        1
        for other in events
        if other is not exclude and other.onset_s <= onset < other.onset_s + other.duration_s
    )


def register_of(pitch: int) -> str:
    if pitch < 52:
        return "low (<E3)"
    if pitch < 64:
        return "mid (E3-E4)"
    return "high (>=E4)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    dev_cache = data_root / "models" / "q6_full_dev_cache"
    sealed_cache = data_root / "models" / "q6_player05_cache"

    cfg = GuitarConfig()
    session = SessionConfig()
    evidence = load_string_evidence()
    table = reference_stiffness_model()

    gold = gold_by_player(data_home, cfg)
    clips = sorted(t for p in DEV_PLAYERS for t in gold[p])
    if args.limit:
        clips = clips[: args.limit]

    print("building leave-one-player-out priors...", flush=True)
    positions, sequences = build_loo_priors(gold, cfg)

    extra_by_interval: Counter[str] = Counter()
    extra_by_mode: Counter[str] = Counter()
    extra_ring_out = 0
    extra_orphan = 0
    extra_total = 0

    missed_by_concurrency: Counter[str] = Counter()
    missed_by_register: Counter[str] = Counter()
    missed_by_mode: Counter[str] = Counter()
    missed_short = 0
    missed_total = 0

    # Base rates. Without these the findings are unreadable: "86% of missed
    # notes have a neighbour sounding" says nothing if 86% of *all* notes do,
    # and guitar music is full of fourths and fifths by construction (standard
    # tuning, chord voicings), so a high fifth/fourth share among spurious
    # detections may simply be the interval content of the music. This repo has
    # twice published a conditional rate without its marginal and had to retract
    # the reading; compute both or report neither.
    base_concurrency: Counter[str] = Counter()
    base_interval: Counter[str] = Counter()
    base_notes = 0
    base_short = 0

    for index, track_id in enumerate(clips, start=1):
        player = track_id[:2]
        mode = "solo" if track_id.endswith("_solo") else "comp"
        cache = sealed_cache if player == BURNED_PLAYER else dev_cache
        events = [
            _event_from_json(item)
            for item in json.loads((cache / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        ordered = sorted(events, key=lambda event: event.onset_s)
        moved, _ = attach_inharmonicity_evidence(
            ordered,
            wav,
            int(sr),
            table,
            cfg,
            weight=evidence.weight,
            min_r2=evidence.min_r2,
            sigma=evidence.sigma,
            isolation=evidence.isolation,
        )
        clip_gold = gold[player][track_id]
        residuals = Residuals()
        # Score exactly as production does, then read the residuals the buckets
        # only counted.
        prepared = _decode(moved, clip_gold, positions[player], sequences[player], cfg, session)
        decompose_errors(prepared, clip_gold, residuals=residuals)

        gold_sorted = sorted(clip_gold, key=lambda e: e.onset_s)

        # Base rates over every gold note in this clip, using the same
        # definitions the conditional counts use.
        for note in gold_sorted:
            base_notes += 1
            concurrency = concurrency_at(note.onset_s, gold_sorted, note)
            if concurrency == 0:
                base_concurrency["alone"] += 1
            elif concurrency <= 2:
                base_concurrency["1-2 others"] += 1
            else:
                base_concurrency["3+ others"] += 1
            if note.duration_s < 0.15:
                base_short += 1
            neighbours = [
                g for g in gold_sorted if g is not note and abs(g.onset_s - note.onset_s) <= NEAR_S
            ]
            if neighbours:
                closest = min(neighbours, key=lambda g: abs(g.onset_s - note.onset_s))
                base_interval[interval_class(note.pitch_midi - closest.pitch_midi)] += 1

        for spurious in residuals.extra:
            extra_total += 1
            extra_by_mode[mode] += 1
            near = [g for g in gold_sorted if abs(g.onset_s - spurious.onset_s) <= NEAR_S]
            if not near:
                # Nothing real anywhere near it: check whether it is the decay
                # of a note that has already ended.
                decayed = [
                    g
                    for g in gold_sorted
                    if g.pitch_midi == spurious.pitch_midi
                    and g.onset_s + g.duration_s <= spurious.onset_s
                    and spurious.onset_s - (g.onset_s + g.duration_s) <= RING_OUT_S
                ]
                if decayed:
                    extra_ring_out += 1
                    extra_by_interval["ring_out"] += 1
                else:
                    extra_orphan += 1
                    extra_by_interval["orphan"] += 1
                continue
            closest = min(near, key=lambda g: abs(g.onset_s - spurious.onset_s))
            extra_by_interval[interval_class(spurious.pitch_midi - closest.pitch_midi)] += 1

        for miss in residuals.missed:
            missed_total += 1
            missed_by_mode[mode] += 1
            missed_by_register[register_of(miss.pitch_midi)] += 1
            concurrency = concurrency_at(miss.onset_s, gold_sorted, miss)
            if concurrency == 0:
                bucket = "alone"
            elif concurrency <= 2:
                bucket = "1-2 others"
            else:
                bucket = "3+ others"
            missed_by_concurrency[bucket] += 1
            if miss.duration_s < 0.15:
                missed_short += 1

        if index % 50 == 0 or index == len(clips):
            print(f"  [{index}/{len(clips)}]", flush=True)

    def shares(counter: Counter[str], total: int) -> dict[str, Any]:
        return {
            key: {"count": value, "share": value / total if total else 0.0}
            for key, value in counter.most_common()
        }

    results = {
        "clips": len(clips),
        "extra_detection": {
            "total": extra_total,
            "by_interval": shares(extra_by_interval, extra_total),
            "by_mode": shares(extra_by_mode, extra_total),
            "ring_out": extra_ring_out,
            "orphan": extra_orphan,
        },
        "missed_onset": {
            "total": missed_total,
            "by_concurrency": shares(missed_by_concurrency, missed_total),
            "by_register": shares(missed_by_register, missed_total),
            "by_mode": shares(missed_by_mode, missed_total),
            "short_notes": missed_short,
        },
    }

    base_interval_total = sum(base_interval.values())
    results["base_rates"] = {
        "gold_notes": base_notes,
        "by_concurrency": shares(base_concurrency, base_notes),
        "by_interval": shares(base_interval, base_interval_total),
        "short_notes": base_short,
    }

    def lift(observed: float, base: float) -> str:
        if base <= 0:
            return "   n/a"
        return f"{observed / base:>5.2f}x"

    print(f"\n=== extra_detection ({extra_total}) vs interval content of the music ===")
    print(f"  {'class':<16}{'observed':>10}{'base':>9}{'lift':>8}")
    for key, value in extra_by_interval.most_common():
        obs = value / max(extra_total, 1)
        base = base_interval.get(key, 0) / max(base_interval_total, 1)
        print(f"  {key:<16}{obs:>10.1%}{base:>9.1%}{lift(obs, base):>8}")

    print(f"\n=== missed_onset ({missed_total}) vs all gold notes ===")
    print(f"  {'concurrency':<16}{'observed':>10}{'base':>9}{'lift':>8}")
    for key, value in missed_by_concurrency.most_common():
        obs = value / max(missed_total, 1)
        base = base_concurrency.get(key, 0) / max(base_notes, 1)
        print(f"  {key:<16}{obs:>10.1%}{base:>9.1%}{lift(obs, base):>8}")
    short_obs = missed_short / max(missed_total, 1)
    short_base = base_short / max(base_notes, 1)
    print(
        f"  {'short (<150ms)':<16}{short_obs:>10.1%}{short_base:>9.1%}"
        f"{lift(short_obs, short_base):>8}"
    )
    print("  by register (no base comparison — registers are not equally populated):")
    for key, value in missed_by_register.most_common():
        print(f"    {key:<16}{value:>7}{value / max(missed_total, 1):>8.1%}")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return 0


def _decode(
    events: list[Any],
    clip_gold: list[TabEvent],
    position: Any,
    sequence: Any,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> list[TabEvent]:
    """Production decode, returning the decoded events themselves."""
    from tabvision.eval.guitarset_audio import score_audio_only
    from tabvision.fusion.playability import set_transition_prior
    from tabvision.fusion.position_prior import apply_pitch_position_prior
    from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT

    prepared = apply_pitch_position_prior(list(events), position)
    set_transition_prior(sequence, weight=SEQUENCE_PRIOR_WEIGHT)
    try:
        scored = score_audio_only(prepared, clip_gold, cfg=cfg, session=session)
    finally:
        set_transition_prior(None)
    return list(scored.decoded)


if __name__ == "__main__":
    raise SystemExit(main())
