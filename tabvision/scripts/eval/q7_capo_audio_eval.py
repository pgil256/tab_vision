"""Accuracy-loop Q7 (ROI deep-dive §4.3) — capo support on real audio.

The entry probe validated the capo-covariant transform at the label level.
This is the build slice's gate: does it move **Tab F1 through `fuse()`** on
audio that actually sounds capoed?

Synthetic capo, exactly as §4.3 specifies. GuitarSet audio is pitch-shifted
up ``C`` semitones and the labels are capo-shifted: a note ``(s0, f0, p0)``
becomes ``(s0, f0+C, p0+C)`` with ``cfg.capo = C``, i.e. the same shape played
``C`` frets up behind a capo. Label-exact augmentation, no NC issues.

Pitch-shifting is not free of artifacts, so the comparison is **paired on
identical shifted audio** — every arm sees the same signal and the same
transcription, and the delta isolates the prior. A capo-0 control arm on
unshifted audio bounds how much the shift itself costs.

Arms:

- ``today``      — no position prior, no sequence prior. This is literally
  what ``resolve_inference_policy`` does for capo>0 today (the two are
  coupled), so it is the honest current-behaviour baseline.
- ``covariant``  — capo-covariant position prior, no sequence prior. Isolates
  the lever §4.3 actually proposes.
- ``covariant+seq`` — adds ``guitarset-seq-v1``. **Caveat measured rather
  than assumed:** the registered artifact uses the ``delta_fret`` scheme,
  which conditions on the *absolute* previous-fret region, so unlike the pure
  ``P(Δstring | Δpitch)`` backbone it is **not** capo-invariant. Its
  fret-region lookups are shifted under a capo and back off to the delta
  table. This arm quantifies whether that costs anything.
- ``naive``      — capo-0 prior applied without the shift, the mistake a
  capo-unaware implementation would make.

Notes whose shifted fret exceeds ``max_fret`` are dropped from gold (they
would be unplayable); the count is reported.

Priors are **leave-one-player-out**. The registered ``guitarset-v1`` was
trained on players 00-04 and these clips are drawn from those players, so
using it would hand every prior arm an in-sample advantage — an early run
with it showed an implausible +0.45, which is what prompted the switch.

**Two stages, deliberately in separate processes.** Interleaving librosa's
pitch shift with repeated highres-backend loads in one process segfaults on
this machine (exit 139) — reproducible in the loop, but not when either is run
alone, so it is an interaction rather than a bug in either. ``--stage shift``
imports librosa and never torch; ``--stage eval`` imports torch and never
librosa. Both are cached and resumable, which is worth having anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import (
    _event_from_json,
    _event_to_json,
    build_oof_priors,
    select_clips,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import (
    load_mono_audio,
    parse_guitarset_jams,
    score_audio_only,
)
from tabvision.fusion.position_prior import (
    apply_pitch_position_prior,
    capo_covariant_prior,
)
from tabvision.pipeline import sequence_decode_context
from tabvision.types import GuitarConfig, SessionConfig, TabEvent

CAPOS = (2, 4)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
ARMS = ("today", "covariant", "covariant+seq", "naive")


def shift_gold(gold: list[TabEvent], capo: int, cfg: GuitarConfig) -> tuple[list[TabEvent], int]:
    """Re-label capo-0 gold as the same shape played behind a capo."""
    out: list[TabEvent] = []
    dropped = 0
    for event in gold:
        fret = event.fret + capo
        if fret > cfg.max_fret:
            dropped += 1
            continue
        out.append(
            TabEvent(
                onset_s=event.onset_s,
                duration_s=event.duration_s,
                string_idx=event.string_idx,
                fret=fret,
                pitch_midi=event.pitch_midi + capo,
                confidence=event.confidence,
            )
        )
    return out, dropped


def run_shift_stage(clips: list[str], data_home: Path, cache_dir: Path) -> None:
    """Generate pitch-shifted audio. Imports librosa; must not import torch."""
    import librosa

    for index, track_id in enumerate(clips, start=1):
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        for capo in CAPOS:
            target = cache_dir / f"{track_id}.shift{capo}.npy"
            if target.is_file():
                continue
            shifted = librosa.effects.pitch_shift(
                y=np.asarray(wav, dtype=np.float32), sr=int(sr), n_steps=capo
            )
            np.save(target, shifted)
        print(f"  [{index}/{len(clips)}] shifted {track_id}", flush=True)


def decode(events, gold, *, cfg, session, prior, sequence: str):
    prepared = list(events) if prior is None else apply_pitch_position_prior(list(events), prior)
    with sequence_decode_context(sequence):
        scored = score_audio_only(prepared, gold, cfg=cfg, session=session)
    from tabvision.eval.error_decomposition import decompose_errors

    return (
        {"tab_f1": scored.tab.f1, "onset_f1": scored.onset.f1, "pitch_f1": scored.pitch.f1},
        decompose_errors(scored.decoded, gold),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--control-cache", type=Path, default=None)
    parser.add_argument("--stage", choices=("shift", "eval"), default="eval")
    parser.add_argument("--comp-clips", type=int, default=10)
    parser.add_argument("--solo-clips", type=int, default=10)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q7_capo_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the full-dev ensemble cache for the unshifted control arm.
    control_cache = args.control_cache or (data_root / "models" / "q6_full_dev_cache")

    clips = select_clips(data_home, "comp", args.comp_clips)
    clips += select_clips(data_home, "solo", args.solo_clips)
    print(f"capo audio eval [{args.stage}]: {len(clips)} clips x capos {CAPOS}", flush=True)

    if args.stage == "shift":
        run_shift_stage(clips, data_home, cache_dir)
        return 0

    cfg0 = GuitarConfig()
    session = SessionConfig()
    # Leave-one-player-out, not the registered guitarset-v1: that artifact was
    # trained on players 00-04, which is exactly where these clips come from,
    # so using it would give every prior arm an in-sample advantage.
    oof_priors = build_oof_priors(data_home, cfg0)

    backend = None
    results: dict[int, dict[str, list[dict[str, float]]]] = {
        capo: {arm: [] for arm in ARMS} for capo in CAPOS
    }
    decomps: dict[int, dict[str, list[ErrorDecomposition]]] = {
        capo: {arm: [] for arm in ARMS} for capo in CAPOS
    }
    control: list[float] = []
    dropped_total = 0
    started = time.perf_counter()

    try:
        for index, track_id in enumerate(clips, start=1):
            wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
            gold0 = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg0)

            # Capo-0 control on unshifted audio: bounds the cost of the shift.
            cache0 = control_cache / f"{track_id}.ensemble.json"
            if cache0.is_file():
                ev0 = [_event_from_json(i) for i in json.loads(cache0.read_text("utf-8"))]
            else:
                if backend is None:
                    from tabvision.audio.highres_ensemble import HighResEnsembleBackend

                    backend = HighResEnsembleBackend()
                ev0 = list(backend.transcribe(wav, int(sr), session))
                cache0.write_text(
                    json.dumps([_event_to_json(e) for e in ev0], indent=1) + "\n", encoding="utf-8"
                )
            base_prior = oof_priors[track_id[:2]]
            m0, _ = decode(
                ev0,
                gold0,
                cfg=cfg0,
                session=session,
                prior=base_prior,
                sequence="guitarset-seq-v1",
            )
            control.append(m0["tab_f1"])

            for capo in CAPOS:
                cfg = GuitarConfig(capo=capo)
                gold, dropped = shift_gold(gold0, capo, cfg)
                dropped_total += dropped
                shift_path = cache_dir / f"{track_id}.shift{capo}.npy"
                if not shift_path.is_file():
                    raise SystemExit(f"missing {shift_path}; run --stage shift first")
                cache = cache_dir / f"{track_id}.capo{capo}.json"
                if cache.is_file():
                    events = [_event_from_json(i) for i in json.loads(cache.read_text("utf-8"))]
                else:
                    if backend is None:
                        from tabvision.audio.highres_ensemble import HighResEnsembleBackend

                        backend = HighResEnsembleBackend()
                    events = list(backend.transcribe(np.load(shift_path), int(sr), session))
                    cache.write_text(
                        json.dumps([_event_to_json(e) for e in events], indent=1) + "\n",
                        encoding="utf-8",
                    )

                covariant = capo_covariant_prior(base_prior, capo)
                arms = {
                    "today": (None, "none"),
                    "covariant": (covariant, "none"),
                    "covariant+seq": (covariant, "guitarset-seq-v1"),
                    "naive": (base_prior, "none"),
                }
                for arm, (prior, sequence) in arms.items():
                    metrics, decomposition = decode(
                        events, gold, cfg=cfg, session=session, prior=prior, sequence=sequence
                    )
                    results[capo][arm].append(metrics)
                    decomps[capo][arm].append(decomposition)

            if index % 5 == 0 or index == len(clips):
                elapsed = (time.perf_counter() - started) / 60
                snapshot = {
                    capo: np.mean(
                        [
                            a["tab_f1"] - b["tab_f1"]
                            for a, b in zip(
                                results[capo]["covariant"], results[capo]["today"], strict=True
                            )
                        ]
                    )
                    for capo in CAPOS
                }
                print(
                    f"  [{index}/{len(clips)}] covariant-today "
                    + " ".join(f"capo{c}={v:+.4f}" for c, v in snapshot.items())
                    + f" ({elapsed:.1f} min)",
                    flush=True,
                )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()

    summary: dict[str, Any] = {
        "clips": clips,
        "capos": list(CAPOS),
        "gold_notes_dropped_above_max_fret": dropped_total,
        "capo0_control_tab_f1": float(np.mean(control)),
        "results": {},
    }
    for capo in CAPOS:
        today = np.asarray([m["tab_f1"] for m in results[capo]["today"]], dtype=np.float64)
        entry: dict[str, Any] = {}
        for arm in ARMS:
            values = np.asarray([m["tab_f1"] for m in results[capo][arm]], dtype=np.float64)
            delta = values - today
            ci = bootstrap_ci(delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            entry[arm] = {
                "tab_f1": float(values.mean()),
                "delta_vs_today": float(delta.mean()),
                "lo95": ci.lower,
                "hi95": ci.upper,
                "onset_f1": float(np.mean([m["onset_f1"] for m in results[capo][arm]])),
                "pitch_f1": float(np.mean([m["pitch_f1"] for m in results[capo][arm]])),
                "decomposition": aggregate_decompositions(decomps[capo][arm]).to_dict(),
            }
        summary["results"][str(capo)] = entry

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\ncapo-0 control (unshifted, full priors) Tab F1 = {summary['capo0_control_tab_f1']:.4f}"
    )
    print(f"gold notes dropped above max_fret: {dropped_total}")
    for capo in CAPOS:
        print(f"\ncapo {capo}:")
        for arm in ARMS:
            row = summary["results"][str(capo)][arm]
            print(
                f"  {arm:>14}: tab={row['tab_f1']:.4f} "
                f"delta_vs_today={row['delta_vs_today']:+.4f} "
                f"[{row['lo95']:+.4f}, {row['hi95']:+.4f}] "
                f"pitch={row['pitch_f1']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
