"""Accuracy-loop Q6 — full development-set OOF run for the physics channel.

The pilot measured +0.0525 on 20 clips with the weight and fit threshold
swept on that same set. This is the honest version: **all 300 GuitarSet
development clips (players 00-04), with the configuration frozen below before
the run started.** No sweep, no arm selection — one candidate against
baseline.

Frozen configuration (deliberately not tuned against this run's result):

    WEIGHT   = 1.0     evidence exponent in the product of experts
    MIN_R2   = 0.50    fit-quality floor below which the channel abstains
    TABLE    = specification-derived steel-string table, no offset

The stiffness table comes from ``stiffness_model_for_session``, so
out-of-domain sessions abstain structurally. Every clip here is clean
steel-string acoustic in standard tuning, so the table applies throughout.

Position prior is leave-one-player-out (``guitarset-v1`` was trained on these
very players) plus ``guitarset-seq-v1`` at w=4.0 — the shipped clean-acoustic
decode. Ensemble events are cached per clip, so an interrupted run resumes
where it stopped.
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
    DEV_PLAYERS,
    _event_from_json,
    _event_to_json,
    _score,
    build_oof_priors,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import load_mono_audio, parse_guitarset_jams
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.string_physics import stiffness_model_for_session
from tabvision.types import GuitarConfig, SessionConfig

WEIGHT = 1.0
MIN_R2 = 0.50
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def dev_clips(data_home: Path) -> list[str]:
    return sorted(
        path.stem
        for path in (data_home / "annotation").glob("*.jams")
        if path.stem[:2] in DEV_PLAYERS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_full_dev_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = GuitarConfig()
    session = SessionConfig()
    model = stiffness_model_for_session(session, cfg)
    if model is None:
        raise SystemExit("no stiffness table for this session — refusing to run")
    priors = build_oof_priors(data_home, cfg)

    clips = dev_clips(data_home)
    if args.limit:
        clips = clips[: args.limit]
    print(f"full dev run: {len(clips)} clips, weight={WEIGHT}, min_r2={MIN_R2}", flush=True)

    backend = None
    base_scores: list[dict[str, float]] = []
    arm_scores: list[dict[str, float]] = []
    base_decomps: list[ErrorDecomposition] = []
    arm_decomps: list[ErrorDecomposition] = []
    rows: list[dict[str, Any]] = []
    coverage = {"events": 0, "isolated": 0, "fitted": 0, "applied": 0}
    started = time.perf_counter()

    try:
        for index, track_id in enumerate(clips, start=1):
            cache = cache_dir / f"{track_id}.ensemble.json"
            wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
            if cache.is_file():
                events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
            else:
                if backend is None:
                    from tabvision.audio.highres_ensemble import HighResEnsembleBackend

                    backend = HighResEnsembleBackend()
                events = list(backend.transcribe(wav, int(sr), session))
                cache.write_text(
                    json.dumps([_event_to_json(e) for e in events], indent=1) + "\n",
                    encoding="utf-8",
                )
            gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
            prior = priors[track_id[:2]]

            base_metrics, base_decomp = _score(events, gold, cfg=cfg, session=session, prior=prior)
            moved, tally = attach_inharmonicity_evidence(
                events, wav, int(sr), model, cfg, weight=WEIGHT, min_r2=MIN_R2
            )
            arm_metrics, arm_decomp = _score(moved, gold, cfg=cfg, session=session, prior=prior)
            for key in coverage:
                coverage[key] += tally[key]

            base_scores.append(base_metrics)
            arm_scores.append(arm_metrics)
            base_decomps.append(base_decomp)
            arm_decomps.append(arm_decomp)
            rows.append(
                {
                    "track_id": track_id,
                    "mode": "solo" if track_id.endswith("_solo") else "comp",
                    "base_tab_f1": base_metrics["tab_f1"],
                    "arm_tab_f1": arm_metrics["tab_f1"],
                    "delta": arm_metrics["tab_f1"] - base_metrics["tab_f1"],
                    "applied": tally["applied"],
                }
            )
            if index % 10 == 0 or index == len(clips):
                elapsed = time.perf_counter() - started
                mean = float(np.mean([r["delta"] for r in rows]))
                print(
                    f"  [{index}/{len(clips)}] mean delta so far {mean:+.4f} "
                    f"({elapsed / 60:.1f} min)",
                    flush=True,
                )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()

    deltas = np.asarray([r["delta"] for r in rows], dtype=np.float64)
    solo = np.asarray([r["mode"] == "solo" for r in rows])
    ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    solo_ci = bootstrap_ci(deltas[solo], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    comp_ci = bootstrap_ci(deltas[~solo], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)

    summary = {
        "frozen_config": {"weight": WEIGHT, "min_r2": MIN_R2, "table": "physics (raw)"},
        "clips": len(rows),
        "coverage": coverage,
        "baseline_tab_f1": float(np.mean([s["tab_f1"] for s in base_scores])),
        "arm_tab_f1": float(np.mean([s["tab_f1"] for s in arm_scores])),
        "delta": float(deltas.mean()),
        "lo95": ci.lower,
        "hi95": ci.upper,
        "solo_delta": float(deltas[solo].mean()),
        "solo_lo95": solo_ci.lower,
        "solo_hi95": solo_ci.upper,
        "comp_delta": float(deltas[~solo].mean()),
        "comp_lo95": comp_ci.lower,
        "comp_hi95": comp_ci.upper,
        "baseline_onset_f1": float(np.mean([s["onset_f1"] for s in base_scores])),
        "arm_onset_f1": float(np.mean([s["onset_f1"] for s in arm_scores])),
        "baseline_pitch_f1": float(np.mean([s["pitch_f1"] for s in base_scores])),
        "arm_pitch_f1": float(np.mean([s["pitch_f1"] for s in arm_scores])),
        "base_decomposition": aggregate_decompositions(base_decomps).to_dict(),
        "arm_decomposition": aggregate_decompositions(arm_decomps).to_dict(),
        "per_clip": rows,
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nTab F1 {summary['baseline_tab_f1']:.4f} -> {summary['arm_tab_f1']:.4f}\n"
        f"delta {summary['delta']:+.4f} [{ci.lower:+.4f}, {ci.upper:+.4f}]\n"
        f"  solo {summary['solo_delta']:+.4f} [{solo_ci.lower:+.4f}, {solo_ci.upper:+.4f}]\n"
        f"  comp {summary['comp_delta']:+.4f} [{comp_ci.lower:+.4f}, {comp_ci.upper:+.4f}]\n"
        f"onset {summary['baseline_onset_f1']:.4f} -> {summary['arm_onset_f1']:.4f} | "
        f"pitch {summary['baseline_pitch_f1']:.4f} -> {summary['arm_pitch_f1']:.4f}\n"
        f"coverage {coverage}"
    )
    print(f"gate (lo-95 > 0): {'PASS' if ci.lower > 0 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
