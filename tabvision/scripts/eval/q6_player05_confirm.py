"""Accuracy-loop Q6 — player-05 sealed confirmation for the physics channel.

The final gate. Player-05 is the held-out player the whole accuracy program
keeps sealed: opened only after config freeze and an explicit user proceed.
Both conditions are met — the full-dev OOF run passed +0.0443
[+0.0339, +0.0555] with the configuration frozen, and the user authorized
this run.

**Nothing is tuned here.** Weight, fit threshold and table are the identical
frozen values the full-dev run used; this file exists only to score them on
data none of the calibration ever touched:

    WEIGHT = 1.0     MIN_R2 = 0.50     TABLE = physics (raw)

Two things make player-05 a clean test of *this* channel specifically:

- the position prior is the **registered** ``guitarset-v1`` (trained on
  players 00-04, player 05 excluded per its manifest), so the decoder's prior
  never saw this player;
- the stiffness table is specification-derived, so it depends on no player at
  all — player 05 is not special to it, which is the whole point of the
  physics route.

Ensemble events cache per clip, so the run is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import _event_from_json, _event_to_json, _score
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import load_mono_audio, parse_guitarset_jams
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.position_prior import load_pitch_position_prior
from tabvision.fusion.string_physics import stiffness_model_for_session
from tabvision.types import GuitarConfig, SessionConfig

HELD_OUT_PLAYER = "05"
WEIGHT = 1.0
MIN_R2 = 0.50
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_player05_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = GuitarConfig()
    session = SessionConfig()
    model = stiffness_model_for_session(session, cfg)
    if model is None:
        raise SystemExit("no stiffness table for this session — refusing to run")
    prior = load_pitch_position_prior("guitarset-v1")

    clips = sorted(
        path.stem
        for path in (data_home / "annotation").glob("*.jams")
        if path.stem[:2] == HELD_OUT_PLAYER
    )
    print(
        f"player-05 confirm: {len(clips)} clips, weight={WEIGHT}, min_r2={MIN_R2}, "
        f"registered guitarset-v1 prior",
        flush=True,
    )

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
        "held_out_player": HELD_OUT_PLAYER,
        "frozen_config": {"weight": WEIGHT, "min_r2": MIN_R2, "table": "physics (raw)"},
        "position_prior": "guitarset-v1 (registered, excludes player 05)",
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
        f"\nplayer-05 Tab F1 {summary['baseline_tab_f1']:.4f} -> {summary['arm_tab_f1']:.4f}\n"
        f"delta {summary['delta']:+.4f} [{ci.lower:+.4f}, {ci.upper:+.4f}]\n"
        f"  solo {summary['solo_delta']:+.4f} [{solo_ci.lower:+.4f}, {solo_ci.upper:+.4f}]\n"
        f"  comp {summary['comp_delta']:+.4f} [{comp_ci.lower:+.4f}, {comp_ci.upper:+.4f}]\n"
        f"onset {summary['baseline_onset_f1']:.4f} -> {summary['arm_onset_f1']:.4f} | "
        f"pitch {summary['baseline_pitch_f1']:.4f} -> {summary['arm_pitch_f1']:.4f}\n"
        f"coverage {coverage}"
    )
    print(f"confirmation (lo-95 > 0): {'PASS' if ci.lower > 0 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
