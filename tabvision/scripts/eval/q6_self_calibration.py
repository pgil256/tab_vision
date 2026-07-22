"""Accuracy-loop Q6 — does the physics channel generalize to any guitar?

The pilot integration lifted Tab F1 by +0.0525, but its stiffness table was
fitted from *other GuitarSet players' labelled notes*. ``B0`` is a property of
the string set and scale length, so that table is an artefact of the
instruments in one dataset and would not transfer to a user's guitar. This
asks whether it has to.

``B ∝ 1/L²`` and the scale length is shared by all six strings, so a
different instrument mostly shifts the whole table. A recording should
therefore be able to calibrate itself: decode once without the physics, take
the provisional string assignments, re-fit ``B0`` from the recording's own
notes, then decode again with the physics term.

Arms:

- ``baseline`` — no inharmonicity evidence.
- ``lopo`` — the pilot: ``B0`` from the *other four players'* gold labels.
  Requires labelled data from a similar instrument; the thing we want to
  eliminate.
- ``self-seeded`` — first decode supplies provisional labels; ``B0`` re-fitted
  from this recording, using the LOPO table only as a fallback for strings
  with too few notes.
- ``self-blind`` — no seed at all, calibrated from the single clip. Nothing
  about any other guitar enters the calculation.
- ``self-pooled`` — **the product case.** No seed either, but calibration
  pools every clip from the same player, i.e. a few minutes of that one
  instrument. Still fully self-labelled: no gold, no other guitars.

If ``self-blind`` matches ``lopo``, the channel is instrument-agnostic and
the reference table can be deleted rather than shipped.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from scripts.eval.n2_muscriptor_merge import (
    _event_from_json,
    _score,
    build_oof_priors,
    select_clips,
)
from scripts.eval.q6_gate_a import LOG2, collect_measurements
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.fusion.inharmonicity import (
    StiffnessObservation,
    StringStiffnessModel,
    attach_inharmonicity_evidence,
    calibrate_from_session,
    measure_events,
)
from tabvision.fusion.string_physics import reference_stiffness_model
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
CALIBRATION_MIN_R2 = 0.70
WEIGHT = 1.0
MIN_R2 = 0.50


def lopo_models(rows: list[dict[str, Any]]) -> dict[str, StringStiffnessModel]:
    usable = [row for row in rows if row["r2"] >= CALIBRATION_MIN_R2]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_player[row["player"]].append(row)
    models: dict[str, StringStiffnessModel] = {}
    for held_out in DEV_PLAYERS:
        train = [r for p, items in by_player.items() if p != held_out for r in items]
        table: dict[int, float] = {}
        for string in range(6):
            values = [r["log_b"] - (r["fret"] / 6.0) * LOG2 for r in train if r["string"] == string]
            if values:
                table[string] = float(np.median(values))
        models[held_out] = StringStiffnessModel(log_b0=table)
    return models


def session_observations(
    events: list[AudioEvent], decoded: list[TabEvent], wav: np.ndarray, sr: int, cfg: GuitarConfig
) -> list[StiffnessObservation]:
    """Measure B, then label each note with the first pass's own assignment."""
    ordered = sorted(events, key=lambda event: event.onset_s)
    fits = measure_events(ordered, wav, sr, cfg)
    by_key: dict[tuple[int, int], TabEvent] = {
        (int(round(item.onset_s * 1000)), item.pitch_midi): item for item in decoded
    }
    observations: list[StiffnessObservation] = []
    for index, fit in fits.items():
        event = ordered[index]
        match = by_key.get((int(round(event.onset_s * 1000)), event.pitch_midi))
        if match is None:
            continue
        observations.append(
            StiffnessObservation(
                string_idx=match.string_idx, fret=match.fret, log_b=fit.log_b, r2=fit.r2
            )
        )
    return observations


def run(
    clips: list[str], *, data_home: Path, workdir: Path, seeds: dict[str, StringStiffnessModel]
):
    cfg = GuitarConfig()
    session = SessionConfig()
    priors = build_oof_priors(data_home, cfg)
    arms = (
        "baseline",
        "lopo",
        "self-seeded",
        "self-blind",
        "self-pooled",
        "physics",
        "physics+offset",
    )

    scores: dict[str, list[dict[str, float]]] = {name: [] for name in arms}
    decomps: dict[str, list[ErrorDecomposition]] = {name: [] for name in arms}
    per_clip: list[dict[str, Any]] = []
    calibration_shift: list[float] = []

    # Phase 1: measure once per clip and pool observations by player, so the
    # pooled arm can calibrate from a few minutes of one instrument.
    pooled: dict[str, list[StiffnessObservation]] = defaultdict(list)
    cached: dict[str, Any] = {}
    for track_id in clips:
        events = [
            _event_from_json(item)
            for item in json.loads((workdir / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        wav, sr = sf.read(
            data_home / "audio_mono-mic" / f"{track_id}_mic.wav", dtype="float32", always_2d=False
        )
        player = track_id[:2]
        decoded = base_decomp_events(events, gold, cfg, session, priors[player])
        observations = session_observations(events, decoded, wav, int(sr), cfg)
        pooled[player].extend(observations)
        cached[track_id] = (events, gold, wav, int(sr), observations)
    # Specification-derived table, plus the same table shifted by one
    # scalar: level error and shape error have very different fixes.
    physics_raw = reference_stiffness_model()
    all_seed = {
        s: float(np.median([seeds[p].log_b0[s] for p in seeds if s in seeds[p].log_b0]))
        for s in range(6)
    }
    shift = float(np.median([physics_raw.log_b0[s] - all_seed[s] for s in all_seed]))
    physics_models = {
        "raw": physics_raw,
        "offset": StringStiffnessModel(
            log_b0={s: v - shift for s, v in physics_raw.log_b0.items()}
        ),
    }
    pooled_models = {
        player: calibrate_from_session(items, seed=None, min_r2=MIN_R2)
        for player, items in pooled.items()
    }

    for track_id in clips:
        events = [
            _event_from_json(item)
            for item in json.loads((workdir / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        wav, sr = sf.read(
            data_home / "audio_mono-mic" / f"{track_id}_mic.wav", dtype="float32", always_2d=False
        )
        player = track_id[:2]
        prior = priors[player]
        seed = seeds[player]
        row: dict[str, Any] = {
            "track_id": track_id,
            "mode": "solo" if track_id.endswith("_solo") else "comp",
        }

        # Pass 1: no physics. Its decode supplies the provisional labels.
        base_metrics, base_decomp = _score(events, gold, cfg=cfg, session=session, prior=prior)
        scores["baseline"].append(base_metrics)
        decomps["baseline"].append(base_decomp)
        row["baseline"] = base_metrics

        observations = session_observations(
            events, base_decomp_events(events, gold, cfg, session, prior), wav, int(sr), cfg
        )
        models = {
            "lopo": seed,
            "self-seeded": calibrate_from_session(observations, seed=seed, min_r2=MIN_R2),
            "self-blind": calibrate_from_session(observations, seed=None, min_r2=MIN_R2),
            "self-pooled": pooled_models.get(player),
            "physics": physics_models["raw"],
            "physics+offset": physics_models["offset"],
        }
        blind = models["self-blind"]
        if blind is not None:
            shared = [
                blind.log_b0[s] - seed.log_b0[s]
                for s in range(6)
                if s in blind.log_b0 and s in seed.log_b0
            ]
            if shared:
                calibration_shift.append(float(np.median(shared)))

        for name in (
            "lopo",
            "self-seeded",
            "self-blind",
            "self-pooled",
            "physics",
            "physics+offset",
        ):
            model = models[name]
            if model is None:
                scores[name].append(base_metrics)
                decomps[name].append(base_decomp)
                row[name] = base_metrics
                continue
            prepared, _tally = attach_inharmonicity_evidence(
                events, wav, int(sr), model, cfg, weight=WEIGHT, min_r2=MIN_R2
            )
            metrics, decomposition = _score(prepared, gold, cfg=cfg, session=session, prior=prior)
            scores[name].append(metrics)
            decomps[name].append(decomposition)
            row[name] = metrics
        per_clip.append(row)
        print(f"{track_id}: " + " ".join(f"{n}={row[n]['tab_f1']:.4f}" for n in arms), flush=True)

    baseline = np.asarray([r["tab_f1"] for r in scores["baseline"]], dtype=np.float64)
    solo_mask = np.asarray([r["mode"] == "solo" for r in per_clip])
    out: dict[str, Any] = {}
    for name in arms:
        tab = np.asarray([r["tab_f1"] for r in scores[name]], dtype=np.float64)
        delta = tab - baseline
        ci = bootstrap_ci(delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        solo_ci = bootstrap_ci(delta[solo_mask], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        out[name] = {
            "tab_f1": float(tab.mean()),
            "delta": float(delta.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
            "solo_delta": float(delta[solo_mask].mean()),
            "solo_lo95": solo_ci.lower,
            "solo_hi95": solo_ci.upper,
            "decomposition": aggregate_decompositions(decomps[name]).to_dict(),
        }
    return {
        "clips": clips,
        "arms": out,
        "per_clip": per_clip,
        "median_blind_vs_lopo_log_b0_shift": (
            float(np.median(calibration_shift)) if calibration_shift else None
        ),
    }


def base_decomp_events(events, gold, cfg, session, prior):
    """Re-run the first pass purely to recover its decoded TabEvents."""
    from scripts.eval.n2_muscriptor_merge import _score as score_again  # noqa: F401
    from tabvision.eval.guitarset_audio import score_audio_only
    from tabvision.fusion.position_prior import apply_pitch_position_prior
    from tabvision.pipeline import sequence_decode_context

    prepared = apply_pitch_position_prior(list(events), prior)
    with sequence_decode_context("guitarset-seq-v1"):
        return list(score_audio_only(prepared, gold, cfg=cfg, session=session).decoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    workdir = args.workdir or (data_root / "models" / "muscriptor_probe")

    print("building LOPO reference tables (for the lopo arm and seeding)...", flush=True)
    seeds = lopo_models(collect_measurements(data_home, DEV_PLAYERS, 0, "mono"))
    clips = select_clips(data_home, "comp", 10) + select_clips(data_home, "solo", 10)
    summary = run(clips, data_home=data_home, workdir=workdir, seeds=seeds)

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    for name, row in summary["arms"].items():
        print(
            f"  {name:>12}: tab={row['tab_f1']:.4f} delta={row['delta']:+.4f} "
            f"[{row['lo95']:+.4f}, {row['hi95']:+.4f}] | "
            f"solo {row['solo_delta']:+.4f} [{row['solo_lo95']:+.4f}, {row['solo_hi95']:+.4f}]"
        )
    print(
        f"\nmedian log-B0 shift, self-blind vs LOPO: {summary['median_blind_vs_lopo_log_b0_shift']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
