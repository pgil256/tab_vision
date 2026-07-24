"""Player-05 sealed confirmation — a 2x2 over table and isolation mode.

Player-05 is the held-out player the accuracy program keeps sealed: opened
only after config freeze and an explicit user proceed. Both conditions are
met. **Every look at a sealed set spends some of its value**, so this run
batches every open question rather than asking them one at a time.

Nothing is tuned here. Weight, fit threshold and sigma are the registered
artifact's own hash-verified values; the arms differ only in the two things
under decision.

Design
------
A full 2x2 plus baseline:

              strict            partial_aware
    raw       raw-strict        raw-partial
    corrected corrected-strict  corrected-partial

The first pass ran only three of the four cells (it omitted ``raw-partial``).
That was enough to refute the level correction but not enough to say what to
ship once the correction is reverted, because N1 had been measured only on top
of the corrected table. Both factors are crossed here so neither answer is
conditional on the other's disposition.

Pre-declared readings
---------------------
**Gate re-seal** (``raw-strict`` vs ``baseline``). Reproduces the Q6 gate on
the artifact as it actually shipped at gate time. PASS if lo-95 > 0.

**Level correction out-of-distribution** (``corrected`` vs ``raw``, at both
isolation modes). The +0.60 constant was located on GuitarSet dev clips
(players 00-04); this is its only out-of-distribution test. Paired on identical
events, so it is far tighter than either arm against baseline.
  * CONFIRMED — delta > 0 and lo-95 > 0
  * DIRECTIONAL — delta > 0, lo-95 <= 0 (right sign; 60 clips cannot resolve an
    effect this size, so the dev estimate stands as the magnitude)
  * REFUTED — delta < 0. The correction does not generalize; revert it.

**N1 partial-aware isolation** (``partial`` vs ``strict``, at both tables).
PASS if lo-95 > 0.

Honest power note, stated *before* the first run: dev measured the correction
at +0.0160 [+0.0088, +0.0233]. Q6's player-05 confirmation on 60 clips produced
a CI of width ~0.058 against baseline. An unpaired effect of +0.016 is
therefore below what 60 clips can resolve; DIRECTIONAL was the expected good
outcome for that leg, and only a negative sign is evidence against it.

Ensemble events come from the Q6 cache, so no re-transcription happens and the
arms score identical detections — every delta is attributable to the table or
the isolation mode alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import _event_from_json, _score
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.guitarset_audio import load_mono_audio, parse_guitarset_jams
from tabvision.fusion.inharmonicity import StringStiffnessModel, attach_inharmonicity_evidence
from tabvision.fusion.position_prior import load_pitch_position_prior
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.types import GuitarConfig, SessionConfig

HELD_OUT_PLAYER = "05"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

LEVEL_CORRECTION_LOG_B = 0.60
"""The correction under test. Local to this script by design.

It briefly lived in ``string_physics`` as a shipped constant; this run refuted
it and it was reverted, so the only place it survives is the experiment that
killed it. Re-running this script reproduces the refutation without
reintroducing the constant to the library.
"""


def level_corrected(model: StringStiffnessModel) -> StringStiffnessModel:
    """``model`` shifted by a uniform offset in log-B."""
    return StringStiffnessModel(
        log_b0={index: value + LEVEL_CORRECTION_LOG_B for index, value in model.log_b0.items()},
        fret_exponent=model.fret_exponent,
    )


ARMS = (
    "baseline",
    "raw-strict",
    "corrected-strict",
    "raw-partial",
    "corrected-partial",
)

# A full 2x2 over (table, isolation) plus baseline. The first pass ran only
# three of the four cells, which was enough to refute the level correction but
# not enough to say what to ship once it is reverted: N1 had been measured only
# on top of the corrected table. The missing cell is the one that matters.
GRID = (
    ("raw-strict", "raw", "strict"),
    ("corrected-strict", "corrected", "strict"),
    ("raw-partial", "raw", "partial_aware"),
    ("corrected-partial", "corrected", "partial_aware"),
)

COMPARISONS = (
    ("gate re-seal (raw)", "raw-strict", "baseline"),
    ("level correction OOD", "corrected-strict", "raw-strict"),
    ("level correction | partial", "corrected-partial", "raw-partial"),
    ("N1 | raw table", "raw-partial", "raw-strict"),
    ("N1 | corrected table", "corrected-partial", "corrected-strict"),
    ("best vs baseline", "raw-partial", "baseline"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_player05_cache")

    cfg = GuitarConfig()
    session = SessionConfig()

    # Decode settings come from the registered artifact, hash-verified, so
    # this run cannot silently score a different configuration than ships.
    evidence = load_string_evidence()
    raw_model = reference_stiffness_model()
    corrected_model = level_corrected(raw_model)
    prior = load_pitch_position_prior("guitarset-v1")

    clips = sorted(
        path.stem
        for path in (data_home / "annotation").glob("*.jams")
        if path.stem[:2] == HELD_OUT_PLAYER
    )
    missing = [t for t in clips if not (cache_dir / f"{t}.ensemble.json").is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} clips are not cached; this run refuses to re-transcribe "
            f"(first missing: {missing[0]})"
        )

    print(
        f"player-05 batched confirm: {len(clips)} clips x {len(ARMS)} arms\n"
        f"  frozen decode: weight={evidence.weight} min_r2={evidence.min_r2} "
        f"sigma={evidence.sigma} (from registered artifact)\n"
        f"  level correction: {LEVEL_CORRECTION_LOG_B:+.2f} log-B\n"
        f"  position prior: guitarset-v1 (registered, excludes player 05)",
        flush=True,
    )

    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    coverage: dict[str, dict[str, int]] = {
        arm: {"events": 0, "isolated": 0, "fitted": 0, "applied": 0}
        for arm in ARMS
        if arm != "baseline"
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, track_id in enumerate(clips, start=1):
        cache = cache_dir / f"{track_id}.ensemble.json"
        events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)

        row: dict[str, Any] = {
            "track_id": track_id,
            "mode": "solo" if track_id.endswith("_solo") else "comp",
        }

        base_metrics, _ = _score(events, gold, cfg=cfg, session=session, prior=prior)
        scores["baseline"].append(base_metrics["tab_f1"])
        row["baseline"] = base_metrics["tab_f1"]

        tables = {"raw": raw_model, "corrected": corrected_model}
        for arm, table, isolation in GRID:
            model = tables[table]
            moved, tally = attach_inharmonicity_evidence(
                events,
                wav,
                int(sr),
                model,
                cfg,
                weight=evidence.weight,
                min_r2=evidence.min_r2,
                sigma=evidence.sigma,
                isolation=isolation,
            )
            metrics, _ = _score(moved, gold, cfg=cfg, session=session, prior=prior)
            scores[arm].append(metrics["tab_f1"])
            row[arm] = metrics["tab_f1"]
            for key in coverage[arm]:
                coverage[arm][key] += tally[key]

        rows.append(row)
        if index % 10 == 0 or index == len(clips):
            elapsed = (time.perf_counter() - started) / 60.0
            lead = float(np.mean([r["raw-strict"] - r["baseline"] for r in rows]))
            print(f"  [{index}/{len(clips)}] raw vs baseline {lead:+.4f} ({elapsed:.1f} min)")

    results: dict[str, Any] = {
        "held_out_player": HELD_OUT_PLAYER,
        "clips": len(rows),
        "frozen_decode": {
            "weight": evidence.weight,
            "min_r2": evidence.min_r2,
            "sigma": evidence.sigma,
            "source": "registered acoustic-physics-v1 artifact",
        },
        "level_correction_log_b": LEVEL_CORRECTION_LOG_B,
        "arm_tab_f1": {arm: float(np.mean(scores[arm])) for arm in ARMS},
        "coverage": coverage,
        "comparisons": {},
        "per_clip": rows,
    }

    print(f"\n{'arm':<20}{'Tab F1':>10}")
    for arm in ARMS:
        print(f"{arm:<20}{float(np.mean(scores[arm])):>10.4f}")

    print(f"\n{'comparison':<24}{'vs':<18}{'delta':>9}{'lo95':>9}{'hi95':>9}  reading")
    solo_mask = np.asarray([r["mode"] == "solo" for r in rows])
    for label, arm, reference in COMPARISONS:
        deltas = np.asarray(scores[arm], dtype=np.float64) - np.asarray(
            scores[reference], dtype=np.float64
        )
        ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        mean = float(deltas.mean())
        if label == "level correction OOD":
            reading = "REFUTED" if mean < 0 else ("CONFIRMED" if ci.lower > 0 else "DIRECTIONAL")
        else:
            reading = "PASS" if ci.lower > 0 else "FAIL"
        solo_ci = bootstrap_ci(deltas[solo_mask], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        comp_ci = bootstrap_ci(deltas[~solo_mask], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        results["comparisons"][label] = {
            "arm": arm,
            "reference": reference,
            "delta": mean,
            "lo95": ci.lower,
            "hi95": ci.upper,
            "reading": reading,
            "solo_delta": float(deltas[solo_mask].mean()),
            "solo_lo95": solo_ci.lower,
            "solo_hi95": solo_ci.upper,
            "comp_delta": float(deltas[~solo_mask].mean()),
            "comp_lo95": comp_ci.lower,
            "comp_hi95": comp_ci.upper,
        }
        print(f"{label:<24}{reference:<18}{mean:+9.4f}{ci.lower:+9.4f}{ci.upper:+9.4f}  {reading}")

    print("\ncoverage (applied / fitted / isolated / events):")
    for arm, tally in coverage.items():
        print(
            f"  {arm:<18}{tally['applied']:>6} /{tally['fitted']:>6} /"
            f"{tally['isolated']:>6} /{tally['events']:>6}"
        )

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
