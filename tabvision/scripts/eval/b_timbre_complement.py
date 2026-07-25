"""Track B — is there room for a timbral model where the physics channel abstains?

Track B was scoped as "fill the empty ``guitarset-timbre-v1`` slot". Before
spending anything on that, two prior closures have to be respected, because
neither was narrow:

- **Phase 2 (2026-07-14)** trained a 35,905-parameter audio ranker on 35,959
  OOF pitch-correct events. Prior-only scored 0.6548; the audio model scored
  **0.6331** and a feature-only variant **0.6027** — both *worse* than using no
  audio at all. Calibration was healthy (ECE 0.0597, all six strings active),
  so the verdict was explicitly "lack of transferable timbral lift", not a
  training bug.
- **Phase 4 (2026-07-16)** went to native 44.1 kHz with multi-resolution
  harmonic envelopes through Nyquist, pick-noise, centroid, rolloff, decay,
  **inharmonicity**, and raw spectral slopes, over 56,742 physically adjacent
  gold-vs-alternative pairs. Position + native audio reached 0.6621 against a
  0.6548 comparator: **+0.0072 [-0.0152, +0.0291]** against a +0.05 gate, with
  player 00 regressing beyond the allowed bound. It closed the compact timbral
  path and said in terms: do not enlarge the window or model, do not tune on
  the failure set, do not open player 05.

Retraining a bigger version of that is exactly what those entries forbid, and
the second one already contained inharmonicity as a feature while the *physics*
channel — same underlying quantity, derived from specifications rather than
fitted — later proved worth +0.05 to +0.07 on its own. That is a strong hint
that the limitation was never model capacity.

**So this probe asks the one question neither closure answers.** Both measured
timbral lift over the position prior across *all* ambiguous notes. Neither
measured it on the population that actually matters now: the ~75% of notes where
the physics channel **abstains**. If a timbral model has a niche, that is
exactly and only where it is.

Two things get measured, both offline against banked fits, no training:

1. **Is the abstain population adversarially selected?** Compare covered vs
   abstain notes on duration and simultaneity. The physics channel abstains when
   it cannot read partials — which happens on short, masked notes. If those are
   also the notes with the worst timbral signal-to-noise, then the complement is
   selected *against* every spectral method, and no timbral model can work there
   however it is built. That would explain both prior closures rather than
   merely adding a third.

2. **What is the oracle ceiling on the complement?** Give the fusion the gold
   string for abstain notes only and score. This is the absolute upper bound for
   any per-note string evidence on that population — the ceiling a perfect
   classifier would reach. The repo's own rule from N4: compute the oracle
   first; if the ceiling is small, no estimator can rescue it.

A small ceiling closes Track B in one iteration. A large ceiling does not open
it — it means the ceiling was never the binding constraint, and the two prior
closures already measured what is reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.a_physics_coverage import load_banked
from scripts.eval.n2_muscriptor_merge import _event_from_json
from scripts.eval.phase0_rotation_baseline import (
    BURNED_PLAYER,
    DEV_PLAYERS,
    build_loo_priors,
    gold_by_player,
    score_leak_free,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import apply_fits, isolation_flags
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
ONSET_TOL_S = 0.05

FROZEN = {"baseline": 0.6083, "shipped": 0.6801}
FROZEN_TOLERANCE = 0.0015


def gold_string_for(event: AudioEvent, gold: list[TabEvent]) -> int | None:
    """The gold string for a detected event, matched on pitch within tolerance."""
    best: TabEvent | None = None
    best_dt = ONSET_TOL_S + 1e-9
    for note in gold:
        if note.pitch_midi != event.pitch_midi:
            continue
        dt = abs(note.onset_s - event.onset_s)
        if dt < best_dt:
            best = note
            best_dt = dt
    return None if best is None else best.string_idx


def oracle_prior(event: AudioEvent, string_idx: int, cfg: GuitarConfig) -> np.ndarray | None:
    """A candidate distribution that puts all mass on the gold string.

    This is the perfect-classifier upper bound: not "a very good timbral model"
    but "a timbral model that is never wrong". Nothing achievable can exceed it.
    """
    candidates = candidate_positions(event.pitch_midi, cfg)
    if not candidates:
        return None
    matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    hit = False
    for candidate in candidates:
        if candidate.string_idx == string_idx:
            matrix[candidate.string_idx, candidate.fret] = 1.0
            hit = True
    if not hit:
        return None
    return matrix


def ambiguous(event: AudioEvent, cfg: GuitarConfig) -> bool:
    """More than one playable position for this pitch."""
    return len(candidate_positions(event.pitch_midi, cfg)) > 1


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
    session = SessionConfig()
    evidence = load_string_evidence()
    table = reference_stiffness_model()
    isolation = evidence.isolation

    gold = gold_by_player(data_home, cfg)
    clips = sorted(t for p in DEV_PLAYERS for t in gold[p])
    if args.limit:
        clips = clips[: args.limit]

    missing = [t for t in clips if not (fit_cache / f"{t}.{isolation}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips unbanked; run track A's `bank` first")

    print("building leave-one-player-out priors...", flush=True)
    positions, sequences = build_loo_priors(gold, cfg)

    arms = ("baseline", "shipped", "oracle_abstain", "oracle_covered", "oracle_all")
    scores: dict[str, list[float]] = {arm: [] for arm in arms}

    covered_stats: dict[str, list[float]] = {"duration": [], "concurrency": []}
    abstain_stats: dict[str, list[float]] = {"duration": [], "concurrency": []}
    counts = {"ambiguous": 0, "covered": 0, "abstain": 0, "abstain_no_gold": 0}

    for index, track_id in enumerate(clips, start=1):
        player = track_id[:2]
        cache = sealed_cache if player == BURNED_PLAYER else dev_cache
        events = [
            _event_from_json(item)
            for item in json.loads((cache / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        ordered = sorted(events, key=lambda event: event.onset_s)
        banked = load_banked(fit_cache / f"{track_id}.{isolation}.json")
        clip_gold = gold[player][track_id]
        flags = isolation_flags(ordered)

        # Which ambiguous events does the shipped channel actually touch?
        covered_idx: set[int] = set()
        for i, fit in banked.items():
            if fit.r2 >= evidence.min_r2 and i < len(ordered) and ambiguous(ordered[i], cfg):
                covered_idx.add(i)

        shipped_events, _ = apply_fits(
            ordered,
            banked,
            table,
            cfg,
            weight=evidence.weight,
            min_r2=evidence.min_r2,
            sigma=evidence.sigma,
            isolation=isolation,
        )

        # Population characterisation, and the two oracle arms.
        oracle_abstain = list(shipped_events)
        oracle_covered = list(shipped_events)
        oracle_all = list(shipped_events)
        for i, event in enumerate(ordered):
            if not ambiguous(event, cfg):
                continue
            counts["ambiguous"] += 1
            duration = event.offset_s - event.onset_s
            # Concurrency from the detected stream, which is what inference sees.
            concurrency = sum(
                1
                for j, other in enumerate(ordered)
                if j != i and other.onset_s < event.offset_s and other.offset_s > event.onset_s
            )
            bucket = covered_stats if i in covered_idx else abstain_stats
            bucket["duration"].append(duration)
            bucket["concurrency"].append(float(concurrency))
            counts["covered" if i in covered_idx else "abstain"] += 1

            string_idx = gold_string_for(event, clip_gold)
            if string_idx is None:
                if i not in covered_idx:
                    counts["abstain_no_gold"] += 1
                continue
            matrix = oracle_prior(event, string_idx, cfg)
            if matrix is None:
                continue
            from dataclasses import replace as _replace

            oracle_all[i] = _replace(oracle_all[i], fret_prior=matrix)
            if i in covered_idx:
                oracle_covered[i] = _replace(oracle_covered[i], fret_prior=matrix)
            else:
                oracle_abstain[i] = _replace(oracle_abstain[i], fret_prior=matrix)

        for arm, arm_events in (
            ("baseline", ordered),
            ("shipped", shipped_events),
            ("oracle_abstain", oracle_abstain),
            ("oracle_covered", oracle_covered),
            ("oracle_all", oracle_all),
        ):
            metrics, _ = score_leak_free(
                arm_events,
                clip_gold,
                position=positions[player],
                sequence=sequences[player],
                cfg=cfg,
                session=session,
            )
            scores[arm].append(metrics["tab_f1"])

        if index % 50 == 0 or index == len(clips):
            print(f"  [{index}/{len(clips)}]", flush=True)

    _ = flags  # isolation flags retained for future population splits

    def summarise(stats: dict[str, list[float]]) -> dict[str, float]:
        if not stats["duration"]:
            return {"n": 0}
        return {
            "n": len(stats["duration"]),
            "median_duration_s": statistics.median(stats["duration"]),
            "mean_concurrency": statistics.fmean(stats["concurrency"]),
            "share_short": sum(1 for d in stats["duration"] if d < 0.15) / len(stats["duration"]),
            "share_masked": sum(1 for c in stats["concurrency"] if c >= 3)
            / len(stats["concurrency"]),
        }

    covered = summarise(covered_stats)
    abstain = summarise(abstain_stats)

    ship = np.asarray(scores["shipped"], dtype=np.float64)
    results: dict[str, Any] = {
        "clips": len(clips),
        "counts": counts,
        "population": {"covered": covered, "abstain": abstain},
        "arms": {},
    }

    print(f"\n{'arm':<18}{'Tab F1':>9}{'vs shipped':>12}{'lo95':>9}{'hi95':>9}")
    for arm in arms:
        values = np.asarray(scores[arm], dtype=np.float64)
        deltas = values - ship
        ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        results["arms"][arm] = {
            "tab_f1": float(values.mean()),
            "delta_vs_shipped": float(deltas.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
        }
        print(
            f"{arm:<18}{values.mean():>9.4f}{deltas.mean():>+12.4f}"
            f"{ci.lower:>+9.4f}{ci.upper:>+9.4f}"
        )

    print("\npopulation (ambiguous notes):")
    print(f"  {'':<10}{'n':>8}{'med dur':>10}{'mean conc':>11}{'short':>8}{'masked':>8}")
    for name, stats in (("covered", covered), ("abstain", abstain)):
        if stats.get("n"):
            print(
                f"  {name:<10}{stats['n']:>8}{stats['median_duration_s']:>10.3f}"
                f"{stats['mean_concurrency']:>11.2f}{stats['share_short']:>8.1%}"
                f"{stats['share_masked']:>8.1%}"
            )

    drift_b = results["arms"]["baseline"]["tab_f1"] - FROZEN["baseline"]
    drift_s = results["arms"]["shipped"]["tab_f1"] - FROZEN["shipped"]
    results["frozen_drift"] = {"baseline": drift_b, "shipped": drift_s}
    print(f"\nfrozen drift: baseline {drift_b:+.4f}  shipped {drift_s:+.4f}")
    if not args.limit and (abs(drift_b) > FROZEN_TOLERANCE or abs(drift_s) > FROZEN_TOLERANCE):
        raise SystemExit("drift from the Phase 0 frozen baseline; nothing above is trustworthy")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
