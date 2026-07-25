"""Track A — bank partial-aware fits, then sweep admission offline.

Coverage, not effect size, is the binding constraint on the physics channel.
Phase 0 measured it at **22.4%** of events: of 51,392 development events only
11,528 receive evidence. The channel is worth +0.05 to +0.07 aggregate on that
quarter, so the obvious question is what the other three quarters would be
worth — and the answer depends entirely on whether the fits we are currently
throwing away are informative or noisy.

Why this script exists in two halves
------------------------------------
The spectral fit is the expensive part (~3 min for 360 clips) and it does not
depend on any admission parameter. The admission decision — ``min_r2``, and
whether to admit at all — is free. So:

``bank``   runs :func:`measure_events` once per clip at the shipped
           ``partial_aware`` isolation and writes every fit to disk, including
           the ones today's ``min_r2`` rejects. **This is what the banked cache
           could not previously contain**: until 2026-07-25 ``measure_events``
           had no isolation parameter and was always strict, so a
           partial-aware cache was unrepresentable.

``sweep``  replays those banked fits through :func:`apply_fits` under whatever
           admission rule is being tested, scores the result, and reports Tab F1
           against the frozen Phase 0 baseline. Seconds per arm, no audio.

Arms
----
``shipped``          the current default, ``min_r2 = 0.5``, hard threshold.
``min_r2=X``         the same rule at other thresholds. Lowering it admits more
                     notes at lower fit quality; the question is where that
                     stops paying.
``confidence``       replaces the hard threshold with a soft one: the evidence
                     weight scales with fit quality, so a marginal fit
                     contributes proportionally rather than being discarded or
                     trusted in full. This is the arm the Phase 0 report
                     proposed — a binary gate on a continuous quality signal is
                     throwing away information at both ends.

Gate: paired ΔTab F1 against the frozen baseline on **dev only**, lo-95 > 0,
plus no drop in accuracy among the notes the shipped arm already covers —
coverage bought by admitting bad fits is not a gain. The sealed player is not
opened here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import _event_from_json
from scripts.eval.phase0_rotation_baseline import (
    BURNED_PLAYER,
    DEV_PLAYERS,
    build_loo_priors,
    gold_by_player,
    score_leak_free,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.inharmonicity import (
    InharmonicityFit,
    apply_fits,
    measure_events,
)
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.types import GuitarConfig, SessionConfig

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

FROZEN_BASELINE = {"dev_baseline": 0.6083, "dev_shipped": 0.6801}
"""Phase 0's frozen dev numbers. Pinned so a harness change cannot pass unseen."""

FROZEN_TOLERANCE = 0.0015


def bank_clip(
    track_id: str,
    *,
    data_home: Path,
    events_cache: Path,
    fit_cache: Path,
    isolation: str,
    cfg: GuitarConfig,
) -> int:
    """Measure and persist every fit for one clip. Returns the fit count."""
    target = fit_cache / f"{track_id}.{isolation}.json"
    if target.is_file():
        return len(json.loads(target.read_text("utf-8")))
    events = [
        _event_from_json(item)
        for item in json.loads((events_cache / f"{track_id}.ensemble.json").read_text("utf-8"))
    ]
    wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
    ordered = sorted(events, key=lambda event: event.onset_s)
    fits = measure_events(ordered, wav, int(sr), cfg, isolation=isolation)
    # `log_b` is a derived property, so `b` is what must round-trip.
    payload = {
        str(index): {
            "f0_hz": fit.f0_hz,
            "b": fit.b,
            "partials": fit.partials,
            "r2": fit.r2,
        }
        for index, fit in fits.items()
    }
    target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return len(payload)


def load_banked(path: Path) -> dict[int, InharmonicityFit]:
    raw = json.loads(path.read_text("utf-8"))
    return {
        int(index): InharmonicityFit(
            f0_hz=float(entry["f0_hz"]),
            b=float(entry["b"]),
            partials=int(entry["partials"]),
            r2=float(entry["r2"]),
        )
        for index, entry in raw.items()
    }


def confidence_scaled(
    fits: dict[int, InharmonicityFit], floor: float, weight: float
) -> tuple[dict[int, InharmonicityFit], dict[int, float]]:
    """Soft admission: keep everything above ``floor``, scale weight by quality.

    A hard ``min_r2`` treats an r2 of 0.49 and 0.01 identically (both discarded)
    and 0.51 and 0.99 identically (both trusted in full). Neither is right. The
    weight here rises linearly from 0 at ``floor`` to ``weight`` at r2 = 1.
    """
    kept = {index: fit for index, fit in fits.items() if fit.r2 >= floor}
    span = max(1.0 - floor, 1e-6)
    weights = {index: weight * (fit.r2 - floor) / span for index, fit in kept.items()}
    return kept, weights


def apply_weighted(
    ordered: list[Any],
    fits: dict[int, InharmonicityFit],
    weights: dict[int, float],
    model: Any,
    cfg: GuitarConfig,
    *,
    sigma: float,
    isolation: str,
) -> tuple[list[Any], dict[str, int]]:
    """`apply_fits` per event with a per-event weight.

    Done one event at a time so each carries its own weight; the shared scoring
    half still does the actual work, so this cannot drift from production.
    """
    out: list[Any] = []
    tally = {"events": len(ordered), "isolated": 0, "fitted": 0, "applied": 0}
    for index, event in enumerate(ordered):
        fit = fits.get(index)
        weight = weights.get(index, 0.0)
        if fit is None or weight <= 0.0:
            out.append(event)
            continue
        moved, sub = apply_fits(
            [event],
            {0: fit},
            model,
            cfg,
            weight=weight,
            min_r2=0.0,
            sigma=sigma,
            isolation="partial_aware",
        )
        out.append(moved[0])
        tally["fitted"] += sub["fitted"]
        tally["applied"] += sub["applied"]
    tally["isolated"] = len(ordered) if isolation == "partial_aware" else 0
    return out, tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bank", "sweep"))
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--fit-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    fit_cache = args.fit_cache or (data_root / "models" / "a_partial_fit_cache")
    fit_cache.mkdir(parents=True, exist_ok=True)
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

    def cache_for(track_id: str) -> Path:
        return sealed_cache if track_id[:2] == BURNED_PLAYER else dev_cache

    if args.command == "bank":
        started = time.perf_counter()
        total = 0
        for index, track_id in enumerate(clips, start=1):
            total += bank_clip(
                track_id,
                data_home=data_home,
                events_cache=cache_for(track_id),
                fit_cache=fit_cache,
                isolation=isolation,
                cfg=cfg,
            )
            if index % 50 == 0 or index == len(clips):
                print(
                    f"  [{index}/{len(clips)}] {total} fits banked "
                    f"({(time.perf_counter() - started) / 60.0:.1f} min)",
                    flush=True,
                )
        print(f"banked {total} {isolation} fits for {len(clips)} clips -> {fit_cache}")
        return 0

    # --- sweep ---
    missing = [t for t in clips if not (fit_cache / f"{t}.{isolation}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} clips unbanked; run `bank` first ({missing[0]})")

    print("building leave-one-player-out priors...", flush=True)
    positions, sequences = build_loo_priors(gold, cfg)

    arms: list[dict[str, Any]] = [{"name": "baseline", "kind": "none"}]
    arms.append({"name": "shipped", "kind": "hard", "min_r2": evidence.min_r2})
    for floor in (0.40, 0.30, 0.20, 0.10, 0.0):
        arms.append({"name": f"min_r2={floor:.2f}", "kind": "hard", "min_r2": floor})
    for floor in (0.30, 0.10, 0.0):
        arms.append({"name": f"confidence>={floor:.2f}", "kind": "soft", "floor": floor})

    scores: dict[str, list[float]] = {arm["name"]: [] for arm in arms}
    coverage: dict[str, int] = {arm["name"]: 0 for arm in arms}
    events_seen = 0
    started = time.perf_counter()

    for index, track_id in enumerate(clips, start=1):
        player = track_id[:2]
        events_path = cache_for(track_id) / f"{track_id}.ensemble.json"
        events = [_event_from_json(item) for item in json.loads(events_path.read_text("utf-8"))]
        ordered = sorted(events, key=lambda event: event.onset_s)
        banked = load_banked(fit_cache / f"{track_id}.{isolation}.json")
        clip_gold = gold[player][track_id]
        events_seen += len(ordered)

        for arm in arms:
            if arm["kind"] == "none":
                arm_events = ordered
            elif arm["kind"] == "hard":
                arm_events, tally = apply_fits(
                    ordered,
                    banked,
                    table,
                    cfg,
                    weight=evidence.weight,
                    min_r2=arm["min_r2"],
                    sigma=evidence.sigma,
                    isolation=isolation,
                )
                coverage[arm["name"]] += tally["applied"]
            else:
                kept, weights = confidence_scaled(banked, arm["floor"], evidence.weight)
                arm_events, tally = apply_weighted(
                    ordered, kept, weights, table, cfg, sigma=evidence.sigma, isolation=isolation
                )
                coverage[arm["name"]] += tally["applied"]
            metrics, _ = score_leak_free(
                arm_events,
                clip_gold,
                position=positions[player],
                sequence=sequences[player],
                cfg=cfg,
                session=session,
            )
            scores[arm["name"]].append(metrics["tab_f1"])

        if index % 50 == 0 or index == len(clips):
            print(
                f"  [{index}/{len(clips)}] ({(time.perf_counter() - started) / 60.0:.1f} min)",
                flush=True,
            )

    base = np.asarray(scores["baseline"], dtype=np.float64)
    results: dict[str, Any] = {
        "clips": len(clips),
        "events": events_seen,
        "isolation": isolation,
        "frozen_baseline": FROZEN_BASELINE,
        "arms": {},
    }

    print(f"\n{'arm':<20}{'Tab F1':>9}{'delta':>9}{'lo95':>9}{'hi95':>9}{'coverage':>11}")
    for arm in arms:
        name = arm["name"]
        values = np.asarray(scores[name], dtype=np.float64)
        deltas = values - base
        ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        cov = coverage[name] / events_seen if events_seen else 0.0
        results["arms"][name] = {
            "tab_f1": float(values.mean()),
            "delta_vs_baseline": float(deltas.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
            "coverage": cov,
            "applied": coverage[name],
        }
        print(
            f"{name:<20}{values.mean():>9.4f}{deltas.mean():>+9.4f}"
            f"{ci.lower:>+9.4f}{ci.upper:>+9.4f}{cov:>10.1%}"
        )

    # Pin against Phase 0 so a harness change cannot pass unnoticed.
    drift_base = results["arms"]["baseline"]["tab_f1"] - FROZEN_BASELINE["dev_baseline"]
    drift_ship = results["arms"]["shipped"]["tab_f1"] - FROZEN_BASELINE["dev_shipped"]
    results["frozen_drift"] = {"baseline": drift_base, "shipped": drift_ship}
    print(f"\nfrozen-baseline drift: baseline {drift_base:+.4f}  shipped {drift_ship:+.4f}")
    if not args.limit and (
        abs(drift_base) > FROZEN_TOLERANCE or abs(drift_ship) > FROZEN_TOLERANCE
    ):
        raise SystemExit(
            "drift from the Phase 0 frozen baseline exceeds tolerance; the banked "
            "replay is not reproducing the live path and no arm above is trustworthy"
        )

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
