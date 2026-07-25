"""Phase 0 — sealed-set rotation, frozen baseline, and current-default decomposition.

This script does three things in one pass, deliberately: it rotates the sealed
confirmation set, re-baselines against it, and decomposes the error profile of
the configuration that actually ships. They share a substrate, so splitting them
would triple the compute and let the three answers drift apart.

Pre-declared design — written before any number below was produced
=================================================================

**The rotation.** Player 05 has been opened twice for the ``acoustic-physics-v1``
artifact (Q6's gate, then the 2026-07-24 batched confirmation). The reports say
so plainly: "its value as a sealed set is correspondingly reduced". Four
incoming workstreams all want confirmation, so the set is rotated now rather
than after the fact.

**The new sealed player is 04**, chosen by a mechanical rule — the next player
index below the burned one — fixed *before* looking at any per-player score.
This matters more than which player is picked: any choice made after seeing
scores would bias every confirmation the rotation is supposed to protect.

Player 05 returns to development. Its sealed value is already spent, and its
clips are still perfectly good training/measurement material. So:

    dev    = {00, 01, 02, 03, 05}   300 clips
    sealed = {04}                    60 clips

Dev keeps its size; only the roles swap.

**Leak-free priors everywhere.** Under the old split the shipped
``guitarset-v1`` excluded player 05, so measuring on 05 was naturally leak-free
while dev measurement needed the house leave-one-player-out protocol. That
asymmetry disappears here: *every* clip is scored under priors rebuilt without
its own player, from a six-player pool, at the registered artifacts' own
hyper-parameters. Each fold therefore trains on five players — the same size as
the shipped artifact — so dev and sealed numbers are directly comparable, and
neither is inflated by memorized fingerings.

The shipped artifacts are **not** modified. This is a measurement substrate, not
a product change. What it estimates is what the product does for a player it has
never seen, which is what the old player-05 setup estimated too.

**The arms.**

``baseline``  — no string-evidence channel. The reference every future track
                measures against.
``shipped``   — the current default: ``acoustic-physics-v1`` at its registered
                weight / min_r2 / sigma, partial-aware isolation.

Both arms score identical detections from the banked ensemble cache, so every
delta is attributable to the fusion stage alone.

**The reproduction check.** Player 05's leave-one-out fold trains on exactly the
five players the registered artifacts saw, so this run must reproduce its
published numbers (0.6340 baseline, 0.7346 shipped). It is asserted, not
eyeballed. The first pass of this script failed it silently: it replayed
evidence through n5's banked ``apply_banked``, whose fits come from
``measure_events`` — which takes no isolation argument and is always strict — so
it measured the ``raw-strict`` arm (0.7119) while labelling itself shipped. The
self-check it had at the time compared the strict replay to the strict live
path and passed. Pinning a published external number catches what an internal
consistency check cannot.

**What this run is not.** It is not a gate and nothing here is tuned. It
produces the frozen baseline and the error profile that Phase 1 prioritizes
against.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import _event_from_json
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, decompose_errors
from tabvision.eval.guitarset_audio import (
    load_mono_audio,
    parse_guitarset_jams,
    score_audio_only,
)
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.playability import set_transition_prior
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
)
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.fusion.transition_prior import TransitionPrior, learn_transition_prior
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT
from tabvision.types import GuitarConfig, SessionConfig, TabEvent

ALL_PLAYERS = ("00", "01", "02", "03", "04", "05")

SEALED_PLAYER = "04"
"""Pre-declared by mechanical rule, before any score was inspected."""

DEV_PLAYERS = tuple(player for player in ALL_PLAYERS if player != SEALED_PLAYER)

BURNED_PLAYER = "05"
"""Previously sealed; opened twice, so it rejoins development."""

# Registered-artifact hyper-parameters. Mirrored here so a fold is built the
# same way the shipped artifact was; asserted against the manifests at startup.
POSITION_ALPHA = 1.0
POSITION_POWER = 2.0
SEQUENCE_SCHEME = "delta_fret"
SEQUENCE_ALPHA = 0.5
SEQUENCE_BACKOFF_KAPPA = 8.0
SEQUENCE_SINGLETON_ONLY = True

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

# Player 05's published numbers, from player05_batched_confirm_2026-07-24.
# Player 05 is now a development player, and its leave-one-out fold trains on
# exactly the five players the registered artifacts saw — so this run must
# reproduce these, and a mismatch means the harness is measuring something
# other than what ships.
#
# This check exists because the first pass of this script did exactly that. It
# replayed evidence through n5's `apply_banked`, whose cached fits are built by
# `measure_events` — which takes no isolation argument and is therefore always
# strict. The run reproduced the *raw-strict* arm (0.7119) while reporting
# itself as shipped (`partial_aware`, 0.7346). Nothing failed: the old
# self-check compared the strict replay against the strict live path, so both
# sides carried the same error. Pinning the published number instead of an
# internal consistency property is what catches this class of bug.
REPRODUCTION = {"baseline": 0.6340, "shipped": 0.7346}
REPRODUCTION_TOLERANCE = 0.0015

ARMS = ("baseline", "shipped")
BUCKETS = (
    "correct",
    "wrong_position_same_pitch",
    "pitch_off",
    "timing_only",
    "missed_onset",
    "extra_detection",
)


def gold_by_player(data_home: Path, cfg: GuitarConfig) -> dict[str, dict[str, list[TabEvent]]]:
    """``player -> track_id -> gold events``, over the full six-player pool."""
    out: dict[str, dict[str, list[TabEvent]]] = {player: {} for player in ALL_PLAYERS}
    for path in sorted((data_home / "annotation").glob("*.jams")):
        player = path.stem[:2]
        if player in out:
            out[player][path.stem] = list(parse_guitarset_jams(path, cfg))
    return out


def build_loo_priors(
    gold: dict[str, dict[str, list[TabEvent]]], cfg: GuitarConfig
) -> tuple[dict[str, PitchPositionPrior], dict[str, TransitionPrior]]:
    """Leave-one-player-out position and sequence priors over all six players.

    Each fold trains on the other five — the same number of players the shipped
    artifacts saw — so no clip is ever scored under a prior that memorized its
    own player, and dev and sealed folds are equally powered.
    """
    positions: dict[str, PitchPositionPrior] = {}
    sequences: dict[str, TransitionPrior] = {}
    for held_out in ALL_PLAYERS:
        events: list[TabEvent] = []
        tracks: list[list[TabEvent]] = []
        for player, by_track in gold.items():
            if player == held_out:
                continue
            for track_events in by_track.values():
                events.extend(track_events)
                tracks.append(track_events)
        positions[held_out] = learn_pitch_position_prior(
            events, cfg=cfg, alpha=POSITION_ALPHA, power=POSITION_POWER
        )
        sequences[held_out] = learn_transition_prior(
            tracks,
            scheme=SEQUENCE_SCHEME,
            alpha=SEQUENCE_ALPHA,
            backoff_kappa=SEQUENCE_BACKOFF_KAPPA,
            singleton_only=SEQUENCE_SINGLETON_ONLY,
        )
    return positions, sequences


def score_leak_free(
    events: list[Any],
    gold: list[TabEvent],
    *,
    position: PitchPositionPrior,
    sequence: TransitionPrior,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> tuple[dict[str, float], ErrorDecomposition]:
    """Shipped clean-acoustic decode with this clip's leave-one-out priors.

    Mirrors ``n2_muscriptor_merge._score`` exactly, except that the sequence
    prior is the fold's rather than the registered artifact's — the registered
    one saw player 04 and would leak on the new sealed set.
    """
    prepared = apply_pitch_position_prior(list(events), position)
    set_transition_prior(sequence, weight=SEQUENCE_PRIOR_WEIGHT)
    try:
        scored = score_audio_only(prepared, gold, cfg=cfg, session=session)
    finally:
        set_transition_prior(None)
    return (
        {
            "onset_f1": scored.onset.f1,
            "pitch_f1": scored.pitch.f1,
            "tab_f1": scored.tab.f1,
            "tab_precision": scored.tab.precision,
            "tab_recall": scored.tab.recall,
        },
        decompose_errors(scored.decoded, gold),
    )


def sum_decompositions(items: list[ErrorDecomposition]) -> dict[str, int]:
    totals = {name: 0 for name in BUCKETS}
    for item in items:
        for field in fields(item):
            if field.name in totals:
                totals[field.name] += int(getattr(item, field.name))
    return totals


def loss_shares(totals: dict[str, int]) -> dict[str, float]:
    loss = sum(value for name, value in totals.items() if name != "correct")
    if loss == 0:
        return {name: 0.0 for name in BUCKETS if name != "correct"}
    return {name: totals[name] / loss for name in BUCKETS if name != "correct"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--dev-cache", type=Path, default=None)
    parser.add_argument("--sealed-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap clips per split")
    parser.add_argument(
        "--skip-reproduction-check",
        action="store_true",
        help="debug only; the check is what makes these numbers trustworthy",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    dev_cache = args.dev_cache or (data_root / "models" / "q6_full_dev_cache")
    sealed_cache = args.sealed_cache or (data_root / "models" / "q6_player05_cache")

    cfg = GuitarConfig()
    session = SessionConfig()
    evidence = load_string_evidence()
    table = reference_stiffness_model()

    print(
        "Phase 0 — rotation, baseline, decomposition\n"
        f"  sealed  : player {SEALED_PLAYER} (pre-declared; was {BURNED_PLAYER})\n"
        f"  dev     : players {', '.join(DEV_PLAYERS)}\n"
        f"  priors  : leave-one-player-out over {len(ALL_PLAYERS)} players, "
        f"5 players per fold\n"
        f"  physics : weight={evidence.weight} min_r2={evidence.min_r2} "
        f"sigma={evidence.sigma} isolation={evidence.isolation}",
        flush=True,
    )

    print("building leave-one-player-out priors...", flush=True)
    gold = gold_by_player(data_home, cfg)
    positions, sequences = build_loo_priors(gold, cfg)

    # Events live in two caches purely for historical reasons (dev vs the old
    # sealed player); the split they belong to is decided by the rotation above,
    # not by which file they happen to sit in.
    def cache_for(track_id: str) -> Path:
        root = sealed_cache if track_id[:2] == BURNED_PLAYER else dev_cache
        return root / f"{track_id}.ensemble.json"

    splits: dict[str, list[str]] = {
        "dev": sorted(t for p in DEV_PLAYERS for t in gold[p]),
        "sealed": sorted(gold[SEALED_PLAYER]),
    }
    if args.limit:
        splits = {name: clips[: args.limit] for name, clips in splits.items()}

    missing = [t for clips in splits.values() for t in clips if not cache_for(t).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} clips are not banked; this run refuses to re-transcribe "
            f"(first missing: {missing[0]})"
        )
    print(f"  banked  : {sum(len(c) for c in splits.values())} clips, all present", flush=True)

    results: dict[str, Any] = {
        "sealed_player": SEALED_PLAYER,
        "dev_players": list(DEV_PLAYERS),
        "previously_sealed": BURNED_PLAYER,
        "prior_protocol": (
            "leave-one-player-out over all six players; each fold trains on five, "
            "at the registered artifacts' hyper-parameters; shipped artifacts unmodified"
        ),
        "frozen_decode": {
            "weight": evidence.weight,
            "min_r2": evidence.min_r2,
            "sigma": evidence.sigma,
            "isolation": evidence.isolation,
            "sequence_prior_weight": SEQUENCE_PRIOR_WEIGHT,
            "source": "registered acoustic-physics-v1 artifact",
        },
        "splits": {},
    }

    started = time.perf_counter()

    for split_name, clips in splits.items():
        scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
        aux: dict[str, list[dict[str, float]]] = {arm: [] for arm in ARMS}
        decomps: dict[str, dict[str, list[ErrorDecomposition]]] = {
            arm: {"solo": [], "comp": []} for arm in ARMS
        }
        rows: list[dict[str, Any]] = []
        applied_total = 0
        event_total = 0

        for index, track_id in enumerate(clips, start=1):
            player = track_id[:2]
            mode = "solo" if track_id.endswith("_solo") else "comp"
            events = [
                _event_from_json(item)
                for item in json.loads(cache_for(track_id).read_text("utf-8"))
            ]
            wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
            clip_gold = gold[player][track_id]

            ordered = sorted(events, key=lambda event: event.onset_s)
            row: dict[str, Any] = {"track_id": track_id, "player": player, "mode": mode}

            for arm in ARMS:
                if arm == "baseline":
                    arm_events = ordered
                else:
                    # The live path, not the banked replay. `measure_events` has
                    # no isolation parameter, so the n5 replay is strict-only and
                    # cannot express the shipped `partial_aware` setting — an
                    # earlier pass of this script used it and silently measured
                    # the strict arm (it reproduced Q6's raw-strict block, 0.7119
                    # on player 05, instead of the shipped 0.7346).
                    arm_events, tally = attach_inharmonicity_evidence(
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
                    applied_total += tally["applied"]
                    event_total += tally["events"]
                metrics, decomposition = score_leak_free(
                    arm_events,
                    clip_gold,
                    position=positions[player],
                    sequence=sequences[player],
                    cfg=cfg,
                    session=session,
                )
                scores[arm].append(metrics["tab_f1"])
                aux[arm].append(metrics)
                decomps[arm][mode].append(decomposition)
                row[arm] = metrics["tab_f1"]

            rows.append(row)
            if index % 25 == 0 or index == len(clips):
                elapsed = (time.perf_counter() - started) / 60.0
                lead = float(np.mean([r["shipped"] - r["baseline"] for r in rows]))
                print(
                    f"  [{split_name} {index}/{len(clips)}] shipped-baseline "
                    f"{lead:+.4f} ({elapsed:.1f} min)",
                    flush=True,
                )

        solo_mask = np.asarray([r["mode"] == "solo" for r in rows])
        split_out: dict[str, Any] = {
            "clips": len(rows),
            "physics_coverage": {"applied": applied_total, "events": event_total},
            "tab_f1": {},
            "tiers": {},
            "decomposition": {},
            "per_clip": rows,
        }

        for arm in ARMS:
            values = np.asarray(scores[arm], dtype=np.float64)
            split_out["tab_f1"][arm] = float(values.mean())
            split_out["tiers"][arm] = {
                "single_line": float(values[solo_mask].mean()),
                "strummed": float(values[~solo_mask].mean()),
            }
            for metric in ("onset_f1", "pitch_f1"):
                split_out.setdefault(metric, {})[arm] = float(
                    np.mean([m[metric] for m in aux[arm]])
                )
            totals_all = sum_decompositions(decomps[arm]["solo"] + decomps[arm]["comp"])
            split_out["decomposition"][arm] = {
                "aggregate": {"counts": totals_all, "shares": loss_shares(totals_all)},
                "single_line": {
                    "counts": (s := sum_decompositions(decomps[arm]["solo"])),
                    "shares": loss_shares(s),
                },
                "strummed": {
                    "counts": (c := sum_decompositions(decomps[arm]["comp"])),
                    "shares": loss_shares(c),
                },
            }

        deltas = np.asarray(scores["shipped"], dtype=np.float64) - np.asarray(
            scores["baseline"], dtype=np.float64
        )
        ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        split_out["shipped_vs_baseline"] = {
            "delta": float(deltas.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
        }
        results["splits"][split_name] = split_out

        print(f"\n=== {split_name} ({len(rows)} clips) ===")
        print(f"{'arm':<12}{'single-line':>13}{'strummed':>11}{'aggregate':>11}")
        for arm in ARMS:
            tier = split_out["tiers"][arm]
            print(
                f"{arm:<12}{tier['single_line']:>13.4f}{tier['strummed']:>11.4f}"
                f"{split_out['tab_f1'][arm]:>11.4f}"
            )
        d = split_out["shipped_vs_baseline"]
        print(f"shipped - baseline: {d['delta']:+.4f} [{d['lo95']:+.4f}, {d['hi95']:+.4f}]")
        print(f"\n{'bucket':<28}{'baseline':>10}{'share':>8}{'shipped':>10}{'share':>8}")
        base_d = split_out["decomposition"]["baseline"]["aggregate"]
        ship_d = split_out["decomposition"]["shipped"]["aggregate"]
        for name in BUCKETS:
            bs = "" if name == "correct" else f"{base_d['shares'][name]:>7.1%}"
            ss = "" if name == "correct" else f"{ship_d['shares'][name]:>7.1%}"
            print(
                f"{name:<28}{base_d['counts'][name]:>10}{bs:>8}{ship_d['counts'][name]:>10}{ss:>8}"
            )

    # Reproduction check, last so the numbers are on screen either way.
    dev_rows = results["splits"].get("dev", {}).get("per_clip", [])
    burned = [row for row in dev_rows if row["player"] == BURNED_PLAYER]
    if not args.skip_reproduction_check and len(burned) == 60:
        observed = {arm: float(np.mean([row[arm] for row in burned])) for arm in ARMS}
        drift = {arm: observed[arm] - REPRODUCTION[arm] for arm in ARMS}
        results["reproduction_check"] = {
            "player": BURNED_PLAYER,
            "published": REPRODUCTION,
            "observed": observed,
            "drift": drift,
            "tolerance": REPRODUCTION_TOLERANCE,
        }
        print(f"\nreproduction check (player {BURNED_PLAYER} vs published):")
        for arm in ARMS:
            print(
                f"  {arm:<10}published {REPRODUCTION[arm]:.4f}  "
                f"observed {observed[arm]:.4f}  drift {drift[arm]:+.4f}"
            )
        failed = [arm for arm in ARMS if abs(drift[arm]) > REPRODUCTION_TOLERANCE]
        if failed:
            raise SystemExit(
                f"reproduction check FAILED on {', '.join(failed)}: this harness is not "
                f"measuring the shipped configuration, so nothing above is trustworthy"
            )
        print("  PASS — this harness measures the configuration that ships")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
