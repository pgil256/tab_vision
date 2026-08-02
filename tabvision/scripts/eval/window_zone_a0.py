"""Phase A0 — can the physics channel cover, and aim, at window scope?

The inharmonicity channel reads position accurately but only on notes that
ring alone (~34% of solo notes), and it never propagates that reading to the
neighbours sharing the same hand position. The +0.2756 fret-zone headroom is
defined per 1 s *window*, so the question that decides the whole program is
whether sparse per-note coverage becomes dense per-*window* coverage.

Measurement only — no mechanism, no Tab F1, no promotion. Every constant is
frozen in ``docs/plans/2026-07-29-per-window-fret-zone-evidence-design.md``
§7a, committed before this script was first run. Run from ``tabvision/``::

    python -m scripts.eval.window_zone_a0 \
        --json ../docs/EVAL_REPORTS/window_zone_a0_2026-07-29.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.string_assignment_oracles import FRET_ZONES, fixed_window_groups
from tabvision.eval.guitarset_audio import DEFAULT_DATA_HOME, load_mono_audio, parse_guitarset_jams
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import (
    MIN_CLEAN_PARTIALS,
    InharmonicityFit,
    StiffnessObservation,
    calibrate_from_session,
    measure_events,
)
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.string_physics import StringStiffnessModel, load_string_evidence
from tabvision.types import AudioEvent, GuitarConfig, TabEvent

# ---------------------------------------------------------------- frozen §7a

DEV_PLAYERS = ("00", "01", "02", "03", "05")
"""Sealed player 04 is not read in A0 (rotation of 2026-07-25)."""

WINDOWS_S = (1.0, 2.0, 4.0)
PRIMARY_WINDOW_S = 1.0

ARMS = ("reference", "self_seeded", "gold_calibrated")
"""Shipped table raw; label-free session refit (gated on); gold refit (ceiling)."""

GATE_COVERAGE = 0.60
GATE_AGREEMENT = 0.75


def _zone_set(frets: Sequence[int]) -> frozenset[int]:
    """Indices of the zones containing *every* fret given."""
    if not frets:
        return frozenset()
    return frozenset(
        index
        for index, (low, high) in enumerate(FRET_ZONES)
        if all(low <= fret <= high for fret in frets)
    )


def _implied_position(
    fit: InharmonicityFit,
    pitch_midi: int,
    model: StringStiffnessModel,
    cfg: GuitarConfig,
) -> tuple[int, int] | None:
    """Candidate minimising |measured log B − predicted log B| (the channel's rule)."""
    best: tuple[float, tuple[int, int]] | None = None
    for candidate in candidate_positions(pitch_midi, cfg):
        predicted = model.predicted_log_b(candidate.string_idx, candidate.fret)
        if predicted is None:
            continue
        distance = abs(fit.log_b - predicted)
        if best is None or distance < best[0]:
            best = (distance, (candidate.string_idx, candidate.fret))
    return None if best is None else best[1]


def _as_audio_events(gold: Sequence[TabEvent]) -> list[AudioEvent]:
    """Gold-timed notes as AudioEvents, onset-ordered (measure_events keys by index)."""
    ordered = sorted(gold, key=lambda event: (float(event.onset_s), int(event.pitch_midi)))
    return [
        AudioEvent(
            onset_s=float(event.onset_s),
            offset_s=float(event.onset_s) + float(event.duration_s),
            pitch_midi=int(event.pitch_midi),
            velocity=1.0,
            confidence=1.0,
        )
        for event in ordered
    ]


def _prior_positions(
    events: Sequence[AudioEvent],
    prior: np.ndarray,
    cfg: GuitarConfig,
) -> dict[int, tuple[int, int]]:
    """Physics-free provisional positions: the position prior's top-1 per note."""
    prepared = apply_pitch_position_prior(list(events), prior, cfg)
    out: dict[int, tuple[int, int]] = {}
    for index, event in enumerate(prepared):
        matrix = event.fret_prior
        candidates = candidate_positions(int(event.pitch_midi), cfg)
        if matrix is None or not candidates:
            continue
        best = max(candidates, key=lambda c: float(np.asarray(matrix)[c.string_idx, c.fret]))
        out[index] = (int(best.string_idx), int(best.fret))
    return out


def analyse_track(
    gold: Sequence[TabEvent],
    wav: np.ndarray,
    sr: int,
    model: StringStiffnessModel,
    cfg: GuitarConfig,
    prior: np.ndarray,
    *,
    min_r2: float,
    isolation: str,
) -> dict[str, Any]:
    """Coverage and aim for one track, at each window size."""
    ordered = sorted(gold, key=lambda event: (float(event.onset_s), int(event.pitch_midi)))
    events = _as_audio_events(ordered)
    if not events:
        return {}

    fits = measure_events(
        events,
        wav,
        sr,
        cfg,
        isolation=isolation,
        min_clean_partials=MIN_CLEAN_PARTIALS,
    )
    readable = {index: fit for index, fit in fits.items() if float(fit.r2) >= min_r2}
    prior_positions = {
        index: position
        for index, position in _prior_positions(events, prior, cfg).items()
        if index in readable
    }

    def implied_for(active: StringStiffnessModel) -> tuple[dict[int, tuple[int, int]], int]:
        table: dict[int, tuple[int, int]] = {}
        correct = 0
        for index, fit in readable.items():
            position = _implied_position(fit, int(ordered[index].pitch_midi), active, cfg)
            if position is None:
                continue
            table[index] = position
            if position == (int(ordered[index].string_idx), int(ordered[index].fret)):
                correct += 1
        return table, correct

    # Arm 1: the shipped reference table applied raw (the floor).
    reference_implied, reference_correct = implied_for(model)

    def refit(provisional: dict[int, tuple[int, int]]) -> StringStiffnessModel | None:
        observations = [
            StiffnessObservation(
                string_idx=position[0],
                fret=position[1],
                log_b=readable[index].log_b,
                r2=float(readable[index].r2),
            )
            for index, position in provisional.items()
        ]
        return calibrate_from_session(observations, seed=model, min_r2=min_r2)

    # Arm 2: label-free session refit seeded from a physics-free first pass.
    # q6's validated `self-seeded` arm takes provisional labels from a first
    # *decode* (~0.65 top-1), not from the physics argmax — seeding from the
    # mis-centred table's own 0.20 output would just recycle its error.
    self_model = refit(prior_positions) or model
    self_implied, self_correct = implied_for(self_model)

    # Arm 3: the same refit with gold provisional positions (the ceiling).
    gold_model = (
        refit(
            {
                index: (int(ordered[index].string_idx), int(ordered[index].fret))
                for index in readable
            }
        )
        or model
    )
    gold_implied, gold_correct = implied_for(gold_model)

    arms = {
        "reference": (reference_implied, reference_correct),
        "self_seeded": (self_implied, self_correct),
        "gold_calibrated": (gold_implied, gold_correct),
    }

    out: dict[str, Any] = {
        "notes": len(ordered),
        "fitted": len(fits),
        "readable": len(readable),
        "ambiguous_notes": sum(
            1 for event in ordered if len(candidate_positions(int(event.pitch_midi), cfg)) > 1
        ),
    }
    for arm, (table, correct) in arms.items():
        out[f"implied_{arm}"] = len(table)
        out[f"note_position_correct_{arm}"] = correct

    all_indices = set(range(len(ordered)))
    for window_s in WINDOWS_S:
        groups = fixed_window_groups(ordered, all_indices, window_s)
        # Coverage and the hand-moved count are arm-independent: which notes
        # are readable is a property of the spectrum, not of the table.
        covered = 0
        hand_moved = 0
        reachable_unreadable = 0
        reachable_unreadable_ambiguous = 0
        agreement: dict[str, dict[str, int]] = {arm: {"scorable": 0, "agreed": 0} for arm in arms}
        for group in groups:
            gold_frets = [int(ordered[i].fret) for i in group if int(ordered[i].fret) > 0]
            group_readable = [i for i in group if i in reference_implied]
            if group_readable:
                covered += 1
                for i in group:
                    if i in reference_implied:
                        continue
                    reachable_unreadable += 1
                    if len(candidate_positions(int(ordered[i].pitch_midi), cfg)) > 1:
                        reachable_unreadable_ambiguous += 1
            if not gold_frets:
                continue
            gold_zones = _zone_set(gold_frets)
            if not gold_zones:
                hand_moved += 1
                continue
            for arm, (table, _correct) in arms.items():
                implied_frets = [table[i][1] for i in group if i in table and table[i][1] > 0]
                if not implied_frets:
                    continue
                agreement[arm]["scorable"] += 1
                if _zone_set(implied_frets) & gold_zones:
                    agreement[arm]["agreed"] += 1
        cell: dict[str, Any] = {
            "windows": len(groups),
            "covered": covered,
            "hand_moved": hand_moved,
            "reachable_unreadable": reachable_unreadable,
            "reachable_unreadable_ambiguous": reachable_unreadable_ambiguous,
        }
        for arm, counts in agreement.items():
            cell[f"scorable_{arm}"] = counts["scorable"]
            cell[f"agreed_{arm}"] = counts["agreed"]
        out[f"w{window_s}"] = cell
    return out


def _tier(track_id: str) -> str:
    return "solo" if track_id.endswith("_solo") else "comp"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=Path(DEFAULT_DATA_HOME))
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N tracks")
    args = parser.parse_args(argv)

    cfg = GuitarConfig()
    evidence = load_string_evidence("acoustic-physics-v1")
    # Physics-free provisional labels for the self-seeded arm. In-sample for
    # players 00-03 (the shipped artifact excluded only 05), which makes arm 2
    # mildly optimistic — A0 is already declared an upper bound.
    prior = load_pitch_position_prior("guitarset-v1", cfg=cfg)
    annotations = sorted((args.data_home / "annotation").glob("*.jams"))
    selected = [p for p in annotations if p.stem.split("_")[0] in DEV_PLAYERS]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit(f"no dev annotations under {args.data_home}")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, jams_path in enumerate(selected, 1):
        track_id = jams_path.stem
        wav_path = args.data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        if not wav_path.is_file():
            continue
        gold = parse_guitarset_jams(jams_path, cfg)
        if not gold:
            continue
        wav, sr = load_mono_audio(wav_path)
        stats = analyse_track(
            gold,
            np.asarray(wav),
            int(sr),
            evidence.model,
            cfg,
            prior,
            min_r2=evidence.min_r2,
            isolation=evidence.isolation,
        )
        if not stats:
            continue
        stats["track_id"] = track_id
        stats["player"] = track_id.split("_")[0]
        stats["tier"] = _tier(track_id)
        rows.append(stats)
        primary = stats[f"w{PRIMARY_WINDOW_S}"]
        print(
            f"[{position:3d}/{len(selected)}] {track_id:28s} {stats['tier']:4s} "
            f"readable {stats['readable']:4d}/{stats['notes']:4d}  "
            f"cov {primary['covered']}/{primary['windows']}  "
            f"agree(self) {primary['agreed_self_seeded']}/{primary['scorable_self_seeded']}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no tracks scored")

    def totals(tier: str | None, window_s: float) -> dict[str, Any]:
        cells = [r for r in rows if tier is None or r["tier"] == tier]
        windows = sum(r[f"w{window_s}"]["windows"] for r in cells)
        covered = sum(r[f"w{window_s}"]["covered"] for r in cells)
        out: dict[str, Any] = {
            "tracks": len(cells),
            "windows": windows,
            "covered": covered,
            "coverage": covered / windows if windows else 0.0,
            "hand_moved": sum(r[f"w{window_s}"]["hand_moved"] for r in cells),
            "reachable_unreadable": sum(r[f"w{window_s}"]["reachable_unreadable"] for r in cells),
            "reachable_unreadable_ambiguous": sum(
                r[f"w{window_s}"]["reachable_unreadable_ambiguous"] for r in cells
            ),
        }
        for arm in ARMS:
            scorable = sum(r[f"w{window_s}"][f"scorable_{arm}"] for r in cells)
            agreed = sum(r[f"w{window_s}"][f"agreed_{arm}"] for r in cells)
            out[f"scorable_{arm}"] = scorable
            out[f"agreed_{arm}"] = agreed
            out[f"agreement_{arm}"] = agreed / scorable if scorable else 0.0
        return out

    notes = sum(r["notes"] for r in rows)
    readable = sum(r["readable"] for r in rows)
    solo_primary = totals("solo", PRIMARY_WINDOW_S)
    gate_arm = "self_seeded"
    passed = (
        solo_primary["coverage"] >= GATE_COVERAGE
        and solo_primary[f"agreement_{gate_arm}"] >= GATE_AGREEMENT
    )

    summary: dict[str, Any] = {
        "phase": "A0",
        "corpus": "GuitarSet dev (players 00,01,02,03,05); sealed 04 not read",
        "tracks": len(rows),
        "notes": notes,
        "readable_notes": readable,
        "note_readable_rate": readable / notes if notes else 0.0,
        "note_position_accuracy_on_readable": {
            arm: (
                sum(r[f"note_position_correct_{arm}"] for r in rows)
                / sum(r[f"implied_{arm}"] for r in rows)
                if sum(r[f"implied_{arm}"] for r in rows)
                else 0.0
            )
            for arm in ARMS
        },
        "physics": {
            "artifact": "acoustic-physics-v1",
            "min_r2": evidence.min_r2,
            "isolation": evidence.isolation,
            "sigma": evidence.sigma,
            "fret_exponent": evidence.model.fret_exponent,
            "min_clean_partials": MIN_CLEAN_PARTIALS,
        },
        "gate": {
            "coverage_bar": GATE_COVERAGE,
            "agreement_bar": GATE_AGREEMENT,
            "read_on": f"solo @ {PRIMARY_WINDOW_S} s, arm {gate_arm}",
            "solo_coverage": solo_primary["coverage"],
            "solo_agreement": solo_primary[f"agreement_{gate_arm}"],
            "pass": passed,
        },
        "wall_seconds": time.perf_counter() - started,
        "per_clip": rows,
    }
    for window_s in WINDOWS_S:
        summary[f"aggregate_w{window_s}"] = {
            "all": totals(None, window_s),
            "solo": totals("solo", window_s),
            "comp": totals("comp", window_s),
        }

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== A0 over {len(rows)} dev tracks, {notes} notes ===")
    print(f"per-note readable {readable}/{notes} = {readable / notes:.4f}")
    for arm in ARMS:
        anchor = "  <- sanity anchor, expect ~0.92" if arm == "gold_calibrated" else ""
        print(
            f"  implied-position accuracy on readable, {arm:16s} "
            f"{summary['note_position_accuracy_on_readable'][arm]:.4f}{anchor}"
        )
    for window_s in WINDOWS_S:
        print(f"\n-- window {window_s:.0f} s --")
        for tier in ("solo", "comp", "all"):
            cell = summary[f"aggregate_w{window_s}"][tier]
            agreements = "  ".join(
                f"{arm}={cell[f'agreed_{arm}']}/{cell[f'scorable_{arm}']}"
                f"={cell[f'agreement_{arm}']:.4f}"
                for arm in ARMS
            )
            print(
                f"  {tier:4s} coverage {cell['covered']:6d}/{cell['windows']:6d} "
                f"= {cell['coverage']:.4f}   hand-moved {cell['hand_moved']:5d}   "
                f"reachable-unreadable {cell['reachable_unreadable']:6d} "
                f"({cell['reachable_unreadable_ambiguous']} ambiguous)"
            )
            print(f"       agreement  {agreements}")
    print(
        f"\nGate A0 (solo @ {PRIMARY_WINDOW_S:.0f} s, arm {gate_arm}: "
        f"coverage >= {GATE_COVERAGE}, agreement >= {GATE_AGREEMENT}): "
        f"{'PASS' if passed else 'FAIL'} "
        f"(coverage {solo_primary['coverage']:.4f}, "
        f"agreement {solo_primary[f'agreement_{gate_arm}']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
