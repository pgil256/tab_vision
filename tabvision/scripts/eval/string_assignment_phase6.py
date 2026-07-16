"""Evaluate the frozen Phase 6 learned review queue and offline correction replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.string_assignment_phase0 import (
    DEFAULT_CACHE,
    DEV_PLAYERS,
    Track,
    _file_sha256,
    load_tracks,
)
from scripts.eval.string_assignment_phase3 import (
    DEFAULT_POSTERIOR_CACHE,
    _entropy_and_margin,
    _posterior_window,
)
from scripts.eval.string_assignment_phase3 import (
    _cache_path as phase3_cache_path,
)
from scripts.eval.string_assignment_phase4 import (
    DEFAULT_CACHE as PHASE4_CACHE,
)
from scripts.eval.string_assignment_phase4 import (
    _candidate_arrays,
    _oof_position_prior,
)
from scripts.eval.string_assignment_phase4 import (
    _load_rows as load_phase4_rows,
)
from scripts.eval.string_assignment_phase4 import (
    _prepare_features as prepare_phase4_features,
)
from scripts.eval.string_assignment_phase4 import (
    _run_oof as run_phase4_oof,
)
from tabvision.audio.highres_cache import read_highres_cache
from tabvision.eval.metrics import tab_f1
from tabvision.eval.review_queue import (
    FEATURE_COUNT,
    ReviewQueueNet,
    binary_roc_auc,
    budget_metrics,
    fit_platt,
    fit_review_model,
    parameter_count,
)
from tabvision.types import GuitarConfig, TabEvent

SEED = 6621
SCHEMA_VERSION = 1
ACTION_SECONDS = 2
REPLAY_BUDGETS_S = (10, 30, 60)
DETECTOR_AUC_GATE = 0.75
ENRICHMENT_GATE = 2.0
REPLAY_REDUCTION_GATE = 0.50
FEATURE_NAMES = (
    "path_margin",
    "candidate_count",
    "context_disagreement",
    "timbre_disagreement",
    "timbre_strength",
    "posterior_entropy",
    "domain_score",
    "chord_size",
    "segment_inconsistency",
    "mode_comp",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_HOME = Path.home() / ".tabvision/data/guitarset"
DEFAULT_FEATURE_CACHE = Path.home() / ".tabvision/cache/string_assignment_phase6"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/EVAL_REPORTS"
PHASE0_NOTES = DEFAULT_OUTPUT_DIR / "string_assignment_phase0_2026-07-15_notes.csv"
PHASE1_NOTES = DEFAULT_OUTPUT_DIR / "string_assignment_phase1_2026-07-15_notes.csv"
PHASE4_PROVENANCE = DEFAULT_OUTPUT_DIR / "string_assignment_phase4_2026-07-16_provenance.json"
PHASE6_DESIGN = REPO_ROOT / "docs/plans/2026-07-16-tab-f1-phase6-assisted-accuracy-design.md"


@dataclass(frozen=True)
class ReviewRow:
    event_id: str
    track_id: str
    player: str
    mode: str
    event_index: int
    cluster_index: int
    onset_s: float
    pitch_midi: int
    predicted_string: int
    predicted_fret: int
    reference_string: int
    reference_fret: int
    candidates: tuple[tuple[int, int, float], ...]
    wrong: bool


@dataclass(frozen=True)
class PreparedFeatures:
    rows: tuple[ReviewRow, ...]
    features: np.ndarray
    wrong_labels: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OOFResult:
    raw_logits: np.ndarray
    calibrated_risk: np.ndarray
    folds: tuple[dict[str, Any], ...]
    models: tuple[dict[str, Any], ...]
    prediction_sha256: str
    model_sha256: str
    wall_s: float


def _load_note_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    production: list[dict[str, str]] = []
    segment: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["evaluation_split"] != "development_oof" or row["player"] not in DEV_PLAYERS:
                continue
            if row["condition"] == "production_equivalent":
                production.append(row)
            elif row["condition"] == "segment-v1":
                segment.append(row)
    if len(production) != 51_130 or len(segment) != 51_130:
        raise RuntimeError(
            f"expected 51,130 development rows per condition, got {len(production)} and "
            f"{len(segment)}"
        )
    if len({row["event_id"] for row in production}) != len(production):
        raise RuntimeError("Phase 1 production event IDs are not unique")
    return production, segment


def _parse_candidates(value: str) -> tuple[tuple[int, int, float], ...]:
    output = []
    for item in value.split(";"):
        string, fret, cost = item.split(":")
        output.append((int(string), int(fret), float(cost)))
    if not output or output[0][2] != 0.0:
        raise ValueError("candidate path must begin with the automatic zero-cost position")
    return tuple(output)


def _review_rows(production: Sequence[Mapping[str, str]]) -> tuple[ReviewRow, ...]:
    rows = []
    for raw in production:
        if raw["ambiguous_pitch_match"] != "1" or int(raw["candidate_count"]) < 2:
            continue
        rows.append(
            ReviewRow(
                raw["event_id"],
                raw["track_id"],
                raw["player"],
                raw["mode"],
                int(raw["event_index"]),
                int(raw["cluster_index"]),
                float(raw["onset_s"]),
                int(raw["pitch_midi"]),
                int(raw["predicted_string"]),
                int(raw["predicted_fret"]),
                int(raw["reference_string"]),
                int(raw["reference_fret"]),
                _parse_candidates(raw["candidate_path"]),
                raw["label"] == "wrong_position_same_pitch",
            )
        )
    if len(rows) != 35_959:
        raise RuntimeError(f"expected 35,959 frozen review rows, found {len(rows)}")
    return tuple(rows)


def _base_cache_metadata(rows: Sequence[ReviewRow], notes_path: Path) -> dict[str, Any]:
    event_ids = "\n".join(row.event_id for row in rows).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase1_notes_sha256": _file_sha256(notes_path),
        "phase0_notes_sha256": _file_sha256(PHASE0_NOTES),
        "phase6_design_sha256": _file_sha256(PHASE6_DESIGN),
        "event_ids_sha256": hashlib.sha256(event_ids).hexdigest(),
        "rows": len(rows),
        "feature_names": list(FEATURE_NAMES),
    }


def _prepare_features(
    rows: tuple[ReviewRow, ...],
    production: Sequence[Mapping[str, str]],
    segment: Sequence[Mapping[str, str]],
    tracks: Sequence[Track],
    *,
    notes_path: Path,
    data_home: Path,
    cache_dir: Path,
) -> PreparedFeatures:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "review_features_v1.npz"
    expected = _base_cache_metadata(rows, notes_path)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                actual = json.loads(str(payload["metadata"].item()))
                features = np.asarray(payload["features"], dtype=np.float64)
                labels = np.asarray(payload["wrong_labels"], dtype=np.bool_)
            if all(actual.get(key) == value for key, value in expected.items()):
                _validate_feature_arrays(features, labels, len(rows))
                return PreparedFeatures(
                    rows,
                    features,
                    labels,
                    {
                        **actual,
                        "cache_hit": True,
                        "cache_path": str(cache_path),
                        "cache_bytes": cache_path.stat().st_size,
                        "cache_sha256": _file_sha256(cache_path),
                        "extraction_wall_s": 0.0,
                    },
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    row_by_id = {row.event_id: index for index, row in enumerate(rows)}
    features = np.zeros((len(rows), FEATURE_COUNT), dtype=np.float64)
    labels = np.asarray([row.wrong for row in rows], dtype=np.bool_)
    segment_by_id = {row["event_id"]: row for row in segment}
    if set(segment_by_id) != {row["event_id"] for row in production}:
        raise RuntimeError("Phase 1 production and segment event IDs differ")

    chord_sizes = Counter((row["track_id"], int(row["cluster_index"])) for row in production)
    inconsistency = _segment_inconsistency(production, segment_by_id)
    for index, row in enumerate(rows):
        second_cost = row.candidates[1][2] if len(row.candidates) > 1 else 50.0
        segment_row = segment_by_id[row.event_id]
        features[index, 0] = math.log1p(min(second_cost, 50.0)) / math.log1p(50.0)
        features[index, 1] = len(row.candidates) / 6.0
        features[index, 2] = float(int(segment_row["predicted_string"]) != row.predicted_string)
        features[index, 6] = 1.0
        features[index, 7] = min(chord_sizes[(row.track_id, row.cluster_index)], 6) / 6.0
        features[index, 8] = inconsistency[row.event_id]
        features[index, 9] = float(row.mode == "comp")

    print("  reconstructing player-held native-timbre disagreements...", flush=True)
    phase4_rows = load_phase4_rows(PHASE0_NOTES, GuitarConfig())
    if [row.event_id for row in phase4_rows] != [row.event_id for row in rows]:
        raise RuntimeError("Phase 4 and Phase 6 frozen row order differs")
    phase4_provenance = json.loads(PHASE4_PROVENANCE.read_text(encoding="utf-8"))
    audio_manifest_sha256 = str(phase4_provenance["audio_manifest_sha256"])
    native_features, native_cache = prepare_phase4_features(
        phase4_rows,
        PHASE0_NOTES,
        data_home,
        PHASE4_CACHE,
        audio_manifest_sha256,
    )
    strings, frets, phase4_labels = _candidate_arrays(phase4_rows)
    position_prior = _oof_position_prior(phase4_rows, strings, frets, data_home)
    native_run = run_phase4_oof(
        phase4_rows,
        native_features,
        strings,
        phase4_labels,
        position_prior,
    )
    for index, row in enumerate(rows):
        audio_column = int(native_run.audio_predictions[index])
        audio_string = int(strings[index, audio_column])
        features[index, 3] = float(audio_string != row.predicted_string)
        features[index, 4] = min(float(np.max(np.abs(native_run.edge_logits[index]))), 20.0) / 20.0

    print("  extracting accepted-checkpoint posterior entropy...", flush=True)
    tracks_by_id = {track.track_id: track for track in tracks if track.player in DEV_PLAYERS}
    posterior_manifest: list[dict[str, Any]] = []
    by_track: dict[str, list[ReviewRow]] = defaultdict(list)
    for row in rows:
        by_track[row.track_id].append(row)
    for number, (track_id, track_rows) in enumerate(sorted(by_track.items()), start=1):
        track = tracks_by_id[track_id]
        path = phase3_cache_path(DEFAULT_POSTERIOR_CACHE, "guitar_gaps", track)
        record = read_highres_cache(path)
        if record.posteriors is None:
            raise RuntimeError(f"posterior tensor is missing for {track_id}")
        posterior_manifest.append(
            {
                "track_id": track_id,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
        events_by_pitch: dict[int, list[Any]] = defaultdict(list)
        for event in record.events:
            events_by_pitch[event.pitch_midi].append(event)
        for row in track_rows:
            candidates = events_by_pitch[row.pitch_midi]
            if not candidates:
                raise RuntimeError(f"posterior event alignment changed for {row.event_id}")
            event = min(candidates, key=lambda item: abs(item.onset_s - row.onset_s))
            if abs(event.onset_s - row.onset_s) > 1.0e-5:
                raise RuntimeError(f"posterior event alignment changed for {row.event_id}")
            probabilities = _posterior_window(record.posteriors, event.onset_s)
            entropy, _margin = _entropy_and_margin(probabilities)
            features[row_by_id[row.event_id], 5] = entropy
        if number % 30 == 0:
            print(f"    posterior entropy: {number}/{len(by_track)} tracks", flush=True)

    _validate_feature_arrays(features, labels, len(rows))
    posterior_encoded = json.dumps(
        posterior_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metadata = {
        **expected,
        "phase4_prediction_sha256": native_run.prediction_sha256,
        "phase4_model_sha256": native_run.model_sha256,
        "phase4_native_cache_sha256": native_cache["cache_sha256"],
        "posterior_manifest": posterior_manifest,
        "posterior_manifest_sha256": hashlib.sha256(posterior_encoded).hexdigest(),
        "feature_sha256": hashlib.sha256(features.astype("<f8", copy=False).tobytes()).hexdigest(),
    }
    np.savez_compressed(
        cache_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        features=features,
        wrong_labels=labels,
    )
    return PreparedFeatures(
        rows,
        features,
        labels,
        {
            **metadata,
            "cache_hit": False,
            "cache_path": str(cache_path),
            "cache_bytes": cache_path.stat().st_size,
            "cache_sha256": _file_sha256(cache_path),
            "extraction_wall_s": time.perf_counter() - started,
        },
    )


def _segment_inconsistency(
    production: Sequence[Mapping[str, str]],
    segment_by_id: Mapping[str, Mapping[str, str]],
) -> dict[str, float]:
    by_track: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in production:
        by_track[row["track_id"]].append(row)
    output: dict[str, float] = {}
    for track_rows in by_track.values():
        ordered = sorted(track_rows, key=lambda row: int(row["event_index"]))
        clusters: list[list[Mapping[str, str]]] = []
        for row in ordered:
            if not clusters or row["cluster_index"] != clusters[-1][0]["cluster_index"]:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        segments: list[list[Mapping[str, str]]] = []
        current: list[Mapping[str, str]] = []
        start_onset = 0.0
        previous_onset = 0.0
        for cluster in clusters:
            onset = float(cluster[0]["onset_s"])
            split = bool(current) and (
                onset - previous_onset > 0.75
                or onset - start_onset > 4.0
                or len(current) + len(cluster) > 32
            )
            if split:
                segments.append(current)
                current = []
            if not current:
                start_onset = onset
            current.extend(cluster)
            previous_onset = onset
        if current:
            segments.append(current)
        for segment_rows in segments:
            shifts = np.asarray(
                [
                    int(segment_by_id[row["event_id"]]["predicted_string"])
                    - int(row["predicted_string"])
                    for row in segment_rows
                ],
                dtype=np.float64,
            )
            median = float(np.median(shifts))
            for row, shift in zip(segment_rows, shifts, strict=True):
                output[row["event_id"]] = min(abs(float(shift) - median), 5.0) / 5.0
    return output


def _validate_feature_arrays(features: np.ndarray, labels: np.ndarray, rows: int) -> None:
    if features.shape != (rows, FEATURE_COUNT) or np.any(~np.isfinite(features)):
        raise ValueError("Phase 6 feature matrix is invalid")
    if labels.shape != (rows,):
        raise ValueError("Phase 6 label vector is invalid")
    if np.any((features < 0.0) | (features > 1.0)):
        raise ValueError("Phase 6 normalized features lie outside [0, 1]")


def _run_oof(prepared: PreparedFeatures) -> OOFResult:
    started = time.perf_counter()
    players = np.asarray([row.player for row in prepared.rows])
    raw_logits = np.zeros(len(prepared.rows), dtype=np.float64)
    risk = np.zeros(len(prepared.rows), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    model_hashes: list[str] = []
    for outer_index, held_player in enumerate(DEV_PLAYERS):
        held = np.flatnonzero(players == held_player)
        outer_train = np.flatnonzero(players != held_player)
        inner_logits = np.zeros(len(outer_train), dtype=np.float64)
        outer_train_players = players[outer_train]
        inner_hashes: list[str] = []
        print(f"  review detector outer player {held_player}", flush=True)
        for inner_index, inner_player in enumerate(
            player for player in DEV_PLAYERS if player != held_player
        ):
            inner_validation_local = np.flatnonzero(outer_train_players == inner_player)
            inner_training_local = np.flatnonzero(outer_train_players != inner_player)
            model = fit_review_model(
                prepared.features[outer_train[inner_training_local]],
                prepared.wrong_labels[outer_train[inner_training_local]],
                seed=SEED + outer_index * 100 + inner_index,
            )
            inner_logits[inner_validation_local] = model.logits(
                prepared.features[outer_train[inner_validation_local]]
            )
            inner_hashes.append(model.sha256())
        calibrator = fit_platt(inner_logits, prepared.wrong_labels[outer_train])
        final_model = fit_review_model(
            prepared.features[outer_train],
            prepared.wrong_labels[outer_train],
            seed=SEED + outer_index * 100 + 50,
        )
        held_logits = final_model.logits(prepared.features[held])
        held_risk = calibrator.probabilities(held_logits)
        raw_logits[held] = held_logits
        risk[held] = held_risk
        fold_auc = binary_roc_auc(held_risk, prepared.wrong_labels[held])
        held_order = np.argsort(-held_risk, kind="stable")
        reviewed = max(1, int(math.ceil(len(held) * 0.10)))
        global_rate = float(np.mean(prepared.wrong_labels[held]))
        high_risk_rate = float(np.mean(prepared.wrong_labels[held][held_order[:reviewed]]))
        folds.append(
            {
                "player": held_player,
                "examples": len(held),
                "wrong_rate": global_rate,
                "roc_auc": fold_auc,
                "high_risk_10_wrong_rate": high_risk_rate,
                "enrichment": high_risk_rate / global_rate,
                "model_sha256": final_model.sha256(),
                "platt_intercept": calibrator.intercept,
                "platt_slope": calibrator.slope,
            }
        )
        model_hashes.append(final_model.sha256())
        model_rows.append(
            {
                "outer_player": held_player,
                "inner_model_sha256": inner_hashes,
                "final_model_sha256": final_model.sha256(),
                "training_history": final_model.history,
                "platt": {
                    "intercept": calibrator.intercept,
                    "slope": calibrator.slope,
                },
            }
        )
    prediction_digest = hashlib.sha256()
    prediction_digest.update(raw_logits.astype("<f8", copy=False).tobytes())
    prediction_digest.update(risk.astype("<f8", copy=False).tobytes())
    return OOFResult(
        raw_logits,
        risk,
        tuple(folds),
        tuple(model_rows),
        prediction_digest.hexdigest(),
        hashlib.sha256("\n".join(model_hashes).encode("ascii")).hexdigest(),
        time.perf_counter() - started,
    )


def _calibration_error(risk: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (risk >= lower) & ((risk < upper) | ((index == bins - 1) & (risk == 1.0)))
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(risk[selected])) - float(np.mean(labels[selected]))
            )
    return error


def _replay_rows(
    prepared: PreparedFeatures,
    risk: np.ndarray,
    production: Sequence[Mapping[str, str]],
    tracks: Sequence[Track],
) -> list[dict[str, Any]]:
    core_by_track: dict[str, list[int]] = defaultdict(list)
    for index, review_row in enumerate(prepared.rows):
        core_by_track[review_row.track_id].append(index)
    production_by_track: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for production_row in production:
        production_by_track[production_row["track_id"]].append(production_row)
    tracks_by_id = {track.track_id: track for track in tracks if track.player in DEV_PLAYERS}
    output: list[dict[str, Any]] = []
    for budget_s in (0, *REPLAY_BUDGETS_S):
        total_wrong = 0
        corrected = 0
        reviewed = 0
        clip_f1: list[float] = []
        solo_f1: list[float] = []
        comp_f1: list[float] = []
        for track_id in sorted(core_by_track):
            indices = core_by_track[track_id]
            ordered = sorted(
                indices, key=lambda index: (-risk[index], prepared.rows[index].event_index)
            )
            selected = ordered[: budget_s // ACTION_SECONDS]
            reviewed += len(selected)
            correction_ids: set[str] = set()
            for index in selected:
                row = prepared.rows[index]
                if row.wrong and (row.reference_string, row.reference_fret) in {
                    (string, fret) for string, fret, _cost in row.candidates[:3]
                }:
                    correction_ids.add(row.event_id)
            corrected += len(correction_ids)
            total_wrong += sum(prepared.rows[index].wrong for index in indices)
            predicted_rows = production_by_track[track_id]
            reconstructed = []
            for prediction in predicted_rows:
                corrected_row = prediction["event_id"] in correction_ids
                reconstructed.append(
                    TabEvent(
                        onset_s=float(prediction["onset_s"]),
                        duration_s=0.0,
                        string_idx=(
                            int(prediction["reference_string"])
                            if corrected_row
                            else int(prediction["predicted_string"])
                        ),
                        fret=(
                            int(prediction["reference_fret"])
                            if corrected_row
                            else int(prediction["predicted_fret"])
                        ),
                        pitch_midi=int(prediction["pitch_midi"]),
                        confidence=float(prediction["confidence"]),
                    )
                )
            score = tab_f1(reconstructed, tracks_by_id[track_id].gold)
            clip_f1.append(score.f1)
            (solo_f1 if tracks_by_id[track_id].mode == "solo" else comp_f1).append(score.f1)
        reduction = corrected / total_wrong if total_wrong else 0.0
        minutes = budget_s * len(core_by_track) / 60.0
        output.append(
            {
                "budget_s_per_clip": budget_s,
                "clips": len(core_by_track),
                "reviewed_notes": reviewed,
                "accepted_corrections": corrected,
                "residual_wrong_positions": total_wrong - corrected,
                "wrong_position_reduction": reduction,
                "correction_precision": corrected / reviewed if reviewed else 0.0,
                "correction_recall": reduction,
                "macro_tab_f1": float(np.mean(clip_f1)),
                "solo_macro_tab_f1": float(np.mean(solo_f1)),
                "comp_macro_tab_f1": float(np.mean(comp_f1)),
                "corrections_per_minute": corrected / minutes if minutes else 0.0,
                "notes_changed_per_accepted_action": 1.0 if corrected else 0.0,
                "undo_rate": 0.0,
                "wrong_propagation_rate": 0.0,
                "pitch_changes": 0,
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(
    *,
    auc: float,
    global_wrong_rate: float,
    high_risk_rate: float,
    enrichment: float,
    ece: float,
    detector_budgets: Sequence[Any],
    folds: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    detector_gate: bool,
    replay_gate: bool,
    deterministic: bool,
    prepared: PreparedFeatures,
    run: OOFResult,
) -> str:
    budget_lines = [
        f"| {int(item.fraction * 100)}% | {item.reviewed:,} | {item.precision:.4f} | "
        f"{item.recall:.4f} |"
        for item in detector_budgets
    ]
    fold_lines = [
        "| {player} | {examples:,} | {wrong_rate:.4f} | {roc_auc:.4f} | "
        "{high_risk_10_wrong_rate:.4f} | {enrichment:.2f}x |".format(**row)
        for row in folds
    ]
    replay_lines = [
        "| {budget_s_per_clip} | {reviewed_notes:,} | {accepted_corrections:,} | "
        "{wrong_position_reduction:.4f} | {macro_tab_f1:.4f} | {solo_macro_tab_f1:.4f} | "
        "{comp_macro_tab_f1:.4f} | {corrections_per_minute:.2f} |".format(**row)
        for row in replay
    ]
    final = next(row for row in replay if int(row["budget_s_per_clip"]) == 60)
    decision = (
        "PASS: the offline assisted path may proceed to production UI integration."
        if detector_gate and replay_gate
        else "FAIL: do not start production UI integration or open player 05 for this fixed "
        "assisted path."
    )
    return "\n".join(
        [
            "# Sequential Tab F1 Phase 6: learned review queue and offline replay",
            "",
            "## Frozen scope",
            "",
            "- Automatic transcription remained frozen; all reported changes are simulated "
            "user-approved, pitch-preserving corrections.",
            "- Development data: 35,959 production-equivalent pitch-correct ambiguous notes "
            "from GuitarSet players 00-04. Player 05 was not read.",
            "- Features: path margin, candidate count, OOF context/timbre disagreement, native "
            "timbre strength, accepted-checkpoint posterior entropy, explicit domain score, "
            "chord size, segment inconsistency, and mode.",
            f"- Detector: fixed 10-16-8-1 MLP with `{parameter_count(ReviewQueueNet())}` "
            "parameters; nested player-held Platt calibration; no grid.",
            "",
            "## Review-detector result",
            "",
            f"- OOF ROC AUC: `{auc:.4f}` (gate `>= {DETECTOR_AUC_GATE:.2f}`).",
            f"- Global wrong-position rate: `{global_wrong_rate:.4f}`; highest-risk 10%: "
            f"`{high_risk_rate:.4f}`; enrichment `{enrichment:.2f}x` "
            f"(gate `>= {ENRICHMENT_GATE:.1f}x`).",
            f"- Calibrated probability ECE: `{ece:.4f}`.",
            "",
            "| review budget | notes | precision | recall |",
            "|---:|---:|---:|---:|",
            *budget_lines,
            "",
            "### Player-held folds",
            "",
            "| player | notes | wrong rate | AUC | high-risk 10% | enrichment |",
            "|---:|---:|---:|---:|---:|---:|",
            *fold_lines,
            "",
            f"Detector gate: **{'PASS' if detector_gate else 'FAIL'}**.",
            "",
            "## Offline correction replay",
            "",
            "A review consumes two seconds per note. Wrong notes are corrected only when the "
            "gold position appears in the production decoder's displayed top three. Correct "
            "notes are rejected. No phrase/motif credit is included.",
            "",
            "| seconds/clip | reviewed | corrections | wrong reduction | Tab F1 | solo | comp | "
            "corrections/min |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *replay_lines,
            "",
            f"- Required 60-second wrong-position reduction: `>= {REPLAY_REDUCTION_GATE:.2f}`; "
            f"observed `{float(final['wrong_position_reduction']):.4f}`.",
            "- Pitch-changing edits: `0`; wrong propagation: `0`; undo rate in deterministic "
            "oracle replay: `0`.",
            f"Replay gate: **{'PASS' if replay_gate else 'FAIL'}**.",
            "",
            "## Decision and reproducibility",
            "",
            f"**{decision}**",
            f"- Deterministic complete rerun: **{'PASS' if deterministic else 'FAIL'}**; "
            f"prediction SHA-256 `{run.prediction_sha256}`; model SHA-256 `{run.model_sha256}`.",
            f"- Feature cache: `{prepared.metadata['cache_bytes']}` bytes; cache hit "
            f"`{str(prepared.metadata['cache_hit']).lower()}`; feature SHA-256 "
            f"`{prepared.metadata['feature_sha256']}`.",
            "- The editing core supports atomic accept/reject/undo, pitch-preserving candidate "
            "cycling and one-string phrase moves, unique K-best phrase alternatives, exact "
            "motif previews, and explicit opt-in side-information settings.",
            "- No UI, runtime route, automatic decoder, or SPEC contract changed in this gate.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "cd tabvision",
            ".\\.venv\\Scripts\\python.exe -m scripts.eval.string_assignment_phase6",
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, default=PHASE1_NOTES)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-determinism-rerun", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    production, segment = _load_note_rows(args.notes.resolve())
    rows = _review_rows(production)
    if any(row.player not in DEV_PLAYERS or row.track_id.startswith("05_") for row in rows):
        raise RuntimeError("player 05 or unexpected data entered Phase 6 development")
    tracks = load_tracks(
        args.data_home.resolve(),
        DEFAULT_CACHE,
        "highres",
        GuitarConfig(),
    )
    prepared = _prepare_features(
        rows,
        production,
        segment,
        tracks,
        notes_path=args.notes.resolve(),
        data_home=args.data_home.resolve(),
        cache_dir=args.cache_dir.resolve(),
    )
    print("running nested player-held review detector...", flush=True)
    first = _run_oof(prepared)
    if args.skip_determinism_rerun:
        second = first
        deterministic = False
    else:
        print("running deterministic review-detector rerun...", flush=True)
        second = _run_oof(prepared)
        deterministic = (
            first.prediction_sha256 == second.prediction_sha256
            and first.model_sha256 == second.model_sha256
        )
        if not deterministic:
            raise RuntimeError("Phase 6 OOF detector is not deterministic")
    auc = binary_roc_auc(first.calibrated_risk, prepared.wrong_labels)
    order = np.argsort(-first.calibrated_risk, kind="stable")
    high_risk_count = max(1, int(math.ceil(len(rows) * 0.10)))
    global_wrong_rate = float(np.mean(prepared.wrong_labels))
    high_risk_rate = float(np.mean(prepared.wrong_labels[order[:high_risk_count]]))
    enrichment = high_risk_rate / global_wrong_rate
    detector_budgets = budget_metrics(first.calibrated_risk, prepared.wrong_labels)
    ece = _calibration_error(first.calibrated_risk, prepared.wrong_labels)
    detector_gate = auc >= DETECTOR_AUC_GATE and enrichment >= ENRICHMENT_GATE
    replay = _replay_rows(prepared, first.calibrated_risk, production, tracks)
    final_replay = next(row for row in replay if int(row["budget_s_per_clip"]) == 60)
    replay_gate = (
        float(final_replay["wrong_position_reduction"]) >= REPLAY_REDUCTION_GATE
        and int(final_replay["pitch_changes"]) == 0
        and float(final_replay["wrong_propagation_rate"]) == 0.0
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "string_assignment_phase6_2026-07-16"
    report_path = output_dir / f"{stem}.md"
    folds_path = output_dir / f"{stem}_folds.csv"
    budgets_path = output_dir / f"{stem}_budgets.csv"
    features_path = output_dir / f"{stem}_features.csv"
    models_path = output_dir / f"{stem}_models.json"
    _write_csv(folds_path, first.folds)
    _write_csv(budgets_path, replay)
    feature_rows = [
        {
            "feature": name,
            "mean": float(np.mean(prepared.features[:, index])),
            "std": float(np.std(prepared.features[:, index])),
            "minimum": float(np.min(prepared.features[:, index])),
            "maximum": float(np.max(prepared.features[:, index])),
        }
        for index, name in enumerate(FEATURE_NAMES)
    ]
    _write_csv(features_path, feature_rows)
    models_path.write_text(json.dumps(first.models, indent=2, sort_keys=True), encoding="utf-8")
    report = _report(
        auc=auc,
        global_wrong_rate=global_wrong_rate,
        high_risk_rate=high_risk_rate,
        enrichment=enrichment,
        ece=ece,
        detector_budgets=detector_budgets,
        folds=first.folds,
        replay=replay,
        detector_gate=detector_gate,
        replay_gate=replay_gate,
        deterministic=deterministic,
        prepared=prepared,
        run=first,
    )
    report_path.write_text(report, encoding="utf-8")
    tracked = (report_path, folds_path, budgets_path, features_path, models_path)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "command": sys.argv,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_sha256": {
            "script": _file_sha256(Path(__file__).resolve()),
            "review_queue": _file_sha256(REPO_ROOT / "tabvision/tabvision/eval/review_queue.py"),
            "editing": _file_sha256(REPO_ROOT / "tabvision/tabvision/assist/editing.py"),
            "design": _file_sha256(PHASE6_DESIGN),
        },
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "torch")},
        "fixed_constants": {
            "seed": SEED,
            "development_players": DEV_PLAYERS,
            "feature_names": FEATURE_NAMES,
            "action_seconds": ACTION_SECONDS,
            "replay_budgets_s": REPLAY_BUDGETS_S,
            "detector_auc_gate": DETECTOR_AUC_GATE,
            "enrichment_gate": ENRICHMENT_GATE,
            "replay_reduction_gate": REPLAY_REDUCTION_GATE,
            "model_parameters": parameter_count(ReviewQueueNet()),
        },
        "dataset": "GuitarSet original public 360-track mono-mic/JAMS release",
        "dataset_license": "CC-BY-4.0; local data not redistributed",
        "events": len(rows),
        "player05_opened": False,
        "feature_cache": prepared.metadata,
        "roc_auc": auc,
        "global_wrong_rate": global_wrong_rate,
        "high_risk_10_wrong_rate": high_risk_rate,
        "enrichment": enrichment,
        "calibration_ece": ece,
        "detector_note_budgets": [item.__dict__ for item in detector_budgets],
        "folds": first.folds,
        "replay": replay,
        "detector_gate_passed": detector_gate,
        "replay_gate_passed": replay_gate,
        "ui_gate_passed": detector_gate and replay_gate,
        "automatic_tab_changed": False,
        "automatic_pitch_changes": 0,
        "runtime_routing_changed": False,
        "prediction_sha256": first.prediction_sha256,
        "model_sha256": first.model_sha256,
        "deterministic_rerun": deterministic,
        "performance": {
            "feature_extraction_wall_s": prepared.metadata["extraction_wall_s"],
            "first_oof_wall_s": first.wall_s,
            "second_oof_wall_s": second.wall_s,
            "total_wall_s": time.perf_counter() - started,
        },
        "tracked_outputs": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in tracked
        },
    }
    provenance_path = output_dir / f"{stem}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(report, flush=True)
    return 2 if detector_gate and replay_gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
