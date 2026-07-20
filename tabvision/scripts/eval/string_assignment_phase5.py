"""Sequential Tab F1 Phase 5 direct per-string gold-pitch gate.

The development-only first stage trains the frozen original six-string
multi-task network on GuitarSet players 00--04. It must clear the gold-pitch
OOF gate before real-event integration or player-05 confirmation is allowed.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from scripts.eval.string_assignment_phase4 import (
    DEV_PLAYERS,
    ProbeRow,
    _audio_manifest,
    _candidate_arrays,
    _file_sha256,
    _gold_by_player,
    _load_rows,
)
from tabvision.eval.direct_per_string import (
    MEL_BANDS,
    MIDI_BEGIN,
    PITCH_CLASSES,
    SAMPLE_RATE,
    STRINGS,
    WINDOW_SAMPLES,
    DirectPerStringNet,
    extract_window,
    gold_pitch_string_scores,
    log_mel_batch,
    multitask_loss,
    parameter_count,
)
from tabvision.fusion.position_prior import learn_pitch_position_prior
from tabvision.types import GuitarConfig, TabEvent

SEED = 5519
SCHEMA_VERSION = 1
MAX_CANDIDATES = 6
BATCH_SIZE = 256
PREPROCESS_BATCH_SIZE = 128
INNER_EPOCHS = 2
FINAL_EPOCHS = 3
GRID: tuple[tuple[float, float], ...] = ((3.0e-4, 1.0e-4), (1.0e-3, 1.0e-5))
DIRECT_PRIOR_WEIGHT = 1.0
BEST_PREVIOUS_TOP1 = 0.6620595678411524
GOLD_PITCH_GATE_DELTA = 0.05
GOLD_PITCH_GATE_TARGET = BEST_PREVIOUS_TOP1 + GOLD_PITCH_GATE_DELTA
ONSET_TOLERANCE_S = 0.05

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NOTES = REPO_ROOT / "docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv"
DEFAULT_DATA_HOME = Path.home() / ".tabvision/data/guitarset"
DEFAULT_CACHE_DIR = Path.home() / ".tabvision/cache/string_assignment_phase5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/EVAL_REPORTS"
PHASE4_FOLDS = REPO_ROOT / "docs/EVAL_REPORTS/string_assignment_phase4_2026-07-16_folds.csv"


@dataclass(frozen=True)
class PreparedData:
    features: np.ndarray
    onset_targets: np.ndarray
    frame_targets: np.ndarray
    global_pitch_targets: np.ndarray
    occupancy_targets: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OOFRun:
    string_scores: np.ndarray
    direct_predictions: np.ndarray
    combined_predictions: np.ndarray
    direct_top3: np.ndarray
    combined_top3: np.ndarray
    folds: tuple[dict[str, Any], ...]
    selections: tuple[dict[str, Any], ...]
    prediction_sha256: str
    model_sha256: str
    training_wall_s: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _source_manifest() -> dict[str, str]:
    paths = {
        "probe_script": Path(__file__).resolve(),
        "model_module": REPO_ROOT / "tabvision/tabvision/eval/direct_per_string.py",
        "phase4_rows": REPO_ROOT / "tabvision/scripts/eval/string_assignment_phase4.py",
        "phase5_design": REPO_ROOT / "docs/plans/2026-07-16-tab-f1-phase5-data-license-design.md",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _cache_metadata(
    rows: Sequence[ProbeRow],
    notes_path: Path,
    audio_manifest_sha256: str,
) -> dict[str, Any]:
    event_ids = "\n".join(row.event_id for row in rows).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "notes_sha256": _file_sha256(notes_path),
        "event_ids_sha256": hashlib.sha256(event_ids).hexdigest(),
        "audio_manifest_sha256": audio_manifest_sha256,
        "source_sha256": _source_manifest(),
        "rows": len(rows),
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "mel_bands": MEL_BANDS,
        "time_frames": 65,
    }


def _prepare_data(
    rows: Sequence[ProbeRow],
    notes_path: Path,
    data_home: Path,
    cache_dir: Path,
    audio_manifest_sha256: str,
) -> PreparedData:
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / "direct_per_string_logmel_v1.npy"
    target_path = cache_dir / "direct_per_string_targets_v1.npz"
    metadata_path = cache_dir / "direct_per_string_metadata_v1.json"
    expected = _cache_metadata(rows, notes_path, audio_manifest_sha256)
    if feature_path.is_file() and target_path.is_file() and metadata_path.is_file():
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
            features = np.load(feature_path, mmap_mode="r")
            with np.load(target_path, allow_pickle=False) as payload:
                targets = {name: np.asarray(payload[name]) for name in payload.files}
            if actual == expected and features.shape == (len(rows), MEL_BANDS, 65):
                prepared = _validated_prepared(features, targets, actual)
                return PreparedData(
                    prepared.features,
                    prepared.onset_targets,
                    prepared.frame_targets,
                    prepared.global_pitch_targets,
                    prepared.occupancy_targets,
                    {
                        **actual,
                        "cache_hit": True,
                        "extraction_wall_s": 0.0,
                        "feature_path": str(feature_path),
                        "feature_bytes": feature_path.stat().st_size,
                        "feature_sha256": _file_sha256(feature_path),
                        "target_path": str(target_path),
                        "target_bytes": target_path.stat().st_size,
                        "target_sha256": _file_sha256(target_path),
                    },
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    temporary_feature = cache_dir / f".{feature_path.name}.{os.getpid()}.tmp.npy"
    features = np.lib.format.open_memmap(
        temporary_feature,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), MEL_BANDS, 65),
    )
    onset_targets = np.zeros((len(rows), STRINGS, PITCH_CLASSES), dtype=np.uint8)
    frame_targets = np.zeros_like(onset_targets)
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[row.track_id].append(index)

    for track_number, (track_id, indices) in enumerate(sorted(by_track.items()), start=1):
        audio_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        annotation_path = data_home / "annotation" / f"{track_id}.jams"
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        signal = np.asarray(waveform, dtype=np.float32)
        if signal.ndim == 2:
            signal = np.asarray(np.mean(signal, axis=1, dtype=np.float32), dtype=np.float32)
        if signal.ndim != 1:
            raise RuntimeError(f"unexpected audio shape for {track_id}: {signal.shape}")
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
            signal = resample_poly(
                signal,
                SAMPLE_RATE // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32)
        gold = parse_gold(annotation_path)
        for batch_start in range(0, len(indices), PREPROCESS_BATCH_SIZE):
            batch_indices = indices[batch_start : batch_start + PREPROCESS_BATCH_SIZE]
            windows = np.stack(
                [extract_window(signal, rows[index].onset_s) for index in batch_indices]
            )
            with torch.inference_mode():
                log_mel = log_mel_batch(torch.from_numpy(windows)).numpy().astype(np.float16)
            features[batch_indices] = log_mel
        for index in indices:
            _set_targets(
                rows[index],
                gold,
                onset_targets[index],
                frame_targets[index],
            )
        if track_number % 10 == 0:
            print(f"  direct-model cache: {track_number}/{len(by_track)} tracks", flush=True)
    features.flush()
    memory_map = getattr(features, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
    del features
    os.replace(temporary_feature, feature_path)

    global_pitch_targets = np.max(onset_targets, axis=1).astype(np.uint8)
    occupancy_targets = np.max(frame_targets, axis=2).astype(np.uint8)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target_path.name}.", suffix=".npz", dir=cache_dir, delete=False
    ) as handle:
        temporary_target = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_target,
            onset_targets=onset_targets,
            frame_targets=frame_targets,
            global_pitch_targets=global_pitch_targets,
            occupancy_targets=occupancy_targets,
        )
        os.replace(temporary_target, target_path)
    finally:
        if temporary_target.exists():
            temporary_target.unlink()
    metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
    features = np.load(feature_path, mmap_mode="r")
    prepared = _validated_prepared(
        features,
        {
            "onset_targets": onset_targets,
            "frame_targets": frame_targets,
            "global_pitch_targets": global_pitch_targets,
            "occupancy_targets": occupancy_targets,
        },
        expected,
    )
    return PreparedData(
        prepared.features,
        prepared.onset_targets,
        prepared.frame_targets,
        prepared.global_pitch_targets,
        prepared.occupancy_targets,
        {
            **expected,
            "cache_hit": False,
            "extraction_wall_s": time.perf_counter() - started,
            "feature_path": str(feature_path),
            "feature_bytes": feature_path.stat().st_size,
            "feature_sha256": _file_sha256(feature_path),
            "target_path": str(target_path),
            "target_bytes": target_path.stat().st_size,
            "target_sha256": _file_sha256(target_path),
        },
    )


def parse_gold(path: Path) -> list[TabEvent]:
    from tabvision.eval.guitarset_audio import parse_guitarset_jams

    return parse_guitarset_jams(path)


def _set_targets(
    row: ProbeRow,
    gold: Sequence[TabEvent],
    onset_target: np.ndarray,
    frame_target: np.ndarray,
) -> None:
    for event in gold:
        pitch_index = event.pitch_midi - MIDI_BEGIN
        if not 0 <= pitch_index < PITCH_CLASSES:
            continue
        if abs(event.onset_s - row.onset_s) <= ONSET_TOLERANCE_S + 1.0e-9:
            onset_target[event.string_idx, pitch_index] = 1
        if event.onset_s <= row.onset_s + 1.0e-9 <= event.onset_s + event.duration_s + 1.0e-9:
            frame_target[event.string_idx, pitch_index] = 1
    reference_pitch = row.pitch_midi - MIDI_BEGIN
    onset_target[row.reference_string, reference_pitch] = 1
    frame_target[row.reference_string, reference_pitch] = 1


def _validated_prepared(
    features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    metadata: dict[str, Any],
) -> PreparedData:
    rows = int(metadata["rows"])
    expected_shapes = {
        "onset_targets": (rows, STRINGS, PITCH_CLASSES),
        "frame_targets": (rows, STRINGS, PITCH_CLASSES),
        "global_pitch_targets": (rows, PITCH_CLASSES),
        "occupancy_targets": (rows, STRINGS),
    }
    if features.shape != (rows, MEL_BANDS, 65) or np.any(~np.isfinite(features)):
        raise ValueError("direct-model feature cache is invalid")
    for name, shape in expected_shapes.items():
        if name not in targets or targets[name].shape != shape:
            raise ValueError(f"direct-model target cache is invalid: {name}")
        if np.any((targets[name] != 0) & (targets[name] != 1)):
            raise ValueError(f"direct-model target cache is non-binary: {name}")
    return PreparedData(
        features,
        targets["onset_targets"],
        targets["frame_targets"],
        targets["global_pitch_targets"],
        targets["occupancy_targets"],
        metadata,
    )


def _balanced_probabilities(indices: np.ndarray, rows: Sequence[ProbeRow]) -> np.ndarray:
    keys = [
        (
            rows[int(index)].reference_string,
            rows[int(index)].pitch_midi // 12,
            rows[int(index)].mode,
        )
        for index in indices
    ]
    counts = Counter(keys)
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)
    return weights / np.sum(weights)


def _train_model(
    prepared: PreparedData,
    rows: Sequence[ProbeRow],
    train_indices: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> tuple[DirectPerStringNet, tuple[dict[str, float], ...]]:
    _seed_everything(seed)
    model = DirectPerStringNet()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    probabilities = _balanced_probabilities(train_indices, rows)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        selected = rng.choice(
            train_indices,
            size=len(train_indices),
            replace=True,
            p=probabilities,
        )
        totals: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for start in range(0, len(selected), BATCH_SIZE):
            batch = selected[start : start + BATCH_SIZE]
            outputs = model(
                torch.from_numpy(np.asarray(prepared.features[batch], dtype=np.float32))
            )
            losses = multitask_loss(
                outputs,
                torch.from_numpy(prepared.onset_targets[batch].astype(np.float32)),
                torch.from_numpy(prepared.frame_targets[batch].astype(np.float32)),
                torch.from_numpy(prepared.global_pitch_targets[batch].astype(np.float32)),
                torch.from_numpy(prepared.occupancy_targets[batch].astype(np.float32)),
            )
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            optimizer.step()
            for name in (
                "total",
                "onset",
                "frame",
                "global_pitch",
                "occupancy",
                "duplicate_inhibition",
            ):
                totals[name] += float(getattr(losses, name).detach())
            batches += 1
        history.append({name: value / batches for name, value in totals.items()})
    return model, tuple(history)


def _score_model(
    model: DirectPerStringNet,
    prepared: PreparedData,
    rows: Sequence[ProbeRow],
    indices: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((len(indices), STRINGS), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start : start + BATCH_SIZE]
            outputs = model(
                torch.from_numpy(np.asarray(prepared.features[batch], dtype=np.float32))
            )
            pitches = torch.as_tensor([rows[int(index)].pitch_midi for index in batch])
            values = gold_pitch_string_scores(outputs, pitches)
            scores[start : start + len(batch)] = values.numpy()
    return scores


def _state_sha256(model: DirectPerStringNet) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _prior_probabilities(
    rows: Sequence[ProbeRow],
    indices: np.ndarray,
    strings: np.ndarray,
    frets: np.ndarray,
    gold_by_player: Mapping[str, Sequence[TabEvent]],
    training_players: Sequence[str],
) -> np.ndarray:
    """Return candidate-normalized position priors learned from training players only."""

    examples = [event for player in training_players for event in gold_by_player[player]]
    prior = learn_pitch_position_prior(examples, cfg=GuitarConfig(), alpha=1.0, power=2.0)
    probabilities = np.zeros((len(indices), MAX_CANDIDATES), dtype=np.float64)
    for output_index, row_index in enumerate(indices):
        row = rows[int(row_index)]
        matrix = prior.matrix_for_pitch(row.pitch_midi)
        if matrix is None:
            raise RuntimeError(f"position prior missing pitch {row.pitch_midi}")
        for column in np.flatnonzero(strings[int(row_index)] >= 0):
            probabilities[output_index, column] = matrix[
                int(strings[int(row_index), column]),
                int(frets[int(row_index), column]),
            ]
        total = float(np.sum(probabilities[output_index]))
        if total <= 0.0:
            raise RuntimeError(f"position prior has zero mass for {row.event_id}")
        probabilities[output_index] /= total
    return probabilities


def _candidate_predictions(
    string_scores: np.ndarray,
    indices: np.ndarray,
    strings: np.ndarray,
    position_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_scores = np.full((len(indices), MAX_CANDIDATES), -np.inf, dtype=np.float64)
    for output_index, row_index in enumerate(indices):
        valid = np.flatnonzero(strings[int(row_index)] >= 0)
        candidate_scores[output_index, valid] = string_scores[
            output_index, strings[int(row_index), valid]
        ]
    combined_scores = candidate_scores + DIRECT_PRIOR_WEIGHT * np.log(
        np.maximum(position_prior, 1.0e-12)
    )
    direct_order = np.argsort(-candidate_scores, axis=1, kind="stable")
    combined_order = np.argsort(-combined_scores, axis=1, kind="stable")
    return (
        direct_order[:, 0].astype(np.int8),
        combined_order[:, 0].astype(np.int8),
        direct_order[:, :3].astype(np.int8),
        combined_order[:, :3].astype(np.int8),
    )


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def _top3_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.any(predictions == labels[:, None], axis=1)))


def _run_oof(
    prepared: PreparedData,
    rows: Sequence[ProbeRow],
    strings: np.ndarray,
    frets: np.ndarray,
    labels: np.ndarray,
    data_home: Path,
) -> OOFRun:
    started = time.perf_counter()
    gold_by_player = _gold_by_player(data_home)
    all_scores = np.zeros((len(rows), STRINGS), dtype=np.float32)
    direct_predictions = np.full(len(rows), -1, dtype=np.int8)
    combined_predictions = np.full(len(rows), -1, dtype=np.int8)
    direct_top3 = np.full((len(rows), 3), -1, dtype=np.int8)
    combined_top3 = np.full((len(rows), 3), -1, dtype=np.int8)
    folds: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    model_hashes: list[str] = []

    for outer_index, held_player in enumerate(DEV_PLAYERS):
        inner_player = DEV_PLAYERS[(outer_index + 1) % len(DEV_PLAYERS)]
        inner_train_players = tuple(
            player for player in DEV_PLAYERS if player not in (held_player, inner_player)
        )
        inner_train = np.asarray(
            [index for index, row in enumerate(rows) if row.player in inner_train_players],
            dtype=np.int64,
        )
        inner_validation = np.asarray(
            [index for index, row in enumerate(rows) if row.player == inner_player],
            dtype=np.int64,
        )
        inner_prior = _prior_probabilities(
            rows,
            inner_validation,
            strings,
            frets,
            gold_by_player,
            inner_train_players,
        )
        grid_rows: list[dict[str, Any]] = []
        for grid_index, (learning_rate, weight_decay) in enumerate(GRID):
            print(
                f"  outer {held_player}: inner {inner_player}, grid {grid_index + 1}/{len(GRID)}",
                flush=True,
            )
            model, history = _train_model(
                prepared,
                rows,
                inner_train,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=INNER_EPOCHS,
                seed=SEED + outer_index * 100 + grid_index,
            )
            scores = _score_model(model, prepared, rows, inner_validation)
            direct, combined, direct3, combined3 = _candidate_predictions(
                scores,
                inner_validation,
                strings,
                inner_prior,
            )
            inner_labels = labels[inner_validation]
            grid_rows.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "inner_direct_accuracy": _accuracy(direct, inner_labels),
                    "inner_combined_accuracy": _accuracy(combined, inner_labels),
                    "inner_direct_top3": _top3_accuracy(direct3, inner_labels),
                    "inner_combined_top3": _top3_accuracy(combined3, inner_labels),
                    "history": history,
                    "model_sha256": _state_sha256(model),
                }
            )
        best = min(
            grid_rows,
            key=lambda item: (
                -float(item["inner_combined_accuracy"]),
                float(item["learning_rate"]),
                float(item["weight_decay"]),
            ),
        )
        outer_train_players = tuple(player for player in DEV_PLAYERS if player != held_player)
        outer_train = np.asarray(
            [index for index, row in enumerate(rows) if row.player in outer_train_players],
            dtype=np.int64,
        )
        held_indices = np.asarray(
            [index for index, row in enumerate(rows) if row.player == held_player],
            dtype=np.int64,
        )
        print(
            f"  outer {held_player}: final lr={best['learning_rate']}, wd={best['weight_decay']}",
            flush=True,
        )
        final_model, final_history = _train_model(
            prepared,
            rows,
            outer_train,
            learning_rate=float(best["learning_rate"]),
            weight_decay=float(best["weight_decay"]),
            epochs=FINAL_EPOCHS,
            seed=SEED + outer_index * 100 + 50,
        )
        held_scores = _score_model(final_model, prepared, rows, held_indices)
        held_prior = _prior_probabilities(
            rows,
            held_indices,
            strings,
            frets,
            gold_by_player,
            outer_train_players,
        )
        direct, combined, direct3, combined3 = _candidate_predictions(
            held_scores,
            held_indices,
            strings,
            held_prior,
        )
        all_scores[held_indices] = held_scores
        direct_predictions[held_indices] = direct
        combined_predictions[held_indices] = combined
        direct_top3[held_indices] = direct3
        combined_top3[held_indices] = combined3
        model_hash = _state_sha256(final_model)
        model_hashes.append(model_hash)
        held_labels = labels[held_indices]
        phase4_baseline = _phase4_fold_accuracy(held_player)
        folds.append(
            {
                "player": held_player,
                "examples": len(held_indices),
                "phase4_best_accuracy": phase4_baseline,
                "direct_accuracy": _accuracy(direct, held_labels),
                "combined_accuracy": _accuracy(combined, held_labels),
                "combined_delta_vs_phase4": _accuracy(combined, held_labels) - phase4_baseline,
                "direct_top3": _top3_accuracy(direct3, held_labels),
                "combined_top3": _top3_accuracy(combined3, held_labels),
                "model_sha256": model_hash,
            }
        )
        selections.append(
            {
                "outer_player": held_player,
                "inner_validation_player": inner_player,
                "inner_training_players": inner_train_players,
                "outer_training_players": outer_train_players,
                "grid": grid_rows,
                "selected_learning_rate": best["learning_rate"],
                "selected_weight_decay": best["weight_decay"],
                "final_history": final_history,
                "final_model_sha256": model_hash,
            }
        )

    if np.any(direct_predictions < 0) or np.any(combined_predictions < 0):
        raise RuntimeError("OOF predictions are incomplete")
    prediction_digest = hashlib.sha256()
    for array in (
        all_scores.astype("<f4", copy=False),
        direct_predictions,
        combined_predictions,
        direct_top3,
        combined_top3,
    ):
        prediction_digest.update(np.asarray(array).tobytes())
    model_digest = hashlib.sha256("\n".join(model_hashes).encode("ascii")).hexdigest()
    return OOFRun(
        all_scores,
        direct_predictions,
        combined_predictions,
        direct_top3,
        combined_top3,
        tuple(folds),
        tuple(selections),
        prediction_digest.hexdigest(),
        model_digest,
        time.perf_counter() - started,
    )


def _phase4_fold_accuracy(player: str) -> float:
    with PHASE4_FOLDS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["player"] == player:
                return float(row["combined_accuracy"])
    raise RuntimeError(f"Phase 4 fold is missing player {player}")


def _phase4_condition(name: str, field: str) -> float:
    path = DEFAULT_OUTPUT_DIR / "string_assignment_phase4_2026-07-16_conditions.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == name:
                return float(row[field])
    raise RuntimeError(f"Phase 4 condition is missing {name}")


def _condition_rows(
    rows: Sequence[ProbeRow], labels: np.ndarray, run: OOFRun
) -> list[dict[str, Any]]:
    production_top1 = float(np.mean([row.baseline_correct for row in rows]))
    production_top3 = float(np.mean([row.baseline_top3 for row in rows]))
    conditions: list[dict[str, Any]] = [
        {
            "condition": "production_prior",
            "ambiguous_top1": production_top1,
            "ambiguous_top3": production_top3,
            "delta_vs_best_previous": production_top1 - BEST_PREVIOUS_TOP1,
        },
        {
            "condition": "best_previous_contextual_timbral",
            "ambiguous_top1": BEST_PREVIOUS_TOP1,
            "ambiguous_top3": _phase4_condition("oof_position_plus_native_audio", "ambiguous_top3"),
            "delta_vs_best_previous": 0.0,
        },
        {
            "condition": "direct_per_string_only",
            "ambiguous_top1": _accuracy(run.direct_predictions, labels),
            "ambiguous_top3": _top3_accuracy(run.direct_top3, labels),
            "delta_vs_best_previous": _accuracy(run.direct_predictions, labels)
            - BEST_PREVIOUS_TOP1,
        },
        {
            "condition": "direct_per_string_plus_oof_position",
            "ambiguous_top1": _accuracy(run.combined_predictions, labels),
            "ambiguous_top3": _top3_accuracy(run.combined_top3, labels),
            "delta_vs_best_previous": _accuracy(run.combined_predictions, labels)
            - BEST_PREVIOUS_TOP1,
        },
    ]
    for condition in conditions:
        condition["same_pitch_wrong_rate"] = 1.0 - float(condition["ambiguous_top1"])
    return conditions


def _error_summary_rows(
    rows: Sequence[ProbeRow], labels: np.ndarray, run: OOFRun
) -> list[dict[str, Any]]:
    dimensions: dict[str, list[str]] = {
        "player": [row.player for row in rows],
        "mode": [row.mode for row in rows],
        "style": [row.style for row in rows],
        "candidate_count": [str(len(row.candidate_strings)) for row in rows],
        "reference_string": [str(row.reference_string) for row in rows],
        "pitch_midi": [str(row.pitch_midi) for row in rows],
    }
    output: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            indices = np.asarray([i for i, item in enumerate(values) if item == value])
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "examples": len(indices),
                    "direct_accuracy": _accuracy(run.direct_predictions[indices], labels[indices]),
                    "combined_accuracy": _accuracy(
                        run.combined_predictions[indices], labels[indices]
                    ),
                    "combined_wrong": int(
                        np.sum(run.combined_predictions[indices] != labels[indices])
                    ),
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


def _peak_rss_bytes() -> int:
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        if not get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
        return int(counters.peak_working_set_size)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
    maximum_rss = int(usage.ru_maxrss)
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def _report(
    conditions: Sequence[Mapping[str, Any]],
    run: OOFRun,
    *,
    gate_passed: bool,
    deterministic: bool,
    cache_metadata: Mapping[str, Any],
    source_duration_s: float,
    peak_rss_bytes: int,
    total_wall_s: float,
) -> str:
    condition_lines = [
        "| {condition} | {ambiguous_top1:.4f} | {ambiguous_top3:.4f} | "
        "{same_pitch_wrong_rate:.4f} | {delta_vs_best_previous:+.4f} |".format(**row)
        for row in conditions
    ]
    fold_lines = [
        "| {player} | {examples:,} | {phase4_best_accuracy:.4f} | {direct_accuracy:.4f} | "
        "{combined_accuracy:.4f} | {combined_delta_vs_phase4:+.4f} |".format(**row)
        for row in run.folds
    ]
    selection_lines = [
        "| {outer_player} | {inner_validation_player} | {selected_learning_rate:.0e} | "
        "{selected_weight_decay:.0e} |".format(**row)
        for row in run.selections
    ]
    primary = next(
        row for row in conditions if row["condition"] == "direct_per_string_plus_oof_position"
    )
    improved_folds = sum(float(row["combined_delta_vs_phase4"]) > 0.0 for row in run.folds)
    decision = (
        "PASS: open real-event second-opinion integration on players 00-04; keep player 05 "
        "sealed until the fixed real-event configuration is frozen."
        if gate_passed
        else "FAIL: close this direct-model branch. Do not open player 05, integrate the "
        "model, enlarge it, or replace the accepted high-resolution backend."
    )
    extraction_per_minute = (
        float(cache_metadata["extraction_wall_s"]) * 60.0 / source_duration_s
        if float(cache_metadata["extraction_wall_s"]) > 0.0
        else 0.0
    )
    return "\n".join(
        [
            "# Sequential Tab F1 Phase 5: direct per-string gold-pitch gate",
            "",
            "## Frozen data and license design",
            "",
            "- Acoustic acceptance core: GuitarSet microphone audio and hex-derived JAMS "
            "labels, CC-BY-4.0, players 00-04 only. Player 05 was not read.",
            "- Guitar-TECHS remains a separate CC-BY-4.0 electric-domain track and was not "
            "mixed into this model.",
            "- GOAT was excluded because no dataset download/license suitable for shipped "
            "derived weights was found. SynthTab, GAPS, and private data were excluded.",
            "- The architecture and training implementation are original project code; no "
            "external model source was copied.",
            "- The complete pre-fit protocol is frozen in "
            "`docs/plans/2026-07-16-tab-f1-phase5-data-license-design.md`.",
            "",
            "## Fixed model and evaluation",
            "",
            f"- Model: shared three-block convolutional encoder with six onset heads, six "
            f"frame/pitch heads, global-pitch head, occupancy head, and duplicate-pitch "
            f"inhibition; `{parameter_count(DirectPerStringNet()):,}` trainable parameters.",
            "- Input: 512 ms event window (64 ms before, 448 ms after), resampled to 16 kHz, "
            "64 log-mel bands, 512-sample FFT, 128-sample hop.",
            "- Examples: 35,959 frozen production-equivalent pitch-correct ambiguous events.",
            "- Validation: five player-held-out folds. Each outer fold selects between only "
            "the two frozen learning-rate/weight-decay settings on the next player, with the "
            "outer player excluded from all fitting and selection.",
            "- Primary score: direct six-string event logit plus player-held-out position-prior "
            "log probability at fixed weight 1.0.",
            "",
            "## OOF gold-pitch result",
            "",
            "| condition | ambiguous top-1 | top-3 | wrong rate | delta vs best previous |",
            "|---|---:|---:|---:|---:|",
            *condition_lines,
            "",
            "### Player folds",
            "",
            "| held player | examples | Phase 4 best | direct only | direct + prior | delta |",
            "|---:|---:|---:|---:|---:|---:|",
            *fold_lines,
            "",
            "### Nested selections",
            "",
            "| outer player | inner validation | learning rate | weight decay |",
            "|---:|---:|---:|---:|",
            *selection_lines,
            "",
            "## Gate decision",
            "",
            f"- Required top-1: best previous `{BEST_PREVIOUS_TOP1:.4f}` plus "
            f"`{GOLD_PITCH_GATE_DELTA:.2f}` = `>= {GOLD_PITCH_GATE_TARGET:.4f}`.",
            f"- Observed primary top-1: `{float(primary['ambiguous_top1']):.4f}` "
            f"(`{float(primary['delta_vs_best_previous']):+.4f}`).",
            f"- Player folds above their Phase 4 comparator: `{improved_folds}/5`.",
            f"- Gold-pitch gate: **{'PASS' if gate_passed else 'FAIL'}**.",
            f"- Decision: **{decision}**",
            "",
            "## Runtime, reproducibility, and production safety",
            "",
            f"- Feature cache: `{cache_metadata['feature_bytes']}` bytes; target cache: "
            f"`{cache_metadata['target_bytes']}` bytes; cache hit "
            f"`{str(cache_metadata['cache_hit']).lower()}`.",
            f"- Uncached feature extraction: `{float(cache_metadata['extraction_wall_s']):.3f} s` "
            f"(`{extraction_per_minute:.3f} s` per source minute).",
            f"- OOF training/evaluation: `{run.training_wall_s:.3f} s`; total command: "
            f"`{total_wall_s:.3f} s`; peak process working set `{peak_rss_bytes}` bytes.",
            f"- Deterministic complete rerun: **{'PASS' if deterministic else 'FAIL'}**; "
            f"prediction SHA-256 `{run.prediction_sha256}`; model SHA-256 "
            f"`{run.model_sha256}`.",
            "- No model artifact was registered and no runtime path was changed at this gate. "
            "Until the real-event gate passes, shipping artifact size and added automatic "
            "latency are zero; onset, pitch, Tab events, and all routing behavior are unchanged.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "cd tabvision",
            ".\\.venv\\Scripts\\python.exe -m scripts.eval.string_assignment_phase5",
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-determinism-rerun",
        action="store_true",
        help="development-only timing aid; canonical evidence must not use this flag",
    )
    parser.add_argument(
        "--compare-to-provenance",
        type=Path,
        help="compare this independent run with a prior complete-run provenance record",
    )
    args = parser.parse_args()
    if args.skip_determinism_rerun and args.compare_to_provenance is not None:
        parser.error("--skip-determinism-rerun and --compare-to-provenance are exclusive")
    comparison: dict[str, Any] | None = None
    if args.compare_to_provenance is not None:
        comparison = json.loads(args.compare_to_provenance.resolve().read_text(encoding="utf-8"))
    started = time.perf_counter()
    cfg = GuitarConfig()
    model_parameters = parameter_count(DirectPerStringNet())
    if model_parameters >= 5_000_000:
        raise RuntimeError(f"Phase 5 architecture exceeds parameter cap: {model_parameters}")
    rows = _load_rows(args.notes.resolve(), cfg)
    if any(row.player not in DEV_PLAYERS or row.track_id.startswith("05_") for row in rows):
        raise RuntimeError("player 05 or an unexpected player entered Phase 5 development")
    print("hashing 300 development microphone files...", flush=True)
    audio_manifest, audio_manifest_sha256 = _audio_manifest(rows, args.data_home.resolve())
    if any(str(item["track_id"]).startswith("05_") for item in audio_manifest):
        raise RuntimeError("player 05 entered the Phase 5 audio manifest")
    source_duration_s = sum(
        float(item["frames"]) / float(item["sample_rate"]) for item in audio_manifest
    )
    print("preparing direct-model features and multi-task labels...", flush=True)
    prepared = _prepare_data(
        rows,
        args.notes.resolve(),
        args.data_home.resolve(),
        args.cache_dir.resolve(),
        audio_manifest_sha256,
    )
    strings, frets, labels = _candidate_arrays(rows)
    print("running fixed nested five-player OOF evaluation...", flush=True)
    first = _run_oof(prepared, rows, strings, frets, labels, args.data_home.resolve())
    if comparison is not None:
        second = first
        deterministic = first.prediction_sha256 == comparison.get(
            "prediction_sha256"
        ) and first.model_sha256 == comparison.get("model_sha256")
        if not deterministic:
            raise RuntimeError("Phase 5 independent runs are not deterministic")
    elif args.skip_determinism_rerun:
        second = first
        deterministic = False
    else:
        print("running complete deterministic OOF rerun...", flush=True)
        second = _run_oof(prepared, rows, strings, frets, labels, args.data_home.resolve())
        deterministic = (
            first.prediction_sha256 == second.prediction_sha256
            and first.model_sha256 == second.model_sha256
        )
        if not deterministic:
            raise RuntimeError("Phase 5 OOF evaluation is not deterministic")

    conditions = _condition_rows(rows, labels, first)
    primary = next(
        row for row in conditions if row["condition"] == "direct_per_string_plus_oof_position"
    )
    gate_passed = float(primary["ambiguous_top1"]) >= GOLD_PITCH_GATE_TARGET
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "string_assignment_phase5_2026-07-16"
    report_path = output_dir / f"{stem}.md"
    conditions_path = output_dir / f"{stem}_conditions.csv"
    folds_path = output_dir / f"{stem}_folds.csv"
    errors_path = output_dir / f"{stem}_errors.csv"
    selections_path = output_dir / f"{stem}_selections.json"
    _write_csv(conditions_path, conditions)
    _write_csv(folds_path, first.folds)
    _write_csv(errors_path, _error_summary_rows(rows, labels, first))
    selections_path.write_text(
        json.dumps(first.selections, indent=2, sort_keys=True), encoding="utf-8"
    )
    total_wall_s = time.perf_counter() - started
    report = _report(
        conditions,
        first,
        gate_passed=gate_passed,
        deterministic=deterministic,
        cache_metadata=prepared.metadata,
        source_duration_s=source_duration_s,
        peak_rss_bytes=_peak_rss_bytes(),
        total_wall_s=total_wall_s,
    )
    report_path.write_text(report, encoding="utf-8")
    tracked_outputs = (
        report_path,
        conditions_path,
        folds_path,
        errors_path,
        selections_path,
    )
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "soundfile", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "command": sys.argv,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "source_sha256": _source_manifest(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "fixed_constants": {
            "seed": SEED,
            "development_players": DEV_PLAYERS,
            "sample_rate": SAMPLE_RATE,
            "window_samples": WINDOW_SAMPLES,
            "mel_bands": MEL_BANDS,
            "batch_size": BATCH_SIZE,
            "inner_epochs": INNER_EPOCHS,
            "final_epochs": FINAL_EPOCHS,
            "grid": GRID,
            "direct_prior_weight": DIRECT_PRIOR_WEIGHT,
            "best_previous_top1": BEST_PREVIOUS_TOP1,
            "gold_pitch_gate_delta": GOLD_PITCH_GATE_DELTA,
            "gold_pitch_gate_target": GOLD_PITCH_GATE_TARGET,
            "parameter_cap": 5_000_000,
            "model_parameters": model_parameters,
        },
        "notes_path": str(args.notes.resolve()),
        "notes_sha256": _file_sha256(args.notes.resolve()),
        "audio_manifest": audio_manifest,
        "audio_manifest_sha256": audio_manifest_sha256,
        "source_duration_s": source_duration_s,
        "dataset_license": "GuitarSet CC-BY-4.0; local evaluation data not redistributed",
        "excluded_data": ["Guitar-TECHS", "GOAT", "SynthTab", "GAPS", "private data"],
        "cache": prepared.metadata,
        "events": len(rows),
        "conditions": conditions,
        "folds": first.folds,
        "selections": first.selections,
        "prediction_sha256": first.prediction_sha256,
        "model_sha256": first.model_sha256,
        "deterministic_rerun": deterministic,
        "comparison_provenance": (
            str(args.compare_to_provenance.resolve())
            if args.compare_to_provenance is not None
            else None
        ),
        "gate_passed": gate_passed,
        "phase5_decision": "open_real_event_gate" if gate_passed else "close_direct_model",
        "player05_opened": False,
        "real_event_integration_run": False,
        "automatic_routing_changed": False,
        "shipping_artifact_bytes": 0,
        "added_automatic_inference_s": 0.0,
        "performance": {
            "feature_extraction_wall_s": prepared.metadata["extraction_wall_s"],
            "feature_extraction_seconds_per_60s": (
                float(prepared.metadata["extraction_wall_s"]) * 60.0 / source_duration_s
            ),
            "first_oof_wall_s": first.training_wall_s,
            "second_oof_wall_s": second.training_wall_s,
            "total_wall_s": total_wall_s,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "tracked_outputs": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in tracked_outputs
        },
    }
    provenance_path = output_dir / f"{stem}_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(report, flush=True)
    return 2 if gate_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
