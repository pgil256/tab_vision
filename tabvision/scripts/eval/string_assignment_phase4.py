"""Sequential Tab F1 Phase 4 native-rate adjacent-string signal probe.

This script uses only GuitarSet players 00--04. It consumes the frozen Phase 0
production-equivalent OOF event table, preserves each microphone WAV at its
native 44.1/48 kHz rate, and evaluates one fixed class-balanced L2-linear model
per physically adjacent string pair. Player 05 is never read by this probe.

Run from ``tabvision/``::

    python -m scripts.eval.string_assignment_phase4 \
        --notes ../docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv \
        --data-home ~/.tabvision/data/guitarset \
        --output-dir ../docs/EVAL_REPORTS
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
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

from tabvision.eval.adjacent_string_probe import (
    BalancedLogisticModel,
    adjacent_candidate_pairs,
    adjacent_gold_pairs,
    candidate_potentials,
    fit_balanced_logistic,
)
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.eval.high_frequency_string import (
    FEATURE_NAMES,
    SUPPORTED_SAMPLE_RATES,
    WINDOW_END_S,
    WINDOW_START_S,
    extract_high_frequency_features,
)
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import learn_pitch_position_prior
from tabvision.types import GuitarConfig, TabEvent

SEED = 4417
DEV_PLAYERS = ("00", "01", "02", "03", "04")
PROBE_SCHEMA_VERSION = 1
MAX_CANDIDATES = 6
LOGISTIC_L2 = 1.0
LOGISTIC_ITERATIONS = 50
AUDIO_EVIDENCE_WEIGHT = 1.0
BOOTSTRAP_RESAMPLES = 10_000
GATE_MIN_TOP1_DELTA = 0.05
GATE_MIN_WORST_FOLD_DELTA = -0.03

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NOTES = REPO_ROOT / "docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv"
DEFAULT_DATA_HOME = Path.home() / ".tabvision/data/guitarset"
DEFAULT_CACHE = Path.home() / ".tabvision/cache/string_assignment_phase4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/EVAL_REPORTS"


@dataclass(frozen=True)
class ProbeRow:
    event_id: str
    track_id: str
    player: str
    mode: str
    style: str
    onset_s: float
    pitch_midi: int
    reference_string: int
    reference_fret: int
    baseline_string: int
    baseline_correct: bool
    baseline_top3: bool
    candidate_strings: tuple[int, ...]
    candidate_frets: tuple[int, ...]


@dataclass(frozen=True)
class PairRows:
    event_indices: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ProbeRun:
    baseline_predictions: np.ndarray
    position_predictions: np.ndarray
    audio_predictions: np.ndarray
    combined_predictions: np.ndarray
    baseline_top3_hits: np.ndarray
    position_top3_hits: np.ndarray
    audio_top3_hits: np.ndarray
    combined_top3_hits: np.ndarray
    edge_logits: np.ndarray
    folds: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    model_payload: dict[str, Any]
    prediction_sha256: str
    model_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def _load_rows(notes_path: Path, cfg: GuitarConfig) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    with notes_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["condition"] != "production_equivalent":
                continue
            if raw["evaluation_split"] != "development_oof":
                continue
            if raw["ambiguous_pitch_match"] != "1" or raw["player"] not in DEV_PLAYERS:
                continue
            pitch = int(raw["pitch_midi"])
            candidates = candidate_positions(pitch, cfg)
            strings = tuple(candidate.string_idx for candidate in candidates)
            frets = tuple(candidate.fret for candidate in candidates)
            reference = (int(raw["reference_string"]), int(raw["reference_fret"]))
            if reference not in set(zip(strings, frets, strict=True)):
                raise RuntimeError(f"gold position is not playable for {raw['event_id']}")
            rows.append(
                ProbeRow(
                    event_id=raw["event_id"],
                    track_id=raw["track_id"],
                    player=raw["player"],
                    mode=raw["mode"],
                    style=raw["style"],
                    onset_s=float(raw["onset_s"]),
                    pitch_midi=pitch,
                    reference_string=reference[0],
                    reference_fret=reference[1],
                    baseline_string=int(raw["predicted_string"]),
                    baseline_correct=raw["candidate_top1"] == "1",
                    baseline_top3=raw["candidate_top3"] == "1",
                    candidate_strings=strings,
                    candidate_frets=frets,
                )
            )
    if len(rows) != 35_959:
        raise RuntimeError(f"expected 35,959 frozen development rows, found {len(rows)}")
    if len({row.event_id for row in rows}) != len(rows):
        raise RuntimeError("frozen Phase 0 event IDs are not unique")
    return rows


def _audio_manifest(rows: Sequence[ProbeRow], data_home: Path) -> tuple[list[dict[str, Any]], str]:
    manifest: list[dict[str, Any]] = []
    for track_id in sorted({row.track_id for row in rows}):
        path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        if not path.is_file():
            raise FileNotFoundError(f"missing GuitarSet microphone audio: {path}")
        info = sf.info(path)
        if info.samplerate not in SUPPORTED_SAMPLE_RATES:
            raise RuntimeError(f"unsupported native sample rate for {track_id}: {info.samplerate}")
        manifest.append(
            {
                "track_id": track_id,
                "bytes": path.stat().st_size,
                "sample_rate": info.samplerate,
                "frames": info.frames,
                "sha256": _file_sha256(path),
            }
        )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return manifest, hashlib.sha256(encoded).hexdigest()


def _feature_cache_metadata(
    rows: Sequence[ProbeRow],
    notes_path: Path,
    audio_manifest_sha256: str,
) -> dict[str, Any]:
    event_ids = "\n".join(row.event_id for row in rows).encode("utf-8")
    feature_module = REPO_ROOT / "tabvision/tabvision/eval/high_frequency_string.py"
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "notes_sha256": _file_sha256(notes_path),
        "event_ids_sha256": hashlib.sha256(event_ids).hexdigest(),
        "audio_manifest_sha256": audio_manifest_sha256,
        "feature_module_sha256": _file_sha256(feature_module),
        "rows": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "window_start_s": WINDOW_START_S,
        "window_end_s": WINDOW_END_S,
        "supported_sample_rates": list(SUPPORTED_SAMPLE_RATES),
    }


def _prepare_features(
    rows: Sequence[ProbeRow],
    notes_path: Path,
    data_home: Path,
    cache_dir: Path,
    audio_manifest_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "native_high_frequency_features_v1.npz"
    expected = _feature_cache_metadata(rows, notes_path, audio_manifest_sha256)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                actual = json.loads(str(payload["metadata"].item()))
                features = np.asarray(payload["features"], dtype=np.float32)
            if actual == expected and features.shape == (len(rows), len(FEATURE_NAMES)):
                if np.any(~np.isfinite(features)):
                    raise ValueError("cached features contain non-finite values")
                return features, {
                    "cache_hit": True,
                    "cache_path": str(cache_path),
                    "cache_bytes": cache_path.stat().st_size,
                    "cache_sha256": _file_sha256(cache_path),
                    "features_sha256": _array_sha256(features),
                    "extraction_wall_s": 0.0,
                    **expected,
                }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    features = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[row.track_id].append(index)
    started = time.perf_counter()
    for track_number, (track_id, indices) in enumerate(sorted(by_track.items()), start=1):
        audio_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        signal = np.asarray(waveform, dtype=np.float32)
        if signal.ndim == 2:
            signal = np.asarray(np.mean(signal, axis=1, dtype=np.float32), dtype=np.float32)
        if signal.ndim != 1:
            raise RuntimeError(f"unexpected audio shape for {track_id}: {signal.shape}")
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise RuntimeError(f"unsupported native sample rate for {track_id}: {sample_rate}")
        for index in indices:
            row = rows[index]
            features[index] = extract_high_frequency_features(
                signal,
                sample_rate,
                row.onset_s,
                row.pitch_midi,
            )
        if track_number % 10 == 0:
            print(f"  native features: {track_number}/{len(by_track)} tracks", flush=True)
    extraction_wall_s = time.perf_counter() - started
    if np.any(~np.isfinite(features)):
        raise RuntimeError("native feature extraction produced non-finite values")
    with tempfile.NamedTemporaryFile(
        prefix=f".{cache_path.name}.", suffix=".npz", dir=cache_dir, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            metadata=np.asarray(json.dumps(expected, sort_keys=True)),
            features=features,
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return features, {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": _file_sha256(cache_path),
        "features_sha256": _array_sha256(features),
        "extraction_wall_s": extraction_wall_s,
        **expected,
    }


def _candidate_arrays(
    rows: Sequence[ProbeRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strings = np.full((len(rows), MAX_CANDIDATES), -1, dtype=np.int8)
    frets = np.full((len(rows), MAX_CANDIDATES), -1, dtype=np.int8)
    labels = np.full(len(rows), -1, dtype=np.int8)
    for index, row in enumerate(rows):
        for column, (string_idx, fret) in enumerate(
            zip(row.candidate_strings, row.candidate_frets, strict=True)
        ):
            strings[index, column] = string_idx
            frets[index, column] = fret
            if (string_idx, fret) == (row.reference_string, row.reference_fret):
                labels[index] = column
        if labels[index] < 0:
            raise RuntimeError(f"missing gold candidate for {row.event_id}")
    return strings, frets, labels


def _gold_by_player(data_home: Path) -> dict[str, list[TabEvent]]:
    output: dict[str, list[TabEvent]] = {player: [] for player in DEV_PLAYERS}
    for player in DEV_PLAYERS:
        for path in sorted((data_home / "annotation").glob(f"{player}_*.jams")):
            output[player].extend(parse_guitarset_jams(path))
    if any(not values for values in output.values()):
        raise RuntimeError("one or more development players have no GuitarSet annotations")
    return output


def _oof_position_prior(
    rows: Sequence[ProbeRow],
    strings: np.ndarray,
    frets: np.ndarray,
    data_home: Path,
) -> np.ndarray:
    cfg = GuitarConfig()
    probabilities = np.zeros((len(rows), MAX_CANDIDATES), dtype=np.float64)
    gold = _gold_by_player(data_home)
    for player in DEV_PLAYERS:
        examples = [event for other, events in gold.items() if other != player for event in events]
        prior = learn_pitch_position_prior(examples, cfg=cfg, alpha=1.0, power=2.0)
        for index, row in enumerate(rows):
            if row.player != player:
                continue
            matrix = prior.matrix_for_pitch(row.pitch_midi)
            if matrix is None:
                raise RuntimeError(f"position prior missing pitch {row.pitch_midi}")
            for column in np.flatnonzero(strings[index] >= 0):
                probabilities[index, column] = matrix[
                    int(strings[index, column]), int(frets[index, column])
                ]
            total = float(np.sum(probabilities[index]))
            if total <= 0.0:
                raise RuntimeError(f"position prior has zero candidate mass for {row.event_id}")
            probabilities[index] /= total
    return probabilities


def _pair_rows(rows: Sequence[ProbeRow]) -> dict[int, PairRows]:
    event_indices: dict[int, list[int]] = {pair: [] for pair in range(5)}
    labels: dict[int, list[bool]] = {pair: [] for pair in range(5)}
    for index, row in enumerate(rows):
        for lower, higher, higher_is_gold in adjacent_gold_pairs(
            row.candidate_strings, row.reference_string
        ):
            if higher - lower != 1:
                raise AssertionError("non-adjacent hard negative escaped filtering")
            event_indices[lower].append(index)
            labels[lower].append(higher_is_gold)
    output = {
        pair: PairRows(
            np.asarray(event_indices[pair], dtype=np.int64),
            np.asarray(labels[pair], dtype=np.bool_),
        )
        for pair in range(5)
    }
    if not sum(len(value.labels) for value in output.values()):
        raise RuntimeError("no adjacent-string hard negatives were constructed")
    return output


def _column_for_string(
    strings: np.ndarray,
    event_indices: np.ndarray,
    string_idx: int,
) -> np.ndarray:
    selected = strings[event_indices] == string_idx
    if np.any(np.sum(selected, axis=1) != 1):
        raise RuntimeError("candidate string is missing or duplicated")
    return np.argmax(selected, axis=1)


def _model_payload(model: BalancedLogisticModel) -> dict[str, Any]:
    return {
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "weights": model.weights.tolist(),
        "sha256": model.sha256(),
    }


def _run_oof(
    rows: Sequence[ProbeRow],
    features: np.ndarray,
    strings: np.ndarray,
    labels: np.ndarray,
    position_prior: np.ndarray,
) -> ProbeRun:
    players = np.asarray([int(row.player) for row in rows], dtype=np.int8)
    hard_pairs = _pair_rows(rows)
    edge_logits = np.zeros((len(rows), 5), dtype=np.float64)
    pair_model_correct = {pair: 0 for pair in range(5)}
    pair_prior_correct = {pair: 0 for pair in range(5)}
    pair_total = {pair: 0 for pair in range(5)}
    pair_higher_gold = {pair: 0 for pair in range(5)}
    model_payload: dict[str, Any] = {}

    for held_player in map(int, DEV_PLAYERS):
        held_events = np.flatnonzero(players == held_player)
        fold_payload: dict[str, Any] = {}
        print(f"  fold {held_player:02d}: {len(held_events)} held events", flush=True)
        for pair in range(5):
            pair_data = hard_pairs[pair]
            train_mask = players[pair_data.event_indices] != held_player
            train_events = pair_data.event_indices[train_mask]
            train_labels = pair_data.labels[train_mask]
            model = fit_balanced_logistic(
                features[train_events],
                train_labels,
                l2=LOGISTIC_L2,
                iterations=LOGISTIC_ITERATIONS,
            )
            fold_payload[f"{pair}-{pair + 1}"] = _model_payload(model)

            has_lower = np.any(strings[held_events] == pair, axis=1)
            has_higher = np.any(strings[held_events] == pair + 1, axis=1)
            score_events = held_events[has_lower & has_higher]
            edge_logits[score_events, pair] = model.decision_function(features[score_events])

            pair_test_mask = players[pair_data.event_indices] == held_player
            test_events = pair_data.event_indices[pair_test_mask]
            test_labels = pair_data.labels[pair_test_mask]
            logits = model.decision_function(features[test_events])
            pair_model_correct[pair] += int(np.sum((logits >= 0.0) == test_labels))
            lower_columns = _column_for_string(strings, test_events, pair)
            higher_columns = _column_for_string(strings, test_events, pair + 1)
            prior_higher = (
                position_prior[test_events, higher_columns]
                > position_prior[test_events, lower_columns]
            )
            pair_prior_correct[pair] += int(np.sum(prior_higher == test_labels))
            pair_total[pair] += len(test_events)
            pair_higher_gold[pair] += int(np.sum(test_labels))
        model_payload[f"player_{held_player:02d}"] = fold_payload

    position_predictions = np.argmax(position_prior, axis=1).astype(np.int8)
    audio_predictions = np.zeros(len(rows), dtype=np.int8)
    combined_predictions = np.zeros(len(rows), dtype=np.int8)
    position_top3_hits = np.zeros(len(rows), dtype=np.bool_)
    audio_top3_hits = np.zeros(len(rows), dtype=np.bool_)
    combined_top3_hits = np.zeros(len(rows), dtype=np.bool_)
    for index, _row in enumerate(rows):
        valid_columns = np.flatnonzero(strings[index] >= 0)
        candidate_strings = [int(strings[index, column]) for column in valid_columns]
        edge_map = {
            pair: float(edge_logits[index, pair[0]])
            for pair in adjacent_candidate_pairs(candidate_strings)
        }
        potentials = candidate_potentials(candidate_strings, edge_map)
        audio_scores = np.asarray(
            [potentials[int(strings[index, column])] for column in valid_columns],
            dtype=np.float64,
        )
        prior_scores = np.log(np.maximum(position_prior[index, valid_columns], 1.0e-12))
        # The position prior is used only to make disconnected/tied audio
        # components deterministic. Its coefficient is too small to change a
        # non-tied audio decision.
        audio_predictions[index] = int(
            valid_columns[np.argmax(audio_scores + 1.0e-12 * prior_scores)]
        )
        combined_predictions[index] = int(
            valid_columns[np.argmax(prior_scores + AUDIO_EVIDENCE_WEIGHT * audio_scores)]
        )
        position_ranking = valid_columns[np.argsort(prior_scores)[::-1]]
        audio_ranking = valid_columns[np.argsort(audio_scores + 1.0e-12 * prior_scores)[::-1]]
        combined_ranking = valid_columns[
            np.argsort(prior_scores + AUDIO_EVIDENCE_WEIGHT * audio_scores)[::-1]
        ]
        position_top3_hits[index] = labels[index] in position_ranking[:3]
        audio_top3_hits[index] = labels[index] in audio_ranking[:3]
        combined_top3_hits[index] = labels[index] in combined_ranking[:3]

    baseline_predictions = np.asarray(
        [
            next(
                column
                for column, string_idx in enumerate(row.candidate_strings)
                if string_idx == row.baseline_string
            )
            for row in rows
        ],
        dtype=np.int8,
    )
    baseline_top3_hits = np.asarray([row.baseline_top3 for row in rows], dtype=np.bool_)
    folds: list[dict[str, Any]] = []
    for player in map(int, DEV_PLAYERS):
        selected = players == player
        baseline_accuracy = float(np.mean(baseline_predictions[selected] == labels[selected]))
        position_accuracy = float(np.mean(position_predictions[selected] == labels[selected]))
        audio_accuracy = float(np.mean(audio_predictions[selected] == labels[selected]))
        combined_accuracy = float(np.mean(combined_predictions[selected] == labels[selected]))
        folds.append(
            {
                "player": f"{player:02d}",
                "examples": int(np.sum(selected)),
                "baseline_accuracy": baseline_accuracy,
                "position_accuracy": position_accuracy,
                "audio_accuracy": audio_accuracy,
                "combined_accuracy": combined_accuracy,
                "combined_delta": combined_accuracy - baseline_accuracy,
            }
        )
    pairs = tuple(
        {
            "pair": f"{pair}-{pair + 1}",
            "examples": pair_total[pair],
            "higher_string_fraction": pair_higher_gold[pair] / pair_total[pair],
            "position_prior_accuracy": pair_prior_correct[pair] / pair_total[pair],
            "audio_pair_accuracy": pair_model_correct[pair] / pair_total[pair],
        }
        for pair in range(5)
    )
    prediction_digest = hashlib.sha256()
    for array in (
        baseline_predictions,
        position_predictions,
        audio_predictions,
        combined_predictions,
        baseline_top3_hits,
        position_top3_hits,
        audio_top3_hits,
        combined_top3_hits,
        edge_logits,
    ):
        prediction_digest.update(np.asarray(array).tobytes())
    model_encoded = json.dumps(model_payload, sort_keys=True, separators=(",", ":")).encode()
    return ProbeRun(
        baseline_predictions=baseline_predictions,
        position_predictions=position_predictions,
        audio_predictions=audio_predictions,
        combined_predictions=combined_predictions,
        baseline_top3_hits=baseline_top3_hits,
        position_top3_hits=position_top3_hits,
        audio_top3_hits=audio_top3_hits,
        combined_top3_hits=combined_top3_hits,
        edge_logits=edge_logits,
        folds=tuple(folds),
        pairs=pairs,
        model_payload=model_payload,
        prediction_sha256=prediction_digest.hexdigest(),
        model_sha256=hashlib.sha256(model_encoded).hexdigest(),
    )


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def _clip_bootstrap(
    rows: Sequence[ProbeRow],
    baseline_correct: np.ndarray,
    combined_correct: np.ndarray,
) -> tuple[float, float]:
    tracks: dict[str, tuple[int, int, int, str]] = {}
    for track_id in sorted({row.track_id for row in rows}):
        selected = np.asarray([row.track_id == track_id for row in rows])
        first = next(row for row in rows if row.track_id == track_id)
        tracks[track_id] = (
            int(np.sum(baseline_correct[selected])),
            int(np.sum(combined_correct[selected])),
            int(np.sum(selected)),
            f"{first.player}|{first.mode}",
        )
    strata: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for baseline, combined, total, stratum in tracks.values():
        strata[stratum].append((baseline, combined, total))
    rng = np.random.default_rng(SEED)
    deltas = np.zeros(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_RESAMPLES):
        baseline_sum = combined_sum = total_sum = 0
        for values in strata.values():
            indices = rng.integers(0, len(values), size=len(values))
            for index in indices:
                baseline, combined, total = values[int(index)]
                baseline_sum += baseline
                combined_sum += combined
                total_sum += total
        deltas[sample] = (combined_sum - baseline_sum) / total_sum
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _condition_rows(run: ProbeRun, labels: np.ndarray) -> list[dict[str, Any]]:
    baseline = _accuracy(run.baseline_predictions, labels)
    return [
        {
            "condition": name,
            "ambiguous_top1": _accuracy(predictions, labels),
            "ambiguous_top3": float(np.mean(top3_hits)),
            "same_pitch_wrong_rate": 1.0 - _accuracy(predictions, labels),
            "delta_vs_production": _accuracy(predictions, labels) - baseline,
        }
        for name, predictions, top3_hits in (
            ("production_prior", run.baseline_predictions, run.baseline_top3_hits),
            ("oof_position_prior", run.position_predictions, run.position_top3_hits),
            ("native_audio_only", run.audio_predictions, run.audio_top3_hits),
            (
                "oof_position_plus_native_audio",
                run.combined_predictions,
                run.combined_top3_hits,
            ),
        )
    ]


def _error_summary_rows(
    rows: Sequence[ProbeRow],
    strings: np.ndarray,
    labels: np.ndarray,
    run: ProbeRun,
) -> list[dict[str, Any]]:
    dimensions = (
        "player",
        "mode",
        "style",
        "pitch_midi",
        "candidate_count",
        "reference_string",
        "predicted_minus_reference_string",
    )
    output: list[dict[str, Any]] = []
    for condition, predictions in (
        ("production_prior", run.baseline_predictions),
        ("oof_position_plus_native_audio", run.combined_predictions),
    ):
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for index, row in enumerate(rows):
            if predictions[index] == labels[index]:
                continue
            predicted_string = int(strings[index, predictions[index]])
            dimension_values: dict[str, object] = {
                "player": row.player,
                "mode": row.mode,
                "style": row.style,
                "pitch_midi": row.pitch_midi,
                "candidate_count": len(row.candidate_strings),
                "reference_string": row.reference_string,
                "predicted_minus_reference_string": predicted_string - row.reference_string,
            }
            for dimension in dimensions:
                counts[(condition, dimension)][str(dimension_values[dimension])] += 1
        for (name, dimension), counter_values in sorted(counts.items()):
            for value, count in sorted(counter_values.items()):
                output.append(
                    {
                        "condition": name,
                        "dimension": dimension,
                        "value": value,
                        "count": count,
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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
    run: ProbeRun,
    ci_lower: float,
    ci_upper: float,
    gate_passed: bool,
    cache_metadata: Mapping[str, Any],
    evaluation_wall_s: float,
    deterministic: bool,
    peak_rss_bytes: int,
    source_duration_s: float,
) -> str:
    condition_lines = [
        f"| `{row['condition']}` | {row['ambiguous_top1']:.4f} | "
        f"{row['ambiguous_top3']:.4f} | {row['same_pitch_wrong_rate']:.4f} | "
        f"{row['delta_vs_production']:+.4f} |"
        for row in conditions
    ]
    fold_lines = [
        f"| {row['player']} | {row['examples']} | {row['baseline_accuracy']:.4f} | "
        f"{row['position_accuracy']:.4f} | {row['audio_accuracy']:.4f} | "
        f"{row['combined_accuracy']:.4f} | {row['combined_delta']:+.4f} |"
        for row in run.folds
    ]
    pair_lines = [
        f"| {row['pair']} | {row['examples']} | {row['higher_string_fraction']:.4f} | "
        f"{row['position_prior_accuracy']:.4f} | {row['audio_pair_accuracy']:.4f} |"
        for row in run.pairs
    ]
    pair_examples = sum(int(row["examples"]) for row in run.pairs)
    combined = next(
        row for row in conditions if row["condition"] == "oof_position_plus_native_audio"
    )
    worst = min(float(row["combined_delta"]) for row in run.folds)
    decision = (
        "PASS: prepare the compact-model cost/license packet; do not start GPU training "
        "without explicit approval."
        if gate_passed
        else "FAIL: close the compact high-frequency timbral path; do not enlarge the model "
        "or open player 05."
    )
    return "\n".join(
        [
            "# Sequential Tab F1 Phase 4: native-rate adjacent-string probe",
            "",
            "## Fixed method",
            "",
            "- Data: GuitarSet microphone WAV and hex-derived per-string JAMS labels; players "
            "00-04 only. Player 05 was not read.",
            "- Examples: 35,959 frozen production-equivalent OOF pitch-correct ambiguous "
            f"events and {pair_examples:,} physically adjacent gold-vs-alternative pairs.",
            "- Audio: native 44.1 kHz in this corpus (44.1/48 kHz accepted); no 16 kHz "
            "backend waveform and no upsampling.",
            "- Window: 64 ms pre-onset plus 448 ms post-onset. Attack, short-sustain, and "
            "long-sustain spectra use fixed 4096/8192/16384 FFTs.",
            "- Features: harmonic-envelope bands through Nyquist, spectral centroid and "
            "85/95% rolloff, 6-18 kHz pick energy/flux, harmonic decay, inharmonicity, "
            "plus separately retained raw RMS and dB/octave spectral slope.",
            "- Model: five separate class-balanced L2-linear logistic models, one per "
            "adjacent string pair; L2=1.0, 50 deterministic Newton steps.",
            "- Fusion: pair log-odds integrate into candidate potentials and are added to "
            "the player-held OOF position prior at the fixed weight 1.0. No grid, "
            "temperature, threshold, or held-player selection.",
            "",
            "## OOF ambiguous-note result",
            "",
            "| condition | ambiguous top-1 | top-3 | wrong rate | delta vs production |",
            "|---|---:|---:|---:|---:|",
            *condition_lines,
            "",
            f"Paired clip-stratified 10,000-resample interval for the combined delta: "
            f"`[{ci_lower:+.4f}, {ci_upper:+.4f}]`.",
            "",
            "### Player folds",
            "",
            "| held player | examples | production | position | audio only | combined | delta |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *fold_lines,
            "",
            "### Adjacent hard-negative diagnostic",
            "",
            "| string pair | pairs | higher-string gold | position pair acc | audio pair acc |",
            "|---|---:|---:|---:|---:|",
            *pair_lines,
            "",
            "## Gate decision",
            "",
            f"- Required aggregate lift: `>= +{GATE_MIN_TOP1_DELTA:.2f}`; observed "
            f"`{float(combined['delta_vs_production']):+.4f}`.",
            f"- Required worst fold: `>= {GATE_MIN_WORST_FOLD_DELTA:+.2f}`; observed "
            f"`{worst:+.4f}`.",
            f"- Free-signal gate: **{'PASS' if gate_passed else 'FAIL'}**.",
            f"- Decision: **{decision}**",
            "",
            "## Runtime, reproducibility, and routing safety",
            "",
            f"- Feature cache: `{cache_metadata['cache_bytes']}` bytes; extraction wall "
            f"`{cache_metadata['extraction_wall_s']:.3f} s`; cache hit "
            f"`{str(cache_metadata['cache_hit']).lower()}`.",
            f"- Feature cache SHA-256 `{cache_metadata['cache_sha256']}`; descriptor-array "
            f"SHA-256 `{cache_metadata['features_sha256']}`.",
            f"- Descriptor extraction rate: "
            f"`{float(cache_metadata['extraction_wall_s']) * 60.0 / source_duration_s:.3f} s` "
            "per 60 seconds of source audio on the first uncached run.",
            f"- Two complete OOF fits/evaluations: `{evaluation_wall_s:.3f} s`; peak process "
            f"working set `{peak_rss_bytes}` bytes.",
            f"- Deterministic rerun: **{'PASS' if deterministic else 'FAIL'}**; prediction "
            f"SHA-256 `{run.prediction_sha256}`; model SHA-256 `{run.model_sha256}`.",
            "- Shipping artifact size and added automatic inference time are both zero: "
            "the failed probe is not registered or integrated. Onset and pitch events are "
            "unchanged. "
            "Automatic clean-acoustic, GAPS classical, Guitar-TECHS electric, capo, "
            "alternate-tuning, distorted, and non-high-resolution routes are unchanged; "
            "their Phase 4 metric/event delta is exactly zero.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "cd tabvision",
            ".\\.venv\\Scripts\\python.exe -m scripts.eval.string_assignment_phase4",
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = GuitarConfig()
    rows = _load_rows(args.notes.resolve(), cfg)
    print("hashing 300 development microphone files...", flush=True)
    audio_manifest, audio_manifest_sha256 = _audio_manifest(rows, args.data_home.resolve())
    source_duration_s = sum(
        float(row["frames"]) / float(row["sample_rate"]) for row in audio_manifest
    )
    features, cache_metadata = _prepare_features(
        rows,
        args.notes.resolve(),
        args.data_home.resolve(),
        args.cache_dir.resolve(),
        audio_manifest_sha256,
    )
    strings, frets, labels = _candidate_arrays(rows)
    frozen_baseline = np.asarray([row.baseline_correct for row in rows], dtype=np.bool_)
    baseline_from_candidates = np.asarray(
        [
            row.candidate_strings[int(label)] == row.baseline_string
            for row, label in zip(rows, labels, strict=True)
        ],
        dtype=np.bool_,
    )
    if not np.array_equal(frozen_baseline, baseline_from_candidates):
        raise RuntimeError("frozen candidate-top1 labels do not match candidate reconstruction")
    position_prior = _oof_position_prior(rows, strings, frets, args.data_home.resolve())
    evaluation_started = time.perf_counter()
    print("running first fixed five-player OOF evaluation...", flush=True)
    first = _run_oof(rows, features, strings, labels, position_prior)
    print("running deterministic OOF rerun...", flush=True)
    second = _run_oof(rows, features, strings, labels, position_prior)
    evaluation_wall_s = time.perf_counter() - evaluation_started
    deterministic = (
        first.prediction_sha256 == second.prediction_sha256
        and first.model_sha256 == second.model_sha256
    )
    if not deterministic:
        raise RuntimeError("Phase 4 OOF evaluation is not deterministic")

    conditions = _condition_rows(first, labels)
    baseline_correct = first.baseline_predictions == labels
    combined_correct = first.combined_predictions == labels
    ci_lower, ci_upper = _clip_bootstrap(rows, baseline_correct, combined_correct)
    combined = next(
        row for row in conditions if row["condition"] == "oof_position_plus_native_audio"
    )
    worst_fold = min(float(row["combined_delta"]) for row in first.folds)
    gate_passed = (
        float(combined["delta_vs_production"]) >= GATE_MIN_TOP1_DELTA
        and worst_fold >= GATE_MIN_WORST_FOLD_DELTA
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "string_assignment_phase4_2026-07-16"
    conditions_path = output_dir / f"{stem}_conditions.csv"
    folds_path = output_dir / f"{stem}_folds.csv"
    pairs_path = output_dir / f"{stem}_pairs.csv"
    errors_path = output_dir / f"{stem}_errors.csv"
    report_path = output_dir / f"{stem}.md"
    _write_csv(conditions_path, conditions)
    _write_csv(folds_path, first.folds)
    _write_csv(pairs_path, first.pairs)
    _write_csv(errors_path, _error_summary_rows(rows, strings, labels, first))
    report = _report(
        conditions,
        first,
        ci_lower,
        ci_upper,
        gate_passed,
        cache_metadata,
        evaluation_wall_s,
        deterministic,
        _peak_rss_bytes(),
        source_duration_s,
    )
    report_path.write_text(report, encoding="utf-8")

    tracked_outputs = (report_path, conditions_path, folds_path, pairs_path, errors_path)
    provenance = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "command": sys.argv,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "source_sha256": {
            "probe_script": _file_sha256(Path(__file__).resolve()),
            "feature_module": _file_sha256(
                REPO_ROOT / "tabvision/tabvision/eval/high_frequency_string.py"
            ),
            "pairwise_module": _file_sha256(
                REPO_ROOT / "tabvision/tabvision/eval/adjacent_string_probe.py"
            ),
        },
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "soundfile")
        },
        "fixed_constants": {
            "seed": SEED,
            "development_players": DEV_PLAYERS,
            "logistic_l2": LOGISTIC_L2,
            "logistic_iterations": LOGISTIC_ITERATIONS,
            "audio_evidence_weight": AUDIO_EVIDENCE_WEIGHT,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "gate_min_top1_delta": GATE_MIN_TOP1_DELTA,
            "gate_min_worst_fold_delta": GATE_MIN_WORST_FOLD_DELTA,
            "window_start_s": WINDOW_START_S,
            "window_end_s": WINDOW_END_S,
        },
        "feature_names": FEATURE_NAMES,
        "notes_path": str(args.notes.resolve()),
        "notes_sha256": _file_sha256(args.notes.resolve()),
        "audio_manifest": audio_manifest,
        "audio_manifest_sha256": audio_manifest_sha256,
        "source_duration_s": source_duration_s,
        "dataset_license": "GuitarSet CC-BY-4.0; local evaluation data not redistributed",
        "cache": cache_metadata,
        "events": len(rows),
        "pair_examples": sum(int(row["examples"]) for row in first.pairs),
        "conditions": conditions,
        "folds": first.folds,
        "pairs": first.pairs,
        "paired_ci": {"lower": ci_lower, "upper": ci_upper},
        "prediction_sha256": first.prediction_sha256,
        "model_sha256": first.model_sha256,
        "model_payload": first.model_payload,
        "deterministic_rerun": deterministic,
        "gate_passed": gate_passed,
        "phase4_decision": "cost_license_gate" if gate_passed else "close_compact_timbral_path",
        "player05_opened": False,
        "automatic_routing_changed": False,
        "shipping_artifact_bytes": 0,
        "added_automatic_inference_s": 0.0,
        "performance": {
            "feature_extraction_wall_s": cache_metadata["extraction_wall_s"],
            "feature_extraction_seconds_per_60s": (
                float(cache_metadata["extraction_wall_s"]) * 60.0 / source_duration_s
            ),
            "evaluation_wall_s": evaluation_wall_s,
            "total_wall_s": time.perf_counter() - started,
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
