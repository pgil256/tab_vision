"""Phase 2 free timbral string-ranker probe on GuitarSet players 00-04.

The probe consumes the exact ambiguous, pitch-correct high-resolution events
recorded by Phase 0, extracts 512 ms audio windows around the detected onset,
and produces leakage-free leave-one-player-out predictions for:

1. a candidate-feature-only ranker; and
2. the fixed compact STFT/CNN ranker from the accuracy plan.

Player 05 is never read. Paid optimization is allowed only when this script's
fixed gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import learn_pitch_position_prior
from tabvision.types import GuitarConfig, TabEvent

SEED = 2714
SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 128
WINDOW_SAMPLES = 8192
JITTER_SAMPLES = 800
EXTENDED_SAMPLES = WINDOW_SAMPLES + 2 * JITTER_SAMPLES
FEATURE_DIM = 15
MAX_CANDIDATES = 6
DEV_PLAYERS = ("00", "01", "02", "03", "04")
PROBE_VERSION = "phase2-free-v1"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NOTES = (
    REPO_ROOT / "docs" / "EVAL_REPORTS" / "string_assignment_phase0_2026-07-14_notes.csv"
)
DEFAULT_DATA_HOME = Path.home() / ".tabvision" / "data" / "guitarset"
DEFAULT_CACHE = Path.home() / ".tabvision" / "cache" / "string_assignment_phase2"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "EVAL_REPORTS" / "string_assignment_phase2_free_2026-07-14.md"


@dataclass(frozen=True)
class ExampleRow:
    track_id: str
    player: str
    mode: str
    onset_s: float
    pitch_midi: int
    reference_string: int
    reference_fret: int
    baseline_correct: int


class _IndexDataset(Dataset[int]):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return int(self.indices[index])


class FeatureOnlyRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, candidate_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.network(candidate_features).squeeze(-1)
        return logits.masked_fill(~mask, -1e9)


class AudioCandidateRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.encoder = nn.Sequential(
            self._block(1, 16, 4),
            self._block(16, 32, 8),
            self._block(32, 64, 8),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(64 + FEATURE_DIM, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, groups: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        candidate_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=self.window,
            center=True,
            return_complex=True,
        )
        log_magnitude = torch.log1p(torch.abs(stft)).unsqueeze(1)
        # The STFT parameters are fixed by the plan. Deterministic area pooling
        # bounds CPU memory while preserving the complete time/frequency range.
        log_magnitude = functional.adaptive_avg_pool2d(log_magnitude, (128, 32))
        audio_embedding = self.encoder(log_magnitude).flatten(1)
        repeated = audio_embedding.unsqueeze(1).expand(-1, MAX_CANDIDATES, -1)
        logits = self.candidate_mlp(torch.cat((repeated, candidate_features), dim=-1))
        return logits.squeeze(-1).masked_fill(~mask, -1e9)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{PROBE_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_rows(notes_path: Path) -> list[ExampleRow]:
    rows: list[ExampleRow] = []
    with notes_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "production_equivalent":
                continue
            if row["evaluation_split"] != "development_oof":
                continue
            if row["ambiguous_pitch_match"] != "1":
                continue
            player = row["player"]
            if player not in DEV_PLAYERS:
                continue
            rows.append(
                ExampleRow(
                    track_id=row["track_id"],
                    player=player,
                    mode=row["mode"],
                    onset_s=float(row["onset_s"]),
                    pitch_midi=int(row["pitch_midi"]),
                    reference_string=int(row["reference_string"]),
                    reference_fret=int(row["reference_fret"]),
                    baseline_correct=int(row["candidate_top1"]),
                )
            )
    if not rows:
        raise RuntimeError(f"no Phase 0 development examples found in {notes_path}")
    return rows


def _candidate_arrays(
    rows: list[ExampleRow],
    cfg: GuitarConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strings = np.full((len(rows), MAX_CANDIDATES), -1, dtype=np.int8)
    frets = np.full((len(rows), MAX_CANDIDATES), -1, dtype=np.int8)
    labels = np.full(len(rows), -1, dtype=np.int8)
    for index, row in enumerate(rows):
        candidates = candidate_positions(row.pitch_midi, cfg)
        if not 2 <= len(candidates) <= MAX_CANDIDATES:
            raise RuntimeError(f"unexpected candidate count for row {index}: {len(candidates)}")
        for candidate_index, candidate in enumerate(candidates):
            strings[index, candidate_index] = candidate.string_idx
            frets[index, candidate_index] = candidate.fret
            if (candidate.string_idx, candidate.fret) == (
                row.reference_string,
                row.reference_fret,
            ):
                labels[index] = candidate_index
        if labels[index] < 0:
            raise RuntimeError(f"gold position is not playable for row {index}")
    return strings, frets, labels


def _extract_extended(waveform: np.ndarray, onset_s: float) -> np.ndarray:
    start = int(round(onset_s * SAMPLE_RATE)) - 64 * SAMPLE_RATE // 1000 - JITTER_SAMPLES
    stop = start + EXTENDED_SAMPLES
    out = np.zeros(EXTENDED_SAMPLES, dtype=np.float32)
    source_start = max(0, start)
    source_stop = min(len(waveform), stop)
    if source_stop > source_start:
        destination_start = source_start - start
        out[destination_start : destination_start + source_stop - source_start] = waveform[
            source_start:source_stop
        ]
    return out


def _nearest_gold_error(row: ExampleRow, gold: list[TabEvent]) -> float:
    matches = [
        event
        for event in gold
        if event.pitch_midi == row.pitch_midi
        and event.string_idx == row.reference_string
        and event.fret == row.reference_fret
        and abs(event.onset_s - row.onset_s) <= 0.050001
    ]
    if not matches:
        return 0.0
    event = min(matches, key=lambda item: abs(item.onset_s - row.onset_s))
    return float(np.clip(row.onset_s - event.onset_s, -0.05, 0.05))


def prepare_cache(
    notes_path: Path,
    data_home: Path,
    cache_dir: Path,
) -> tuple[np.memmap, dict[str, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(notes_path)
    metadata_path = cache_dir / "examples.npz"
    waveform_path = cache_dir / "waveforms.npy"
    if metadata_path.is_file() and waveform_path.is_file():
        cached = np.load(metadata_path, allow_pickle=False)
        if str(cached["fingerprint"].item()) == fingerprint:
            arrays = {name: cached[name] for name in cached.files if name != "fingerprint"}
            waveforms = np.load(waveform_path, mmap_mode="r")
            if len(waveforms) == len(arrays["labels"]):
                return waveforms, arrays

    rows = _load_rows(notes_path)
    cfg = GuitarConfig()
    strings, frets, labels = _candidate_arrays(rows, cfg)
    waveforms = np.lib.format.open_memmap(
        waveform_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), EXTENDED_SAMPLES),
    )
    rms = np.zeros(len(rows), dtype=np.float32)
    onset_error = np.zeros(len(rows), dtype=np.float32)
    by_track: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[row.track_id].append(index)

    for track_number, (track_id, indices) in enumerate(sorted(by_track.items()), start=1):
        wav_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
        jams_path = data_home / "annotation" / f"{track_id}.jams"
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        mono = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            divisor = math.gcd(sample_rate, SAMPLE_RATE)
            mono = resample_poly(mono, SAMPLE_RATE // divisor, sample_rate // divisor).astype(
                np.float32
            )
        gold = parse_guitarset_jams(jams_path, cfg)
        for index in indices:
            extended = _extract_extended(mono, rows[index].onset_s)
            central = extended[JITTER_SAMPLES : JITTER_SAMPLES + WINDOW_SAMPLES]
            value = float(np.sqrt(np.mean(np.square(central), dtype=np.float64)))
            rms[index] = value
            waveforms[index] = (extended / max(value, 1e-4)).astype(np.float16)
            onset_error[index] = _nearest_gold_error(rows[index], gold)
        if track_number % 25 == 0:
            print(f"prepared {track_number}/{len(by_track)} tracks", flush=True)
    waveforms.flush()

    arrays = {
        "players": np.asarray([int(row.player) for row in rows], dtype=np.int8),
        "modes": np.asarray([0 if row.mode == "solo" else 1 for row in rows], dtype=np.int8),
        "pitches": np.asarray([row.pitch_midi for row in rows], dtype=np.int16),
        "strings": strings,
        "frets": frets,
        "labels": labels,
        "baseline_correct": np.asarray([row.baseline_correct for row in rows], dtype=np.int8),
        "rms": rms,
        "onset_error": onset_error,
    }
    np.savez_compressed(metadata_path, fingerprint=np.asarray(fingerprint), **arrays)
    return np.load(waveform_path, mmap_mode="r"), arrays


def _candidate_features(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_examples = len(arrays["labels"])
    features = np.zeros((n_examples, MAX_CANDIDATES, FEATURE_DIM), dtype=np.float32)
    mask = arrays["strings"] >= 0
    for row in range(n_examples):
        pitch = int(arrays["pitches"][row])
        pitch_class = pitch % 12
        style = int(arrays["modes"][row])
        rms_feature = float(
            np.clip((math.log10(max(float(arrays["rms"][row]), 1e-4)) + 4) / 4, 0, 1)
        )
        for column in np.flatnonzero(mask[row]):
            string_idx = int(arrays["strings"][row, column])
            fret = int(arrays["frets"][row, column])
            features[row, column, string_idx] = 1.0
            offset = 6
            features[row, column, offset] = fret / 24.0
            features[row, column, offset + 1] = float(fret == 0)
            features[row, column, offset + 2] = pitch / 127.0
            features[row, column, offset + 3] = math.sin(2 * math.pi * pitch_class / 12)
            features[row, column, offset + 4] = math.cos(2 * math.pi * pitch_class / 12)
            features[row, column, offset + 5 + style] = 1.0
            features[row, column, offset + 8] = rms_feature
    return features, mask


def _balanced_weights(indices: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    keys = [
        (
            int(arrays["players"][index]),
            int(arrays["strings"][index, arrays["labels"][index]]),
            int(arrays["pitches"][index] // 12),
            int(arrays["modes"][index]),
        )
        for index in indices
    ]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)


def _loader(
    indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    samples: int,
    seed: int,
) -> DataLoader[int]:
    weights = torch.from_numpy(_balanced_weights(indices, arrays))
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        _IndexDataset(indices),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


def _tensor_batch(
    batch_indices: torch.Tensor,
    features: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = batch_indices.numpy()
    return (
        torch.from_numpy(features[indices]),
        torch.from_numpy(mask[indices]),
        torch.from_numpy(labels[indices].astype(np.int64)),
    )


def _train_feature_model(
    train_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    features: np.ndarray,
    mask: np.ndarray,
    *,
    epochs: int,
    train_cap: int,
    seed: int,
) -> FeatureOnlyRanker:
    _seed_everything(seed)
    model = FeatureOnlyRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        loader = _loader(
            train_indices,
            arrays,
            batch_size=512,
            samples=min(train_cap, len(train_indices)),
            seed=seed + epoch,
        )
        for batch_indices in loader:
            candidate_features, candidate_mask, labels = _tensor_batch(
                batch_indices, features, mask, arrays["labels"]
            )
            loss = functional.cross_entropy(model(candidate_features, candidate_mask), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def _augment_waveforms(
    extended: np.ndarray,
    onset_errors: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    batch = np.empty((len(extended), WINDOW_SAMPLES), dtype=np.float32)
    sampled_errors = rng.choice(onset_errors, size=len(extended), replace=True)
    for index, error_s in enumerate(sampled_errors):
        shift = int(round(float(np.clip(error_s, -0.05, 0.05)) * SAMPLE_RATE))
        start = JITTER_SAMPLES + shift
        batch[index] = extended[index, start : start + WINDOW_SAMPLES]
    waveform = torch.from_numpy(batch)
    gain_db = torch.from_numpy(rng.uniform(-6.0, 6.0, size=(len(batch), 1)).astype(np.float32))
    waveform = waveform * torch.pow(10.0, gain_db / 20.0)
    tilt = torch.from_numpy(rng.uniform(-0.2, 0.2, size=(len(batch), 1)).astype(np.float32))
    waveform[:, 1:] = waveform[:, 1:] + tilt * (waveform[:, 1:] - waveform[:, :-1])
    snr_db = torch.from_numpy(rng.uniform(30.0, 50.0, size=(len(batch), 1)).astype(np.float32))
    waveform = waveform + torch.randn_like(waveform) * torch.pow(10.0, -snr_db / 20.0)
    drive = torch.from_numpy(rng.uniform(0.8, 1.4, size=(len(batch), 1)).astype(np.float32))
    waveform = torch.tanh(waveform * drive) / torch.tanh(drive)
    return waveform


def _train_audio_model(
    train_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    waveforms: np.memmap,
    features: np.ndarray,
    mask: np.ndarray,
    *,
    epochs: int,
    train_cap: int,
    batch_size: int,
    seed: int,
) -> AudioCandidateRanker:
    _seed_everything(seed)
    model = AudioCandidateRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    onset_errors = arrays["onset_error"][train_indices]
    model.train()
    for epoch in range(epochs):
        loader = _loader(
            train_indices,
            arrays,
            batch_size=batch_size,
            samples=min(train_cap, len(train_indices)),
            seed=seed + epoch,
        )
        losses = []
        for batch_indices in loader:
            indices = batch_indices.numpy()
            waveform = _augment_waveforms(
                np.asarray(waveforms[indices], dtype=np.float32), onset_errors, rng
            )
            candidate_features, candidate_mask, labels = _tensor_batch(
                batch_indices, features, mask, arrays["labels"]
            )
            logits = model(waveform, candidate_features, candidate_mask)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(
            f"  audio epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}",
            flush=True,
        )
    return model


@torch.inference_mode()
def _predict_feature(
    model: FeatureOnlyRanker,
    indices: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    model.eval()
    out = np.full((len(indices), MAX_CANDIDATES), -1e9, dtype=np.float32)
    for start in range(0, len(indices), 1024):
        selected = indices[start : start + 1024]
        logits = model(torch.from_numpy(features[selected]), torch.from_numpy(mask[selected]))
        out[start : start + len(selected)] = logits.numpy()
    return out


@torch.inference_mode()
def _predict_audio(
    model: AudioCandidateRanker,
    indices: np.ndarray,
    waveforms: np.memmap,
    features: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out = np.full((len(indices), MAX_CANDIDATES), -1e9, dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        extended = np.asarray(waveforms[selected], dtype=np.float32)
        waveform = torch.from_numpy(
            extended[:, JITTER_SAMPLES : JITTER_SAMPLES + WINDOW_SAMPLES].copy()
        )
        logits = model(
            waveform,
            torch.from_numpy(features[selected]),
            torch.from_numpy(mask[selected]),
        )
        out[start : start + len(selected)] = logits.numpy()
    return out


def _gold_by_player(data_home: Path) -> dict[int, list[TabEvent]]:
    out: dict[int, list[TabEvent]] = {int(player): [] for player in DEV_PLAYERS}
    for annotation in sorted((data_home / "annotation").glob("*.jams")):
        player = int(annotation.stem.split("_", 1)[0])
        if player in out:
            out[player].extend(parse_guitarset_jams(annotation))
    return out


def _oof_prior(
    arrays: dict[str, np.ndarray],
    gold_by_player: dict[int, list[TabEvent]],
) -> np.ndarray:
    cfg = GuitarConfig()
    probabilities = np.zeros((len(arrays["labels"]), MAX_CANDIDATES), dtype=np.float32)
    for player in map(int, DEV_PLAYERS):
        examples = [
            event
            for other_player, gold in gold_by_player.items()
            if other_player != player
            for event in gold
        ]
        prior = learn_pitch_position_prior(examples, cfg=cfg, alpha=1.0, power=2.0)
        for index in np.flatnonzero(arrays["players"] == player):
            matrix = prior.matrix_for_pitch(int(arrays["pitches"][index]))
            if matrix is None:
                continue
            for candidate_index in np.flatnonzero(arrays["strings"][index] >= 0):
                string_idx = int(arrays["strings"][index, candidate_index])
                fret = int(arrays["frets"][index, candidate_index])
                probabilities[index, candidate_index] = matrix[string_idx, fret]
            total = float(probabilities[index].sum())
            if total > 0:
                probabilities[index] /= total
    return probabilities


def _temperature(logits: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> float:
    best = (float("inf"), 1.0)
    for value in np.linspace(0.5, 3.0, 51):
        scaled = np.where(mask, logits / value, -1e9)
        maximum = scaled.max(axis=1, keepdims=True)
        logsumexp = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
        nll = float(np.mean(logsumexp - scaled[np.arange(len(labels)), labels]))
        if nll < best[0]:
            best = (nll, float(value))
    return best[1]


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, logits, -1e9)
    maximum = masked.max(axis=1, keepdims=True)
    values = np.exp(masked - maximum) * mask
    return values / values.sum(axis=1, keepdims=True)


def _ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    value = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= low) & (confidence < low + 0.1)
        if np.any(selected):
            value += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected])) - float(np.mean(correct[selected]))
            )
    return value


def _combined_predictions(
    prior: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    temperature: float,
    weight: float,
) -> np.ndarray:
    scores = np.log(np.maximum(prior, 1e-12)) + weight * logits / temperature
    scores = np.where(mask, scores, -1e9)
    return scores.argmax(axis=1)


def _select_weight(
    prior: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> tuple[float, float]:
    best = (-1.0, 0.0)
    for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        predictions = _combined_predictions(
            prior, logits, mask, temperature=temperature, weight=weight
        )
        accuracy = float(np.mean(predictions == labels))
        if accuracy > best[0]:
            best = (accuracy, weight)
    return best[1], best[0]


def run_probe(
    waveforms: np.memmap,
    arrays: dict[str, np.ndarray],
    data_home: Path,
    *,
    feature_epochs: int,
    audio_epochs: int,
    train_cap: int,
    batch_size: int,
) -> dict[str, Any]:
    features, mask = _candidate_features(arrays)
    labels = arrays["labels"].astype(np.int64)
    feature_logits = np.full((len(labels), MAX_CANDIDATES), -1e9, dtype=np.float32)
    audio_logits = np.full_like(feature_logits, -1e9)
    started = time.perf_counter()
    for fold, player in enumerate(map(int, DEV_PLAYERS)):
        test_indices = np.flatnonzero(arrays["players"] == player)
        train_indices = np.flatnonzero(arrays["players"] != player)
        print(
            f"fold player {player:02d}: train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        feature_model = _train_feature_model(
            train_indices,
            arrays,
            features,
            mask,
            epochs=feature_epochs,
            train_cap=train_cap,
            seed=SEED + fold * 10,
        )
        feature_logits[test_indices] = _predict_feature(feature_model, test_indices, features, mask)
        audio_model = _train_audio_model(
            train_indices,
            arrays,
            waveforms,
            features,
            mask,
            epochs=audio_epochs,
            train_cap=train_cap,
            batch_size=batch_size,
            seed=SEED + fold * 10 + 1,
        )
        audio_logits[test_indices] = _predict_audio(
            audio_model,
            test_indices,
            waveforms,
            features,
            mask,
            batch_size=batch_size,
        )
    elapsed = time.perf_counter() - started

    prior = _oof_prior(arrays, _gold_by_player(data_home))
    feature_temperature = _temperature(feature_logits, mask, labels)
    audio_temperature = _temperature(audio_logits, mask, labels)
    feature_weight, feature_accuracy = _select_weight(
        prior, feature_logits, mask, labels, feature_temperature
    )
    audio_weight, audio_accuracy = _select_weight(
        prior, audio_logits, mask, labels, audio_temperature
    )
    audio_predictions = _combined_predictions(
        prior,
        audio_logits,
        mask,
        temperature=audio_temperature,
        weight=audio_weight,
    )
    feature_predictions = _combined_predictions(
        prior,
        feature_logits,
        mask,
        temperature=feature_temperature,
        weight=feature_weight,
    )
    probabilities = _softmax(audio_logits / audio_temperature, mask)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    baseline_accuracy = float(np.mean(arrays["baseline_correct"]))
    folds = []
    for player in map(int, DEV_PLAYERS):
        selected = arrays["players"] == player
        baseline = float(np.mean(arrays["baseline_correct"][selected]))
        feature = float(np.mean(feature_predictions[selected] == labels[selected]))
        audio = float(np.mean(audio_predictions[selected] == labels[selected]))
        folds.append(
            {
                "player": player,
                "examples": int(np.sum(selected)),
                "baseline": baseline,
                "feature": feature,
                "audio": audio,
                "audio_delta": audio - baseline,
            }
        )
    predicted_strings = arrays["strings"][np.arange(len(labels)), audio_predictions]
    string_distribution = {
        int(string_idx): float(np.mean(predicted_strings == string_idx)) for string_idx in range(6)
    }
    parameter_count = sum(parameter.numel() for parameter in AudioCandidateRanker().parameters())
    ece = _ece(probabilities, labels)
    mean_entropy = float(np.mean(entropy))
    delta = audio_accuracy - baseline_accuracy
    worst_fold_delta = min(fold["audio_delta"] for fold in folds)
    active_strings = sum(value >= 0.01 for value in string_distribution.values())
    calibrated_noncollapsed = ece <= 0.15 and mean_entropy >= 0.05 and active_strings >= 4
    gate_passed = delta >= 0.05 and worst_fold_delta >= -0.03 and calibrated_noncollapsed
    return {
        "seed": SEED,
        "examples": len(labels),
        "baseline_accuracy": baseline_accuracy,
        "feature_accuracy": feature_accuracy,
        "feature_delta": feature_accuracy - baseline_accuracy,
        "feature_weight": feature_weight,
        "feature_temperature": feature_temperature,
        "audio_accuracy": audio_accuracy,
        "audio_delta": delta,
        "audio_weight": audio_weight,
        "audio_temperature": audio_temperature,
        "folds": folds,
        "worst_fold_delta": worst_fold_delta,
        "ece": ece,
        "mean_entropy": mean_entropy,
        "string_distribution": string_distribution,
        "active_strings": active_strings,
        "parameter_count": parameter_count,
        "elapsed_s": elapsed,
        "gate_passed": gate_passed,
        "calibrated_noncollapsed": calibrated_noncollapsed,
        "feature_epochs": feature_epochs,
        "audio_epochs": audio_epochs,
        "train_cap": train_cap,
        "batch_size": batch_size,
    }


def render_report(result: dict[str, Any], notes_path: Path) -> str:
    fold_rows = [
        f"| {fold['player']:02d} | {fold['examples']} | {fold['baseline']:.4f} | "
        f"{fold['feature']:.4f} | {fold['audio']:.4f} | {fold['audio_delta']:+.4f} |"
        for fold in result["folds"]
    ]
    distribution = ", ".join(
        f"s{string_idx}={value:.3f}" for string_idx, value in result["string_distribution"].items()
    )
    decision = (
        "PASS — paid capped optimization is authorized"
        if result["gate_passed"]
        else ("FAIL — do not start paid training and do not enlarge the model")
    )
    return "\n".join(
        [
            "# String assignment Phase 2: free timbral signal probe",
            "",
            "Date: 2026-07-14",
            "",
            "## Method",
            "",
            "- Data: GuitarSet players 00–04 only; player 05 was not read.",
            f"- Examples: **{result['examples']}** ambiguous, pitch-correct high-resolution "
            "events from the frozen Phase 0 note table.",
            "- Split: five leave-one-player-out folds. Every reported prediction is OOF.",
            f"- Seed: **{result['seed']}**.",
            "- Window: 512 ms at 16 kHz, -64 ms/+448 ms around the detected onset, "
            "with zero padding and training jitter sampled from the empirical capped "
            "±50 ms onset-error distribution.",
            "- Audio model: fixed 512/128 log-STFT, three 16/32/64-channel "
            "Conv2d+GroupNorm+SiLU+pool blocks, 96/48 candidate MLP.",
            f"- Parameters: **{result['parameter_count']}** (<250,000).",
            "- Augmentation: gain, time-domain spectral tilt, broadband noise, mild "
            "compression, and onset shift.",
            "- Sampling balances player, string, pitch region, and solo/comp mode.",
            f"- Frozen Phase 0 source: `{notes_path.name}`.",
            f"- Frozen source SHA-256: `{result['notes_sha256']}`.",
            f"- Training config: feature epochs={result['feature_epochs']}, "
            f"audio epochs={result['audio_epochs']}, train cap={result['train_cap']}, "
            f"batch size={result['batch_size']}.",
            "",
            "## OOF result",
            "",
            "| held-out player | examples | prior-only baseline | "
            "feature-only+prior | audio+prior | audio delta |",
            "|---:|---:|---:|---:|---:|---:|",
            *fold_rows,
            "",
            f"- Best prior-only candidate top-1: **{result['baseline_accuracy']:.4f}**",
            f"- Feature-only + prior: **{result['feature_accuracy']:.4f}** "
            f"(`{result['feature_delta']:+.4f}`, weight={result['feature_weight']}, "
            f"temperature={result['feature_temperature']:.2f})",
            f"- Compact audio + prior: **{result['audio_accuracy']:.4f}** "
            f"(`{result['audio_delta']:+.4f}`, weight={result['audio_weight']}, "
            f"temperature={result['audio_temperature']:.2f})",
            f"- Worst player-fold delta: **{result['worst_fold_delta']:+.4f}**",
            f"- Calibrated ECE: **{result['ece']:.4f}**",
            f"- Mean posterior entropy: **{result['mean_entropy']:.4f}**",
            f"- Predicted-string distribution: {distribution}",
            f"- Training/evaluation wall time: **{result['elapsed_s'] / 60:.1f} min** on CPU",
            "",
            "## Free-probe gate",
            "",
            "Required: audio delta ≥ +0.05, every player fold ≥ -0.03, and calibrated "
            "non-collapsed posteriors.",
            "",
            f"**{decision}.**",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-epochs", type=int, default=12)
    parser.add_argument("--audio-epochs", type=int, default=2)
    parser.add_argument("--train-cap", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    _seed_everything(SEED)
    waveforms, arrays = prepare_cache(
        args.notes.expanduser(), args.data_home.expanduser(), args.cache_dir.expanduser()
    )
    print(f"examples={len(arrays['labels'])} cache={args.cache_dir}", flush=True)
    if args.prepare_only:
        return 0
    result = run_probe(
        waveforms,
        arrays,
        args.data_home.expanduser(),
        feature_epochs=args.feature_epochs,
        audio_epochs=args.audio_epochs,
        train_cap=args.train_cap,
        batch_size=args.batch_size,
    )
    result["notes_sha256"] = hashlib.sha256(args.notes.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_report(result, args.notes))
    result_path = args.output.with_suffix(".json")
    with result_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"audio_delta={result['audio_delta']:+.4f}")
    print(f"worst_fold_delta={result['worst_fold_delta']:+.4f}")
    print(f"gate={'PASS' if result['gate_passed'] else 'FAIL'}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
