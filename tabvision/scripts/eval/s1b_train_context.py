"""Accuracy-loop Q2 (ROI deep-dive §3.2) — contextual string model (pretrain).

A small transformer encoder over a window of notes that predicts, per note,
a distribution over the six strings from **pitch and timing context alone**.
It never sees a string as input, so it cannot copy the answer; everything it
knows about position comes from the surrounding phrase.

That is deliberately the weakest form of the MIDI-to-Tab / Fretting-
Transformer recipe: no masked-string conditioning on neighbouring strings, no
autoregressive decoding. It is chosen because the integration target
(`fuse()` emissions) consumes exactly a per-note distribution — the
string-to-string coupling is already the transition prior's job — and because
a single forward pass is cheap enough to keep the whole probe offline. If
context carries signal at all, this measures it; if it passes, partial
masking and a fine-tune are the obvious next levers rather than prerequisites.

Trained on the SynthTab symbolic corpus (`s1b_extract_symbolic.py`), held out
by track. CC-BY-NC-4.0 inherited (LICENSES.md) — this checkpoint is a probe
artifact, not a registered one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

WINDOW = 64
PITCH_LOW = 40
PITCH_HIGH = 88
PITCH_VOCAB = PITCH_HIGH - PITCH_LOW + 2  # + 1 pad slot
GAP_BUCKETS_MS = (10, 40, 80, 160, 320, 640, 1280)
GAP_VOCAB = len(GAP_BUCKETS_MS) + 2  # + overflow + pad
SEED = 42


def gap_bucket(delta_ms: np.ndarray) -> np.ndarray:
    """Bucket inter-onset gaps; bucket 0 is 'same cluster' (< 10 ms)."""
    return np.digitize(delta_ms, GAP_BUCKETS_MS).astype(np.int64) + 1


class ContextStringModel(nn.Module):
    """Transformer encoder → per-note 6-way string logits."""

    def __init__(self, *, d_model: int = 128, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.pitch = nn.Embedding(PITCH_VOCAB, d_model, padding_idx=0)
        self.gap = nn.Embedding(GAP_VOCAB, d_model, padding_idx=0)
        self.position = nn.Embedding(WINDOW, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 6)

    def forward(self, pitch: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(pitch.shape[1], device=pitch.device)
        hidden = self.pitch(pitch) + self.gap(gap) + self.position(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=(pitch == 0))
        return self.head(hidden)


def build_windows(corpus_path: Path, *, max_windows: int = 0) -> dict[str, np.ndarray]:
    """Chop the corpus into fixed windows of consecutive notes per track."""
    with np.load(corpus_path) as payload:
        pitch = payload["pitch"].astype(np.int64)
        string = payload["string"].astype(np.int64)
        onset = payload["onset_ms"].astype(np.int64)
        offsets = payload["track_offset"].astype(np.int64)

    pitch_tokens = np.clip(pitch - PITCH_LOW, 0, PITCH_HIGH - PITCH_LOW) + 1
    windows_pitch: list[np.ndarray] = []
    windows_gap: list[np.ndarray] = []
    windows_string: list[np.ndarray] = []
    window_track: list[int] = []

    # Visit tracks in seeded random order: ``max_windows`` stops the walk
    # early, and the archive is grouped by source, so taking the first N
    # tracks would sample one corner of the repertoire rather than the corpus.
    track_order = np.random.default_rng(SEED).permutation(len(offsets) - 1)
    total_windows = 0
    for track_index in track_order:
        start, end = int(offsets[track_index]), int(offsets[track_index + 1])
        length = end - start
        if length < 2:
            continue
        track_onset = onset[start:end]
        deltas = np.diff(track_onset, prepend=track_onset[0])
        gaps = gap_bucket(np.maximum(deltas, 0))
        count = length // WINDOW
        if count == 0:
            continue
        usable = count * WINDOW
        windows_pitch.append(pitch_tokens[start : start + usable].reshape(count, WINDOW))
        windows_gap.append(gaps[:usable].reshape(count, WINDOW))
        windows_string.append(string[start : start + usable].reshape(count, WINDOW))
        window_track.extend([int(track_index)] * count)
        total_windows += count
        if max_windows and total_windows >= max_windows:
            break

    data = {
        "pitch": np.concatenate(windows_pitch),
        "gap": np.concatenate(windows_gap),
        "string": np.concatenate(windows_string),
        "track": np.asarray(window_track, dtype=np.int64),
    }
    if max_windows:
        data = {key: value[:max_windows] for key, value in data.items()}
    return data


def _ambiguous_mask(pitch_tokens: torch.Tensor) -> torch.Tensor:
    """True where the pitch is playable at more than one standard position."""
    midi = pitch_tokens + PITCH_LOW - 1
    open_midi = torch.tensor([40, 45, 50, 55, 59, 64], device=pitch_tokens.device)
    frets = midi[..., None] - open_midi
    return ((frets >= 0) & (frets <= 24)).sum(dim=-1) > 1


def train(
    data: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    max_seconds: float,
) -> tuple[ContextStringModel, dict[str, Any]]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    tracks = np.unique(data["track"])
    rng.shuffle(tracks)
    val_count = max(1, int(len(tracks) * val_fraction))
    val_tracks = set(tracks[:val_count].tolist())
    is_val = np.fromiter((int(t) in val_tracks for t in data["track"]), bool, len(data["track"]))

    tensors = {key: torch.from_numpy(data[key]) for key in ("pitch", "gap", "string")}
    train_idx = np.flatnonzero(~is_val)
    val_idx = np.flatnonzero(is_val)

    model = ContextStringModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        running = 0.0
        seen = 0
        for offset in range(0, len(order), batch_size):
            batch = torch.from_numpy(order[offset : offset + batch_size])
            logits = model(tensors["pitch"][batch], tensors["gap"][batch])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 6), tensors["string"][batch].reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
            if max_seconds and time.perf_counter() - started > max_seconds:
                stopped_early = True
                break
        metrics = evaluate(model, tensors, val_idx, batch_size)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        history.append(metrics)
        print(
            f"epoch {epoch}: train_loss={metrics['train_loss']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f} "
            f"val_amb_acc={metrics['val_ambiguous_accuracy']:.4f} "
            f"({metrics['seconds']:.0f}s)",
            flush=True,
        )
        if stopped_early:
            print("stopping early: training time budget reached", flush=True)
            break

    return model, {
        "history": history,
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "val_tracks": int(len(val_tracks)),
        "stopped_early": stopped_early,
    }


@torch.no_grad()
def evaluate(
    model: ContextStringModel,
    tensors: dict[str, torch.Tensor],
    val_idx: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    correct = total = amb_correct = amb_total = 0
    for offset in range(0, len(val_idx), batch_size):
        batch = torch.from_numpy(val_idx[offset : offset + batch_size])
        pitch = tensors["pitch"][batch]
        logits = model(pitch, tensors["gap"][batch])
        predicted = logits.argmax(dim=-1)
        gold = tensors["string"][batch]
        hits = predicted == gold
        correct += int(hits.sum())
        total += hits.numel()
        mask = _ambiguous_mask(pitch)
        amb_correct += int((hits & mask).sum())
        amb_total += int(mask.sum())
    return {
        "val_accuracy": correct / max(total, 1),
        "val_ambiguous_accuracy": amb_correct / max(amb_total, 1),
    }


class ContextScorer:
    """Adapter: scores a rescoring-harness ``Track`` with the trained model."""

    name = "context"

    def __init__(self, model: ContextStringModel) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def log_probs(self, track: Any) -> np.ndarray:
        notes = track.notes
        if not notes:
            return np.zeros((0, 6), dtype=np.float64)
        pitch = (
            np.clip(
                np.asarray([note.pitch_midi for note in notes], dtype=np.int64) - PITCH_LOW,
                0,
                PITCH_HIGH - PITCH_LOW,
            )
            + 1
        )
        onset_ms = np.asarray([round(note.onset_s * 1000.0) for note in notes], dtype=np.int64)
        gaps = gap_bucket(np.maximum(np.diff(onset_ms, prepend=onset_ms[0]), 0))

        padded = math.ceil(len(notes) / WINDOW) * WINDOW
        pitch_padded = np.zeros(padded, dtype=np.int64)
        gap_padded = np.zeros(padded, dtype=np.int64)
        pitch_padded[: len(notes)] = pitch
        gap_padded[: len(notes)] = gaps
        logits = self.model(
            torch.from_numpy(pitch_padded.reshape(-1, WINDOW)),
            torch.from_numpy(gap_padded.reshape(-1, WINDOW)),
        )
        flat = torch.log_softmax(logits, dim=-1).reshape(-1, 6).numpy()
        return flat[: len(notes)].astype(np.float64)


def load_context_scorer(checkpoint: Path) -> ContextScorer:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ContextStringModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return ContextScorer(model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    corpus = args.corpus or (data_root / "models" / "s1b_symbolic" / "synthtab_all.npz")
    output = args.output or (data_root / "models" / "s1b_symbolic" / "context_v2.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"building windows from {corpus}", flush=True)
    data = build_windows(corpus, max_windows=args.max_windows)
    print(f"windows={len(data['pitch'])} notes={data['pitch'].size}", flush=True)

    model, summary = train(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        max_seconds=args.max_seconds,
    )
    config = {"d_model": 128, "layers": 3, "heads": 4}
    torch.save({"state_dict": model.state_dict(), "config": config}, output)
    summary.update(
        {
            "corpus": str(corpus),
            "checkpoint": str(output),
            "window": WINDOW,
            "config": config,
            "parameters": sum(p.numel() for p in model.parameters()),
            "seed": SEED,
        }
    )
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
