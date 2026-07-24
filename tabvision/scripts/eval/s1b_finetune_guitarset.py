"""Accuracy-loop Q2 (ROI deep-dive §3.2) — recipe stage 2: GuitarSet fine-tune.

The pretrain-only probe reached ambiguous top-1 +0.0302 [+0.0163, +0.0446]
against a +0.05 bar (`s1b_context_probe_2026-07-22.md`). MIDI-to-Tab prices
the fine-tune step alone at +4.0 pp string agreement, and the diagnosis there
was domain: SynthTab is notation-derived, GuitarSet is what players' hands
actually did.

**Leave-one-player-out is not optional here.** The lattice this is gated on
is players 00-04, so a model fine-tuned on all of 00-04 would be scored on
its own training data. Instead, for each dev player P this trains a separate
fold on the *other four* players and emits `context_v2_oof_<P>.pt`; the
rescoring harness then scores player P's tracks only with the fold that never
saw P. Same protocol as `string_assignment_phase4.py::_oof_position_prior`
and the OOF position prior used throughout the accuracy program.

Player-05 is never read by this script — it stays sealed.

Fine-tune data is GuitarSet gold tablature (CC-BY-4.0); the initialization is
the NC SynthTab-pretrained checkpoint, so the fine-tuned folds inherit NC
(LICENSES.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.eval.s1b_train_context import (
    PITCH_HIGH,
    PITCH_LOW,
    SEED,
    WINDOW,
    ContextStringModel,
    gap_bucket,
)
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.types import GuitarConfig

DEV_PLAYERS = ("00", "01", "02", "03", "04")


def load_guitarset_symbolic(data_home: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per-player windowed tensors from GuitarSet gold tablature."""
    cfg = GuitarConfig()
    per_player: dict[str, dict[str, list[np.ndarray]]] = {
        player: {"pitch": [], "gap": [], "string": []} for player in DEV_PLAYERS
    }
    for path in sorted((data_home / "annotation").glob("*.jams")):
        player = path.stem[:2]
        if player not in per_player:
            continue
        events = parse_guitarset_jams(path, cfg)
        if len(events) < WINDOW:
            continue
        events = sorted(events, key=lambda event: (event.onset_s, event.string_idx))
        pitch = (
            np.clip(
                np.asarray([event.pitch_midi for event in events], dtype=np.int64) - PITCH_LOW,
                0,
                PITCH_HIGH - PITCH_LOW,
            )
            + 1
        )
        onset_ms = np.asarray([round(event.onset_s * 1000.0) for event in events], dtype=np.int64)
        gaps = gap_bucket(np.maximum(np.diff(onset_ms, prepend=onset_ms[0]), 0))
        strings = np.asarray([event.string_idx for event in events], dtype=np.int64)

        count = len(events) // WINDOW
        usable = count * WINDOW
        per_player[player]["pitch"].append(pitch[:usable].reshape(count, WINDOW))
        per_player[player]["gap"].append(gaps[:usable].reshape(count, WINDOW))
        per_player[player]["string"].append(strings[:usable].reshape(count, WINDOW))

    return {
        player: {key: np.concatenate(value) for key, value in arrays.items()}
        for player, arrays in per_player.items()
        if arrays["pitch"]
    }


def finetune_fold(
    pretrained: Path,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[ContextStringModel, dict[str, Any]]:
    """Fine-tune the pretrained checkpoint on four players, score on the fifth."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    payload = torch.load(pretrained, map_location="cpu", weights_only=True)
    model = ContextStringModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_tensors = {key: torch.from_numpy(value) for key, value in train_data.items()}
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    total = len(train_data["pitch"])

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(total)
        running = 0.0
        for offset in range(0, total, batch_size):
            batch = torch.from_numpy(order[offset : offset + batch_size])
            logits = model(train_tensors["pitch"][batch], train_tensors["gap"][batch])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 6), train_tensors["string"][batch].reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch)
        metrics = score_fold(model, val_data, batch_size)
        metrics.update({"epoch": epoch, "train_loss": running / max(total, 1)})
        history.append(metrics)

    return model, {
        "history": history,
        "train_windows": total,
        "val_windows": int(len(val_data["pitch"])),
        "seconds": round(time.perf_counter() - started, 1),
    }


@torch.no_grad()
def score_fold(
    model: ContextStringModel, data: dict[str, np.ndarray], batch_size: int
) -> dict[str, float]:
    """Held-out-player string accuracy, overall and on ambiguous pitches."""
    from scripts.eval.s1b_train_context import _ambiguous_mask

    model.eval()
    correct = total = amb_correct = amb_total = 0
    tensors = {key: torch.from_numpy(value) for key, value in data.items()}
    for offset in range(0, len(data["pitch"]), batch_size):
        span = slice(offset, offset + batch_size)
        pitch = tensors["pitch"][span]
        predicted = model(pitch, tensors["gap"][span]).argmax(dim=-1)
        hits = predicted == tensors["string"][span]
        correct += int(hits.sum())
        total += hits.numel()
        mask = _ambiguous_mask(pitch)
        amb_correct += int((hits & mask).sum())
        amb_total += int(mask.sum())
    return {
        "accuracy": correct / max(total, 1),
        "ambiguous_accuracy": amb_correct / max(amb_total, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    pretrained = args.pretrained or (data_root / "models" / "s1b_symbolic" / "context_v2.pt")
    output_dir = args.output_dir or (data_root / "models" / "s1b_symbolic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading GuitarSet symbolic from {data_home}", flush=True)
    per_player = load_guitarset_symbolic(data_home)
    for player, arrays in sorted(per_player.items()):
        print(f"  player {player}: {len(arrays['pitch'])} windows", flush=True)

    folds: dict[str, Any] = {}
    for held_out in DEV_PLAYERS:
        train_data = {
            key: np.concatenate(
                [arrays[key] for player, arrays in per_player.items() if player != held_out]
            )
            for key in ("pitch", "gap", "string")
        }
        model, summary = finetune_fold(
            pretrained,
            train_data,
            per_player[held_out],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        destination = output_dir / f"context_v2_oof_{held_out}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "config": {"d_model": 128, "layers": 3, "heads": 4}},
            destination,
        )
        summary["checkpoint"] = str(destination)
        summary["held_out_player"] = held_out
        folds[held_out] = summary
        final = summary["history"][-1]
        print(
            f"fold {held_out}: val_acc={final['accuracy']:.4f} "
            f"val_amb_acc={final['ambiguous_accuracy']:.4f} "
            f"({summary['seconds']:.0f}s)",
            flush=True,
        )

    payload = {
        "pretrained": str(pretrained),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": SEED,
        "protocol": "leave-one-player-out over dev players 00-04; player-05 never read",
        "folds": folds,
        "mean_held_out_ambiguous_accuracy": float(
            np.mean([folds[p]["history"][-1]["ambiguous_accuracy"] for p in folds])
        ),
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"mean held-out ambiguous accuracy: {payload['mean_held_out_ambiguous_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
