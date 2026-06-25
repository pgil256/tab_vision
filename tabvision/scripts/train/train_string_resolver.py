"""Train the WS4 learned string-resolution model on extracted GAPS crops.

Core training loop (``train_string_resolver``) callable locally (CPU smoke) or
from the Modal GPU wrapper (``string_resolver_modal.py``). Reads the dataset
produced by ``extract_string_dataset.py`` (``manifest.jsonl`` + ``crops/``),
splits **by clip** (no same-clip leakage between train/val), filters on
alignment ``peak_ratio`` to curb label noise, and fine-tunes a pretrained
ResNet-18 to a 6-way string classifier.

IMPORTANT: no horizontal/vertical flips in augmentation — the string identity is
encoded in the across-neck position, so a flip would corrupt the label. Only
photometric jitter is applied.

Usage::

    python -m scripts.train.train_string_resolver \
        --data-dir ~/.tabvision/cache/gaps_string_dataset --epochs 15
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tabvision.video.hand.learned_string import StringResolverNet, preprocess_crops


def _read_manifest(data_dir: Path, *, min_peak_ratio: float) -> list[dict]:
    rows: list[dict] = []
    with open(data_dir / "manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(r.get("peak_ratio", 0.0)) >= min_peak_ratio:
                rows.append(r)
    return rows


def split_by_clip(rows: list[dict], *, val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Clip-disjoint train/val split (no crop from a val clip appears in train).

    Falls back to a row-level split when there are <2 distinct clips (smoke /
    tiny datasets), so the loader is never handed an empty split.
    """
    stems = sorted({r["stem"] for r in rows})
    rng = random.Random(seed)
    if len(stems) < 2:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac)))
        return shuffled[n_val:], shuffled[:n_val]
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_stems = set(stems[:n_val])
    train = [r for r in rows if r["stem"] not in val_stems]
    val = [r for r in rows if r["stem"] in val_stems]
    return train, val


class StringCropDataset(Dataset):
    """(crop tensor, string_idx) from manifest rows; photometric jitter only."""

    def __init__(self, rows: list[dict], data_dir: Path, *, augment: bool) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        import cv2

        r = self.rows[i]
        crop = cv2.imread(str(self.data_dir / r["jpg"]))
        if crop is None:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        if self.augment:
            # Photometric only (brightness/contrast) — never geometric flips.
            alpha = 1.0 + np.random.uniform(-0.15, 0.15)  # contrast
            beta = np.random.uniform(-15, 15)  # brightness
            crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        x = preprocess_crops(crop)[0]  # (3, H, W)
        return x, int(r["string_idx"])


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    return correct / total if total else 0.0


def train_string_resolver(
    data_dir: Path,
    out_dir: Path,
    *,
    epochs: int = 15,
    batch: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    min_peak_ratio: float = 2.0,
    seed: int = 0,
    device: str | None = None,
    num_workers: int = 0,
    limit: int | None = None,
) -> dict:
    """Fine-tune the string classifier; saves best.pt, returns metrics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows = _read_manifest(data_dir, min_peak_ratio=min_peak_ratio)
    if limit is not None:
        rows = rows[:limit]
    train_rows, val_rows = split_by_clip(rows, val_frac=val_frac, seed=seed)
    print(
        f"dataset: {len(rows)} crops ({len(train_rows)} train / {len(val_rows)} val), "
        f"device={device}",
        flush=True,
    )

    train_ds = StringCropDataset(train_rows, data_dir, augment=True)
    val_ds = StringCropDataset(val_rows, data_dir, augment=False)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=num_workers)

    model = StringResolverNet(n_strings=6, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(1, len(train_ds))
        val_acc = _evaluate(model, val_ld, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "n_strings": 6}, out_dir / "best.pt")

    metrics = {
        "best_val_acc": best_acc,
        "history": history,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"best val_acc={best_acc:.4f} -> {out_dir / 'best.pt'}", flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_string_dataset",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / ".tabvision" / "data" / "models" / "string_resolver",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-peak-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap crops (smoke)")
    args = ap.parse_args(argv)

    train_string_resolver(
        args.data_dir,
        args.out_dir,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        val_frac=args.val_frac,
        min_peak_ratio=args.min_peak_ratio,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
