"""Robust local-CPU YOLO-OBB training run (v1.1 chunk-2).

Standalone so a background launch survives turn boundaries and any error lands in the log
rather than vanishing. Saves periodic checkpoints (save_period) so a mid-run death costs at
most a few epochs, and copies the best (or newest) weights to the stable runtime path.

Usage: python -m scripts.train._cpu_train_guitar_obb [--epochs N] [--batch N] [--resume]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

DATA = r"C:\Users\patri\.tabvision\data\datasets\roboflow-b101-guitar-3-4efcd-v2\data.yaml"
PROJECT = r"C:\Users\patri\.tabvision\data\models\runs"
NAME = "guitar-obb-finetune"
STABLE = Path(r"C:\Users\patri\.tabvision\data\models\guitar-yolo-obb-finetuned.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(PROJECT) / NAME
    ckpt_last = run_dir / "weights" / "last.pt"
    if args.resume and ckpt_last.exists():
        print(f"[train] resuming from {ckpt_last}", flush=True)
        model = YOLO(str(ckpt_last))
        train_kwargs = {"resume": True}
    else:
        model = YOLO("yolo11n-obb.pt")
        train_kwargs = {
            "data": DATA,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": 640,
            "device": "cpu",
            "seed": 0,
            "deterministic": True,
            "patience": args.patience,
            "project": PROJECT,
            "name": NAME,
            "exist_ok": True,
            "verbose": False,
            "plots": False,
            "workers": 0,
            "cache": False,
            "save_period": 5,
        }

    try:
        results = model.train(**train_kwargs)
    except Exception:  # noqa: BLE001 — capture any failure to the log
        print("[train] TRAINING FAILED:\n" + traceback.format_exc(), flush=True)
        return 1

    best = Path(results.save_dir) / "weights" / "best.pt"
    src = best if best.exists() else ckpt_last
    if src.exists():
        STABLE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, STABLE)
        print(f"[train] DONE. weights {src} -> {STABLE}", flush=True)
        try:
            mv = model.val(data=DATA, split="val", device="cpu", verbose=False)
            print(f"[train] VAL mAP50={mv.box.map50:.4f} mAP50-95={mv.box.map:.4f}", flush=True)
        except Exception:  # noqa: BLE001
            print("[train] val failed:\n" + traceback.format_exc(), flush=True)
    else:
        print(f"[train] NO weights found at {best} or {ckpt_last}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
