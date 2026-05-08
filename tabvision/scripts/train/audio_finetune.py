"""Phase 7 audio fine-tuning scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "script": "audio_finetune",
        "phase": 7,
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "status": "ready" if args.dry_run else "blocked",
        "inputs": {
            "train_manifest": str(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "base_backend": args.base_backend,
        },
        "outputs": {
            "checkpoint_dir": str(args.checkpoint_dir),
            "report": str(args.output),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "steps": [
            "load augmented audio manifests",
            "initialize pretrained high-resolution guitar transcription backend",
            "fine-tune onset and pitch heads with fixed seeds",
            "write checkpoint metadata and validation metrics",
        ],
        "blockers": []
        if args.dry_run
        else ["full audio fine-tuning requires GPU-capable torch and training data"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default="data/augmented/audio/train.json")
    parser.add_argument("--validation-manifest", default="data/eval/manifest.toml")
    parser.add_argument("--base-backend", default="highres")
    parser.add_argument("--checkpoint-dir", default="data/augmented/checkpoints/audio")
    parser.add_argument("--output", type=Path, default=Path("audio_finetune_plan.json"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
