"""Fine-tune YOLO-OBB on the Roboflow guitar dataset — Phase 3.

Per ``docs/DECISIONS.md`` (2026-05-05 "Phase 3 detector path"), we accept
ultralytics AGPL-3.0 contagion in exchange for a working guitar detector
with rotation. The trained checkpoint is saved alongside the dataset.

Prereqs:
    pip install '.[vision]'
    export ROBOFLOW_API_KEY=...
    python -m scripts.acquire.datasets roboflow-guitar
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"
DEFAULT_BASE_MODEL = "yolo11n-obb.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640


def _data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", DEFAULT_DATA_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-OBB on guitar dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to the dataset directory (must contain data.yaml). "
        "Defaults to $TABVISION_DATA_ROOT/datasets/roboflow-b101-guitar-3-v3.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default=None, help="auto-pick if unset (cpu / 0 / cuda)")
    parser.add_argument(
        "--name",
        default="guitar-obb-finetune",
        help="ultralytics run name; outputs go to runs/obb/<name>/",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="ultralytics project dir; defaults to $TABVISION_DATA_ROOT/models/runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducibility",
    )
    args = parser.parse_args(argv)

    if args.dataset is None:
        args.dataset = _data_root() / "datasets" / "roboflow-b101-guitar-3-v3"
    if args.project is None:
        args.project = _data_root() / "models" / "runs"
    args.project.mkdir(parents=True, exist_ok=True)

    data_yaml = args.dataset / "data.yaml"
    if not data_yaml.exists():
        print(
            f"error: dataset not found at {args.dataset}\n"
            "Run: python -m scripts.acquire.datasets roboflow-guitar",
            file=sys.stderr,
        )
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: ultralytics not installed. "
            "Install with: pip install '.[vision]' (AGPL — see LICENSES.md)",
            file=sys.stderr,
        )
        return 2

    print(f"loading base model {args.base_model}")
    model = YOLO(args.base_model)

    print(f"fine-tuning on {data_yaml}")
    print(f"  epochs={args.epochs}, batch={args.batch}, imgsz={args.img_size}, lr0={args.lr0}")
    print(f"  device={args.device or 'auto'}")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        exist_ok=False,
    )

    final_dir = Path(results.save_dir)
    best = final_dir / "weights" / "best.pt"
    if best.exists():
        # Symlink to a stable path the runtime can pick up without knowing
        # the run name.
        stable = _data_root() / "models" / "guitar-yolo-obb-finetuned.pt"
        stable.parent.mkdir(parents=True, exist_ok=True)
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(best.resolve())
        print(f"\n✓ best weights linked at {stable} → {best}")
    else:
        print(f"⚠ no best.pt found in {final_dir}/weights/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
