"""Modal cloud-GPU runner for the YOLO-OBB fretboard-parts fine-tune.

Mirrors the pattern in ``tabvision-server/tools/finetune_basic_pitch_modal.py``
(v0 Phase-1 audio fine-tune). Runs ``ultralytics.YOLO.train`` on a Modal L4
GPU, uploads the Roboflow YOLO-OBB dataset as an image layer (cached
across runs), then tars + downloads the trained checkpoint back to
``$TABVISION_DATA_ROOT/models/runs/``.

**License note:** ultralytics is AGPL-3.0; running this script taints the
combined work as AGPL-3.0. See LICENSES.md and DECISIONS.md
(2026-05-05 "Phase 3 detector path").

Two modes:
  --smoke   3 epochs, batch 4 — validates GPU + image + dataset round-trip.
            ~5 min, ~$0.10.
  (default) 50 epochs, full split, batch 16 — the real Phase-3 fine-tune.
            ~25-40 min on L4, ~$0.30-0.50.

Usage::

    modal run scripts/train/yolo_guitar_obb_modal.py --smoke
    modal run scripts/train/yolo_guitar_obb_modal.py
    modal run scripts/train/yolo_guitar_obb_modal.py --epochs 100 --batch 32
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

# ----- local paths -----

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # tabvision/
DATA_ROOT = Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))

DATASET_LOCAL = DATA_ROOT / "datasets" / "roboflow-b101-guitar-3-4efcd-v2"
OUTPUT_LOCAL = DATA_ROOT / "models" / "runs"
STABLE_WEIGHT_LINK = DATA_ROOT / "models" / "guitar-yolo-obb-finetuned.pt"

# ----- remote paths -----

REMOTE_DATASET = "/data/dataset"
REMOTE_OUTPUT = "/output"
REMOTE_RUN_NAME = "guitar-obb-finetune"

# ----- Modal image -----

# Ultralytics ships an official image with CUDA + PyTorch + their package.
# We pin ultralytics here for reproducibility; AGPL-tainted but explicit.
image = (
    modal.Image.from_registry("ultralytics/ultralytics:latest", add_python=None)
    .pip_install("ultralytics>=8.3", "numpy<2", "opencv-python-headless")
    .add_local_dir(str(DATASET_LOCAL), REMOTE_DATASET)
)

app = modal.App("tabvision-yolo-obb-finetune", image=image)


@app.function(gpu="L4", timeout=60 * 60)
def finetune(
    *,
    smoke: bool = False,
    epochs: int = 50,
    batch: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    base_model: str = "yolo11n-obb.pt",
    seed: int = 0,
) -> bytes:
    """Run YOLO-OBB fine-tune on a remote L4 and return the run dir as a tar."""
    import logging

    import torch
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("yolo-obb")

    log.info("torch=%s cuda=%s gpus=%d", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible to torch")

    # Hyperparameters: smoke is for the wiring check only.
    if smoke:
        epochs = 3
        batch = 4

    data_yaml = Path(REMOTE_DATASET) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}; check image-mount step")

    os.makedirs(REMOTE_OUTPUT, exist_ok=True)

    log.info(
        "training: base=%s epochs=%d batch=%d imgsz=%d lr0=%g seed=%d",
        base_model, epochs, batch, img_size, lr0, seed,
    )
    t0 = time.time()
    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        lr0=lr0,
        seed=seed,
        deterministic=True,
        project=REMOTE_OUTPUT,
        name=REMOTE_RUN_NAME,
        exist_ok=False,
        device=0,
        verbose=True,
    )
    log.info("training finished in %.1fs", time.time() - t0)

    # Validate on the held-out split for sanity.
    metrics = model.val(data=str(data_yaml), split="val")
    log.info("val metrics: mAP50=%.4f mAP50-95=%.4f", metrics.box.map50, metrics.box.map)

    # Tar the run dir back to the caller.
    run_dir = Path(results.save_dir)
    log.info("packaging %s", run_dir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(run_dir), arcname="run")
    log.info("tarball size %.1f MB", buf.tell() / 1e6)
    return buf.getvalue()


@app.local_entrypoint()
def main(
    smoke: bool = False,
    epochs: int = 50,
    batch: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    base_model: str = "yolo11n-obb.pt",
    seed: int = 0,
) -> None:
    if not DATASET_LOCAL.exists():
        print(
            f"[modal] dataset not found at {DATASET_LOCAL}\n"
            "        Run: python -m scripts.acquire.datasets roboflow-guitar "
            "--project guitar-3-4efcd",
            file=sys.stderr,
        )
        sys.exit(2)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_LOCAL / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[modal] timestamp={timestamp}\n"
        f"[modal] dataset: {DATASET_LOCAL}\n"
        f"[modal] output dir: {out_dir}\n"
        f"[modal] smoke={smoke} epochs={epochs} batch={batch} imgsz={img_size} lr0={lr0}",
        file=sys.stderr,
    )

    tarball = finetune.remote(
        smoke=smoke,
        epochs=epochs,
        batch=batch,
        img_size=img_size,
        lr0=lr0,
        base_model=base_model,
        seed=seed,
    )

    archive = out_dir / "run.tar.gz"
    archive.write_bytes(tarball)
    print(f"[modal] artifact ({len(tarball)/1e6:.1f} MB) -> {archive}", file=sys.stderr)

    with tarfile.open(archive) as tar:
        tar.extractall(out_dir)
    extracted = out_dir / "run"
    best = extracted / "weights" / "best.pt"
    print(f"[modal] extracted -> {extracted}", file=sys.stderr)
    if best.exists():
        STABLE_WEIGHT_LINK.parent.mkdir(parents=True, exist_ok=True)
        if STABLE_WEIGHT_LINK.exists() or STABLE_WEIGHT_LINK.is_symlink():
            STABLE_WEIGHT_LINK.unlink()
        STABLE_WEIGHT_LINK.symlink_to(best.resolve())
        print(f"[modal] best weights linked at {STABLE_WEIGHT_LINK}", file=sys.stderr)
    else:
        print(f"[modal] WARNING: no best.pt at {best}", file=sys.stderr)
