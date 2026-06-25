"""Modal cloud-GPU runner for the WS4 learned string-resolution fine-tune.

Mirrors ``scripts/train/yolo_guitar_obb_modal.py``: the extracted crop dataset
lives on a persistent **Modal Volume**; an L4 GPU function fine-tunes the
ResNet-18 string classifier (``train_string_resolver``) and returns the run dir
(``best.pt`` + ``metrics.json``) as a tarball.

Requires ``pip install modal`` + ``modal token new`` (the user's Modal account).
``torch``/``torchvision``/``opencv`` are training-only; the runtime inference path
needs only ``torch`` to load the checkpoint.

Steps::

    # 1. one-time: upload the extracted dataset to the volume (CLI is fastest)
    modal volume create tabvision-string-dataset
    modal volume put tabvision-string-dataset \
        ~/.tabvision/cache/gaps_string_dataset /dataset

    # 2. smoke (wiring check) then the full run
    modal run scripts/train/string_resolver_modal.py --smoke
    modal run scripts/train/string_resolver_modal.py --epochs 20 --batch 128
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

VOLUME_NAME = "tabvision-string-dataset"
REMOTE_MOUNT = "/vol"
REMOTE_DATASET = f"{REMOTE_MOUNT}/dataset"
REMOTE_OUTPUT = "/output"

# Bring the package + training script into the image so the remote function can
# import ``tabvision.*`` and ``scripts.train.*``. Run from the repo's tabvision/.
_REPO = Path(__file__).resolve().parents[2]  # .../tabvision (the project subdir)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "opencv-python-headless", "numpy<2")
    .add_local_dir(str(_REPO / "tabvision"), "/root/tabvision", copy=True)
    .add_local_dir(str(_REPO / "scripts"), "/root/scripts", copy=True)
    .workdir("/root")
)
app = modal.App("tabvision-string-resolver", image=image)


@app.function(gpu="L4", timeout=60 * 90, volumes={REMOTE_MOUNT: volume})
def finetune(
    *,
    smoke: bool = False,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 3e-4,
    min_peak_ratio: float = 2.0,
    seed: int = 0,
) -> bytes:
    """Run the string-resolver fine-tune on a remote L4; return the run dir tar."""
    import torch

    from scripts.train.train_string_resolver import train_string_resolver

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible to torch")
    volume.reload()
    if smoke:
        epochs, batch = 2, 64

    out = Path(REMOTE_OUTPUT)
    train_string_resolver(
        Path(REMOTE_DATASET),
        out,
        epochs=epochs,
        batch=batch,
        lr=lr,
        min_peak_ratio=min_peak_ratio,
        seed=seed,
        device="cuda",
        num_workers=4,
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(out), arcname="run")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    smoke: bool = False,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 3e-4,
    min_peak_ratio: float = 2.0,
    seed: int = 0,
) -> None:
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))
    out_dir = data_root / "models" / "string_resolver" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[modal] dataset volume={VOLUME_NAME} smoke={smoke} epochs={epochs} -> {out_dir}",
        file=sys.stderr,
    )

    tarball = finetune.remote(
        smoke=smoke, epochs=epochs, batch=batch, lr=lr, min_peak_ratio=min_peak_ratio, seed=seed
    )
    archive = out_dir / "run.tar.gz"
    archive.write_bytes(tarball)
    with tarfile.open(archive) as tar:
        tar.extractall(out_dir)
    best = out_dir / "run" / "best.pt"
    if best.exists():
        stable = data_root / "models" / "string-resolver.pt"
        stable.parent.mkdir(parents=True, exist_ok=True)
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(best.resolve())
        print(f"[modal] best checkpoint linked at {stable}", file=sys.stderr)
    else:
        print(f"[modal] WARNING: no best.pt under {out_dir}", file=sys.stderr)
