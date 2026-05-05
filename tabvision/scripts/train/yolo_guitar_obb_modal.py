"""Modal cloud-GPU runner for the YOLO-OBB fretboard-parts fine-tune.

Mirrors the pattern in ``tabvision-server/tools/finetune_basic_pitch_modal.py``
(v0 Phase-1 audio fine-tune). Runs ``ultralytics.YOLO.train`` on a Modal L4
GPU. The Roboflow dataset lives on a persistent **Modal Volume** so we
don't re-upload 230 MB every run — populate it once via
``--upload-dataset``, then training runs mount the volume read-only.

**License note:** ultralytics is AGPL-3.0; running this script taints the
combined work as AGPL-3.0. See LICENSES.md and DECISIONS.md
(2026-05-05 "Phase 3 detector path").

Modes:
  --upload-dataset  one-time: copy the local dataset directory into the
                    Modal Volume. ~30s on a fast connection.
  --smoke           3 epochs, batch 4 — wiring check, ~5 min, ~$0.10.
  (default)         50 epochs, batch 16 — Phase-3 fine-tune, ~30 min, ~$0.40.

Usage::

    modal run scripts/train/yolo_guitar_obb_modal.py --upload-dataset
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

# ----- local paths (only relevant in the local entrypoint) -----


def _local_data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))


def _local_dataset_path() -> Path:
    return _local_data_root() / "datasets" / "roboflow-b101-guitar-3-4efcd-v2"


def _local_output_root() -> Path:
    return _local_data_root() / "models" / "runs"


def _local_stable_weight_link() -> Path:
    return _local_data_root() / "models" / "guitar-yolo-obb-finetuned.pt"

# ----- remote paths -----

VOLUME_NAME = "tabvision-yolo-guitar-3"
DATASET_SUBDIR = "roboflow-b101-guitar-3-4efcd-v2"
REMOTE_VOLUME_MOUNT = "/data"
REMOTE_DATASET = f"{REMOTE_VOLUME_MOUNT}/{DATASET_SUBDIR}"
REMOTE_OUTPUT = "/output"
REMOTE_RUN_NAME = "guitar-obb-finetune"

# ----- Modal image + volume -----

# Persistent volume holds the dataset. Populate once via --upload-dataset;
# subsequent training runs mount it read-only.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("ultralytics/ultralytics:latest", add_python=None)
    .pip_install("ultralytics>=8.3", "numpy<2", "opencv-python-headless")
)

app = modal.App("tabvision-yolo-obb-finetune", image=image)


@app.function(volumes={REMOTE_VOLUME_MOUNT: volume}, timeout=60 * 30)
def upload_dataset_files(name_to_bytes: dict[str, bytes]) -> int:
    """Upload-side helper: write a batch of files into the volume.

    Called from the local entrypoint with chunks of files keyed by the
    relative path under DATASET_SUBDIR/. Idempotent — overwrites on
    repeat. Returns the number of bytes written.
    """
    total = 0
    base = Path(REMOTE_VOLUME_MOUNT) / DATASET_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    for rel, data in name_to_bytes.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        total += len(data)
    volume.commit()
    return total


@app.function(gpu="L4", timeout=60 * 60, volumes={REMOTE_VOLUME_MOUNT: volume})
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

    # Volume is mounted read-only fresh each invocation; rehydrate metadata.
    volume.reload()
    data_yaml = Path(REMOTE_DATASET) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml}; populate the volume "
            "first with: modal run scripts/train/yolo_guitar_obb_modal.py --upload-dataset"
        )

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
    upload_dataset: bool = False,
    smoke: bool = False,
    epochs: int = 50,
    batch: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    base_model: str = "yolo11n-obb.pt",
    seed: int = 0,
) -> None:
    if upload_dataset:
        return _upload_dataset()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _local_output_root() / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[modal] timestamp={timestamp}\n"
        f"[modal] volume: {VOLUME_NAME} (populate with --upload-dataset first)\n"
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
        stable = _local_stable_weight_link()
        stable.parent.mkdir(parents=True, exist_ok=True)
        if stable.exists() or stable.is_symlink():
            stable.unlink()
        stable.symlink_to(best.resolve())
        print(f"[modal] best weights linked at {stable}", file=sys.stderr)
    else:
        print(f"[modal] WARNING: no best.pt at {best}", file=sys.stderr)


def _upload_dataset() -> None:
    """Upload the local Roboflow dataset to the persistent Modal Volume.

    Streams files in chunks (~50 MB each) to avoid huge single RPCs and
    to make progress visible. Idempotent: overwrites on re-run.

    Note: in practice ``modal volume put -f tabvision-yolo-guitar-3 <local>
    /<remote>`` from the CLI is faster than this Python loop (no per-chunk
    container cold-start) — kept for completeness.
    """
    dataset_local = _local_dataset_path()
    if not dataset_local.exists():
        print(
            f"[modal] dataset not found at {dataset_local}\n"
            "        Run: python -m scripts.acquire.datasets roboflow-guitar "
            "--project guitar-3-4efcd",
            file=sys.stderr,
        )
        sys.exit(2)

    files = sorted(p for p in dataset_local.rglob("*") if p.is_file())
    print(f"[modal] uploading {len(files)} files to volume {VOLUME_NAME!r}", file=sys.stderr)

    chunk_max_bytes = 50 * 1024 * 1024
    chunk: dict[str, bytes] = {}
    chunk_size = 0
    total_bytes = 0
    chunks_sent = 0

    for f in files:
        rel = f.relative_to(dataset_local).as_posix()
        data = f.read_bytes()
        chunk[rel] = data
        chunk_size += len(data)
        if chunk_size >= chunk_max_bytes:
            written = upload_dataset_files.remote(chunk)
            total_bytes += written
            chunks_sent += 1
            print(
                f"[modal]   chunk {chunks_sent}: {len(chunk)} files, "
                f"{written / 1e6:.1f} MB; cumulative {total_bytes / 1e6:.1f} MB",
                file=sys.stderr,
            )
            chunk = {}
            chunk_size = 0

    if chunk:
        written = upload_dataset_files.remote(chunk)
        total_bytes += written
        chunks_sent += 1
        print(
            f"[modal]   final chunk: {len(chunk)} files, {written / 1e6:.1f} MB; "
            f"total {total_bytes / 1e6:.1f} MB across {chunks_sent} chunks",
            file=sys.stderr,
        )

    print(
        f"[modal] upload complete. Volume contains the dataset at "
        f"{REMOTE_DATASET} for subsequent runs.",
        file=sys.stderr,
    )
