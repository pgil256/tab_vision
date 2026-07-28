"""Modal cloud-GPU runner for the Phase E2 fret-keypoint (pose) fine-tune.

Mirrors ``scripts/train/yolo_guitar_obb_modal.py``. Fine-tunes ``yolo11n-pose``
on ``s-workspace-y3mjn/guitar-fret-6pt`` (CC BY 4.0, 926 images, classes
``fret``/``nut``, ``kpt_shape: [6, 3]``), where the six keypoints per instance
are the wire's intersections with the six strings — so the labels supply the
string axis and the fret axis together.

That is the point of E2: ``calibrate.py`` currently *reconstructs* that lattice
by RANSAC-fitting rule-of-18 to noisy OBB centres, and Phase A established the
reconstruction is the binding constraint (fit rate drove the entire +0.151, and
its partial-evidence failures cause the ``118``-class regressions). Go bar:
keypoint-derived fret registration beats ``calibrate.py``'s consensus fit on
wire-sparse clips.

**No horizontal flips.** The export declares ``flip_idx: [0,1,2,3,4,5]`` — an
*identity* mapping — but mirroring an image reverses the six string
intersections, so the correct mapping would be ``[5,4,3,2,1,0]``. With
ultralytics' default ``fliplr=0.5`` every flipped sample would carry silently
transposed string labels. ``fliplr`` is pinned to 0.0 here for the same reason
WS4 banned flips: string identity is encoded in across-neck position.

**License note:** ultralytics is AGPL-3.0; running this script taints the
combined work as AGPL-3.0. See LICENSES.md and DECISIONS.md (2026-05-05
"Phase 3 detector path"). The dataset is CC BY 4.0 — attribution is owed to
**both** ``s-workspace-y3mjn`` and ``b101`` (the image set is almost certainly a
re-annotation of ``b101/guitar-3``: identical 926 count and split).

Modes:
  --upload-dataset  one-time: copy the local dataset into the Modal Volume (56 MB).
  --smoke           3 epochs, batch 4 — wiring check, ~5 min, ~$0.10.
  (default)         100 epochs, batch 16 — E2 fine-tune, ~30-40 min, ~$0.40.

Usage::

    modal run scripts/train/yolo_fret_keypoints_modal.py --upload-dataset
    modal run scripts/train/yolo_fret_keypoints_modal.py --smoke
    modal run scripts/train/yolo_fret_keypoints_modal.py
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

DATASET_SUBDIR = "roboflow-s-workspace-y3mjn-guitar-fret-6pt-v1"


def _local_data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))


def _local_dataset_path() -> Path:
    return _local_data_root() / "datasets" / DATASET_SUBDIR


def _local_output_root() -> Path:
    return _local_data_root() / "models" / "runs"


def _local_stable_weight_link() -> Path:
    return _local_data_root() / "models" / "guitar-yolo-pose-fret6pt.pt"


# ----- remote paths -----

VOLUME_NAME = "tabvision-yolo-fret-6pt"
REMOTE_VOLUME_MOUNT = "/data"
REMOTE_DATASET = f"{REMOTE_VOLUME_MOUNT}/{DATASET_SUBDIR}"
REMOTE_OUTPUT = "/output"
REMOTE_RUN_NAME = "fret-keypoints-finetune"

# ----- Modal image + volume -----

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.from_registry("ultralytics/ultralytics:latest", add_python=None).pip_install(
    "ultralytics>=8.3", "numpy<2", "opencv-python-headless"
)

app = modal.App("tabvision-fret-keypoints", image=image)


@app.function(volumes={REMOTE_VOLUME_MOUNT: volume}, timeout=60 * 30)
def upload_dataset_files(name_to_bytes: dict[str, bytes]) -> int:
    """Write a batch of files into the volume. Idempotent; returns bytes written."""
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


def _normalized_data_yaml(dataset_dir: Path) -> Path:
    """Rewrite the Roboflow data.yaml with absolute split paths.

    Roboflow exports carry ``train: ../train/images``, which resolves relative to
    the *parent* of the yaml and silently points outside the dataset dir. Writing
    absolute paths makes a missing split fail loudly at load time instead of
    training on a surprise directory.
    """
    import yaml

    src = dataset_dir / "data.yaml"
    spec = yaml.safe_load(src.read_text())
    for split, sub in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = dataset_dir / sub / "images"
        if images.is_dir():
            spec[split] = str(images)
        else:
            spec.pop(split, None)
    for required in ("train", "val"):
        if required not in spec:
            raise FileNotFoundError(f"{required} split missing under {dataset_dir}")
    spec.pop("path", None)
    dst = dataset_dir.parent / "data.normalized.yaml"
    dst.write_text(yaml.safe_dump(spec, sort_keys=False))
    return dst


@app.function(gpu="L4", timeout=60 * 90, volumes={REMOTE_VOLUME_MOUNT: volume})
def finetune(
    *,
    smoke: bool = False,
    epochs: int = 100,
    batch: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    base_model: str = "yolo11n-pose.pt",
    seed: int = 0,
) -> bytes:
    """Fine-tune yolo11n-pose on the fret-keypoint set; return the run dir as a tar."""
    import logging

    import torch
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("fret-kpt")

    log.info(
        "torch=%s cuda=%s gpus=%d",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible to torch")

    if smoke:
        epochs = 3
        batch = 4

    volume.reload()
    dataset_dir = Path(REMOTE_DATASET)
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"data.yaml not found under {dataset_dir}; populate the volume first with: "
            "modal run scripts/train/yolo_fret_keypoints_modal.py --upload-dataset"
        )
    data_yaml = _normalized_data_yaml(dataset_dir)
    log.info("normalized data.yaml -> %s\n%s", data_yaml, data_yaml.read_text())

    os.makedirs(REMOTE_OUTPUT, exist_ok=True)
    log.info(
        "training: base=%s epochs=%d batch=%d imgsz=%d lr0=%g seed=%d fliplr=0.0",
        base_model,
        epochs,
        batch,
        img_size,
        lr0,
        seed,
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
        # See the module docstring: flip_idx is an identity mapping, so a
        # horizontal flip would transpose the six string intersections.
        fliplr=0.0,
    )
    log.info("training finished in %.1fs", time.time() - t0)

    metrics = model.val(data=str(data_yaml), split="val")
    log.info(
        "val: box mAP50=%.4f mAP50-95=%.4f | pose mAP50=%.4f mAP50-95=%.4f",
        metrics.box.map50,
        metrics.box.map,
        metrics.pose.map50,
        metrics.pose.map,
    )

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
    epochs: int = 100,
    batch: int = 16,
    img_size: int = 640,
    lr0: float = 0.01,
    base_model: str = "yolo11n-pose.pt",
    seed: int = 0,
) -> None:
    if upload_dataset:
        return _upload_dataset()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _local_output_root() / f"fret6pt-{timestamp}"
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
    print(f"[modal] artifact ({len(tarball) / 1e6:.1f} MB) -> {archive}", file=sys.stderr)

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
    """Upload the local Roboflow keypoint dataset to the persistent Modal Volume."""
    dataset_local = _local_dataset_path()
    if not dataset_local.exists():
        print(
            f"[modal] dataset not found at {dataset_local}\n"
            "        Acquire guitar-fret-6pt (CC BY 4.0) first.",
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
                f"[modal]   chunk {chunks_sent}: {len(chunk)} files, {written / 1e6:.1f} MB",
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

    print(f"[modal] upload complete -> {REMOTE_DATASET}", file=sys.stderr)
