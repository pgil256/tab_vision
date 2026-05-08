"""Modal cloud-GPU runner for the Basic Pitch fine-tune.

Runs `basic_pitch.train.main` on a Modal L4 GPU with our pretrained-weight
loader monkey-patched in. Uploads the GuitarSet TFRecord splits + the
`app/training` package as image layers (Modal dedups across runs), then
downloads the trained checkpoint back to
`tools/outputs/finetune/<timestamp>/`.

Two modes:
  --smoke   5 epochs, batch 2, 5 steps/epoch — validates GPU + image +
            pretrained-load round-trip end-to-end. ~3 min, ~$0.05.
  (default) 20 epochs, full 300-clip train split, batch 8, lr 1e-4 — the
            real Phase-1 fine-tune. ~30-45 min, ~$0.40.

Usage:
    modal run tools/finetune_basic_pitch_modal.py --smoke
    modal run tools/finetune_basic_pitch_modal.py
    modal run tools/finetune_basic_pitch_modal.py --epochs 30 --learning-rate 5e-5
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import time
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TFRECORD_LOCAL = PROJECT_ROOT / "tools" / "outputs" / "tfrecords"
LOAD_PRETRAINED_LOCAL = PROJECT_ROOT / "app" / "training" / "load_pretrained.py"
OUTPUT_LOCAL = PROJECT_ROOT / "tools" / "outputs" / "finetune"

REMOTE_TFRECORDS = "/data/tfrecords"
REMOTE_CODE = "/code"
REMOTE_OUTPUT = "/output"


image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu", add_python=None)
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "basic-pitch==0.4.0",
        "librosa>=0.10",
        "soundfile",
        "numpy<2",
    )
    .add_local_dir(str(TFRECORD_LOCAL), REMOTE_TFRECORDS)
    .add_local_file(str(LOAD_PRETRAINED_LOCAL), f"{REMOTE_CODE}/load_pretrained.py")
)

app = modal.App("tabvision-basic-pitch-finetune", image=image)


@app.function(gpu="L4", timeout=60 * 60)
def finetune(
    *,
    smoke: bool = False,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    no_contours: bool = False,
    freeze_bn: bool = False,
) -> bytes:
    """Run fine-tune on the remote GPU. Returns a tar.gz of the output dir."""
    import logging
    import sys as _sys

    import numpy as np
    import tensorflow as tf

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("finetune")

    gpus = tf.config.list_physical_devices("GPU")
    log.info("TF=%s GPUs=%s", tf.__version__, gpus)
    if not gpus:
        raise RuntimeError("no GPU visible to TensorFlow")

    _sys.path.insert(0, REMOTE_CODE)
    from load_pretrained import load_pretrained_basic_pitch_weights

    from basic_pitch import models as bp_models
    from basic_pitch import train as bp_train

    _orig_model = bp_models.model

    def _model_with_pretrained(*args, **kwargs):
        m = _orig_model(*args, **kwargs)
        summary = load_pretrained_basic_pitch_weights(m, strict=True)
        log.info("pretrained weights loaded: %s", summary)
        if freeze_bn:
            n_frozen = 0
            for layer in m.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                    n_frozen += 1
            log.info("froze %d BatchNormalization layers (running stats + gamma/beta locked)", n_frozen)
        return m

    bp_models.model = _model_with_pretrained
    if hasattr(bp_train, "model"):
        bp_train.model = _model_with_pretrained

    class _NoopVis(tf.keras.callbacks.Callback):
        def __init__(self, *a, **k):
            super().__init__()

    bp_train.VisualizeCallback = _NoopVis

    os.makedirs(REMOTE_OUTPUT, exist_ok=True)

    if smoke:
        epochs = 5
        batch_size = 2
        steps_per_epoch = 5
        validation_steps = 1
        learning_rate = 1e-3
    else:
        # 300 train / 60 val examples (counted on 2026-04-30). basic_pitch's
        # TFRecord pipeline uses .repeat(), so steps_per_epoch=None would
        # iterate forever. Floor-divide so drop_remainder=True doesn't bite.
        steps_per_epoch = 300 // batch_size
        validation_steps = max(1, 60 // batch_size)

    log.info(
        "starting train.main: smoke=%s epochs=%d batch_size=%d lr=%g",
        smoke, epochs, batch_size, learning_rate,
    )
    t0 = time.time()
    bp_train.main(
        source=REMOTE_TFRECORDS,
        output=REMOTE_OUTPUT,
        batch_size=batch_size,
        shuffle_size=64,
        learning_rate=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        size_evaluation_callback_datasets=2,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1.0]),
        no_sonify=True,
        no_contours=no_contours,
        weighted_onset_loss=False,
        positive_onset_weight=0.5,
    )
    log.info("train.main finished in %.1fs", time.time() - t0)

    out_path = Path(REMOTE_OUTPUT)
    artifacts = sorted(p for p in out_path.rglob("*") if p.is_file())
    log.info("output dir contains %d files (top-level: %s)",
             len(artifacts), [p.name for p in out_path.iterdir()])

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REMOTE_OUTPUT, arcname="finetune_output")
    log.info("returning tarball of size %.1fMB", buf.tell() / 1e6)
    return buf.getvalue()


@app.local_entrypoint()
def main(
    smoke: bool = False,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    freeze_bn: bool = False,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_LOCAL / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[modal] timestamp={timestamp}", file=sys.stderr)
    print(f"[modal] output dir: {out_dir}", file=sys.stderr)
    print(f"[modal] smoke={smoke} epochs={epochs} batch_size={batch_size} lr={learning_rate} "
          f"freeze_bn={freeze_bn}",
          file=sys.stderr)

    tarball = finetune.remote(
        smoke=smoke,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_bn=freeze_bn,
    )

    archive = out_dir / "finetune_output.tar.gz"
    archive.write_bytes(tarball)
    print(f"[modal] artifact ({len(tarball)/1e6:.1f}MB) -> {archive}", file=sys.stderr)

    with tarfile.open(archive) as tar:
        tar.extractall(out_dir)
    print(f"[modal] extracted -> {out_dir}/finetune_output/", file=sys.stderr)
