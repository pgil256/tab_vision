"""Modal GPU runner for the v1 GuitarSet highres audio eval.

Runs the same evaluator as ``scripts/eval/guitarset_audio_eval.py`` on a
Modal L4 GPU, with raw GuitarSet hydrated into a persistent Modal Volume
from the ``taohu/guitarset`` Hugging Face mirror.

Usage from repo root:

    tabvision-server/venv/bin/modal run tabvision/scripts/eval/guitarset_highres_modal.py
    tabvision-server/venv/bin/modal run tabvision/scripts/eval/guitarset_highres_modal.py --limit 3
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal


def _local_repo_root() -> Path:
    """Find the repo root when this file is imported by the local Modal CLI.

    Modal re-imports the script inside the remote container from ``/root``;
    in that context the local source tree is not present, so fall back to
    cwd instead of indexing fixed parents.
    """
    script = Path(__file__).resolve()
    for candidate in (Path.cwd().resolve(), *script.parents):
        if (candidate / "tabvision" / "tabvision").is_dir():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _local_repo_root()
V1_ROOT = REPO_ROOT / "tabvision"
PACKAGE_LOCAL = V1_ROOT / "tabvision"
OUTPUT_LOCAL = REPO_ROOT / "tabvision-server" / "tools" / "outputs"

APP_NAME = "tabvision-guitarset-highres-eval"
VOLUME_NAME = "tabvision-guitarset"
HF_REPO = "taohu/guitarset"
HF_REPO_TYPE = "dataset"
SHARD_FILES = [f"data/train-{i:05d}-of-00005.parquet" for i in range(5)]

REMOTE_CODE = "/code"
REMOTE_VOLUME = "/data"
REMOTE_GUITARSET = f"{REMOTE_VOLUME}/guitarset"
REMOTE_OUTPUT = "/output"
SENTINEL = f"{REMOTE_GUITARSET}/.complete"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python=None)
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "git+https://github.com/xavriley/hf_midi_transcription.git@96f6797881e9497cbfc8f8e5deccea9c1f2f7adc",
        "huggingface-hub>=0.16.0",
        "librosa>=0.10.0",
        "mir_eval>=0.7",
        "numpy<2",
        "pretty_midi>=0.2.10",
        "pyarrow>=15.0",
        "safetensors>=0.3.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
    )
    .add_local_dir(str(PACKAGE_LOCAL), f"{REMOTE_CODE}/tabvision")
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="L4", timeout=60 * 120, volumes={REMOTE_VOLUME: volume})
def run_highres_eval(
    limit: int | None = None,
    force_data_refresh: bool = False,
    position_prior: str = "guitarset-train",
    melodic_prior: bool = False,
) -> dict:
    """Run validation eval on a GPU and return report artifacts."""
    import logging
    import os
    import shutil
    import time
    from pathlib import Path as RemotePath

    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("guitarset-highres-eval")

    log.info(
        "torch=%s cuda=%s gpus=%d",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible to torch")

    sys.path.insert(0, REMOTE_CODE)
    _ensure_guitarset_data(force=force_data_refresh, log=log)

    from tabvision.eval.guitarset_audio import run_eval, write_report

    os.makedirs(REMOTE_OUTPUT, exist_ok=True)
    t0 = time.time()
    results, summary = run_eval(
        backend_name="highres",
        data_home=REMOTE_GUITARSET,
        split="validation",
        limit=limit,
        position_prior_name=position_prior,
        backend_kwargs={"device": "cuda"},
        melodic_prior_enabled=melodic_prior,
    )
    log.info(
        "eval finished in %.1fs: tracks=%d onset=%.4f pitch=%.4f tab=%.4f",
        time.time() - t0,
        summary.n_tracks,
        summary.micro_onset.f1,
        summary.micro_pitch.f1,
        summary.micro_tab.f1,
    )

    md_path, csv_path = write_report(results, summary, output_dir=REMOTE_OUTPUT)
    # Keep the remote output directory tidy between calls.
    for child in RemotePath(REMOTE_OUTPUT).iterdir():
        if child not in {md_path, csv_path}:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    return {
        "md_name": md_path.name,
        "md": md_path.read_bytes(),
        "csv_name": csv_path.name,
        "csv": csv_path.read_bytes(),
        "tracks": summary.n_tracks,
        "onset_f1": summary.micro_onset.f1,
        "pitch_f1": summary.micro_pitch.f1,
        "tab_f1": summary.micro_tab.f1,
    }


def _ensure_guitarset_data(*, force: bool, log) -> None:
    """Hydrate raw GuitarSet JAMS/WAV files into the mounted Modal volume."""
    from pathlib import Path as RemotePath

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    root = RemotePath(REMOTE_GUITARSET)
    annotation_dir = root / "annotation"
    audio_dir = root / "audio_mono-mic"
    sentinel = RemotePath(SENTINEL)
    if sentinel.exists() and not force:
        log.info("GuitarSet volume already hydrated at %s", root)
        return

    annotation_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    n_tracks = 0
    for shard in SHARD_FILES:
        log.info("downloading %s from %s", shard, HF_REPO)
        local = hf_hub_download(repo_id=HF_REPO, filename=shard, repo_type=HF_REPO_TYPE)
        table = pq.read_table(local, columns=["track_id", "jams", "audio_mic"])
        log.info("  shard rows: %d", len(table))
        for row in table.to_pylist():
            track_id = row["track_id"]
            (annotation_dir / f"{track_id}.jams").write_text(row["jams"], encoding="utf-8")
            (audio_dir / f"{track_id}_mic.wav").write_bytes(row["audio_mic"]["bytes"])
            n_tracks += 1
    sentinel.write_text(f"tracks={n_tracks}\n", encoding="utf-8")
    volume.commit()
    log.info(
        "hydrated %d tracks into %s (annotation=%d audio=%d)",
        n_tracks,
        root,
        len(list(annotation_dir.glob("*.jams"))),
        len(list(audio_dir.glob("*_mic.wav"))),
    )


@app.local_entrypoint()
def main(
    limit: int | None = None,
    force_data_refresh: bool = False,
    position_prior: str = "guitarset-train",
    melodic_prior: bool = False,
) -> None:
    print(
        f"[modal] app={APP_NAME} gpu=L4 limit={limit} "
        f"position_prior={position_prior} melodic_prior={melodic_prior}",
        file=sys.stderr,
    )
    payload = run_highres_eval.remote(
        limit=limit,
        force_data_refresh=force_data_refresh,
        position_prior=position_prior,
        melodic_prior=melodic_prior,
    )

    OUTPUT_LOCAL.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_LOCAL / payload["md_name"]
    csv_path = OUTPUT_LOCAL / payload["csv_name"]
    md_path.write_bytes(payload["md"])
    csv_path.write_bytes(payload["csv"])
    print(
        "[modal] tracks={tracks} onset_f1={onset:.4f} pitch_f1={pitch:.4f} tab_f1={tab:.4f}".format(
            tracks=payload["tracks"],
            onset=payload["onset_f1"],
            pitch=payload["pitch_f1"],
            tab=payload["tab_f1"],
        ),
        file=sys.stderr,
    )
    print(f"[modal] report={md_path}", file=sys.stderr)
    print(f"[modal] csv={csv_path}", file=sys.stderr)
