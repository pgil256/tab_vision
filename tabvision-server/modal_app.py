"""Modal deployment entrypoint for the TabVision Flask API."""
from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "tabvision-api"
DATA_VOLUME_NAME = "tabvision-prod-data"
JOBS_DICT_NAME = "tabvision-prod-jobs"

SERVER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SERVER_ROOT.parent
APP_LOCAL = SERVER_ROOT / "app"
TABVISION_PACKAGE_LOCAL = REPO_ROOT / "tabvision" / "tabvision"

REMOTE_CODE = "/code"
REMOTE_TABVISION_PACKAGE = f"{REMOTE_CODE}/tabvision"
REMOTE_DATA = "/data"
REMOTE_UPLOADS = f"{REMOTE_DATA}/uploads"
REMOTE_RESULTS = f"{REMOTE_DATA}/results"
REMOTE_HF_HOME = f"{REMOTE_DATA}/huggingface"
REMOTE_TORCH_HOME = f"{REMOTE_DATA}/torch"

FRONTEND_URL = "https://tabvision.patbuilds.dev"
MAX_WORKER_CONTAINERS = 2

WORKER_REQUIREMENTS = [
    "flask==3.0.0",
    "flask-cors==4.0.0",
    "numpy>=1.24.0",
    "librosa>=0.10.0",
    "resampy>=0.4.3",
    "soundfile>=0.12.0",
    "scipy>=1.10.0",
    "pretty_midi>=0.2.10",
    "torch>=2.0.0",
    "torchlibrosa>=0.1.0",
    "huggingface-hub>=0.16.0",
    "safetensors>=0.3.0",
    "mido>=1.3.2",
    "hf-midi-transcription @ git+https://github.com/xavriley/hf_midi_transcription.git@96f6797881e9497cbfc8f8e5deccea9c1f2f7adc",
]


data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
job_records = modal.Dict.from_name(JOBS_DICT_NAME, create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("flask==3.0.0", "flask-cors==4.0.0")
    .add_local_dir(str(APP_LOCAL), f"{REMOTE_CODE}/app")
)

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libsndfile1")
    .pip_install(*WORKER_REQUIREMENTS)
    .add_local_dir(str(APP_LOCAL), f"{REMOTE_CODE}/app")
    .add_local_dir(str(TABVISION_PACKAGE_LOCAL), REMOTE_TABVISION_PACKAGE)
)

app = modal.App(APP_NAME)


@app.function(
    image=worker_image,
    gpu="L4",
    timeout=60 * 60,
    max_containers=MAX_WORKER_CONTAINERS,
    volumes={REMOTE_DATA: data_volume},
)
def process_job_modal(job_id: str) -> None:
    import os
    import sys

    sys.path.insert(0, REMOTE_CODE)
    os.makedirs(REMOTE_RESULTS, exist_ok=True)
    os.makedirs(REMOTE_HF_HOME, exist_ok=True)
    os.makedirs(REMOTE_TORCH_HOME, exist_ok=True)
    os.environ.setdefault("HF_HOME", REMOTE_HF_HOME)
    os.environ.setdefault("TORCH_HOME", REMOTE_TORCH_HOME)
    os.environ.setdefault("TABVISION_PIPELINE", "v1")
    os.environ.setdefault("TABVISION_AUDIO_BACKEND", "highres")
    os.environ.setdefault("TABVISION_FALLBACK_AUDIO_BACKEND", "none")
    os.environ.setdefault("TABVISION_POSITION_PRIOR", "guitarset-v1")
    os.environ.setdefault("TABVISION_VIDEO_ENABLED", "false")
    os.environ.setdefault("TABVISION_MELODIC_PRIOR_ENABLED", "false")
    os.environ.setdefault("TABVISION_ACCURACY_MODE", "accurate")

    from app.modal_storage import ModalJobStorage
    from app.processing import process_job

    data_volume.reload()
    process_job(
        job_id,
        ModalJobStorage(job_records),
        REMOTE_RESULTS,
        result_saved_hook=data_volume.commit,
    )
    data_volume.commit()


@app.function(
    image=web_image,
    timeout=300,
    volumes={REMOTE_DATA: data_volume},
)
@modal.wsgi_app()
def flask_app():
    import os
    import sys

    sys.path.insert(0, REMOTE_CODE)
    os.environ["FRONTEND_URL"] = FRONTEND_URL
    os.makedirs(REMOTE_UPLOADS, exist_ok=True)
    os.makedirs(REMOTE_RESULTS, exist_ok=True)

    from app import create_app
    from app.modal_storage import ModalJobStorage

    storage = ModalJobStorage(job_records)

    def dispatch_job(job_id: str, _storage, _results_folder: str) -> None:
        process_job_modal.spawn(job_id)

    return create_app({
        "UPLOAD_FOLDER": REMOTE_UPLOADS,
        "RESULTS_FOLDER": REMOTE_RESULTS,
        "JOB_STORAGE": storage,
        "JOB_DISPATCHER": dispatch_job,
        "UPLOAD_SAVED_HOOK": data_volume.commit,
        "RESULTS_RELOAD_HOOK": data_volume.reload,
        "PREWARM_ML": False,
    })
