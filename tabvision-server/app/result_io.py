"""Lightweight result-file loading helpers."""
import json
import os

from app.models import Job


def load_result(job: Job) -> dict:
    """Load a completed job result without importing the processing pipeline."""
    if not job.result_path or not os.path.exists(job.result_path):
        raise FileNotFoundError(f"Result not found for job {job.id}")

    with open(job.result_path) as f:
        return json.load(f)
