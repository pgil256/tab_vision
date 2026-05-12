"""Tests for Modal-backed job storage."""
from app.modal_storage import ModalJobStorage
from app.models import Job


def test_modal_job_storage_serializes_jobs_into_mapping(tmp_path):
    backing_store = {}
    storage = ModalJobStorage(backing_store)
    job = Job.create(video_path=str(tmp_path / "input.mp4"), capo_fret=4)
    job.status = "processing"
    job.current_stage = "extracting_audio"
    job.progress = 0.1

    storage.save(job)

    assert backing_store[job.id] == job.to_record()
    assert storage.get(job.id) == job


def test_modal_job_storage_returns_none_for_missing_job():
    storage = ModalJobStorage({})

    assert storage.get("missing") is None
