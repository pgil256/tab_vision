"""Tests for JobStorage."""
from app.storage import JobStorage
from app.models import Job


def test_storage_save_and_get():
    storage = JobStorage()
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=0)

    storage.save(job)
    retrieved = storage.get(job.id)

    assert retrieved is not None
    assert retrieved.id == job.id


def test_storage_get_nonexistent():
    storage = JobStorage()
    retrieved = storage.get("nonexistent-id")

    assert retrieved is None
