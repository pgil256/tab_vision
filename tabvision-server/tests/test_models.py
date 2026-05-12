# tabvision-server/tests/test_models.py
from app.models import Job

def test_job_creation():
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=0)

    assert job.id is not None
    assert job.status == "pending"
    assert job.video_path == "/uploads/test.mp4"
    assert job.capo_fret == 0
    assert job.progress == 0.0
    assert job.current_stage == "uploading"
    assert job.result_path is None
    assert job.error_message is None

def test_job_to_dict():
    job = Job.create(video_path="/uploads/test.mp4", capo_fret=2)
    data = job.to_dict()

    assert data["id"] == job.id
    assert data["status"] == "pending"
    assert data["progress"] == 0.0
    assert data["current_stage"] == "uploading"


def test_job_with_roi_fields():
    """Job can be created with ROI coordinates."""
    job = Job.create(video_path="/test.mp4", capo_fret=0)
    job.roi_x1 = 0.1
    job.roi_y1 = 0.2
    job.roi_x2 = 0.8
    job.roi_y2 = 0.9

    assert job.roi_x1 == 0.1
    assert job.roi_y1 == 0.2
    assert job.roi_x2 == 0.8
    assert job.roi_y2 == 0.9


def test_job_roi_defaults_to_none():
    """Job ROI fields default to None."""
    job = Job.create(video_path="/test.mp4", capo_fret=0)

    assert job.roi_x1 is None
    assert job.roi_y1 is None
    assert job.roi_x2 is None
    assert job.roi_y2 is None


def test_job_context_fields_round_trip_record():
    """Guided upload context survives durable storage serialization."""
    job = Job.create(
        video_path="/test.mp4",
        capo_fret=3,
        instrument="electric",
        tone="distorted",
        style="fingerstyle",
        accuracy_mode="fast",
    )
    job.roi_x1 = 0.1
    job.roi_y1 = 0.2
    job.roi_x2 = 0.8
    job.roi_y2 = 0.9
    job.result_path = "/results/test.json"
    job.error_message = "example error"

    restored = Job.from_record(job.to_record())

    assert restored.instrument == "electric"
    assert restored.tone == "distorted"
    assert restored.style == "fingerstyle"
    assert restored.accuracy_mode == "fast"
    assert restored.capo_fret == 3
    assert restored.roi_x1 == 0.1
    assert restored.roi_y1 == 0.2
    assert restored.roi_x2 == 0.8
    assert restored.roi_y2 == 0.9
    assert restored.result_path == "/results/test.json"
    assert restored.error_message == "example error"
