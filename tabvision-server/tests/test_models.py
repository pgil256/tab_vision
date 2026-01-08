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
