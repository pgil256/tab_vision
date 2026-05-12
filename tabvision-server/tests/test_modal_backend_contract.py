"""Contracts needed by the Modal production backend."""
import io
import json
from datetime import datetime, timezone

from app import create_app
from app.models import Job
from app.storage import JobStorage


class SpyStorage(JobStorage):
    def __init__(self):
        super().__init__()
        self.saved_ids = []

    def save(self, job):
        self.saved_ids.append(job.id)
        super().save(job)


def test_job_record_round_trip_preserves_durable_fields(tmp_path):
    job = Job.create(video_path=str(tmp_path / "input.mp4"), capo_fret=3)
    job.status = "failed"
    job.progress = 0.7
    job.current_stage = "analyzing_audio"
    job.result_path = str(tmp_path / "result.json")
    job.error_message = "boom"
    job.roi_x1 = 0.1
    job.roi_y1 = 0.2
    job.roi_x2 = 0.8
    job.roi_y2 = 0.9

    record = job.to_record()
    restored = Job.from_record(record)

    assert record["created_at"] == job.created_at.isoformat()
    assert record["updated_at"] == job.updated_at.isoformat()
    assert restored == job


def test_routes_use_configured_storage_and_dispatcher(tmp_path):
    storage = SpyStorage()
    dispatched = []
    hooks = []

    def dispatcher(job_id, job_storage, results_folder):
        dispatched.append((job_id, job_storage, results_folder))

    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "RESULTS_FOLDER": str(tmp_path / "results"),
        "JOB_STORAGE": storage,
        "JOB_DISPATCHER": dispatcher,
        "UPLOAD_SAVED_HOOK": lambda: hooks.append("upload_saved"),
        "PREWARM_ML": False,
    })

    with app.test_client() as client:
        response = client.post(
            "/jobs",
            data={
                "video": (io.BytesIO(b"fake video content"), "test.mp4"),
                "capo_fret": "2",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 201
        job_id = response.get_json()["job_id"]
        stored_job = storage.get(job_id)
        assert stored_job is not None
        assert stored_job.capo_fret == 2
        assert hooks == ["upload_saved"]
        assert dispatched == [(job_id, storage, str(tmp_path / "results"))]

        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.get_json()["id"] == job_id


def test_result_route_uses_configured_storage_and_result_loader(tmp_path):
    storage = JobStorage()
    job = Job.create(video_path=str(tmp_path / "input.mp4"), capo_fret=0)
    job.status = "completed"
    job.result_path = str(tmp_path / "result.json")
    storage.save(job)

    calls = []
    hooks = []

    def result_loader(loaded_job):
        calls.append(loaded_job.id)
        return {"notes": [], "metadata": {"source": "test"}}

    app = create_app({
        "TESTING": True,
        "JOB_STORAGE": storage,
        "RESULT_LOADER": result_loader,
        "RESULTS_RELOAD_HOOK": lambda: hooks.append("results_reloaded"),
        "PREWARM_ML": False,
    })

    with app.test_client() as client:
        response = client.get(f"/jobs/{job.id}/result")

        assert response.status_code == 200
        assert response.get_json()["metadata"]["source"] == "test"
        assert hooks == ["results_reloaded"]
        assert calls == [job.id]


def test_job_from_record_accepts_iso_datetimes():
    record = {
        "id": "job-1",
        "status": "processing",
        "created_at": "2026-05-08T12:00:00+00:00",
        "updated_at": "2026-05-08T12:01:00+00:00",
        "video_path": "/data/uploads/job-1.mp4",
        "capo_fret": 1,
        "progress": 0.5,
        "current_stage": "analyzing_video",
        "result_path": None,
        "error_message": None,
        "roi_x1": None,
        "roi_y1": None,
        "roi_x2": None,
        "roi_y2": None,
    }
    expected = {
        **record,
        "instrument": "acoustic",
        "tone": "clean",
        "style": "mixed",
        "accuracy_mode": "accurate",
    }

    job = Job.from_record(record)

    assert job.created_at == datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    assert job.updated_at == datetime(2026, 5, 8, 12, 1, tzinfo=timezone.utc)
    assert job.to_record() == expected
