"""Tests for processing module."""
import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from app.processing import process_job, save_result, load_result
from app.models import Job
from app.storage import JobStorage
from app.audio_pipeline import DetectedNote
from app.fusion_engine import TabNote


@pytest.fixture
def job_storage():
    """Create a fresh job storage for each test."""
    return JobStorage()


@pytest.fixture
def sample_job(tmp_path):
    """Create a sample job with a fake video file."""
    video_path = tmp_path / "test_video.mp4"
    video_path.write_bytes(b"fake video content")
    return Job.create(video_path=str(video_path), capo_fret=0)


class TestSaveAndLoadResult:
    """Tests for result persistence."""

    def test_save_result_creates_json_file(self, tmp_path, sample_job):
        """save_result should create a JSON file with TabDocument."""
        tab_notes = [
            TabNote(
                id="test-1",
                timestamp=1.0,
                string=1,
                fret=5,
                confidence=0.9,
                confidence_level="high",
                midi_note=69,
            ),
        ]

        result_path = save_result(sample_job, tab_notes, str(tmp_path))

        assert os.path.exists(result_path)
        with open(result_path) as f:
            data = json.load(f)
        assert "notes" in data
        assert len(data["notes"]) == 1
        assert data["notes"][0]["string"] == 1
        assert data["notes"][0]["fret"] == 5

    def test_load_result_reads_saved_data(self, tmp_path, sample_job):
        """load_result should read back saved TabDocument."""
        tab_notes = [
            TabNote(
                id="test-1",
                timestamp=1.0,
                string=1,
                fret=5,
                confidence=0.9,
                confidence_level="high",
                midi_note=69,
            ),
        ]
        result_path = save_result(sample_job, tab_notes, str(tmp_path))
        sample_job.result_path = result_path

        result = load_result(sample_job)

        assert "notes" in result
        assert len(result["notes"]) == 1


class TestProcessJob:
    """Tests for job processing orchestration."""

    @patch('app.processing.extract_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_updates_status_on_success(
        self, mock_fuse, mock_analyze, mock_extract, job_storage, sample_job, tmp_path
    ):
        """Successful processing should update job status to completed."""
        # Setup mocks
        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_analyze.return_value = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.9)
        ]
        mock_fuse.return_value = [
            TabNote(
                id="test-1",
                timestamp=1.0,
                string=1,
                fret=5,
                confidence=0.9,
                confidence_level="high",
                midi_note=69,
            )
        ]

        job_storage.save(sample_job)

        process_job(sample_job.id, job_storage, str(tmp_path))

        job = job_storage.get(sample_job.id)
        assert job.status == "completed"
        assert job.progress == 1.0
        assert job.current_stage == "complete"
        assert job.result_path is not None

    @patch('app.processing.extract_audio')
    def test_process_job_sets_failed_on_error(
        self, mock_extract, job_storage, sample_job, tmp_path
    ):
        """Processing errors should set job status to failed."""
        mock_extract.side_effect = RuntimeError("ffmpeg error")

        job_storage.save(sample_job)

        process_job(sample_job.id, job_storage, str(tmp_path))

        job = job_storage.get(sample_job.id)
        assert job.status == "failed"
        assert "ffmpeg error" in job.error_message

    @patch('app.processing.extract_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_progresses_through_stages(
        self, mock_fuse, mock_analyze, mock_extract, job_storage, sample_job, tmp_path
    ):
        """Processing should progress through defined stages."""
        stages_seen = []

        def track_stage(*args, **kwargs):
            job = job_storage.get(sample_job.id)
            stages_seen.append(job.current_stage)

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_extract.side_effect = lambda *a, **k: (track_stage(), str(tmp_path / "audio.wav"))[1]
        mock_analyze.return_value = []
        mock_analyze.side_effect = lambda *a, **k: (track_stage(), [])[1]
        mock_fuse.return_value = []
        mock_fuse.side_effect = lambda *a, **k: (track_stage(), [])[1]

        job_storage.save(sample_job)

        process_job(sample_job.id, job_storage, str(tmp_path))

        # Should have seen extracting_audio and analyzing_audio stages
        assert "extracting_audio" in stages_seen or "analyzing_audio" in stages_seen
