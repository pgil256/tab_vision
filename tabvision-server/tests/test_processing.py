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

    def test_save_result_creates_missing_output_directory(self, tmp_path, sample_job):
        """save_result should create the configured result directory."""
        output_dir = tmp_path / "missing" / "results"

        result_path = save_result(sample_job, [], str(output_dir))

        assert os.path.exists(result_path)
        assert output_dir.exists()


class TestProcessJob:
    """Tests for job processing orchestration."""

    @patch('app.processing.quantize_notes')
    @patch('app.processing.detect_muted_notes')
    @patch('app.processing.extract_audio')
    @patch('app.processing.preprocess_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.detect_with_ensemble')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_persists_progress_and_terminal_state(
        self, mock_fuse, mock_ensemble, mock_analyze, mock_preprocess, mock_extract,
        mock_muted, mock_quantize, sample_job, tmp_path
    ):
        """Processing should save every progress update for remote pollers."""
        class RecordingStorage(JobStorage):
            def __init__(self):
                super().__init__()
                self.snapshots = []

            def save(self, job):
                self.snapshots.append((job.status, job.current_stage, job.progress))
                super().save(job)

        recording_storage = RecordingStorage()
        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_preprocess.return_value = str(tmp_path / "audio_preprocessed.wav")
        detected = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.9)
        ]
        mock_analyze.return_value = detected
        mock_ensemble.return_value = detected
        mock_muted.return_value = []
        tab_notes = [
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
        mock_fuse.return_value = tab_notes
        mock_quantize.return_value = tab_notes

        recording_storage.save(sample_job)
        recording_storage.snapshots.clear()

        process_job(sample_job.id, recording_storage, str(tmp_path))

        assert ("processing", "extracting_audio", 0.1) in recording_storage.snapshots
        assert ("processing", "analyzing_audio", 0.3) in recording_storage.snapshots
        assert ("processing", "saving", 0.9) in recording_storage.snapshots
        assert ("completed", "complete", 1.0) in recording_storage.snapshots

    @patch('app.processing.quantize_notes')
    @patch('app.processing.detect_muted_notes')
    @patch('app.processing.extract_audio')
    @patch('app.processing.preprocess_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.detect_with_ensemble')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_runs_result_saved_hook_before_completed_save(
        self, mock_fuse, mock_ensemble, mock_analyze, mock_preprocess, mock_extract,
        mock_muted, mock_quantize, sample_job, tmp_path
    ):
        """Remote file commits should happen before completed status is visible."""
        events = []

        class RecordingStorage(JobStorage):
            def save(self, job):
                events.append(("save", job.status, job.current_stage))
                super().save(job)

        recording_storage = RecordingStorage()
        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_preprocess.return_value = str(tmp_path / "audio_preprocessed.wav")
        mock_analyze.return_value = []
        mock_ensemble.return_value = []
        mock_muted.return_value = []
        mock_fuse.return_value = []
        mock_quantize.return_value = []

        recording_storage.save(sample_job)
        events.clear()

        process_job(
            sample_job.id,
            recording_storage,
            str(tmp_path),
            result_saved_hook=lambda: events.append(("result_saved",)),
        )

        result_saved_index = events.index(("result_saved",))
        completed_index = events.index(("save", "completed", "complete"))
        assert result_saved_index < completed_index

    @patch('app.processing.quantize_notes')
    @patch('app.processing.detect_muted_notes')
    @patch('app.processing.extract_audio')
    @patch('app.processing.preprocess_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.detect_with_ensemble')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_updates_status_on_success(
        self, mock_fuse, mock_ensemble, mock_analyze, mock_preprocess, mock_extract,
        mock_muted, mock_quantize, job_storage, sample_job, tmp_path
    ):
        """Successful processing should update job status to completed."""
        # Setup mocks
        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_preprocess.return_value = str(tmp_path / "audio_preprocessed.wav")
        detected = [
            DetectedNote(start_time=1.0, end_time=1.5, midi_note=69, confidence=0.9)
        ]
        mock_analyze.return_value = detected
        mock_ensemble.return_value = detected
        mock_muted.return_value = []
        tab_notes = [
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
        mock_fuse.return_value = tab_notes
        mock_quantize.return_value = tab_notes

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

    @patch('app.processing.quantize_notes')
    @patch('app.processing.detect_muted_notes')
    @patch('app.processing.extract_audio')
    @patch('app.processing.preprocess_audio')
    @patch('app.processing.analyze_pitch')
    @patch('app.processing.detect_with_ensemble')
    @patch('app.processing.fuse_audio_only')
    def test_process_job_progresses_through_stages(
        self, mock_fuse, mock_ensemble, mock_analyze, mock_preprocess, mock_extract,
        mock_muted, mock_quantize, job_storage, sample_job, tmp_path
    ):
        """Processing should progress through defined stages."""
        stages_seen = []

        def track_stage(*args, **kwargs):
            job = job_storage.get(sample_job.id)
            stages_seen.append(job.current_stage)

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_extract.side_effect = lambda *a, **k: (track_stage(), str(tmp_path / "audio.wav"))[1]
        mock_preprocess.return_value = str(tmp_path / "audio_preprocessed.wav")
        mock_analyze.return_value = []
        mock_analyze.side_effect = lambda *a, **k: (track_stage(), [])[1]
        mock_ensemble.return_value = []
        mock_muted.return_value = []
        mock_fuse.return_value = []
        mock_fuse.side_effect = lambda *a, **k: (track_stage(), [])[1]
        mock_quantize.return_value = []

        job_storage.save(sample_job)

        process_job(sample_job.id, job_storage, str(tmp_path))

        # Should have seen extracting_audio and analyzing_audio stages
        assert "extracting_audio" in stages_seen or "analyzing_audio" in stages_seen
