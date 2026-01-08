"""Background job processing orchestration."""
import json
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict

from app.models import Job
from app.storage import JobStorage
from app.audio_pipeline import extract_audio, analyze_pitch
from app.fusion_engine import fuse_audio_only, TabNote


def update_job(job: Job, stage: str, progress: float) -> None:
    """Update job status and progress."""
    job.current_stage = stage
    job.progress = progress
    job.status = "processing"
    job.updated_at = datetime.now(timezone.utc)


def save_result(job: Job, tab_notes: list[TabNote], output_dir: str) -> str:
    """Save TabDocument to JSON file.

    Args:
        job: The job being processed
        tab_notes: List of tab notes to save
        output_dir: Directory to save results

    Returns:
        Path to the saved JSON file
    """
    # Convert TabNotes to frontend-compatible format
    notes_data = []
    for note in tab_notes:
        notes_data.append({
            "id": note.id,
            "timestamp": note.timestamp,
            "string": note.string,
            "fret": note.fret,
            "confidence": note.confidence,
            "confidenceLevel": note.confidence_level,
        })

    tab_document = {
        "notes": notes_data,
        "metadata": {
            "job_id": job.id,
            "capo_fret": job.capo_fret,
        }
    }

    result_path = os.path.join(output_dir, f"{job.id}_result.json")
    with open(result_path, "w") as f:
        json.dump(tab_document, f, indent=2)

    return result_path


def load_result(job: Job) -> dict:
    """Load TabDocument from saved JSON file.

    Args:
        job: The completed job

    Returns:
        TabDocument dictionary

    Raises:
        FileNotFoundError: If result file doesn't exist
    """
    if not job.result_path or not os.path.exists(job.result_path):
        raise FileNotFoundError(f"Result not found for job {job.id}")

    with open(job.result_path) as f:
        return json.load(f)


def process_job(
    job_id: str,
    storage: JobStorage,
    output_dir: str = None
) -> None:
    """Process a job through the audio pipeline.

    Args:
        job_id: ID of the job to process
        storage: JobStorage instance
        output_dir: Directory for temp files and results
    """
    job = storage.get(job_id)
    if job is None:
        return

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    try:
        # Stage 1: Extract audio
        update_job(job, "extracting_audio", 0.1)
        audio_path = os.path.join(output_dir, f"{job.id}_audio.wav")
        extract_audio(job.video_path, audio_path)

        # Stage 2: Analyze with pitch detector
        update_job(job, "analyzing_audio", 0.3)
        detected_notes = analyze_pitch(audio_path)

        # Stage 3: Fuse into tab notes
        update_job(job, "fusing", 0.7)
        tab_notes = fuse_audio_only(detected_notes, job.capo_fret)

        # Stage 4: Save result
        update_job(job, "saving", 0.9)
        result_path = save_result(job, tab_notes, output_dir)
        job.result_path = result_path

        # Complete
        job.status = "completed"
        job.current_stage = "complete"
        job.progress = 1.0
        job.updated_at = datetime.now(timezone.utc)

        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.updated_at = datetime.now(timezone.utc)
