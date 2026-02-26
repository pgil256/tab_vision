"""Background job processing orchestration."""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict

from app.models import Job
from app.storage import JobStorage
from app.audio_pipeline import (
    extract_audio, analyze_pitch, AudioAnalysisConfig, detect_note_onsets
)
from app.fusion_engine import fuse_audio_only, fuse_audio_video, TabNote, FusionConfig
from app.video_pipeline import analyze_video_at_timestamps, VideoAnalysisConfig
from app.fretboard_detection import (
    detect_fretboard_from_video, track_fretboard_temporal, FretboardDetectionConfig
)

logger = logging.getLogger(__name__)


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
        note_data = {
            "id": note.id,
            "timestamp": note.timestamp,
            "string": note.string,
            "fret": note.fret,
            "confidence": note.confidence,
            "confidenceLevel": note.confidence_level,
            "isEdited": False,
        }

        # Add enhanced attributes if present
        if note.end_time is not None:
            note_data["endTime"] = note.end_time
        if note.technique:
            note_data["technique"] = note.technique
        if note.is_part_of_chord:
            note_data["isPartOfChord"] = True
            note_data["chordId"] = note.chord_id
        if note.video_matched:
            note_data["videoMatched"] = True

        notes_data.append(note_data)

    # Calculate duration from notes
    if tab_notes:
        # Use end_time if available, otherwise timestamp
        max_time = max(
            (n.end_time if n.end_time else n.timestamp for n in tab_notes),
            default=0
        )
    else:
        max_time = 0

    # Calculate statistics for metadata
    total_notes = len(tab_notes)
    high_conf = sum(1 for n in tab_notes if n.confidence_level == "high")
    med_conf = sum(1 for n in tab_notes if n.confidence_level == "medium")
    low_conf = sum(1 for n in tab_notes if n.confidence_level == "low")
    video_confirmed = sum(1 for n in tab_notes if n.video_matched)

    tab_document = {
        "id": job.id,
        "createdAt": job.created_at.isoformat(),
        "duration": max_time + 1,  # Add 1 second buffer
        "capoFret": job.capo_fret,
        "tuning": ["E", "B", "G", "D", "A", "E"],  # Standard tuning
        "notes": notes_data,
        # Processing metadata
        "metadata": {
            "totalNotes": total_notes,
            "highConfidenceNotes": high_conf,
            "mediumConfidenceNotes": med_conf,
            "lowConfidenceNotes": low_conf,
            "videoConfirmedNotes": video_confirmed,
            "averageConfidence": sum(n.confidence for n in tab_notes) / total_notes if total_notes > 0 else 0,
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
    output_dir: str = None,
    audio_config: AudioAnalysisConfig = None,
    video_config: VideoAnalysisConfig = None,
    fretboard_config: FretboardDetectionConfig = None,
    fusion_config: FusionConfig = None
) -> None:
    """Process a job through the audio and video analysis pipelines.

    Args:
        job_id: ID of the job to process
        storage: JobStorage instance
        output_dir: Directory for temp files and results
        audio_config: Audio analysis configuration
        video_config: Video analysis configuration
        fretboard_config: Fretboard detection configuration
        fusion_config: Fusion engine configuration
    """
    job = storage.get(job_id)
    if job is None:
        return

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Use default configs if not provided
    if audio_config is None:
        audio_config = AudioAnalysisConfig()
    if video_config is None:
        video_config = VideoAnalysisConfig()
    if fusion_config is None:
        fusion_config = FusionConfig()

    try:
        # Stage 1: Extract audio
        update_job(job, "extracting_audio", 0.1)
        audio_path = os.path.join(output_dir, f"{job.id}_audio.wav")
        extract_audio(job.video_path, audio_path)

        # Stage 2: Analyze with pitch detector
        update_job(job, "analyzing_audio", 0.3)
        detected_notes = analyze_pitch(audio_path, audio_config)
        logger.info(f"Audio analysis: detected {len(detected_notes)} notes")

        # Stage 3: Analyze video (optional - graceful fallback if fails)
        update_job(job, "analyzing_video", 0.5)
        video_observations = {}
        fretboard = None

        try:
            # Get onset timestamps from detected notes
            timestamps = detect_note_onsets(detected_notes)
            logger.info(f"Detected {len(timestamps)} note onsets for video analysis")

            if timestamps:
                # Detect fretboard geometry using multiple frames for robustness
                fretboard = detect_fretboard_from_video(
                    job.video_path,
                    num_sample_frames=5
                )

                if fretboard:
                    logger.info(
                        f"Fretboard detected with confidence {fretboard.detection_confidence:.2f}, "
                        f"{len(fretboard.fret_positions)} frets"
                    )

                    # For longer videos, track fretboard across time
                    if len(timestamps) > 10:
                        fretboard_timeline = track_fretboard_temporal(
                            job.video_path,
                            timestamps[::5],  # Sample every 5th onset
                            fretboard
                        )
                        # Use best fretboard from timeline if available
                        if fretboard_timeline:
                            best_fb = max(
                                fretboard_timeline.values(),
                                key=lambda fb: fb.detection_confidence
                            )
                            if best_fb.detection_confidence > fretboard.detection_confidence:
                                fretboard = best_fb
                                logger.info(
                                    f"Updated fretboard from temporal tracking: "
                                    f"confidence {fretboard.detection_confidence:.2f}"
                                )

                    # Analyze hand positions at note onset times
                    video_observations = analyze_video_at_timestamps(
                        job.video_path, timestamps, video_config
                    )
                    logger.info(
                        f"Video analysis: {len(video_observations)} hand observations "
                        f"from {len(timestamps)} onsets "
                        f"({len(video_observations)/len(timestamps)*100:.0f}% detection rate)"
                    )
                else:
                    logger.info("No fretboard detected, using audio-only mode")

        except Exception as video_err:
            # Video analysis is optional - log and continue
            logger.warning(f"Video analysis failed: {video_err}, using audio-only mode")

        # Stage 4: Fuse into tab notes
        update_job(job, "fusing", 0.7)

        # Quality gate: only use video fusion if fretboard and video are reliable
        use_video = False
        if fretboard and video_observations:
            detection_rate = len(video_observations) / len(timestamps) if timestamps else 0
            if (fretboard.detection_confidence > 0.5 and detection_rate > 0.2):
                use_video = True
                logger.info(
                    f"Video quality gate passed: fretboard_conf={fretboard.detection_confidence:.2f}, "
                    f"detection_rate={detection_rate:.0%}"
                )
            else:
                logger.info(
                    f"Video quality gate failed: fretboard_conf={fretboard.detection_confidence:.2f}, "
                    f"detection_rate={detection_rate:.0%} — using audio-only"
                )

        if use_video:
            # Use audio + video fusion
            tab_notes = fuse_audio_video(
                detected_notes,
                video_observations,
                fretboard,
                job.capo_fret,
                fusion_config
            )
            logger.info(
                f"Fusion complete: {len(tab_notes)} tab notes, "
                f"{sum(1 for n in tab_notes if n.video_matched)} video-confirmed"
            )
        else:
            # Fall back to audio-only
            tab_notes = fuse_audio_only(detected_notes, job.capo_fret, fusion_config)
            logger.info(f"Audio-only fusion: {len(tab_notes)} tab notes")

        # Log confidence distribution
        if tab_notes:
            high_conf = sum(1 for n in tab_notes if n.confidence_level == "high")
            med_conf = sum(1 for n in tab_notes if n.confidence_level == "medium")
            low_conf = sum(1 for n in tab_notes if n.confidence_level == "low")
            logger.info(
                f"Confidence distribution: high={high_conf}, medium={med_conf}, low={low_conf}"
            )

        # Stage 5: Save result
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
        logger.exception(f"Job {job_id} failed")
        job.status = "failed"
        job.error_message = str(e)
        job.updated_at = datetime.now(timezone.utc)
