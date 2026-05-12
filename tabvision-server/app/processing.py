"""Background job processing orchestration."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.models import Job

logger = logging.getLogger(__name__)


# Legacy v0 dependencies are loaded only when the legacy pipeline is selected.
# Tests patch these module attributes directly, so keep the names available
# without importing TensorFlow/Basic Pitch/OpenCV at module import time.
AudioAnalysisConfig = None
AudioPreprocessConfig = None
EnsembleConfig = None
FretboardDetectionConfig = None
FusionConfig = None
QuantizationConfig = None
SpectralResidualConfig = None
VideoAnalysisConfig = None

analyze_pitch = None
analyze_spectral_residual = None
analyze_video_at_timestamps = None
detect_fretboard_from_video = None
detect_muted_notes = None
detect_note_onsets = None
detect_with_ensemble = None
extract_audio = None
fuse_audio_only = None
fuse_audio_video = None
preprocess_audio = None
quantize_notes = None
track_fretboard_temporal = None


@dataclass
class EnhancedAudioConfig:
    """Configuration for enhanced audio processing (ensemble + spectral)."""
    ensemble: Optional[EnsembleConfig] = None
    spectral_residual: Optional[SpectralResidualConfig] = None


def _ensure_legacy_dependencies() -> None:
    """Import legacy v0 dependencies lazily for local/threaded processing."""
    global AudioAnalysisConfig, AudioPreprocessConfig
    global EnsembleConfig, FretboardDetectionConfig, FusionConfig, QuantizationConfig
    global SpectralResidualConfig, VideoAnalysisConfig
    global analyze_pitch, analyze_spectral_residual, analyze_video_at_timestamps
    global detect_fretboard_from_video, detect_muted_notes, detect_note_onsets
    global detect_with_ensemble, extract_audio, fuse_audio_only, fuse_audio_video
    global preprocess_audio, quantize_notes, track_fretboard_temporal

    if (
        AudioAnalysisConfig is None
        or AudioPreprocessConfig is None
        or analyze_pitch is None
        or detect_note_onsets is None
        or extract_audio is None
        or preprocess_audio is None
        or detect_muted_notes is None
    ):
        from app.audio_pipeline import (
            AudioAnalysisConfig as _AudioAnalysisConfig,
            AudioPreprocessConfig as _AudioPreprocessConfig,
            analyze_pitch as _analyze_pitch,
            detect_muted_notes as _detect_muted_notes,
            detect_note_onsets as _detect_note_onsets,
            extract_audio as _extract_audio,
            preprocess_audio as _preprocess_audio,
        )

        AudioAnalysisConfig = AudioAnalysisConfig or _AudioAnalysisConfig
        AudioPreprocessConfig = AudioPreprocessConfig or _AudioPreprocessConfig
        analyze_pitch = analyze_pitch or _analyze_pitch
        detect_muted_notes = detect_muted_notes or _detect_muted_notes
        detect_note_onsets = detect_note_onsets or _detect_note_onsets
        extract_audio = extract_audio or _extract_audio
        preprocess_audio = preprocess_audio or _preprocess_audio

    if QuantizationConfig is None or quantize_notes is None:
        from app.beat_quantization import (
            QuantizationConfig as _QuantizationConfig,
            quantize_notes as _quantize_notes,
        )

        QuantizationConfig = QuantizationConfig or _QuantizationConfig
        quantize_notes = quantize_notes or _quantize_notes

    if (
        FretboardDetectionConfig is None
        or detect_fretboard_from_video is None
        or track_fretboard_temporal is None
    ):
        from app.fretboard_detection import (
            FretboardDetectionConfig as _FretboardDetectionConfig,
            detect_fretboard_from_video as _detect_fretboard_from_video,
            track_fretboard_temporal as _track_fretboard_temporal,
        )

        FretboardDetectionConfig = FretboardDetectionConfig or _FretboardDetectionConfig
        detect_fretboard_from_video = (
            detect_fretboard_from_video or _detect_fretboard_from_video
        )
        track_fretboard_temporal = track_fretboard_temporal or _track_fretboard_temporal

    if FusionConfig is None or fuse_audio_only is None or fuse_audio_video is None:
        from app.fusion_engine import (
            FusionConfig as _FusionConfig,
            fuse_audio_only as _fuse_audio_only,
            fuse_audio_video as _fuse_audio_video,
        )

        FusionConfig = FusionConfig or _FusionConfig
        fuse_audio_only = fuse_audio_only or _fuse_audio_only
        fuse_audio_video = fuse_audio_video or _fuse_audio_video

    if EnsembleConfig is None or detect_with_ensemble is None:
        from app.secondary_pitch_detector import (
            EnsembleConfig as _EnsembleConfig,
            detect_with_ensemble as _detect_with_ensemble,
        )

        EnsembleConfig = EnsembleConfig or _EnsembleConfig
        detect_with_ensemble = detect_with_ensemble or _detect_with_ensemble

    if SpectralResidualConfig is None or analyze_spectral_residual is None:
        from app.spectral_residual import (
            SpectralResidualConfig as _SpectralResidualConfig,
            analyze_spectral_residual as _analyze_spectral_residual,
        )

        SpectralResidualConfig = SpectralResidualConfig or _SpectralResidualConfig
        analyze_spectral_residual = (
            analyze_spectral_residual or _analyze_spectral_residual
        )

    if VideoAnalysisConfig is None or analyze_video_at_timestamps is None:
        from app.video_pipeline import (
            VideoAnalysisConfig as _VideoAnalysisConfig,
            analyze_video_at_timestamps as _analyze_video_at_timestamps,
        )

        VideoAnalysisConfig = VideoAnalysisConfig or _VideoAnalysisConfig
        analyze_video_at_timestamps = (
            analyze_video_at_timestamps or _analyze_video_at_timestamps
        )


def update_job(job: Job, stage: str, progress: float, storage=None) -> None:
    """Update job status and progress."""
    job.current_stage = stage
    job.progress = progress
    job.status = "processing"
    job.updated_at = datetime.now(timezone.utc)
    if storage is not None:
        storage.save(job)


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
        if note.pitch_bend:
            note_data["pitchBend"] = note.pitch_bend

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
            "pipelineVersion": "legacy_v0",
            "audioBackend": "basicpitch",
            "positionPrior": "none",
            "videoEnabled": video_confirmed > 0,
            "accuracyMode": job.accuracy_mode,
            "noteCountRatio": None,
            "diagnostics": {
                "notesAffectedByVideo": video_confirmed,
            },
        }
    }

    os.makedirs(output_dir, exist_ok=True)
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
    storage,
    output_dir: str = None,
    audio_config: AudioAnalysisConfig = None,
    preprocess_config: AudioPreprocessConfig = None,
    video_config: VideoAnalysisConfig = None,
    fretboard_config: FretboardDetectionConfig = None,
    fusion_config: FusionConfig = None,
    enhanced_audio_config: EnhancedAudioConfig = None,
    quantization_config: QuantizationConfig = None,
    result_saved_hook=None,
) -> None:
    """Process a job through the audio and video analysis pipelines.

    Args:
        job_id: ID of the job to process
        storage: JobStorage instance
        output_dir: Directory for temp files and results
        audio_config: Audio analysis configuration
        preprocess_config: Audio preprocessing configuration (normalization, filtering)
        video_config: Video analysis configuration
        fretboard_config: Fretboard detection configuration
        fusion_config: Fusion engine configuration
        enhanced_audio_config: Enhanced audio processing (ensemble + spectral)
        result_saved_hook: Optional callback after result JSON is written but before completion is saved
    """
    job = storage.get(job_id)
    if job is None:
        return

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    pipeline_name = os.getenv("TABVISION_PIPELINE", "legacy_v0").strip().lower()
    if pipeline_name in {"v1", "tabvision_v1"}:
        from app.v1_adapter import process_v1_job

        process_v1_job(
            job,
            storage,
            output_dir,
            result_saved_hook=result_saved_hook,
        )
        return

    _ensure_legacy_dependencies()

    # Use default configs if not provided
    if audio_config is None:
        audio_config = AudioAnalysisConfig()
    if video_config is None:
        video_config = VideoAnalysisConfig()
    if fusion_config is None:
        fusion_config = FusionConfig()

    # Wire ROI from job to video pipeline configs
    roi = None
    if job.roi_x1 is not None:
        roi = {
            'x1': job.roi_x1, 'y1': job.roi_y1,
            'x2': job.roi_x2, 'y2': job.roi_y2,
        }
        video_config.roi = roi

    try:
        # Stage 1: Extract audio
        update_job(job, "extracting_audio", 0.1, storage)
        audio_path = os.path.join(output_dir, f"{job.id}_audio.wav")
        extract_audio(job.video_path, audio_path)

        # Stage 1b: Preprocess audio (normalize, filter, noise gate)
        if preprocess_config is None:
            preprocess_config = AudioPreprocessConfig()
        preprocessed_path = os.path.join(output_dir, f"{job.id}_audio_preprocessed.wav")
        preprocess_audio(audio_path, preprocessed_path, preprocess_config)
        analysis_audio_path = preprocessed_path

        # Stage 2: Analyze with pitch detector
        update_job(job, "analyzing_audio", 0.3, storage)
        detected_notes = analyze_pitch(analysis_audio_path, audio_config)
        logger.info(f"Audio analysis: detected {len(detected_notes)} notes")

        # Stage 2b: Enhanced audio (ensemble + spectral residual)
        if enhanced_audio_config is None:
            enhanced_audio_config = EnhancedAudioConfig(
                ensemble=EnsembleConfig(),
            )
        if enhanced_audio_config:
            if enhanced_audio_config.ensemble and enhanced_audio_config.ensemble.enabled:
                pre_count = len(detected_notes)
                detected_notes = detect_with_ensemble(
                    analysis_audio_path, detected_notes, enhanced_audio_config.ensemble
                )
                logger.info(
                    f"Ensemble: {pre_count} -> {len(detected_notes)} notes "
                    f"(+{len(detected_notes) - pre_count})"
                )

            if enhanced_audio_config.spectral_residual and enhanced_audio_config.spectral_residual.enabled:
                pre_count = len(detected_notes)
                detected_notes = analyze_spectral_residual(
                    analysis_audio_path, detected_notes, enhanced_audio_config.spectral_residual
                )
                logger.info(
                    f"Spectral residual: {pre_count} -> {len(detected_notes)} notes "
                    f"(+{len(detected_notes) - pre_count})"
                )

        # Stage 2c: Detect muted notes (percussive transients without pitch)
        muted_notes = []
        try:
            muted_notes = detect_muted_notes(analysis_audio_path, detected_notes)
            if muted_notes:
                logger.info(f"Detected {len(muted_notes)} potential muted notes")
        except Exception as muted_err:
            logger.warning(f"Muted note detection failed: {muted_err}")

        # Stage 3: Analyze video (optional - graceful fallback if fails)
        update_job(job, "analyzing_video", 0.5, storage)
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
                    num_sample_frames=5,
                    roi=roi
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
        update_job(job, "fusing", 0.7, storage)

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
                fusion_config,
                muted_notes=muted_notes,
            )
            logger.info(
                f"Fusion complete: {len(tab_notes)} tab notes, "
                f"{sum(1 for n in tab_notes if n.video_matched)} video-confirmed"
            )
        else:
            # Fall back to audio-only
            tab_notes = fuse_audio_only(
                detected_notes, job.capo_fret, fusion_config, muted_notes=muted_notes
            )
            logger.info(f"Audio-only fusion: {len(tab_notes)} tab notes")

        # Log confidence distribution
        if tab_notes:
            high_conf = sum(1 for n in tab_notes if n.confidence_level == "high")
            med_conf = sum(1 for n in tab_notes if n.confidence_level == "medium")
            low_conf = sum(1 for n in tab_notes if n.confidence_level == "low")
            logger.info(
                f"Confidence distribution: high={high_conf}, medium={med_conf}, low={low_conf}"
            )

        # Stage 4b: Beat quantization
        if quantization_config is None:
            quantization_config = QuantizationConfig()
        if quantization_config.enabled and tab_notes:
            try:
                tab_notes = quantize_notes(tab_notes, analysis_audio_path, quantization_config)
            except Exception as quant_err:
                logger.warning(f"Beat quantization failed: {quant_err}, using unquantized notes")

        # Stage 5: Save result
        update_job(job, "saving", 0.9, storage)
        result_path = save_result(job, tab_notes, output_dir)
        if result_saved_hook:
            result_saved_hook()
        job.result_path = result_path

        # Complete
        job.status = "completed"
        job.current_stage = "complete"
        job.progress = 1.0
        job.updated_at = datetime.now(timezone.utc)
        storage.save(job)

        # Cleanup temp audio files
        for path in [audio_path, preprocessed_path]:
            if os.path.exists(path):
                os.remove(path)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"Job {job_id} failed")
        job.status = "failed"
        job.error_message = f"{e}\n\nTraceback:\n{tb}"
        job.updated_at = datetime.now(timezone.utc)
        storage.save(job)
