"""Video-only transcription pipeline for guitar tabs.

This module provides transcription capabilities when no audio track is available,
relying entirely on video analysis of finger positions on the fretboard.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from app.video_pipeline import (
    VideoAnalysisConfig, HandObservation, FingerPosition,
    detect_hand_landmarks, _get_hand_landmarker
)
from app.fretboard_detection import (
    FretboardDetectionConfig, FretboardGeometry,
    detect_fretboard, map_finger_to_position, VideoPosition
)
from app.fusion_engine import TabNote, get_confidence_level


@dataclass
class VideoOnlyConfig:
    """Configuration for video-only transcription."""
    # Frame sampling rate (frames per second to analyze)
    sample_fps: float = 10.0

    # Minimum confidence for a detected position
    min_position_confidence: float = 0.3

    # Minimum time between note events (debouncing)
    min_note_gap: float = 0.05  # seconds

    # Change detection threshold (normalized position change)
    position_change_threshold: float = 0.03

    # ROI for fretboard detection (normalized coordinates)
    # Default focuses on middle of frame where guitar neck typically appears
    fretboard_roi: Optional[dict] = field(
        default_factory=lambda: {'y_start': 0.45, 'y_end': 0.75, 'x_start': 0.0, 'x_end': 1.0}
    )

    # Video analysis config
    video_config: VideoAnalysisConfig = field(default_factory=VideoAnalysisConfig)

    # Fretboard detection config
    fretboard_config: FretboardDetectionConfig = field(default_factory=FretboardDetectionConfig)


@dataclass
class FrameAnalysis:
    """Analysis result for a single video frame."""
    timestamp: float
    hand_observation: Optional[HandObservation]
    finger_positions: list[VideoPosition]  # Fret/string positions for each detected finger
    fretboard: Optional[FretboardGeometry]


def _positions_changed(
    prev_positions: list[VideoPosition],
    curr_positions: list[VideoPosition],
    threshold: float
) -> bool:
    """Check if finger positions have changed significantly."""
    if len(prev_positions) != len(curr_positions):
        return True

    # Compare each position
    prev_set = {(p.string, p.fret) for p in prev_positions}
    curr_set = {(p.string, p.fret) for p in curr_positions}

    return prev_set != curr_set


def _extract_pressed_positions(
    observation: HandObservation,
    fretboard: FretboardGeometry
) -> list[VideoPosition]:
    """Extract fret/string positions from hand observation."""
    positions = []

    if observation is None or fretboard is None:
        return positions

    # Focus on pressing fingers (more likely to be fretting)
    pressing_fingers = observation.get_pressing_finger_positions()
    if not pressing_fingers:
        # Fall back to extended fingers
        pressing_fingers = [f for f in observation.fingers if f.is_extended and f.finger_id > 0]

    for finger in pressing_fingers:
        # Convert normalized coordinates to pixel coordinates
        finger_x = finger.x * fretboard.frame_width
        finger_y = finger.y * fretboard.frame_height

        pos = map_finger_to_position(
            finger_x, finger_y, fretboard,
            finger_z=finger.z,
            finger_id=finger.finger_id
        )

        if pos is not None and pos.confidence >= 0.3:
            positions.append(pos)

    return positions


def analyze_video_continuous(
    video_path: str,
    config: Optional[VideoOnlyConfig] = None,
    progress_callback=None
) -> list[FrameAnalysis]:
    """Analyze video continuously for finger positions.

    Args:
        video_path: Path to video file
        config: Analysis configuration
        progress_callback: Optional callback(progress: float) for status updates

    Returns:
        List of FrameAnalysis objects for each sampled frame
    """
    if config is None:
        config = VideoOnlyConfig()

    # Apply ROI to fretboard config
    if config.fretboard_roi:
        config.fretboard_config.roi = config.fretboard_roi

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if fps <= 0:
        cap.release()
        return []

    # Calculate frame step for desired sample rate
    frame_step = max(1, int(fps / config.sample_fps))

    analyses = []
    fretboard = None  # Will be detected and cached

    # Create hand landmarker once for efficiency
    try:
        landmarker = _get_hand_landmarker(config.video_config)
    except (ImportError, FileNotFoundError):
        cap.release()
        return []

    # First pass: find best fretboard detection from multiple sample frames
    # Sample frames around 1 second where guitar playing typically starts
    sample_frames = []
    sample_timestamps = [0.8, 1.0, 1.2, 1.5, 2.0]  # Seconds
    for ts in sample_timestamps:
        sample_frame_idx = int(ts * fps)
        if sample_frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
            ret, frame = cap.read()
            if ret:
                detected = detect_fretboard(frame, config.fretboard_config)
                if detected:
                    detected.frame_width = frame.shape[1]
                    detected.frame_height = frame.shape[0]
                    # Also check that the detected fretboard has reasonable dimensions
                    # (not too narrow which would indicate partial detection)
                    if detected.height > 200:  # At least 200 pixels height
                        sample_frames.append((detected, ts))

    # Use the detection with highest confidence and largest height (covers most strings)
    if sample_frames:
        # Sort by confidence * height to prefer larger, high-confidence detections
        sample_frames.sort(key=lambda x: -(x[0].detection_confidence * x[0].height))
        fretboard = sample_frames[0][0]

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Only analyze at sample rate
            if frame_idx % frame_step == 0:
                # Use pre-detected fretboard (already set above)
                # Fall back to detecting if we don't have one
                if fretboard is None:
                    fretboard = detect_fretboard(frame, config.fretboard_config)
                    if fretboard:
                        fretboard.frame_width = frame.shape[1]
                        fretboard.frame_height = frame.shape[0]

                # Detect hand landmarks
                observation = detect_hand_landmarks(frame, config.video_config, landmarker)
                if observation:
                    observation = HandObservation(
                        timestamp=timestamp,
                        fingers=observation.fingers,
                        is_left_hand=observation.is_left_hand,
                        wrist_position=observation.wrist_position,
                        hand_confidence=observation.hand_confidence,
                        pressing_fingers=observation.pressing_fingers,
                    )

                # Extract finger positions on fretboard
                positions = _extract_pressed_positions(observation, fretboard)

                analysis = FrameAnalysis(
                    timestamp=timestamp,
                    hand_observation=observation,
                    finger_positions=positions,
                    fretboard=fretboard,
                )
                analyses.append(analysis)

                # Progress callback
                if progress_callback:
                    progress_callback(timestamp / duration)

            frame_idx += 1

    finally:
        landmarker.close()
        cap.release()

    return analyses


def convert_to_tab_notes(
    analyses: list[FrameAnalysis],
    config: Optional[VideoOnlyConfig] = None
) -> list[TabNote]:
    """Convert frame analyses to tab notes.

    Uses change detection to identify when notes are played.

    Args:
        analyses: Frame analysis results
        config: Configuration

    Returns:
        List of TabNote objects
    """
    if config is None:
        config = VideoOnlyConfig()

    tab_notes = []
    prev_positions = []
    prev_timestamp = -1.0  # Start at -1 so first frame triggers change detection
    first_frame_processed = False

    for analysis in analyses:
        curr_positions = analysis.finger_positions

        # For the first frame with positions, record all positions as initial notes
        if not first_frame_processed and curr_positions:
            first_frame_processed = True
            for pos in curr_positions:
                note = TabNote(
                    id=str(uuid4()),
                    timestamp=analysis.timestamp,
                    string=pos.string,
                    fret=pos.fret,
                    confidence=pos.confidence,
                    confidence_level=get_confidence_level(pos.confidence),
                    midi_note=0,
                    video_matched=True,
                    video_confidence=pos.confidence,
                )
                tab_notes.append(note)
            prev_positions = curr_positions
            prev_timestamp = analysis.timestamp
            continue

        # Check if positions changed significantly
        if _positions_changed(prev_positions, curr_positions, config.position_change_threshold):
            # Debounce - require minimum time gap
            if analysis.timestamp - prev_timestamp >= config.min_note_gap:
                # Record new note events
                for pos in curr_positions:
                    # Check if this is a new position (not in previous)
                    is_new = not any(
                        p.string == pos.string and p.fret == pos.fret
                        for p in prev_positions
                    )

                    if is_new:
                        # Create tab note
                        note = TabNote(
                            id=str(uuid4()),
                            timestamp=analysis.timestamp,
                            string=pos.string,
                            fret=pos.fret,
                            confidence=pos.confidence,
                            confidence_level=get_confidence_level(pos.confidence),
                            midi_note=0,  # Unknown from video only
                            video_matched=True,
                            video_confidence=pos.confidence,
                        )
                        tab_notes.append(note)

                prev_timestamp = analysis.timestamp

        prev_positions = curr_positions

    # Sort by timestamp
    tab_notes.sort(key=lambda n: n.timestamp)

    return tab_notes


def transcribe_video_only(
    video_path: str,
    config: Optional[VideoOnlyConfig] = None,
    progress_callback=None
) -> list[TabNote]:
    """Full video-only transcription pipeline.

    Args:
        video_path: Path to video file
        config: Analysis configuration
        progress_callback: Optional progress callback

    Returns:
        List of TabNote objects
    """
    if config is None:
        config = VideoOnlyConfig()

    # Analyze video frames
    analyses = analyze_video_continuous(video_path, config, progress_callback)

    # Convert to tab notes
    tab_notes = convert_to_tab_notes(analyses, config)

    return tab_notes
