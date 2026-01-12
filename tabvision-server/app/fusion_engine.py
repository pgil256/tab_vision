"""Fusion engine for combining audio and video analysis into tab notes."""
from dataclasses import dataclass
from uuid import uuid4
from app.audio_pipeline import DetectedNote
from app.guitar_mapping import get_candidate_positions, pick_lowest_fret, Position
from app.video_pipeline import HandObservation
from app.fretboard_detection import FretboardGeometry, map_finger_to_position


@dataclass
class TabNote:
    """A note in the guitar tablature."""
    id: str
    timestamp: float        # seconds
    string: int             # 1-6
    fret: int               # 0-24
    confidence: float       # 0.0-1.0
    confidence_level: str   # "high", "medium", "low"
    midi_note: int          # Original MIDI note for debugging


def get_confidence_level(confidence: float) -> str:
    """Map confidence score to level.

    Args:
        confidence: Score from 0.0 to 1.0

    Returns:
        "high" (>0.8), "medium" (0.5-0.8), or "low" (<0.5)
    """
    if confidence > 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def fuse_audio_only(
    detected_notes: list[DetectedNote],
    capo_fret: int = 0
) -> list[TabNote]:
    """Convert detected audio notes to TabNotes using lowest-fret heuristic.

    Args:
        detected_notes: Notes detected from audio analysis
        capo_fret: Fret where capo is placed (0 = no capo)

    Returns:
        List of TabNote objects
    """
    tab_notes = []

    for note in detected_notes:
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue  # Skip notes outside guitar range

        position = pick_lowest_fret(candidates)
        if position is None:
            continue

        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=note.start_time,
            string=position.string,
            fret=position.fret,
            confidence=note.confidence,
            confidence_level=get_confidence_level(note.confidence),
            midi_note=note.midi_note,
        ))

    return tab_notes


def find_nearest_observation(
    observations: dict[float, HandObservation],
    timestamp: float,
    tolerance: float = 0.1
) -> HandObservation | None:
    """Find the video observation nearest to a timestamp.

    Args:
        observations: Dict mapping timestamps to HandObservation
        timestamp: Target timestamp in seconds
        tolerance: Maximum time difference in seconds

    Returns:
        Nearest HandObservation within tolerance, or None
    """
    if not observations:
        return None

    nearest_ts = min(observations.keys(), key=lambda t: abs(t - timestamp))
    if abs(nearest_ts - timestamp) <= tolerance:
        return observations[nearest_ts]
    return None


def match_video_to_candidates(
    observation: HandObservation,
    fretboard: FretboardGeometry,
    candidates: list[Position],
    frame_width: int = 640,
    frame_height: int = 480
) -> Position | None:
    """Try to match video finger positions to audio candidates.

    Args:
        observation: Hand detection from video
        fretboard: Detected fretboard geometry
        candidates: Candidate positions from audio analysis
        frame_width: Video frame width for coordinate conversion
        frame_height: Video frame height for coordinate conversion

    Returns:
        Matching Position if found, None otherwise
    """
    for finger in observation.fingers:
        # Convert normalized coordinates to pixel coordinates
        finger_x = finger.x * frame_width
        finger_y = finger.y * frame_height

        video_pos = map_finger_to_position(finger_x, finger_y, fretboard)
        if video_pos is None:
            continue

        # Check if video position matches any audio candidate
        for candidate in candidates:
            if candidate.string == video_pos.string and candidate.fret == video_pos.fret:
                return candidate

    return None


def has_open_string_candidate(candidates: list[Position]) -> Position | None:
    """Check if fret 0 (open string) is among the candidates.

    Args:
        candidates: List of possible positions

    Returns:
        Position with fret 0 if found, None otherwise
    """
    for candidate in candidates:
        if candidate.fret == 0:
            return candidate
    return None


def fuse_audio_video(
    detected_notes: list[DetectedNote],
    video_observations: dict[float, HandObservation],
    fretboard: FretboardGeometry | None,
    capo_fret: int = 0
) -> list[TabNote]:
    """Combine audio and video signals for tab generation.

    For each detected note:
    1. Get audio candidates (from guitar_mapping)
    2. Get video observation (if available)
    3. If video agrees with audio candidate → high confidence
    4. If no finger match but fret 0 is valid → use open string (medium confidence)
    5. If no video data → use audio only (lowest-fret heuristic)

    Args:
        detected_notes: Notes detected from audio analysis
        video_observations: Dict mapping timestamp to HandObservation
        fretboard: Detected fretboard geometry (None if not detected)
        capo_fret: Fret where capo is placed (0 = no capo)

    Returns:
        List of TabNote objects
    """
    # If no fretboard detected, fall back to audio-only
    if fretboard is None:
        return fuse_audio_only(detected_notes, capo_fret)

    tab_notes = []

    for note in detected_notes:
        # Get audio candidates
        candidates = get_candidate_positions(note.midi_note, capo_fret)
        if not candidates:
            continue

        # Try to get video observation at this timestamp
        video_obs = find_nearest_observation(video_observations, note.start_time)
        video_position = None

        if video_obs:
            # Try to match video fingers to audio candidates
            video_position = match_video_to_candidates(
                video_obs, fretboard, candidates
            )

        # Determine final position and confidence
        if video_position:
            # Video agrees with an audio candidate - boost confidence
            position = video_position
            confidence = min(1.0, note.confidence + 0.2)
        elif video_obs:
            # We have video observation but no finger matched any candidate
            # Check if this could be an open string (fret 0)
            open_string = has_open_string_candidate(candidates)
            if open_string:
                # No finger on fretboard + fret 0 is valid = likely open string
                position = open_string
                confidence = 0.65  # Medium confidence for open string inference
            else:
                # Fall back to lowest-fret heuristic
                position = pick_lowest_fret(candidates)
                if position is None:
                    continue
                confidence = note.confidence
        else:
            # No video observation - fall back to lowest-fret heuristic
            position = pick_lowest_fret(candidates)
            if position is None:
                continue
            confidence = note.confidence

        tab_notes.append(TabNote(
            id=str(uuid4()),
            timestamp=note.start_time,
            string=position.string,
            fret=position.fret,
            confidence=confidence,
            confidence_level=get_confidence_level(confidence),
            midi_note=note.midi_note,
        ))

    return tab_notes
