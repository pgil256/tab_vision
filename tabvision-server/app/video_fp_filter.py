"""Video-based false positive filter.

Uses negative video evidence (hand is NOT at a given position) to suppress
low-confidence audio detections that contradict what the video shows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from app.video_pipeline import HandObservation
from app.fretboard_detection import FretboardGeometry, map_finger_to_position

if TYPE_CHECKING:
    from app.fusion_engine import TabNote


@dataclass
class VideoFPFilterConfig:
    """Configuration for video-based false positive filtering."""
    enabled: bool = False
    # Minimum fretboard detection confidence to trust video evidence
    min_fretboard_confidence: float = 0.6
    # Minimum hand detection confidence to use as negative evidence
    min_hand_confidence: float = 0.7
    # Only suppress notes below this confidence
    max_note_confidence_to_suppress: float = 0.75
    # String distance threshold: suppress if no finger within this many strings
    empty_string_distance: int = 2
    # Time tolerance for matching tab notes to video observations
    time_tolerance: float = 0.15


def compute_hand_coverage(
    observation: HandObservation,
    fretboard: FretboardGeometry,
) -> set[tuple[int, int]]:
    """Map finger positions to a set of covered (string, fret) regions.

    Returns positions that the hand is actively covering (pressing or nearby).

    Args:
        observation: Hand detection from video frame
        fretboard: Detected fretboard geometry

    Returns:
        Set of (string, fret) tuples that the hand is covering
    """
    covered = set()
    frame_width = fretboard.frame_width
    frame_height = fretboard.frame_height

    for finger in observation.fingers:
        finger_x = finger.x * frame_width
        finger_y = finger.y * frame_height

        video_pos = map_finger_to_position(
            finger_x, finger_y, fretboard,
            finger_z=finger.z,
            finger_id=finger.finger_id
        )
        if video_pos is None:
            continue

        # The finger covers its detected position
        covered.add((video_pos.string, video_pos.fret))
        # Also consider adjacent frets as "covered" (tolerance for detection error)
        if video_pos.fret > 0:
            covered.add((video_pos.string, video_pos.fret - 1))
        covered.add((video_pos.string, video_pos.fret + 1))

    return covered


def _get_hand_string_range(observation: HandObservation, fretboard: FretboardGeometry) -> Optional[tuple[int, int]]:
    """Get the range of strings the hand is covering.

    Args:
        observation: Hand detection
        fretboard: Fretboard geometry

    Returns:
        (min_string, max_string) tuple, or None if no fingers detected on fretboard
    """
    frame_width = fretboard.frame_width
    frame_height = fretboard.frame_height
    strings_covered = []

    for finger in observation.fingers:
        finger_x = finger.x * frame_width
        finger_y = finger.y * frame_height

        video_pos = map_finger_to_position(
            finger_x, finger_y, fretboard,
            finger_z=finger.z,
            finger_id=finger.finger_id
        )
        if video_pos:
            strings_covered.append(video_pos.string)

    if not strings_covered:
        return None

    return (min(strings_covered), max(strings_covered))


def filter_false_positives(
    tab_notes: list[TabNote],
    video_observations: dict[float, HandObservation],
    fretboard: FretboardGeometry,
    config: VideoFPFilterConfig,
) -> list[TabNote]:
    """Suppress tab notes that contradict video evidence.

    A note is suppressed if:
    1. Its confidence is below max_note_confidence_to_suppress
    2. A video observation exists near its timestamp
    3. The video shows the hand is NOT at the note's position
       (specifically: no finger within empty_string_distance strings)

    Args:
        tab_notes: Tab notes to filter
        video_observations: Dict mapping timestamp to HandObservation
        fretboard: Detected fretboard geometry
        config: Filter configuration

    Returns:
        Filtered tab notes
    """
    if not tab_notes or not video_observations or not fretboard:
        return tab_notes

    if fretboard.detection_confidence < config.min_fretboard_confidence:
        return tab_notes

    # Sort observation timestamps for efficient lookup
    obs_timestamps = sorted(video_observations.keys())

    result = []
    for note in tab_notes:
        # High-confidence notes are never suppressed
        if note.confidence >= config.max_note_confidence_to_suppress:
            result.append(note)
            continue

        # Video-matched notes are never suppressed
        if note.video_matched:
            result.append(note)
            continue

        # Find nearest video observation
        nearest_obs = None
        nearest_dist = float('inf')
        for ts in obs_timestamps:
            dist = abs(ts - note.timestamp)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_obs = video_observations[ts]
            elif dist > nearest_dist:
                break  # Timestamps are sorted, past the nearest

        if nearest_obs is None or nearest_dist > config.time_tolerance:
            result.append(note)
            continue

        if nearest_obs.confidence < config.min_hand_confidence:
            result.append(note)
            continue

        # Check if hand covers the note's string area
        string_range = _get_hand_string_range(nearest_obs, fretboard)
        if string_range is None:
            result.append(note)
            continue

        min_str, max_str = string_range
        note_string = note.string

        # Suppress if the note's string is far from all detected fingers
        dist_to_hand = 0
        if note_string < min_str:
            dist_to_hand = min_str - note_string
        elif note_string > max_str:
            dist_to_hand = note_string - max_str

        if dist_to_hand >= config.empty_string_distance:
            # Video shows hand is far from this string - suppress
            continue

        result.append(note)

    return result
