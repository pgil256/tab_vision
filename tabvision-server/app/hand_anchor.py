"""Hand-position anchor derived from video landmarks.

Computes a timeline of (timestamp -> fret anchor) by projecting the fretting
hand's centroid onto the detected fretboard. This anchor replaces the
audio-pick-derived hand_position_fret in fusion when available, breaking the
circular dependency where early wrong picks drag later picks with them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from bisect import bisect_left, insort
import math
import logging

from app.video_pipeline import HandObservation
from app.fretboard_detection import FretboardGeometry

logger = logging.getLogger(__name__)


@dataclass
class HandAnchorPoint:
    """One smoothed hand-anchor sample on the timeline."""
    timestamp: float
    anchor_fret: float     # Fractional fret, absolute fret space
    confidence: float      # [0, 1]


# --- Configuration (exposed for tuning, rarely changed) --------------------

MIN_EXTENDED_FINGERS = 2       # Need >= 2 extended non-thumb fingers for fingertip centroid
MIN_FRAME_CONFIDENCE = 0.4     # Drop raw samples below this
OUTLIER_WINDOW_SEC = 0.3       # Median window for outlier rejection
OUTLIER_FRET_THRESHOLD = 3.0   # Samples > this many frets from median...
OUTLIER_CONF_THRESHOLD = 0.7   # ...AND below this confidence are dropped
GAP_RESET_SEC = 0.5            # EMA resets if dt > this
BOUNDS_MARGIN = 0.15           # rel_x allowed outside [0, 1] by this much before penalty
DEFAULT_MAX_QUERY_GAP = 0.3    # Queries farther than this from nearest anchor return None


# --- Projection ------------------------------------------------------------


def _compute_palm_xy(observation: HandObservation) -> Optional[tuple[float, float]]:
    """Return a normalized (x, y) estimate of the fretting-hand centroid on the fretboard.

    Primary: centroid of extended non-thumb fingertips (these are on the fretboard
    when fretting and their centroid tracks the hand's fret span).
    Fallback: wrist position.

    Returns None if neither is available.
    """
    extended = [f for f in observation.fingers if f.finger_id != 0 and f.is_extended]
    if len(extended) >= MIN_EXTENDED_FINGERS:
        palm_x = sum(f.x for f in extended) / len(extended)
        palm_y = sum(f.y for f in extended) / len(extended)
        return palm_x, palm_y
    if observation.wrist_position is not None:
        return observation.wrist_position[0], observation.wrist_position[1]
    return None


def _interpolate_fret_from_rel_x(
    rel_x: float,
    fret_positions: list[float],
    actual_fret_numbers: Optional[list[int]] = None,
    starting_fret: int = 0,
) -> Optional[float]:
    """Return a fractional fret number in absolute fret space for a rel_x along the neck.

    Unlike `_find_nearest_fret_smart`, this linearly interpolates between the two
    bracketing detected fret positions, returning a float. Used for the anchor
    (which wants a smooth value, not a snapped integer).

    Args:
        rel_x: Relative x position along the neck (0 = nut-side, 1 = body-side).
        fret_positions: Detected fret x positions, normalized [0, 1].
        actual_fret_numbers: Absolute fret number for each detected position.
        starting_fret: Fallback for the first detected position's fret number
            when actual_fret_numbers is not supplied.

    Returns None if fret_positions is too short to interpolate.
    """
    if not fret_positions or len(fret_positions) < 2:
        return None

    # Filter detected positions to remove near-duplicates (matches _find_nearest_fret_smart)
    filtered: list[tuple[float, int]] = []
    prev_pos = -1.0
    for i, pos in enumerate(sorted(fret_positions)):
        if prev_pos < 0 or pos - prev_pos > 0.03:
            fret_num = (
                actual_fret_numbers[i]
                if actual_fret_numbers is not None and i < len(actual_fret_numbers)
                else starting_fret + i
            )
            filtered.append((pos, fret_num))
            prev_pos = pos

    if len(filtered) < 2:
        return None

    positions = [p for p, _ in filtered]
    frets = [f for _, f in filtered]

    # Before the first detected fret: extrapolate linearly using the first gap.
    if rel_x <= positions[0]:
        gap_pos = positions[1] - positions[0]
        gap_fret = frets[1] - frets[0]
        if gap_pos <= 0:
            return float(frets[0])
        return frets[0] + (rel_x - positions[0]) / gap_pos * gap_fret

    # Past the last detected fret: extrapolate using the last gap.
    if rel_x >= positions[-1]:
        gap_pos = positions[-1] - positions[-2]
        gap_fret = frets[-1] - frets[-2]
        if gap_pos <= 0:
            return float(frets[-1])
        return frets[-1] + (rel_x - positions[-1]) / gap_pos * gap_fret

    # Between positions i and i+1: linear interpolation.
    idx = bisect_left(positions, rel_x)
    # bisect_left returns insertion index; since rel_x > positions[0] and
    # rel_x < positions[-1], idx is in [1, len-1].
    lo_pos, lo_fret = positions[idx - 1], frets[idx - 1]
    hi_pos, hi_fret = positions[idx], frets[idx]
    span = hi_pos - lo_pos
    if span <= 0:
        return float(lo_fret)
    return lo_fret + (rel_x - lo_pos) / span * (hi_fret - lo_fret)


def project_palm_to_fret(
    observation: HandObservation,
    fretboard: FretboardGeometry,
) -> tuple[Optional[float], float]:
    """Project the fretting-hand centroid onto the neck axis → fractional fret.

    Returns (anchor_fret, confidence). `anchor_fret` is None when projection fails
    (hand off-fretboard, not enough fingers, or degenerate geometry).
    """
    palm = _compute_palm_xy(observation)
    if palm is None:
        return None, 0.0

    palm_x_norm, palm_y_norm = palm
    palm_x_px = palm_x_norm * fretboard.frame_width
    palm_y_px = palm_y_norm * fretboard.frame_height

    # Project onto neck axis (mirrors fretboard_detection.map_finger_to_position).
    neck_vec_x = fretboard.top_right[0] - fretboard.top_left[0]
    neck_vec_y = fretboard.top_right[1] - fretboard.top_left[1]
    neck_length = math.hypot(neck_vec_x, neck_vec_y)
    if neck_length <= 0:
        return None, 0.0
    neck_unit_x = neck_vec_x / neck_length
    neck_unit_y = neck_vec_y / neck_length

    finger_vec_x = palm_x_px - fretboard.top_left[0]
    finger_vec_y = palm_y_px - fretboard.top_left[1]
    rel_x = (finger_vec_x * neck_unit_x + finger_vec_y * neck_unit_y) / neck_length

    # Bounds penalty — if palm is well outside the fretboard, discard.
    if rel_x < -BOUNDS_MARGIN or rel_x > 1 + BOUNDS_MARGIN:
        return None, 0.0
    if rel_x < 0 or rel_x > 1:
        bounds_penalty = 1.0 - min(1.0, abs(rel_x - max(0.0, min(1.0, rel_x))) / BOUNDS_MARGIN)
    else:
        bounds_penalty = 1.0

    anchor_fret = _interpolate_fret_from_rel_x(
        rel_x,
        fretboard.fret_positions,
        actual_fret_numbers=fretboard.actual_fret_numbers,
        starting_fret=fretboard.starting_fret,
    )
    if anchor_fret is None:
        return None, 0.0

    confidence = (
        fretboard.detection_confidence
        * observation.hand_confidence
        * bounds_penalty
    )
    return anchor_fret, max(0.0, min(1.0, confidence))


# --- Timeline build --------------------------------------------------------


def _median(values: list[float]) -> float:
    n = len(values)
    s = sorted(values)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def build_hand_position_timeline(
    video_observations: dict[float, HandObservation],
    fretboard: Optional[FretboardGeometry],
    require_fretting_hand: bool = False,
) -> list[HandAnchorPoint]:
    """Project every observation to a fret anchor, reject outliers, smooth over time.

    Output is sorted by timestamp. Samples that fail projection or fall below
    MIN_FRAME_CONFIDENCE are dropped. Samples more than OUTLIER_FRET_THRESHOLD
    frets from the local median AND below OUTLIER_CONF_THRESHOLD are dropped.
    Retained samples are EMA-smoothed; gaps > GAP_RESET_SEC restart the filter.

    Args:
        video_observations: Frame-indexed hand detections.
        fretboard: Detected fretboard geometry (None → empty timeline).
        require_fretting_hand: When True, drop observations whose selected hand
            is NOT the fretting hand (is_left_hand=False under MediaPipe's
            mirrored labeling for a right-handed player). Prevents the anchor
            from tracking the picking hand when MediaPipe only detected one
            hand and picked the wrong one.
    """
    if not video_observations or fretboard is None:
        return []

    # Step 1: raw projection per timestamp.
    raw: list[tuple[float, float, float]] = []  # (t, fret, conf)
    for t in sorted(video_observations):
        obs = video_observations[t]
        if require_fretting_hand and not obs.is_left_hand:
            continue
        fret, conf = project_palm_to_fret(obs, fretboard)
        if fret is None or conf < MIN_FRAME_CONFIDENCE:
            continue
        raw.append((t, fret, conf))

    if len(raw) < 2:
        # Too few samples — return whatever we have as a single-point timeline.
        return [HandAnchorPoint(t, f, c) for t, f, c in raw]

    # Step 2: outlier rejection via median in ±OUTLIER_WINDOW_SEC neighborhood.
    kept: list[tuple[float, float, float]] = []
    for i, (t, fret, conf) in enumerate(raw):
        neighbors = [
            raw[j][1]
            for j in range(len(raw))
            if j != i and abs(raw[j][0] - t) <= OUTLIER_WINDOW_SEC
        ]
        if len(neighbors) >= 2:
            med = _median(neighbors)
            if abs(fret - med) > OUTLIER_FRET_THRESHOLD and conf < OUTLIER_CONF_THRESHOLD:
                continue
        kept.append((t, fret, conf))

    if not kept:
        return []

    # Step 3: confidence-weighted EMA with gap reset.
    smoothed: list[HandAnchorPoint] = []
    for t, fret, conf in kept:
        if not smoothed:
            smoothed.append(HandAnchorPoint(t, fret, conf))
            continue
        dt = t - smoothed[-1].timestamp
        if dt > GAP_RESET_SEC:
            smoothed.append(HandAnchorPoint(t, fret, conf))
            continue
        alpha = max(0.1, min(0.6, dt * 2.0)) * conf
        new_fret = (1 - alpha) * smoothed[-1].anchor_fret + alpha * fret
        # Confidence of the smoothed point = geometric mean of recent confidences
        new_conf = math.sqrt(smoothed[-1].confidence * conf)
        smoothed.append(HandAnchorPoint(t, new_fret, new_conf))

    return smoothed


# --- Query -----------------------------------------------------------------


def get_hand_anchor_at(
    timeline: list[HandAnchorPoint],
    timestamp: float,
    max_gap: float = DEFAULT_MAX_QUERY_GAP,
) -> tuple[Optional[float], float]:
    """Return (anchor_fret, confidence) at `timestamp`, interpolating between samples.

    If the nearest timeline point is farther than `max_gap`, returns (None, 0.0).
    """
    if not timeline:
        return None, 0.0

    timestamps = [p.timestamp for p in timeline]
    idx = bisect_left(timestamps, timestamp)

    if idx == 0:
        p = timeline[0]
        if abs(p.timestamp - timestamp) <= max_gap:
            return p.anchor_fret, p.confidence
        return None, 0.0

    if idx >= len(timeline):
        p = timeline[-1]
        if abs(p.timestamp - timestamp) <= max_gap:
            return p.anchor_fret, p.confidence
        return None, 0.0

    # Between timeline[idx-1] and timeline[idx].
    lo = timeline[idx - 1]
    hi = timeline[idx]

    # Gap check: ensure at least ONE bracketing point is within max_gap.
    # If both are farther than max_gap, we've landed in a silence/occlusion gap.
    if min(timestamp - lo.timestamp, hi.timestamp - timestamp) > max_gap:
        return None, 0.0

    span = hi.timestamp - lo.timestamp
    if span <= 0:
        return lo.anchor_fret, lo.confidence
    w = (timestamp - lo.timestamp) / span
    fret = (1 - w) * lo.anchor_fret + w * hi.anchor_fret
    conf = (1 - w) * lo.confidence + w * hi.confidence
    return fret, conf
