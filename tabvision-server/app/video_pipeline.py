"""Video pipeline for extracting frames and detecting hand landmarks.

Uses MediaPipe Tasks API (v0.10+) for hand landmark detection.
"""
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class FingerPosition:
    """Position of a fingertip in frame coordinates."""
    finger_id: int      # 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    x: float            # Normalized x coordinate (0-1)
    y: float            # Normalized y coordinate (0-1)
    z: float            # Depth estimate (relative, negative = closer to camera)
    # Additional finger state info
    is_extended: bool = True  # Whether finger appears extended (not curled)
    angle: float = 0.0  # Finger angle relative to hand
    confidence: float = 1.0  # Detection confidence for this finger


@dataclass
class HandObservation:
    """Hand detection result for a single frame."""
    timestamp: float
    fingers: list[FingerPosition]
    is_left_hand: bool
    # Additional hand info
    wrist_position: Optional[tuple[float, float, float]] = None
    hand_confidence: float = 1.0
    # Derived analysis
    pressing_fingers: list[int] = field(default_factory=list)  # finger_ids that appear to be pressing
    muting_fingers: list[int] = field(default_factory=list)   # finger_ids that appear to be muting (light contact)

    def get_pressing_finger_positions(self) -> list[FingerPosition]:
        """Get only fingers that appear to be pressing on the fretboard."""
        return [f for f in self.fingers if f.finger_id in self.pressing_fingers]

    def get_muting_finger_positions(self) -> list[FingerPosition]:
        """Get only fingers that appear to be muting strings (light contact)."""
        return [f for f in self.fingers if f.finger_id in self.muting_fingers]


@dataclass
class VideoAnalysisConfig:
    """Configuration for video analysis."""
    # MediaPipe detection confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Number of hands to detect (2 = detect both, then filter to fretting hand)
    max_num_hands: int = 2
    # Static vs video mode
    static_image_mode: bool = False  # Use False for better temporal tracking
    # Frame extraction settings
    frame_buffer_before: float = 0.066  # Seconds before onset to analyze (~2 frames at 30fps)
    frame_buffer_after: float = 0.066   # Seconds after onset to analyze
    frames_per_onset: int = 5           # Number of frames to sample per onset
    # Finger state detection
    finger_curl_threshold: float = 0.6  # Ratio threshold for curled fingers
    pressing_z_threshold: float = -0.02  # Z value threshold for pressing detection
    muting_z_threshold: float = -0.005   # Z threshold for muting (light contact, between 0 and pressing)
    # Model path for MediaPipe Tasks API
    model_path: Optional[str] = None  # None = use default path
    # Region of interest (normalized 0-1 coordinates)
    roi: Optional[dict] = None  # {'x1': float, 'y1': float, 'x2': float, 'y2': float}


# MediaPipe landmark indices
# Fingertip landmarks
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
# Finger base landmarks (MCP joints)
FINGER_BASE_INDICES = [2, 5, 9, 13, 17]  # thumb base, index MCP, middle MCP, ring MCP, pinky MCP
# Finger middle landmarks (PIP joints for fingers, IP for thumb)
FINGER_MIDDLE_INDICES = [3, 6, 10, 14, 18]
# DIP joints (for fingers) / tip for thumb
FINGER_DIP_INDICES = [4, 7, 11, 15, 19]

FINGER_NAMES = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}  # Map landmark index to finger_id
FINGER_ID_TO_NAME = {0: "thumb", 1: "index", 2: "middle", 3: "ring", 4: "pinky"}

# Default model path
DEFAULT_MODEL_PATH = os.path.expanduser("~/.mediapipe/models/hand_landmarker.task")


def _get_hand_landmarker(config: VideoAnalysisConfig):
    """Create a MediaPipe HandLandmarker instance.

    Uses the MediaPipe Tasks API (v0.10+).

    Args:
        config: Video analysis configuration

    Returns:
        HandLandmarker instance

    Raises:
        ImportError: If mediapipe is not installed
        FileNotFoundError: If model file is not found
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        raise ImportError(
            "mediapipe is not installed. "
            "Install with: pip install mediapipe"
        )

    # Get model path
    model_path = config.model_path or DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. "
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=config.max_num_hands,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )

    return vision.HandLandmarker.create_from_options(options)


def extract_frame(
    video_path: str,
    timestamp: float,
    roi: dict = None
) -> np.ndarray | None:
    """Extract a single frame from video at given timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame at
        roi: Optional dict with x1, y1, x2, y2 (normalized 0-1) to crop frame

    Returns:
        BGR numpy array of the frame, or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Get video FPS to calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    # Calculate target frame number
    frame_number = int(timestamp * fps)

    # Seek to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Apply ROI cropping if provided
    if roi is not None:
        h, w = frame.shape[:2]
        x1 = int(roi['x1'] * w)
        y1 = int(roi['y1'] * h)
        x2 = int(roi['x2'] * w)
        y2 = int(roi['y2'] * h)
        frame = frame[y1:y2, x1:x2]

    return frame


def detect_hand_landmarks(
    frame: np.ndarray,
    config: Optional[VideoAnalysisConfig] = None,
    landmarker=None
) -> HandObservation | None:
    """Detect hand and finger positions using MediaPipe Tasks API.

    Args:
        frame: BGR numpy array from video frame
        config: Analysis configuration
        landmarker: Optional pre-created HandLandmarker (for batch processing)

    Returns:
        HandObservation with fingertip landmarks, or None if no hand detected
    """
    if frame is None or frame.size == 0:
        return None

    if config is None:
        config = VideoAnalysisConfig()

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except ImportError:
        raise ImportError(
            "mediapipe is not installed. "
            "Install with: pip install mediapipe"
        )

    # Create landmarker if not provided
    close_landmarker = False
    if landmarker is None:
        try:
            landmarker = _get_hand_landmarker(config)
            close_landmarker = True
        except FileNotFoundError as e:
            logger.warning(f"Hand landmarker model not available: {e}")
            return None

    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        results = landmarker.detect(mp_image)

        if not results.hand_landmarks or not results.handedness:
            return None

        # Select the fretting hand from detected hands
        hand_landmarks, hand_info = _select_fretting_hand(
            results.hand_landmarks, results.handedness
        )

        # Determine if left or right hand
        # MediaPipe labels are from the camera's perspective (mirrored)
        is_left_hand = hand_info[0].category_name == "Right"  # Player's left hand
        hand_confidence = hand_info[0].score

        # Get wrist position for reference
        wrist = hand_landmarks[0]
        wrist_position = (wrist.x, wrist.y, wrist.z)

        # Extract finger positions with extended state analysis
        fingers = []
        pressing_fingers = []
        muting_fingers = []

        for i, tip_idx in enumerate(FINGERTIP_INDICES):
            tip = hand_landmarks[tip_idx]
            base_idx = FINGER_BASE_INDICES[i]
            mid_idx = FINGER_MIDDLE_INDICES[i]

            base = hand_landmarks[base_idx]
            mid = hand_landmarks[mid_idx]

            # Determine if finger is extended or curled
            is_extended = _is_finger_extended(tip, mid, base, i == 0)  # thumb is special

            # Calculate finger angle (useful for detecting press direction)
            angle = _calculate_finger_angle(tip, base)

            # Determine if finger appears to be pressing (firm contact)
            is_pressing = (
                is_extended and
                tip.z < config.pressing_z_threshold
            )

            # Determine if finger appears to be muting (light contact)
            # Muting: z between muting_z_threshold and pressing_z_threshold
            is_muting = (
                is_extended and
                not is_pressing and
                tip.z < config.muting_z_threshold
            )

            finger_pos = FingerPosition(
                finger_id=i,
                x=tip.x,
                y=tip.y,
                z=tip.z,
                is_extended=is_extended,
                angle=angle,
                confidence=hand_confidence,
            )
            fingers.append(finger_pos)

            if i > 0:  # Exclude thumb
                if is_pressing:
                    pressing_fingers.append(i)
                elif is_muting:
                    muting_fingers.append(i)

        return HandObservation(
            timestamp=0.0,  # Caller should set this
            fingers=fingers,
            is_left_hand=is_left_hand,
            wrist_position=wrist_position,
            hand_confidence=hand_confidence,
            pressing_fingers=pressing_fingers,
            muting_fingers=muting_fingers,
        )
    finally:
        if close_landmarker and landmarker:
            landmarker.close()


def _select_fretting_hand(
    hand_landmarks_list: list,
    handedness_list: list,
) -> tuple:
    """Select the fretting hand from detected hands.

    For right-handed players, the fretting hand is the left hand (labeled
    "Right" by MediaPipe since it mirrors). We prefer the fretting hand
    because its finger positions map to fret/string coordinates. If only
    one hand is detected, we use that hand.

    Heuristics when handedness is ambiguous:
    - Fretting hand fingers tend to be more spread (wider x-range)
    - Fretting hand is typically on the left side of frame (right-handed player)

    Args:
        hand_landmarks_list: List of hand landmarks from MediaPipe
        handedness_list: List of handedness info from MediaPipe

    Returns:
        Tuple of (hand_landmarks, hand_info) for the selected hand
    """
    if len(hand_landmarks_list) == 1:
        return hand_landmarks_list[0], handedness_list[0]

    # With 2 hands detected, prefer the fretting hand
    # For right-handed player: fretting hand = player's left = MediaPipe "Right"
    fretting_idx = None
    for i, hand_info in enumerate(handedness_list):
        label = hand_info[0].category_name
        if label == "Right":  # Player's left (fretting) hand
            fretting_idx = i
            break

    if fretting_idx is not None:
        return hand_landmarks_list[fretting_idx], handedness_list[fretting_idx]

    # Fallback: pick hand with more finger spread (fretting hand has spread fingers)
    best_idx = 0
    best_spread = 0.0
    for i, landmarks in enumerate(hand_landmarks_list):
        tips = [landmarks[idx] for idx in FINGERTIP_INDICES]
        xs = [t.x for t in tips]
        spread = max(xs) - min(xs)
        if spread > best_spread:
            best_spread = spread
            best_idx = i

    return hand_landmarks_list[best_idx], handedness_list[best_idx]


def _get_landmark_coords(landmark):
    """Extract x, y, z coordinates from a landmark.

    Works with both legacy MediaPipe landmarks and new Tasks API landmarks.

    Args:
        landmark: MediaPipe landmark object

    Returns:
        Tuple of (x, y, z) coordinates
    """
    return (landmark.x, landmark.y, landmark.z)


def _is_finger_extended(
    tip,
    mid,
    base,
    is_thumb: bool = False
) -> bool:
    """Determine if a finger is extended based on joint positions.

    A finger is considered extended if the tip is farther from the base
    than the mid joint, relative to the overall finger length.

    Args:
        tip: Fingertip landmark
        mid: Middle joint landmark
        base: Base joint landmark
        is_thumb: Whether this is the thumb (different geometry)

    Returns:
        True if finger appears extended
    """
    tip_x, tip_y, _ = _get_landmark_coords(tip)
    mid_x, mid_y, _ = _get_landmark_coords(mid)
    base_x, base_y, _ = _get_landmark_coords(base)

    # Calculate distances
    tip_to_base = np.sqrt(
        (tip_x - base_x)**2 +
        (tip_y - base_y)**2
    )
    mid_to_base = np.sqrt(
        (mid_x - base_x)**2 +
        (mid_y - base_y)**2
    )

    if mid_to_base == 0:
        return True

    # For extended finger, tip should be roughly 1.5-2x the distance from base as mid
    extension_ratio = tip_to_base / mid_to_base

    if is_thumb:
        # Thumb has different geometry
        return extension_ratio > 1.2
    else:
        return extension_ratio > 1.3


def _calculate_finger_angle(tip, base) -> float:
    """Calculate angle of finger from base to tip.

    Args:
        tip: Fingertip landmark
        base: Finger base landmark

    Returns:
        Angle in degrees (0 = pointing right, 90 = pointing down)
    """
    tip_x, tip_y, _ = _get_landmark_coords(tip)
    base_x, base_y, _ = _get_landmark_coords(base)

    dx = tip_x - base_x
    dy = tip_y - base_y
    return np.arctan2(dy, dx) * 180 / np.pi


def analyze_video_at_timestamps(
    video_path: str,
    timestamps: list[float],
    config: Optional[VideoAnalysisConfig] = None
) -> dict[float, HandObservation]:
    """Analyze video frames at given timestamps.

    Extracts frames at each timestamp and runs hand detection.
    Uses multiple frames per onset for better accuracy.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to analyze
        config: Analysis configuration

    Returns:
        Dictionary mapping timestamp to HandObservation
        (only includes timestamps where a hand was detected)
    """
    if config is None:
        config = VideoAnalysisConfig()

    observations = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return observations

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return observations

    # Create a single landmarker instance for all frames (more efficient)
    try:
        landmarker = _get_hand_landmarker(config)
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Cannot create hand landmarker: {e}")
        cap.release()
        return observations

    try:
        # Process each timestamp
        for ts in timestamps:
            best_observation = None
            best_score = 0.0

            # Sample multiple frames around the onset with onset-biased weighting
            total_span = config.frame_buffer_before + config.frame_buffer_after
            for i in range(config.frames_per_onset):
                offset = (i - config.frames_per_onset // 2) * (
                    total_span / config.frames_per_onset
                )
                sample_ts = ts + offset

                if sample_ts < 0:
                    continue

                # Onset-biased weighting: frames closer to onset time are weighted higher
                # onset=1.0, ±33ms≈0.7, ±66ms≈0.4
                proximity_weight = max(0.1, 1.0 - abs(offset) / total_span)

                frame_idx = int(sample_ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Apply ROI cropping if configured
                if config.roi is not None:
                    h, w = frame.shape[:2]
                    x1 = int(config.roi['x1'] * w)
                    y1 = int(config.roi['y1'] * h)
                    x2 = int(config.roi['x2'] * w)
                    y2 = int(config.roi['y2'] * h)
                    frame = frame[y1:y2, x1:x2]

                observation = detect_hand_landmarks(frame, config, landmarker)
                if observation is not None:
                    # Score = detection confidence * temporal proximity weight
                    score = observation.hand_confidence * proximity_weight
                    if score > best_score:
                        best_observation = observation
                        best_score = score

            if best_observation is not None:
                # Update timestamp to the original onset time
                best_observation = HandObservation(
                    timestamp=ts,
                    fingers=best_observation.fingers,
                    is_left_hand=best_observation.is_left_hand,
                    wrist_position=best_observation.wrist_position,
                    hand_confidence=best_observation.hand_confidence,
                    pressing_fingers=best_observation.pressing_fingers,
                    muting_fingers=best_observation.muting_fingers,
                )
                observations[ts] = best_observation
    finally:
        landmarker.close()
        cap.release()

    return observations


def analyze_video_continuous(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    sample_interval: float = 0.1,
    config: Optional[VideoAnalysisConfig] = None
) -> list[HandObservation]:
    """Analyze video continuously at regular intervals.

    Useful for building a temporal model of hand movement.

    Args:
        video_path: Path to video file
        start_time: Start timestamp in seconds
        end_time: End timestamp (None = end of video)
        sample_interval: Time between samples in seconds
        config: Analysis configuration

    Returns:
        List of HandObservation objects with timestamps
    """
    if config is None:
        config = VideoAnalysisConfig()
        config.static_image_mode = False  # Use tracking mode

    observations = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return observations

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        return observations

    if end_time is None:
        end_time = total_frames / fps

    # Create landmarker for continuous processing
    try:
        import mediapipe as mp
        landmarker = _get_hand_landmarker(config)
    except (ImportError, FileNotFoundError) as e:
        logger.warning(f"Cannot create hand landmarker: {e}")
        cap.release()
        return observations

    try:
        current_time = start_time
        frame_step = int(sample_interval * fps)
        frame_step = max(1, frame_step)

        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame
        while current_time <= end_time:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Only process at sample intervals
            if (frame_count - start_frame) % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = landmarker.detect(mp_image)

                if results.hand_landmarks and results.handedness:
                    hand_landmarks = results.hand_landmarks[0]
                    hand_info = results.handedness[0]

                    observation = _create_observation_from_landmarks(
                        hand_landmarks, hand_info, current_time, config
                    )
                    if observation:
                        observations.append(observation)

            current_time = frame_count / fps
            frame_count += 1
    finally:
        landmarker.close()
        cap.release()

    return observations


def _create_observation_from_landmarks(
    hand_landmarks,
    hand_info,
    timestamp: float,
    config: VideoAnalysisConfig
) -> HandObservation | None:
    """Create HandObservation from MediaPipe Tasks API landmarks.

    Args:
        hand_landmarks: List of NormalizedLandmark objects
        hand_info: List of Category objects containing handedness info
        timestamp: Frame timestamp
        config: Video analysis configuration

    Returns:
        HandObservation or None
    """
    # Tasks API returns list of Category, not Classification
    is_left_hand = hand_info[0].category_name == "Right"  # Player's left hand
    hand_confidence = hand_info[0].score

    wrist = hand_landmarks[0]
    wrist_position = (wrist.x, wrist.y, wrist.z)

    fingers = []
    pressing_fingers = []
    muting_fingers = []

    for i, tip_idx in enumerate(FINGERTIP_INDICES):
        tip = hand_landmarks[tip_idx]
        base_idx = FINGER_BASE_INDICES[i]
        mid_idx = FINGER_MIDDLE_INDICES[i]

        base = hand_landmarks[base_idx]
        mid = hand_landmarks[mid_idx]

        is_extended = _is_finger_extended(tip, mid, base, i == 0)
        angle = _calculate_finger_angle(tip, base)

        is_pressing = (
            is_extended and
            tip.z < config.pressing_z_threshold
        )

        is_muting = (
            is_extended and
            not is_pressing and
            tip.z < config.muting_z_threshold
        )

        finger_pos = FingerPosition(
            finger_id=i,
            x=tip.x,
            y=tip.y,
            z=tip.z,
            is_extended=is_extended,
            angle=angle,
            confidence=hand_confidence,
        )
        fingers.append(finger_pos)

        if i > 0:
            if is_pressing:
                pressing_fingers.append(i)
            elif is_muting:
                muting_fingers.append(i)

    return HandObservation(
        timestamp=timestamp,
        fingers=fingers,
        is_left_hand=is_left_hand,
        wrist_position=wrist_position,
        hand_confidence=hand_confidence,
        pressing_fingers=pressing_fingers,
        muting_fingers=muting_fingers,
    )
