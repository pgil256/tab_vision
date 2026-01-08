"""Video pipeline for extracting frames and detecting hand landmarks."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FingerPosition:
    """Position of a fingertip in frame coordinates."""
    finger_id: int      # 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
    x: float            # Normalized x coordinate (0-1)
    y: float            # Normalized y coordinate (0-1)
    z: float            # Depth estimate (relative)


@dataclass
class HandObservation:
    """Hand detection result for a single frame."""
    timestamp: float
    fingers: list[FingerPosition]
    is_left_hand: bool


# MediaPipe fingertip landmark indices
# MediaPipe hand landmarks: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_NAMES = {4: 0, 8: 1, 12: 2, 16: 3, 20: 4}  # Map landmark index to finger_id


def extract_frame(video_path: str, timestamp: float) -> np.ndarray | None:
    """Extract a single frame from video at given timestamp.

    Args:
        video_path: Path to video file
        timestamp: Time in seconds to extract frame at

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

    return frame


def detect_hand_landmarks(frame: np.ndarray) -> HandObservation | None:
    """Detect hand and finger positions using MediaPipe.

    Args:
        frame: BGR numpy array from video frame

    Returns:
        HandObservation with fingertip landmarks, or None if no hand detected
    """
    if frame is None or frame.size == 0:
        return None

    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "mediapipe is not installed. "
            "Install with: pip install mediapipe"
        )

    mp_hands = mp.solutions.hands

    # Initialize MediaPipe Hands with static_image_mode for single frame analysis
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return None

        # Get the first (and only, since max_num_hands=1) hand
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_info = results.multi_handedness[0]

        # Determine if left or right hand
        # MediaPipe labels are from the camera's perspective (mirrored)
        # So "Left" in MediaPipe means the player's right hand when facing camera
        is_left_hand = hand_info.classification[0].label == "Right"  # Player's left hand

        # Extract fingertip positions
        fingers = []
        for landmark_idx in FINGERTIP_INDICES:
            landmark = hand_landmarks.landmark[landmark_idx]
            fingers.append(FingerPosition(
                finger_id=FINGER_NAMES[landmark_idx],
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
            ))

        return HandObservation(
            timestamp=0.0,  # Caller should set this
            fingers=fingers,
            is_left_hand=is_left_hand,
        )


def analyze_video_at_timestamps(
    video_path: str,
    timestamps: list[float]
) -> dict[float, HandObservation]:
    """Analyze video frames at given timestamps.

    Extracts frames at each timestamp and runs hand detection.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps (in seconds) to analyze

    Returns:
        Dictionary mapping timestamp to HandObservation
        (only includes timestamps where a hand was detected)
    """
    observations = {}

    for ts in timestamps:
        frame = extract_frame(video_path, ts)
        if frame is None:
            continue

        observation = detect_hand_landmarks(frame)
        if observation is not None:
            # Update the timestamp in the observation
            observation = HandObservation(
                timestamp=ts,
                fingers=observation.fingers,
                is_left_hand=observation.is_left_hand,
            )
            observations[ts] = observation

    return observations
