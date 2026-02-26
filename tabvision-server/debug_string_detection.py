"""Debug script to understand why strings 1-2 aren't being detected."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from app.fretboard_detection import (
    FretboardDetectionConfig, detect_fretboard, map_finger_to_position
)
from app.video_pipeline import detect_hand_landmarks, VideoAnalysisConfig, _get_hand_landmarker


def analyze_string_detection():
    """Check finger positions at specific timestamps where strings 1-2 should be played."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Timestamps where strings 1-2 are in ground truth
    # s2f5 @ 0.58s, s1f5 @ 0.82s, s1f7 @ 1.05s, s2f5 @ 1.28s, s1f7 @ 1.75s
    test_timestamps = [0.58, 0.82, 1.05, 1.28, 1.75, 2.0, 3.0, 4.0]

    # First get fretboard geometry from a good frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.0 * fps))
    ret, frame = cap.read()

    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.75, 'x_start': 0.0, 'x_end': 1.0}

    geometry = detect_fretboard(frame, config)
    if not geometry:
        print("No fretboard detected")
        return

    geometry.frame_width = frame.shape[1]
    geometry.frame_height = frame.shape[0]

    print("Fretboard geometry:")
    print(f"  top_left: {geometry.top_left}")
    print(f"  bottom_left: {geometry.bottom_left}")
    print(f"  Height (string direction): {geometry.height:.1f} pixels")
    print(f"  String positions (normalized): {geometry.string_positions}")
    print()

    # Calculate pixel positions for each string
    print("String pixel positions (at nut):")
    for i, sp in enumerate(geometry.string_positions):
        string_num = 6 - i
        # Interpolate between top_left (string 6) and bottom_left (string 1)
        px = geometry.top_left[0] + sp * (geometry.bottom_left[0] - geometry.top_left[0])
        py = geometry.top_left[1] + sp * (geometry.bottom_left[1] - geometry.top_left[1])
        print(f"  String {string_num}: ({px:.1f}, {py:.1f}) - rel_y = {sp:.2f}")
    print()

    video_config = VideoAnalysisConfig()
    landmarker = _get_hand_landmarker(video_config)

    try:
        for ts in test_timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            height, width = frame.shape[:2]

            print(f"\n=== Timestamp {ts:.2f}s ===")

            observation = detect_hand_landmarks(frame, video_config, landmarker)

            if not observation:
                print("No hand detected")
                continue

            print(f"Hand detected, {len(observation.fingers)} fingers")

            # Check all fingers
            for finger in observation.fingers:
                finger_x = finger.x * width
                finger_y = finger.y * height

                pos = map_finger_to_position(
                    finger_x, finger_y, geometry,
                    finger_z=finger.z,
                    finger_id=finger.finger_id
                )

                # Calculate relative position for debugging
                import math

                neck_vec_x = geometry.top_right[0] - geometry.top_left[0]
                neck_vec_y = geometry.top_right[1] - geometry.top_left[1]
                string_vec_x = geometry.bottom_left[0] - geometry.top_left[0]
                string_vec_y = geometry.bottom_left[1] - geometry.top_left[1]

                neck_length = math.sqrt(neck_vec_x**2 + neck_vec_y**2)
                string_width = math.sqrt(string_vec_x**2 + string_vec_y**2)

                neck_unit_x = neck_vec_x / neck_length
                neck_unit_y = neck_vec_y / neck_length
                string_unit_x = string_vec_x / string_width
                string_unit_y = string_vec_y / string_width

                finger_vec_x = finger_x - geometry.top_left[0]
                finger_vec_y = finger_y - geometry.top_left[1]

                rel_x = (finger_vec_x * neck_unit_x + finger_vec_y * neck_unit_y) / neck_length
                rel_y = (finger_vec_x * string_unit_x + finger_vec_y * string_unit_y) / string_width

                if pos:
                    print(f"  Finger {finger.finger_id}: pixel ({finger_x:.0f}, {finger_y:.0f}) -> "
                          f"rel_y={rel_y:.3f} -> string {pos.string}, fret {pos.fret}")
                else:
                    print(f"  Finger {finger.finger_id}: pixel ({finger_x:.0f}, {finger_y:.0f}) -> "
                          f"rel_y={rel_y:.3f} -> OUT OF BOUNDS")

                # Check if this should be on string 1-2
                if rel_y > 0.7:
                    expected_string = 1 if rel_y > 0.9 else 2
                    print(f"    *** rel_y={rel_y:.3f} should map to string ~{expected_string}")
    finally:
        landmarker.close()

    cap.release()


if __name__ == '__main__':
    analyze_string_detection()
