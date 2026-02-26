"""Debug script to check finger joint positions vs fingertip positions."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import math
from app.fretboard_detection import FretboardDetectionConfig, detect_fretboard
from app.video_pipeline import VideoAnalysisConfig, _get_hand_landmarker

# MediaPipe landmark indices
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
FINGER_DIP_INDICES = [3, 7, 11, 15, 19]  # DIP joints (closer to base)
FINGER_PIP_INDICES = [2, 6, 10, 14, 18]  # PIP joints (even closer to base)


def analyze_joint_positions():
    """Compare fingertip vs DIP joint positions."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get fretboard
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.0 * fps))
    ret, frame = cap.read()

    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.75, 'x_start': 0.0, 'x_end': 1.0}

    geometry = detect_fretboard(frame, config)
    geometry.frame_width = frame.shape[1]
    geometry.frame_height = frame.shape[0]

    # Calculate basis vectors for coordinate transformation
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

    def pixel_to_rel(px, py):
        """Convert pixel position to relative fretboard coordinates."""
        finger_vec_x = px - geometry.top_left[0]
        finger_vec_y = py - geometry.top_left[1]
        rel_x = (finger_vec_x * neck_unit_x + finger_vec_y * neck_unit_y) / neck_length
        rel_y = (finger_vec_x * string_unit_x + finger_vec_y * string_unit_y) / string_width
        return rel_x, rel_y

    video_config = VideoAnalysisConfig()
    landmarker = _get_hand_landmarker(video_config)

    import mediapipe as mp

    try:
        # Analyze at key timestamps
        timestamps = [0.58, 0.82, 1.05]

        for ts in timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_image)

            if not results.hand_landmarks:
                print(f"\n=== t={ts:.2f}s: No hand detected ===")
                continue

            hand_landmarks = results.hand_landmarks[0]

            print(f"\n=== t={ts:.2f}s ===")
            print("Finger | Fingertip rel_y | DIP rel_y | PIP rel_y | Tip String | DIP String")
            print("-" * 80)

            for i in range(1, 5):  # Skip thumb (index 0)
                tip_idx = FINGERTIP_INDICES[i]
                dip_idx = FINGER_DIP_INDICES[i]
                pip_idx = FINGER_PIP_INDICES[i]

                tip = hand_landmarks[tip_idx]
                dip = hand_landmarks[dip_idx]
                pip = hand_landmarks[pip_idx]

                tip_px, tip_py = tip.x * width, tip.y * height
                dip_px, dip_py = dip.x * width, dip.y * height
                pip_px, pip_py = pip.x * width, pip.y * height

                tip_rel_x, tip_rel_y = pixel_to_rel(tip_px, tip_py)
                dip_rel_x, dip_rel_y = pixel_to_rel(dip_px, dip_py)
                pip_rel_x, pip_rel_y = pixel_to_rel(pip_px, pip_py)

                # Determine strings
                def rel_y_to_string(rel_y):
                    if rel_y < 0 or rel_y > 1.2:
                        return "OUT"
                    string_idx = int(round(rel_y * 5))
                    string_idx = max(0, min(5, string_idx))
                    return str(6 - string_idx)

                tip_string = rel_y_to_string(tip_rel_y)
                dip_string = rel_y_to_string(dip_rel_y)

                finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                print(f"{finger_names[i]:7} | {tip_rel_y:15.3f} | {dip_rel_y:9.3f} | {pip_rel_y:9.3f} | "
                      f"{tip_string:10} | {dip_string}")

            # Check if any joint reaches string 1-2
            print("\nJoints reaching string 1-2 (rel_y > 0.7):")
            found_any = False
            for landmark_name, indices in [("Fingertip", FINGERTIP_INDICES),
                                           ("DIP", FINGER_DIP_INDICES),
                                           ("PIP", FINGER_PIP_INDICES)]:
                for i, idx in enumerate(indices):
                    if i == 0:  # Skip thumb
                        continue
                    lm = hand_landmarks[idx]
                    px, py = lm.x * width, lm.y * height
                    rel_x, rel_y = pixel_to_rel(px, py)
                    if rel_y > 0.7:
                        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                        expected_string = 1 if rel_y > 0.9 else 2
                        print(f"  {landmark_name} of {finger_names[i]}: rel_y={rel_y:.3f} -> string ~{expected_string}")
                        found_any = True
            if not found_any:
                print("  None found!")

    finally:
        landmarker.close()
    cap.release()


if __name__ == '__main__':
    analyze_joint_positions()
