"""Save frames at specific timestamps for visual inspection."""
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


def save_frames():
    """Save frames with annotations at key timestamps."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Timestamps - 0.58s should be s2f5 (string 2, fret 5)
    timestamps = [0.58, 0.82, 1.05]

    # Get fretboard geometry
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.0 * fps))
    ret, frame = cap.read()

    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.75, 'x_start': 0.0, 'x_end': 1.0}

    geometry = detect_fretboard(frame, config)
    geometry.frame_width = frame.shape[1]
    geometry.frame_height = frame.shape[0]

    video_config = VideoAnalysisConfig()
    landmarker = _get_hand_landmarker(video_config)

    try:
        for i, ts in enumerate(timestamps):
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                continue

            height, width = frame.shape[:2]
            debug_frame = frame.copy()

            # Draw fretboard box
            corners = [geometry.top_left, geometry.top_right,
                      geometry.bottom_right, geometry.bottom_left]
            pts = np.array([[int(c[0]), int(c[1])] for c in corners], np.int32)
            cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 2)

            # Draw string lines and label where strings 1-2 should be
            for si, sp in enumerate(geometry.string_positions):
                string_num = 6 - si

                # Calculate line from nut to body for this string
                left_x = geometry.top_left[0] + sp * (geometry.bottom_left[0] - geometry.top_left[0])
                left_y = geometry.top_left[1] + sp * (geometry.bottom_left[1] - geometry.top_left[1])
                right_x = geometry.top_right[0] + sp * (geometry.bottom_right[0] - geometry.top_right[0])
                right_y = geometry.top_right[1] + sp * (geometry.bottom_right[1] - geometry.top_right[1])

                # Color: strings 1-2 in red (the missing ones)
                color = (0, 0, 255) if string_num <= 2 else (255, 200, 0)
                thickness = 3 if string_num <= 2 else 1

                cv2.line(debug_frame, (int(left_x), int(left_y)),
                        (int(right_x), int(right_y)), color, thickness)
                cv2.putText(debug_frame, f"S{string_num}", (int(right_x)+5, int(right_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Detect and draw hand
            observation = detect_hand_landmarks(frame, video_config, landmarker)

            if observation:
                for finger in observation.fingers:
                    finger_x = int(finger.x * width)
                    finger_y = int(finger.y * height)

                    # Blue for fingers
                    cv2.circle(debug_frame, (finger_x, finger_y), 10, (255, 0, 0), -1)
                    cv2.putText(debug_frame, f"F{finger.finger_id}",
                               (finger_x+12, finger_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Add timestamp and expected note info
            expected = {0.58: "s2f5", 0.82: "s1f5", 1.05: "s1f7+s4f6"}
            cv2.putText(debug_frame, f"t={ts:.2f}s Expected: {expected.get(ts, '?')}",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Save
            output_path = f"/home/gilhooleyp/projects/tab_vision/frame_debug_{i}_{ts:.2f}s.png"
            cv2.imwrite(output_path, debug_frame)
            print(f"Saved {output_path}")

    finally:
        landmarker.close()

    cap.release()


if __name__ == '__main__':
    save_frames()
