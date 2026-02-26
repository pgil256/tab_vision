"""Visualize fretboard and hand detection on video frames."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from app.fretboard_detection import (
    FretboardDetectionConfig, detect_fretboard
)
from app.video_pipeline import detect_hand_landmarks, VideoAnalysisConfig, _get_hand_landmarker


def visualize_detection():
    """Create visualization of fretboard and hand detection."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Go to frame at 1 second (similar to where tabs start)
    target_time = 1.0
    target_frame = int(target_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return

    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Configure fretboard detection
    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.70, 'x_start': 0.0, 'x_end': 1.0}

    # Calculate ROI boundaries
    roi_y_start = int(0.45 * height)
    roi_y_end = int(0.70 * height)
    roi_x_start = 0
    roi_x_end = width

    print(f"ROI: y={roi_y_start}-{roi_y_end}, x={roi_x_start}-{roi_x_end}")

    # Draw ROI on frame
    debug_frame = frame.copy()
    cv2.rectangle(debug_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 2)
    cv2.putText(debug_frame, "ROI", (10, roi_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Detect fretboard
    geometry = detect_fretboard(frame, config)

    if geometry is None:
        print("No fretboard detected")
    else:
        print(f"\nFretboard detected:")
        print(f"  top_left: ({geometry.top_left[0]:.1f}, {geometry.top_left[1]:.1f})")
        print(f"  top_right: ({geometry.top_right[0]:.1f}, {geometry.top_right[1]:.1f})")
        print(f"  bottom_left: ({geometry.bottom_left[0]:.1f}, {geometry.bottom_left[1]:.1f})")
        print(f"  bottom_right: ({geometry.bottom_right[0]:.1f}, {geometry.bottom_right[1]:.1f})")
        print(f"  confidence: {geometry.detection_confidence:.2f}")
        print(f"  starting_fret: {geometry.starting_fret}")

        # Draw fretboard corners
        corners = [
            geometry.top_left, geometry.top_right,
            geometry.bottom_right, geometry.bottom_left
        ]
        corner_colors = [(0, 255, 0), (0, 200, 0), (0, 150, 0), (0, 100, 0)]
        corner_labels = ["TL(str6)", "TR(str6)", "BR(str1)", "BL(str1)"]

        for i, (corner, color, label) in enumerate(zip(corners, corner_colors, corner_labels)):
            pt = (int(corner[0]), int(corner[1]))
            cv2.circle(debug_frame, pt, 12, color, -1)
            cv2.putText(debug_frame, label, (pt[0] + 15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw fretboard boundary
        pts = np.array([
            [int(c[0]), int(c[1])] for c in corners
        ], np.int32)
        cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 2)

        # Draw string lines
        for i, string_pos in enumerate(geometry.string_positions):
            # Interpolate position
            left_x = geometry.top_left[0] + string_pos * (geometry.bottom_left[0] - geometry.top_left[0])
            left_y = geometry.top_left[1] + string_pos * (geometry.bottom_left[1] - geometry.top_left[1])
            right_x = geometry.top_right[0] + string_pos * (geometry.bottom_right[0] - geometry.top_right[0])
            right_y = geometry.top_right[1] + string_pos * (geometry.bottom_right[1] - geometry.top_right[1])

            string_num = 6 - i
            cv2.line(debug_frame, (int(left_x), int(left_y)), (int(right_x), int(right_y)), (255, 200, 0), 1)
            cv2.putText(debug_frame, f"S{string_num}", (int(right_x) + 5, int(right_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # Detect hands
    video_config = VideoAnalysisConfig()
    landmarker = _get_hand_landmarker(video_config)

    try:
        observation = detect_hand_landmarks(frame, video_config, landmarker)

        if observation:
            print(f"\nHand detected: {'left' if observation.is_left_hand else 'right'}")

            # Draw finger positions
            for finger in observation.fingers:
                finger_x = int(finger.x * width)
                finger_y = int(finger.y * height)

                color = (255, 0, 0) if finger.finger_id == 0 else (255, 100, 100)
                cv2.circle(debug_frame, (finger_x, finger_y), 8, color, -1)
                cv2.putText(debug_frame, f"F{finger.finger_id}", (finger_x + 10, finger_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Show pixel coordinates
                print(f"  Finger {finger.finger_id}: pixel=({finger_x}, {finger_y})")

                # If we have fretboard, show relative position
                if geometry:
                    # Calculate relative position within fretboard
                    # Using the same calculation as map_finger_to_position
                    import math
                    from app.fretboard_detection import map_finger_to_position

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

                    # Find string
                    string_idx = int(round(rel_y * 5))
                    string_num = 6 - string_idx if 0 <= string_idx <= 5 else None

                    # Also use map_finger_to_position to see full result
                    pos = map_finger_to_position(finger_x, finger_y, geometry, finger.z, finger.finger_id)
                    pos_str = f"s{pos.string}f{pos.fret} conf={pos.confidence:.2f}" if pos else "None"

                    print(f"    rel_x={rel_x:.3f}, rel_y={rel_y:.3f} -> string {string_num}, mapped: {pos_str}")
    finally:
        landmarker.close()

    cap.release()

    # Save debug image
    output_path = "/home/gilhooleyp/projects/tab_vision/detection_visualization.png"
    cv2.imwrite(output_path, debug_frame)
    print(f"\nSaved visualization to {output_path}")


if __name__ == '__main__':
    visualize_detection()
