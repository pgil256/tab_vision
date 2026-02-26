"""Debug fret mapping to understand why high frets (11, 14, 17, 19) are being detected."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import math
from app.fretboard_detection import (
    FretboardDetectionConfig, detect_fretboard, map_finger_to_position,
    STANDARD_FRET_RATIOS
)
from app.video_pipeline import detect_hand_landmarks, VideoAnalysisConfig, _get_hand_landmarker


def analyze_fret_mapping():
    """Understand the fret mapping at different positions."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get geometry
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(1.0 * fps))
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.75, 'x_start': 0.0, 'x_end': 1.0}

    geometry = detect_fretboard(frame, config)
    geometry.frame_width = width
    geometry.frame_height = height

    print("Fretboard geometry:")
    print(f"  Detected fret positions: {geometry.fret_positions}")
    print(f"  Number of frets: {len(geometry.fret_positions)}")
    print()

    # Filter fret positions like the algorithm does
    filtered_positions = []
    prev_pos = -1
    for pos in sorted(geometry.fret_positions):
        if prev_pos < 0 or pos - prev_pos > 0.03:
            filtered_positions.append(pos)
            prev_pos = pos

    print(f"Filtered fret positions: {filtered_positions}")
    print(f"Number after filtering: {len(filtered_positions)}")
    print()

    # Show the mapping for each filtered position
    print("Position -> Fret mapping:")
    print("Detected Pos | Scaled Pos | Best Fret | Theoretical Pos")
    print("-" * 60)

    POSITION_SCALE_FACTOR = 1.2

    for pos in filtered_positions:
        scaled_pos = pos / POSITION_SCALE_FACTOR

        best_fret = 0
        best_diff = float('inf')
        for fret_num in range(20):
            theoretical_pos = STANDARD_FRET_RATIOS[fret_num]
            diff = abs(theoretical_pos - scaled_pos)
            if diff < best_diff:
                best_diff = diff
                best_fret = fret_num

        theoretical = STANDARD_FRET_RATIOS[best_fret]
        print(f"  {pos:.4f}      | {scaled_pos:.4f}     | {best_fret:9} | {theoretical:.4f}")

    # Now check what happens at different rel_x values
    print()
    print("rel_x -> fret mapping across the visible fretboard:")
    print("rel_x | Detected Idx | Det Pos | Scaled Pos | Fret")
    print("-" * 60)

    for rel_x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Find which filtered position we're behind
        detected_idx = None
        for i, fret_pos in enumerate(filtered_positions):
            if rel_x < fret_pos:
                if i == 0:
                    detected_idx = 0
                else:
                    prev_pos = filtered_positions[i - 1]
                    fret_space = fret_pos - prev_pos
                    threshold = prev_pos + fret_space * 0.4
                    if rel_x >= threshold:
                        detected_idx = i
                    else:
                        detected_idx = i - 1
                break

        if detected_idx is None:
            detected_idx = len(filtered_positions) - 1

        detected_position = filtered_positions[detected_idx] if detected_idx < len(filtered_positions) else 1.0
        scaled_position = detected_position / POSITION_SCALE_FACTOR

        best_fret = 0
        best_diff = float('inf')
        for fret_num in range(20):
            theoretical_pos = STANDARD_FRET_RATIOS[fret_num]
            diff = abs(theoretical_pos - scaled_position)
            if diff < best_diff:
                best_diff = diff
                best_fret = fret_num

        print(f"{rel_x:.2f}  | {detected_idx:12} | {detected_position:.4f}  | {scaled_position:.4f}     | {best_fret}")

    cap.release()


if __name__ == '__main__':
    analyze_fret_mapping()
