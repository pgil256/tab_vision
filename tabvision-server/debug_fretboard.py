"""Debug script to visualize fretboard detection and finger mapping."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from app.fretboard_detection import (
    FretboardDetectionConfig, FretboardGeometry,
    detect_fretboard, map_finger_to_position, STANDARD_FRET_RATIOS
)
from app.video_pipeline import detect_hand_landmarks, VideoAnalysisConfig, _get_hand_landmarker


def analyze_fretboard_detection():
    """Analyze what the fretboard detection is finding."""
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip to frame with likely finger positions (around 1 second in based on tabs)
    target_time = 1.0  # Start of the song
    target_frame = int(target_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        return

    height, width = frame.shape[:2]
    print(f"Frame size: {width}x{height}")

    # Configure fretboard detection with ROI
    config = FretboardDetectionConfig()
    config.roi = {'y_start': 0.45, 'y_end': 0.70, 'x_start': 0.0, 'x_end': 1.0}

    # Detect fretboard
    geometry = detect_fretboard(frame, config)

    # Import to test identify_fret_numbers
    from app.fretboard_detection import identify_fret_numbers

    if geometry is None:
        print("No fretboard detected")
        return

    print("\n=== FRETBOARD GEOMETRY ===")
    print(f"Detection confidence: {geometry.detection_confidence:.2f}")
    print(f"Rotation angle: {geometry.rotation_angle:.1f}°")
    print(f"Corners:")
    print(f"  top_left: ({geometry.top_left[0]:.1f}, {geometry.top_left[1]:.1f})")
    print(f"  top_right: ({geometry.top_right[0]:.1f}, {geometry.top_right[1]:.1f})")
    print(f"  bottom_left: ({geometry.bottom_left[0]:.1f}, {geometry.bottom_left[1]:.1f})")
    print(f"  bottom_right: ({geometry.bottom_right[0]:.1f}, {geometry.bottom_right[1]:.1f})")
    print(f"Dimensions: {geometry.width:.1f}w x {geometry.height:.1f}h")
    print(f"Aspect ratio: {geometry.width / geometry.height:.1f}")

    print(f"\n=== FRET POSITIONS (normalized) ===")
    print(f"Number of detected fret lines: {len(geometry.fret_positions)}")
    for i, pos in enumerate(geometry.fret_positions):
        print(f"  Detected fret {i}: {pos:.4f}")

    # Calculate spacing between detected frets
    print(f"\n=== FRET SPACING ANALYSIS ===")
    spacings = []
    for i in range(len(geometry.fret_positions) - 1):
        spacing = geometry.fret_positions[i+1] - geometry.fret_positions[i]
        spacings.append(spacing)
        print(f"  Spacing {i}->{i+1}: {spacing:.4f}")

    if spacings:
        avg_spacing = sum(spacings) / len(spacings)
        print(f"\nAverage spacing: {avg_spacing:.4f}")

        # In standard fret spacing, each fret is ~5.6% closer than the previous
        # So fret 1 to 2 is slightly larger than fret 2 to 3, etc.
        # Let's see if spacing decreases (indicating higher frets visible) or increases

        spacing_trend = spacings[-1] - spacings[0] if len(spacings) > 1 else 0
        if spacing_trend < 0:
            print("Spacing DECREASING: We're looking at higher frets (towards body)")
        else:
            print("Spacing INCREASING or CONSTANT")

    # Compare to theoretical fret spacing to estimate which frets are visible
    print(f"\n=== FRET IDENTIFICATION ===")
    print("Comparing detected spacing to theoretical guitar fret ratios...")

    # Calculate theoretical spacing between consecutive frets
    theoretical_spacings = []
    for i in range(1, 15):  # Frets 1-14
        spacing = STANDARD_FRET_RATIOS[i+1] - STANDARD_FRET_RATIOS[i]
        theoretical_spacings.append(spacing)
        # print(f"  Theoretical fret {i} to {i+1}: {spacing:.4f}")

    # Try to match our detected spacings to theoretical ones
    # But first filter out likely non-fret lines
    print(f"\n=== FILTERING FRET CANDIDATES ===")

    # Real guitar frets have decreasing spacing as you go up the neck
    # Also, the spacing should be relatively consistent (within ~30%)
    # Filter detected positions to find ones that look like real frets

    # For this video, the camera likely shows frets 0-12 range
    # Let's estimate based on spacing pattern
    if len(spacings) >= 3:
        # Calculate median spacing
        sorted_spacings = sorted(spacings)
        median_spacing = sorted_spacings[len(sorted_spacings)//2]
        print(f"Median spacing: {median_spacing:.4f}")

        # Filter positions where spacing is close to median (+/- 50%)
        filtered_positions = [geometry.fret_positions[0]]
        for i, spacing in enumerate(spacings):
            if 0.5 * median_spacing <= spacing <= 1.5 * median_spacing:
                filtered_positions.append(geometry.fret_positions[i+1])
            else:
                print(f"  Rejecting position {i+1} - spacing {spacing:.4f} too different from median")

        print(f"Filtered to {len(filtered_positions)} likely fret positions")

    # For the actual mapping, we need a different approach
    # The key insight is that guitar fret spacing follows a known formula
    print(f"\n=== FIRST FEW THEORETICAL FRET POSITIONS ===")
    for i in range(13):
        print(f"  Fret {i}: {STANDARD_FRET_RATIOS[i]:.4f}")

    # Check what the fret number identification detected
    print(f"\n=== DETECTED FRET NUMBER MAPPING ===")
    print(f"Starting fret: {geometry.starting_fret}")
    print(f"Actual fret numbers: {geometry.actual_fret_numbers}")

    # Check string mapping - which direction is "top" of fretboard?
    print("\n=== STRING ORIENTATION CHECK ===")
    print("In guitar tab/video: Low E (string 6) is typically at top, High E (string 1) at bottom")
    print(f"Fretboard top_left: ({geometry.top_left[0]:.1f}, {geometry.top_left[1]:.1f})")
    print(f"Fretboard bottom_left: ({geometry.bottom_left[0]:.1f}, {geometry.bottom_left[1]:.1f})")
    if geometry.top_left[1] < geometry.bottom_left[1]:
        print("top_left is ABOVE bottom_left (y increases downward)")
        print("So string 6 (low E) is at top_left, string 1 (high E) is at bottom_left")
    else:
        print("bottom_left is ABOVE top_left - GEOMETRY MAY BE INVERTED!")

    # Now test with hand detection
    print("\n\n=== HAND DETECTION TEST ===")

    video_config = VideoAnalysisConfig()
    landmarker = _get_hand_landmarker(video_config)

    try:
        observation = detect_hand_landmarks(frame, video_config, landmarker)

        if observation:
            print(f"Hand detected: {'left' if observation.is_left_hand else 'right'}")
            print(f"Number of fingers: {len(observation.fingers)}")

            # Get pressing fingers
            pressing = observation.get_pressing_finger_positions()
            print(f"Pressing fingers: {len(pressing)}")

            print("\n=== FINGER POSITION MAPPING ===")
            for finger in observation.fingers:
                if finger.finger_id == 0:  # Skip thumb for now
                    continue

                # Convert normalized to pixel
                finger_x = finger.x * width
                finger_y = finger.y * height

                # Map to fretboard
                pos = map_finger_to_position(
                    finger_x, finger_y, geometry,
                    finger_z=finger.z,
                    finger_id=finger.finger_id
                )

                if pos:
                    print(f"\nFinger {finger.finger_id}:")
                    print(f"  Pixel position: ({finger_x:.1f}, {finger_y:.1f})")
                    print(f"  Finger Z depth: {finger.z:.4f}")
                    print(f"  Mapped to: string {pos.string}, fret {pos.fret}")
                    print(f"  Confidence: {pos.confidence:.2f}")

                    # If we determined the starting fret above, calculate actual fret
                    if 'best_match_fret' in dir():
                        actual_fret = best_match_fret + pos.fret
                        print(f"  ADJUSTED FRET (if starting at fret {best_match_fret}): {actual_fret}")
        else:
            print("No hand detected in this frame")
    finally:
        landmarker.close()

    cap.release()

    # Save debug visualization
    save_debug_image(frame, geometry, "/home/gilhooleyp/projects/tab_vision/debug_fretboard.png")
    print("\nDebug image saved to debug_fretboard.png")


def save_debug_image(frame, geometry, output_path):
    """Save a debug image with fretboard overlay."""
    debug_img = frame.copy()

    # Draw fretboard corners
    corners = [
        geometry.top_left, geometry.top_right,
        geometry.bottom_right, geometry.bottom_left
    ]

    for i, corner in enumerate(corners):
        pt = (int(corner[0]), int(corner[1]))
        cv2.circle(debug_img, pt, 10, (0, 255, 0), -1)
        cv2.putText(debug_img, f"C{i}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw fretboard boundary
    for i in range(4):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1]))
        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)

    # Draw fret lines
    for i, fret_pos in enumerate(geometry.fret_positions):
        # Interpolate position along top and bottom edges
        top_x = geometry.top_left[0] + fret_pos * (geometry.top_right[0] - geometry.top_left[0])
        top_y = geometry.top_left[1] + fret_pos * (geometry.top_right[1] - geometry.top_left[1])
        bot_x = geometry.bottom_left[0] + fret_pos * (geometry.bottom_right[0] - geometry.bottom_left[0])
        bot_y = geometry.bottom_left[1] + fret_pos * (geometry.bottom_right[1] - geometry.bottom_left[1])

        cv2.line(debug_img, (int(top_x), int(top_y)), (int(bot_x), int(bot_y)), (255, 0, 0), 2)
        cv2.putText(debug_img, f"F{i}", (int(top_x), int(top_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imwrite(output_path, debug_img)


if __name__ == '__main__':
    analyze_fretboard_detection()
