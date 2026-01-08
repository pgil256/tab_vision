"""Fretboard detection using OpenCV edge detection and Hough transform."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FretboardGeometry:
    """Detected fretboard geometry."""
    # Corner points of fretboard region (for homography)
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_left: tuple[float, float]
    bottom_right: tuple[float, float]

    # Detected fret positions (x coordinates in normalized space)
    # Index 0 = nut, 1 = fret 1, etc.
    fret_positions: list[float]

    # Detected string positions (y coordinates in normalized space)
    # Index 0 = string 6 (low E), 5 = string 1 (high E)
    string_positions: list[float]


@dataclass
class VideoPosition:
    """A fret/string position detected from video."""
    string: int         # 1-6
    fret: int           # 0-24
    confidence: float   # 0-1


def detect_fretboard(frame: np.ndarray) -> FretboardGeometry | None:
    """Detect fretboard region and geometry in frame.

    Uses Canny edge detection and Hough line transform to identify
    the fretboard structure.

    Args:
        frame: BGR numpy array from video frame

    Returns:
        FretboardGeometry if detected, None if detection fails
    """
    if frame is None or frame.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Hough line transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None or len(lines) < 4:
        return None

    # Separate lines into horizontal (frets) and vertical (strings/edges)
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Horizontal lines (frets): angle close to 0 or 180
        if angle < 20 or angle > 160:
            horizontal_lines.append((x1, y1, x2, y2))
        # Vertical lines (strings/fretboard edges): angle close to 90
        elif 70 < angle < 110:
            vertical_lines.append((x1, y1, x2, y2))

    # Need at least 2 horizontal lines (frets) and 2 vertical lines (edges)
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Cluster horizontal lines by y-position to find distinct frets
    fret_y_positions = _cluster_line_positions(
        horizontal_lines, axis='y', min_distance=10
    )

    # Cluster vertical lines by x-position to find fretboard boundaries
    edge_x_positions = _cluster_line_positions(
        vertical_lines, axis='x', min_distance=20
    )

    if len(fret_y_positions) < 2 or len(edge_x_positions) < 2:
        return None

    # Get bounding box of detected fretboard
    height, width = frame.shape[:2]

    # Sort positions
    fret_y_positions.sort()
    edge_x_positions.sort()

    # Take the outermost fret lines as top/bottom boundaries
    top_y = fret_y_positions[0]
    bottom_y = fret_y_positions[-1]

    # Take the outermost vertical lines as left/right boundaries
    left_x = edge_x_positions[0]
    right_x = edge_x_positions[-1]

    # Normalize fret positions (0=nut end, 1=body end)
    fret_width = right_x - left_x
    fret_height = bottom_y - top_y

    if fret_width <= 0 or fret_height <= 0:
        return None

    # Normalize fret y-positions within the detected region
    normalized_frets = [
        (y - top_y) / fret_height for y in fret_y_positions
    ]

    # Generate 6 string positions (evenly distributed)
    # String 6 (low E) at top, String 1 (high E) at bottom
    normalized_strings = [i / 5.0 for i in range(6)]

    return FretboardGeometry(
        top_left=(float(left_x), float(top_y)),
        top_right=(float(right_x), float(top_y)),
        bottom_left=(float(left_x), float(bottom_y)),
        bottom_right=(float(right_x), float(bottom_y)),
        fret_positions=normalized_frets,
        string_positions=normalized_strings,
    )


def _cluster_line_positions(
    lines: list[tuple[int, int, int, int]],
    axis: str,
    min_distance: int
) -> list[float]:
    """Cluster line positions to find distinct frets/strings.

    Args:
        lines: List of (x1, y1, x2, y2) tuples
        axis: 'x' or 'y' to indicate which axis to cluster
        min_distance: Minimum pixel distance between clusters

    Returns:
        List of averaged positions for each cluster
    """
    if axis == 'y':
        positions = [(line[1] + line[3]) / 2 for line in lines]
    else:  # axis == 'x'
        positions = [(line[0] + line[2]) / 2 for line in lines]

    positions.sort()

    if not positions:
        return []

    # Cluster nearby positions
    clusters = [[positions[0]]]
    for pos in positions[1:]:
        if pos - clusters[-1][-1] < min_distance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    # Return average position for each cluster
    return [sum(cluster) / len(cluster) for cluster in clusters]


def map_finger_to_position(
    finger_x: float,
    finger_y: float,
    geometry: FretboardGeometry
) -> VideoPosition | None:
    """Map a finger position to fret/string coordinates.

    Args:
        finger_x: Normalized x coordinate (0-1) in frame
        finger_y: Normalized y coordinate (0-1) in frame
        geometry: Detected fretboard geometry

    Returns:
        VideoPosition if finger is on fretboard, None otherwise
    """
    # Get frame dimensions from geometry corners
    # (assuming geometry coordinates are in pixel space)
    fb_left = geometry.top_left[0]
    fb_right = geometry.top_right[0]
    fb_top = geometry.top_left[1]
    fb_bottom = geometry.bottom_left[1]

    # The finger coordinates are normalized (0-1), but geometry is in pixels
    # We need to know the frame dimensions to convert
    # For now, assume geometry corners define the relevant area

    # Convert normalized finger position to fretboard-relative position
    fb_width = fb_right - fb_left
    fb_height = fb_bottom - fb_top

    if fb_width <= 0 or fb_height <= 0:
        return None

    # Calculate relative position within fretboard
    # Note: finger_x/y are normalized to frame, geometry is in pixels
    # This function expects the caller to handle coordinate conversion
    # Here we assume finger_x/y are already in pixel coordinates

    rel_x = (finger_x - fb_left) / fb_width
    rel_y = (finger_y - fb_top) / fb_height

    # Check if finger is within fretboard bounds (with margin)
    margin = 0.1
    if rel_x < -margin or rel_x > 1 + margin:
        return None
    if rel_y < -margin or rel_y > 1 + margin:
        return None

    # Find nearest fret
    fret = _find_nearest_fret(rel_x, geometry.fret_positions)
    if fret is None:
        return None

    # Find nearest string
    string = _find_nearest_string(rel_y, geometry.string_positions)
    if string is None:
        return None

    # Calculate confidence based on proximity to exact fret/string position
    fret_confidence = _calculate_proximity_confidence(
        rel_x, geometry.fret_positions, fret
    )
    string_confidence = _calculate_proximity_confidence(
        rel_y, geometry.string_positions, string - 1  # Convert to 0-indexed
    )

    confidence = (fret_confidence + string_confidence) / 2

    return VideoPosition(
        string=string,
        fret=fret,
        confidence=confidence
    )


def _find_nearest_fret(rel_x: float, fret_positions: list[float]) -> int | None:
    """Find the nearest fret to a relative x position.

    Args:
        rel_x: Relative x position (0-1) within fretboard
        fret_positions: List of fret positions (normalized)

    Returns:
        Fret number (0 = nut/open, 1+ = frets), or None
    """
    if not fret_positions:
        return None

    # Find the fret position that the finger is closest to (but behind)
    # A finger at position 0.15 between fret 0 (0.0) and fret 1 (0.2)
    # is playing fret 1

    for i, fret_pos in enumerate(fret_positions):
        if rel_x < fret_pos:
            return i

    # Finger is past the last fret
    return len(fret_positions)


def _find_nearest_string(rel_y: float, string_positions: list[float]) -> int | None:
    """Find the nearest string to a relative y position.

    Args:
        rel_y: Relative y position (0-1) within fretboard
        string_positions: List of string positions (normalized)

    Returns:
        String number (1-6), or None
    """
    if not string_positions or len(string_positions) < 6:
        return None

    # Find closest string
    min_dist = float('inf')
    closest_idx = 0

    for i, string_pos in enumerate(string_positions):
        dist = abs(rel_y - string_pos)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # String positions are indexed 0-5 (string 6 to string 1)
    # Convert to string number 1-6
    # Index 0 = string 6 (low E), Index 5 = string 1 (high E)
    string_number = 6 - closest_idx

    return string_number


def _calculate_proximity_confidence(
    position: float,
    reference_positions: list[float],
    target_index: int
) -> float:
    """Calculate confidence based on proximity to target position.

    Args:
        position: Current position (0-1)
        reference_positions: List of reference positions
        target_index: Index of target position

    Returns:
        Confidence score (0-1)
    """
    if target_index < 0 or target_index >= len(reference_positions):
        # Position is outside the detected frets, lower confidence
        return 0.5

    target_pos = reference_positions[target_index]
    distance = abs(position - target_pos)

    # Calculate spacing to adjacent positions for normalization
    if target_index > 0:
        spacing_before = target_pos - reference_positions[target_index - 1]
    else:
        spacing_before = 0.1  # Default for first position

    if target_index < len(reference_positions) - 1:
        spacing_after = reference_positions[target_index + 1] - target_pos
    else:
        spacing_after = 0.1  # Default for last position

    avg_spacing = (spacing_before + spacing_after) / 2

    # Confidence decreases as distance from target increases
    # Full confidence when exactly on target, decreasing to 0.5 at half spacing
    if avg_spacing > 0:
        normalized_distance = distance / avg_spacing
        confidence = max(0.5, 1.0 - normalized_distance)
    else:
        confidence = 0.5

    return confidence


def detect_fretboard_from_video(video_path: str) -> FretboardGeometry | None:
    """Detect fretboard geometry from the first frame of a video.

    Args:
        video_path: Path to video file

    Returns:
        FretboardGeometry if detected, None otherwise
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    return detect_fretboard(frame)
