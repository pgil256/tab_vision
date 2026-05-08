"""Fretboard detection using OpenCV edge detection and Hough transform."""
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
import math


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

    # Confidence in the detection (0-1)
    detection_confidence: float = 0.5

    # Rotation angle of the fretboard (degrees, 0 = horizontal)
    rotation_angle: float = 0.0

    # Frame dimensions for coordinate conversion
    frame_width: int = 640
    frame_height: int = 480

    # Actual fret numbers corresponding to fret_positions
    # Maps each index in fret_positions to actual fret number (0-24)
    # If not set, assumes index = fret number
    actual_fret_numbers: list[int] = None

    # The starting fret visible in the detection
    starting_fret: int = 0

    @property
    def width(self) -> float:
        """Width of fretboard in pixels (along neck direction)."""
        # Use actual distance between corners (works for rotated rectangles)
        dx = self.top_right[0] - self.top_left[0]
        dy = self.top_right[1] - self.top_left[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def height(self) -> float:
        """Height of fretboard in pixels (perpendicular to neck)."""
        # Use actual distance between corners (works for rotated rectangles)
        dx = self.bottom_left[0] - self.top_left[0]
        dy = self.bottom_left[1] - self.top_left[1]
        return math.sqrt(dx * dx + dy * dy)

    def is_valid(self) -> bool:
        """Check if geometry is physically plausible for a guitar."""
        # Fretboard should have positive dimensions
        w = self.width
        h = self.height
        if w <= 0 or h <= 0:
            return False

        # Calculate aspect ratio (length along neck / width across strings)
        aspect_ratio = w / h

        # Guitar fretboards typically have aspect ratio 3:1 to 20:1
        # (relaxed from 5:1 to accommodate partial fretboard visibility)
        if aspect_ratio < 1.5 or aspect_ratio > 25.0:
            return False

        # Should have detected at least a few frets
        if len(self.fret_positions) < 3:
            return False

        return True


@dataclass
class VideoPosition:
    """A fret/string position detected from video."""
    string: int         # 1-6
    fret: int           # 0-24
    confidence: float   # 0-1
    # Additional detail about the detection
    finger_id: Optional[int] = None  # Which finger (0-4)
    is_pressing: bool = True  # Whether finger appears to be pressing


@dataclass
class FretboardDetectionConfig:
    """Configuration for fretboard detection."""
    # Canny edge detection thresholds
    canny_low: int = 30
    canny_high: int = 120

    # Hough transform parameters
    hough_threshold: int = 60
    hough_min_line_length: int = 40
    hough_max_line_gap: int = 15

    # Line angle tolerances (degrees)
    horizontal_angle_tolerance: float = 25.0  # For fret lines
    vertical_angle_tolerance: float = 25.0    # For string/edge lines

    # Clustering parameters
    fret_cluster_distance: int = 8
    string_cluster_distance: int = 15

    # Minimum required detections
    min_fret_lines: int = 2
    min_vertical_lines: int = 2

    # Adaptive thresholding
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Multi-scale detection
    use_multi_scale: bool = True
    scale_factors: list = field(default_factory=lambda: [1.0, 0.75, 0.5])

    # Angle-adaptive detection for angled guitars
    auto_detect_orientation: bool = True
    # Expected neck angle range (degrees from horizontal, 0=horizontal neck)
    neck_angle_range: tuple = (-60.0, 60.0)
    # Minimum lines to determine dominant angle
    min_lines_for_angle: int = 10

    # Region of interest (normalized 0-1 coordinates)
    # Used to focus detection on area where guitar is visible
    roi: Optional[dict] = None  # {'y_start': 0.4, 'y_end': 0.85, 'x_start': 0.0, 'x_end': 1.0}


# Standard guitar fret spacing ratios (based on 12-TET tuning)
# Distance from nut to fret n = scale_length * (1 - 2^(-n/12))
def calculate_fret_ratios(num_frets: int = 24) -> list[float]:
    """Calculate theoretical fret position ratios from the nut.

    Returns:
        List of positions as ratios of scale length (0 = nut, approaching 1 at high frets)
    """
    ratios = [0.0]  # Nut position
    for n in range(1, num_frets + 1):
        ratio = 1.0 - (2.0 ** (-n / 12.0))
        ratios.append(ratio)
    return ratios


STANDARD_FRET_RATIOS = calculate_fret_ratios(24)


def identify_fret_numbers(
    detected_positions: list[float],
    max_visible_frets: int = 15
) -> tuple[int, list[int]]:
    """Identify actual fret numbers from detected normalized positions.

    Uses RANSAC-style matching against 12-TET theoretical fret spacing ratios
    to determine which frets are visible. The key insight is that the ratio
    between consecutive fret spacings is constant (~0.9439), independent of
    scale length or camera zoom.

    Args:
        detected_positions: List of normalized fret positions (0-1), sorted
        max_visible_frets: Maximum number of frets expected to be visible

    Returns:
        Tuple of (starting_fret, fret_number_map) where:
        - starting_fret: The actual fret number of the first detected fret
        - fret_number_map: List mapping each detected index to actual fret number
    """
    if not detected_positions or len(detected_positions) < 2:
        return 0, list(range(len(detected_positions)))

    # Step 1: Filter noise from detected positions
    detected_spacings = []
    for i in range(len(detected_positions) - 1):
        detected_spacings.append(detected_positions[i + 1] - detected_positions[i])

    if not detected_spacings:
        return 0, list(range(len(detected_positions)))

    sorted_spacings = sorted(detected_spacings)
    median_spacing = sorted_spacings[len(sorted_spacings) // 2]

    # Keep positions with reasonable spacing (filter noise lines)
    filtered_indices = [0]
    for i, spacing in enumerate(detected_spacings):
        if 0.3 * median_spacing <= spacing <= 2.5 * median_spacing:
            filtered_indices.append(i + 1)

    filtered_positions = [detected_positions[i] for i in filtered_indices]

    if len(filtered_positions) < 3:
        return 0, list(range(len(detected_positions)))

    # Step 2: Calculate spacing ratios between consecutive filtered positions
    filtered_spacings = []
    for i in range(len(filtered_positions) - 1):
        filtered_spacings.append(filtered_positions[i + 1] - filtered_positions[i])

    # Step 3: RANSAC-style matching against 12-TET theory
    # Try each possible starting fret and score how well the detected spacing
    # pattern matches the theoretical pattern
    best_start_fret = 0
    best_score = -1.0
    num_detected = len(filtered_positions)

    # Precompute theoretical spacings for all possible starting frets
    for start_fret in range(0, 20):
        end_fret = start_fret + num_detected - 1
        if end_fret > 24:
            break

        # Get theoretical spacings for this starting fret
        theoretical_spacings = []
        for n in range(start_fret, start_fret + num_detected - 1):
            if n + 1 < len(STANDARD_FRET_RATIOS):
                theoretical_spacings.append(
                    STANDARD_FRET_RATIOS[n + 1] - STANDARD_FRET_RATIOS[n]
                )

        if len(theoretical_spacings) != len(filtered_spacings):
            continue

        # Normalize both spacing vectors to sum to 1 for scale-invariant comparison
        theo_sum = sum(theoretical_spacings)
        det_sum = sum(filtered_spacings)
        if theo_sum <= 0 or det_sum <= 0:
            continue

        theo_norm = [s / theo_sum for s in theoretical_spacings]
        det_norm = [s / det_sum for s in filtered_spacings]

        # Score: 1 - mean absolute difference of normalized spacings
        diffs = [abs(t - d) for t, d in zip(theo_norm, det_norm)]
        score = 1.0 - sum(diffs) / len(diffs)

        if score > best_score:
            best_score = score
            best_start_fret = start_fret

    # Step 4: Require reasonable confidence to override fret 0 assumption
    # If the score difference between best and fret-0 is small, prefer fret 0
    # (most common in guitar videos)
    if best_start_fret != 0 and best_score < 0.85:
        # Check score for start_fret=0
        end_fret_0 = num_detected - 1
        if end_fret_0 < len(STANDARD_FRET_RATIOS):
            theo_0 = []
            for n in range(num_detected - 1):
                if n + 1 < len(STANDARD_FRET_RATIOS):
                    theo_0.append(STANDARD_FRET_RATIOS[n + 1] - STANDARD_FRET_RATIOS[n])
            if len(theo_0) == len(filtered_spacings):
                t_sum = sum(theo_0)
                d_sum = sum(filtered_spacings)
                if t_sum > 0 and d_sum > 0:
                    tn = [s / t_sum for s in theo_0]
                    dn = [s / d_sum for s in filtered_spacings]
                    score_0 = 1.0 - sum(abs(t - d) for t, d in zip(tn, dn)) / len(tn)
                    # Only use non-zero start if significantly better
                    if best_score - score_0 < 0.1:
                        best_start_fret = 0

    # Step 5: Build mapping from detected indices to actual fret numbers
    fret_number_map = []
    for i, det_pos in enumerate(detected_positions):
        # Find closest filtered position
        min_dist = float('inf')
        closest_filtered_idx = 0
        for j, filt_pos in enumerate(filtered_positions):
            dist = abs(det_pos - filt_pos)
            if dist < min_dist:
                min_dist = dist
                closest_filtered_idx = j

        actual_fret = best_start_fret + closest_filtered_idx
        fret_number_map.append(actual_fret)

    return best_start_fret, fret_number_map


def _estimate_neck_orientation(
    lines: list,
    config: FretboardDetectionConfig
) -> tuple[float, float]:
    """Estimate the guitar neck orientation from detected lines.

    A guitar fretboard has two dominant line orientations:
    - Strings/edges: along the neck direction
    - Frets: perpendicular to the neck direction (90° offset from strings)

    We find these by analyzing the distribution of line angles.

    Args:
        lines: List of lines from Hough transform
        config: Detection configuration

    Returns:
        Tuple of (neck_angle, fret_angle) in degrees
        neck_angle: Direction of strings/neck (0 = horizontal pointing right)
        fret_angle: Direction of frets (should be ~neck_angle + 90)
    """
    if lines is None or len(lines) < config.min_lines_for_angle:
        return 0.0, 90.0  # Default to horizontal neck

    # Collect angles weighted by line length
    angle_weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Normalize to -90 to +90 (direction doesn't matter for orientation)
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle_weights.append((angle, line_length))

    # Bin angles into 5-degree buckets
    bins = {}
    for angle, weight in angle_weights:
        bin_idx = round(angle / 5) * 5
        if bin_idx not in bins:
            bins[bin_idx] = 0.0
        bins[bin_idx] += weight

    # Find the strongest angle clusters
    sorted_bins = sorted(bins.items(), key=lambda x: -x[1])

    if len(sorted_bins) < 2:
        return 0.0, 90.0

    # The dominant angle is likely the neck/string direction
    # Look for a second strong cluster that's roughly 90° away (frets)
    neck_angle = sorted_bins[0][0]

    # Expected fret angle (perpendicular to neck)
    expected_fret_angle = neck_angle + 90
    if expected_fret_angle > 90:
        expected_fret_angle -= 180
    elif expected_fret_angle < -90:
        expected_fret_angle += 180

    # Find the best matching fret angle among strong clusters
    fret_angle = expected_fret_angle  # Default to calculated
    best_fret_weight = 0

    for candidate_angle, candidate_weight in sorted_bins[1:15]:  # Check top 15
        # Check how close this is to expected perpendicular
        angle_diff = min(
            abs(candidate_angle - expected_fret_angle),
            abs(candidate_angle - expected_fret_angle + 180),
            abs(candidate_angle - expected_fret_angle - 180)
        )
        if angle_diff < 25 and candidate_weight > best_fret_weight:
            fret_angle = candidate_angle
            best_fret_weight = candidate_weight

    # Determine which is neck vs fret based on "horizontal-ness"
    # Guitar necks are typically more horizontal than vertical in videos
    neck_horiz_ness = 90 - abs(neck_angle)  # Higher = more horizontal
    fret_horiz_ness = 90 - abs(fret_angle)

    if fret_horiz_ness > neck_horiz_ness:
        # Swap - the "fret" angle is actually more horizontal (the neck)
        neck_angle, fret_angle = fret_angle, neck_angle

    return neck_angle, fret_angle


def detect_fretboard(
    frame: np.ndarray,
    config: Optional[FretboardDetectionConfig] = None
) -> FretboardGeometry | None:
    """Detect fretboard region and geometry in frame.

    Uses Canny edge detection and Hough line transform to identify
    the fretboard structure. Includes adaptive thresholding and
    multi-scale detection for robustness.

    Args:
        frame: BGR numpy array from video frame
        config: Detection configuration (uses defaults if None)

    Returns:
        FretboardGeometry if detected, None if detection fails
    """
    if frame is None or frame.size == 0:
        return None

    if config is None:
        config = FretboardDetectionConfig()

    height, width = frame.shape[:2]

    # Apply ROI if specified
    roi_offset_x = 0
    roi_offset_y = 0
    working_frame = frame

    if config.roi is not None:
        y_start = int(config.roi.get('y_start', 0) * height)
        y_end = int(config.roi.get('y_end', 1) * height)
        x_start = int(config.roi.get('x_start', 0) * width)
        x_end = int(config.roi.get('x_end', 1) * width)

        working_frame = frame[y_start:y_end, x_start:x_end]
        roi_offset_x = x_start
        roi_offset_y = y_start

    # Try detection at multiple scales if enabled
    if config.use_multi_scale:
        for scale in config.scale_factors:
            if scale != 1.0:
                scaled_frame = cv2.resize(
                    working_frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA
                )
            else:
                scaled_frame = working_frame

            result = _detect_fretboard_single_scale(scaled_frame, config)
            if result is not None:
                # Scale coordinates back to original frame size
                if scale != 1.0:
                    result = _scale_geometry(result, 1.0 / scale)

                # Offset coordinates for ROI
                if config.roi is not None:
                    result = _offset_geometry(result, roi_offset_x, roi_offset_y)

                result.frame_width = width
                result.frame_height = height
                return result
        return None
    else:
        result = _detect_fretboard_single_scale(working_frame, config)
        if result is not None:
            if config.roi is not None:
                result = _offset_geometry(result, roi_offset_x, roi_offset_y)
            result.frame_width = width
            result.frame_height = height
        return result


def _offset_geometry(geometry: FretboardGeometry, offset_x: float, offset_y: float) -> FretboardGeometry:
    """Offset geometry coordinates by adding x and y offsets."""
    return FretboardGeometry(
        top_left=(geometry.top_left[0] + offset_x, geometry.top_left[1] + offset_y),
        top_right=(geometry.top_right[0] + offset_x, geometry.top_right[1] + offset_y),
        bottom_left=(geometry.bottom_left[0] + offset_x, geometry.bottom_left[1] + offset_y),
        bottom_right=(geometry.bottom_right[0] + offset_x, geometry.bottom_right[1] + offset_y),
        fret_positions=geometry.fret_positions,
        string_positions=geometry.string_positions,
        detection_confidence=geometry.detection_confidence,
        rotation_angle=geometry.rotation_angle,
        frame_width=geometry.frame_width,
        frame_height=geometry.frame_height,
        actual_fret_numbers=geometry.actual_fret_numbers,
        starting_fret=geometry.starting_fret,
    )


def _detect_fretboard_single_scale(
    frame: np.ndarray,
    config: FretboardDetectionConfig
) -> FretboardGeometry | None:
    """Detect fretboard at a single scale."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing
    if config.use_adaptive_threshold:
        # Use CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter preserves edges while reducing noise
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with configured thresholds
    edges = cv2.Canny(blurred, config.canny_low, config.canny_high)

    # Optional: dilate edges slightly to connect broken lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Hough line transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap
    )

    if lines is None or len(lines) < 4:
        return None

    # Auto-detect neck orientation if enabled
    if config.auto_detect_orientation:
        neck_angle, fret_angle = _estimate_neck_orientation(lines, config)
    else:
        neck_angle = 0.0
        fret_angle = 90.0

    # Separate lines by angle relative to detected orientation
    # - Lines aligned with neck_angle are strings/edges
    # - Lines aligned with fret_angle are frets
    string_lines = []  # Strings and edges (along neck)
    fret_lines = []    # Frets (perpendicular to neck)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Normalize to -90 to +90
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Check if aligned with neck direction (strings)
        angle_diff_neck = min(
            abs(angle - neck_angle),
            abs(angle - neck_angle + 180),
            abs(angle - neck_angle - 180)
        )
        if angle_diff_neck < config.horizontal_angle_tolerance:
            string_lines.append((x1, y1, x2, y2, line_length, angle))
            continue

        # Check if aligned with fret direction
        angle_diff_fret = min(
            abs(angle - fret_angle),
            abs(angle - fret_angle + 180),
            abs(angle - fret_angle - 180)
        )
        if angle_diff_fret < config.vertical_angle_tolerance:
            fret_lines.append((x1, y1, x2, y2, line_length, angle))

    # Need minimum lines
    if len(fret_lines) < config.min_fret_lines or \
       len(string_lines) < config.min_vertical_lines:
        return None

    # Use detected orientation for reference
    rotation_angle = neck_angle

    # For angled fretboards, we need to cluster lines differently
    # Project fret lines onto the axis perpendicular to the neck
    # and project string lines onto the axis parallel to the neck

    # Calculate projection axis based on neck angle
    neck_rad = math.radians(rotation_angle)
    # Perpendicular to neck (for clustering frets)
    perp_x = math.cos(neck_rad + math.pi/2)
    perp_y = math.sin(neck_rad + math.pi/2)
    # Parallel to neck (for clustering strings)
    para_x = math.cos(neck_rad)
    para_y = math.sin(neck_rad)

    # Cluster fret lines by their projection onto the perpendicular axis
    fret_projections = []
    for l in fret_lines:
        mid_x = (l[0] + l[2]) / 2
        mid_y = (l[1] + l[3]) / 2
        # Project onto neck direction to get position along neck
        proj = mid_x * para_x + mid_y * para_y
        fret_projections.append((proj, l[4]))  # (projection, weight=length)

    fret_x_positions = _cluster_projections_weighted(
        fret_projections,
        min_distance=config.fret_cluster_distance * 2  # Scale for projection
    )

    # Cluster string lines by their projection onto the parallel axis
    string_projections = []
    for l in string_lines:
        mid_x = (l[0] + l[2]) / 2
        mid_y = (l[1] + l[3]) / 2
        # Project onto perpendicular to get string position
        proj = mid_x * perp_x + mid_y * perp_y
        string_projections.append((proj, l[4]))

    edge_y_positions = _cluster_projections_weighted(
        string_projections,
        min_distance=config.string_cluster_distance * 2
    )

    # Fallback: if clustering collapsed all string lines into one group
    # (can happen when the fretboard is small in the frame and strings are
    # close together), use the full spread (min/max) of the raw projections.
    if len(edge_y_positions) < 2 and string_projections:
        all_string_projs = [p for p, _ in string_projections]
        span = max(all_string_projs) - min(all_string_projs)
        if span > 10:  # Only if there is meaningful spread
            edge_y_positions = [min(all_string_projs), max(all_string_projs)]

    if len(fret_x_positions) < 2 or len(edge_y_positions) < 2:
        return None

    # Sort positions (these are now projections onto the neck axis)
    fret_x_positions.sort()
    edge_y_positions.sort()

    # Projection bounds
    neck_start = fret_x_positions[0]  # Nut end
    neck_end = fret_x_positions[-1]   # Body end
    string_top = edge_y_positions[0]  # Top string (string 6)
    string_bottom = edge_y_positions[-1]  # Bottom string (string 1)

    # Calculate dimensions in projection space
    fret_width = neck_end - neck_start
    fret_height = string_bottom - string_top

    if fret_width <= 0 or fret_height <= 0:
        return None

    # Convert projection coordinates back to pixel coordinates for the corner points
    # For an angled neck, the corners form a parallelogram
    neck_rad = math.radians(rotation_angle)
    cos_a = math.cos(neck_rad)
    sin_a = math.sin(neck_rad)
    # Perpendicular direction
    cos_p = math.cos(neck_rad + math.pi/2)
    sin_p = math.sin(neck_rad + math.pi/2)

    # Find the fretboard region by computing the oriented bounding box
    # Use the extreme projection values and convert back to pixel coordinates

    # Find a reference point (center of detected lines)
    all_mid_x = []
    all_mid_y = []
    for l in fret_lines + string_lines:
        all_mid_x.append((l[0] + l[2]) / 2)
        all_mid_y.append((l[1] + l[3]) / 2)

    if not all_mid_x:
        return None

    center_x = sum(all_mid_x) / len(all_mid_x)
    center_y = sum(all_mid_y) / len(all_mid_y)

    # The corners of the oriented bounding box in projection space
    # neck_start/end are along the neck axis (para_x, para_y)
    # string_top/bottom are along the perpendicular axis (perp_x, perp_y)

    # Convert projection bounds back to pixel coordinates
    # For each corner, we need to find the pixel coordinates that correspond
    # to the projection values (neck_start, string_top), (neck_end, string_top), etc.

    # The projection equations are:
    #   proj_along_neck = x * para_x + y * para_y
    #   proj_perp_neck = x * perp_x + y * perp_y

    # To invert, we solve for x and y given the two projections
    # This is a linear system: [para_x, para_y; perp_x, perp_y] * [x; y] = [proj_along; proj_perp]
    # The inverse is simply the transpose since the vectors are orthonormal

    def proj_to_pixel(proj_along: float, proj_perp: float) -> tuple[float, float]:
        # Since (para_x, para_y) and (perp_x, perp_y) are orthonormal
        # x = proj_along * para_x + proj_perp * perp_x
        # y = proj_along * para_y + proj_perp * perp_y
        x = proj_along * para_x + proj_perp * perp_x
        y = proj_along * para_y + proj_perp * perp_y
        return (x, y)

    # The four corners (along neck direction, perpendicular direction)
    top_left = proj_to_pixel(neck_start, string_top)
    top_right = proj_to_pixel(neck_end, string_top)
    bottom_left = proj_to_pixel(neck_start, string_bottom)
    bottom_right = proj_to_pixel(neck_end, string_bottom)

    # Normalize fret positions within the detected region
    normalized_frets = [
        (x - neck_start) / fret_width for x in fret_x_positions
    ]

    # Identify actual fret numbers by matching to theoretical spacing
    starting_fret, actual_fret_numbers = identify_fret_numbers(normalized_frets)

    # Generate 6 string positions (evenly distributed)
    # String 6 (low E) at top, String 1 (high E) at bottom
    normalized_strings = [i / 5.0 for i in range(6)]

    # Calculate detection confidence
    confidence = _calculate_detection_confidence(
        normalized_frets, fret_width, fret_height, len(fret_lines)
    )

    geometry = FretboardGeometry(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        fret_positions=normalized_frets,
        string_positions=normalized_strings,
        detection_confidence=confidence,
        rotation_angle=rotation_angle,
        actual_fret_numbers=actual_fret_numbers,
        starting_fret=starting_fret,
    )

    # Validate geometry
    if not geometry.is_valid():
        return None

    return geometry


def _scale_geometry(geometry: FretboardGeometry, scale: float) -> FretboardGeometry:
    """Scale geometry coordinates by a factor."""
    return FretboardGeometry(
        top_left=(geometry.top_left[0] * scale, geometry.top_left[1] * scale),
        top_right=(geometry.top_right[0] * scale, geometry.top_right[1] * scale),
        bottom_left=(geometry.bottom_left[0] * scale, geometry.bottom_left[1] * scale),
        bottom_right=(geometry.bottom_right[0] * scale, geometry.bottom_right[1] * scale),
        fret_positions=geometry.fret_positions,  # Normalized, no scaling needed
        string_positions=geometry.string_positions,
        detection_confidence=geometry.detection_confidence,
        rotation_angle=geometry.rotation_angle,
        actual_fret_numbers=geometry.actual_fret_numbers,
        starting_fret=geometry.starting_fret,
    )


def _calculate_detection_confidence(
    fret_positions: list[float],
    width: float,
    height: float,
    num_lines: int
) -> float:
    """Calculate confidence score for fretboard detection.

    Args:
        fret_positions: Normalized fret positions
        width: Fretboard width in pixels
        height: Fretboard height in pixels
        num_lines: Number of horizontal lines detected

    Returns:
        Confidence score 0-1
    """
    confidence = 0.5  # Base confidence

    # Bonus for detecting more frets (up to +0.2)
    fret_bonus = min(0.2, len(fret_positions) * 0.02)
    confidence += fret_bonus

    # Bonus for good aspect ratio (up to +0.15)
    aspect_ratio = width / height if height > 0 else 0
    if 5.0 <= aspect_ratio <= 12.0:  # Ideal range for guitar fretboard
        confidence += 0.15
    elif 3.0 <= aspect_ratio <= 15.0:  # Acceptable range
        confidence += 0.08

    # Bonus for consistent fret spacing (up to +0.15)
    if len(fret_positions) >= 3:
        spacings = [
            fret_positions[i+1] - fret_positions[i]
            for i in range(len(fret_positions) - 1)
        ]
        if spacings:
            mean_spacing = sum(spacings) / len(spacings)
            if mean_spacing > 0:
                variance = sum((s - mean_spacing)**2 for s in spacings) / len(spacings)
                # Low variance = consistent spacing
                if variance < 0.01:
                    confidence += 0.15
                elif variance < 0.03:
                    confidence += 0.08

    return min(1.0, confidence)


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


def _cluster_projections_weighted(
    projections: list[tuple[float, float]],
    min_distance: float
) -> list[float]:
    """Cluster projection values with weighting.

    Args:
        projections: List of (projection_value, weight) tuples
        min_distance: Minimum distance between clusters

    Returns:
        List of weighted average positions for each cluster
    """
    if not projections:
        return []

    # Sort by projection value
    sorted_projs = sorted(projections, key=lambda x: x[0])

    if not sorted_projs:
        return []

    # Cluster nearby projections
    clusters = [[sorted_projs[0]]]
    for proj, weight in sorted_projs[1:]:
        last_proj = clusters[-1][-1][0]
        if proj - last_proj < min_distance:
            clusters[-1].append((proj, weight))
        else:
            clusters.append([(proj, weight)])

    # Return weighted average for each cluster
    result = []
    for cluster in clusters:
        total_weight = sum(w for _, w in cluster)
        if total_weight > 0:
            weighted_avg = sum(p * w for p, w in cluster) / total_weight
        else:
            weighted_avg = sum(p for p, _ in cluster) / len(cluster)
        result.append(weighted_avg)

    return result


def _cluster_line_positions_weighted(
    lines: list[tuple[int, int, int, int]],
    weights: list[float],
    axis: str,
    min_distance: int
) -> list[float]:
    """Cluster line positions with weighting (e.g., by line length).

    Args:
        lines: List of (x1, y1, x2, y2) tuples
        weights: Weight for each line (e.g., line length)
        axis: 'x' or 'y' to indicate which axis to cluster
        min_distance: Minimum pixel distance between clusters

    Returns:
        List of weighted average positions for each cluster
    """
    if not lines:
        return []

    # Calculate positions with weights
    if axis == 'y':
        pos_weight_pairs = [
            ((line[1] + line[3]) / 2, w)
            for line, w in zip(lines, weights)
        ]
    else:
        pos_weight_pairs = [
            ((line[0] + line[2]) / 2, w)
            for line, w in zip(lines, weights)
        ]

    # Sort by position
    pos_weight_pairs.sort(key=lambda x: x[0])

    if not pos_weight_pairs:
        return []

    # Cluster nearby positions
    clusters = [[pos_weight_pairs[0]]]
    for pw in pos_weight_pairs[1:]:
        pos, _ = pw
        last_pos = clusters[-1][-1][0]
        if pos - last_pos < min_distance:
            clusters[-1].append(pw)
        else:
            clusters.append([pw])

    # Return weighted average position for each cluster
    result = []
    for cluster in clusters:
        total_weight = sum(w for _, w in cluster)
        if total_weight > 0:
            weighted_avg = sum(p * w for p, w in cluster) / total_weight
        else:
            weighted_avg = sum(p for p, _ in cluster) / len(cluster)
        result.append(weighted_avg)

    return result


def map_finger_to_position(
    finger_x: float,
    finger_y: float,
    geometry: FretboardGeometry,
    finger_z: float = 0.0,
    finger_id: Optional[int] = None
) -> VideoPosition | None:
    """Map a finger position to fret/string coordinates.

    Handles angled fretboards by transforming finger coordinates into the
    fretboard's local coordinate system.

    Args:
        finger_x: X coordinate in pixels (or normalized if frame dims unknown)
        finger_y: Y coordinate in pixels (or normalized if frame dims unknown)
        geometry: Detected fretboard geometry
        finger_z: Depth value from hand detection (for press detection)
        finger_id: Which finger (0-4, for tracking)

    Returns:
        VideoPosition if finger is on fretboard, None otherwise
    """
    # For angled fretboards, we need to transform the finger position
    # into the fretboard's local coordinate system.
    #
    # The fretboard is defined by 4 corners forming a parallelogram:
    # top_left (nut, string 6) -- top_right (body, string 6)
    #     |                           |
    # bottom_left (nut, string 1) -- bottom_right (body, string 1)
    #
    # We want to find (rel_x, rel_y) where:
    # - rel_x = 0 at nut, rel_x = 1 at body (along neck)
    # - rel_y = 0 at string 6 (top), rel_y = 1 at string 1 (bottom)

    # Calculate the basis vectors for the fretboard coordinate system
    # Vector along the neck (from nut to body at string 6)
    neck_vec_x = geometry.top_right[0] - geometry.top_left[0]
    neck_vec_y = geometry.top_right[1] - geometry.top_left[1]

    # Vector across strings (from string 6 to string 1 at nut)
    string_vec_x = geometry.bottom_left[0] - geometry.top_left[0]
    string_vec_y = geometry.bottom_left[1] - geometry.top_left[1]

    # Calculate the neck length and string width
    neck_length = math.sqrt(neck_vec_x**2 + neck_vec_y**2)
    string_width = math.sqrt(string_vec_x**2 + string_vec_y**2)

    if neck_length <= 0 or string_width <= 0:
        return None

    # Normalize the basis vectors
    neck_unit_x = neck_vec_x / neck_length
    neck_unit_y = neck_vec_y / neck_length
    string_unit_x = string_vec_x / string_width
    string_unit_y = string_vec_y / string_width

    # Vector from top_left (origin) to finger position
    finger_vec_x = finger_x - geometry.top_left[0]
    finger_vec_y = finger_y - geometry.top_left[1]

    # Project finger position onto the fretboard coordinate system
    # rel_x = dot(finger_vec, neck_unit) / neck_length
    # rel_y = dot(finger_vec, string_unit) / string_width
    rel_x = (finger_vec_x * neck_unit_x + finger_vec_y * neck_unit_y) / neck_length
    rel_y = (finger_vec_x * string_unit_x + finger_vec_y * string_unit_y) / string_width

    # Check if finger is within fretboard bounds (with margin)
    margin_x = 0.15  # Along neck
    margin_y = 0.2   # Across strings

    if rel_x < -margin_x or rel_x > 1 + margin_x:
        return None
    if rel_y < -margin_y or rel_y > 1 + margin_y:
        return None

    # Determine if finger is pressing (based on z-depth)
    is_pressing = finger_z < -0.02

    # Find nearest fret (where finger should be pressing)
    fret = _find_nearest_fret_smart(
        rel_x,
        geometry.fret_positions,
        actual_fret_numbers=geometry.actual_fret_numbers,
        starting_fret=geometry.starting_fret
    )
    if fret is None:
        return None

    # Find nearest string
    string = _find_nearest_string(rel_y, geometry.string_positions)
    if string is None:
        return None

    # Calculate confidence based on multiple factors
    confidence = _calculate_position_confidence(
        rel_x, rel_y, fret, string, geometry, finger_z
    )

    return VideoPosition(
        string=string,
        fret=fret,
        confidence=confidence,
        finger_id=finger_id,
        is_pressing=is_pressing,
    )


def _calculate_position_confidence(
    rel_x: float,
    rel_y: float,
    fret: int,
    string: int,
    geometry: FretboardGeometry,
    finger_z: float
) -> float:
    """Calculate confidence for a finger-to-position mapping.

    Considers:
    - Proximity to detected fret position
    - Proximity to string position
    - Finger depth (z) suggesting actual pressing
    - Overall fretboard detection confidence

    Args:
        rel_x: Relative x position (0-1)
        rel_y: Relative y position (0-1)
        fret: Detected fret number
        string: Detected string number (1-6)
        geometry: Fretboard geometry
        finger_z: Finger depth from hand detection

    Returns:
        Combined confidence score (0-1)
    """
    # Base confidence from fretboard detection
    base_confidence = geometry.detection_confidence * 0.3

    # Fret proximity confidence
    fret_conf = _calculate_proximity_confidence(
        rel_x, geometry.fret_positions, fret
    )

    # String proximity confidence
    string_idx = 6 - string  # Convert string 1-6 to index 0-5
    string_conf = _calculate_proximity_confidence(
        rel_y, geometry.string_positions, string_idx
    )

    # Depth confidence (more negative z = more likely pressing)
    if finger_z < -0.05:
        depth_conf = 0.9
    elif finger_z < -0.02:
        depth_conf = 0.7
    elif finger_z < 0:
        depth_conf = 0.5
    else:
        depth_conf = 0.3

    # Weighted combination
    combined = (
        base_confidence +
        fret_conf * 0.3 +
        string_conf * 0.25 +
        depth_conf * 0.15
    )

    return min(1.0, combined)


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


def _find_nearest_fret_smart(
    rel_x: float,
    fret_positions: list[float],
    actual_fret_numbers: list[int] = None,
    starting_fret: int = 0
) -> int | None:
    """Find the fret being played based on finger position.

    Uses the detected fret line positions directly when available.
    The detected fret lines, even with some noise, give a better indication
    of actual fret locations than theoretical spacing.

    Guitar fretting technique: finger presses BEHIND the fret wire.
    The fret number is determined by the fret wire immediately in front
    of (higher position than) the finger.

    Args:
        rel_x: Relative x position (0-1) within fretboard
        fret_positions: List of fret positions (normalized)
        actual_fret_numbers: Optional list mapping each index to actual fret number
        starting_fret: The actual fret number of the first detected fret

    Returns:
        Actual fret number being played, or None
    """
    # Clamp to valid range
    rel_x = max(0.0, min(1.0, rel_x))

    # Very close to start = open string (fret 0)
    if rel_x < 0.02:
        return 0

    # Filter detected fret positions to remove noise
    # Keep only positions with reasonable spacing
    filtered_positions = []
    prev_pos = -1
    for pos in sorted(fret_positions):
        if prev_pos < 0 or pos - prev_pos > 0.03:  # At least 3% spacing
            filtered_positions.append(pos)
            prev_pos = pos

    # If we have good detected fret positions, use them directly
    if len(filtered_positions) >= 4:
        # Find which detected fret position the finger is before
        for i, fret_pos in enumerate(filtered_positions):
            if rel_x < fret_pos:
                if i == 0:
                    # Before first detected fret
                    if starting_fret == 0:
                        return 0  # Open string
                    else:
                        return starting_fret
                else:
                    # Between detected fret i-1 and i
                    prev_pos = filtered_positions[i - 1]
                    fret_space = fret_pos - prev_pos
                    # Finger in latter 60% of fret space = pressing this fret
                    threshold = prev_pos + fret_space * 0.4
                    # Use actual_fret_numbers if available, else offset from starting_fret
                    fret_at_i = actual_fret_numbers[i] if actual_fret_numbers and i < len(actual_fret_numbers) else starting_fret + i
                    fret_at_prev = actual_fret_numbers[i - 1] if actual_fret_numbers and (i - 1) < len(actual_fret_numbers) else starting_fret + i - 1
                    if rel_x >= threshold:
                        return fret_at_i
                    else:
                        return fret_at_prev

        # Finger is past all detected frets
        last_fret = actual_fret_numbers[-1] if actual_fret_numbers else starting_fret + len(filtered_positions) - 1
        return last_fret

    # Fall back to theoretical mapping if not enough detected positions
    # Use a conservative scale factor
    MAX_VISIBLE_FRET = 10
    VISIBLE_RANGE_SCALE = 2.0  # Conservative: assume fret 10 at rel_x=1.0

    for fret_num in range(1, MAX_VISIBLE_FRET + 1):
        fret_wire_pos = STANDARD_FRET_RATIOS[fret_num]
        scaled_fret_pos = fret_wire_pos * VISIBLE_RANGE_SCALE

        if rel_x < scaled_fret_pos:
            if fret_num == 1:
                if rel_x < scaled_fret_pos * 0.5:
                    return 0
                else:
                    return 1
            else:
                prev_fret_pos = STANDARD_FRET_RATIOS[fret_num - 1] * VISIBLE_RANGE_SCALE
                fret_space = scaled_fret_pos - prev_fret_pos
                threshold = prev_fret_pos + fret_space * 0.4
                if rel_x >= threshold:
                    return fret_num
                else:
                    return fret_num - 1

    return MAX_VISIBLE_FRET


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


def detect_fretboard_from_video(
    video_path: str,
    num_sample_frames: int = 5,
    roi: dict = None
) -> FretboardGeometry | None:
    """Detect fretboard geometry from video using multiple frames.

    Samples multiple frames and uses the detection with highest confidence.

    Args:
        video_path: Path to video file
        num_sample_frames: Number of frames to sample
        roi: Optional ROI dict with x1, y1, x2, y2 (normalized 0-1)

    Returns:
        FretboardGeometry with highest confidence, or None
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Sample frames evenly throughout the video (skip first/last 10%)
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    frame_step = max(1, (end_frame - start_frame) // num_sample_frames)

    best_geometry = None
    best_confidence = 0.0

    for i in range(num_sample_frames):
        frame_idx = start_frame + i * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Apply ROI cropping if provided
        if roi is not None:
            h, w = frame.shape[:2]
            x1 = int(roi['x1'] * w)
            y1 = int(roi['y1'] * h)
            x2 = int(roi['x2'] * w)
            y2 = int(roi['y2'] * h)
            frame = frame[y1:y2, x1:x2]

        geometry = detect_fretboard(frame)
        if geometry and geometry.detection_confidence > best_confidence:
            best_geometry = geometry
            best_confidence = geometry.detection_confidence

    cap.release()
    return best_geometry


def track_fretboard_temporal(
    video_path: str,
    timestamps: list[float],
    initial_geometry: Optional[FretboardGeometry] = None
) -> dict[float, FretboardGeometry]:
    """Track fretboard geometry across multiple timestamps.

    Uses temporal consistency to improve detection reliability.

    Args:
        video_path: Path to video file
        timestamps: List of timestamps to track
        initial_geometry: Optional starting geometry to use as reference

    Returns:
        Dictionary mapping timestamps to FretboardGeometry
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {}

    results = {}
    previous_geometry = initial_geometry

    for ts in sorted(timestamps):
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        geometry = detect_fretboard(frame)

        if geometry:
            # Check temporal consistency with previous detection
            if previous_geometry:
                geometry = _apply_temporal_smoothing(geometry, previous_geometry)

            results[ts] = geometry
            previous_geometry = geometry
        elif previous_geometry:
            # Use previous geometry if detection fails (with lower confidence)
            smoothed = FretboardGeometry(
                top_left=previous_geometry.top_left,
                top_right=previous_geometry.top_right,
                bottom_left=previous_geometry.bottom_left,
                bottom_right=previous_geometry.bottom_right,
                fret_positions=previous_geometry.fret_positions,
                string_positions=previous_geometry.string_positions,
                detection_confidence=previous_geometry.detection_confidence * 0.8,
                rotation_angle=previous_geometry.rotation_angle,
                frame_width=previous_geometry.frame_width,
                frame_height=previous_geometry.frame_height,
            )
            results[ts] = smoothed

    cap.release()
    return results


def _apply_temporal_smoothing(
    current: FretboardGeometry,
    previous: FretboardGeometry,
    smoothing_factor: float = 0.3
) -> FretboardGeometry:
    """Apply temporal smoothing to reduce jitter in fretboard detection.

    Args:
        current: Current frame's detection
        previous: Previous frame's detection
        smoothing_factor: How much to blend with previous (0-1, higher = more smoothing)

    Returns:
        Smoothed geometry
    """
    # Blend corner positions
    def blend_point(curr: tuple, prev: tuple) -> tuple:
        return (
            curr[0] * (1 - smoothing_factor) + prev[0] * smoothing_factor,
            curr[1] * (1 - smoothing_factor) + prev[1] * smoothing_factor,
        )

    # Only smooth if geometries are similar (not a major change)
    position_diff = abs(current.top_left[0] - previous.top_left[0])
    if position_diff > current.width * 0.3:
        # Major change - don't smooth, use current
        return current

    smoothed = FretboardGeometry(
        top_left=blend_point(current.top_left, previous.top_left),
        top_right=blend_point(current.top_right, previous.top_right),
        bottom_left=blend_point(current.bottom_left, previous.bottom_left),
        bottom_right=blend_point(current.bottom_right, previous.bottom_right),
        fret_positions=current.fret_positions,  # Don't smooth internal structure
        string_positions=current.string_positions,
        detection_confidence=max(current.detection_confidence, previous.detection_confidence * 0.9),
        rotation_angle=current.rotation_angle * (1 - smoothing_factor) + previous.rotation_angle * smoothing_factor,
        frame_width=current.frame_width,
        frame_height=current.frame_height,
    )

    return smoothed
