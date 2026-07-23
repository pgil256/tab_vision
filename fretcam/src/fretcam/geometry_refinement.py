"""Training-free live refinement of FretCam's fret and string coordinate frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import RULE_OF_18_RATIO

MIN_BODY_JOINT_FRET = 10
MAX_BODY_JOINT_FRET = 19
UNEXPLAINED_PEAK_PENALTY = 0.65


@dataclass(frozen=True)
class GeometryRefinement:
    """One conservative local refinement derived from the rectified neck."""

    homography: Homography
    fret_centers: np.ndarray | None
    string_curves: tuple[tuple[float, float, float], ...]
    fret_support: float
    string_support: float
    nut_x: float | None
    body_x: float | None
    distortion_residual: float
    boundary_support: float = 0.0
    body_joint_fret: int | None = None


def _wire_xs(fret_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(fret_centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or not np.all(np.isfinite(centers)):
        return np.empty(0, dtype=np.float64)
    first_center = 1.0 - RULE_OF_18_RATIO**0.5
    second_center = 1.0 - RULE_OF_18_RATIO**1.5
    denominator = second_center - first_center
    if abs(denominator) <= 1e-12:
        return np.empty(0, dtype=np.float64)
    scale = float((centers[1] - centers[0]) / denominator)
    origin = float(centers[0] - scale * first_center)
    frets = np.arange(centers.size + 1, dtype=np.float64)
    return origin + scale * (1.0 - np.power(RULE_OF_18_RATIO, frets))


def _project_to_image(homography: Homography, points: np.ndarray) -> np.ndarray:
    canonical = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate(
        [canonical, np.ones((canonical.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = homogeneous @ homography.H.T
    safe = np.abs(projected[:, 2]) > 1e-9
    output = np.full((canonical.shape[0], 2), np.nan, dtype=np.float64)
    output[safe] = projected[safe, :2] / projected[safe, 2, None]
    return output


def _local_peak(
    profile: np.ndarray,
    expected: float,
    radius: int,
) -> tuple[float, float] | None:
    center = int(round(expected))
    left = max(0, center - radius)
    right = min(profile.size, center + radius + 1)
    if right - left < 3:
        return None
    window = np.asarray(profile[left:right], dtype=np.float64)
    peak_index = int(np.argmax(window))
    peak = float(window[peak_index])
    baseline = float(np.median(window))
    spread = float(np.median(np.abs(window - baseline))) + 1e-6
    prominence = (peak - baseline) / spread
    if prominence < 2.0:
        return None
    return float(left + peak_index), float(np.clip(prominence / 10.0, 0.0, 1.0))


def _profile_peaks(profile: np.ndarray, *, minimum_distance: int) -> list[int]:
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 3:
        return []
    smoothed = cv2.GaussianBlur(
        values.reshape(1, -1),
        (0, 0),
        sigmaX=1.2,
    ).reshape(-1)
    baseline = float(np.median(smoothed))
    spread = float(np.median(np.abs(smoothed - baseline))) + 1e-6
    threshold = baseline + 2.5 * spread
    candidates = [
        index
        for index in range(1, smoothed.size - 1)
        if smoothed[index] >= threshold
        and smoothed[index] >= smoothed[index - 1]
        and smoothed[index] > smoothed[index + 1]
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda value: smoothed[value], reverse=True):
        if all(abs(index - prior) >= minimum_distance for prior in selected):
            selected.append(index)
    return sorted(selected)


def _unexplained_peak_fraction(
    peaks: list[int],
    expected: np.ndarray,
    tolerances: np.ndarray,
    *,
    nut_px: float,
    body_px: float,
) -> float:
    """Return the share of observed neck peaks not explained by a candidate axis."""
    observed = [peak for peak in peaks if nut_px <= peak <= body_px]
    if not observed:
        return 1.0
    explained = sum(
        any(
            abs(float(peak) - float(expected_px)) <= float(tolerance)
            for expected_px, tolerance in zip(expected, tolerances, strict=True)
        )
        for peak in observed
    )
    return 1.0 - explained / len(observed)


def _infer_rule18_axis(
    profile: np.ndarray,
    *,
    max_fret: int,
) -> tuple[np.ndarray, float, float, int, float] | None:
    """Detect nut/body boundaries and infer their fret number from wire support."""
    values = np.asarray(profile, dtype=np.float64)
    width = values.size
    if width < 80:
        return None
    peaks = _profile_peaks(values, minimum_distance=max(4, width // 100))
    left_candidates = [index for index in peaks if index <= 0.14 * (width - 1)]
    right_candidates = [index for index in peaks if index >= 0.86 * (width - 1)]
    if not left_candidates or not right_candidates:
        return None
    nut_px = min(left_candidates)
    body_px = max(right_candidates)
    if body_px - nut_px < 0.65 * (width - 1):
        return None

    baseline = float(np.median(values))
    spread = float(np.median(np.abs(values - baseline))) + 1e-6
    best: tuple[float, int] | None = None
    maximum_body_fret = min(MAX_BODY_JOINT_FRET, max_fret)
    for body_fret in range(MIN_BODY_JOINT_FRET, maximum_body_fret + 1):
        denominator = 1.0 - RULE_OF_18_RATIO**body_fret
        expected = (
            nut_px
            + (body_px - nut_px)
            * (
                1.0
                - np.power(
                    RULE_OF_18_RATIO,
                    np.arange(body_fret + 1, dtype=np.float64),
                )
            )
            / denominator
        )
        qualities: list[float] = []
        tolerances: list[float] = []
        hits = 0
        for index, expected_px in enumerate(expected):
            left_gap = (
                expected_px - expected[index - 1]
                if index > 0
                else expected[1] - expected[0]
            )
            right_gap = (
                expected[index + 1] - expected_px
                if index + 1 < expected.size
                else left_gap
            )
            local_gap = min(left_gap, right_gap)
            tolerances.append(max(2.0, 0.35 * local_gap))
            radius = max(2, round(0.25 * local_gap))
            peak = _local_peak(values, float(expected_px), radius)
            if peak is None:
                qualities.append(0.0)
                continue
            observed_px, prominence = peak
            distance_score = math.exp(
                -0.5 * ((observed_px - expected_px) / max(0.35 * local_gap, 1.0)) ** 2
            )
            qualities.append(prominence * distance_score)
            hits += 1
        coverage = hits / max(len(expected), 1)
        unexplained_fraction = _unexplained_peak_fraction(
            peaks,
            expected,
            np.asarray(tolerances, dtype=np.float64),
            nut_px=float(nut_px),
            body_px=float(body_px),
        )
        explanation_score = 1.0 - UNEXPLAINED_PEAK_PENALTY * unexplained_fraction
        score = (
            coverage
            * (float(np.mean(qualities)) if qualities else 0.0)
            * explanation_score
        )
        if best is None or score > best[0]:
            best = (score, body_fret)
    if best is None or best[0] < 0.12:
        return None

    body_fret = best[1]
    nut_x = nut_px / (width - 1)
    body_x = body_px / (width - 1)
    denominator = 1.0 - RULE_OF_18_RATIO**body_fret
    cell_frets = np.arange(max_fret + 1, dtype=np.float64) + 0.5
    centers = (
        nut_x
        + (body_x - nut_x)
        * (1.0 - np.power(RULE_OF_18_RATIO, cell_frets))
        / denominator
    )
    boundary_strength = min(
        1.0,
        max(0.0, (values[nut_px] - baseline) / (8.0 * spread)),
        max(0.0, (values[body_px] - baseline) / (8.0 * spread)),
    )
    support = float(np.clip(0.65 * best[0] + 0.35 * boundary_strength, 0.0, 1.0))
    return centers, float(nut_x), float(body_x), body_fret, support


def _fit_string_curves(
    rectified_gray: np.ndarray,
    *,
    n_strings: int,
) -> tuple[tuple[tuple[float, float, float], ...], float, float]:
    height, width = rectified_gray.shape
    if n_strings < 2 or width < 30 or height < 30:
        return (), 0.0, 0.0
    gradient = np.abs(cv2.Sobel(rectified_gray, cv2.CV_32F, 0, 1, ksize=3))
    x_samples = np.linspace(0.10, 0.90, 7, dtype=np.float64)
    curves: list[tuple[float, float, float]] = []
    qualities: list[float] = []
    residuals: list[float] = []
    search_radius = max(3, round(height / (2.4 * n_strings)))
    for string_index in range(n_strings):
        expected_y = string_index * (height - 1) / (n_strings - 1)
        xs: list[float] = []
        ys: list[float] = []
        qs: list[float] = []
        for x_fraction in x_samples:
            x = int(round(x_fraction * (width - 1)))
            x0 = max(0, x - max(2, width // 40))
            x1 = min(width, x + max(3, width // 40))
            profile = np.median(gradient[:, x0:x1], axis=1)
            peak = _local_peak(profile, expected_y, search_radius)
            if peak is None:
                continue
            y, quality = peak
            xs.append(float(x_fraction))
            ys.append(y / max(height - 1, 1))
            qs.append(quality)
        if len(ys) < 2:
            # Preserve the physical string index. Consumers ignore non-finite
            # curves instead of accidentally renumbering a partially observed
            # six-string set.
            curves.append((math.nan, math.nan, math.nan))
            continue
        degree = min(2, len(ys) - 1)
        sample_xs = np.asarray(xs, dtype=np.float64)
        coefficients = np.polyfit(sample_xs, np.asarray(ys), degree)
        if degree == 1:
            coefficients = np.asarray((0.0, coefficients[0], coefficients[1]))
        elif degree == 0:
            coefficients = np.asarray((0.0, 0.0, coefficients[0]))
        prediction = np.polyval(coefficients, sample_xs)
        residuals.extend(np.abs(prediction - np.asarray(ys)).tolist())
        curves.append(tuple(float(value) for value in coefficients))
        qualities.append(float(np.mean(qs)))
    distortion = float(max(residuals, default=0.0))
    spacing_deficit = _string_curve_spacing_deficit(
        tuple(curves),
        n_strings=n_strings,
    )
    distortion = max(distortion, spacing_deficit)
    support = len(qualities) / max(n_strings, 1)
    residual_quality = math.exp(-distortion / 0.015)
    quality = (
        support * (float(np.mean(qualities)) if qualities else 0.0) * residual_quality
    )
    if spacing_deficit > 0.0:
        return (
            tuple((math.nan, math.nan, math.nan) for _ in range(n_strings)),
            0.0,
            distortion,
        )
    return tuple(curves), float(np.clip(quality, 0.0, 1.0)), distortion


def _string_curve_spacing_deficit(
    curves: tuple[tuple[float, float, float], ...],
    *,
    n_strings: int,
) -> float:
    """Return how far fitted curves violate a conservative non-crossing gap."""
    valid_curves = [
        (index, np.asarray(curve, dtype=np.float64))
        for index, curve in enumerate(curves)
        if np.all(np.isfinite(curve))
    ]
    spacing_probe = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    maximum_deficit = 0.0
    for (left_index, left_curve), (right_index, right_curve) in zip(
        valid_curves,
        valid_curves[1:],
        strict=False,
    ):
        gaps = np.polyval(right_curve, spacing_probe) - np.polyval(
            left_curve,
            spacing_probe,
        )
        expected_gap = (right_index - left_index) / max(n_strings - 1, 1)
        maximum_deficit = max(
            maximum_deficit,
            float(max(0.0, 0.30 * expected_gap - np.min(gaps))),
        )
    return maximum_deficit


def refine_live_geometry(
    frame: np.ndarray,
    homography: Homography,
    fret_centers: np.ndarray | None,
    cfg: GuitarConfig,
    *,
    rectified_width: int = 720,
    rectified_height: int = 180,
) -> GeometryRefinement:
    """Refine a detected board from local fret-wire and string image evidence.

    The detector remains authoritative. Local gradients may move its coordinate
    frame only when enough expected wires/strings have nearby support, and the
    resulting canonical-corner innovation stays small.
    """
    if homography.confidence <= 0.0 or frame.ndim != 3 or frame.shape[-1] != 3:
        return GeometryRefinement(
            homography=homography,
            fret_centers=fret_centers,
            string_curves=(),
            fret_support=0.0,
            string_support=0.0,
            nut_x=None,
            body_x=None,
            distortion_residual=0.0,
        )
    scale = np.asarray(
        [
            [rectified_width - 1.0, 0.0, 0.0],
            [0.0, rectified_height - 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        image_to_rectified = scale @ np.linalg.inv(homography.H)
    except np.linalg.LinAlgError:
        return GeometryRefinement(
            homography=homography,
            fret_centers=fret_centers,
            string_curves=(),
            fret_support=0.0,
            string_support=0.0,
            nut_x=None,
            body_x=None,
            distortion_residual=0.0,
        )
    rectified = cv2.warpPerspective(
        frame,
        image_to_rectified,
        (rectified_width, rectified_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    string_curves, string_support, distortion = _fit_string_curves(
        gray,
        n_strings=cfg.n_strings,
    )

    detector_centers = (
        None
        if fret_centers is None
        else np.asarray(fret_centers, dtype=np.float64).copy()
    )
    x_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    profile = np.median(
        x_gradient[
            max(1, rectified_height // 10) : max(
                2, rectified_height - rectified_height // 10
            )
        ],
        axis=0,
    )
    inferred_axis = _infer_rule18_axis(
        profile,
        max_fret=cfg.max_fret,
    )
    if inferred_axis is None:
        centers = detector_centers
        detected_nut_x = None
        detected_body_x = None
        body_joint_fret = None
        boundary_support = 0.0
    else:
        (
            centers,
            detected_nut_x,
            detected_body_x,
            body_joint_fret,
            boundary_support,
        ) = inferred_axis
    wires = np.empty(0, dtype=np.float64) if centers is None else _wire_xs(centers)
    observed_wires: list[tuple[int, float, float, float]] = []
    if wires.size:
        in_frame = [
            (index, float(value))
            for index, value in enumerate(wires)
            if 0.0 <= value <= 1.0
        ]
        for sequence_index, (wire_index, expected_x) in enumerate(in_frame):
            left_gap = (
                abs(expected_x - in_frame[sequence_index - 1][1])
                if sequence_index > 0
                else 0.04
            )
            right_gap = (
                abs(in_frame[sequence_index + 1][1] - expected_x)
                if sequence_index + 1 < len(in_frame)
                else left_gap
            )
            radius = max(
                2,
                round(
                    0.35 * max(0.01, min(left_gap, right_gap)) * (rectified_width - 1)
                ),
            )
            peak = _local_peak(
                profile,
                expected_x * (rectified_width - 1),
                radius,
            )
            if peak is None:
                continue
            observed_px, quality = peak
            observed_wires.append(
                (
                    wire_index,
                    expected_x,
                    observed_px / (rectified_width - 1),
                    quality,
                )
            )
    fret_support = float(
        np.clip(
            len(observed_wires)
            / max(min(8, len(wires)), 1)
            * (
                float(np.mean([item[3] for item in observed_wires]))
                if observed_wires
                else 0.0
            ),
            0.0,
            1.0,
        )
    )

    refined_homography = homography
    finite_string_curves = [
        (index, curve)
        for index, curve in enumerate(string_curves)
        if np.all(np.isfinite(curve))
    ]
    if len(observed_wires) >= 4 and len(finite_string_curves) >= 2:
        expected_points: list[tuple[float, float]] = []
        observed_points: list[tuple[float, float]] = []
        edge_curves = (finite_string_curves[0], finite_string_curves[-1])
        expected_ys = tuple(
            index / max(cfg.n_strings - 1, 1) for index, _ in edge_curves
        )
        for _, expected_x, observed_x, _ in observed_wires:
            for expected_y, (_, curve) in zip(
                expected_ys,
                edge_curves,
                strict=True,
            ):
                observed_y = float(np.polyval(np.asarray(curve), observed_x))
                expected_points.append((expected_x, expected_y))
                observed_points.append((observed_x, observed_y))
        observed_image = _project_to_image(
            homography,
            np.asarray(observed_points, dtype=np.float64),
        )
        if np.all(np.isfinite(observed_image)):
            fitted, mask = cv2.findHomography(
                np.asarray(expected_points, dtype=np.float64),
                observed_image,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
            )
            if fitted is not None and mask is not None:
                old_corners = _project_to_image(
                    homography,
                    np.asarray(
                        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                        dtype=np.float64,
                    ),
                )
                candidate = Homography(
                    H=np.asarray(fitted, dtype=np.float64),
                    confidence=float(
                        np.clip(
                            homography.confidence
                            * (0.75 + 0.25 * min(fret_support, string_support)),
                            0.0,
                            1.0,
                        )
                    ),
                    method=f"{homography.method}+local_fret_string",
                )
                new_corners = _project_to_image(
                    candidate,
                    np.asarray(
                        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                        dtype=np.float64,
                    ),
                )
                diagonal = max(
                    1.0,
                    float(
                        np.linalg.norm(
                            old_corners.max(axis=0) - old_corners.min(axis=0)
                        )
                    ),
                )
                innovation = float(
                    np.max(np.linalg.norm(new_corners - old_corners, axis=1)) / diagonal
                )
                if innovation <= 0.04:
                    refined_homography = candidate
    return GeometryRefinement(
        homography=refined_homography,
        fret_centers=centers,
        string_curves=string_curves,
        fret_support=fret_support,
        string_support=string_support,
        nut_x=detected_nut_x,
        body_x=detected_body_x,
        distortion_residual=distortion,
        boundary_support=boundary_support,
        body_joint_fret=body_joint_fret,
    )


def nearest_string(
    canonical_x: float,
    canonical_y: float,
    *,
    n_strings: int,
    string_curves: tuple[tuple[float, float, float], ...] = (),
) -> int:
    """Return a one-based string index with optional lens/rolling correction."""
    if len(string_curves) == n_strings and all(
        np.all(np.isfinite(curve)) for curve in string_curves
    ):
        candidates = [
            (
                index + 1,
                float(np.polyval(np.asarray(curve, dtype=np.float64), canonical_x)),
            )
            for index, curve in enumerate(string_curves)
        ]
        if candidates:
            return min(
                candidates,
                key=lambda item: abs(item[1] - canonical_y),
            )[0]
    return int(
        np.clip(
            round(float(canonical_y) * (n_strings - 1)) + 1,
            1,
            n_strings,
        )
    )


def transform_string_curves(
    string_curves: tuple[tuple[float, float, float], ...],
    canonical_transform: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    """Move base-canonical string curves into a corrected canonical frame.

    ``canonical_transform`` maps corrected canonical coordinates into the base
    canonical coordinates (the ``C`` in ``H_refined = H_base @ C``).
    """
    if not string_curves:
        return ()
    transform = np.asarray(canonical_transform, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("canonical_transform must be a finite 3x3 matrix")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return string_curves
    sample_xs = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    output: list[tuple[float, float, float]] = []
    for curve in string_curves:
        coefficients = np.asarray(curve, dtype=np.float64)
        if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
            output.append((math.nan, math.nan, math.nan))
            continue
        old_points = np.column_stack(
            (
                sample_xs,
                np.polyval(coefficients, sample_xs),
                np.ones_like(sample_xs),
            )
        )
        projected = old_points @ inverse.T
        safe = np.abs(projected[:, 2]) > 1e-9
        new_points = projected[safe, :2] / projected[safe, 2, None]
        finite = np.all(np.isfinite(new_points), axis=1)
        new_points = new_points[finite]
        if len(new_points) < 2 or np.ptp(new_points[:, 0]) <= 1e-6:
            output.append((math.nan, math.nan, math.nan))
            continue
        degree = min(2, len(new_points) - 1)
        fitted = np.polyfit(new_points[:, 0], new_points[:, 1], degree)
        if degree == 1:
            fitted = np.asarray((0.0, fitted[0], fitted[1]))
        output.append(tuple(float(value) for value in fitted))
    return tuple(output)


__all__ = [
    "GeometryRefinement",
    "nearest_string",
    "refine_live_geometry",
    "transform_string_curves",
]
