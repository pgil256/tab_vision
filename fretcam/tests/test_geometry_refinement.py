from __future__ import annotations

import cv2
import numpy as np
import pytest

from fretcam.geometry_refinement import (
    _string_curve_spacing_deficit,
    nearest_string,
    refine_live_geometry,
    transform_string_curves,
)
from tabvision.types import GuitarConfig, Homography
from tabvision.video.fretboard.calibrate import RULE_OF_18_RATIO


def _fixture_geometry(
    *,
    body_joint_fret: int = 12,
) -> tuple[np.ndarray, Homography, np.ndarray, GuitarConfig]:
    cfg = GuitarConfig()
    homography = Homography(
        H=np.asarray(
            ((600.0, 0.0, 20.0), (0.0, 120.0, 20.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        confidence=0.9,
        method="fixture",
    )
    body_fraction = 1.0 - RULE_OF_18_RATIO**body_joint_fret
    frets = np.arange(cfg.max_fret + 1, dtype=np.float64) + 0.5
    centers = (1.0 - np.power(RULE_OF_18_RATIO, frets)) / body_fraction
    wire_frets = np.arange(centers.size + 1, dtype=np.float64)
    wires = (1.0 - np.power(RULE_OF_18_RATIO, wire_frets)) / body_fraction

    frame = np.zeros((160, 640, 3), dtype=np.uint8)
    for wire in wires[(wires >= 0.0) & (wires <= 1.0)]:
        x = round(20.0 + 600.0 * float(wire))
        cv2.line(frame, (x, 20), (x, 140), (230, 230, 230), 2)
    for index in range(cfg.n_strings):
        fraction = index / (cfg.n_strings - 1)
        y = round(20.0 + 120.0 * fraction)
        cv2.line(frame, (20, y), (620, y), (180, 180, 180), 1)
    return frame, homography, centers, cfg


def test_live_refinement_finds_fret_strings_nut_and_body_boundary() -> None:
    frame, homography, centers, cfg = _fixture_geometry()

    result = refine_live_geometry(frame, homography, centers, cfg)

    assert result.fret_support > 0.15
    assert result.string_support > 0.10
    assert result.nut_x == pytest.approx(0.0, abs=0.04)
    assert result.body_x == pytest.approx(1.0, abs=0.05)
    assert result.body_joint_fret == 12
    assert result.boundary_support > 0.10
    assert result.fret_centers is not None
    assert np.ptp(np.diff(result.fret_centers[:12])) > 0.01
    assert len(result.string_curves) == cfg.n_strings


def test_nearest_string_uses_complete_uniform_fallback_when_curve_missing() -> None:
    curves = (
        (0.02, 0.01, 0.00),
        (0.02, 0.01, 0.20),
        (np.nan, np.nan, np.nan),
        (0.02, 0.01, 0.60),
        (0.02, 0.01, 0.80),
        (0.02, 0.01, 1.00),
    )

    assert (
        nearest_string(
            0.5,
            0.62,
            n_strings=6,
            string_curves=curves,
        )
        == 4
    )
    assert (
        nearest_string(
            0.5,
            0.40,
            n_strings=6,
            string_curves=curves,
        )
        == 3
    )


@pytest.mark.parametrize("body_joint_fret", (14, 17, 18, 19))
def test_body_joint_fret_is_inferred_across_common_guitar_joints(
    body_joint_fret: int,
) -> None:
    frame, homography, centers, cfg = _fixture_geometry(body_joint_fret=body_joint_fret)

    result = refine_live_geometry(frame, homography, centers, cfg)

    assert result.body_joint_fret == body_joint_fret
    assert result.boundary_support > 0.10


def test_invalid_geometry_abstains_without_modifying_homography() -> None:
    _, homography, centers, cfg = _fixture_geometry()
    missing = Homography(
        H=homography.H,
        confidence=0.0,
        method="missing",
    )

    result = refine_live_geometry(
        np.zeros((20, 20, 3), dtype=np.uint8),
        missing,
        centers,
        cfg,
    )

    assert result.homography is missing
    assert result.fret_support == 0.0
    assert result.string_support == 0.0


def test_string_curves_move_into_corrected_canonical_coordinates() -> None:
    curves = ((0.0, 0.0, 0.4),)
    # H_refined = H_base @ C, so C maps corrected (new) canonical points
    # into base (old) canonical points.
    correction = np.asarray(
        ((1.0, 0.0, 0.10), (0.0, 1.0, 0.05), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    transformed = transform_string_curves(curves, correction)

    assert np.polyval(transformed[0], 0.5) == pytest.approx(0.35, abs=1e-6)


def test_crossing_string_curves_are_rejected() -> None:
    curves = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.2),
        (0.0, 0.15, 0.4),
        (0.0, -0.15, 0.6),
        (0.0, 0.0, 0.8),
        (0.0, 0.0, 1.0),
    )

    assert _string_curve_spacing_deficit(curves, n_strings=6) > 0.0
