"""Grounded framing and tracking guidance for the live FretCam HUD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from fretcam.detection import FrameDetection
from fretcam.position import PositionEstimate

GuidanceLevel = Literal["good", "info", "warning"]


@dataclass(frozen=True)
class Guidance:
    code: str
    message: str
    level: GuidanceLevel

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def assess_guidance(
    detection: FrameDetection,
    estimate: PositionEstimate,
    *,
    frame_width: int,
    frame_height: int,
    edge_margin: float = 0.03,
    weak_lock_confidence: float = 0.35,
) -> Guidance:
    """Return advice supported by the current geometry and hand signals."""
    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    if not 0.0 <= edge_margin < 0.5:
        raise ValueError("edge_margin must be in [0, 0.5)")

    if not detection.neck_locked or len(detection.neck_quad) != 4:
        return Guidance("frame_neck", "Frame the full guitar neck", "warning")
    if detection.geometry_status == "stale":
        return Guidance(
            "stale_board",
            "Fretboard tracking is stale - hold still while it reacquires",
            "warning",
        )

    x_margin = frame_width * edge_margin
    y_margin = frame_height * edge_margin
    if any(
        x <= x_margin
        or x >= frame_width - x_margin
        or y <= y_margin
        or y >= frame_height - y_margin
        for x, y in detection.neck_quad
    ):
        return Guidance(
            "neck_at_edge",
            "Neck partly out of frame - move camera back",
            "warning",
        )

    if detection.homography_confidence < weak_lock_confidence:
        return Guidance(
            "weak_board_lock",
            "Board lock is weak - hold the camera steady and reduce glare",
            "warning",
        )

    if (
        detection.hand_points
        and "off_neck" in detection.confidence_factors.blockers
        and estimate.state in {"acquiring", "lost"}
    ):
        return Guidance(
            "few_fingertips_on_neck",
            "Hand seen - place 3 or more fingertips on the neck to lock; "
            "single-finger notes may not lock",
            "warning",
        )
    if detection.hand_points and (
        estimate.reason == "low_confidence"
        or "low_confidence" in detection.confidence_factors.blockers
    ):
        return Guidance(
            "low_confidence",
            "Position evidence is weak - keep several fingertips visible",
            "warning",
        )
    if not detection.hand_points or estimate.state == "lost":
        return Guidance("show_hand", "Show your fretting hand on the neck", "warning")
    if estimate.state == "acquiring":
        return Guidance("acquiring", "Hold position while the readout locks", "info")
    if estimate.state == "shifting":
        return Guidance("shifting", "Shift in progress", "info")
    if estimate.state == "holding":
        return Guidance(
            "holding", "Hand briefly hidden - holding the last position", "info"
        )
    return Guidance(
        "locked",
        "Locked - keep the neck visible; open strings remain possible",
        "good",
    )


__all__ = ["Guidance", "GuidanceLevel", "assess_guidance"]
