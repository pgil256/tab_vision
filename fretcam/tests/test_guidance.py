from __future__ import annotations

from dataclasses import replace

from fretcam.detection import FrameDetection, HandPoint, StageLatency
from fretcam.guidance import assess_guidance
from fretcam.position import PositionEstimate
from tabvision.video.hand.neck_anchor import HandNeckAnchor


def _detection() -> FrameDetection:
    return FrameDetection(
        timestamp_s=1.0,
        detector_ran=True,
        neck_locked=True,
        fret_map_locked=True,
        homography_confidence=0.8,
        homography_method="fixture",
        neck_quad=((10.0, 10.0), (90.0, 10.0), (90.0, 40.0), (10.0, 40.0)),
        fret_ticks=(),
        hand_points=(HandPoint("index", 40.0, 25.0),),
        index_fret=5.2,
        anchor=HandNeckAnchor(5.2, 4.0, 8.0, 0.72, "fixture"),
        stage_latency=StageLatency(1.0, 1.0, 1.0, 1.0, 4.0),
    )


def _estimate(state: str = "locked") -> PositionEstimate:
    return PositionEstimate(
        timestamp_s=1.0,
        state=state,  # type: ignore[arg-type]
        label="Position V",
        raw_index_fret=5.2,
        smoothed_index_fret=5.2,
        position=5 if state in {"locked", "holding"} else None,
        previous_position=None,
        window_frets=(0, 4, 5, 6, 7, 8, 9),
        confidence=0.72,
        temporal_agreement=1.0,
    )


def test_guidance_reports_locked_state_only_from_complete_signals() -> None:
    result = assess_guidance(
        _detection(), _estimate(), frame_width=100, frame_height=50
    )

    assert result.code == "locked"
    assert result.level == "good"
    assert "open strings" in result.message


def test_guidance_prioritizes_missing_or_clipped_neck() -> None:
    missing = replace(_detection(), neck_locked=False, neck_quad=())
    clipped = replace(
        _detection(),
        neck_quad=((0.0, 10.0), (90.0, 10.0), (90.0, 40.0), (0.0, 40.0)),
    )

    assert assess_guidance(
        missing, _estimate(), frame_width=100, frame_height=50
    ).code == "frame_neck"
    clipped_result = assess_guidance(
        clipped, _estimate(), frame_width=100, frame_height=50
    )
    assert clipped_result.code == "neck_at_edge"
    assert "move camera back" in clipped_result.message


def test_guidance_distinguishes_weak_lock_missing_hand_and_transition() -> None:
    weak = replace(_detection(), homography_confidence=0.3)
    no_hand = replace(_detection(), hand_points=())

    assert assess_guidance(
        weak, _estimate(), frame_width=100, frame_height=50
    ).code == "weak_board_lock"
    assert assess_guidance(
        no_hand, _estimate("lost"), frame_width=100, frame_height=50
    ).code == "show_hand"
    assert assess_guidance(
        _detection(), _estimate("shifting"), frame_width=100, frame_height=50
    ).code == "shifting"
