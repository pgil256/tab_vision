"""Phase D extraction changes: sustain-window sampling + hand-tight crops.

Both target the banked WS4 negative (net −0.117 Tab F1, val 6-way accuracy
plateauing at ~0.30): the documented root cause was that a whole-neck crop
squished to 224² starves the model, compounded by onset-frame label noise.
"""

from __future__ import annotations

import numpy as np

from scripts.train.extract_string_dataset import hand_tight_rect, sustain_frame_index
from tabvision.video.hand.fingertip_to_fret import FingerSample, HandSample


def _hand(wrist=(100.0, 100.0), tips=((110.0, 90.0), (120.0, 95.0))) -> HandSample:
    return HandSample(
        wrist_xy=wrist,
        wrist_z=0.0,
        is_left_hand=True,
        confidence=0.9,
        fingers={
            f"f{i}": FingerSample(name=f"f{i}", tip_xy=t, tip_z=0.0, curl_ratio=0.9)
            for i, t in enumerate(tips)
        },
    )


class TestSustainFrameIndex:
    def test_lands_inside_the_sustain_window(self) -> None:
        fps = 25.0
        fi = sustain_frame_index(10.0, next_onset_s=None, offset_s=0.0, fps=fps)
        t = fi / fps
        assert 10.08 <= t <= 10.40

    def test_never_crosses_into_the_next_note(self) -> None:
        fps = 100.0
        fi = sustain_frame_index(5.0, next_onset_s=5.20, offset_s=0.0, fps=fps)
        assert fi / fps <= 5.20 - 0.040 + 1e-9

    def test_dense_notes_do_not_borrow_the_next_frame(self) -> None:
        fps = 100.0
        # Notes 50 ms apart: closer than lead (80 ms) + guard (40 ms).
        fi = sustain_frame_index(2.0, next_onset_s=2.05, offset_s=0.0, fps=fps)
        assert 2.0 - 1e-9 <= fi / fps < 2.05

    def test_offset_is_applied(self) -> None:
        fps = 50.0
        a = sustain_frame_index(1.0, None, offset_s=0.0, fps=fps)
        b = sustain_frame_index(1.0, None, offset_s=2.0, fps=fps)
        assert b - a == int(round(2.0 * fps))

    def test_differs_from_the_onset_frame(self) -> None:
        fps = 30.0
        onset_fi = int(round(7.0 * fps))
        assert sustain_frame_index(7.0, None, 0.0, fps) > onset_fi


class TestHandTightRect:
    def test_square_and_centred_on_the_hand(self) -> None:
        rect = hand_tight_rect(_hand(), 640, 480, min_extent_px=0)
        assert rect is not None
        x0, y0, x1, y1 = rect
        assert abs((x1 - x0) - (y1 - y0)) <= 2  # square up to rounding
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        assert 95 <= cx <= 125 and 85 <= cy <= 115

    def test_min_extent_floor_applies_to_a_small_hand(self) -> None:
        tight = _hand(wrist=(300.0, 300.0), tips=((302.0, 301.0), (303.0, 302.0)))
        rect = hand_tight_rect(tight, 640, 480, min_extent_px=160)
        assert rect is not None
        x0, _y0, x1, _y1 = rect
        assert x1 - x0 >= 150  # floored, not a 3-pixel crop

    def test_clamped_to_frame_bounds(self) -> None:
        edge = _hand(wrist=(5.0, 5.0), tips=((8.0, 4.0), (2.0, 9.0)))
        rect = hand_tight_rect(edge, 640, 480, min_extent_px=200)
        assert rect is not None
        x0, y0, x1, y1 = rect
        assert x0 == 0 and y0 == 0 and x1 <= 640 and y1 <= 480

    def test_rejects_non_finite_landmarks(self) -> None:
        bad = _hand(tips=((float("nan"), 90.0), (120.0, 95.0)))
        assert hand_tight_rect(bad, 640, 480) is None

    def test_is_tighter_than_a_whole_neck_crop(self) -> None:
        """The WS4 root cause, expressed as a test."""
        rect = hand_tight_rect(_hand(), 1280, 720, min_extent_px=160)
        assert rect is not None
        x0, y0, x1, y1 = rect
        hand_area = (x1 - x0) * (y1 - y0)
        neck_area = 1280 * 300  # a typical full-width neck band
        assert hand_area < neck_area / 4

    def test_all_landmarks_are_enclosed(self) -> None:
        hand = _hand(wrist=(200.0, 200.0), tips=((260.0, 180.0), (150.0, 240.0)))
        rect = hand_tight_rect(hand, 640, 480, pad_mult=1.6, min_extent_px=0)
        assert rect is not None
        x0, y0, x1, y1 = rect
        pts = np.array([(200.0, 200.0), (260.0, 180.0), (150.0, 240.0)])
        assert (pts[:, 0] >= x0).all() and (pts[:, 0] <= x1).all()
        assert (pts[:, 1] >= y0).all() and (pts[:, 1] <= y1).all()
