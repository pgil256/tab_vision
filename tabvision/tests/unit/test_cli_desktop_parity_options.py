"""Desktop/browser parity options carried by the local CLI contract."""

from __future__ import annotations

import numpy as np
import pytest

from tabvision.cli import _build_parser
from tabvision.pipeline import _crop_frame_iterator


def test_parity_options_default_to_standard_accurate_without_roi() -> None:
    args = _build_parser().parse_args(["transcribe", "in.mp4"])

    assert args.tuning == "standard"
    assert args.accuracy_mode == "accurate"
    assert args.roi is None


@pytest.mark.parametrize(
    "tuning",
    ["standard", "drop-d", "eb-standard", "d-standard", "drop-c", "dadgad", "open-g"],
)
def test_every_browser_tuning_is_accepted(tuning: str) -> None:
    args = _build_parser().parse_args(["transcribe", "in.mp4", "--tuning", tuning])

    assert args.tuning == tuning


def test_normalized_roi_is_parsed_in_display_order() -> None:
    args = _build_parser().parse_args(["transcribe", "in.mp4", "--roi", "0.1", "0.2", "0.8", "0.9"])

    assert args.roi == [0.1, 0.2, 0.8, 0.9]


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "not-a-number"])
def test_roi_rejects_non_normalized_coordinates(value: str) -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["transcribe", "in.mp4", "--roi", value, "0", "1", "1"])


def test_frame_roi_crops_lazily_and_preserves_timestamp() -> None:
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)

    cropped = list(_crop_frame_iterator(iter([(1.25, frame)]), (0.2, 0.25, 0.8, 0.75)))

    assert cropped[0][0] == 1.25
    assert cropped[0][1].shape == (4, 6, 3)
    np.testing.assert_array_equal(cropped[0][1], frame[2:6, 2:8])
