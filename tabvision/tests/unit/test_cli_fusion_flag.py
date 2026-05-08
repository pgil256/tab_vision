"""CLI parser smoke for the transcribe-subcommand flags.

Covers ``--fusion-lambda-vision``, ``--no-video``, ``--video-stride``.
"""

from __future__ import annotations

import pytest

from tabvision.cli import _build_parser

# ---------- --fusion-lambda-vision ----------


def test_default_lambda_vision_is_one():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.fusion_lambda_vision == 1.0


def test_explicit_lambda_vision_parsed():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "2.5"])
    assert args.fusion_lambda_vision == pytest.approx(2.5)


def test_lambda_vision_zero_accepted():
    """``--fusion-lambda-vision 0`` is the audio-only ablation knob."""
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--fusion-lambda-vision", "0"])
    assert args.fusion_lambda_vision == 0.0


def test_lambda_vision_only_on_transcribe():
    """The ``check`` subcommand has no fusion stage, so the flag should
    not be exposed there."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--fusion-lambda-vision", "1.0"])


# ---------- --no-video ----------


def test_no_video_default_false():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.no_video is False


def test_no_video_flag_sets_true():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--no-video"])
    assert args.no_video is True


# ---------- --video-stride ----------


def test_video_stride_default_three():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.video_stride == 3


def test_video_stride_explicit_value():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--video-stride", "1"])
    assert args.video_stride == 1


def test_video_stride_only_on_transcribe():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "in.mp4", "--video-stride", "5"])
