"""CLI parser smoke for ``--fusion-lambda-vision``.

Verifies the flag parses with the right default, accepts user-supplied
values, and surfaces zero (the audio-only-equivalent setting). The
actual pass-through to ``fuse()`` is one line of code in
``_cmd_transcribe`` — see ``tabvision/cli.py``.
"""

from __future__ import annotations

import pytest

from tabvision.cli import _build_parser


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
