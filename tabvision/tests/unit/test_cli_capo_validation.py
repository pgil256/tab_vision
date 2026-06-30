"""CLI validation for ``--capo``.

The flag is documented as ``capo fret (0-7)``; this enforces that range at the
parser boundary so a negative or out-of-range value fails fast with a clear
message rather than silently corrupting the rendered tab.
"""

from __future__ import annotations

import pytest

from tabvision.cli import _build_parser, _capo_arg


def test_capo_default_zero():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4"])
    assert args.capo == 0


@pytest.mark.parametrize("value", [0, 3, 7])
def test_capo_in_range_accepted(value):
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "in.mp4", "--capo", str(value)])
    assert args.capo == value


@pytest.mark.parametrize("value", ["-1", "8", "24"])
def test_capo_out_of_range_rejected(value):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--capo", value])


def test_capo_non_integer_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["transcribe", "in.mp4", "--capo", "high"])


def test_capo_validated_on_diagnose():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["diagnose", "in.mp4", "--capo", "99"])


# ---------- the validator in isolation ----------


@pytest.mark.parametrize("value", [-1, 8, 100])
def test_capo_arg_raises_on_out_of_range(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _capo_arg(str(value))


def test_capo_arg_raises_on_non_integer():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _capo_arg("not-a-number")
