"""Deterministic pipeline fixture for the desktop sidecar integration gate."""

from __future__ import annotations

import sys
from collections.abc import Callable

import tabvision.pipeline
from tabvision.cli import main
from tabvision.types import TabEvent


def _fixture_pipeline(*_args: object, **kwargs: object) -> list[TabEvent]:
    progress_callback = kwargs.get("progress_callback")
    if isinstance(progress_callback, Callable):
        for stage in (
            "demux",
            "model_load",
            "audio_inference",
            "video_analysis",
            "decode",
        ):
            progress_callback(stage)

    return [
        TabEvent(
            onset_s=0.125,
            duration_s=0.25,
            string_idx=5,
            fret=5,
            pitch_midi=69,
            confidence=0.32,
        )
    ]


def run() -> int:
    tabvision.pipeline.run_pipeline = _fixture_pipeline
    return main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(run())
