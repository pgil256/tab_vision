"""Tests for the UT-Austin tablature ``.npy`` parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tabvision.eval.parsers import get_parser
from tabvision.types import GuitarConfig


def test_utaustin_parser_converts_new_finger_placements_to_tab_events(tmp_path: Path) -> None:
    root = tmp_path
    label_dir = root / "tablature_labels"
    label_dir.mkdir()
    (root / "timestamps.csv").write_text(
        "frame,timestamp\n0_0.png,0.0\n0_1.png,0.1\n0_2.png,0.2\n",
        encoding="utf-8",
    )

    # Shape: frames, four fingers, [active, fret, their_string].
    # Frame 0 starts a note at their string 6 / fret 3 -> TabVision string 0.
    # Frame 1 holds the same note -> no new event.
    # Frame 2 adds their string 5 / fret 5 -> TabVision string 1.
    labels = np.zeros((3, 4, 3), dtype=np.float32)
    labels[0, 0] = [1, 3, 6]
    labels[1, 0] = [1, 3, 6]
    labels[2, 0] = [1, 3, 6]
    labels[2, 1] = [1, 5, 5]
    np.save(label_dir / "0.npy", labels)

    parser = get_parser("utaustin_tablature_npy")
    events = parser(label_dir / "0.npy", GuitarConfig())

    assert [(event.onset_s, event.string_idx, event.fret) for event in events] == [
        (0.0, 0, 3),
        (0.2, 1, 5),
    ]
    assert [event.pitch_midi for event in events] == [43, 50]
