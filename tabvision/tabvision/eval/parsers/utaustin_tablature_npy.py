"""UT-Austin guitar-transcription label parser.

The Kaggle/UT-Austin tablature dataset stores one ``.npy`` label file per
clip. Each frame has four finger slots with ``[active, fret, string]`` labels;
the companion ``timestamps.csv`` maps frame names to seconds. This parser uses
the same gold derivation as the v1.1 real-video probes: new finger placements
become note onsets, same-string simultaneous placements collapse to the highest
fret, and dataset strings are mapped to TabVision string indices with
``our_string_idx = 6 - their_string``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tabvision.eval.parsers.registry import register_parser
from tabvision.types import GuitarConfig, TabEvent

FORMAT_NAME = "utaustin_tablature_npy"


def _load_timestamps(root: Path) -> dict[str, float]:
    with open(root / "timestamps.csv", newline="", encoding="utf-8") as fh:
        return {row["frame"]: float(row["timestamp"]) for row in csv.DictReader(fh)}


def parse(
    annotation_path: str | Path,
    cfg: GuitarConfig | None = None,
    *,
    default_dur: float = 0.3,
) -> list[TabEvent]:
    """Parse a UT-Austin ``tablature_labels/<clip>.npy`` file into tab events."""

    label_path = Path(annotation_path)
    if cfg is None:
        cfg = GuitarConfig()

    root = label_path.parent.parent
    timestamps = _load_timestamps(root)
    clip_id = label_path.stem
    arr = np.load(label_path)

    gold: list[TabEvent] = []
    prev: set[tuple[int, int]] = set()
    for frame_index in range(arr.shape[0]):
        cur = {
            (int(arr[frame_index, finger_index, 1]), int(arr[frame_index, finger_index, 2]))
            for finger_index in range(arr.shape[1])
            if arr[frame_index, finger_index].any()
        }

        highest_by_string: dict[int, int] = {}
        for fret, their_string in cur - prev:
            highest_by_string[their_string] = max(fret, highest_by_string.get(their_string, -1))

        for their_string, fret in sorted(highest_by_string.items()):
            string_idx = 6 - their_string
            onset_s = timestamps.get(f"{clip_id}_{frame_index}.png")
            if (
                onset_s is None
                or not (0 <= string_idx < cfg.n_strings)
                or not (0 <= fret <= cfg.max_fret)
            ):
                continue
            gold.append(
                TabEvent(
                    onset_s=onset_s,
                    duration_s=default_dur,
                    string_idx=string_idx,
                    fret=fret,
                    pitch_midi=cfg.tuning_midi[string_idx] + fret,
                    confidence=1.0,
                )
            )
        prev = cur

    gold.sort(key=lambda event: (event.onset_s, event.string_idx, event.fret))
    return gold


register_parser(FORMAT_NAME, parse)


__all__ = ["FORMAT_NAME", "parse"]
