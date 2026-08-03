"""User-supplied reference tabs as ground-truth labels.

A gold tab is the user's statement of exactly what they played — strings
and frets in play order. Unlike the FretCam window harvest (which only
labels notes the camera happened to pin down), a gold tab labels every
note, with no camera in the loop, which is what makes it usable as a
training substrate for video analysis itself: the labels carry no
selection bias from the very models they would train.

Format (JSON):

    {"notes": [{"string": 6, "fret": 3}, {"string": 5, "fret": 2}, ...]}

``string`` uses tab convention — **1 = high E through 6 = low E** — because
that is how guitarists write tabs. It is converted to TabVision's
``string_idx`` (0 = low E) on load. An optional ``pitch_midi`` per note is
cross-checked against the tuning and rejected on mismatch, catching
off-by-one string errors before they poison a corpus.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tabvision.types import GuitarConfig

GOLD_TAB_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GoldNote:
    """One reference note in play order, in TabVision's own conventions."""

    string_idx: int
    fret: int
    pitch_midi: int


def load_gold_tab(path: str | Path, cfg: GuitarConfig | None = None) -> list[GoldNote]:
    """Parse and validate a gold tab file into ordered :class:`GoldNote` rows.

    Standard tuning at capo 0 only: the corpus and the personal-prior store
    are both capo-0 indexed, and a mis-declared tuning would silently
    mislabel every note. Any structural or consistency problem raises —
    a gold tab is ground truth, so it does not get to be almost right.
    """
    cfg = cfg or GuitarConfig()
    if cfg.capo != 0:
        raise ValueError("gold-tab ingest requires capo 0; the stores are capo-0 indexed")

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: gold tab is not valid JSON") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("notes"), list):
        raise ValueError(f"{source}: gold tab must be an object with a 'notes' list")

    notes: list[GoldNote] = []
    for index, raw in enumerate(payload["notes"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: note {index} must be an object")
        try:
            string = int(raw["string"])
            fret = int(raw["fret"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: note {index} needs integer 'string' and 'fret'") from exc
        if not 1 <= string <= cfg.n_strings:
            raise ValueError(
                f"{source}: note {index} string must be 1..{cfg.n_strings} "
                "(tab convention, 1 = high E)"
            )
        if not 0 <= fret <= cfg.max_fret:
            raise ValueError(f"{source}: note {index} fret must be 0..{cfg.max_fret}")
        string_idx = cfg.n_strings - string
        pitch = cfg.tuning_midi[string_idx] + fret
        declared = raw.get("pitch_midi")
        if declared is not None and int(declared) != pitch:
            raise ValueError(
                f"{source}: note {index} declares pitch {declared} but string "
                f"{string} fret {fret} sounds {pitch} — check the string number"
            )
        notes.append(GoldNote(string_idx=string_idx, fret=fret, pitch_midi=pitch))
    if not notes:
        raise ValueError(f"{source}: gold tab contains no notes")
    return notes


__all__ = ["GOLD_TAB_SCHEMA_VERSION", "GoldNote", "load_gold_tab"]
