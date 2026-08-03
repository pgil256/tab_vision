"""Corrected studio transcriptions as gold labels.

The studio loop: the user records a take, the pipeline transcribes it, and
the user fixes the transcription in the web editor. The corrected document
is a gold tab that labels *itself* — every note already carries its
performed timestamp from the original audio decode, so unlike
:mod:`tabvision.personal.gold_tab` no sequence alignment is needed, and
unedited notes count as confirmed by the review.

Notes arrive in the editor's own JSON shape (``TabDocument.notes``):
``{"timestamp": seconds, "string": 1-6 (tab convention, 1 = high E),
"fret": int | "X"}``. Muted ``"X"`` notes carry no fret truth and are
skipped rather than guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from tabvision.personal.alignment import AlignedNote
from tabvision.personal.gold_tab import GoldNote
from tabvision.types import GuitarConfig

CORRECTION_SOURCE = "studio-correction"


def correction_notes_to_matches(
    notes: Sequence[Mapping[str, Any]],
    cfg: GuitarConfig | None = None,
) -> list[AlignedNote]:
    """Validate corrected editor notes into onset-stamped gold matches.

    Human-reviewed notes are ground truth, so confidence is 1.0 by
    definition. Structural problems raise — a correction set does not get
    to be almost right — but muted ``"X"`` frets are skipped silently
    because the editor legitimately produces them.
    """
    cfg = cfg or GuitarConfig()
    if cfg.capo != 0:
        raise ValueError("gold-session ingest requires capo 0; the stores are capo-0 indexed")

    matches: list[AlignedNote] = []
    for index, raw in enumerate(notes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"note {index} must be an object")
        fret_raw = raw.get("fret")
        if fret_raw == "X":
            continue
        try:
            timestamp = float(raw["timestamp"])
            string = int(raw["string"])
            fret = int(fret_raw)  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"note {index} needs a numeric 'timestamp', integer 'string', "
                "and integer or 'X' 'fret'"
            ) from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"note {index} timestamp must be finite and non-negative")
        if not 1 <= string <= cfg.n_strings:
            raise ValueError(
                f"note {index} string must be 1..{cfg.n_strings} (tab convention, 1 = high E)"
            )
        if not 0 <= fret <= cfg.max_fret:
            raise ValueError(f"note {index} fret must be 0..{cfg.max_fret}")
        string_idx = cfg.n_strings - string
        matches.append(
            AlignedNote(
                note=GoldNote(
                    string_idx=string_idx,
                    fret=fret,
                    pitch_midi=cfg.tuning_midi[string_idx] + fret,
                ),
                onset_s=timestamp,
                confidence=1.0,
            )
        )
    if not matches:
        raise ValueError("correction set contains no playable notes")
    return matches


__all__ = ["CORRECTION_SOURCE", "correction_notes_to_matches"]
