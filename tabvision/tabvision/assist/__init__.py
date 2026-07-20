"""Pitch-preserving assisted-correction primitives."""

from tabvision.assist.candidates import compute_note_candidates
from tabvision.assist.editing import (
    AssistOptions,
    BatchEdit,
    EditSession,
    MotifPreview,
    PositionEdit,
    cycle_candidate_edit,
    matched_motif_previews,
    move_phrase_edit,
    phrase_alternatives,
)

__all__ = [
    "AssistOptions",
    "BatchEdit",
    "EditSession",
    "MotifPreview",
    "PositionEdit",
    "compute_note_candidates",
    "cycle_candidate_edit",
    "matched_motif_previews",
    "move_phrase_edit",
    "phrase_alternatives",
]
