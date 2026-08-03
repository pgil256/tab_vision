"""Align a performed take against its gold tab by pitch sequence.

The gold tab has no timestamps; the audio backend's events have no
string/fret truth. Needleman–Wunsch over the two pitch sequences joins
them: each aligned pair stamps one gold note with the performance's onset
time. Only exact pitch matches are emitted — a substitution means either a
playing mistake or a detection error, and a ground-truth corpus must
contain neither.

Insertions (extra detections, string noise) and deletions (missed or
undetected notes) are absorbed as gaps: the surrounding notes still align.
The caller gates on ``matched_fraction`` — a take that diverges too far
from its tab is refused outright rather than salvaged, because "accurate
tab, accurately played" is the premise that makes this data gold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabvision.personal.gold_tab import GoldNote

if TYPE_CHECKING:
    from tabvision.types import AudioEvent

_MATCH = 2
_MISMATCH = -3
_GAP = -1


@dataclass(frozen=True)
class AlignedNote:
    """One gold note stamped with its performed onset."""

    note: GoldNote
    onset_s: float
    confidence: float


@dataclass(frozen=True)
class AlignmentResult:
    matches: tuple[AlignedNote, ...]
    gold_count: int
    event_count: int

    @property
    def matched_fraction(self) -> float:
        return len(self.matches) / self.gold_count if self.gold_count else 0.0


def align_gold_notes(
    audio_events: list[AudioEvent] | tuple[AudioEvent, ...],
    gold_notes: list[GoldNote],
) -> AlignmentResult:
    """Return gold notes stamped with onsets via global pitch alignment."""
    events = sorted(audio_events, key=lambda event: event.onset_s)
    n, m = len(events), len(gold_notes)
    if n == 0 or m == 0:
        return AlignmentResult(matches=(), gold_count=m, event_count=n)

    # Needleman–Wunsch with traceback: rows = events, columns = gold notes.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for row in range(1, n + 1):
        score[row][0] = row * _GAP
    for col in range(1, m + 1):
        score[0][col] = col * _GAP
    for row in range(1, n + 1):
        event_pitch = events[row - 1].pitch_midi
        for col in range(1, m + 1):
            paired = _MATCH if event_pitch == gold_notes[col - 1].pitch_midi else _MISMATCH
            score[row][col] = max(
                score[row - 1][col - 1] + paired,
                score[row - 1][col] + _GAP,
                score[row][col - 1] + _GAP,
            )

    matches: list[AlignedNote] = []
    row, col = n, m
    while row > 0 and col > 0:
        event = events[row - 1]
        note = gold_notes[col - 1]
        paired = _MATCH if event.pitch_midi == note.pitch_midi else _MISMATCH
        if score[row][col] == score[row - 1][col - 1] + paired:
            if event.pitch_midi == note.pitch_midi:
                matches.append(
                    AlignedNote(
                        note=note,
                        onset_s=event.onset_s,
                        confidence=event.confidence,
                    )
                )
            row -= 1
            col -= 1
        elif score[row][col] == score[row - 1][col] + _GAP:
            row -= 1
        else:
            col -= 1
    matches.reverse()
    return AlignmentResult(matches=tuple(matches), gold_count=m, event_count=n)


__all__ = ["AlignedNote", "AlignmentResult", "align_gold_notes"]
