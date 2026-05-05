"""Single-line Viterbi decode + audio-only fallback.

Public entrypoint: ``fuse(events, fingerings, cfg, session, lambda_vision)``.

Phase 1: when ``fingerings`` is empty (video stubs), degenerate to a
greedy "lowest-fret with continuity bonus" decoder per SPEC.md §7 Phase 1.

Phase 5 replaces the body with a proper Viterbi over candidate states
using ``tabvision.fusion.playability`` transition costs. The public
signature stays stable.
"""

from __future__ import annotations

from typing import Sequence

from tabvision.errors import FusionError
from tabvision.fusion.candidates import Candidate, candidate_positions
from tabvision.types import (
    AudioEvent,
    FrameFingering,
    GuitarConfig,
    SessionConfig,
    TabEvent,
)

# Continuity bonus: amount subtracted from a candidate's "cost" when its
# string matches the previous note's string. A small constant; Phase 5
# will calibrate.
STRING_CONTINUITY_BONUS = 0.5
# Penalty per fret of distance from the previous note's fret. Small
# enough that the lowest-fret bias still wins for distant pitches.
FRET_DISTANCE_PENALTY = 0.05
# Penalty per fret position (lower-fret preference).
LOWER_FRET_BIAS = 0.10


def fuse(
    events: Sequence[AudioEvent],
    fingerings: Sequence[FrameFingering],
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    lambda_vision: float = 1.0,
) -> list[TabEvent]:
    """Decode AudioEvents into TabEvents.

    Phase 1: ``fingerings`` is empty / uniform; falls back to greedy
    audio-only decode. The ``lambda_vision`` weight is accepted for
    interface stability but ignored until Phase 5.
    """
    if cfg is None:
        cfg = GuitarConfig()
    if session is None:
        session = SessionConfig()

    has_video = any(_has_evidence(f) for f in fingerings)
    if has_video:
        # Phase 5 deliverable: Viterbi over (string, fret) states with
        # vision-evidence + playability costs. Not yet implemented.
        raise FusionError(
            "video-aware fusion not implemented in Phase 1 — "
            "this is a Phase 5 deliverable"
        )

    return _greedy_audio_only(events, cfg)


def _has_evidence(f: FrameFingering) -> bool:
    """A FrameFingering carries info if its logits are not all-zero."""
    arr = f.finger_pos_logits
    return arr is not None and bool(arr.size) and bool((arr != 0).any())


def _greedy_audio_only(
    events: Sequence[AudioEvent], cfg: GuitarConfig
) -> list[TabEvent]:
    """Pick (string, fret) per event by lowest-fret + continuity."""
    out: list[TabEvent] = []
    prev: Candidate | None = None

    for ev in events:
        candidates = candidate_positions(ev.pitch_midi, cfg)
        if not candidates:
            # Out-of-range pitch; skip rather than emit a phantom note.
            continue
        pick = _pick_candidate(candidates, prev)
        out.append(
            TabEvent(
                onset_s=ev.onset_s,
                duration_s=max(0.0, ev.offset_s - ev.onset_s),
                string_idx=pick.string_idx,
                fret=pick.fret,
                pitch_midi=ev.pitch_midi,
                confidence=ev.confidence,
                techniques=ev.tags,
            )
        )
        prev = pick

    return out


def _pick_candidate(
    candidates: list[Candidate], prev: Candidate | None
) -> Candidate:
    """Score each candidate; lower cost wins."""

    def cost(c: Candidate) -> float:
        score = LOWER_FRET_BIAS * c.fret
        if prev is not None:
            score += FRET_DISTANCE_PENALTY * abs(c.fret - prev.fret)
            if c.string_idx == prev.string_idx:
                score -= STRING_CONTINUITY_BONUS
        return score

    return min(candidates, key=cost)


__all__ = ["fuse"]
