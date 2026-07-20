"""Runtime per-note pitch-preserving candidate rankings for assisted review.

Phase 6 measured assisted review against "the production decoder's
pitch-preserving min-marginal candidates" (top-3 recall 0.9986 on ambiguous
held-out notes). This module exposes exactly that ranking at runtime: it
re-decodes the pipeline's retained ``AudioEvent`` stream with the shipped
costs via the analysis decoder and aligns each note's ranked candidates to
the produced ``TabEvent`` order.

It never mutates the automatic result — the output is advisory metadata for
the review UI (2026-07-20 personal-posture program; DECISIONS.md same date
supersedes the Phase 6 "terminal" rule for this assisted path).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Sequence

from tabvision.eval.string_assignment import RankedCandidate, decode_with_analysis
from tabvision.types import AudioEvent, GuitarConfig, TabEvent

logger = logging.getLogger(__name__)

_ONSET_KEY_DECIMALS = 6


def _note_key(onset_s: float, pitch_midi: int) -> tuple[float, int]:
    return (round(float(onset_s), _ONSET_KEY_DECIMALS), int(pitch_midi))


def compute_note_candidates(
    tab_events: Sequence[TabEvent],
    audio_events: Sequence[AudioEvent],
    *,
    cfg: GuitarConfig | None = None,
    sequence_prior: str | None = "none",
    max_candidates: int = 4,
) -> list[tuple[RankedCandidate, ...] | None]:
    """Ranked pitch-preserving candidates aligned to ``tab_events`` order.

    ``sequence_prior`` must be the *resolved* policy name the production
    decode used (or ``"none"``) so the analysis decode mirrors its costs.
    Notes that cannot be aligned to the analysis decode — or whose emitted
    position is missing from the ranking (an alignment red flag) — get
    ``None`` instead of a guessed list. Failures never raise beyond this
    function's contract; callers may still wrap it for belt-and-suspenders.
    """
    cfg = cfg or GuitarConfig()
    if not tab_events:
        return []

    # Imported lazily to keep module import light and the pipeline dependency
    # one-directional at import time.
    from tabvision.pipeline import sequence_decode_context

    with sequence_decode_context(sequence_prior):
        analysis = decode_with_analysis(audio_events, cfg=cfg, k_paths=1)

    index_by_key: dict[tuple[float, int], deque[int]] = defaultdict(deque)
    for idx, audio_event in enumerate(analysis.audio_events):
        index_by_key[_note_key(audio_event.onset_s, audio_event.pitch_midi)].append(idx)

    out: list[tuple[RankedCandidate, ...] | None] = []
    unmatched = 0
    for note in tab_events:
        queue = index_by_key.get(_note_key(note.onset_s, note.pitch_midi))
        if not queue:
            unmatched += 1
            out.append(None)
            continue
        idx = queue.popleft()
        ranked = analysis.candidate_ranks[idx] if idx < len(analysis.candidate_ranks) else ()
        current = (int(note.string_idx), int(note.fret))
        if not any((cand.string_idx, cand.fret) == current for cand in ranked):
            # The production position must appear among the decode's playable
            # candidates; if it does not, the alignment is not trustworthy.
            unmatched += 1
            out.append(None)
            continue
        trimmed = list(ranked[: max(1, max_candidates)])
        if not any((cand.string_idx, cand.fret) == current for cand in trimmed):
            trimmed.append(next(cand for cand in ranked if (cand.string_idx, cand.fret) == current))
        out.append(tuple(trimmed))
    if unmatched:
        logger.info(
            "assist candidates: %d/%d notes could not be aligned to the analysis decode",
            unmatched,
            len(tab_events),
        )
    return out


__all__ = ["RankedCandidate", "compute_note_candidates"]
