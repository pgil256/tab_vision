"""Playability emission + transition costs — Phase 5 deliverable.

All functions return **negative log-probs** in nats: lower cost = better.
Costs decompose into per-candidate emission terms (audio prior + vision
evidence + open-string bonus + low-fret bias) and pairwise transition
terms (string continuity + position shift + hand-span barrier).

See ``docs/plans/2026-05-06-phase5-fusion-design.md`` §2 for the formulae
and ``SPEC.md`` §5 for acceptance bars.

Port targets: ``tabvision-server/app/fusion_engine.py`` —
``_score_position_heuristic``, ``_correct_slide_positions``, the
hand-anchor/position-shift logic.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

from tabvision.fusion.candidates import Candidate
from tabvision.fusion.transition_prior import TransitionPrior, load_transition_prior
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig

# --- emission term weights ---
LOW_FRET_BIAS = 0.10
"""Cost added per fret index. Keeps the decoder honest when audio + vision
are flat — picks the lower fret all else equal. Same magnitude as the legacy
``viterbi.LOWER_FRET_BIAS``."""

OPEN_STRING_BONUS = 0.5
"""Cost subtracted when the candidate is an open string (fret 0).

Open strings are systematically under-represented by MediaPipe-derived
``marginal_string_fret`` because there is no fingertip pressing — this
bonus re-introduces them. Magnitude calibrated to roughly cancel the
vision-floor cost (``-log(VISION_FLOOR)`` over a uniform marginal)."""

VISION_FLOOR = 1e-3
"""Minimum probability used when computing ``-log P_vision``. Caps the
vision evidence's contribution at ``-log(1e-3) ≈ 6.9`` per candidate so
a confident wrong fingering can still be overridden by strong audio +
playability evidence."""

# --- transition term weights ---
SAME_STRING_BONUS = 0.5
"""Cost subtracted when ``prev.string_idx == curr.string_idx``. Direct
port of legacy ``STRING_CONTINUITY_BONUS``."""

POSITION_SHIFT_COST = float(os.environ.get("TABVISION_POSITION_SHIFT_COST", "2.5"))
"""Cost per fret of ``|curr.fret - prev.fret|`` (after normalisation by
``SPAN_NORM``). Hand-position-continuity weight. **Default 2.5** (raised from
0.05 on 2026-06-02): on GuitarSet validation it lifts single-line Tab F1
0.508 → 0.523 and strummed 0.671 → 0.676 with no regression — the old 0.05
left continuity effectively off. Env-overridable (``TABVISION_POSITION_SHIFT_COST``)
for sweeps. See docs/EVAL_REPORTS/acoustic_single_line_2026-06-02.md."""

SPAN_NORM = 12
"""Normalisation for ``POSITION_SHIFT_COST`` — one octave."""

MAX_HAND_SPAN = 5
"""Frets — beyond this distance the hand-span barrier kicks in."""

HAND_SPAN_BARRIER = 5.0
"""Cost added per fret of overshoot beyond ``MAX_HAND_SPAN``. Steep
enough to act as a soft hard-constraint while still allowing a jump
when audio + vision agree strongly."""

TRANSITION_PRIOR_WEIGHT = float(os.environ.get("TABVISION_TRANSITION_PRIOR_WEIGHT", "1.0"))
"""Weight on the learned transition-prior term (A15). Only consulted when a
prior is installed — default is OFF (no prior). Env-overridable for sweeps
(``TABVISION_TRANSITION_PRIOR_WEIGHT``), matching ``POSITION_SHIFT_COST``."""

_TRANSITION_PRIOR: TransitionPrior | None = None
_TRANSITION_PRIOR_ENV_READ = False

EPS = 1e-9


def set_transition_prior(prior: TransitionPrior | None, weight: float | None = None) -> None:
    """Install (or clear, with ``None``) the learned transition prior.

    Probe/sweep entrypoint; CLI runs install via the
    ``TABVISION_TRANSITION_PRIOR`` env var instead.
    """
    global _TRANSITION_PRIOR, _TRANSITION_PRIOR_ENV_READ, TRANSITION_PRIOR_WEIGHT
    _TRANSITION_PRIOR = prior
    _TRANSITION_PRIOR_ENV_READ = True  # explicit install wins over the env var
    if weight is not None:
        TRANSITION_PRIOR_WEIGHT = weight


def active_transition_prior() -> TransitionPrior | None:
    """The installed transition prior, lazily loading from the env once."""
    global _TRANSITION_PRIOR, _TRANSITION_PRIOR_ENV_READ
    if not _TRANSITION_PRIOR_ENV_READ:
        _TRANSITION_PRIOR_ENV_READ = True
        name = os.environ.get("TABVISION_TRANSITION_PRIOR", "").strip()
        if name and name.lower() != "none":
            _TRANSITION_PRIOR = load_transition_prior(name)
    return _TRANSITION_PRIOR


def find_fingering_at(t: float, fingerings: Sequence[FrameFingering]) -> FrameFingering | None:
    """Return the ``FrameFingering`` whose ``.t`` is closest to ``t``.

    Returns ``None`` when ``fingerings`` is empty or no entry carries
    evidence (logits None, empty, or all-zero). Ties broken by earliest.
    """
    if not fingerings:
        return None
    best: FrameFingering | None = None
    best_dt = math.inf
    for f in fingerings:
        if f.homography_confidence <= 0.0:
            continue
        if f.finger_pos_logits is None or f.finger_pos_logits.size == 0:
            continue
        if not (f.finger_pos_logits != 0).any():
            continue
        dt = abs(f.t - t)
        if dt < best_dt:
            best = f
            best_dt = dt
    return best


def emission_cost(
    candidate: Candidate,
    event: AudioEvent,
    fingering: FrameFingering | None,
    cfg: GuitarConfig,
    *,
    lambda_vision: float = 1.0,
) -> float:
    """Emission cost (negative log-prob) for ``candidate`` given ``event``.

    Decomposition (lower = better):

    - ``-log(event.confidence)`` — per-event constant (does not affect
      ranking within a single event but matters across events).
    - ``-log(event.fret_prior[s, f])`` — only when the audio backend or
      video neck-anchor path provides a prior. A one-dimensional fret-only
      prior is also accepted and read as ``event.fret_prior[f]``.
    - ``lambda_vision * -log(P_vision[s, f])`` — vision marginal at
      ``event.onset_s``. Skipped when ``fingering is None``.
    - ``LOW_FRET_BIAS * fret`` — gentle low-fret preference.
    - ``-OPEN_STRING_BONUS`` when ``fret == 0``.
    """
    cost = -math.log(max(event.confidence, EPS))

    if event.fret_prior is not None:
        prior = _candidate_prior(event.fret_prior, candidate)
        cost += -math.log(max(prior, EPS))

    if fingering is not None:
        marginal = fingering.marginal_string_fret()
        p = float(marginal[candidate.string_idx, candidate.fret])
        vision_weight = lambda_vision * max(0.0, min(1.0, fingering.homography_confidence))
        cost += vision_weight * (-math.log(max(p, VISION_FLOOR)))

    cost += LOW_FRET_BIAS * candidate.fret
    if candidate.fret == 0:
        cost -= OPEN_STRING_BONUS

    return cost


def _candidate_prior(prior: object, candidate: Candidate) -> float:
    """Read a candidate prior from either a 2D position prior or 1D fret prior."""
    try:
        arr = prior  # keep mypy's object handling local to this helper
        shape = getattr(arr, "shape", ())
        if len(shape) == 2:
            return float(arr[candidate.string_idx, candidate.fret])  # type: ignore[index]
        if len(shape) == 1:
            return float(arr[candidate.fret])  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return 0.0
    return 0.0


def transition_cost(
    prev: Candidate,
    curr: Candidate,
    cfg: GuitarConfig,
    *,
    use_sequence_prior: bool = True,
) -> float:
    """Transition cost from ``prev`` to ``curr``.

    - String continuity: ``-SAME_STRING_BONUS`` when on the same string.
    - Position shift: ``POSITION_SHIFT_COST * |Δfret| / SPAN_NORM``.
    - Hand-span barrier: ``HAND_SPAN_BARRIER * max(0, |Δfret| - MAX_HAND_SPAN)``.
    - Learned sequence prior (A15): ``TRANSITION_PRIOR_WEIGHT * -log P``
      when a prior is installed — default off. Callers pass
      ``use_sequence_prior=False`` for transitions the prior should not
      shape (the Viterbi gates it to singleton→singleton cluster moves;
      chord-to-chord movement stays on the hand-coded terms — that is
      chord-dictionary territory, see roadmap A5/A15).
    """
    cost = 0.0
    delta = abs(curr.fret - prev.fret)
    cost += POSITION_SHIFT_COST * delta / SPAN_NORM
    if delta > MAX_HAND_SPAN:
        cost += HAND_SPAN_BARRIER * (delta - MAX_HAND_SPAN)
    if curr.string_idx == prev.string_idx:
        cost -= SAME_STRING_BONUS
    if use_sequence_prior:
        prior = active_transition_prior()
        if prior is not None:
            cost += TRANSITION_PRIOR_WEIGHT * prior.cost(prev, curr, cfg)
    return cost


__all__ = [
    "find_fingering_at",
    "emission_cost",
    "transition_cost",
    "set_transition_prior",
    "active_transition_prior",
    "TRANSITION_PRIOR_WEIGHT",
    "LOW_FRET_BIAS",
    "OPEN_STRING_BONUS",
    "VISION_FLOOR",
    "SAME_STRING_BONUS",
    "POSITION_SHIFT_COST",
    "SPAN_NORM",
    "MAX_HAND_SPAN",
    "HAND_SPAN_BARRIER",
]
