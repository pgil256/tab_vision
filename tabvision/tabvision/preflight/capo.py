"""Capo detection for preflight (ROI deep-dive §4.3, piece 1).

The capo-covariant prior is worth ~+0.37 Tab F1 to a capo user
(`q7_capo_audio_2026-07-23.md`), but only if something knows the capo. Today
it must be passed on the command line. This estimates it from the recording.

**Pitch content alone cannot identify a capo.** A capo at ``C`` playing a
shape produces exactly the pitches of capo 0 playing the same music
transposed up ``C``; the note sets are identical. Two signals break the tie:

1. **The physical floor.** With a capo at ``C`` nothing below
   ``open_midi[0] + C`` is playable. That is a hard *upper bound* on the capo
   (a piece may simply avoid low notes), never a point estimate.
2. **Inharmonicity.** A capo shortens every string, and ``B`` scales as
   ``2^(n/6)`` in the absolute fret — so a capoed instrument is measurably
   stiffer than an open one playing the same pitches. This is the only
   evidence that is *causally* tied to the capo rather than to repertoire,
   and it is the same measurement the string-evidence channel already makes.

Both estimators are provided so the second can be judged against the first.
Neither changes routing: this module reports, and the caller decides.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.inharmonicity import LOG2, StringStiffnessModel, measure_events
from tabvision.types import AudioEvent, GuitarConfig

DEFAULT_MAX_CAPO = 7
FLOOR_QUANTILE = 0.02
"""Low quantile used for the physical floor, so one spurious low detection
does not veto every capo hypothesis."""


@dataclass(frozen=True)
class CapoEstimate:
    """One capo estimate with the evidence behind it."""

    capo: int
    confidence: float
    method: str
    upper_bound: int
    """Highest capo consistent with the lowest pitch actually heard."""

    scores: tuple[float, ...]
    """Per-hypothesis score, index = candidate capo."""

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"capo {self.capo} ({self.method}, confidence {self.confidence:.2f})"


def _physical_upper_bound(events: Sequence[AudioEvent], cfg: GuitarConfig, max_capo: int) -> int:
    """Highest capo consistent with the lowest pitch heard."""
    if not events:
        return 0
    pitches = np.asarray([event.pitch_midi for event in events], dtype=np.float64)
    floor = float(np.quantile(pitches, FLOOR_QUANTILE))
    return int(max(0, min(max_capo, math.floor(floor - min(cfg.tuning_midi)))))


def detect_capo_from_video(
    video_capo: int | None,
    video_confidence: float,
    events: Sequence[AudioEvent],
    cfg: GuitarConfig | None = None,
    *,
    max_capo: int = DEFAULT_MAX_CAPO,
) -> CapoEstimate:
    """Combine a camera's capo estimate with the audio physical bound.

    Q7 refuted pitch-based detection from first principles — a capo at ``C``
    and a transposition up ``C`` have identical note sets — so the only audio
    signal that survives is the *upper bound* from the lowest pitch heard,
    which is sound but weak (it held 60/60, but a piece avoiding low notes
    permits a capo it does not have).

    A camera supplies what audio cannot: a point estimate. A capo is a large,
    static, full-width bar, which is the one fretboard feature low-resolution
    video resolves well. The two evidence types are complementary in exactly
    the right way — **the bound cannot locate a capo, but it can refute one.**
    A video estimate above the physical bound is impossible, so it is rejected
    rather than reported.

    ``video_capo`` is a plain fret number so this module stays independent of
    the quarantined FretCam package; the caller adapts.

    Reports only. Its field accuracy on real capo footage is unmeasured — no
    such footage with ground truth exists in this repository — so callers must
    confirm with a human rather than route on it silently.
    """
    cfg = cfg or GuitarConfig()
    bound = _physical_upper_bound(events, cfg, max_capo)
    scores = [0.0] * (max_capo + 1)

    if video_capo is None:
        return CapoEstimate(0, 0.0, "video", bound, tuple(scores))

    capo = int(video_capo)
    if not 0 <= capo <= max_capo:
        return CapoEstimate(0, 0.0, "video-out-of-range", bound, tuple(scores))

    confidence = float(max(0.0, min(1.0, video_confidence)))
    if capo > bound:
        # The recording contains pitches this capo makes unplayable. The bound
        # is a hard physical constraint; the camera is not. Believe the bound.
        return CapoEstimate(0, 0.0, "video-refuted-by-bound", bound, tuple(scores))

    scores[capo] = confidence
    return CapoEstimate(capo, confidence, "video", bound, tuple(scores))


def detect_capo_from_pitches(
    events: Sequence[AudioEvent],
    cfg: GuitarConfig | None = None,
    *,
    max_capo: int = DEFAULT_MAX_CAPO,
) -> CapoEstimate:
    """Baseline estimator: physical floor plus open-string occupancy.

    Scores each hypothesis by how much of the recording lands exactly on that
    capo's open-string pitches, since the notes at the capo are the most
    comfortable and tend to be the most played. This is a *repertoire*
    heuristic, not physics, and the module docstring explains why it cannot
    fully separate a capo from a transposition.
    """
    cfg = cfg or GuitarConfig()
    bound = _physical_upper_bound(events, cfg, max_capo)
    if not events:
        return CapoEstimate(0, 0.0, "pitches", bound, ())

    pitches = [event.pitch_midi for event in events]
    scores: list[float] = []
    for capo in range(max_capo + 1):
        if capo > bound:
            scores.append(0.0)
            continue
        open_set = {open_midi + capo for open_midi in cfg.tuning_midi}
        scores.append(sum(1 for pitch in pitches if pitch in open_set) / len(pitches))

    best = max(range(max_capo + 1), key=lambda c: (scores[c], -c))
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    confidence = float(max(0.0, min(1.0, margin / max(ordered[0], 1e-9))))
    return CapoEstimate(best, confidence, "pitches", bound, tuple(scores))


def detect_capo_from_inharmonicity(
    events: Sequence[AudioEvent],
    wav: np.ndarray,
    sr: int,
    model: StringStiffnessModel,
    cfg: GuitarConfig | None = None,
    *,
    max_capo: int = DEFAULT_MAX_CAPO,
    min_r2: float = 0.5,
    min_notes: int = 8,
) -> CapoEstimate:
    """Physics estimator: which capo makes the measured stiffness consistent?

    For each hypothesis the notes are placed at their lowest playable position
    under that capo and the stiffness those positions imply is compared with
    the stiffness actually measured. ``B`` depends on the *absolute* fret, so
    hypothesising a capo that is too low predicts a slacker, longer string
    than the recording actually contains and leaves a systematic positive
    residual — which is exactly the asymmetry pitch content lacks.

    Scored by ``-|median residual|``, so the best hypothesis is the one whose
    predicted stiffness is unbiased rather than merely close.
    """
    cfg = cfg or GuitarConfig()
    bound = _physical_upper_bound(events, cfg, max_capo)
    ordered = sorted(events, key=lambda event: event.onset_s)
    fits = measure_events(ordered, wav, sr, cfg)
    usable = [(index, fit) for index, fit in fits.items() if fit.r2 >= min_r2]
    if len(usable) < min_notes:
        # Not enough measurable notes to say anything; defer to the floor.
        return CapoEstimate(0, 0.0, "inharmonicity (insufficient)", bound, ())

    scores: list[float] = []
    for capo in range(max_capo + 1):
        if capo > bound:
            scores.append(-math.inf)
            continue
        probe = GuitarConfig(
            tuning_midi=cfg.tuning_midi,
            capo=capo,
            max_fret=cfg.max_fret,
            n_strings=cfg.n_strings,
        )
        residuals: list[float] = []
        for index, fit in usable:
            candidates = candidate_positions(ordered[index].pitch_midi, probe)
            if not candidates:
                continue
            # Lowest playable position: assumption-free, and its bias under a
            # wrong capo is what carries the signal.
            pick = min(candidates, key=lambda c: (c.fret, c.string_idx))
            base = model.log_b0.get(pick.string_idx)
            if base is None:
                continue
            predicted = base + model.fret_exponent * (pick.fret / 6.0) * LOG2
            residuals.append(fit.log_b - predicted)
        scores.append(-abs(float(np.median(residuals))) if residuals else -math.inf)

    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return CapoEstimate(0, 0.0, "inharmonicity (no hypothesis)", bound, ())
    best = max(range(max_capo + 1), key=lambda c: (scores[c], -c))
    ranked = sorted(finite, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    # Residual gaps are in log-B units; 0.1 (~10%) is a decisive separation.
    confidence = float(max(0.0, min(1.0, abs(margin) / 0.1)))
    return CapoEstimate(best, confidence, "inharmonicity", bound, tuple(scores))


__all__ = [
    "CapoEstimate",
    "DEFAULT_MAX_CAPO",
    "detect_capo_from_inharmonicity",
    "detect_capo_from_pitches",
    "detect_capo_from_video",
]
