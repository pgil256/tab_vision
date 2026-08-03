"""Session-level capo detection from the fretboard geometry FretCam already has.

Why this and not something else. Q7
(``docs/EVAL_REPORTS/q7_capo_detect_2026-07-23.md``) established two things:
the capo-covariant position prior is worth roughly **+0.37 Tab F1 to a capo
user**, and **audio cannot recover the capo** — a capo at fret ``C`` playing a
shape produces exactly the pitch set of capo 0 playing the same music
transposed up ``C``, so no amount of pitch analysis separates them. Pitch-based
detection measured 1/60. Today the capo must be typed in by hand.

A capo is the one thing on a guitar neck that suits this camera. It is large,
high-contrast, spans every string, and — decisively — **does not move for the
entire session**. That is the opposite of the per-note, per-frame precision
that 640×360 footage cannot deliver
(``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``), so session-level
integration sidesteps FretCam's coverage problem instead of fighting it.

**Method.** ``FrameDetection.fret_ticks`` already publishes each fret wire as a
line segment across the neck in image coordinates. A capo is a narrow, dark
band spanning the strings immediately body-side of one of those wires, in
essentially every frame. Each observed frame contributes a darkness profile
over candidate frets; the session estimate is the fret that is both darkest by
a margin and *persistent*. Working straight off the published ticks means this
needs no homography of its own and cannot drift from the geometry the rest of
the chain uses.

**Persistence is what separates a capo from a barre chord**, which looks nearly
identical in a single frame. A barre comes and goes; a capo is always there. A
fret must lead in at least :data:`MIN_PERSISTENCE` of observed frames to be
reported at all.

**Status: reporting only.** This never sets ``cfg.capo``. It produces an
estimate and a confidence for a human to confirm, which is the disposition Q7
reached ("report the bound and ask", since auto-setting from pitch would be
wrong ~98% of the time). Its field accuracy on real capo footage is
**unmeasured** — no such footage with ground truth exists in this repository —
so it must not be wired to anything that acts without confirmation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class FretTickLike(Protocol):
    """One fret wire as a segment across the neck, in image coordinates.

    Structurally satisfied by :class:`fretcam.detection.FretTick`. The
    members are read-only properties, not plain attributes: ``FretTick`` is
    a frozen dataclass, and mypy only lets a frozen (read-only) attribute
    satisfy a protocol member declared read-only.
    """

    @property
    def fret(self) -> int: ...

    @property
    def start(self) -> tuple[float, float]: ...

    @property
    def end(self) -> tuple[float, float]: ...


MAX_CAPO_FRET = 7
"""Highest fret considered. Matches ``tabvision.preflight.capo.DEFAULT_MAX_CAPO``."""

STRING_SPAN = (0.15, 0.85)
"""Canonical y range sampled, trimmed to avoid the neck edges."""

BAND_WIDTH_CANONICAL = 0.35
"""Sampling band width as a fraction of the cell a capo occupies.

A capo at fret ``N`` clamps inside cell ``N`` — between wire ``N-1`` and wire
``N`` — pressed up against wire ``N``, which is what makes the string speak at
fret ``N``. So the band runs *back* from wire ``N`` toward wire ``N-1``, not
forward. Sampling the whole cell would average the capo with bare fretboard and
wash the signal out.
"""

SAMPLES_ACROSS = 12
"""Sample points across the strings, per band."""

SAMPLES_ALONG = 3
"""Sample points along the neck axis, per band."""

MIN_MARGIN = 0.06
"""Minimum normalized darkness lead over the runner-up fret, in [0, 1] units."""

MIN_PERSISTENCE = 0.60
"""Fraction of observed frames a fret must lead to be called a capo."""

MIN_STRING_COVERAGE = 0.80
"""Fraction of across-string samples that must be dark for a capo candidate.

Persistence alone does **not** separate a capo from a stationary fretting hand:
a player who stays in one position darkens that cell in most frames, which is
precisely a capo's temporal signature. Measured on the capo-free GAPS control,
fret 4 led 26/90 frames on `027_Zpswc` and 44/90 on `179_pM1wc` — the hand, not
the nut.

What a hand cannot fake is *width*. A capo is a rigid bar clamping every string;
a hand darkens the two or three strings its fingers are on and leaves the rest
bare. So a candidate must be dark across essentially the whole string span, not
merely dark on average.
"""

MIN_FRAMES = 12
"""Minimum observed frames before any estimate is reported."""

SEMITONE_RATIO = 2.0 ** (-1.0 / 12.0)
"""Rule of 18, used for the neck-quad fallback geometry."""


@dataclass(frozen=True)
class _DerivedTick:
    fret: int
    start: tuple[float, float]
    end: tuple[float, float]


def fret_ticks_from_neck_quad(
    neck_quad: Sequence[tuple[float, float]],
    body_joint_fret: int,
    *,
    count: int,
) -> tuple[_DerivedTick, ...]:
    """Derive approximate fret wires from the neck quad alone.

    ``FrameDetection.fret_ticks`` is emitted only when the *calibrated* fret map
    has locked, which on the GAPS negative control happened on 5 of 12 clips —
    the other 7 abstained blind. ``neck_locked`` runs far higher (97–98% on
    clips where the fret map never locked), so this recovers the difference.

    The quad is the canonical ``[0,1]²`` corners projected to the image, in the
    order emitted by ``detection._neck_quad``: ``(0,0), (1,0), (1,1), (0,1)``,
    x running nut→body joint. Rule of 18 then places wire ``n`` at
    ``(1 - r^n) / (1 - r^J)`` along that axis, with ``J`` the body-joint fret.

    Deliberately coarser than the calibrated map. That is acceptable *here* and
    nowhere else in this package: identifying which of seven frets carries a
    full-width bar is a far weaker demand than assigning a note's fret, and the
    persistence gate absorbs the extra jitter.
    """
    if len(neck_quad) != 4 or body_joint_fret < 1 or count < 1:
        return ()
    corners = [(float(x), float(y)) for x, y in neck_quad]
    if not all(_finite_point(corner) for corner in corners):
        return ()
    nut_top, body_top, body_bottom, nut_bottom = corners

    denominator = 1.0 - SEMITONE_RATIO**body_joint_fret
    if abs(denominator) < 1e-12:
        return ()

    ticks: list[_DerivedTick] = []
    for fret in range(count):
        along = (1.0 - SEMITONE_RATIO**fret) / denominator
        if not math.isfinite(along):
            return ()
        top = (
            nut_top[0] + along * (body_top[0] - nut_top[0]),
            nut_top[1] + along * (body_top[1] - nut_top[1]),
        )
        bottom = (
            nut_bottom[0] + along * (body_bottom[0] - nut_bottom[0]),
            nut_bottom[1] + along * (body_bottom[1] - nut_bottom[1]),
        )
        ticks.append(_DerivedTick(fret=fret, start=top, end=bottom))
    return tuple(ticks)


@dataclass(frozen=True)
class CapoObservation:
    """A session-level capo estimate, or an explicit abstention."""

    fret: int | None
    confidence: float
    frames_observed: int
    frames_supporting: int
    margin: float
    reason: str

    @property
    def detected(self) -> bool:
        return self.fret is not None


class CapoDetector:
    """Accumulate per-frame darkness profiles into one session estimate."""

    def __init__(self, *, max_capo_fret: int = MAX_CAPO_FRET) -> None:
        if max_capo_fret < 1:
            raise ValueError("max_capo_fret must be >= 1")
        self.max_capo_fret = int(max_capo_fret)
        self._frames = 0
        self._leader_counts = np.zeros(self.max_capo_fret + 1, dtype=np.int64)
        self._margins: list[float] = []
        self._leaders: list[int] = []

    # -- accumulation ----------------------------------------------------

    def observe(
        self,
        frame_bgr: np.ndarray,
        fret_ticks: Sequence[FretTickLike],
        *,
        neck_quad: Sequence[tuple[float, float]] = (),
        body_joint_fret: int | None = None,
    ) -> int | None:
        """Score one frame; return the fret leading it, or ``None`` if unusable.

        ``fret_ticks`` is ``FrameDetection.fret_ticks`` — wire segments in image
        coordinates, each carrying its physical fret number. When the calibrated
        fret map has not locked those are empty, so ``neck_quad`` plus
        ``body_joint_fret`` (both also on ``FrameDetection``) provide a coarser
        rule-of-18 fallback rather than abstaining blind.
        """
        profile = self._darkness_profile(frame_bgr, fret_ticks)
        if profile is None and neck_quad and body_joint_fret:
            derived = fret_ticks_from_neck_quad(
                neck_quad, int(body_joint_fret), count=self.max_capo_fret + 2
            )
            if derived:
                profile = self._darkness_profile(frame_bgr, derived)
        if profile is None:
            return None
        self._frames += 1
        if float(profile.max()) <= 0.0:
            # Nothing spanned the strings. The frame still counts: it is
            # evidence *against* a capo, and dropping it would let a handful of
            # lucky frames carry the persistence fraction.
            return None
        order = np.argsort(profile)[::-1]
        leader = int(order[0]) + 1  # profile index 0 is fret 1
        runner_up = float(profile[order[1]]) if profile.size > 1 else 0.0
        margin = float(profile[order[0]]) - runner_up
        self._leader_counts[leader] += 1
        self._margins.append(margin)
        self._leaders.append(leader)
        return leader

    def _darkness_profile(
        self,
        frame_bgr: np.ndarray,
        fret_ticks: Sequence[FretTickLike],
    ) -> np.ndarray | None:
        """Normalized darkness just body-side of each candidate fret wire."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None
        by_fret = {
            int(tick.fret): tick
            for tick in fret_ticks
            if _finite_point(tick.start) and _finite_point(tick.end)
        }
        # Fret N's cell is bounded by wires N-1 and N, so wire 0 (the nut) is
        # needed and wire max+1 is not.
        needed = range(0, self.max_capo_fret + 1)
        if any(fret not in by_fret for fret in needed):
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        height, width = gray.shape[:2]
        offsets = np.linspace(STRING_SPAN[0], STRING_SPAN[1], SAMPLES_ACROSS)
        steps = np.linspace(0.0, BAND_WIDTH_CANONICAL, SAMPLES_ALONG)

        values = np.empty(self.max_capo_fret, dtype=np.float64)
        # Per-string means, kept so a candidate can be required to be dark
        # across the *whole* span rather than dark on average — the one thing
        # a fretting hand cannot imitate.
        per_string = np.full((self.max_capo_fret, SAMPLES_ACROSS), np.nan)
        for fret in range(1, self.max_capo_fret + 1):
            wire = by_fret[fret]
            behind = by_fret[fret - 1]
            samples: list[float] = []
            for offset_index, offset in enumerate(offsets):
                string_samples: list[float] = []
                # Walk across the strings on both wires, then step back from
                # wire N toward wire N-1 — the sliver a capo occupies.
                ax = wire.start[0] + offset * (wire.end[0] - wire.start[0])
                ay = wire.start[1] + offset * (wire.end[1] - wire.start[1])
                bx = behind.start[0] + offset * (behind.end[0] - behind.start[0])
                by = behind.start[1] + offset * (behind.end[1] - behind.start[1])
                for step in steps:
                    px = ax + step * (bx - ax)
                    py = ay + step * (by - ay)
                    if not (math.isfinite(px) and math.isfinite(py)):
                        continue
                    # Bounds-check the rounded index, not the float: 639.6
                    # passes `< 640` and then rounds to 640.
                    col, row = int(round(px)), int(round(py))
                    if not (0 <= col < width and 0 <= row < height):
                        continue
                    string_samples.append(gray[row, col])
                if string_samples:
                    per_string[fret - 1, offset_index] = float(np.mean(string_samples))
                    samples.extend(string_samples)
            if len(samples) < 0.75 * SAMPLES_ACROSS * SAMPLES_ALONG:
                return None
            values[fret - 1] = float(np.mean(samples))

        # Darkness relative to the neck's own brightness range, so the profile
        # is comparable across lighting and across frames.
        lo, hi = float(values.min()), float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            return np.zeros_like(values)
        darkness = (hi - values) / (hi - lo)

        # Across-string coverage: what fraction of the span is darker than the
        # midpoint of this frame's own range. A clamped bar darkens all of it;
        # fingers darken the strings they are on.
        midpoint = 0.5 * (lo + hi)
        coverage = np.zeros(self.max_capo_fret, dtype=np.float64)
        for fret_index in range(self.max_capo_fret):
            row = per_string[fret_index]
            valid = row[~np.isnan(row)]
            if valid.size:
                coverage[fret_index] = float(np.mean(valid < midpoint))
        # A candidate that does not span the strings is not a capo, whatever
        # its mean darkness.
        darkness[coverage < MIN_STRING_COVERAGE] = 0.0
        return darkness

    # -- readout ---------------------------------------------------------

    def estimate(self) -> CapoObservation:
        """Return the session estimate, abstaining unless the evidence holds."""
        if self._frames < MIN_FRAMES:
            return CapoObservation(
                fret=None,
                confidence=0.0,
                frames_observed=self._frames,
                frames_supporting=0,
                margin=0.0,
                reason="insufficient_frames",
            )

        leader = int(np.argmax(self._leader_counts))
        supporting = int(self._leader_counts[leader])
        persistence = supporting / self._frames
        leader_margins = [
            margin
            for margin, who in zip(self._margins, self._leaders, strict=True)
            if who == leader
        ]
        margin = float(np.median(leader_margins)) if leader_margins else 0.0

        if persistence < MIN_PERSISTENCE:
            return CapoObservation(
                fret=None,
                confidence=0.0,
                frames_observed=self._frames,
                frames_supporting=supporting,
                margin=margin,
                # Intermittent leadership is what a barre chord looks like.
                reason="not_persistent",
            )
        if margin < MIN_MARGIN:
            return CapoObservation(
                fret=None,
                confidence=0.0,
                frames_observed=self._frames,
                frames_supporting=supporting,
                margin=margin,
                reason="no_dark_band",
            )

        confidence = float(
            np.clip(persistence * min(1.0, margin / (2.0 * MIN_MARGIN)), 0.0, 1.0)
        )
        return CapoObservation(
            fret=leader,
            confidence=confidence,
            frames_observed=self._frames,
            frames_supporting=supporting,
            margin=margin,
            reason="detected",
        )


def _finite_point(point: tuple[float, float]) -> bool:
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return False
    return bool(np.isfinite(x) and np.isfinite(y))


__all__ = [
    "MAX_CAPO_FRET",
    "MIN_FRAMES",
    "MIN_MARGIN",
    "MIN_PERSISTENCE",
    "CapoDetector",
    "CapoObservation",
    "FretTickLike",
]
