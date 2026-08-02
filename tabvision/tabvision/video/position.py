"""Implementation-only coarse fret-position observations.

This module deliberately lives outside :mod:`tabvision.types`: the records
below connect the current FretCam implementation to pipeline orchestration
without changing the immutable SPEC section-8 contracts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

PositionObservationState = Literal["locked", "holding"]


@dataclass(frozen=True)
class PositionWindowObservation:
    """One confidence-gated fret-position window on the media timeline."""

    timestamp_s: float
    position: int
    window_frets: tuple[int, ...]
    confidence: float
    state: PositionObservationState


@dataclass(frozen=True)
class FingerContactObservation:
    """Per-finger string/fret contacts on the media timeline.

    The coarse window above collapses a whole hand into one integer position
    and is emitted only while the position estimator holds a lock. These
    records carry the finger-level detail that collapse discards, and are
    emitted whenever contacts exist — the estimator's lock criterion is a
    display requirement, not an evidence requirement.

    ``positions`` holds ``(string_idx, fret)`` pairs in TabVision's own
    convention: ``string_idx`` zero-based from the low E. Producers converting
    from FretCam's ``FingerContact.string`` must apply
    ``string_idx = n_strings - string``, because FretCam numbers strings the
    way tab notation does (1 = high E). The direct mapping measures a
    likelihood ratio of exactly 1.00 — see
    ``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``.
    """

    timestamp_s: float
    positions: tuple[tuple[int, int], ...]
    confidence: float


@dataclass(frozen=True)
class SessionCapoObservation:
    """One whole-session capo estimate from video, or an abstention.

    A capo is the one fretboard feature a camera reads well: large, static, and
    spanning every string. It is also the one thing audio provably *cannot*
    read — a capo at fret ``C`` and a transposition up ``C`` have identical
    pitch content (``docs/EVAL_REPORTS/q7_capo_detect_2026-07-23.md``, where
    pitch-based detection measured 1/60).

    Reporting only. ``fret`` is never routed on without human confirmation; see
    :func:`tabvision.preflight.capo.detect_capo_from_video`, which additionally
    refutes any estimate the audio's physical bound rules out.
    """

    fret: int | None
    confidence: float
    frames_observed: int
    reason: str


@dataclass(frozen=True)
class VideoObservations:
    """Everything one analyzer pass extracted from the frames."""

    windows: tuple[PositionWindowObservation, ...] = ()
    contacts: tuple[FingerContactObservation, ...] = ()
    capo: SessionCapoObservation | None = None


class PositionAnalyzer(Protocol):
    """Analyze timestamped BGR frames into stable coarse fret windows."""

    def analyze(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        stride: int = 1,
    ) -> list[PositionWindowObservation]: ...


class ContactAwarePositionAnalyzer(PositionAnalyzer, Protocol):
    """A analyzer that can also return finger contacts from the same pass.

    Frame decoding and inference are the expensive part, so contacts must come
    out of the *same* traversal as the windows rather than a second one.
    Callers detect this capability with :func:`supports_contacts`.
    """

    def analyze_all(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        *,
        stride: int = 1,
    ) -> VideoObservations: ...


def supports_contacts(analyzer: object) -> bool:
    """Return whether ``analyzer`` can emit finger contacts in one pass."""
    return callable(getattr(analyzer, "analyze_all", None))


__all__ = [
    "ContactAwarePositionAnalyzer",
    "FingerContactObservation",
    "PositionAnalyzer",
    "PositionObservationState",
    "PositionWindowObservation",
    "SessionCapoObservation",
    "VideoObservations",
    "supports_contacts",
]
