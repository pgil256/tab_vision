"""Canonical shared types — see SPEC.md §8.

These dataclasses and protocols are the immutable contracts between modules.
Signatures are stable within the v1 cycle. Implementations behind them may
evolve; signatures may not, except by explicit user approval and a
corresponding update to SPEC.md.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

# Standard tuning, low E to high E
DEFAULT_TUNING_MIDI: tuple[int, ...] = (40, 45, 50, 55, 59, 64)

InstrumentType = Literal["acoustic", "classical", "electric"]
ToneType = Literal["clean", "distorted"]
PlayingStyle = Literal["fingerstyle", "strumming", "mixed"]


@dataclass(frozen=True)
class GuitarConfig:
    """Physical instrument properties. Stable per-instrument."""

    tuning_midi: tuple[int, ...] = DEFAULT_TUNING_MIDI
    capo: int = 0
    max_fret: int = 24
    n_strings: int = 6


@dataclass(frozen=True)
class SessionConfig:
    """Per-recording context. Affects model selection and fusion behavior."""

    instrument: InstrumentType = "acoustic"
    tone: ToneType = "clean"
    style: PlayingStyle = "mixed"


@dataclass
class AudioEvent:
    onset_s: float
    offset_s: float
    pitch_midi: int
    velocity: float
    confidence: float
    pitch_logits: np.ndarray | None = None
    fret_prior: np.ndarray | None = None
    tags: tuple[str, ...] = ()


@dataclass
class DemuxResult:
    """Output of ``demux(video_path)`` — see SPEC.md §8.

    ``wav`` is mono float32 audio at ``sample_rate`` Hz. ``frame_iterator``
    yields ``(timestamp_s, frame_bgr)`` tuples on demand; iteration is
    one-pass and not seekable.
    """

    wav: np.ndarray
    sample_rate: int
    duration_s: float
    fps: float
    frame_iterator: Iterator[tuple[float, np.ndarray]] = field(default_factory=iter)


@dataclass
class GuitarBBox:
    """Bounding box of the guitar in image coords.

    Default is axis-aligned (``rotation_deg=0``). YOLO-OBB backends emit
    rotated boxes; consumers that don't care about rotation can read
    (x, y, w, h) as the enclosing axis-aligned box per backend convention
    and ignore ``rotation_deg``. Rotation is in degrees, counter-clockwise
    from the +x axis, in the same image coordinate system.

    Spec extension recorded 2026-05-05 (Phase 3 entry): ``rotation_deg``
    added to support the YOLO-OBB detector path. Backward-compatible —
    default 0.0 keeps the axis-aligned semantics.
    """

    x: float
    y: float
    w: float
    h: float
    confidence: float
    rotation_deg: float = 0.0


@dataclass
class GuitarTrack:
    """Per-frame guitar bbox + smoothed track."""

    boxes: list[GuitarBBox]
    fps: float
    stability_px: float


@dataclass
class PreflightFinding:
    severity: Literal["info", "warn", "fail"]
    code: str
    message: str


@dataclass
class PreflightReport:
    """Output of the preflight tool."""

    passed: bool
    findings: list[PreflightFinding]
    suggested_actions: list[str]


@dataclass
class Homography:
    H: np.ndarray
    confidence: float
    method: str


@dataclass
class FrameFingering:
    t: float
    finger_pos_logits: np.ndarray
    homography_confidence: float

    def marginal_string_fret(self) -> np.ndarray:
        """Marginalize over fingers → shape (n_strings, max_fret+1) softmax.

        Aggregates each finger's per-cell logits via log-sum-exp ("any
        finger here?"), then softmax-normalises so the output sums to 1.
        Backed by ``video.hand.fingertip_to_fret.marginal_string_fret``.
        """
        from tabvision.video.hand.fingertip_to_fret import marginal_string_fret
        return marginal_string_fret(self.finger_pos_logits)


@dataclass
class TabEvent:
    onset_s: float
    duration_s: float
    string_idx: int
    fret: int
    pitch_midi: int
    confidence: float
    techniques: tuple[str, ...] = ()


# ----- Backend protocols -----


class AudioBackend(Protocol):
    name: str

    def transcribe(
        self, wav: np.ndarray, sr: int, session: SessionConfig
    ) -> Sequence[AudioEvent]:
        ...


class GuitarBackend(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> GuitarBBox:
        ...


class FretboardBackend(Protocol):
    name: str

    def detect(self, frame: np.ndarray, guitar_box: GuitarBBox) -> Homography:
        ...


class HandBackend(Protocol):
    name: str

    def detect(
        self,
        frame: np.ndarray,
        H: Homography,  # noqa: N803 — math-convention name baked into the §8 contract
        cfg: GuitarConfig,
    ) -> FrameFingering:
        ...
