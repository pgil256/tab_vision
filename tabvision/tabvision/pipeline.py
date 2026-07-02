"""End-to-end transcription pipeline.

Public entrypoint: :func:`run_pipeline`. Composes ``demux`` → audio
backend → guitar / fretboard / hand backends → :func:`fuse` into a
single call returning ``list[TabEvent]``. The video stack runs in a
single forward pass over frames (subsampled by ``video_stride``); audio
is processed in parallel from the same demux result.

Graceful degradation:

- **Missing import** (e.g. ``mediapipe`` or ``cv2`` not installed) →
  audio-only fallback with a logged warning. Caller still gets tab events.
- **Missing YOLO checkpoint** (when video was explicitly requested) →
  :class:`tabvision.errors.BackendError` with the acquire command in
  the message. Silent fallback would mask a fixable user-side problem.
- **`video_enabled=False`** → skip the video stack entirely; audio-only.

See ``docs/plans/2026-05-06-video-pipeline-integration-design.md`` for
the design.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from tabvision.demux import demux
from tabvision.fusion import TimedNeckAnchor, apply_neck_anchor_priors, fuse
from tabvision.fusion.melodic_prior import apply_melodic_segment_prior
from tabvision.fusion.neck_prior import NeckAnchorLike
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.types import (
    AudioBackend,
    AudioEvent,
    FrameFingering,
    FretboardBackend,
    GuitarBackend,
    GuitarConfig,
    HandBackend,
    Homography,
    SessionConfig,
    TabEvent,
)

if TYPE_CHECKING:
    from tabvision.audio.filters import AudioFilterConfig

logger = logging.getLogger(__name__)


class _VideoImportError(RuntimeError):
    """Internal signal: a soft-optional video dep failed to import."""


@dataclass(frozen=True)
class _VideoStackResult:
    fingerings: list[FrameFingering]
    neck_anchors: list[TimedNeckAnchor]


def run_pipeline(
    video_path: str | Path,
    *,
    audio_backend: AudioBackend | None = None,
    audio_backend_name: str = "highres",
    guitar_backend: GuitarBackend | None = None,
    fretboard_backend: FretboardBackend | None = None,
    hand_backend: HandBackend | None = None,
    lambda_vision: float = 1.0,
    video_stride: int = 3,
    video_enabled: bool = True,
    position_prior: str | None = None,
    melodic_prior_enabled: bool = False,
    audio_filters: bool | AudioFilterConfig | None = None,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[TabEvent]:
    """Run the full transcription pipeline on ``video_path``.

    Backends are injectable for testing; when omitted, the function
    constructs production backends on demand.

    ``audio_filters`` controls the constructed backend's post-detection
    filtering (``tabvision.audio.filters``): ``None`` (default) keeps each
    backend's built-in default — basicpitch on, highres off — so the default
    path is unchanged; ``True``/``False`` or an explicit ``AudioFilterConfig``
    forces it. Ignored when ``audio_backend`` is supplied (the injected
    backend carries its own filter config).

    ``progress_callback``, when given, is invoked with a stage name as each
    pipeline stage *starts*: ``demux`` → ``model_load`` → ``audio_inference``
    → ``video_analysis`` (only when the video stack runs) → ``decode``.
    Rendering happens outside this function, so callers wanting a render
    stage emit it themselves after this returns. Callback exceptions are
    logged and swallowed — progress reporting must never break a
    transcription. Default ``None`` is a strict no-op.
    """
    cfg = cfg or GuitarConfig()
    session = session or SessionConfig()

    def _notify(stage: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage)
        except Exception as exc:  # noqa: BLE001 — progress must never break transcription
            logger.warning("progress callback failed on %r: %s", stage, exc)

    _notify("demux")
    logger.info("demuxing %s", video_path)
    demuxed = demux(video_path)

    # Tone toggle: "auto" routes to the backend for the session's instrument
    # (electric → highres-electric, else acoustic highres). Explicit names pass through.
    _notify("model_load")
    if audio_backend is None and audio_backend_name == "auto":
        audio_backend_name = audio_backend_for_session(session)
    if audio_backend is not None:
        audio = audio_backend
    else:
        # audio_filters None ⇒ keep the backend's built-in default (basicpitch
        # on, highres off); a bool / AudioFilterConfig overrides it.
        filter_kwargs = {} if audio_filters is None else {"filter_config": audio_filters}
        audio = _make_audio_backend(audio_backend_name, **filter_kwargs)
    # Backends load checkpoints lazily on first transcribe, so the load time
    # lands in the audio_inference stage; both map to "analyzing audio" UX-side.
    _notify("audio_inference")
    logger.info("transcribing audio with %s", audio.name)
    audio_events = audio.transcribe(demuxed.wav, demuxed.sample_rate, session)
    logger.info("audio backend produced %d events", len(audio_events))

    if position_prior and position_prior != "none":
        prior = load_pitch_position_prior(position_prior, cfg=cfg)
        audio_events = apply_pitch_position_prior(audio_events, prior)
        logger.info("attached pitch-position prior %s", position_prior)

    if melodic_prior_enabled:
        audio_events = apply_melodic_segment_prior(audio_events, cfg)
        logger.info("attached melodic segment prior")

    fingerings: list[FrameFingering] = []
    neck_anchors: list[TimedNeckAnchor] = []
    if video_enabled:
        _notify("video_analysis")
        try:
            video_result = _run_video_stack(
                demuxed.frame_iterator,
                stride=video_stride,
                cfg=cfg,
                guitar_backend=guitar_backend,
                fretboard_backend=fretboard_backend,
                hand_backend=hand_backend,
            )
            fingerings = video_result.fingerings
            neck_anchors = video_result.neck_anchors
        except _VideoImportError as exc:
            logger.warning(
                "video stack unavailable, falling back to audio-only: %s",
                exc,
            )

    if lambda_vision > 0.0 and neck_anchors:
        audio_events = apply_neck_anchor_priors(audio_events, neck_anchors, cfg)
        logger.info("attached %d hand-neck anchors as audio fret priors", len(neck_anchors))

    _notify("decode")
    logger.info(
        "running fuse() with %d audio events, %d fingerings, lambda_vision=%.2f",
        len(audio_events),
        len(fingerings),
        lambda_vision,
    )
    return list(fuse(audio_events, fingerings, cfg, session, lambda_vision=lambda_vision))


# ---------------------------------------------------------------------------
# Video stack
# ---------------------------------------------------------------------------


def _run_video_stack(
    frames: Iterable[tuple[float, np.ndarray]] | Iterator[tuple[float, np.ndarray]],
    *,
    stride: int,
    cfg: GuitarConfig,
    guitar_backend: GuitarBackend | None,
    fretboard_backend: FretboardBackend | None,
    hand_backend: HandBackend | None,
) -> _VideoStackResult:
    """Single-pass walk producing one ``FrameFingering`` per sampled frame.

    Skipped-by-stride frames produce nothing; sampled frames produce
    either a real fingering (full pipeline succeeded) or an empty
    fingering at the right timestamp (any earlier stage failed). Empty
    fingerings let ``playability.find_fingering_at`` see the timestamp
    grid without contributing evidence.
    """
    if stride < 1:
        raise ValueError(f"video_stride must be >= 1, got {stride}")

    if guitar_backend is None:
        guitar_backend = _make_guitar_backend()
    if fretboard_backend is None:
        fretboard_backend = _make_fretboard_backend()
    if hand_backend is None:
        hand_backend = _make_hand_backend()

    fingerings: list[FrameFingering] = []
    neck_anchors: list[TimedNeckAnchor] = []
    n_fingers = 4  # fretting fingers; matches Phase 4 convention.
    empty_logits = np.zeros((n_fingers, cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)

    for frame_idx, (t, frame) in enumerate(frames):
        if frame_idx % stride != 0:
            continue
        bbox = guitar_backend.detect(frame)
        if bbox is None or bbox.confidence <= 0.0:
            fingerings.append(
                FrameFingering(
                    t=t,
                    finger_pos_logits=empty_logits.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        H = fretboard_backend.detect(frame, bbox)  # noqa: N806
        if H.confidence <= 0.0:
            fingerings.append(
                FrameFingering(
                    t=t,
                    finger_pos_logits=empty_logits.copy(),
                    homography_confidence=0.0,
                )
            )
            continue
        ff = hand_backend.detect(frame, H, cfg)
        # Backends produce a degenerate t=0.0; stamp the real timestamp here.
        fingerings.append(replace(ff, t=t))
        anchor = _detect_neck_anchor(hand_backend, frame, H, cfg)
        if anchor is not None and anchor.confidence > 0.0:
            neck_anchors.append((t, anchor))

    return _VideoStackResult(fingerings=fingerings, neck_anchors=neck_anchors)


def _detect_neck_anchor(
    hand_backend: HandBackend,
    frame: np.ndarray,
    H: Homography,  # noqa: N803 — optional extension outside the §8 protocol
    cfg: GuitarConfig,
) -> NeckAnchorLike | None:
    """Use a backend's optional coarse neck-anchor hook when available."""
    detect_anchor = getattr(hand_backend, "detect_anchor", None)
    if detect_anchor is None:
        return None
    try:
        return cast(NeckAnchorLike | None, detect_anchor(frame, H, cfg))
    except Exception as exc:  # noqa: BLE001 — optional evidence must degrade softly
        logger.debug("hand-neck anchor unavailable on frame: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend factories — deferred imports so audio-only callers don't pay the
# vision-extras cost.
# ---------------------------------------------------------------------------


def audio_backend_for_session(session: SessionConfig) -> str:
    """Audio backend for a session's declared instrument — the user-facing toggle.

    Electric → the separately fine-tuned electric checkpoint (``highres-electric``);
    acoustic / classical → the acoustic ``highres`` default. Separate checkpoints,
    so the acoustic model is never disturbed (see
    ``docs/plans/2026-06-02-electric-backbone-finetune-design.md``). Used when
    ``run_pipeline`` is called with ``audio_backend_name="auto"``.
    """
    if session.instrument == "electric":
        return "highres-electric"
    return "highres"


def _make_audio_backend(name: str, **kwargs) -> AudioBackend:  # type: ignore[no-untyped-def]
    from tabvision.audio.backend import make

    return make(name, **kwargs)


def _make_guitar_backend() -> GuitarBackend:
    try:
        from tabvision.video.guitar.yolo_backend import YoloOBBBackend
    except ImportError as exc:
        raise _VideoImportError(f"YOLO backend import failed: {exc}") from exc
    return YoloOBBBackend()


def _make_fretboard_backend() -> FretboardBackend:
    try:
        from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
    except ImportError as exc:
        raise _VideoImportError(f"keypoint fretboard backend import failed: {exc}") from exc
    return KeypointFretboardBackend()


def _make_hand_backend() -> HandBackend:
    try:
        from tabvision.video.hand.mediapipe_backend import MediaPipeHandBackend
    except ImportError as exc:
        raise _VideoImportError(f"MediaPipe hand backend import failed: {exc}") from exc
    return MediaPipeHandBackend()


# Re-export AudioEvent / TabEvent for ergonomic ``from tabvision.pipeline import TabEvent``.
__all__ = ["run_pipeline", "AudioEvent", "TabEvent"]
