"""End-to-end transcription pipeline.

Public entrypoint: :func:`run_pipeline`. Composes ``demux`` → audio
backend → guitar / fretboard / hand backends → :func:`fuse` into a
single call returning ``list[TabEvent]``. The video stack runs in a
single forward pass over frames (subsampled by ``video_stride``); audio
is processed in parallel from the same demux result.

The video stack is **opt-in** (``video_enabled=False`` by default since
2026-07-28): the ungated legacy chain measured −0.15 to −0.20 aggregate
Tab F1 against the audio-only decode, its gated form is a measured
no-op, and every published Tab F1 figure is audio-only. See
DECISIONS.md 2026-07-28.

Graceful degradation (once video is enabled):

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
import math
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from tabvision.demux import demux
from tabvision.errors import BackendError
from tabvision.fusion import TimedNeckAnchor, apply_neck_anchor_priors, fuse
from tabvision.fusion.contact_prior import apply_contact_priors
from tabvision.fusion.inference_policy import ResolvedInferencePolicy, resolve_inference_policy
from tabvision.fusion.melodic_prior import apply_melodic_segment_prior
from tabvision.fusion.neck_prior import NeckAnchorLike
from tabvision.fusion.playability import set_transition_prior
from tabvision.fusion.position_prior import (
    apply_pitch_position_prior,
    capo_covariant_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.position_window_prior import apply_position_window_priors
from tabvision.fusion.transition_prior import load_transition_prior
from tabvision.fusion.viterbi import assignment_decoder_context
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
from tabvision.video.position import (
    ContactAwarePositionAnalyzer,
    FingerContactObservation,
    PositionAnalyzer,
    PositionWindowObservation,
    SessionCapoObservation,
    supports_contacts,
)

if TYPE_CHECKING:
    from tabvision.audio.filters import AudioFilterConfig

logger = logging.getLogger(__name__)


class _VideoImportError(RuntimeError):
    """Internal signal: a soft-optional video dep failed to import."""


SEQUENCE_PRIOR_DEFAULT = "guitarset-seq-v1"
"""Artifact the ``"auto"`` sequence-prior setting resolves to (A15)."""

SEQUENCE_PRIOR_WEIGHT = 4.0
"""Gate-accepted weight for the coupled sequence-prior default — the
val24 real-audio + 60-clip confirm runs both measured w=4.0
(DECISIONS.md 2026-07-02)."""

_SEQUENCE_DECODE_LOCK = threading.RLock()
"""Guards the legacy process-global transition prior during one decode."""


def _install_sequence_prior(
    sequence_prior: str | None,
    *,
    position_prior_active: bool | None = None,
) -> None:
    """Install (or clear) the learned fingering-sequence prior (A15).

    ``"auto"`` couples the prior to the position prior: active iff the
    position prior is. The coupling is load-bearing — uncoupled
    (position prior off) the sequence prior is a banked GAPS regression
    (0.647→0.593), while under the guitarset-v1 config it passes the
    val24 + 60-clip gates (DECISIONS.md 2026-07-02). An explicit
    ``TABVISION_TRANSITION_PRIOR`` env var wins over the argument and
    keeps its own weight semantics (sweep knob — see
    :mod:`tabvision.fusion.playability`).
    """
    if os.environ.get("TABVISION_TRANSITION_PRIOR", "").strip():
        return  # env-driven install path; playability's lazy loader owns it
    resolved = sequence_prior or "none"
    if resolved == "auto":
        resolved = SEQUENCE_PRIOR_DEFAULT if position_prior_active else "none"
    if resolved == "none":
        # Explicit clear: a prior installed by an earlier run_pipeline call
        # in this process must not leak into a prior-less run (the banked
        # uncoupled regression).
        set_transition_prior(None)
        return
    weight = None if "TABVISION_TRANSITION_PRIOR_WEIGHT" in os.environ else SEQUENCE_PRIOR_WEIGHT
    set_transition_prior(load_transition_prior(resolved), weight=weight)
    logger.info("installed fingering-sequence prior %s", resolved)


@contextmanager
def sequence_decode_context(sequence_prior: str | None) -> Iterator[None]:
    """Serialize a decode that depends on the process-global sequence prior.

    Any decode outside :func:`run_pipeline_with_artifacts` that must mirror
    production costs (e.g. the assisted-review candidate ranking) has to see
    the same installed transition prior without racing a concurrent pipeline
    run. This acquires the same lock and installs ``sequence_prior`` (a
    resolved policy name or ``"none"``) for the duration of the block.
    """
    with _SEQUENCE_DECODE_LOCK:
        _install_sequence_prior(sequence_prior)
        yield


@dataclass(frozen=True)
class _VideoStackResult:
    fingerings: list[FrameFingering]
    neck_anchors: list[TimedNeckAnchor]


@dataclass(frozen=True)
class PipelineArtifacts:
    """Additive detailed result for API metadata and evaluation tooling."""

    tab_events: tuple[TabEvent, ...]
    audio_events: tuple[AudioEvent, ...]
    policy: ResolvedInferencePolicy
    # Additive (2026-07-20): the backend that actually ran after "auto"
    # routing (e.g. "highres-ensemble"). Empty string on legacy constructors.
    resolved_audio_backend: str = ""
    # Additive FretCam bridge diagnostics. ``"none"`` means video was disabled
    # or its soft-optional legacy stack was unavailable.
    resolved_video_backend: str = "none"
    position_observation_count: int = 0
    notes_affected_by_video: int = 0
    # Additive (2026-07-26): whole-session capo estimate from video, when the
    # FretCam route ran. Reporting only — nothing in this pipeline routes on it,
    # because its field accuracy on real capo footage is unmeasured. Cross-check
    # it with `tabvision.preflight.capo.detect_capo_from_video` before showing a
    # number to a user; that refutes estimates the audio bound rules out.
    video_capo: SessionCapoObservation | None = None
    # Additive (2026-08-02): the stabilized window observations themselves, for
    # the opt-in personal-label harvest (SPEC §1.5 carve-out). Empty whenever
    # the FretCam route did not run; the count field above stays for callers
    # that only report.
    position_observations: tuple[PositionWindowObservation, ...] = ()


def run_pipeline_with_artifacts(
    video_path: str | Path,
    *,
    audio_backend: AudioBackend | None = None,
    audio_backend_name: str = "highres",
    guitar_backend: GuitarBackend | None = None,
    fretboard_backend: FretboardBackend | None = None,
    hand_backend: HandBackend | None = None,
    position_analyzer: PositionAnalyzer | None = None,
    lambda_vision: float = 1.0,
    video_stride: int = 3,
    video_enabled: bool = False,
    video_backend: str = "legacy",
    contact_evidence: bool = False,
    position_prior: str | None = "auto",
    sequence_prior: str | None = "auto",
    string_evidence: str | None = "auto",
    assignment_decoder: str | None = None,
    melodic_prior_enabled: bool = False,
    audio_filters: bool | AudioFilterConfig | None = None,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineArtifacts:
    """Run the full transcription pipeline on ``video_path``.

    Backends are injectable for testing; when omitted, the function
    constructs production backends on demand.

    ``video_enabled`` defaults to ``False`` (audio-only): the default
    decode then matches the published eval convention exactly — with no
    fingerings and no anchors the vision term in
    :func:`tabvision.fusion.playability.emission_cost` never fires, so
    ``lambda_vision`` is inert. Pass ``video_enabled=True`` to opt in to
    a video backend (DECISIONS.md 2026-07-28).

    ``audio_filters`` controls the constructed backend's post-detection
    filtering (``tabvision.audio.filters``): ``None`` (default) keeps each
    backend's built-in default — basicpitch on, highres off — so the default
    path is unchanged; ``True``/``False`` or an explicit ``AudioFilterConfig``
    forces it. Ignored when ``audio_backend`` is supplied (the injected
    backend carries its own filter config).

    ``sequence_prior`` selects the learned fingering-sequence prior (A15):
    ``"auto"`` (default) activates it iff ``position_prior`` is active —
    see :func:`_install_sequence_prior` for why the coupling is mandatory;
    ``"none"`` disables it; an artifact name forces it on.

    ``contact_evidence`` (default ``False``, experimental) additionally applies
    FretCam's per-finger ``(string, fret)`` contacts as a capped fusion prior.
    It requires ``video_backend="fretcam"`` and an analyzer that can return
    contacts from the same traversal. Off by default: the channel is measured
    but not gated — see
    ``docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md``.

    ``progress_callback``, when given, is invoked with a stage name as each
    pipeline stage *starts*: ``demux`` → ``model_load`` → ``audio_inference``
    → ``video_analysis`` (only when the video stack runs) → ``decode``.
    Rendering happens outside this function, so callers wanting a render
    stage emit it themselves after this returns. Callback exceptions are
    logged and swallowed — progress reporting must never break a
    transcription. Default ``None`` is a strict no-op.

    ``video_backend="fretcam"`` runs the stabilized FretCam position solver
    over the demuxer's media timestamps and contributes only its coarse,
    bounded fret-window evidence. It deliberately does not also run the
    legacy per-string hand posterior, which would double-count visual
    evidence. ``"legacy"`` remains the rollback/default until the controlled
    live acceptance gate is measured.
    """
    cfg = cfg or GuitarConfig()
    session = session or SessionConfig()
    if video_backend not in {"legacy", "fretcam"}:
        raise ValueError(f"video_backend must be 'legacy' or 'fretcam', got {video_backend!r}")
    if position_analyzer is not None and video_backend != "fretcam":
        raise ValueError("position_analyzer requires video_backend='fretcam'")
    if contact_evidence and video_backend != "fretcam":
        raise ValueError("contact_evidence requires video_backend='fretcam'")
    if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
        raise ValueError(f"video_stride must be a positive integer, got {video_stride!r}")
    if isinstance(lambda_vision, bool):
        raise ValueError(f"lambda_vision must be finite and >= 0, got {lambda_vision!r}")
    try:
        lambda_vision = float(lambda_vision)
    except (TypeError, ValueError):
        raise ValueError(f"lambda_vision must be finite and >= 0, got {lambda_vision!r}") from None
    if not math.isfinite(lambda_vision) or lambda_vision < 0.0:
        raise ValueError(f"lambda_vision must be finite and >= 0, got {lambda_vision!r}")
    resolved_position_analyzer = position_analyzer

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
    if video_enabled and video_backend == "fretcam" and resolved_position_analyzer is None:
        resolved_position_analyzer = _make_fretcam_position_analyzer(cfg)
    if audio_backend is None and audio_backend_name == "auto":
        audio_backend_name = audio_backend_for_session(session)
    if audio_backend is not None:
        audio = audio_backend
    else:
        # audio_filters None ⇒ keep the backend's built-in default (basicpitch
        # on, highres off); a bool / AudioFilterConfig overrides it.
        filter_kwargs = {} if audio_filters is None else {"filter_config": audio_filters}
        audio = _make_audio_backend(audio_backend_name, **filter_kwargs)
    resolved_audio_backend_name = str(getattr(audio, "name", audio_backend_name))
    policy = resolve_inference_policy(
        requested_position_prior=position_prior,
        requested_sequence_prior=sequence_prior,
        requested_string_evidence=string_evidence,
        requested_assignment_decoder=assignment_decoder,
        cfg=cfg,
        session=session,
        audio_backend_name=resolved_audio_backend_name,
    )
    # Backends load checkpoints lazily on first transcribe, so the load time
    # lands in the audio_inference stage; both map to "analyzing audio" UX-side.
    _notify("audio_inference")
    logger.info("transcribing audio with %s", audio.name)
    audio_events = audio.transcribe(demuxed.wav, demuxed.sample_rate, session)
    logger.info("audio backend produced %d events", len(audio_events))

    # Inharmonicity evidence runs *before* the corpus prior so the two combine
    # as a product of experts inside apply_pitch_position_prior — the same
    # order the gates measured. The channel abstains on any note it cannot
    # measure, so this is a no-op wherever the physics is unreadable.
    if policy.resolved_string_evidence != "none":
        from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
        from tabvision.fusion.string_physics import load_string_evidence

        evidence = load_string_evidence(policy.resolved_string_evidence)
        audio_events, tally = attach_inharmonicity_evidence(
            audio_events,
            demuxed.wav,
            demuxed.sample_rate,
            evidence.model,
            cfg,
            weight=evidence.weight,
            min_r2=evidence.min_r2,
            sigma=evidence.sigma,
            isolation=evidence.isolation,
        )
        logger.info(
            "attached string evidence %s to %d/%d events",
            policy.resolved_string_evidence,
            tally["applied"],
            tally["events"],
        )

    if policy.resolved_position_prior != "none":
        prior = load_pitch_position_prior(policy.resolved_position_prior, cfg=cfg)
        if cfg.capo > 0:
            # A capo shifts the fretboard uniformly, so the capo-0 prior
            # applies once both axes are re-indexed. Without this the prior
            # would be read at absolute coordinates — the "naive" arm, which
            # Q7 measured as no better than no prior at all by capo 4.
            prior = capo_covariant_prior(prior, cfg.capo)
        audio_events = apply_pitch_position_prior(audio_events, prior, cfg)
        logger.info(
            "attached pitch-position prior %s%s",
            policy.resolved_position_prior,
            f" (capo-covariant, capo {cfg.capo})" if cfg.capo > 0 else "",
        )

    if melodic_prior_enabled:
        audio_events = apply_melodic_segment_prior(audio_events, cfg)
        logger.info("attached melodic segment prior")

    fingerings: list[FrameFingering] = []
    neck_anchors: list[TimedNeckAnchor] = []
    position_observations: list[PositionWindowObservation] = []
    contact_observations: list[FingerContactObservation] = []
    video_capo: SessionCapoObservation | None = None
    resolved_video_backend = "none"
    if video_enabled:
        _notify("video_analysis")
        try:
            if video_backend == "fretcam":
                if resolved_position_analyzer is None:  # pragma: no cover - guarded above
                    raise AssertionError("FretCam analyzer was not resolved")
                if contact_evidence and supports_contacts(resolved_position_analyzer):
                    # One traversal, both evidence types: frame decode and
                    # inference dominate the cost, so contacts must not cost a
                    # second pass.
                    contact_analyzer = cast(
                        ContactAwarePositionAnalyzer, resolved_position_analyzer
                    )
                    bundle = contact_analyzer.analyze_all(
                        demuxed.frame_iterator,
                        stride=video_stride,
                    )
                    position_observations = list(bundle.windows)
                    contact_observations = list(bundle.contacts)
                    video_capo = bundle.capo
                else:
                    position_observations = list(
                        resolved_position_analyzer.analyze(
                            demuxed.frame_iterator,
                            stride=video_stride,
                        )
                    )
                resolved_video_backend = "fretcam"
            else:
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
                    resolved_video_backend = "legacy"
                except _VideoImportError as exc:
                    logger.warning(
                        "video stack unavailable, falling back to audio-only: %s",
                        exc,
                    )
        finally:
            _close_frame_iterator(demuxed.frame_iterator)

    if lambda_vision > 0.0 and neck_anchors:
        audio_events = apply_neck_anchor_priors(audio_events, neck_anchors, cfg)
        logger.info("attached %d hand-neck anchors as audio fret priors", len(neck_anchors))

    notes_affected_by_video = 0
    if lambda_vision > 0.0 and position_observations:
        enriched_audio_events = apply_position_window_priors(
            audio_events,
            position_observations,
            cfg,
            vision_weight=lambda_vision,
        )
        notes_affected_by_video = sum(
            before is not after
            for before, after in zip(audio_events, enriched_audio_events, strict=True)
        )
        audio_events = enriched_audio_events
        logger.info(
            "attached %d stabilized FretCam observations to %d ambiguous audio events",
            len(position_observations),
            notes_affected_by_video,
        )

    if lambda_vision > 0.0 and contact_observations:
        enriched_audio_events = apply_contact_priors(
            audio_events,
            contact_observations,
            cfg,
            vision_weight=lambda_vision,
        )
        notes_affected_by_contacts = sum(
            before is not after
            for before, after in zip(audio_events, enriched_audio_events, strict=True)
        )
        notes_affected_by_video += notes_affected_by_contacts
        audio_events = enriched_audio_events
        logger.info(
            "attached %d FretCam contact observations to %d ambiguous audio events",
            len(contact_observations),
            notes_affected_by_contacts,
        )

    _notify("decode")
    logger.info(
        "running fuse() with %d audio events, %d fingerings, lambda_vision=%.2f",
        len(audio_events),
        len(fingerings),
        lambda_vision,
    )
    # ``playability`` predates the server and exposes the transition prior as
    # process-global state. Resolve/load work can happen concurrently, but the
    # install + decode pair must be atomic so an acoustic and classical job
    # cannot observe each other's policy.
    with _SEQUENCE_DECODE_LOCK:
        _install_sequence_prior(policy.resolved_sequence_prior)
        with assignment_decoder_context(policy.resolved_assignment_decoder):
            tab_events = tuple(
                fuse(audio_events, fingerings, cfg, session, lambda_vision=lambda_vision)
            )
    return PipelineArtifacts(
        tab_events=tab_events,
        audio_events=tuple(audio_events),
        policy=policy,
        resolved_audio_backend=resolved_audio_backend_name,
        resolved_video_backend=resolved_video_backend,
        position_observation_count=len(position_observations),
        notes_affected_by_video=notes_affected_by_video,
        video_capo=video_capo,
        position_observations=tuple(position_observations),
    )


def run_pipeline(
    video_path: str | Path,
    *,
    audio_backend: AudioBackend | None = None,
    audio_backend_name: str = "highres",
    guitar_backend: GuitarBackend | None = None,
    fretboard_backend: FretboardBackend | None = None,
    hand_backend: HandBackend | None = None,
    position_analyzer: PositionAnalyzer | None = None,
    lambda_vision: float = 1.0,
    video_stride: int = 3,
    video_enabled: bool = False,
    video_backend: str = "legacy",
    position_prior: str | None = "auto",
    sequence_prior: str | None = "auto",
    string_evidence: str | None = "auto",
    assignment_decoder: str | None = None,
    melodic_prior_enabled: bool = False,
    audio_filters: bool | AudioFilterConfig | None = None,
    cfg: GuitarConfig | None = None,
    session: SessionConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[TabEvent]:
    """Run the pipeline and preserve the original list-returning contract."""

    result = run_pipeline_with_artifacts(
        video_path,
        audio_backend=audio_backend,
        audio_backend_name=audio_backend_name,
        guitar_backend=guitar_backend,
        fretboard_backend=fretboard_backend,
        hand_backend=hand_backend,
        position_analyzer=position_analyzer,
        lambda_vision=lambda_vision,
        video_stride=video_stride,
        video_enabled=video_enabled,
        video_backend=video_backend,
        position_prior=position_prior,
        sequence_prior=sequence_prior,
        string_evidence=string_evidence,
        assignment_decoder=assignment_decoder,
        melodic_prior_enabled=melodic_prior_enabled,
        audio_filters=audio_filters,
        cfg=cfg,
        session=session,
        progress_callback=progress_callback,
    )
    return list(result.tab_events)


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


def _close_frame_iterator(frames: object) -> None:
    """Release a demux-owned iterator even when a video analyzer returns early."""
    close = getattr(frames, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask decode results
        logger.debug("frame iterator cleanup failed: %s", exc)


# ---------------------------------------------------------------------------
# Backend factories — deferred imports so audio-only callers don't pay the
# vision-extras cost.
# ---------------------------------------------------------------------------


def audio_backend_for_session(session: SessionConfig) -> str:
    """Audio backend for a session's declared instrument — the user-facing toggle.

    Electric → the separately fine-tuned electric checkpoint (``highres-electric``);
    clean acoustic → the registered GAPS+FL ``highres-ensemble`` (Phase 3 gate
    passed; promoted to the default by the 2026-07-20 personal-use decision —
    player-05 aggregate +0.0213, onset/pitch improve, ~2× audio inference time);
    classical / distorted-acoustic → the single-checkpoint ``highres``. The
    ensemble backend itself deterministically rolls back to the single GAPS
    checkpoint outside clean acoustic, so this routing is belt-and-suspenders.
    Used when ``run_pipeline`` is called with ``audio_backend_name="auto"``.
    """
    if session.instrument == "electric":
        return "highres-electric"
    if session.instrument == "acoustic" and session.tone == "clean":
        return "highres-ensemble"
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


def _make_fretcam_position_analyzer(cfg: GuitarConfig) -> PositionAnalyzer:
    """Construct the separately packaged FretCam batch adapter on demand."""
    try:
        from fretcam.tabvision_adapter import FretCamPositionAnalyzer
    except ModuleNotFoundError as exc:
        if exc.name == "fretcam":
            raise BackendError(
                "FretCam video backend is not installed. Install the repository's "
                "fretcam package (from tabvision/: pip install -e ../fretcam) or "
                "select --video-backend legacy."
            ) from exc
        if exc.name == "fretcam.tabvision_adapter":
            raise BackendError(
                "The installed FretCam package does not include the TabVision "
                "adapter. Reinstall this repository's sibling fretcam package."
            ) from exc
        raise BackendError(
            f"FretCam dependency {exc.name!r} could not be imported: {exc}. "
            "Install TabVision's vision extra and run "
            "`python -m scripts.acquire.models status`."
        ) from exc
    except ImportError as exc:
        raise BackendError(
            f"FretCam adapter import failed: {exc}. Reinstall the sibling "
            "fretcam package or select --video-backend legacy."
        ) from exc
    return FretCamPositionAnalyzer(cfg)


# Re-export AudioEvent / TabEvent for ergonomic ``from tabvision.pipeline import TabEvent``.
__all__ = [
    "PipelineArtifacts",
    "run_pipeline",
    "run_pipeline_with_artifacts",
    "AudioEvent",
    "TabEvent",
]
