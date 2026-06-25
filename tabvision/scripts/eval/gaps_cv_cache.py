"""Shared *light* cache layer for the GAPS video CV chain — v1.1 chunk-6 WS0.

Splits the numpy-only cache helpers out of the probe so geometry / orientation /
diagnostic code can run entirely from cached CV intermediates **without importing
cv2 / mediapipe / ultralytics**. The probe
(``scripts.eval.v1_1_gaps_video_chain_probe``) writes the rich cache during a CV
re-run; this module reads it back and reconstructs ``FrameFingering`` objects.

Why a *rich* (v2) cache. The chunk-5 probe cached only the final
``FrameFingering`` per frame — the string was already baked in by
``compute_fingering`` at detection time, so the *only* cache-only lever was the
four discrete orientation flips (which cannot undo GAPS's graded −1..−4 string
offset). The v2 cache instead persists the raw inputs ``compute_fingering`` needs
— the YOLO ``OBBPredictions`` (nut/fret/neck anchors), the fitted ``Homography``,
and the selected fretting ``HandSample`` — so per-clip board calibration,
orientation, and posterior changes (chunk-6 WS1/WS2/WS4) become re-runnable from
cache (seconds), not a full CV re-run (tens of minutes).

Cache files live under ``--cache-dir`` (default ``~/.tabvision/cache/gaps_video_chain``):

- ``{stem}.rawcv.c{conf:.2f}.pkl`` — v2 rich cache, ``dict[int, RawFrameCV | None]``.
- ``{stem}.frames.c{conf:.2f}.pkl`` — legacy chunk-5 cache, ``dict[int, FrameFingering | None]``
  (read-only fallback so the diagnostic still runs against pre-WS0 caches).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tabvision.types import FrameFingering, GuitarConfig, Homography
from tabvision.video.guitar.yolo_backend import OBBPredictions
from tabvision.video.hand.fingertip_to_fret import HandSample, PosteriorConfig, compute_fingering

# Bump when the RawFrameCV layout changes in a way that invalidates old pickles.
RAWCV_CACHE_VERSION = 2

# A per-clip geometry re-fit: given a frame's raw CV intermediates, return the
# homography to project through and an optional nonlinear ``fret_xs`` map (chunk-6
# WS1/WS2). Injected by the caller (the probe / diagnostic) so this module stays
# numpy-only (no cv2 / mediapipe / ultralytics) — the cv2-using re-fit lives in
# ``tabvision.video.fretboard``. ``None`` means "replay the cached chain".
CalibrateFn = Callable[["RawFrameCV"], "tuple[Homography, np.ndarray | None]"]


@dataclass(frozen=True)
class RawFrameCV:
    """Raw CV intermediates for one frame — everything ``compute_fingering`` needs.

    Persisting these (rather than only the final ``FrameFingering``) is the WS0
    enabler: the homography fit, the orientation, and the fingertip→string
    posterior can all be re-derived downstream from cache, so chunk-6 geometry
    experiments do not pay the MediaPipe/YOLO cost on every iteration.

    All three fields are plain dataclasses/ndarrays, so the record pickles
    cleanly. ``preds`` carries the per-clip calibration cues (nut OBB edge,
    ordered fret OBBs) that WS1 will exploit; ``homography`` + ``hand`` are the
    direct inputs to :func:`fingering_from_raw`.
    """

    preds: OBBPredictions
    homography: Homography
    hand: HandSample


def fingering_from_raw(
    rec: RawFrameCV | None,
    cfg: GuitarConfig,
    *,
    t: float,
    calibrate: CalibrateFn | None = None,
    posterior_cfg: PosteriorConfig | None = None,
) -> FrameFingering | None:
    """Reconstruct a frame's ``FrameFingering`` from its cached CV intermediates.

    With no hooks this is **bit-identical** to the chunk-5 chain, which stored
    ``replace(compute_fingering(hand, H, cfg), t=t)``, so cached baselines
    reproduce exactly. Returns ``None`` when the frame had no usable detection
    (mirrors the chunk-5 ``None`` sentinel), so consumers treat it the same as a
    frame that was never cached.

    Args:
        calibrate: optional per-clip geometry re-fit (chunk-6 WS1/WS2). When
            given, it maps ``rec`` to ``(homography, fret_xs)`` and the fingering
            is rebuilt with that homography + nonlinear fret map instead of the
            cached homography + uniform partition. This is the single seam both
            eval gates funnel through, so a re-fit shows up cache-only in each.
        posterior_cfg: optional ``PosteriorConfig`` override (else the default).
    """
    if rec is None:
        return None
    if calibrate is not None:
        homography, fret_xs = calibrate(rec)
    else:
        homography, fret_xs = rec.homography, None
    fingering = compute_fingering(rec.hand, homography, cfg, posterior_cfg, fret_xs=fret_xs)
    return replace(fingering, t=t)


def make_fret_xs_calibrator(cfg: GuitarConfig) -> CalibrateFn:
    """Build the chunk-6 WS1 calibrate hook for :func:`fingering_from_raw`.

    Keeps the cached homography and re-derives only the per-clip **nonlinear**
    (rule-of-18) fret map from the cached fret/nut OBBs. Lazily imports the
    (numpy-only) calibration so this module's top stays free of
    cv2/mediapipe/ultralytics. When the calibration can't be trusted for a frame
    (too few/garbled fret detections) it returns ``fret_xs=None`` and
    ``compute_fingering`` falls back to the uniform partition — i.e. exactly the
    pre-WS1 behaviour for that frame, preserving the no-regression invariant.
    """
    from tabvision.video.fretboard.calibrate import calibrate_fret_xs

    def _calibrate(rec: RawFrameCV) -> tuple[Homography, np.ndarray | None]:
        return rec.homography, calibrate_fret_xs(rec.preds, rec.homography, cfg)

    return _calibrate


def rawcv_cache_path(cache_dir: Path, stem: str, conf: float) -> Path:
    """Path of the v2 rich cache for ``stem`` at YOLO confidence ``conf``."""
    return cache_dir / f"{stem}.rawcv.c{conf:.2f}.pkl"


def legacy_frames_cache_path(cache_dir: Path, stem: str, conf: float) -> Path:
    """Path of the chunk-5 legacy ``FrameFingering`` cache for ``stem``."""
    return cache_dir / f"{stem}.frames.c{conf:.2f}.pkl"


def load_frame_fingerings(
    cache_dir: Path,
    stem: str,
    *,
    conf: float,
    cfg: GuitarConfig,
    fps: float,
    calibrate: CalibrateFn | None = None,
    posterior_cfg: PosteriorConfig | None = None,
) -> dict[int, FrameFingering | None]:
    """Per-frame ``FrameFingering``s for a clip, from cache (no CV re-run).

    Prefers the v2 rich cache — reconstructs each frame via
    :func:`fingering_from_raw`, so any later change to ``compute_fingering`` or
    the homography flows through automatically. Falls back to the legacy
    ``FrameFingering`` cache when the rich one is absent, so the diagnostic can
    run against pre-WS0 caches. The reconstructed timestamps (``t = fi / fps``)
    match the chunk-5 frame-iterator convention, so rich- and legacy-sourced
    fingerings are interchangeable downstream (fusion overrides ``t`` with the
    event onset anyway).

    ``calibrate`` / ``posterior_cfg`` are forwarded to :func:`fingering_from_raw`
    (chunk-6 WS1/WS2 re-fit hook); they apply only on the rich-cache path. The
    legacy ``FrameFingering`` cache has no raw intermediates to re-fit from, so a
    calibrated run requires the rich cache.

    Raises:
        FileNotFoundError: if neither cache exists for ``(stem, conf)``.
    """
    rich = rawcv_cache_path(cache_dir, stem, conf)
    if rich.exists():
        with open(rich, "rb") as fh:
            raw: dict[int, RawFrameCV | None] = pickle.load(fh)
        return {
            fi: fingering_from_raw(
                rec, cfg, t=fi / fps, calibrate=calibrate, posterior_cfg=posterior_cfg
            )
            for fi, rec in raw.items()
        }

    legacy = legacy_frames_cache_path(cache_dir, stem, conf)
    if legacy.exists():
        with open(legacy, "rb") as fh:
            legacy_cache: dict[int, FrameFingering | None] = pickle.load(fh)
        return legacy_cache

    raise FileNotFoundError(
        f"no CV cache for {stem!r} at conf={conf:.2f} under {cache_dir} "
        f"(looked for {rich.name} then {legacy.name}); build it by running "
        "scripts.eval.v1_1_gaps_video_chain_probe first."
    )


def needed_frames(
    onsets: list[float],
    offset_s: float,
    fps: float,
    *,
    window_s: float,
    max_frames: int,
) -> tuple[set[int], dict[int, list[int]]]:
    """Union of frame indices near each onset (+ per-onset nearest-frame lists).

    ``video_time = onset + offset_s``; for each onset the nearest ``max_frames``
    frame indices within ``±window_s`` are selected (and returned time-sorted).
    Pure (numpy only) — shared by the probe's frame extraction and the
    diagnostic so both group frames identically.
    """
    needed: set[int] = set()
    per_onset: dict[int, list[int]] = {}
    span = int(np.ceil(window_s * fps))
    for i, onset in enumerate(onsets):
        vt = onset + offset_s
        center = int(round(vt * fps))
        cands = [fi for fi in range(center - span, center + span + 1) if fi >= 0]
        cands.sort(key=lambda fi: abs(fi / fps - vt))
        cands = sorted(cands[:max_frames], key=lambda fi: fi / fps)
        per_onset[i] = cands
        needed.update(cands)
    return needed, per_onset


__all__ = [
    "RAWCV_CACHE_VERSION",
    "CalibrateFn",
    "RawFrameCV",
    "fingering_from_raw",
    "make_fret_xs_calibrator",
    "rawcv_cache_path",
    "legacy_frames_cache_path",
    "load_frame_fingerings",
    "needed_frames",
]
