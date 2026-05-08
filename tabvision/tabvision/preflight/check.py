"""Preflight check — see SPEC.md §7 Phase 3.

Runs a short pass (default 5 seconds) over the head of a video, reports
whether the framing / lighting / stability are good enough to attempt a
full transcription. Bad framing here is much cheaper to surface than
after a 60-second take has already been recorded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tabvision.errors import InvalidInputError
from tabvision.types import GuitarBBox, PreflightFinding, PreflightReport

logger = logging.getLogger(__name__)

# Thresholds. Tuned against typical iPhone-on-lap shots; revisit at Phase 3
# acceptance with the labeled good/bad framing set.
DEFAULT_PREVIEW_SECONDS = 5.0
DEFAULT_MIN_GUITAR_DETECT_RATE = 0.6  # ≥60% of preview frames must have a bbox
DEFAULT_BBOX_AREA_FRAC_RANGE = (0.05, 0.85)  # bbox area / frame area
DEFAULT_BBOX_CENTER_FRAC_RANGE = (0.15, 0.85)  # cx, cy normalized
DEFAULT_BBOX_STABILITY_FRAC = 0.05  # center std / frame width
DEFAULT_LUMA_RANGE = (40.0, 220.0)  # mean luma 0–255
DEFAULT_BBOX_ASPECT_RANGE = (1.5, 25.0)  # match v0 fretboard plausibility


def check(video_path: str | Path, *, strict: bool = False) -> PreflightReport:
    """Run preflight on the first few seconds of a video.

    Args:
        video_path: input mp4/mov.
        strict: if True, soft "warn" findings escalate to a failed report.

    Returns:
        PreflightReport with findings + actionable suggestions.
    """
    path = Path(video_path)
    if not path.exists():
        raise InvalidInputError(f"video file not found: {path}")

    frames, _fps = _sample_preview_frames(path, DEFAULT_PREVIEW_SECONDS)
    if not frames:
        return _fail_report(
            "EMPTY_VIDEO",
            "could not sample any frames from the preview window",
            ["Re-record the clip; the file may be empty or corrupt."],
        )

    detections, detector_available = _detect_guitar_in_frames(frames)
    findings: list[PreflightFinding] = []
    suggestions: list[str] = []

    if detector_available:
        _check_guitar_visibility(detections, findings, suggestions)
        _check_framing_position(frames, detections, findings, suggestions)
        _check_bbox_stability(frames, detections, findings, suggestions)
        _check_aspect_ratio(detections, findings, suggestions)
    else:
        findings.append(
            PreflightFinding(
                "info",
                "DETECTOR_UNAVAILABLE",
                "guitar detector not loaded — skipping framing checks "
                "(install '.[vision]' and train weights for full preflight)",
            )
        )

    # Lighting check works without the detector.
    _check_lighting(frames, findings, suggestions)

    has_fail = any(f.severity == "fail" for f in findings)
    has_warn = any(f.severity == "warn" for f in findings)
    passed = not has_fail and (not strict or not has_warn)

    return PreflightReport(
        passed=passed,
        findings=findings,
        suggested_actions=suggestions,
    )


# ----- frame sampling -----


def _sample_preview_frames(
    path: Path, max_seconds: float
) -> tuple[list[np.ndarray], float]:
    """Decode the first ``max_seconds`` of video into a list of BGR frames."""
    try:
        import cv2
    except ImportError as exc:
        raise InvalidInputError(
            "preflight requires opencv-python. Install with: pip install '.[vision]'"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_max = int(round(fps * max_seconds))
    # Sample at most 30 frames across the preview window (1 every ~6 frames at
    # 30 fps × 5 s); enough to estimate stability without paying full decode.
    n_sample = min(30, n_max)
    if n_sample <= 0:
        cap.release()
        return [], fps

    sample_indices = set(
        int(round(i)) for i in np.linspace(0, n_max - 1, n_sample, dtype=float)
    )
    frames: list[np.ndarray] = []
    idx = 0
    while idx < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in sample_indices:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames, fps


# ----- detector pass -----


def _detect_guitar_in_frames(
    frames: list[np.ndarray],
) -> tuple[list[GuitarBBox | None], bool]:
    """Run the YOLO-OBB backend on each preview frame.

    Returns ``(detections, detector_available)``. If the backend cannot
    load (no checkpoint, ultralytics not installed) we return all-None
    detections plus ``detector_available=False`` so the caller can emit a
    degraded-mode INFO finding instead of a hard GUITAR_NOT_DETECTED fail.
    """
    try:
        from tabvision.video.guitar.yolo_backend import YoloOBBBackend

        backend = YoloOBBBackend()
        # Probe the first frame to surface load errors early.
        try:
            _ = backend.detect(frames[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning("guitar detector unavailable; preflight in degraded mode: %s", exc)
            return [None] * len(frames), False
        detections: list[GuitarBBox | None] = []
        for f in frames:
            try:
                detections.append(backend.detect(f))
            except Exception as exc:  # noqa: BLE001
                logger.debug("detector raised on frame: %s", exc)
                detections.append(None)
        return detections, True
    except Exception as exc:  # noqa: BLE001 — module import or constructor failure
        logger.warning("guitar detector unavailable; preflight in degraded mode: %s", exc)
        return [None] * len(frames), False


# ----- individual checks -----


def _check_guitar_visibility(
    detections: list[GuitarBBox | None],
    findings: list[PreflightFinding],
    suggestions: list[str],
) -> None:
    n = len(detections)
    if n == 0:
        return
    rate = sum(1 for d in detections if d is not None) / n
    if rate >= DEFAULT_MIN_GUITAR_DETECT_RATE:
        findings.append(
            PreflightFinding("info", "GUITAR_VISIBLE", f"guitar detected in {rate:.0%} of preview frames")
        )
        return
    if rate == 0.0:
        findings.append(
            PreflightFinding(
                "fail",
                "GUITAR_NOT_DETECTED",
                "no guitar found in any preview frame",
            )
        )
        suggestions.append("Move the guitar fully into frame; check that it isn't off-screen or occluded.")
    else:
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_INTERMITTENT",
                f"guitar only detected in {rate:.0%} of preview frames",
            )
        )
        suggestions.append("Hold the guitar steadier; partial occlusion or motion is reducing detection reliability.")


def _check_framing_position(
    frames: list[np.ndarray],
    detections: list[GuitarBBox | None],
    findings: list[PreflightFinding],
    suggestions: list[str],
) -> None:
    boxes = [d for d in detections if d is not None]
    if not boxes or not frames:
        return
    h, w = frames[0].shape[:2]
    cx = np.mean([b.x + b.w / 2 for b in boxes]) / w
    cy = np.mean([b.y + b.h / 2 for b in boxes]) / h
    area_frac = np.mean([b.w * b.h for b in boxes]) / (w * h)

    cmin, cmax = DEFAULT_BBOX_CENTER_FRAC_RANGE
    if not (cmin < cx < cmax) or not (cmin < cy < cmax):
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_OFF_CENTER",
                f"guitar center at ({cx:.2f}, {cy:.2f}) — outside the {cmin}–{cmax} range",
            )
        )
        if cx < cmin:
            suggestions.append("Move the guitar (or the camera) right.")
        elif cx > cmax:
            suggestions.append("Move the guitar (or the camera) left.")
        if cy < cmin:
            suggestions.append("Lower the guitar in frame (tilt phone up or move guitar down).")
        elif cy > cmax:
            suggestions.append("Raise the guitar in frame (tilt phone down or move guitar up).")

    amin, amax = DEFAULT_BBOX_AREA_FRAC_RANGE
    if area_frac < amin:
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_TOO_SMALL",
                f"guitar fills only {area_frac:.1%} of the frame",
            )
        )
        suggestions.append("Move the camera closer or zoom in so the guitar fills more of the frame.")
    elif area_frac > amax:
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_TOO_LARGE",
                f"guitar fills {area_frac:.0%} of the frame — likely cropping the neck",
            )
        )
        suggestions.append("Pull the camera back so the entire fretboard is visible.")


def _check_bbox_stability(
    frames: list[np.ndarray],
    detections: list[GuitarBBox | None],
    findings: list[PreflightFinding],
    suggestions: list[str],
) -> None:
    boxes = [d for d in detections if d is not None]
    if len(boxes) < 3 or not frames:
        return
    w = frames[0].shape[1]
    cx = np.array([b.x + b.w / 2 for b in boxes])
    cy = np.array([b.y + b.h / 2 for b in boxes])
    drift = float(np.sqrt(np.std(cx) ** 2 + np.std(cy) ** 2))
    drift_frac = drift / w
    if drift_frac > DEFAULT_BBOX_STABILITY_FRAC:
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_DRIFT",
                f"bbox center drift {drift:.0f} px ({drift_frac:.1%} of frame width)",
            )
        )
        suggestions.append("Stabilize the camera — drift across the take will hurt fretboard tracking.")


def _check_lighting(
    frames: list[np.ndarray],
    findings: list[PreflightFinding],
    suggestions: list[str],
) -> None:
    if not frames:
        return
    luma = np.mean([np.mean(f[..., 1]) for f in frames])  # rough luma proxy via G channel of BGR
    lo, hi = DEFAULT_LUMA_RANGE
    if luma < lo:
        findings.append(
            PreflightFinding("warn", "LIGHTING_DIM", f"mean luma {luma:.0f}/255 — likely too dark")
        )
        suggestions.append("Add light: an overhead lamp or open blinds typically rescues fretboard detection.")
    elif luma > hi:
        findings.append(
            PreflightFinding("warn", "LIGHTING_BRIGHT", f"mean luma {luma:.0f}/255 — frames may be over-exposed")
        )
        suggestions.append("Reduce light or move out of direct sun; clipped highlights hurt edge detection.")


def _check_aspect_ratio(
    detections: list[GuitarBBox | None],
    findings: list[PreflightFinding],
    suggestions: list[str],
) -> None:
    boxes = [d for d in detections if d is not None]
    if not boxes:
        return
    aspects = [(b.w / b.h) if b.h > 0 else 0.0 for b in boxes]
    mean_aspect = float(np.mean(aspects))
    lo, hi = DEFAULT_BBOX_ASPECT_RANGE
    if not (lo <= mean_aspect <= hi):
        findings.append(
            PreflightFinding(
                "warn",
                "GUITAR_ANGLE_IMPLAUSIBLE",
                f"mean bbox aspect {mean_aspect:.2f} — fretboard angle may be too oblique",
            )
        )
        suggestions.append("Tilt the phone so the fretboard runs roughly horizontally across the frame.")


# ----- error helpers -----


def _fail_report(code: str, message: str, suggestions: list[str]) -> PreflightReport:
    return PreflightReport(
        passed=False,
        findings=[PreflightFinding("fail", code, message)],
        suggested_actions=suggestions,
    )


__all__ = ["check"]
