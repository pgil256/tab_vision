"""Phase 3 acceptance harnesses — preflight + keypoint fretboard.

Driven by labels collected with the ``scripts.annotate.label_clips``
tool. Each test skips when its fixtures dir is empty so a fresh clone
doesn't fail CI; once you've labeled at least the minimum number of
clips per the SPEC §7 Phase 3 acceptance criteria, the tests exercise
the live pipeline against the labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.annotate import storage

# Acceptance constants from SPEC §7 Phase 3.
PREFLIGHT_MIN_CLIPS = 10
PREFLIGHT_REQUIRED_CORRECT_FRACTION = 0.9  # ≥ 9/10

FRETBOARD_MIN_CLIPS = 5
FRETBOARD_MAX_PIXEL_ERROR = 5.0  # median across-frame error ≤ 5 px


# ----- preflight -----


@pytest.mark.preflight_eval
def test_preflight_classifies_good_vs_bad_framing(tmp_path):
    """SPEC §7 Phase 3 acceptance: ≥ 9/10 correct on labeled good/bad
    framing clips. Skips when fewer than ``PREFLIGHT_MIN_CLIPS`` are
    available; reports an honest fail otherwise."""
    pytest.importorskip("cv2")
    from tabvision.preflight.check import check

    labeled_ids = storage.list_labeled_clips("framing")
    if len(labeled_ids) < PREFLIGHT_MIN_CLIPS:
        pytest.skip(
            f"need ≥ {PREFLIGHT_MIN_CLIPS} framing labels, have {len(labeled_ids)}. "
            f"Label more with: python -m scripts.annotate.label_clips --clips ..."
        )

    correct = 0
    total = 0
    failures: list[str] = []
    for cid in labeled_ids:
        label = storage.load_framing(_clip_path_from_id(cid, "framing"))
        if label is None:
            continue
        clip = Path(label.clip_path)
        if not clip.exists():
            failures.append(f"{cid}: clip file missing at {clip}")
            continue
        try:
            report = check(str(clip))
        except Exception as exc:  # noqa: BLE001 — surface anything as a failure
            failures.append(f"{cid}: preflight raised {exc}")
            continue
        predicted = "good" if report.passed else "bad"
        total += 1
        if predicted == label.label:
            correct += 1
        else:
            failures.append(f"{cid}: predicted {predicted}, labeled {label.label}")

    assert total >= PREFLIGHT_MIN_CLIPS, f"only {total} usable labels"
    accuracy = correct / total
    assert accuracy >= PREFLIGHT_REQUIRED_CORRECT_FRACTION, (
        f"preflight accuracy {accuracy:.2f} < {PREFLIGHT_REQUIRED_CORRECT_FRACTION}; "
        f"failures: {failures}"
    )


# ----- keypoint fretboard -----


@pytest.mark.fretboard_eval
def test_keypoint_fretboard_pixel_error_within_5px():
    """SPEC §7 Phase 3 acceptance: median per-frame homography error within
    5 px of hand-clicked GT on 4 reference fret intersections, measured on
    ≥ 5 user eval clips."""
    pytest.importorskip("cv2")

    labeled_ids = storage.list_labeled_clips("fretboard")
    if len(labeled_ids) < FRETBOARD_MIN_CLIPS:
        pytest.skip(
            f"need ≥ {FRETBOARD_MIN_CLIPS} fretboard labels, have {len(labeled_ids)}. "
            f"Label more with: python -m scripts.annotate.label_clips --clips ..."
        )

    from scripts.annotate.frames import probe_clip, read_frame
    from tabvision.types import GuitarBBox
    from tabvision.video.fretboard.keypoint import KeypointFretboardBackend

    backend = KeypointFretboardBackend()
    per_clip_median: list[float] = []
    for cid in labeled_ids:
        label = storage.load_fretboard(_clip_path_from_id(cid, "fretboard"))
        if label is None or not label.is_complete():
            continue
        clip = Path(label.clip_path)
        if not clip.exists():
            continue
        meta = probe_clip(clip)
        frame = read_frame(clip, label.frame_idx)
        bbox = GuitarBBox(0.0, 0.0, float(meta.width), float(meta.height), 1.0)
        homog = backend.detect(frame, bbox)
        if homog.confidence == 0.0:
            per_clip_median.append(float("inf"))
            continue

        errors = []
        for p in label.points:
            # Predicted position of (p.fret, p.edge) projected via the
            # current backend's homography.  Canonical x for fret k uses
            # the spec's equal-tempered "x=1 corresponds to fret 12"
            # convention (matches the keypoint backend's drawing
            # convention; see scripts.viz.overlay_fretboard).
            canon_x = (1 - 1.0 / (2 ** (p.fret / 12.0))) / (1 - 1.0 / 2.0)
            canon_y = 0.0 if p.edge == "top" else 1.0
            proj = homog.H @ np.array([canon_x, canon_y, 1.0])
            px = float(proj[0] / proj[2])
            py = float(proj[1] / proj[2])
            errors.append(float(np.hypot(p.x - px, p.y - py)))
        per_clip_median.append(float(np.median(errors)))

    assert len(per_clip_median) >= FRETBOARD_MIN_CLIPS, (
        f"only {len(per_clip_median)} usable fretboard labels"
    )
    overall_median = float(np.median(per_clip_median))
    assert overall_median <= FRETBOARD_MAX_PIXEL_ERROR, (
        f"keypoint fretboard median pixel error {overall_median:.2f} > "
        f"{FRETBOARD_MAX_PIXEL_ERROR}; per-clip medians: {per_clip_median}"
    )


def _clip_path_from_id(cid: str, kind: str) -> str:
    """Recover the clip_path stored inside the label JSON.

    The on-disk filename is the clip_id (a slug); the actual clip path
    lives inside the JSON as ``clip_path``.  We need the path for things
    like ``cv2.VideoCapture`` lookups, so dispatch via the loader.
    """
    root = storage.default_eval_root() / kind
    payload_path = root / f"{cid}.json"
    if not payload_path.exists():
        return ""
    import json
    return json.loads(payload_path.read_text()).get("clip_path", "")
