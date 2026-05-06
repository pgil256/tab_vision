"""Phase 4 acceptance harness — fingertip → (string, fret) accuracy.

Driven by labels collected with ``scripts.annotate.label_clips
fingering``. Skips when fewer than ``HAND_MIN_FRAMES`` labeled frames
are available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.annotate import storage

# SPEC §7 Phase 4 acceptance: top-1 fingertip position ≥ 0.75 accuracy
# across 100 hand-labeled frames.
HAND_MIN_FRAMES = 100
HAND_MIN_TOP1_ACCURACY = 0.75


@pytest.mark.hand_eval
def test_fingertip_top1_accuracy_above_threshold():
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")  # not installable on numpy<2 envs

    from scripts.annotate.frames import probe_clip, read_frame
    from tabvision.types import GuitarBBox, GuitarConfig
    from tabvision.video.fretboard.keypoint import KeypointFretboardBackend
    from tabvision.video.hand.fingertip_to_fret import FRETTING_FINGERS
    from tabvision.video.hand.mediapipe_backend import MediaPipeHandBackend

    labeled_ids = storage.list_labeled_clips("fingering")

    # Aggregate over all clips' frames.
    per_finger_correct = 0
    per_finger_total = 0
    cfg = GuitarConfig(max_fret=12)
    keypoint = KeypointFretboardBackend()
    hand = MediaPipeHandBackend()

    for cid in labeled_ids:
        label = storage.load_fingering(_clip_path_from_id(cid))
        if label is None:
            continue
        clip = Path(label.clip_path)
        if not clip.exists():
            continue
        meta = probe_clip(clip)
        bbox = GuitarBBox(0.0, 0.0, float(meta.width), float(meta.height), 1.0)
        for frame_label in label.frames:
            frame = read_frame(clip, frame_label.frame_idx)
            homog = keypoint.detect(frame, bbox)
            ff = hand.detect(frame, homog, cfg)
            for fl in frame_label.fingers:
                if not fl.is_fretting:
                    continue
                fi = FRETTING_FINGERS.index(fl.finger)
                logits = ff.finger_pos_logits[fi]
                s_pred, f_pred = np.unravel_index(int(logits.argmax()), logits.shape)
                # Convert v1 0-indexed-from-low-E to user's 1-indexed-from-high-E
                # convention used in the labeler (string=1 = high E).
                s_pred_user = cfg.n_strings - int(s_pred)  # 0->n, 5->1
                per_finger_total += 1
                if (s_pred_user == fl.string) and (int(f_pred) == fl.fret):
                    per_finger_correct += 1

    if per_finger_total < HAND_MIN_FRAMES:
        pytest.skip(
            f"need ≥ {HAND_MIN_FRAMES} fretting-finger labels, have "
            f"{per_finger_total}. Label more frames with: "
            f"python -m scripts.annotate.label_clips --clips ..."
        )

    accuracy = per_finger_correct / per_finger_total
    assert accuracy >= HAND_MIN_TOP1_ACCURACY, (
        f"fingertip top-1 accuracy {accuracy:.3f} < {HAND_MIN_TOP1_ACCURACY} "
        f"({per_finger_correct}/{per_finger_total})"
    )


def _clip_path_from_id(cid: str) -> str:
    """Look up the clip_path inside a saved fingering label."""
    p = storage.default_eval_root() / "fingering" / f"{cid}.json"
    if not p.exists():
        return ""
    import json
    return json.loads(p.read_text()).get("clip_path", "")
