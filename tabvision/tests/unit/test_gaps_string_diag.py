"""Unit tests for the chunk-6 WS0 rich CV cache + string-resolution diagnostic.

Covers the two cache-only pieces that let chunk-6 iterate without re-running the
CV models:

* :mod:`scripts.eval.gaps_cv_cache` — the rich (v2) cache reconstructs a frame's
  ``FrameFingering`` bit-identically to the chunk-5 chain
  (``replace(compute_fingering(hand, H, cfg), t=t)``), the loader prefers the rich
  cache and falls back to the legacy one, and ``needed_frames`` groups frames
  deterministically.
* :func:`scripts.eval.v1_1_gaps_string_diag.diagnose_clip_strings` — the leading
  indicator: ambiguous-only, evidence-gated, best-orientation string accuracy and
  ``pred − gold`` offset histograms.

All synthetic — no media, no MediaPipe/YOLO, no ffmpeg.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.eval.gaps_cv_cache import (
    RawFrameCV,
    fingering_from_raw,
    legacy_frames_cache_path,
    load_frame_fingerings,
    needed_frames,
    rawcv_cache_path,
)
from scripts.eval.v1_1_gaps_string_diag import diagnose_clip_strings
from tabvision.types import FrameFingering, GuitarConfig, Homography, TabEvent
from tabvision.video.guitar.yolo_backend import OBBPredictions
from tabvision.video.hand.fingertip_to_fret import (
    FingerSample,
    HandSample,
    compute_fingering,
)

CFG = GuitarConfig()


def _hand() -> HandSample:
    """A hand whose index fingertip projects onto the board under identity H."""
    return HandSample(
        wrist_xy=(0.5, 0.9),
        wrist_z=0.0,
        is_left_hand=True,
        confidence=1.0,
        fingers={"index": FingerSample(name="index", tip_xy=(0.3, 0.6), tip_z=0.0, curl_ratio=1.0)},
    )


def _raw_record() -> RawFrameCV:
    return RawFrameCV(
        preds=OBBPredictions(),
        homography=Homography(H=np.eye(3, dtype=np.float64), confidence=0.8, method="test"),
        hand=_hand(),
    )


def _peaked(t: float, string_idx: int, fret: int) -> FrameFingering:
    """A FrameFingering whose marginal peaks on ``(string_idx, fret)``."""
    logits = np.full((4, CFG.n_strings, CFG.max_fret + 1), -10.0)
    logits[0, string_idx, fret] = 5.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=1.0)


def _gold(onset_s: float, string_idx: int, fret: int, pitch_midi: int) -> TabEvent:
    return TabEvent(
        onset_s=onset_s,
        duration_s=0.3,
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch_midi,
        confidence=1.0,
    )


# --------------------------------------------------------------------------- #
# Rich cache reconstruction
# --------------------------------------------------------------------------- #
def test_fingering_from_raw_matches_compute_fingering() -> None:
    rec = _raw_record()
    expected = compute_fingering(rec.hand, rec.homography, CFG)

    got = fingering_from_raw(rec, CFG, t=1.25)

    assert got is not None
    assert got.t == 1.25  # the chunk-5 chain stamps the frame timestamp
    assert got.homography_confidence == expected.homography_confidence
    np.testing.assert_array_equal(got.finger_pos_logits, expected.finger_pos_logits)


def test_fingering_from_raw_survives_pickle_roundtrip() -> None:
    rec = _raw_record()
    expected = compute_fingering(rec.hand, rec.homography, CFG)

    restored = pickle.loads(pickle.dumps(rec))
    got = fingering_from_raw(restored, CFG, t=2.0)

    assert got is not None
    np.testing.assert_array_equal(got.finger_pos_logits, expected.finger_pos_logits)


def test_fingering_from_raw_none_passthrough() -> None:
    # ``None`` is the sentinel for "no usable detection" — preserved verbatim.
    assert fingering_from_raw(None, CFG, t=0.0) is None


# --------------------------------------------------------------------------- #
# Cache loader: rich preferred, legacy fallback
# --------------------------------------------------------------------------- #
def test_load_prefers_rich_cache_and_reconstructs(tmp_path) -> None:
    fps = 25.0
    rec = _raw_record()
    rich = {3: rec, 7: None}
    rawcv_cache_path(tmp_path, "clip", 0.25).write_bytes(pickle.dumps(rich))
    # A *different* legacy cache present too — must be ignored when rich exists.
    legacy = {3: _peaked(0.0, 0, 0)}
    legacy_frames_cache_path(tmp_path, "clip", 0.25).write_bytes(pickle.dumps(legacy))

    out = load_frame_fingerings(tmp_path, "clip", conf=0.25, cfg=CFG, fps=fps)

    assert set(out) == {3, 7}
    assert out[7] is None
    assert out[3] is not None
    # Reconstructed from the rich record (t = fi / fps), not the legacy entry.
    assert out[3].t == pytest.approx(3 / fps)
    expected = compute_fingering(rec.hand, rec.homography, CFG)
    np.testing.assert_array_equal(out[3].finger_pos_logits, expected.finger_pos_logits)


def test_load_falls_back_to_legacy_cache(tmp_path) -> None:
    legacy = {5: _peaked(0.2, 1, 4), 6: None}
    legacy_frames_cache_path(tmp_path, "clip", 0.25).write_bytes(pickle.dumps(legacy))

    out = load_frame_fingerings(tmp_path, "clip", conf=0.25, cfg=CFG, fps=24.0)

    assert set(out) == {5, 6}
    assert out[6] is None
    np.testing.assert_array_equal(out[5].finger_pos_logits, legacy[5].finger_pos_logits)


def test_load_missing_cache_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frame_fingerings(tmp_path, "absent", conf=0.25, cfg=CFG, fps=24.0)


# --------------------------------------------------------------------------- #
# needed_frames
# --------------------------------------------------------------------------- #
def test_needed_frames_single_frame_picks_nearest() -> None:
    needed, per_onset = needed_frames([1.0], 0.0, 10.0, window_s=0.06, max_frames=1)
    assert per_onset == {0: [10]}
    assert needed == {10}


def test_needed_frames_multi_frame_is_time_sorted() -> None:
    needed, per_onset = needed_frames([1.0], 0.0, 10.0, window_s=0.06, max_frames=3)
    # span = ceil(0.06 * 10) = 1 -> window {9, 10, 11}; time-sorted on return.
    assert per_onset == {0: [9, 10, 11]}
    assert needed == {9, 10, 11}


def test_needed_frames_applies_offset() -> None:
    # video_time = onset + offset; a +0.2 s offset shifts the centre frame by 2.
    _needed, per_onset = needed_frames([1.0], 0.2, 10.0, window_s=0.01, max_frames=1)
    assert per_onset == {0: [12]}


# --------------------------------------------------------------------------- #
# diagnose_clip_strings
# --------------------------------------------------------------------------- #
def test_diag_skips_unambiguous_and_counts_only_evidenced() -> None:
    # 3 notes: A ambiguous w/ evidence, B unambiguous (skipped), C ambiguous w/o evidence.
    gold = [
        _gold(1.0, string_idx=5, fret=0, pitch_midi=64),  # E4 — playable on all 6 strings
        _gold(2.0, string_idx=0, fret=0, pitch_midi=40),  # low-E open — only 1 candidate
        _gold(3.0, string_idx=5, fret=0, pitch_midi=64),  # ambiguous but no cached frame
    ]
    # fps/offset so onsets map to frames 10, 20, 30; only 10 and 20 are cached.
    per_frame = {10: _peaked(1.0, 5, 0), 20: _peaked(2.0, 0, 0)}

    d = diagnose_clip_strings(gold, per_frame, 0.0, 10.0, CFG, window_s=0.06, max_frames=1)

    assert d.n_gold == 3
    assert d.n_ambiguous == 2  # A and C; B (pitch 40) is single-candidate
    assert d.have_cv == 1  # only A has CV evidence near its onset
    assert d.correct == 1
    assert d.str_acc == pytest.approx(1.0)
    assert d.str_hist == {0: 1}
    assert d.fret_hist == {0: 1}


def test_diag_picks_orientation_maximizing_string_correctness() -> None:
    # gold is the HIGH-fret candidate (string 0, fret 24, pitch 64) so it is never
    # the audio-only default (string 5, fret 0). The raw fingering peaks at
    # (string 5, fret 24): only flip-string maps that onto the gold cell, so the
    # best-orientation search must select 'flip-string' and score it correct.
    gold = [_gold(1.0, string_idx=0, fret=24, pitch_midi=64)]
    per_frame = {10: _peaked(1.0, string_idx=5, fret=24)}

    d = diagnose_clip_strings(gold, per_frame, 0.0, 10.0, CFG, window_s=0.06, max_frames=1)

    assert d.best_orient == "flip-string"
    assert d.have_cv == 1
    assert d.correct == 1
    assert d.str_hist == {0: 1}
    assert d.fret_hist == {0: 1}


def test_diag_records_string_offset_when_wrong() -> None:
    # Evidence peaks at (string 2, fret 2) — off every candidate cell for pitch 64
    # under all four orientations, so each orientation falls back to the arg-max
    # default (string 5, fret 0). gold is (string 0, fret 24), so no orientation
    # can recover it: the prediction is wrong under all flips and the bass-side
    # string offset (+5) is recorded.
    gold = [_gold(1.0, string_idx=0, fret=24, pitch_midi=64)]
    per_frame = {10: _peaked(1.0, string_idx=2, fret=2)}

    d = diagnose_clip_strings(gold, per_frame, 0.0, 10.0, CFG, window_s=0.06, max_frames=1)

    assert d.have_cv == 1
    assert d.correct == 0
    assert d.best_orient == "none"  # all orientations tie at 0 correct → first wins
    # pred string 5 - gold string 0 = +5; pred fret 0 - gold fret 24 = -24.
    assert d.str_hist == {5: 1}
    assert d.fret_hist == {-24: 1}
