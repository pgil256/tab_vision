"""Beat-grid detection for display and export timing (advisory only).

Detects a tempo and tracked beat times from the demuxed audio so the editor
can draw measure lines and exports can quantize on read. Nothing here moves
note onsets: SPEC §9.2 accuracy metrics are defined on raw model onsets, and
onset snapping was measured negative (accuracy-loop Q5).
"""

from __future__ import annotations

import logging

import numpy as np

from tabvision.types import BeatGrid

logger = logging.getLogger(__name__)

# Below any of these the tracker output is more likely noise than musical
# structure; callers fall back to a plain seconds axis.
MIN_CLIP_SECONDS = 4.0
MIN_BEATS = 8
MIN_BPM = 40.0
MAX_BPM = 240.0


def detect_beat_grid(wav: np.ndarray, sample_rate: int) -> BeatGrid | None:
    """Detect tempo + beat times from mono audio; ``None`` when unreliable.

    Fail-open by design: librosa absence or any tracking failure returns
    ``None`` rather than raising — the grid is advisory metadata and must
    never fail a transcription.
    """
    if wav is None or wav.size < MIN_CLIP_SECONDS * sample_rate:
        return None
    try:
        import librosa
    except Exception:  # noqa: BLE001
        logger.warning("librosa unavailable; skipping beat-grid detection")
        return None
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            y=np.asarray(wav, dtype=np.float32), sr=sample_rate
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    except Exception as exc:  # noqa: BLE001 — advisory, never fails a job
        logger.warning("beat tracking failed: %s", exc)
        return None

    # librosa >= 0.10 may return tempo as a one-element ndarray.
    if hasattr(tempo, "__len__"):
        tempo_bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_bpm = float(tempo)

    if len(beat_times) < MIN_BEATS or not (MIN_BPM <= tempo_bpm <= MAX_BPM):
        logger.info(
            "beat grid rejected (tempo=%.1f BPM, beats=%d)", tempo_bpm, len(beat_times)
        )
        return None
    return BeatGrid(
        tempo_bpm=tempo_bpm,
        beat_times=tuple(float(t) for t in beat_times),
    )
