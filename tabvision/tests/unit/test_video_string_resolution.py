"""The fusion resolver uses a per-note ``FrameFingering`` to pick the string that
audio cannot — the v1.1 lever. Guards the path validated by the 2026-06-03 oracle
probe (``docs/EVAL_REPORTS/v1_1_oracle_string_probe_2026-06-03.md``): a confident
hand signal overrides the audio-only string choice, and an absent hand signal
leaves the audio path exactly unchanged (the no-regression guarantee).
"""

from __future__ import annotations

import numpy as np

from tabvision.fusion import fuse
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import AudioEvent, FrameFingering, GuitarConfig


def _oracle_fingering(t: float, string_idx: int, fret: int, cfg: GuitarConfig) -> FrameFingering:
    """A FrameFingering whose ``marginal_string_fret`` is peaked on ``(string, fret)``."""
    logits = np.full((4, cfg.n_strings, cfg.max_fret + 1), -10.0)
    logits[0, string_idx, fret] = 5.0
    return FrameFingering(t=t, finger_pos_logits=logits, homography_confidence=1.0)


def test_oracle_fingering_resolves_ambiguous_string() -> None:
    cfg = GuitarConfig()
    pitch = 64  # E4 — playable on every string, maximally string-ambiguous from audio
    cands = candidate_positions(pitch, cfg)
    assert len(cands) >= 2
    target = cands[-1]  # highest-fret position; never the audio-only low-fret default

    ev = AudioEvent(onset_s=1.0, offset_s=1.5, pitch_midi=pitch, velocity=1.0, confidence=1.0)

    audio_only = fuse([ev], [], cfg)
    with_oracle = fuse([ev], [_oracle_fingering(1.0, target.string_idx, target.fret, cfg)], cfg)

    assert len(with_oracle) == 1
    assert (with_oracle[0].string_idx, with_oracle[0].fret) == (target.string_idx, target.fret)
    # The hand signal actually changed the decision vs audio-only.
    assert len(audio_only) == 1
    assert (audio_only[0].string_idx, audio_only[0].fret) != (target.string_idx, target.fret)


def test_absent_fingering_is_pure_audio_decode() -> None:
    """No-regression guarantee: empty/absent fingerings == the audio-only decode."""
    cfg = GuitarConfig()
    ev = AudioEvent(onset_s=0.0, offset_s=0.4, pitch_midi=60, velocity=1.0, confidence=1.0)
    out = fuse([ev], [], cfg)
    assert len(out) == 1
    assert out[0].pitch_midi == 60
    # Deterministic and unaffected by an all-zero (evidence-free) fingering.
    zero = FrameFingering(
        t=0.0,
        finger_pos_logits=np.zeros((4, cfg.n_strings, cfg.max_fret + 1)),
        homography_confidence=0.0,
    )
    assert fuse([ev], [zero], cfg) == out
