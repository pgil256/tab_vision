"""Tests for the N4 ritual-validation study.

Three things carry the study's validity: the ritual take must be selected the
way a guided take would be (frets *spread* per string, since the exponent is a
slope), the level offset must be the robust statistic it claims to be, and the
bleed guard must actually reject a note whose labelled channel is not the one
carrying it — otherwise a bled-through neighbour supplies the "measurement"
for a string that was never plucked.
"""

from __future__ import annotations

import json
import math

import numpy as np

from scripts.eval.n4_ritual_validation import (
    MAX_FRETS_PER_STRING,
    admit_note,
    level_offset,
    load_cached_fits,
    select_ritual,
)
from tabvision.fusion.inharmonicity import (
    LOG2,
    StringStiffnessModel,
    calibrate_from_ritual,
)
from tabvision.fusion.string_physics import reference_stiffness_model

SR = 44100


def _cells(spec: dict[int, list[int]], log_b: float = -9.0):
    cells = {(s, f): log_b + 0.01 * f for s, frets in spec.items() for f in frets}
    r2 = {key: 0.9 for key in cells}
    return cells, r2


def _stiff_string(f0: float, b_value: float, seconds: float = 0.45) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(t)
    for k in range(1, 11):
        freq = k * f0 * math.sqrt(1.0 + b_value * k * k)
        if freq > SR / 2.5:
            break
        signal += np.sin(2 * np.pi * freq * t) / k
    return signal


def test_select_ritual_caps_frets_per_string() -> None:
    cells, r2 = _cells({0: [0, 1, 2, 3, 4, 5, 7, 9, 12]})
    observations = select_ritual(cells, r2)
    assert len(observations) == MAX_FRETS_PER_STRING


def test_select_ritual_spreads_frets_rather_than_clustering() -> None:
    """The exponent is a slope; three frets at the nut measure it badly."""
    cells, r2 = _cells({3: [0, 1, 2, 3, 10, 11, 12]})
    frets = sorted(item.fret for item in select_ritual(cells, r2))
    assert frets[0] == 0
    assert frets[-1] == 12
    assert frets[-1] - frets[0] > 6


def test_select_ritual_keeps_every_covered_string() -> None:
    cells, r2 = _cells({s: [0, 5] for s in range(6)})
    observations = select_ritual(cells, r2)
    assert sorted({item.string_idx for item in observations}) == list(range(6))


def test_select_ritual_ignores_absent_strings() -> None:
    cells, r2 = _cells({1: [0, 3], 4: [2]})
    assert sorted({item.string_idx for item in select_ritual(cells, r2)}) == [1, 4]


def test_level_offset_is_the_median_shift() -> None:
    reference = reference_stiffness_model()
    moved = StringStiffnessModel(log_b0={s: value + 0.4 for s, value in reference.log_b0.items()})
    assert abs(level_offset(moved, reference) - 0.4) < 1e-12


def test_level_offset_resists_a_single_wild_string() -> None:
    """Median, not mean — one bad string must not move the level."""
    reference = reference_stiffness_model()
    shifted = {s: value + 0.4 for s, value in reference.log_b0.items()}
    shifted[2] += 9.0
    assert abs(level_offset(StringStiffnessModel(log_b0=shifted), reference) - 0.4) < 1e-9


def test_level_offset_is_none_without_shared_strings() -> None:
    assert level_offset(StringStiffnessModel(log_b0={}), reference_stiffness_model()) is None


def test_ritual_round_trip_recovers_a_known_table() -> None:
    """Synthesise a take from a known model; calibration must return it."""
    truth = StringStiffnessModel(log_b0={s: -9.5 - 0.2 * s for s in range(6)}, fret_exponent=1.0)
    cells = {}
    r2 = {}
    for string in range(6):
        for fret in (0, 5, 10):
            cells[(string, fret)] = truth.log_b0[string] + (fret / 6.0) * LOG2
            r2[(string, fret)] = 0.95
    model = calibrate_from_ritual(select_ritual(cells, r2))
    assert model is not None
    assert abs(model.fret_exponent - 1.0) < 1e-9
    for string in range(6):
        assert abs(model.log_b0[string] - truth.log_b0[string]) < 1e-9


def test_load_cached_fits_round_trips(tmp_path) -> None:
    path = tmp_path / "clip.fits.json"
    assert load_cached_fits(path) is None
    path.write_text(json.dumps({"3": [-9.25, 0.81]}), encoding="utf-8")
    assert load_cached_fits(path) == {3: (-9.25, 0.81)}


def test_admit_note_accepts_the_channel_that_carries_the_note() -> None:
    pitch_f0 = 196.0  # G3
    audio = np.random.default_rng(42).standard_normal((int(SR * 0.45), 6)) * 1e-4
    audio[:, 3] += _stiff_string(pitch_f0, 3.0e-4)
    result = admit_note(audio, SR, 0, audio.shape[0], pitch_f0, 3, min_r2=0.5)
    assert result is not None
    log_b, r2 = result
    assert r2 >= 0.5
    assert abs(math.exp(log_b) - 3.0e-4) < 1.5e-4


def test_admit_note_rejects_when_a_neighbour_carries_the_note() -> None:
    """The bleed guard: label says string 2, the signal is on channel 3."""
    pitch_f0 = 196.0
    audio = np.random.default_rng(42).standard_normal((int(SR * 0.45), 6)) * 1e-4
    audio[:, 3] += _stiff_string(pitch_f0, 3.0e-4)
    audio[:, 2] += 0.02 * _stiff_string(pitch_f0, 3.0e-4)  # faint bleed
    assert admit_note(audio, SR, 0, audio.shape[0], pitch_f0, 2, min_r2=0.5) is None


def test_admit_note_rejects_a_silent_channel() -> None:
    pitch_f0 = 196.0
    audio = np.zeros((int(SR * 0.45), 6))
    audio[:, 3] += _stiff_string(pitch_f0, 3.0e-4)
    assert admit_note(audio, SR, 0, audio.shape[0], pitch_f0, 0, min_r2=0.5) is None
