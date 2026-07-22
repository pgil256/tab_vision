"""Inharmonicity string evidence — physical per-note evidence for fusion.

A stiff string's partials run sharp of the harmonic series,
``f_k = k*f0*sqrt(1 + B*k^2)``, and the inharmonicity coefficient ``B`` scales
as ``1/L^2``. Fretting shortens the speaking length by ``2^(-n/12)``, so for a
given string

    B(s, n) = B0_s * 2^(n/6)

Two positions playing the *same pitch* therefore differ in ``B`` — 4-5 frets
apart on this instrument, a 1.6-1.8x ratio before plain-vs-wound construction
differences (`docs/EVAL_REPORTS/q6_separability_2026-07-22.md`). That makes
``B`` a *causal* cue for string identity rather than a learned convention,
which is what distinguishes it from every prior attempt in this program.

Measured behaviour (same report): 0.92 string accuracy on isolated notes
against a 0.65 count-prior control, and it transfers to the ensemble's
detected stream essentially unchanged. **Coverage is the constraint** — a
note must ring alone for the partial structure to be readable, which is ~34%
of solo notes and ~1% of strummed ones. This channel is a single-line
instrument by construction.

Evidence is emitted as a bounded product-of-experts term: the caller's
``weight`` is an exponent on the log-probabilities, so it cannot overrule the
corpus prior, and notes whose fit is poor contribute nothing at all rather
than contributing noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.evidence import combine_candidate_evidence
from tabvision.types import AudioEvent, GuitarConfig

LOG2 = math.log(2.0)

SKIP_ATTACK_S = 0.030
"""Attack transient is inharmonic noise; the steady state carries the fit."""

MIN_WINDOW_S = 0.120
MAX_WINDOW_S = 0.400
ZERO_PAD = 4
MAX_PARTIALS = 10
REL_TOLERANCE = 2 ** (60.0 / 1200.0) - 1.0
MIN_PARTIALS = 4
DEFAULT_MIN_R2 = 0.50
DEFAULT_SIGMA = 0.35
DEFAULT_WEIGHT = 0.5


@dataclass(frozen=True)
class InharmonicityFit:
    """One note's stiff-string fit."""

    f0_hz: float
    b: float
    partials: int
    r2: float

    @property
    def log_b(self) -> float:
        return math.log(self.b)


@dataclass(frozen=True)
class StringStiffnessModel:
    """Per-string open-string ``log B0``, from which ``B(s, n)`` follows.

    Calibrated from tab-labelled notes (see
    ``scripts/eval/q6_gate_a.py``); a per-session bootstrap is the intended
    production path and is not implemented here.
    """

    log_b0: Mapping[int, float]

    def predicted_log_b(self, string_idx: int, fret: int) -> float | None:
        base = self.log_b0.get(int(string_idx))
        if base is None:
            return None
        return base + (fret / 6.0) * LOG2


def _parabolic_peak(spectrum: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(spectrum) - 1:
        return float(index)
    left = math.log(max(float(spectrum[index - 1]), 1e-12))
    centre = math.log(max(float(spectrum[index]), 1e-12))
    right = math.log(max(float(spectrum[index + 1]), 1e-12))
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    return index + 0.5 * (left - right) / denominator


def _find_partials(
    spectrum: np.ndarray,
    freqs_per_bin: float,
    f0_guess: float,
    b_guess: float,
    sr: int,
    noise_floor: float,
) -> tuple[list[float], list[float]]:
    """Locate partials around the stiff-string prediction for ``b_guess``.

    The search half-width is capped at ``0.4*f0``. Without the cap a relative
    tolerance widens faster than partials separate, and by k~10 the window
    swallows its neighbour — the fit then locks onto the wrong peaks and
    reports a confidently biased answer.
    """
    ks: list[float] = []
    measured: list[float] = []
    for k in range(1, MAX_PARTIALS + 1):
        predicted = k * f0_guess * math.sqrt(1.0 + b_guess * k * k)
        if predicted > sr / 2.0 * 0.9:
            break
        tolerance = min(predicted * REL_TOLERANCE, 0.4 * f0_guess)
        low = int((predicted - tolerance) / freqs_per_bin)
        high = int((predicted + tolerance) / freqs_per_bin) + 1
        if low < 1 or high >= len(spectrum):
            break
        band = spectrum[low:high]
        peak = int(np.argmax(band))
        if float(band[peak]) <= noise_floor:
            continue
        ks.append(float(k))
        measured.append(_parabolic_peak(spectrum, low + peak) * freqs_per_bin)
    return ks, measured


def _fit(ks: Sequence[float], measured: Sequence[float]) -> tuple[float, float, float] | None:
    """Linearised fit: ``(f_k/k)^2 = f0^2 + (f0^2 B) k^2``."""
    if len(ks) < MIN_PARTIALS:
        return None
    k_arr = np.asarray(ks, dtype=np.float64)
    f_arr = np.asarray(measured, dtype=np.float64)
    x = k_arr**2
    y = (f_arr / k_arr) ** 2
    slope, intercept = np.polyfit(x, y, 1)
    if intercept <= 0.0:
        return None
    residual = y - (slope * x + intercept)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return float(math.sqrt(intercept)), float(slope / intercept), r2


def estimate_inharmonicity(
    segment: np.ndarray, sr: int, nominal_f0: float
) -> InharmonicityFit | None:
    """Fit ``B`` for one note segment, or ``None`` if it is not measurable."""
    if segment.size < int(MIN_WINDOW_S * sr):
        return None
    if not np.any(np.abs(segment) > 0.0):
        return None
    windowed = np.asarray(segment, dtype=np.float64) * np.hanning(segment.size)
    n_fft = int(2 ** math.ceil(math.log2(segment.size * ZERO_PAD)))
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    peak_magnitude = float(spectrum.max())
    if peak_magnitude <= 0.0:
        return None
    freqs_per_bin = sr / n_fft
    noise_floor = max(float(np.median(spectrum)) * 4.0, peak_magnitude * 1e-4)

    guess = 0.0
    fitted: tuple[float, float, float] | None = None
    used = 0
    # Pass 1 assumes a harmonic series; pass 2 re-centres on the partials the
    # fitted B actually predicts, which matters at high k where the stiffness
    # shift exceeds the search window.
    for _ in range(2):
        ks, measured = _find_partials(spectrum, freqs_per_bin, nominal_f0, guess, sr, noise_floor)
        fitted = _fit(ks, measured)
        if fitted is None:
            return None
        used = len(ks)
        guess = max(fitted[1], 0.0)
    if fitted is None or fitted[1] <= 0.0:
        return None
    return InharmonicityFit(f0_hz=fitted[0], b=fitted[1], partials=used, r2=fitted[2])


def inharmonicity_matrix(
    pitch_midi: int,
    cfg: GuitarConfig,
    log_b: float,
    model: StringStiffnessModel,
    *,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray | None:
    """Per-candidate likelihood of the measured ``log B``, as a prior matrix.

    Gaussian in log-B space: the separability study frames the decision as a
    mean-separation problem, and ``sigma`` is the estimator's relative error.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    candidates = candidate_positions(pitch_midi, cfg)
    if len(candidates) < 2:
        return None
    matrix = np.zeros((cfg.n_strings, cfg.max_fret + 1), dtype=np.float64)
    any_scored = False
    for candidate in candidates:
        predicted = model.predicted_log_b(candidate.string_idx, candidate.fret)
        if predicted is None:
            continue
        delta = (log_b - predicted) / sigma
        matrix[candidate.string_idx, candidate.fret] = math.exp(-0.5 * delta * delta)
        any_scored = True
    if not any_scored:
        return None
    total = float(matrix.sum())
    if total <= 0.0:
        return None
    return matrix / total


def _isolated_flags(events: Sequence[AudioEvent]) -> list[bool]:
    """True where no other event sounds during this note's analysis window.

    Isolation is judged on the *detected* stream, since that is all inference
    has. An undetected neighbour will violate it silently — which is why the
    fit quality gate below is load-bearing rather than cosmetic.
    """
    flags: list[bool] = []
    for event in events:
        duration = event.offset_s - event.onset_s
        start = event.onset_s + SKIP_ATTACK_S
        end = start + min(MAX_WINDOW_S, max(duration - SKIP_ATTACK_S, 0.0))
        flags.append(
            not any(
                other is not event and other.onset_s < end and other.offset_s > start
                for other in events
            )
        )
    return flags


def attach_inharmonicity_evidence(
    events: Sequence[AudioEvent],
    wav: np.ndarray,
    sr: int,
    model: StringStiffnessModel,
    cfg: GuitarConfig | None = None,
    *,
    weight: float = DEFAULT_WEIGHT,
    min_r2: float = DEFAULT_MIN_R2,
    sigma: float = DEFAULT_SIGMA,
) -> tuple[list[AudioEvent], dict[str, int]]:
    """Fold inharmonicity evidence into each event's ``fret_prior``.

    Returns the rewritten events and a coverage tally. Events that are not
    isolated, are unambiguous, or whose fit is below ``min_r2`` are returned
    untouched — the channel abstains rather than guessing.
    """
    cfg = cfg or GuitarConfig()
    if weight < 0.0:
        raise ValueError("weight must be non-negative")

    audio = np.asarray(wav, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ordered = sorted(events, key=lambda event: event.onset_s)
    isolated = _isolated_flags(ordered)

    tally = {"events": len(ordered), "isolated": 0, "fitted": 0, "applied": 0}
    out: list[AudioEvent] = []
    for event, is_isolated in zip(ordered, isolated, strict=True):
        if not is_isolated or weight == 0.0:
            out.append(event)
            continue
        tally["isolated"] += 1
        duration = event.offset_s - event.onset_s
        if duration < MIN_WINDOW_S + SKIP_ATTACK_S:
            out.append(event)
            continue
        start = int((event.onset_s + SKIP_ATTACK_S) * sr)
        stop = start + int(min(MAX_WINDOW_S, duration - SKIP_ATTACK_S) * sr)
        if start < 0 or stop > audio.size:
            out.append(event)
            continue
        nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
        fit = estimate_inharmonicity(audio[start:stop], sr, nominal)
        if fit is None or fit.r2 < min_r2:
            out.append(event)
            continue
        tally["fitted"] += 1
        matrix = inharmonicity_matrix(event.pitch_midi, cfg, fit.log_b, model, sigma=sigma)
        if matrix is None:
            out.append(event)
            continue
        combined = combine_candidate_evidence(
            event.pitch_midi,
            cfg,
            {
                "existing": (event.fret_prior, 1.0),
                "inharmonicity": (matrix, weight),
            },
        )
        if combined is None:
            out.append(event)
            continue
        tally["applied"] += 1
        out.append(replace(event, fret_prior=combined))
    return out, tally


__all__ = [
    "InharmonicityFit",
    "StringStiffnessModel",
    "attach_inharmonicity_evidence",
    "estimate_inharmonicity",
    "inharmonicity_matrix",
]
