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

**Generalizing across instruments.** ``B0_s`` is a property of the physical
string set and scale length, so a table fitted on one guitar does not
transfer to another. It does not need to: ``B ∝ 1/L²`` and the scale length
is shared by all six strings, so a different instrument largely *shifts* the
whole table rather than reshaping it. :func:`calibrate_from_session` exploits
that — it measures the recording's own notes against provisional string
assignments and re-fits ``B0``, per string where there is enough evidence and
as a single shared offset where there is not. A session therefore calibrates
itself from unlabelled audio, which is what makes this usable on a guitar the
project has never seen.
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
MIN_CLEAN_PARTIALS = 4
"""Surviving partials required when a note lost some to a collision.

A contaminated note is fitted on whatever partials remain, so the fit can be
supported by as few as ``MIN_PARTIALS`` points and is correspondingly easier
to satisfy by accident. Demanding more *surviving* evidence is the per-note,
generalizable way to separate a usable overlapped measurement from a lucky
one — as opposed to splitting on playing tier, which would be selection on
the evaluation set.
"""

SEPARATION_FACTOR = 3.0
"""Partial separability guard, in units of ``1/window_seconds``.

A Hann-windowed peak has a main lobe ~4/T wide, so two partials closer than
roughly that are one blob and the louder wins. 3/T is a slightly permissive
choice, validated rather than assumed (see the N1 coverage report).
"""
MIN_NOTES_PER_STRING = 8
"""Below this, a string re-uses the shared offset instead of its own median.

A handful of notes on one string is easily a handful of *misassigned* notes;
the shared offset is the robust fallback because scale length — the dominant
per-instrument term — is common to all six.
"""

MIN_NOTES_FOR_OFFSET = 12
"""Below this the session is too thin to calibrate at all; keep the seed."""


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

    ``fret_exponent`` is the ``k`` in ``B(s, n) = B0_s * 2^(k*n/6)``. Ideal
    fretting gives ``k = 1`` — stiff-string theory derives it, see
    :mod:`tabvision.fusion.string_physics` — but a real fret and fingertip
    terminate the string differently from the nut, so a calibration ritual
    that measures several frets per string can fit ``k`` rather than trust it.
    """

    log_b0: Mapping[int, float]
    fret_exponent: float = 1.0

    def predicted_log_b(self, string_idx: int, fret: int) -> float | None:
        base = self.log_b0.get(int(string_idx))
        if base is None:
            return None
        return base + self.fret_exponent * (fret / 6.0) * LOG2


@dataclass(frozen=True)
class StiffnessObservation:
    """One measured note with the position it is believed to have been played.

    ``string_idx``/``fret`` are *provisional* — for self-calibration they come
    from a first decode pass, not from labels.
    """

    string_idx: int
    fret: int
    log_b: float
    r2: float


def calibrate_from_session(
    observations: Sequence[StiffnessObservation],
    *,
    seed: StringStiffnessModel | None = None,
    min_r2: float = DEFAULT_MIN_R2,
) -> StringStiffnessModel | None:
    """Fit ``B0`` for the instrument in this recording.

    Two-tier, because per-string evidence is uneven and provisional labels are
    wrong some of the time:

    * a string with at least :data:`MIN_NOTES_PER_STRING` usable observations
      takes the **median** of ``log B - (fret/6)·log 2`` over its own notes —
      median rather than mean so a minority of misassigned notes cannot drag
      it;
    * every other string takes ``seed + shared_offset``, where the offset is
      the median residual against the seed across *all* strings. This is the
      scale-length term, which is common to the instrument.

    Returns ``None`` when the session is too thin to calibrate and no seed is
    available, so the caller can decline to apply evidence rather than apply
    badly-calibrated evidence.
    """
    usable = [item for item in observations if item.r2 >= min_r2]
    if not usable:
        return seed

    by_string: dict[int, list[float]] = {}
    for item in usable:
        by_string.setdefault(item.string_idx, []).append(item.log_b - (item.fret / 6.0) * LOG2)

    shared_offset = 0.0
    if seed is not None:
        residuals = [
            value - base
            for string, values in by_string.items()
            if (base := seed.log_b0.get(string)) is not None
            for value in values
        ]
        if len(residuals) >= MIN_NOTES_FOR_OFFSET:
            shared_offset = float(np.median(residuals))

    table: dict[int, float] = {}
    for string in range(6):
        values = by_string.get(string, [])
        if len(values) >= MIN_NOTES_PER_STRING:
            table[string] = float(np.median(values))
        elif seed is not None and string in seed.log_b0:
            table[string] = seed.log_b0[string] + shared_offset
    if not table:
        return seed
    return StringStiffnessModel(log_b0=table)


def calibrate_from_ritual(
    observations: Sequence[StiffnessObservation],
    *,
    min_r2: float = DEFAULT_MIN_R2,
    min_strings_for_exponent: int = 3,
    min_frets_per_string: int = 2,
) -> StringStiffnessModel | None:
    """Fit an instrument from a guided calibration take.

    Unlike :func:`calibrate_from_session` the labels here are *certain*: the
    application asked the player for a specific string and fret, so there is
    no decoder in the loop and none of the +0.30 bootstrap bias that sank
    self-calibration.

    With several frets per string the fret exponent becomes measurable rather
    than assumed. Each string contributes a least-squares slope of ``log B``
    against fret; the shared exponent is the median of those slopes, and each
    string's ``log B0`` is then its intercept under that shared exponent.
    Strings with a single fret still contribute ``B0`` — they just cannot vote
    on the exponent.
    """
    usable = [item for item in observations if item.r2 >= min_r2]
    if not usable:
        return None

    by_string: dict[int, list[tuple[int, float]]] = {}
    for item in usable:
        by_string.setdefault(item.string_idx, []).append((item.fret, item.log_b))

    slopes: list[float] = []
    for points in by_string.values():
        frets = sorted({fret for fret, _ in points})
        if len(frets) < min_frets_per_string:
            continue
        x = np.asarray([float(fret) for fret, _ in points])
        y = np.asarray([value for _, value in points])
        slope = float(np.polyfit(x, y, 1)[0])
        slopes.append(slope * 6.0 / LOG2)

    exponent = float(np.median(slopes)) if len(slopes) >= min_strings_for_exponent else 1.0

    table: dict[int, float] = {}
    for string, points in by_string.items():
        intercepts = [value - exponent * (fret / 6.0) * LOG2 for fret, value in points]
        table[string] = float(np.median(intercepts))
    if not table:
        return None
    return StringStiffnessModel(log_b0=table, fret_exponent=exponent)


def measure_events(
    events: Sequence[AudioEvent],
    wav: np.ndarray,
    sr: int,
    cfg: GuitarConfig | None = None,
    *,
    isolation: str = "strict",
    min_clean_partials: int = MIN_CLEAN_PARTIALS,
) -> dict[int, InharmonicityFit]:
    """Fit ``B`` for every measurable event; keyed by index into onset order.

    The **whole spectral half** of the channel. :func:`attach_inharmonicity_evidence`
    calls this and then does only the scoring half, so the two cannot drift, and
    so a caller can bank these fits to disk and replay table or admission
    variants against them for free.

    ``isolation`` selects how neighbours are handled and must match the mode the
    caller intends to score, because it changes *which* events get a fit and how
    contaminated partials are dropped:

    - ``"strict"`` requires a note to sound alone. The N1 diagnostic measured
      this as 88% of all lost coverage.
    - ``"partial_aware"`` drops only the partials a simultaneous note actually
      collides with and fits the rest, so an overlapped note is still measured
      whenever enough of its harmonic series survives. A contaminated fit
      resting on fewer than ``min_clean_partials`` surviving partials is
      discarded rather than trusted.

    Note the parameter is **not** cosmetic: it did not exist until 2026-07-25,
    which meant the banked-replay path could only ever express ``"strict"``.
    A Phase 0 run replayed banked fits while reporting itself as measuring the
    shipped ``partial_aware`` configuration, and silently scored the wrong arm.

    ``min_r2`` is deliberately *not* applied here — it is an admission
    threshold, not a measurement, so it belongs to the caller and can be swept
    against a fixed set of banked fits.
    """
    cfg = cfg or GuitarConfig()
    if isolation not in {"strict", "partial_aware"}:
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    audio = np.asarray(wav, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    ordered = sorted(events, key=lambda event: event.onset_s)
    isolated = _isolated_flags(ordered)
    fits: dict[int, InharmonicityFit] = {}
    for index, (event, is_isolated) in enumerate(zip(ordered, isolated, strict=True)):
        if isolation == "strict" and not is_isolated:
            continue
        duration = event.offset_s - event.onset_s
        if duration < MIN_WINDOW_S + SKIP_ATTACK_S:
            continue
        start = int((event.onset_s + SKIP_ATTACK_S) * sr)
        stop = start + int(min(MAX_WINDOW_S, duration - SKIP_ATTACK_S) * sr)
        if start < 0 or stop > audio.size:
            continue
        nominal = 440.0 * 2 ** ((event.pitch_midi - 69) / 12.0)
        blocked: list[float] = []
        separation = 0.0
        if isolation == "partial_aware" and not is_isolated:
            window_s = (stop - start) / sr
            separation = SEPARATION_FACTOR / max(window_s, 1e-6)
            for other in _overlapping(
                ordered, index, event.onset_s + SKIP_ATTACK_S, event.onset_s + duration
            ):
                blocked.extend(_harmonic_frequencies(other.pitch_midi))
        fit = estimate_inharmonicity(
            audio[start:stop],
            sr,
            nominal,
            blocked_hz=blocked,
            min_separation_hz=separation,
        )
        if fit is None:
            continue
        if blocked and fit.partials < min_clean_partials:
            # Contaminated and thinly supported: abstain rather than trust a
            # fit resting on a handful of surviving partials.
            continue
        fits[index] = fit
    return fits


def isolation_flags(events: Sequence[AudioEvent]) -> list[bool]:
    """Public view of the isolation gate, in onset order.

    Exposed so a banked-replay harness can reconstruct the coverage tally
    without redoing the spectral work.
    """
    return _isolated_flags(sorted(events, key=lambda event: event.onset_s))


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
    blocked_hz: Sequence[float] = (),
    min_separation_hz: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Locate partials around the stiff-string prediction for ``b_guess``.

    The search half-width is capped at ``0.4*f0``. Without the cap a relative
    tolerance widens faster than partials separate, and by k~10 the window
    swallows its neighbour — the fit then locks onto the wrong peaks and
    reports a confidently biased answer.

    ``blocked_hz`` lists frequencies belonging to *other* notes sounding at
    the same time. A partial within ``min_separation_hz`` of one is dropped
    rather than measured, because the two are unresolvable and the peak would
    report the louder note. Dropping the contaminated partials instead of the
    whole note is what lets an overlapped note still be fitted.
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
        if any(abs(predicted - other) < min_separation_hz for other in blocked_hz):
            continue
        band = spectrum[low:high]
        peak = int(np.argmax(band))
        if float(band[peak]) <= noise_floor:
            continue
        refined = _parabolic_peak(spectrum, low + peak) * freqs_per_bin
        # Re-check the located peak, not just the predicted centre: a strong
        # interferer inside the search window would otherwise be measured as
        # this note's partial.
        if any(abs(refined - other) < min_separation_hz for other in blocked_hz):
            continue
        ks.append(float(k))
        measured.append(refined)
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
    segment: np.ndarray,
    sr: int,
    nominal_f0: float,
    *,
    blocked_hz: Sequence[float] = (),
    min_separation_hz: float = 0.0,
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
        ks, measured = _find_partials(
            spectrum,
            freqs_per_bin,
            nominal_f0,
            guess,
            sr,
            noise_floor,
            blocked_hz,
            min_separation_hz,
        )
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
    for candidate in candidates:
        predicted = model.predicted_log_b(candidate.string_idx, candidate.fret)
        if predicted is None:
            # An uncalibrated string must make the channel abstain on this
            # note, not score the candidate at zero — a zero is a hard veto
            # that would silently force the note onto whichever strings the
            # calibration happened to cover.
            return None
        delta = (log_b - predicted) / sigma
        matrix[candidate.string_idx, candidate.fret] = math.exp(-0.5 * delta * delta)
    total = float(matrix.sum())
    if total <= 0.0:
        return None
    return matrix / total


def _harmonic_frequencies(pitch_midi: int, count: int = MAX_PARTIALS) -> list[float]:
    """Approximate partial frequencies of an interfering note.

    The harmonic series is close enough for a collision test: ``B`` shifts a
    partial by well under the resolution limit that makes two partials
    unresolvable in the first place.
    """
    f0 = 440.0 * 2 ** ((pitch_midi - 69) / 12.0)
    return [k * f0 for k in range(1, count + 1)]


def _overlapping(
    events: Sequence[AudioEvent], index: int, start: float, end: float
) -> list[AudioEvent]:
    """Events other than ``index`` sounding during ``[start, end]``."""
    return [
        other
        for position, other in enumerate(events)
        if position != index and other.onset_s < end and other.offset_s > start
    ]


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
    model: StringStiffnessModel | None,
    cfg: GuitarConfig | None = None,
    *,
    weight: float = DEFAULT_WEIGHT,
    min_r2: float = DEFAULT_MIN_R2,
    sigma: float = DEFAULT_SIGMA,
    isolation: str = "strict",
    min_clean_partials: int = MIN_CLEAN_PARTIALS,
) -> tuple[list[AudioEvent], dict[str, int]]:
    """Fold inharmonicity evidence into each event's ``fret_prior``.

    Returns the rewritten events and a coverage tally. Events that are not
    isolated, are unambiguous, or whose fit is below ``min_r2`` are returned
    untouched — the channel abstains rather than guessing.

    ``isolation`` selects how neighbours are handled. ``"strict"`` requires a
    note to sound alone, which the N1 diagnostic measured as **88% of all lost
    coverage**. ``"partial_aware"`` instead drops only the partials a
    simultaneous note actually collides with and fits the rest, so an
    overlapped note is still measured whenever enough of its harmonic series
    survives.
    """
    cfg = cfg or GuitarConfig()
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    if isolation not in {"strict", "partial_aware"}:
        raise ValueError(f"unknown isolation mode: {isolation!r}")
    if model is None:
        # No table describes this instrument's strings — see
        # ``string_physics.stiffness_model_for_session``. Returning the stream
        # untouched keeps out-of-domain sessions bit-identical to baseline.
        return list(events), {"events": len(events), "isolated": 0, "fitted": 0, "applied": 0}

    ordered = sorted(events, key=lambda event: event.onset_s)
    if weight == 0.0:
        return list(ordered), {
            "events": len(ordered),
            "isolated": 0,
            "fitted": 0,
            "applied": 0,
        }

    fits = measure_events(
        ordered,
        wav,
        sr,
        cfg,
        isolation=isolation,
        min_clean_partials=min_clean_partials,
    )
    return apply_fits(
        ordered,
        fits,
        model,
        cfg,
        weight=weight,
        min_r2=min_r2,
        sigma=sigma,
        isolation=isolation,
    )


def apply_fits(
    ordered: Sequence[AudioEvent],
    fits: dict[int, InharmonicityFit],
    model: StringStiffnessModel,
    cfg: GuitarConfig | None = None,
    *,
    weight: float = DEFAULT_WEIGHT,
    min_r2: float = DEFAULT_MIN_R2,
    sigma: float = DEFAULT_SIGMA,
    isolation: str = "strict",
) -> tuple[list[AudioEvent], dict[str, int]]:
    """The scoring half: fold measured fits into each event's ``fret_prior``.

    Separated from the spectral half so admission thresholds (``min_r2``),
    ``sigma``, ``weight`` and the stiffness table can be swept against banked
    fits without re-analysing audio. ``ordered`` must be in onset order and
    ``fits`` keyed by index into it — exactly what :func:`measure_events`
    returns for the same event list.
    """
    cfg = cfg or GuitarConfig()
    isolated = _isolated_flags(ordered)
    tally = {"events": len(ordered), "isolated": 0, "fitted": 0, "applied": 0}
    out: list[AudioEvent] = []
    for index, (event, is_isolated) in enumerate(zip(ordered, isolated, strict=True)):
        if isolation == "strict" and not is_isolated:
            out.append(event)
            continue
        tally["isolated"] += 1
        fit = fits.get(index)
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
    "StiffnessObservation",
    "StringStiffnessModel",
    "calibrate_from_ritual",
    "calibrate_from_session",
    "measure_events",
    "apply_fits",
    "isolation_flags",
    "attach_inharmonicity_evidence",
    "estimate_inharmonicity",
    "inharmonicity_matrix",
]
