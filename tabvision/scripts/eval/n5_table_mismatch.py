"""Accuracy-loop N5 — how wrong can the physics table be before it stops helping?

Q6 shipped `acoustic-physics-v1`: one light-gauge phosphor-bronze set on a
25.4" scale, derived from published specifications. Every gate it passed was
measured on GuitarSet, whose instruments resemble that table. The pending
default-on decision therefore rests on an *argument* — "``B`` follows from
specifications, so it ports" — rather than a measurement. This study replaces
the argument with a tolerance curve.

Two physically distinct axes, because a different guitar moves the table in
two different ways:

**Uniform offset.** ``B = pi^3 E d^4 / (256 mu L^4 f^2)``, so at fixed strings
and fixed pitch a change of scale length multiplies every string's ``B`` by
``(L_ref / L)^4`` — one shared factor, i.e. a pure additive offset in log-B.
Everything else that scales all six strings together (a global core-diameter
or modulus error) lands on this same axis, which is what makes it the natural
summary of "the whole table is off".

**String set.** Gauge changes each string differently: for a plain string
``mu ~ d^2`` so ``B ~ d^2``, while a wound string's core and total mass move
independently. This reshapes the table rather than shifting it.

Both axes are *derived*, not invented. Alternative sets are built from
published gauges plus a wound-string model whose two free constants — the
wrap packing factor and the core-diameter/gauge relation — are fitted to the
shipped table itself. Each variant is then applied to the registered table as
a *difference* rather than replacing it, so the wound model's own fit
residual (up to 0.09 log-B, a quarter of the channel's sigma) cancels and an
arm's effect is attributable to the string change alone.

Method: the measured ``B`` of a note does not depend on the table (the table
enters only when candidates are scored), so measurements are banked once per
clip and every variant replays against them in milliseconds. The replay is
asserted equal to `attach_inharmonicity_evidence` under the shipped table
before any variant is trusted.

PRE-DECLARED READING (fixed before the run):

* **Robust** — every derived real-guitar set keeps lo-95 > 0, and the offset
  tolerance band that holds lo-95 > 0 covers at least +/-0.10 log-B (the
  span of real acoustic scale lengths, 24.75"-25.6"). Default-on is then
  defensible for an unknown steel-string acoustic.
* **Conditional** — some derived sets stay positive and others wash out, but
  none is significantly negative. Default-on is defensible as a
  non-harmful-in-expectation change; self-calibration is the upgrade path.
* **Fragile** — any derived real-guitar set is significantly negative
  (hi-95 < 0). Default-on then needs calibration or a gauge input first, and
  the recommendation flips.

The sigma arm is diagnostic only: a mismatched table has larger effective
error, so widening the posterior's sigma should trade peak gain for
robustness. It is reported as a direction, never as a proposed default —
choosing sigma on this run's result would be tuning on the test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import (
    DEV_PLAYERS,
    _event_from_json,
    _score,
    build_oof_priors,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.guitarset_audio import load_mono_audio, parse_guitarset_jams
from tabvision.fusion.evidence import combine_candidate_evidence
from tabvision.fusion.inharmonicity import (
    StringStiffnessModel,
    attach_inharmonicity_evidence,
    inharmonicity_matrix,
    measure_events,
)
from tabvision.fusion.string_physics import (
    ACOUSTIC_LIGHT_SET,
    DEFAULT_SCALE_LENGTH_IN,
    IN_TO_M,
    LB_PER_IN_TO_KG_PER_M,
    STEEL_YOUNGS_MODULUS_PA,
    StringSpec,
    inharmonicity_coefficient,
    reference_stiffness_model,
)
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig

# Frozen decode configuration — the registered acoustic-physics-v1 settings.
WEIGHT = 1.0
MIN_R2 = 0.50
SIGMA = 0.35
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

STEEL_DENSITY_KG_M3 = 7850.0
"""Music-wire steel. Recovers the shipped plain-string unit weights to <1%."""

WRAP_DENSITY_KG_M3 = 8800.0
"""Phosphor bronze. Only enters the fitted packing factor, which absorbs any
error in it, so the alternative sets are insensitive to this constant."""

# Published gauges, high-E first, converted to repo order (low-E first) below.
EXTRA_LIGHT_GAUGES = (0.010, 0.014, 0.023, 0.030, 0.039, 0.047)
LIGHT_GAUGES = (0.012, 0.016, 0.024, 0.032, 0.042, 0.053)
MEDIUM_GAUGES = (0.013, 0.017, 0.026, 0.035, 0.045, 0.056)

OPEN_MIDI = (40, 45, 50, 55, 59, 64)
WOUND = (True, True, True, True, False, False)
STRING_NAMES = ("E2", "A2", "D3", "G3", "B3", "E4")

OFFSETS = (-0.60, -0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40, 0.60)
"""Uniform log-B offsets. +/-0.10 spans real acoustic scale lengths; the wider
points exist to locate the edge of the usable band, not because a guitar
lives there."""

SIGMA_ARM = 0.60
"""Diagnostic only — see the module docstring."""


@dataclass(frozen=True)
class WoundModel:
    """The two constants that turn a gauge into a wound-string specification.

    Both are least-squares fits to :data:`ACOUSTIC_LIGHT_SET`, so a derived
    set reproduces the shipped table exactly at the shipped gauges.
    """

    core_intercept_in: float
    core_slope: float
    packing: float

    def core_diameter_in(self, gauge_in: float) -> float:
        return self.core_intercept_in + self.core_slope * gauge_in

    def unit_weight_lb_per_in(self, gauge_in: float) -> float:
        core_r = self.core_diameter_in(gauge_in) * IN_TO_M / 2.0
        outer_r = gauge_in * IN_TO_M / 2.0
        core_mu = STEEL_DENSITY_KG_M3 * math.pi * core_r**2
        wrap_mu = self.packing * WRAP_DENSITY_KG_M3 * math.pi * (outer_r**2 - core_r**2)
        return (core_mu + wrap_mu) / LB_PER_IN_TO_KG_PER_M


def fit_wound_model(specs: tuple[StringSpec, ...] = ACOUSTIC_LIGHT_SET) -> WoundModel:
    """Recover the shipped table's own wound-string model.

    Fitting rather than asserting keeps every alternative set anchored to the
    gate-passed table: at the shipped gauges the model reproduces the shipped
    core diameters and unit weights, so a variant's difference is attributable
    to the gauge change alone.
    """
    wound = [spec for spec in specs if spec.wound]
    gauges = np.asarray([spec.gauge_in for spec in wound], dtype=np.float64)
    cores = np.asarray([spec.core_diameter_in for spec in wound], dtype=np.float64)
    slope, intercept = np.polyfit(gauges, cores, 1)
    packings: list[float] = []
    for spec in wound:
        core_r = spec.core_diameter_in * IN_TO_M / 2.0
        outer_r = spec.gauge_in * IN_TO_M / 2.0
        total_mu = spec.unit_weight_lb_per_in * LB_PER_IN_TO_KG_PER_M
        core_mu = STEEL_DENSITY_KG_M3 * math.pi * core_r**2
        annulus = math.pi * (outer_r**2 - core_r**2)
        packings.append((total_mu - core_mu) / (WRAP_DENSITY_KG_M3 * annulus))
    return WoundModel(
        core_intercept_in=float(intercept),
        core_slope=float(slope),
        packing=float(np.mean(packings)),
    )


def plain_unit_weight_lb_per_in(gauge_in: float) -> float:
    radius_m = gauge_in * IN_TO_M / 2.0
    return (STEEL_DENSITY_KG_M3 * math.pi * radius_m**2) / LB_PER_IN_TO_KG_PER_M


def derive_set(
    gauges_high_first: tuple[float, ...],
    model: WoundModel,
    *,
    core_scale: float = 1.0,
) -> tuple[StringSpec, ...]:
    """Build a specification set from published gauges alone.

    ``core_scale`` perturbs only the wound cores, which is how round-core and
    hex-core constructions of the same nominal gauge actually differ.
    """
    gauges = tuple(reversed(gauges_high_first))
    specs: list[StringSpec] = []
    for index, gauge in enumerate(gauges):
        if WOUND[index]:
            core = model.core_diameter_in(gauge) * core_scale
            unit_weight = model.unit_weight_lb_per_in(gauge)
        else:
            core = gauge
            unit_weight = plain_unit_weight_lb_per_in(gauge)
        specs.append(
            StringSpec(
                name=STRING_NAMES[index],
                gauge_in=gauge,
                core_diameter_in=core,
                unit_weight_lb_per_in=unit_weight,
                wound=WOUND[index],
                open_midi=OPEN_MIDI[index],
            )
        )
    return tuple(specs)


def model_from_specs(
    specs: tuple[StringSpec, ...], *, scale_length_in: float = DEFAULT_SCALE_LENGTH_IN
) -> StringStiffnessModel:
    table = {
        index: math.log(
            inharmonicity_coefficient(
                spec,
                scale_length_in=scale_length_in,
                youngs_modulus_pa=STEEL_YOUNGS_MODULUS_PA,
            )
        )
        for index, spec in enumerate(specs)
    }
    return StringStiffnessModel(log_b0=table)


def offset_model(base: StringStiffnessModel, offset: float) -> StringStiffnessModel:
    return StringStiffnessModel(
        log_b0={index: value + offset for index, value in base.log_b0.items()},
        fret_exponent=base.fret_exponent,
    )


def perturbed_model(
    variant_specs: tuple[StringSpec, ...],
    *,
    scale_length_in: float = DEFAULT_SCALE_LENGTH_IN,
) -> StringStiffnessModel:
    """The shipped table moved by the *difference* this variant implies.

    Building a variant table from scratch would fold in the wound model's own
    fit residual — up to 0.09 in log-B on G3, a quarter of the channel's
    sigma — so a measured effect would mix the string change with modelling
    noise. Taking the difference against the same model's light-set
    prediction cancels that residual exactly: at the shipped gauges and scale
    the variant *is* the registered table, bit for bit, and every arm differs
    from it only by the physics of the change under test.
    """
    reference = model_from_specs(derive_set(LIGHT_GAUGES, fit_wound_model()))
    variant = model_from_specs(variant_specs, scale_length_in=scale_length_in)
    shipped = reference_stiffness_model()
    return StringStiffnessModel(
        log_b0={
            index: shipped.log_b0[index] + (variant.log_b0[index] - reference.log_b0[index])
            for index in shipped.log_b0
        }
    )


def build_variants() -> list[dict[str, Any]]:
    """Every arm, declared before the run. Order is the report's order."""
    shipped = reference_stiffness_model()
    wound = fit_wound_model()
    variants: list[dict[str, Any]] = []
    for offset in OFFSETS:
        variants.append(
            {
                "label": f"offset{offset:+.2f}",
                "axis": "uniform_offset",
                "offset": offset,
                "sigma": SIGMA,
                "model": offset_model(shipped, offset),
                "note": "shipped table"
                if offset == 0.0
                else f"all six strings x{math.exp(offset):.2f}",
            }
        )
    real: list[tuple[str, StringStiffnessModel, str]] = [
        (
            "set:extra-light",
            perturbed_model(derive_set(EXTRA_LIGHT_GAUGES, wound)),
            ".010-.047 on a 25.4in scale",
        ),
        (
            "set:medium",
            perturbed_model(derive_set(MEDIUM_GAUGES, wound)),
            ".013-.056 on a 25.4in scale",
        ),
        (
            "scale:24.75in",
            perturbed_model(derive_set(LIGHT_GAUGES, wound), scale_length_in=24.75),
            "short-scale body, shipped gauges",
        ),
        (
            "scale:25.6in",
            perturbed_model(derive_set(LIGHT_GAUGES, wound), scale_length_in=25.6),
            "long-scale body, shipped gauges",
        ),
        (
            "core:round-0.90",
            perturbed_model(derive_set(LIGHT_GAUGES, wound, core_scale=0.90)),
            "wound cores 10% thinner at the same gauge",
        ),
        (
            "core:hex-1.10",
            perturbed_model(derive_set(LIGHT_GAUGES, wound, core_scale=1.10)),
            "wound cores 10% thicker at the same gauge",
        ),
    ]
    for label, model, note in real:
        variants.append(
            {
                "label": label,
                "axis": "real_set",
                "offset": None,
                "sigma": SIGMA,
                "model": model,
                "note": note,
            }
        )
    worst = max(OFFSETS, key=abs)
    variants.append(
        {
            "label": f"sigma{SIGMA_ARM:.2f}@offset{worst:+.2f}",
            "axis": "sigma_diagnostic",
            "offset": worst,
            "sigma": SIGMA_ARM,
            "model": offset_model(shipped, worst),
            "note": "diagnostic: does a wider posterior buy robustness?",
        }
    )
    variants.append(
        {
            "label": f"sigma{SIGMA_ARM:.2f}@offset+0.00",
            "axis": "sigma_diagnostic",
            "offset": 0.0,
            "sigma": SIGMA_ARM,
            "model": shipped,
            "note": "diagnostic: cost of the wider posterior when the table is right",
        }
    )
    return variants


def apply_banked(
    ordered: list[AudioEvent],
    fits: dict[int, tuple[float, float]],
    model: StringStiffnessModel,
    cfg: GuitarConfig,
    *,
    weight: float,
    min_r2: float,
    sigma: float,
) -> tuple[list[AudioEvent], int]:
    """Replay of `attach_inharmonicity_evidence`'s scoring half.

    ``fits`` maps an index into ``ordered`` to ``(log_b, r2)``. Only the
    scoring half is replayed, because that is the only half a table change can
    affect; :func:`self_check` proves the two agree.
    """
    out: list[AudioEvent] = []
    applied = 0
    for index, event in enumerate(ordered):
        banked = fits.get(index)
        if banked is None or banked[1] < min_r2:
            out.append(event)
            continue
        matrix = inharmonicity_matrix(event.pitch_midi, cfg, banked[0], model, sigma=sigma)
        if matrix is None:
            out.append(event)
            continue
        combined = combine_candidate_evidence(
            event.pitch_midi,
            cfg,
            {"existing": (event.fret_prior, 1.0), "inharmonicity": (matrix, weight)},
        )
        if combined is None:
            out.append(event)
            continue
        applied += 1
        out.append(replace(event, fret_prior=combined))
    return out, applied


def self_check(
    events: list[AudioEvent],
    wav: np.ndarray,
    sr: int,
    fits: dict[int, tuple[float, float]],
    cfg: GuitarConfig,
) -> None:
    """The replay must equal the shipped code path, or nothing here is valid."""
    shipped_events, tally = attach_inharmonicity_evidence(
        events, wav, sr, reference_stiffness_model(), cfg, weight=WEIGHT, min_r2=MIN_R2, sigma=SIGMA
    )
    ordered = sorted(events, key=lambda event: event.onset_s)
    replayed, applied = apply_banked(
        ordered, fits, reference_stiffness_model(), cfg, weight=WEIGHT, min_r2=MIN_R2, sigma=SIGMA
    )
    if applied != tally["applied"]:
        raise SystemExit(f"replay applied {applied}, shipped path applied {tally['applied']}")
    for left, right in zip(shipped_events, replayed, strict=True):
        if not np.array_equal(left.fret_prior, right.fret_prior):
            raise SystemExit(f"replay diverged from the shipped path at {left.onset_s:.3f}s")


def dev_clips(data_home: Path) -> list[str]:
    return sorted(
        path.stem
        for path in (data_home / "annotation").glob("*.jams")
        if path.stem[:2] in DEV_PLAYERS
    )


def load_fits(
    cache: Path, events: list[AudioEvent], wav: np.ndarray, sr: int, cfg: GuitarConfig
) -> dict[int, tuple[float, float]]:
    if cache.is_file():
        raw = json.loads(cache.read_text("utf-8"))
        return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
    measured = measure_events(events, wav, sr, cfg)
    fits = {index: (fit.log_b, fit.r2) for index, fit in measured.items()}
    cache.write_text(
        json.dumps({str(k): [v[0], v[1]] for k, v in fits.items()}, indent=1) + "\n",
        encoding="utf-8",
    )
    return fits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--events-cache", type=Path, default=None)
    parser.add_argument("--fit-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--check-clips", type=int, default=3)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    events_cache = args.events_cache or (data_root / "models" / "q6_full_dev_cache")
    fit_cache = args.fit_cache or (data_root / "models" / "n5_fit_cache")
    fit_cache.mkdir(parents=True, exist_ok=True)

    cfg = GuitarConfig()
    session = SessionConfig()
    priors = build_oof_priors(data_home, cfg)
    variants = build_variants()
    wound = fit_wound_model()
    print(
        f"wound model: core = {wound.core_intercept_in:.5f} + {wound.core_slope:.4f}*gauge, "
        f"packing = {wound.packing:.3f}",
        flush=True,
    )

    clips = dev_clips(data_home)
    if args.limit:
        clips = clips[: args.limit]
    missing = [c for c in clips if not (events_cache / f"{c}.ensemble.json").is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} clips have no banked ensemble events (e.g. {missing[0]}); "
            "run scripts/eval/q6_full_dev.py first"
        )
    print(f"n5 table mismatch: {len(clips)} clips x {len(variants)} variants", flush=True)

    deltas: dict[str, list[float]] = {v["label"]: [] for v in variants}
    applied_total: dict[str, int] = {v["label"]: 0 for v in variants}
    base_tab: list[float] = []
    started = time.perf_counter()

    for position, track_id in enumerate(clips, start=1):
        events = [
            _event_from_json(item)
            for item in json.loads((events_cache / f"{track_id}.ensemble.json").read_text("utf-8"))
        ]
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        fits = load_fits(fit_cache / f"{track_id}.fits.json", events, wav, int(sr), cfg)
        if position <= args.check_clips:
            self_check(events, wav, int(sr), fits, cfg)
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        prior = priors[track_id[:2]]
        ordered = sorted(events, key=lambda event: event.onset_s)

        base_metrics, _ = _score(ordered, gold, cfg=cfg, session=session, prior=prior)
        base_tab.append(base_metrics["tab_f1"])
        for variant in variants:
            moved, applied = apply_banked(
                ordered,
                fits,
                variant["model"],
                cfg,
                weight=WEIGHT,
                min_r2=MIN_R2,
                sigma=variant["sigma"],
            )
            metrics, _ = _score(moved, gold, cfg=cfg, session=session, prior=prior)
            deltas[variant["label"]].append(metrics["tab_f1"] - base_metrics["tab_f1"])
            applied_total[variant["label"]] += applied
        if position % 10 == 0 or position == len(clips):
            elapsed = (time.perf_counter() - started) / 60.0
            shipped_mean = float(np.mean(deltas["offset+0.00"]))
            print(
                f"  [{position}/{len(clips)}] shipped-table delta {shipped_mean:+.4f} "
                f"({elapsed:.1f} min)",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        values = np.asarray(deltas[variant["label"]], dtype=np.float64)
        ci = bootstrap_ci(values, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        rows.append(
            {
                "label": variant["label"],
                "axis": variant["axis"],
                "offset": variant["offset"],
                "sigma": variant["sigma"],
                "note": variant["note"],
                "log_b0": {str(k): v for k, v in variant["model"].log_b0.items()},
                "delta": float(values.mean()),
                "lo95": ci.lower,
                "hi95": ci.upper,
                "applied": applied_total[variant["label"]],
            }
        )

    summary = {
        "frozen_config": {"weight": WEIGHT, "min_r2": MIN_R2, "sigma": SIGMA},
        "clips": len(clips),
        "baseline_tab_f1": float(np.mean(base_tab)),
        "wound_model": {
            "core_intercept_in": wound.core_intercept_in,
            "core_slope": wound.core_slope,
            "packing": wound.packing,
        },
        "variants": rows,
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nbaseline Tab F1 {summary['baseline_tab_f1']:.4f}")
    print(f"{'variant':<28}{'delta':>9}{'lo95':>9}{'hi95':>9}  note")
    for row in rows:
        print(
            f"{row['label']:<28}{row['delta']:+9.4f}{row['lo95']:+9.4f}{row['hi95']:+9.4f}  "
            f"{row['note']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
