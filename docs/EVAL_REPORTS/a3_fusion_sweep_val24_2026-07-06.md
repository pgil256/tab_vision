# A3/A4 — fusion-constants sweep (val24)

**Date:** 2026-07-06 · **Branch:** `v1.1/a3-a4-fusion-sweep`

**Status: harness validated, NO default changed. The real movers are a
domain-sensitive trap; A4 is a wash.** The baseline reproduces the roadmap
val24 numbers exactly (0.4820 / 0.7951), validating the in-process harness.

**Key reading (do not adopt any of these without the gate):**
- The three biggest movers — **`LOW_FRET_BIAS=0.0`** (+0.0386; single-line
  0.4820→**0.5728**), **`FRET_PRIOR_WEIGHT=1.5`** (+0.0297; 0.5306/0.8060, both
  tiers up), **prior `power=3.0`** (+0.0297, identical) — are all the *same
  lever*: **trust the `guitarset-v1` prior more** (stop `LOW_FRET_BIAS` from
  fighting it, or up-weight/sharpen it). But **val24 IS GuitarSet**, the
  in-domain corpus that prior was built from, so this is *guaranteed* to help
  here. On GAPS the same prior is a measured **−0.138** (A2). These candidates
  are almost certainly **GuitarSet-overfit and would fail the GAPS clean-12
  no-regression gate** — the exact domain-sensitivity hard-gate from the
  2026-07-02 A2 negative. Flagged, NOT adopted.
- **A4 gap-decay (`TRANSITION_GAP_TAU`) is a wash** — best is TAU=1.0 at
  **+0.0005**, most values slightly negative. Short TAU trades single-line for
  strummed (TAU=0.25: single-line 0.4545 / strummed 0.8094) but nets negative.
  Keep the default `inf` (off). Banked negative for the A4 hypothesis.
- **Domain-neutral candidates** (small, roadmap-consistent, worth the gate):
  **`OPEN_STRING_BONUS=0.0`** lifts strummed 0.7951→**0.8140** with single-line
  flat (the docstring admitted the bonus was calibrated against a now-absent
  vision floor — this confirms it slightly hurts strummed); `SPAN_NORM=6.0`
  +0.0066; `CHORD_MAX_GAP_S=0.04` +0.0055.
- **Prior `alpha` is inert** (all values identical) — the power-sharpening
  washes it out; drop it from future sweeps.

**Next step (not done here — measurement discipline):** the domain-neutral
`OPEN_STRING_BONUS=0.0` candidate is the safest; take it (and, if desired, the
prior-trust movers) through a 60-clip player-05 lower-95 confirm **and** a GAPS
clean-12 per-clip no-regression before any default change. The prior-trust
movers must clear GAPS or they don't ship — expect them to fail it.

Config: `highres` + `guitarset-v1` prior, **no sequence prior**, splits `validation,test`. 1-D marginal sweeps around defaults.
**Caveat:** the shipped default couples an A15 sequence prior with the position
prior; this sweep omits it, so any winner must be re-confirmed under the coupled
default too.

**Baseline** (single-line | strummed | aggregate): **0.4820 | 0.7951 | 0.6386** (roadmap val24 baseline 0.4820 / 0.7951 — harness validation).

Δ columns are vs this baseline aggregate; **best** marks the top aggregate per axis.

## `POSITION_SHIFT_COST` (default 2.5)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 1.5 | 0.4838 | 0.7959 | 0.6398 | +0.0013 |
| 2.0 | 0.4855 | 0.7933 | 0.6394 | +0.0008 |
| 2.5 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 3.0 | 0.4848 | 0.7972 | 0.6410 | +0.0025 |
| 3.5 | 0.4872 | 0.7940 | 0.6406 | +0.0020 |
| 4.0 | 0.4922 | 0.7903 | 0.6413 | +0.0027 **best** |

## `FRET_PRIOR_WEIGHT` (default 1.0)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.5 | 0.3830 | 0.6840 | 0.5335 | -0.1051 |
| 0.75 | 0.4516 | 0.7445 | 0.5980 | -0.0405 |
| 1.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 1.25 | 0.5128 | 0.7911 | 0.6520 | +0.0134 |
| 1.5 | 0.5306 | 0.8060 | 0.6683 | +0.0297 **best** |
| 2.0 | 0.5414 | 0.7939 | 0.6677 | +0.0291 |

## `OPEN_STRING_BONUS` (default 0.5)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4846 | 0.8140 | 0.6493 | +0.0107 **best** |
| 0.25 | 0.4846 | 0.8086 | 0.6466 | +0.0081 |
| 0.5 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 0.75 | 0.4754 | 0.7586 | 0.6170 | -0.0216 |
| 1.0 | 0.4648 | 0.7317 | 0.5983 | -0.0403 |

## `SAME_STRING_BONUS` (default 0.5)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4796 | 0.8010 | 0.6403 | +0.0017 |
| 0.25 | 0.4796 | 0.7986 | 0.6391 | +0.0005 |
| 0.5 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 0.75 | 0.4918 | 0.7916 | 0.6417 | +0.0032 **best** |
| 1.0 | 0.5002 | 0.7724 | 0.6363 | -0.0023 |

## `LOW_FRET_BIAS` (default 0.1)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.0 | 0.5728 | 0.7815 | 0.6771 | +0.0386 **best** |
| 0.05 | 0.5475 | 0.7746 | 0.6611 | +0.0225 |
| 0.1 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 0.2 | 0.4159 | 0.7260 | 0.5710 | -0.0676 |
| 0.3 | 0.3247 | 0.6435 | 0.4841 | -0.1545 |

## `HAND_SPAN_BARRIER` (default 5.0)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 2.5 | 0.4811 | 0.7958 | 0.6385 | -0.0001 |
| 5.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 7.5 | 0.4795 | 0.7951 | 0.6373 | -0.0012 |
| 10.0 | 0.4795 | 0.7951 | 0.6373 | -0.0012 |

## `SPAN_NORM` (default 12.0)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 6.0 | 0.5031 | 0.7872 | 0.6451 | +0.0066 **best** |
| 9.0 | 0.4884 | 0.7949 | 0.6416 | +0.0031 |
| 12.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 18.0 | 0.4855 | 0.7959 | 0.6407 | +0.0021 |

## `CHORD_MAX_GAP_S` (default 0.08)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.04 | 0.4863 | 0.8017 | 0.6440 | +0.0055 **best** |
| 0.06 | 0.4834 | 0.8017 | 0.6426 | +0.0040 |
| 0.08 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 0.1 | 0.4802 | 0.7870 | 0.6336 | -0.0049 |
| 0.12 | 0.4733 | 0.7610 | 0.6171 | -0.0214 |

## `TRANSITION_GAP_TAU` (default inf)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| inf | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 4.0 | 0.4843 | 0.7919 | 0.6381 | -0.0004 |
| 2.0 | 0.4834 | 0.7915 | 0.6375 | -0.0011 |
| 1.0 | 0.4831 | 0.7949 | 0.6390 | +0.0005 **best** |
| 0.5 | 0.4762 | 0.7986 | 0.6374 | -0.0012 |
| 0.25 | 0.4545 | 0.8094 | 0.6320 | -0.0066 |

## prior `power` (default 2.0)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 1.0 | 0.3830 | 0.6840 | 0.5335 | -0.1051 |
| 1.5 | 0.4516 | 0.7445 | 0.5980 | -0.0405 |
| 2.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 2.5 | 0.5128 | 0.7911 | 0.6520 | +0.0134 |
| 3.0 | 0.5306 | 0.8060 | 0.6683 | +0.0297 **best** |

## prior `alpha` (default 1.0)

| value | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.5 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 1.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 1.5 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |
| 2.0 | 0.4820 | 0.7951 | 0.6386 | +0.0000 |

## Verdict

Best single-axis point: **LOW_FRET_BIAS=0.0** → aggregate 0.6771 (**+0.0386** vs baseline). Candidate only — needs the 60-clip lower-95 confirm + GAPS clean-12 no-regression before any default change.

