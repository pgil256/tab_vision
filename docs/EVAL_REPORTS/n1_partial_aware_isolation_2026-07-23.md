# N1 — partial-aware isolation: coverage 2.6×, +0.0182 over the shipped mode

Accuracy-loop N1 (coverage extension for the Q6 physics channel). The channel
is 0.92 accurate on the notes it reaches but reached only 8.3% of detections,
so coverage — not accuracy — bounded its value.

## The diagnostic corrected the plan

The proposed program said fit success was the cheap lever ("only 44% of
isolated notes produce a usable fit"). Counting every rejection reason
separately over 60 clips / 12,733 events says otherwise:

| rejection reason | share |
|---|---:|
| **not_isolated** | **87.81%** |
| low_r2 | 2.23% |
| too_short | 1.19% |
| too_few_partials | **0.00%** |
| fit_failed | **0.00%** |
| applied | 5.24% |

Partial finding never fails (9-10 partials located essentially always) and
fits never fail. The earlier 44% figure conflated unambiguous notes that were
fitted and then dropped later at the matrix stage. **Isolation is the entire
problem.**

## The change

A neighbouring note only ruins the measurement if its partials *collide* with
the target's. `partial_aware` computes the interferers' harmonic frequencies,
drops the target partials within `3/T` Hz of one (under the `~4/T` main lobe
of a Hann window, so genuinely unresolvable), and fits the survivors. The
located peak is re-checked against blocked bands, so a loud interferer inside
the search window cannot be measured as this note's partial.

`strict` remains the default and is bit-identical to shipped v1; all 26
pre-existing tests pass unchanged.

## Result — full development set, 300 clips

| arm | Tab F1 | Δ vs baseline [lo-95, hi-95] | solo Δ | comp Δ | coverage |
|---|---:|---|---:|---:|---:|
| baseline | 0.6031 | — | — | — | — |
| strict (shipped v1) | 0.6474 | +0.0443 [+0.0339, +0.0555] | +0.0860 | +0.0026 | 8.26% |
| **pa4** | **0.6660** | **+0.0629 [+0.0481, +0.0792]** | **+0.1139** | **+0.0120** | **21.69%** |
| pa6 | 0.6656 | +0.0625 [+0.0484, +0.0781] | +0.1130 | +0.0120 | 17.51% |
| pa8 | 0.6625 | +0.0594 [+0.0462, +0.0739] | +0.1087 | +0.0100 | 13.20% |

**Head-to-head: pa6 − strict = +0.0182 [+0.0111, +0.0256]** — the interval
excludes zero, so the improvement over the shipped mode is significant, not
just a larger point estimate.

Coverage rises **8.26% → 21.69%** (2.6×) and solo gains **+0.0860 → +0.1139**.
Onset F1 is 0.9182 in every arm, bit-identical, as it must be.

Decomposition against `strict`, still one-for-one:

| bucket | strict | pa4 | Δ |
|---|---:|---:|---:|
| correct | 31,956 | 32,602 | **+646** |
| wrong_position_same_pitch | 11,659 | 11,013 | **−646** |
| pitch_off / timing_only / missed_onset / extra_detection | — | — | **0 each** |

## The 20-clip pilot was wrong, and this is the more useful finding

The pilot reported comp **regressing** (+0.0000 → −0.0347) and concluded this
was a solo-only improvement with a chord-tier cost. On 300 clips comp
**improves** (+0.0026 → +0.0120). The regression was a 10-comp-clip artifact
and the sign flipped.

The pilot also suggested `min_clean_partials` mattered, trading solo against
comp monotonically. On the full set pa4 and pa6 are statistically
indistinguishable (+0.0629 vs +0.0625) and pa8 is only slightly worse — the
threshold is close to immaterial. That apparent sensitivity was noise too.

**Recommended arm: pa4**, i.e. no extra contaminated-note gate beyond the
estimator's existing `MIN_PARTIALS`. It matches pa6 within noise, and adding
a threshold that buys nothing is a parameter to defend for no gain.

Worth stating plainly because it cuts against my own earlier write-up: a
20-clip pilot was enough to detect the coverage change but not to get the
sign of a per-tier effect right, and I reported the wrong conclusion before
the full run corrected it.

## Gate status

- **Full-dev OOF: PASSED** — +0.0629 [+0.0481, +0.0792], and significant
  against the shipped mode.
- **GAPS cross-domain: satisfied by construction, unchanged.** The domain
  guard is untouched, so classical/electric/capo/alt-tuning still abstain and
  the mode never runs there.
- **player-05 confirmation: NOT RUN.** v1 was registered only after the
  sealed hold-out confirmed it; v2 should clear the same bar, and player-05 is
  user-gated. Nothing is registered and `auto` is unchanged.

## Honest limits

- The threshold was chosen after seeing full-dev results. It barely matters
  (all three arms beat `strict` significantly), so the *decision* to adopt
  partial-aware does not rest on it — but the specific +0.0629 for pa4 is
  mildly optimistic.
- Still GuitarSet. Contamination behaviour on denser or differently-voiced
  material is unmeasured.
- `_overlapping` is O(n²) per clip; the full sweep took ~25 min where the
  strict path takes seconds. Fine offline, but it needs an interval index
  before this could run inside the 5-min-per-60s latency budget.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n1_coverage_diagnostic.py --clips 60 \
  --json ../docs/EVAL_REPORTS/n1_coverage_diagnostic_2026-07-23.json
python scripts/eval/n1_isolation_sweep.py --all-dev \
  --json ../docs/EVAL_REPORTS/n1_isolation_fulldev_2026-07-23.json
```
