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
- ~~`_overlapping` is O(n²) per clip ... needs an interval index before this
  could run inside the 5-min-per-60s latency budget.~~ **Retracted — this was
  asserted without measurement and is wrong on both counts. See the latency
  appendix below.**

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n1_coverage_diagnostic.py --clips 60 \
  --json ../docs/EVAL_REPORTS/n1_coverage_diagnostic_2026-07-23.json
python scripts/eval/n1_isolation_sweep.py --all-dev \
  --json ../docs/EVAL_REPORTS/n1_isolation_fulldev_2026-07-23.json
```


---

## Appendix — latency claim retracted (measured 2026-07-23)

The limits section above originally warned that partial-aware isolation
"needs an interval index before this could run inside the 5-min-per-60s
latency budget". That was asserted from reading the code, not measured, and
it is wrong twice over.

**Measured cost on the three densest dev clips** (SPEC §1.4 allows 300 s per
60 s clip; transcription alone is ~40 s):

| clip | events | audio | strict | partial_aware |
|---|---:|---:|---:|---:|
| 00_Rock2-85-F_comp | 802 | 45.2 s | 0.10 s/60s | **4.45 s/60s** |
| 04_SS1-68-E_comp | 784 | 42.3 s | 0.14 s/60s | **4.43 s/60s** |
| 00_Rock1-90-C#_comp | 654 | 32.0 s | 0.09 s/60s | **4.32 s/60s** |

**~4.4 s per 60 s of audio is 1.5% of the budget.** Partial-aware is ~45×
slower than strict in relative terms and irrelevant in absolute ones.

**The bottleneck is not the overlap scan.** `strict` rejects non-isolated
notes *before* fitting, so it runs an FFT on ~19% of events; `partial_aware`
fits nearly all of them. The extra spectral work dominates, not the O(n²)
neighbour search.

**The O(n²) is real but never material at realistic sizes.** Isolating the
scan on synthetic events:

| events | ≈ audio | scan cost | per 60 s audio |
|---:|---:|---:|---:|
| 800 | 1.6 min | 0.03 s | 0.02 s |
| 2,000 | 4.0 min | 0.31 s | 0.08 s |
| 5,000 | 10 min | 1.85 s | 0.18 s |
| 10,000 | 20 min | 17.10 s | 0.86 s |

The absolute cost grows quadratically while the budget grows linearly, so it
does eventually bite — but reaching even 10% of budget needs roughly 60,000
events, about two hours of continuously dense playing in one recording.

**No optimisation made.** An interval index would be premature against 1.5%
of budget, and any change to this path would invalidate the full-dev gate it
just passed. The measurement is recorded so a future session can skip the
question, and so the earlier claim does not propagate.
