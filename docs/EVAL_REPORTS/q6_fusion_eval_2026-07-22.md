# Q6 fusion integration — inharmonicity evidence lifts Tab F1

Accuracy-loop iteration 8 (ROI deep-dive §4.1). Gates A/B measured string
*classification*; the detected-notes probe showed it survives real onsets and
pitches. This measures the only thing that decides promotion: **Tab F1
through the real `fuse()`**, with the evidence folded in as a bounded
product-of-experts term beside the corpus prior.

Offline replay of the banked 20-clip ensemble cache (10 comp + 10 solo),
leave-one-player-out position prior, `guitarset-seq-v1` @ w=4.0. The
stiffness model is calibrated per fold from *other* players' gold notes.
**`auto` is untouched** — the module is new package code, not wired into any
default path.

## Result — CI-significant on every arm

| arm | Tab F1 | ΔTab F1 [lo-95, hi-95] | solo Δ [lo-95, hi-95] | onset F1 | pitch F1 |
|---|---:|---|---|---:|---:|
| baseline | 0.6773 | — | — | 0.9325 | 0.9131 |
| `w=0.5, r²≥0.50` | 0.7167 | +0.0394 [+0.0121, +0.0720] | +0.0789 [+0.0322, +0.1282] | 0.9325 | 0.9131 |
| **`w=1.0, r²≥0.50`** | **0.7298** | **+0.0525 [+0.0208, +0.0888]** | **+0.1050 [+0.0553, +0.1537]** | 0.9325 | 0.9131 |
| `w=0.5, r²≥0.70` | 0.7124 | +0.0351 [+0.0104, +0.0652] | +0.0702 [+0.0277, +0.1167] | 0.9325 | 0.9131 |

Every arm's lower bound is above zero. **Onset and pitch F1 are
bit-identical across all arms** — the channel rewrites `fret_prior` only and
cannot move a detection, which the unit tests assert by construction rather
than leaving to luck.

The solo lift of **+0.1050** matches the detected-notes probe's prediction of
"~+0.10 on the solo tier" almost exactly. That estimate was made before this
run, from coverage and per-note accuracy alone, so the agreement is a real
out-of-sample check on the reasoning rather than a fit.

## The decomposition is a one-for-one conversion

| bucket | baseline | w=1.0 | Δ |
|---|---:|---:|---:|
| correct | 1411 | 1477 | **+66** |
| wrong_position_same_pitch | 443 | 377 | **−66** |
| pitch_off | 132 | 132 | 0 |
| timing_only | 15 | 15 | 0 |
| missed_onset | 196 | 196 | 0 |
| extra_detection | 102 | 102 | 0 |

This is the cleanest bucket result in the program. 66 wrong-position errors
become correct notes and **nothing else changes at all** — exactly what a
string-assignment channel that cannot touch detection must look like. The
deep-dive's §6.3 rule ("a gain in the wrong bucket is a red flag for
leakage") is satisfied about as strictly as it can be.

66 fixed out of 213 covered notes is 31%, consistent with the decoder having
already been right on roughly two thirds of them.

## Coverage

| stage | count | share |
|---|---:|---:|
| detected events | 2,105 | 100% |
| isolated | 449 | 21.3% |
| fit succeeds (r² ≥ 0.50) | 213 | 10.1% |
| evidence applied | 213 | 10.1% |

The channel abstains on ~90% of notes and contributes essentially nothing on
strummed material. That is the design, not a defect: it fires only where the
physics is readable.

## Why `w=1.0` rather than a smaller weight

The weight is an exponent on log-probabilities in
`combine_candidate_evidence`, so `w=1.0` gives the channel parity with the
corpus prior — not dominance. It is already bounded three ways: it abstains
below the fit threshold, it abstains on non-isolated notes, and its
distribution is a normalised Gaussian over candidates rather than a hard
assignment. Within that, more weight is better on this set.

## What this is not

- **20 clips.** A dev-set pilot, not a ship gate. The real gates are full-dev
  OOF, GAPS clean-12 strict no-regression, then player-05 — none of which
  have run.
- **Weight and r² threshold were chosen on the reported set.** Four arms is
  mild selection, but the honest reading is that +0.0525 is the optimistic
  end. A held-out weight selection would very likely land lower.
- **Calibration is GuitarSet-specific.** All five players use similar
  acoustic guitars, and the stiffness table is fitted from their gold notes.
  A user's own instrument needs its own `B0`, which §4.1's per-session EM
  bootstrap sketches and this work does not implement. **This is the single
  biggest gap between "works on GuitarSet" and "works on your recording."**
- **Aggregate is carried by solo.** Comp barely moves, so the headline number
  on a strum-heavy set will be far smaller than +0.0525.

## Files

- `tabvision/tabvision/fusion/inharmonicity.py` — the evidence channel
  (package code, mypy-clean, 10 unit tests).
- `tabvision/scripts/eval/q6_fusion_eval.py` — this evaluation.
- `tabvision/tests/unit/test_inharmonicity_evidence.py`.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q6_fusion_eval.py \
  --json ../docs/EVAL_REPORTS/q6_fusion_eval_2026-07-22.json
```
