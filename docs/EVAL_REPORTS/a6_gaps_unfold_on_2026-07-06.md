# A6: GAPS test-22 (repeat unfold ON)

**Date:** 2026-07-06
**Branch:** `v1.1/a6-gaps-gold-coverage`
**Status:** **A6 lands — the honest GAPS single-line Tab F1 is 0.6969, not 0.6468.**
Controlled A/B (highres, `--position-prior none`, `--splits test`; predictions
cache-shared, only the parser gold differs via `TABVISION_GAPS_NO_UNFOLD`):

| | gold notes | Tab F1 mean | lower-95 | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|
| **unfold OFF** (pre-A6) | 14,699 | **0.6468** | 0.5734 | 0.8277 | 0.8185 |
| **unfold ON** (A6) | 16,079 | **0.6969** | 0.6256 | 0.8796 | 0.8703 |
| **Δ** | **+1,380 (+9.4%)** | **+0.0501** | +0.0522 | +0.0519 | +0.0518 |

The OFF run reproduces the banked baseline **0.6468 exactly**, validating the
harness and the env guard (only the gold changed between runs).

**Honest interpretation (do not misread the +0.05 as the model improving):**
A6 added 1,380 gold notes — the repeat/volta traversals the model was already
transcribing but getting *no credit for* (they had no score counterpart and
counted as false positives). Unfolding turns those into true positives, which
is why all three F1s rise together. This is a **coverage-accounting
correction**: the honest GAPS single-line number is 0.6969, and GAPS-tuned
work should measure against that. It restores gold on the 11/14 test-22 repeat
clips whose repeat structure the unfold reproduces (the 3 nonstandard voltas
fall back safely; the 8 non-repeat clips are unchanged). See the OFF companion
`a6_gaps_unfold_off_2026-07-06.md`.

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 22 | 16079 | 0.6969 | 0.6256 | 0.45 | pass | 0.8796 | 0.8703 |
| clean_acoustic_strummed | 0 | 0 | — | — | 0.60 | missing | — | — |
| clean_electric | 0 | 0 | — | — | 0.90 | missing | — | — |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 video-assisted target; audio-only is string-resolution-limited, like single-line Tab F1 (SPEC §1.4.1).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 22 | 0.6821 | 0.6324 |
| clean_acoustic_strummed | 0 | — | — |
| clean_electric | 0 | — | — |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_acoustic_single_line | GAPS | 22 | 0.6969 | 0.8796 | 0.8703 |

## Methodology

- Manifest: `data\eval\gaps.toml`
- Audio backend: `highres`
- Position prior: `none`
- Eval-harness SHA: `b46c175`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

