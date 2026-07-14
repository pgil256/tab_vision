# Correct-pitch / wrong-string Phase 0 benchmark

High-resolution audio events are cached once. Development numbers are out-of-fold: each of players 00-04 is decoded with priors trained on the other four. Player 05 is the untouched final confirmation. Video and the retired melodic prior are disabled.

Primary Tab F1 is the mean of standard per-clip Tab F1; micro Tab F1 is included as a cross-check. Confidence intervals are paired, clip-stratified 10,000-resample bootstraps.

## Development: leave-one-player-out players 00-04

| condition | solo Tab F1 | comp Tab F1 | all Tab F1 | micro | ambiguous top-1 | top-3 | same-pitch wrong rate | AUC | ECE | delta vs production [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.3202 | 0.4576 | 0.3889 | 0.4378 | 0.4879 | 0.9965 | 0.5121 | 0.6705 | 0.3131 | -0.1692 [-0.1928, -0.1463] |
| `position_only` | 0.5275 | 0.5678 | 0.5477 | 0.5515 | 0.6454 | 0.9967 | 0.3546 | 0.6585 | 0.2214 | -0.0104 [-0.0175, -0.0035] |
| `production_equivalent` | 0.5460 | 0.5702 | 0.5581 | 0.5577 | 0.6548 | 0.9967 | 0.3452 | 0.6147 | 0.2564 | baseline |
| `mode_specific` | 0.5523 | 0.5533 | 0.5528 | 0.5525 | 0.6480 | 0.9966 | 0.3520 | 0.6365 | 0.2596 | -0.0053 [-0.0172, +0.0067] |

## Final confirmation: held-out player 05

| condition | solo Tab F1 | comp Tab F1 | all Tab F1 | micro | ambiguous top-1 | top-3 | same-pitch wrong rate | AUC | ECE | delta vs production [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.2183 | 0.5051 | 0.3617 | 0.4051 | 0.4153 | 0.9961 | 0.5847 | 0.6016 | 0.3734 | -0.2509 [-0.3100, -0.1905] |
| `position_only` | 0.5230 | 0.6816 | 0.6023 | 0.6221 | 0.6698 | 0.9985 | 0.3302 | 0.6365 | 0.2368 | -0.0103 [-0.0287, +0.0071] |
| `production_equivalent` | 0.5418 | 0.6834 | 0.6126 | 0.6279 | 0.6770 | 0.9986 | 0.3230 | 0.6241 | 0.2640 | baseline |
| `mode_specific` | 0.6074 | 0.6447 | 0.6260 | 0.6090 | 0.6576 | 0.9986 | 0.3424 | 0.6888 | 0.2488 | +0.0134 [-0.0154, +0.0425] |
| `checked_in_production` | 0.5418 | 0.6834 | 0.6126 | 0.6279 | 0.6770 | 0.9986 | 0.3230 | 0.6241 | 0.2640 | +0.0000 [+0.0000, +0.0000] |

### Mode-specific prior deltas by target mode

| split | mode | mean delta | paired 95% CI |
|---|---|---:|---:|
| development OOF | solo | +0.0063 | [-0.0129, +0.0255] |
| development OOF | comp | -0.0168 | [-0.0319, -0.0026] |
| held-out 05 context | solo | +0.0656 | [+0.0286, +0.1034] |
| held-out 05 context | comp | -0.0387 | [-0.0839, +0.0043] |

Promotion is development-gated. The held-out per-mode rows are reported for confirmation context only and are not used to select a prior.

### Held-out production string confusion matrix

| reference \ predicted | 0 | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 158 | 0 | 0 | 0 | 0 |
| 1 | 17 | 541 | 325 | 3 | 0 | 0 |
| 2 | 0 | 76 | 1123 | 493 | 12 | 0 |
| 3 | 0 | 0 | 134 | 1302 | 539 | 14 |
| 4 | 0 | 0 | 0 | 158 | 1184 | 301 |
| 5 | 0 | 0 | 0 | 0 | 70 | 606 |

## Phrase oracle

Phrases: **586**; infeasible gold-anchor phrases: **1**; ambiguous pitch-matched notes: **7121**. Infeasible phrases count as no improvement rather than being dropped.

| baseline | one gold anchor, top-1 | lift | best of 3 | lift over anchored top-1 |
|---:|---:|---:|---:|---:|
| 0.6770 | 0.7384 | +0.0614 | 0.7950 | +0.0566 |

Refinement build gate (`>= +0.10`): **FAIL**. Multiple-alternative gate (`>= +0.05` over anchored top-1): **PASS**.

## Reproduction and provenance

```powershell
cd tabvision
& .\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase0 --data-home $HOME\.tabvision\data\guitarset --output-dir ..\docs\EVAL_REPORTS
```

Artifact and dataset provenance: `string_assignment_phase0_2026-07-14_provenance.json`. The grouped diagnostic summary is checked in as the sibling summary CSV. The raw note CSV is generated locally and git-ignored because it is reproducible and approximately 26 MB.
