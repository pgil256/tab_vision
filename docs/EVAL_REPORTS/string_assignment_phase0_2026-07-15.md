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

Phase 0 baseline reproduction gate (`abs(delta) <= 1e-4`): **PASS** (expected `0.6126`, observed `0.6126`).

### Held-out production string confusion matrix

| reference \ predicted | 0 | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 158 | 0 | 0 | 0 | 0 |
| 1 | 17 | 541 | 325 | 3 | 0 | 0 |
| 2 | 0 | 76 | 1123 | 493 | 12 | 0 |
| 3 | 0 | 0 | 134 | 1302 | 539 | 14 |
| 4 | 0 | 0 | 0 | 158 | 1184 | 301 |
| 5 | 0 | 0 | 0 | 0 | 70 | 606 |

### Held-out audio-event metrics

| onset F1 | pitch F1 |
|---:|---:|
| 0.9302 | 0.9154 |

The segment/oracle transforms below operate only on string/fret assignment. The frozen baseline audio events are unchanged.

## Segment and fret-zone diagnostic ceilings

Each row chooses one gold state per track or fixed window. Fixed windows are cluster-safe; every selected position is a real candidate for the unchanged MIDI pitch. Impossible shifted candidates are dropped.

| oracle | ambiguous accuracy | lift | macro Tab F1 | lift | micro Tab F1 | dropped impossible |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.6770 | +0.0000 | 0.6126 | +0.0000 | 0.6279 | 0 |
| `offset_track` | 0.7566 | +0.0796 | 0.6828 | +0.0702 | 0.6954 | 52 |
| `offset_1s` | 0.9041 | +0.2271 | 0.8189 | +0.2063 | 0.8163 | 38 |
| `offset_2s` | 0.8687 | +0.1917 | 0.7850 | +0.1724 | 0.7885 | 68 |
| `offset_4s` | 0.8249 | +0.1479 | 0.7508 | +0.1382 | 0.7538 | 99 |
| `offset_8s` | 0.7881 | +0.1111 | 0.7189 | +0.1063 | 0.7226 | 80 |
| `offset_16s` | 0.7616 | +0.0845 | 0.6898 | +0.0772 | 0.7001 | 68 |
| `fret_zone_track` | 0.8437 | +0.1667 | 0.7690 | +0.1565 | 0.7665 | 36 |
| `fret_zone_1s` | 0.9760 | +0.2990 | 0.8881 | +0.2756 | 0.8740 | 9 |
| `fret_zone_2s` | 0.9594 | +0.2824 | 0.8734 | +0.2608 | 0.8606 | 14 |
| `fret_zone_4s` | 0.9309 | +0.2539 | 0.8528 | +0.2403 | 0.8385 | 42 |
| `fret_zone_8s` | 0.8881 | +0.2111 | 0.8199 | +0.2074 | 0.8027 | 29 |
| `fret_zone_16s` | 0.8645 | +0.1875 | 0.7917 | +0.1791 | 0.7831 | 25 |
| `joint_track` | 0.7469 | +0.0699 | 0.6918 | +0.0792 | 0.7035 | 448 |
| `joint_1s` | 0.9039 | +0.2269 | 0.8231 | +0.2105 | 0.8203 | 124 |
| `joint_2s` | 0.8662 | +0.1892 | 0.7890 | +0.1765 | 0.7921 | 191 |
| `joint_4s` | 0.8217 | +0.1446 | 0.7570 | +0.1444 | 0.7593 | 284 |
| `joint_8s` | 0.7804 | +0.1034 | 0.7235 | +0.1109 | 0.7241 | 268 |
| `joint_16s` | 0.7531 | +0.0761 | 0.6978 | +0.0852 | 0.7056 | 370 |

Phase 0 segment-signal gate (four-second joint lift `>= +0.10`): **PASS** (0.6770 -> 0.8217, +0.1446).

## Phrase oracle

Phrases: **586**; infeasible gold-anchor phrases: **1**; ambiguous pitch-matched notes: **7121**. Infeasible phrases count as no improvement rather than being dropped.

| baseline ambiguous | one gold anchor, top-1 | lift | best of 3 | lift over anchored top-1 | baseline Tab F1 | anchored Tab F1 | best-of-3 Tab F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6770 | 0.7384 | +0.0614 | 0.7950 | +0.0566 | 0.6126 | 0.6833 | 0.7529 |

Refinement build gate (`>= +0.10`): **FAIL**. Multiple-alternative gate (`>= +0.05` over anchored top-1): **PASS**.

## Reproduction and provenance

```powershell
cd tabvision
& .\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase0 --data-home $HOME\.tabvision\data\guitarset --output-dir ..\docs\EVAL_REPORTS
```

Artifact and dataset provenance: `string_assignment_phase0_2026-07-15_provenance.json`. The grouped diagnostic summary includes candidate-rank, displacement, player/mode, pitch, style, and clip slices. The raw stable note table is generated locally and git-ignored because it is reproducible and approximately 30 MB. Runtime/peak-memory observations, exact command/environment, cache identities, and deterministic output hashes live in provenance.
