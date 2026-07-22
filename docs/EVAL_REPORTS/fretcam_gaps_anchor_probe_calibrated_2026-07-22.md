# F7 corrected: cache-only calibrated GAPS hand-centroid anchor probe

**Date:** 2026-07-22
**Status:** **POSITIVE EVIDENCE for the later M4 bridge verdict.** The cache-only
result does not bypass FretCam's controlled-live acceptance contract or
authorize integration.

## Fixed protocol

- Corpus: public GAPS clean-12; gold-pitch ambiguous-note lattice decoded with
  A14's frozen mirrored cluster Viterbi (the comparator's banked audio
  mechanism).
- Video: rich cache only (`rawcv.c0.25.pkl`); no inference, download, or
  training.
- Anchor: cached predictions + `HandSample` through F2b's `calibrate_board` and
  `compute_position_anchor`; use the orientation-aware nonlinear fret map when
  available and the rule-of-18 fret-12 body-joint fallback otherwise;
  `N=max(1,floor(center_fret))`; window `[N-1,N+4] union {0}`.
- Timestamp: nearest cached frame within ±60 ms of `onset-30 ms` (the cache
  contains onset-near frames, not a purpose-built pre-onset sample stream).
- Comparator, window, clips, timestamp, cache, and audio decoder are unchanged
  from the superseded run. The only experimental change is the F2b coordinate
  correction.

Script: `fretcam/src/fretcam/gaps_anchor_probe.py`. Reproduce from the repo root:

```powershell
$env:PYTHONPATH = ((Resolve-Path 'fretcam/src').Path + ';' + (Resolve-Path 'tabvision').Path)
tabvision/.venv/Scripts/python.exe -m fretcam.gaps_anchor_probe
```

## Result

- **P(gold fret in window | audio wrong, anchor present) = 1195/1566 = 0.763**
  (Wilson 95% CI **0.741–0.783**).
- This is **+0.478** versus A14's 0.285 anti-enrichment reference and **+0.048**
  versus the corrected anchor marginal **0.715**.
- The primary corrected conditional is **−0.015** versus the 0.778 audio-prior
  scale comparator. The current decoder's actual audio prior is
  **7959/10182 = 0.782**, reproducing 0.778 within +0.004.
- Audio-wrong anchor coverage is **1566/2223 = 0.704**; all-ambiguous coverage
  is **7777/10182 = 0.764**.

The 95% interval is wholly above the frozen 0.285 comparator, and the corrected
conditional is also above the anchor marginal. This is strong cache-only
evidence for taking a controlled-live signal to F8 after L2. It is not
authorization to write integration code.

## Wrong-audio discrimination diagnostic

| gold fret in window | audio choice in window | notes | share | interpretation |
|---|---|---:|---:|---|
| yes | no | 639 | 0.408 | potential rescue |
| yes | yes | 556 | 0.355 | no discrimination |
| no | yes | 255 | 0.163 | favors wrong choice |
| no | no | 116 | 0.074 | no usable support |

The gold-only share exceeds the wrong-choice-only share by **0.245** (639 versus
255 notes). This is a diagnostic, not an end-to-end fusion effect estimate.

## Geometry correction audit

| Run | gold in window / wrong-audio anchors | Primary | vs 0.285 | vs marginal | boundary-clipped anchors |
|---|---:|---:|---:|---:|---:|
| Superseded `canonical_x × 24` | 387 / 1566 | 0.247 | −0.038 | −0.135 | 1654 / 7777 (0.213) |
| Corrected calibrated/fret-12 | 1195 / 1566 | **0.763** | **+0.478** | **+0.048** | 651 / 7777 (0.084) |

- Corrected anchor paths: calibrated fret map **1,274**; rule-of-18 fret-12
  fallback **6,503**; missing board calibration **0**.
- Selected-frame lag relative to the intended pre-onset target: median
  **+25.5 ms**, range −41.6 to +52.7 ms. The median selected frame is therefore
  about **−4.5 ms** from onset.
- The parser/decoder still yields 10,182 ambiguous notes rather than A14's
  banked 10,072 while reproducing the audio prior within 0.4 percentage points.
  The frozen A14 report remains the comparator.

The 0.516 absolute flip is therefore attributable to the already-approved F2b
coordinate repair, not threshold, corpus, decoder, window, or timestamp tuning.
The older 0.247 report remains preserved as superseded evidence.

## Per-clip breakdown

| clip | ambiguous | audio wrong | anchors on wrong | gold in window | rate |
|---|---:|---:|---:|---:|---:|
| 027_Zpswc | 1443 | 362 | 362 | 302 | 0.834 |
| 031_vpswc | 827 | 119 | 108 | 94 | 0.870 |
| 043_bc1wc | 1352 | 334 | 153 | 90 | 0.588 |
| 063_bV1wc | 824 | 243 | 74 | 13 | 0.176 |
| 104_xf1wc | 398 | 82 | 81 | 75 | 0.926 |
| 118_VD1wc | 768 | 55 | 55 | 22 | 0.400 |
| 142_GD1wc | 663 | 113 | 101 | 93 | 0.921 |
| 179_pM1wc | 484 | 71 | 71 | 53 | 0.746 |
| 212_y41wc | 886 | 279 | 7 | 2 | 0.286 |
| 235_Ny1wc | 1471 | 462 | 462 | 378 | 0.818 |
| 294_BSswc | 423 | 36 | 36 | 34 | 0.944 |
| 341_1M1wc | 643 | 67 | 56 | 39 | 0.696 |

## Verdict

Bank this as **positive cache-only F7 evidence**. Do not tune the window,
orientation, clip set, or confidence threshold against the result. L1/L2 remain
the controlled-live acceptance gate. F8 is now blocked only on an L2 pass and
must stop for user sign-off before any TabVision integration code.
