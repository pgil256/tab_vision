# F7: cache-only GAPS hand-centroid anchor probe

**Date:** 2026-07-22
**Status:** **CLOSED-NEGATIVE for the GAPS bridge probe.** This does not close
FretCam because GAPS uses the explicitly different uncontrolled-footage capture
contract.

## Fixed protocol

- Corpus: public GAPS clean-12; gold-pitch ambiguous-note lattice decoded with
  A14's frozen mirrored cluster Viterbi (the comparator's banked audio
  mechanism).
- Video: rich cache only (`rawcv.c0.25.pkl`); no inference, download, or
  training.
- Anchor: cached `HandSample` + cached homography through
  `compute_neck_anchor`; `N=max(1,floor(center_fret))`; window
  `[N-1,N+4] union {0}`.
- Timestamp: nearest cached frame within +/-60 ms of `onset-30 ms` (the cache
  contains onset-near frames, not a purpose-built pre-onset sample stream).

Script: `fretcam/src/fretcam/gaps_anchor_probe.py`. Reproduce from the repo root
using the existing TabVision evaluation environment (no install required):

```powershell
$env:PYTHONPATH = (Resolve-Path 'fretcam/src').Path
tabvision/.venv/Scripts/python.exe -m fretcam.gaps_anchor_probe
```

## Result

- **P(gold fret in window | audio wrong, anchor present) = 387/1566 = 0.247**
  (Wilson 95% CI 0.226-0.269).
- This is **-0.038** versus A14's 0.285 anti-enrichment reference and below the
  anchor marginal **0.382**.
- Current audio prior = **7959/10182 = 0.782** versus the requested 0.778
  reference (+0.004).
- Audio-wrong anchor coverage = **1566/2223 = 0.704**; all-ambiguous coverage =
  **7777/10182 = 0.764**.

The conditional is lower than both the 0.285 comparator and the anchor's own
marginal. The cached centroid signal is therefore anti-enriched where audio
fails; it is not evidence for wiring this GAPS signal into fusion.

## Wrong-audio discrimination diagnostic

| gold fret in window | audio choice in window | notes | share | interpretation |
|---|---|---:|---:|---|
| yes | no | 204 | 0.130 | potential rescue |
| yes | yes | 183 | 0.117 | no discrimination |
| no | yes | 426 | 0.272 | favors wrong choice |
| no | no | 753 | 0.481 | no usable support |

## Cache and geometry diagnostics

- Selected-frame lag relative to the intended pre-onset target: median
  **+25.5 ms**, range -41.6 to +52.7 ms. The median selected frame is thus
  about **-4.5 ms** from onset.
- Centroids clipped to a neck boundary: **1654/7777 = 0.213** (nut 633;
  bridge 1021).
- The current parser/decoder yields 10,182 ambiguous decoded notes rather than
  A14's banked 10,072, while reproducing its audio prior within 0.4 pp. Counts
  in this report are from the current checkout and the same local public cache;
  the comparator remains the frozen A14 report. Current `fuse` has evolved
  since A14, so this probe intentionally uses A14's decoder instead of claiming
  parity with today's implementation.

## Per-clip breakdown

| clip | ambiguous | audio wrong | anchors on wrong | gold in window | rate |
|---|---:|---:|---:|---:|---:|
| 027_Zpswc | 1443 | 362 | 362 | 68 | 0.188 |
| 031_vpswc | 827 | 119 | 108 | 41 | 0.380 |
| 043_bc1wc | 1352 | 334 | 153 | 87 | 0.569 |
| 063_bV1wc | 824 | 243 | 74 | 13 | 0.176 |
| 104_xf1wc | 398 | 82 | 81 | 12 | 0.148 |
| 118_VD1wc | 768 | 55 | 55 | 29 | 0.527 |
| 142_GD1wc | 663 | 113 | 101 | 33 | 0.327 |
| 179_pM1wc | 484 | 71 | 71 | 21 | 0.296 |
| 212_y41wc | 886 | 279 | 7 | 2 | 0.286 |
| 235_Ny1wc | 1471 | 462 | 462 | 50 | 0.108 |
| 294_BSswc | 423 | 36 | 36 | 3 | 0.083 |
| 341_1M1wc | 643 | 67 | 56 | 28 | 0.500 |

## Verdict

Bank this as a **negative for uncontrolled GAPS footage**. Do not tune the
window, orientation, clip set, or confidence threshold against this result.
The only valid FretCam reopen path remains a new controlled-live capture
contract; the build path itself is still paused at F2's independent 2/3-clip
failed gate.
