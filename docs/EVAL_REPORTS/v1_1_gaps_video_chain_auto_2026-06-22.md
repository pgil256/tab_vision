# v1.1 chunk-5 — GAPS video real-chain (video half)

**Date:** 2026-06-22
**Branch:** `v1.1/oracle-string-resolution`
**Status:** Video half — the chunk-3 confidence-gated CV chain (YOLO-OBB + MediaPipe + homography + fusion) run over the GAPS clean-12, with source video acquired via yt-dlp and aligned to the GAPS audio crop by onset-envelope cross-correlation.

GAPS is non-commercial, offline-eval-only: source media stays local and is never committed or redistributed. Clean-12 = the >=80% gold-coverage standard-tuning test clips from the chunk-5 audio half.

## 1. Video<->audio crop-offset alignment (net-new)

GAPS audio is a crop of the YouTube upload, so frame time != gold time. Per clip, the offset is recovered by cross-correlating onset-strength envelopes (`video_time = gold_onset + offset`); the peak ratio is the correlation peak over the best competitor outside a +-5-frame guard band.

| Clip | Gold | offset (s) | xcorr peak ratio | wav dur | vid dur |
|---|---:|---:|---:|---:|---:|
| 027_Zpswc | 1607 | +0.040 | 7.91 | 404.9 | 405.0 |
| 031_vpswc | 887 | +0.040 | 5.59 | 186.5 | 186.5 |
| 043_bc1wc | 1401 | +0.050 | 11.24 | 328.0 | 328.0 |
| 063_bV1wc | 841 | +0.050 | 10.03 | 181.5 | 181.4 |
| 104_xf1wc | 422 | +0.040 | 9.57 | 184.4 | 184.4 |
| 118_VD1wc | 678 | +0.040 | 2.32 | 120.7 | 120.7 |
| 142_GD1wc | 701 | +0.040 | 4.97 | 238.9 | 238.9 |
| 179_pM1wc | 515 | +0.020 | 4.71 | 189.1 | 189.0 |
| 212_y41wc | 946 | +0.040 | 2.49 | 322.5 | 322.4 |
| 235_Ny1wc | 1582 | +0.040 | 8.58 | 415.8 | 415.8 |
| 294_BSswc | 474 | +0.040 | 3.24 | 125.0 | 125.0 |
| 341_1M1wc | 691 | +0.010 | 3.94 | 127.4 | 127.4 |

Offsets are sub-frame (~|offset| < 0.042 s, one frame at 24 fps) and corroborated by the near-equal wav/video durations — the upload is essentially the GAPS crop.

## 2. Video-assisted Tab F1

Conditions per clip: `audio-only` (strings from the playability prior), `+real (auto)` (CV chain resolves the string; orientation auto-selected — the honest number), `+real (best-orient)` (best over the 4 fixed orientations — a diagnostic ceiling on orientation selection), `+oracle` (gold strings — the absolute ceiling). Gate params: vote 1 frames / +-60 ms, min coverage 0.71.

#### Audio source: `gold`

| Clip | audio-only | +real (auto) | +real (best-orient) | +oracle | lift (auto) | auto/best orient | kept/cov |
|---|---:|---:|---:|---:|---:|---|---:|
| 027_Zpswc | 0.7671 | 0.7671 | 0.7671 | 0.9568 | +0.0000 | none/none | 0/0.47 |
| 031_vpswc | 0.8658 | 0.8658 | 0.8658 | 0.9312 | +0.0000 | none/none | 0/0.45 |
| 043_bc1wc | 0.7616 | 0.7616 | 0.7616 | 0.9850 | +0.0000 | flip-string/flip-string | 0/0.27 |
| 063_bV1wc | 0.7111 | 0.7111 | 0.7111 | 0.9727 | +0.0000 | none/none | 0/0.20 |
| 104_xf1wc | 0.8014 | 0.8014 | 0.8014 | 0.9774 | +0.0000 | none/none | 0/0.49 |
| 118_VD1wc | 0.9336 | 0.9336 | 0.9336 | 0.9779 | +0.0000 | flip-fret/flip-fret | 0/0.32 |
| 142_GD1wc | 0.8388 | 0.8388 | 0.8388 | 0.9815 | +0.0000 | none/none | 0/0.53 |
| 179_pM1wc | 0.8621 | 0.8621 | 0.8621 | 0.9845 | +0.0000 | none/none | 0/0.58 |
| 212_y41wc | 0.7051 | 0.7051 | 0.7051 | 0.9820 | +0.0000 | flip-string/flip-string | 0/0.06 |
| 235_Ny1wc | 0.7035 | 0.7035 | 0.7035 | 0.9768 | +0.0000 | flip-fret/flip-fret | 0/0.54 |
| 294_BSswc | 0.9241 | 0.9241 | 0.9241 | 0.9937 | +0.0000 | none/none | 0/0.40 |
| 341_1M1wc | 0.9030 | 0.9030 | 0.9030 | 0.9522 | +0.0000 | flip-fret/flip-fret | 0/0.41 |
| **mean** | **0.8148** | **0.8148** | **0.8148** | **0.9726** | **+0.0000** | — | — |

+real (auto-orientation) Tab F1: mean **0.8148**, bootstrap lower-95 **0.7679** (n=12). Best-fixed-orientation ceiling: mean **0.8148**.
vs the 0.94 single-line video target (auto): **below 0.94 bar**.

Error decomposition (+real, auto-orientation):

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 8494 | — |
| wrong_position_same_pitch | 2213 | 98.3% |
| pitch_off | 2 | 0.1% |
| timing_only | 2 | 0.1% |
| missed_onset | 34 | 1.5% |
| extra_detection | 0 | 0.0% |

#### Audio source: `highres`

| Clip | audio-only | +real (auto) | +real (best-orient) | +oracle | lift (auto) | auto/best orient | kept/cov |
|---|---:|---:|---:|---:|---:|---|---:|
| 027_Zpswc | 0.6877 | 0.6877 | 0.6877 | 0.8698 | +0.0000 | none/none | 0/0.46 |
| 031_vpswc | 0.8594 | 0.8594 | 0.8594 | 0.9226 | +0.0000 | none/none | 0/0.43 |
| 043_bc1wc | 0.7067 | 0.7067 | 0.7067 | 0.9169 | +0.0000 | flip-fret/flip-fret | 0/0.25 |
| 063_bV1wc | 0.6431 | 0.6431 | 0.6431 | 0.8874 | +0.0000 | none/none | 0/0.20 |
| 104_xf1wc | 0.7049 | 0.7049 | 0.7049 | 0.8889 | +0.0000 | none/none | 0/0.50 |
| 118_VD1wc | 0.7927 | 0.7927 | 0.7927 | 0.8359 | +0.0000 | flip-fret/flip-fret | 0/0.31 |
| 142_GD1wc | 0.8127 | 0.8127 | 0.8127 | 0.9438 | +0.0000 | none/none | 0/0.54 |
| 179_pM1wc | 0.8571 | 0.8571 | 0.8571 | 0.9796 | +0.0000 | none/none | 0/0.58 |
| 212_y41wc | 0.6744 | 0.6744 | 0.6744 | 0.9177 | +0.0000 | flip-string/flip-string | 0/0.06 |
| 235_Ny1wc | 0.6151 | 0.6151 | 0.6151 | 0.8550 | +0.0000 | flip-fret/flip-fret | 0/0.54 |
| 294_BSswc | 0.9037 | 0.9037 | 0.9037 | 0.9757 | +0.0000 | none/none | 0/0.39 |
| 341_1M1wc | 0.8768 | 0.8768 | 0.8768 | 0.9261 | +0.0000 | flip-fret/flip-fret | 0/0.40 |
| **mean** | **0.7612** | **0.7612** | **0.7612** | **0.9099** | **+0.0000** | — | — |

+real (auto-orientation) Tab F1: mean **0.7612**, bootstrap lower-95 **0.7063** (n=12). Best-fixed-orientation ceiling: mean **0.7612**.

Error decomposition (+real, auto-orientation):

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 7719 | — |
| wrong_position_same_pitch | 2019 | 54.9% |
| pitch_off | 381 | 10.4% |
| timing_only | 48 | 1.3% |
| missed_onset | 578 | 15.7% |
| extra_detection | 652 | 17.7% |

