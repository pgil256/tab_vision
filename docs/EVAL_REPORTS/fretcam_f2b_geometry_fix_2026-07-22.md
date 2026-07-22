# FretCam F2b: calibrated fret-axis geometry fix

**Date:** 2026-07-22
**Status:** **PASS — original F2 three-clip gate is unblocked.**

## Approved scope

Diagnose the `027_Zpswc` boundary clipping, make the smallest quarantined
FretCam geometry correction, and rerun the original three clips without
changing their order, time range, sampling cadence, confidence gates, or
plausibility predicate. No model inference settings, downloads, dependencies,
training, or TabVision package code changed.

## Root cause

F2 passed the homography and hand landmarks to
`tabvision.video.hand.neck_anchor.compute_neck_anchor`, which converts the
canonical x coordinate with `x * cfg.max_fret`. FretCam uses the default
`max_fret=24`, but its keypoint homography describes physical distance along a
unit neck and its calibrator already returns a nonlinear per-fret map. The old
adapter ignored that map and treated the body-joint end of the unit neck as
fret 24.

The direct `027_Zpswc` diagnostic confirmed strong neck/hand detections but
canonical fingertip x values around 1.04–1.09, so the old conversion clipped
the median to 24.0. On `031_vpswc`, the detected fret map runs in the opposite
canonical direction (fret-0 center x=0.957); the old conversion reported
17.862 while orientation-aware interpolation reports 2.756.

## Correction

`fretcam.detection.compute_position_anchor` now:

1. projects the same wrist and four fretting fingertips through the same
   homography;
2. interpolates canonical x against the detected fret-center map, accepting
   either monotonic direction; or
3. when no map exists, applies the repository's established unit-neck
   convention (`x=0` nut, `x=1` fret 12) with rule-of-18 spacing.

The fallback endpoint is existing project geometry, not a constant selected
from the gate result. Confidence and one-fret span margins retain the original
coarse-anchor calculation.

## Unchanged F2 gate rerun

Command: `cd fretcam; .venv/Scripts/python.exe -m fretcam.replay_gaps`

| clip | pass | first usable time | center fret | anchor conf | H conf | fitted map |
|---|---|---:|---:|---:|---:|---|
| `027_Zpswc` | yes | 8.5 s | 12.000 | 0.965 | 0.984 | no; rule-of-18 fallback |
| `031_vpswc` | yes | 2.0 s | 2.756 | 0.518 | 0.634 | yes |
| `043_bc1wc` | yes | 2.0 s | 9.381 | 0.574 | 0.620 | no; rule-of-18 fallback |

**Gate: 3/3, PASS** (required 3/3). Each success also retained lock on the
next non-detector tracking frame.

Warm-path medians in this rerun: detector 98.719 ms, hand 20.358 ms, anchor
0.012 ms, total 122.504 ms. Cold-start total was 5.138 s and is retained in
the raw command output as initialization, not reported as warm latency.

## Verification and limits

- 12 FretCam tests passed, including new ascending-map, descending-map, and
  no-map body-joint regression coverage.
- Ruff check and formatting checks passed on the changed code.
- This gate verifies cross-clip lock and non-boundary geometry only. It does
  not provide position ground truth; the live A2 protocol remains the accuracy
  gate.
- F7 used the old `x * 24` conversion. Its 0.247 result remains banked as
  historical evidence but is superseded for the corrected implementation and
  must be rerun before F8.
