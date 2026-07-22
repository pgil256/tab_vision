# FretCam F4b — physical fret numbering and robust position lock

**Date:** 2026-07-22

**Scope:** Explicitly approved, quarantined FretCam-only diagnostic/fix

**Verdict:** PASS headlessly; L1/L2 remain pending live gates

## Problem reproduced

On public cached GAPS clip `104_xf1wc`, source time 12–19 s, the visible
index-hand move is Position II to Position VI. The nonlinear fret map stayed
locked on every sampled frame with homography confidence 0.92–0.97, but the
HUD reported Position I and then remained in `Shifting…`.

The trace isolated two implementation defects after the already-correct board
geometry:

1. The calibrated fret-cell array stores the first physical fret cell at
   index 0. FretCam passed that zero-based cell coordinate directly to the
   classical-position estimator, making the readout one position low.
2. Isolated MediaPipe/projection failures produced raw coordinates 24.00 and
   18.35. Each reset the five-frame shift hysteresis even though adjacent
   frames returned to the expected hand region.

## Bounded correction

- `compute_index_fret` now normalizes calibrated and rule-of-18 fallback
  coordinates to a one-based, physical, cell-centred fret number.
- Position candidates use nearest physical fret with 0.4-fret locked-state
  slack. The slack remains below half a fret and absorbs sub-fret fingertip
  placement/projection drift without hiding a real multi-fret shift.
- A one-frame jump greater than 10 frets holds the last accepted coordinate.
  A second nearby frame confirms and admits a genuine large relocation, so
  the guard cannot permanently suppress a sustained move. The design's
  I→V→IX acceptance shifts remain below the guard and incur no added delay.

No model, dependency, training, threshold in `tabvision/`, §8 contract, or
main-pipeline behavior changed.

## Verification

- Full FretCam suite: **33 passed** (one pre-existing Starlette deprecation
  warning).
- Ruff: all FretCam source/tests pass; the four changed files are formatted.
- Public full-neck replay: 70 frames at 10 FPS, `104_xf1wc` 12–19 s.
  - first Position II lock: source 12.4 s (0.4 s acquisition);
  - Position VI lock: source 16.6 s, 0.4 s after stable VI frames begin;
  - locked positions: only II (43 frames) and VI (13 frames); no false III,
    XVIII, XIX, or XXIV label;
  - estimator latency: 0.0399 ms median, 0.0613 ms p95, 0.2655 ms max.

Generated artifacts (local, not committed):

- `fretcam_104_full_neck_position_fix.mp4`
- `fretcam_104_full_neck_position_fix.png`
- `fretcam_104_full_neck_position_VI.png`

## Gate interpretation

This closes the reproduced public-footage HUD defect and shows that a new fret
CV model is not required for this failure: the existing calibrated fret map
was locked and the error was downstream coordinate semantics plus temporal
outlier handling. It is not a substitute for L1 or the 15-hold A2/L2 live
accuracy protocol.
