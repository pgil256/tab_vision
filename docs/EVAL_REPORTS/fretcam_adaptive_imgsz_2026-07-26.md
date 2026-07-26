# FretCam adaptive detector scale — closed negative

**Date:** 2026-07-26
**Verdict:** **CLOSED NEGATIVE.** Implemented, measured, and left **off by
default**. It buys +0.006 dev coverage and costs a 0.000 → 0.0497 false-lock
rate.
**Scope:** `fretcam/` plus one additive optional keyword on
`tabvision.video.guitar.yolo_backend.YoloOBBBackend.predict_all`. No dependency,
model, download, or training run.

## 1. The hypothesis, and why it looked strong

`YoloOBBBackend` ran `model.predict(frame)` at ultralytics' default
`imgsz=640`, and `HudFrameProcessor` caps live frames at 640×480, so fret wires
— thin, small objects — were never detected above native resolution. The
calibrated rule-of-18 fret map was absent on 51% of benchmark frames
(`fret_map_locked` 135/276), falling back to `rule18_fret12_fallback`.

Holding the checkpoint fixed and varying only the inference scale over 54 dev
stable frames produced a clean, strongly-signed result:

| framing | imgsz | neck detected | mean fret detections | ≥4 wires | fret map fitted |
|---|---:|---:|---:|---:|---:|
| full-neck (36) | **640** | 1.000 | **6.56** | 0.389 | 0.333 |
| full-neck | 960 | 1.000 | **14.42** | **0.861** | 0.556 |
| full-neck | 1280 | 0.972 | 14.31 | 0.861 | **0.667** |
| close (18) | **640** | 1.000 | 14.44 | 1.000 | **1.000** |
| close | 960 | 0.889 | 14.67 | 1.000 | 0.889 |
| close | 1280 | 0.611 | 10.11 | 1.000 | **0.111** |

Upscaling more than doubles fret-wire yield at full-neck framing and lifts the
fret-map fit rate from 0.333 to 0.667 — while at close framing the same change
starts losing the **neck OBB itself**. So a fixed larger `imgsz` is wrong and
the rule has to be adaptive.

## 2. The rule

Measuring the detected neck's long edge at native scale separates the two
framings cleanly:

| framing | neck long edge p10 / median / p90 (px) | fret map fitted |
|---|---:|---:|
| close | 416.7 / **432.9** / 448.9 | 1.000 |
| full-neck | 205.6 / **249.9** / 306.7 | 0.333 |

The detector responds to the neck's size *in its own input*, so the target is an
absolute pixel extent:

```
rendered_neck_px = neck_px · imgsz / frame_long_px
⇒ imgsz = TARGET_NECK_PX · frame_long_px / neck_px,   TARGET_NECK_PX = 430
```

clamped to [640, 1280] and snapped to stride 32. One constant reproduces both
measured optima: close framing (433 px) → **640** (native, preserving its 1.000
fit rate), full-neck (250 px) → **1088** (inside the measured 960–1280 band).
The rule is resolution-independent by construction, so it transfers to a live
1080p camera unchanged. Scale is taken from the geometry the tracker already
holds, so it is one detector pass behind — the first pass must find a neck
before its size can inform anything.

## 3. Result — it regressed

Dev split, on top of the four coverage fixes:

| build | valid obs | coverage | displayed precision | **false-lock** | negative control |
|---|---:|---:|---:|---:|---:|
| four fixes | **0.7267** | 0.4783 | **0.9535** | **0.0000** | 0.0000 |
| + adaptive scale | 0.6708 | 0.4845 | 0.8621 | **0.0497** | 0.0000 |

Coverage moved +0.006 — noise. Everything else moved the wrong way:
valid-observation rate **fell** 0.056, and eight stable frames locked to a wrong
position where the previous build had none.

**Every one of the eight is at `full-neck` framing** — precisely the case the
change targets — and they are not adjacent slips:

| predicted − truth | frames | sequence |
|---:|---:|---|
| −7 | 4 | `dev_341_shift_to_ix` (predicted II, truth IX) |
| +3 | 2 | `dev_142_note_v` (predicted VIII, truth V) |
| −1 | 2 | `dev_142_chord_iii` |

A constant seven-position offset is a **mis-anchored fret axis**, not a noisy
contact. `calibrate_fret_xs` searches the first visible wire's index over
`k0 ∈ 1..6` and takes the largest inlier consensus. More wires — especially
wires further up the neck, which is exactly what upscaling adds — let a *wrong*
`k0` gather a winning consensus, and the whole axis shifts by the difference.
The `no_hand` blocker also rose 26 → 37, because the displaced neck quad feeds
`_select_neck_hand` and the crop rectangles.

## 4. What this actually establishes

**Fret-map *fit rate* is not fret-map *correctness*, and optimising the
intermediate metric moved the end metric backwards.** §1's table is still true:
upscaling really does deliver the wires. The failure is downstream — the
rule-of-18 anchor search is not robust enough to consume them, and it fails
*silently*, producing a confidently wrong axis rather than declining to fit.

That reframes the open lead. The blocker on fret geometry is **not** wire yield.
It is the `k0` anchor, which is currently decided by inlier count alone with no
temporal agreement, no nut-anchored cross-check, and no penalty for an axis that
disagrees with the previous frame's. Wire yield only becomes worth harvesting
after that is fixed — at which point this change is already implemented and
waiting behind its flag.

## 5. Disposition

- `DetectionChain(adaptive_detector_scale=...)` defaults to **`False`**. The
  default build's call into the detector is byte-identical to before.
- `YoloOBBBackend.predict_all(frame, *, imgsz=None)` is additive and optional;
  `None` preserves ultralytics' default. The `Detector` protocol is unchanged
  and FretCam feature-detects the keyword, so test doubles and any other caller
  are untouched.
- Retained rather than reverted because the §1 measurement is reusable and the
  implementation is the cheap part; the anchor work is the expensive part.

## 6. Verification

- `283 passed, 1 skipped` in the FretCam suite, including eight new tests
  covering the scale rule, its clamping and stride alignment, its
  resolution-independence, the no-board case, the feature-detection path for
  detectors without the keyword, and that **the default is off**.
- `56 passed` across the tabvision detector tests (`test_yolo_backend`,
  `test_fretboard_keypoint`, `test_fretboard_calibrate`,
  `test_video_orchestrators`).
- `ruff check` / `ruff format --check` pass on all changed files; `mypy` clean
  on `yolo_backend.py`.
- The default-off build re-runs the dev benchmark to the four-fix numbers
  exactly, confirming the flag is inert.

## 7. Reproduce

```bash
./fretcam/.venv/Scripts/python.exe -m fretcam.position_benchmark --split dev --output-json <machine-local-output.json>
```
