# FretCam accuracy-improvement handoff

**Date:** 2026-07-22  
**Purpose:** Self-contained handoff for continuing FretCam accuracy work in a
new Codex session.  
**Status:** Planning only. This document does not authorize a new build item,
model download, dependency, or training run.

## 1. Current repository state

- Working branch: `fretcam/f4d-fret-contact-semantics`
- Current workspace HEAD when this handoff was written: `30785e9`
  (`desktop-shell: pass correctness gate`)
- FretCam F4d implementation commit: `c5306da`
  (`fix(fretcam): classify barre contact by fret wires`)
- FretCam remains quarantined. Accuracy work belongs under `fretcam/` plus
  FretCam-specific plans, reports, state, and decision records.
- Do not modify `tabvision/`, `SPEC.md`, or the SPEC section 8 contracts for
  this work without separate explicit approval.
- The worktree already contains unrelated user changes and untracked planning
  files. Preserve them and do not stage or rewrite them as part of FretCam
  work.

Canonical operating material:

- `docs/prompts/fretcam-loop.md`
- `docs/fretcam-loop-state.md`
- `docs/plans/2026-07-22-fretcam-live-position-hud-design.md`
- `docs/EVAL_REPORTS/fretcam_f4d_fret_contact_semantics_2026-07-22.md`

## 2. What the current product does

The live chain currently performs:

1. YOLO OBB guitar/neck/fret detection at 2 Hz.
2. A neck homography and calibrated rule-of-18 fret map.
3. MediaPipe fretting-hand landmarks every frame.
4. Strict rejection of hands outside the neck and boundary-clipped fret
   geometry.
5. Technique-aware index contact classification.
6. EMA, hysteresis, jump rejection, and dropout holding before emitting a
   Roman-numeral playing position.

Accuracy fixes already shipped:

- **F4b:** corrected physical fret numbering and rejected isolated large
  landmark jumps.
- **F4c:** rejected off-neck hands and clipped geometry before temporal lock.
- **F4d:** classified contact by physical fret-wire cells instead of rounding
  a continuous fingertip coordinate. Confirmed full-spanning barres use their
  PIP/DIP/tip contact axis and a local behind-wire deadband; curled and
  partial-span fingers retain exact wire-cell containment.
- The raw continuous fingertip coordinate remains available as `tip-x` for
  diagnostics, while `contact` is the discrete fret cell used for locking.

F4d verification passed with 44 FretCam tests and Ruff. It did not add a model,
dependency, training run, or change to the main TabVision pipeline.

## 3. Latest five-source visual benchmark

These were freshly regenerated from the current F4d build. They are five
different public GAPS source videos, not five windows from one source.

| source | source window | frames | emitted positions | important observation |
|---|---:|---:|---|---|
| `023_Ypswc` | 2-8 s | 60 | VI: 4, VII: 19, VIII: 3 | Multiple high-neck position changes; substantial `shifting` time |
| `031_vpswc` | 2-8 s | 60 | I: 34, II: 3 | Barre correction reads contact 1 while the tip is around 1.83 |
| `104_xf1wc` | 12-19 s | 70 | II: 27, VI: 17 | User-verified Position II to VI move is preserved |
| `178_SM1wc` | 16-22 s | 60 | II: 18, V: 4 | Includes lost and shifting intervals that need labeled truth to judge |
| `238_My1wc` | 16-22 s | 60 | II: 43 | Stable Position II after acquisition |

Local H.264 artifacts, all fully decoded after generation:

```text
C:\Users\patri\.codex\visualizations\2026\07\22\019f89ec-b791-7721-8d35-8762f3723126\fretcam-five-sources-f4d\fretcam_01_023_Ypswc_h264.mp4
C:\Users\patri\.codex\visualizations\2026\07\22\019f89ec-b791-7721-8d35-8762f3723126\fretcam-five-sources-f4d\fretcam_02_031_vpswc_h264.mp4
C:\Users\patri\.codex\visualizations\2026\07\22\019f89ec-b791-7721-8d35-8762f3723126\fretcam-five-sources-f4d\fretcam_03_104_xf1wc_h264.mp4
C:\Users\patri\.codex\visualizations\2026\07\22\019f89ec-b791-7721-8d35-8762f3723126\fretcam-five-sources-f4d\fretcam_04_178_SM1wc_h264.mp4
C:\Users\patri\.codex\visualizations\2026\07\22\019f89ec-b791-7721-8d35-8762f3723126\fretcam-five-sources-f4d\fretcam_05_238_My1wc_h264.mp4
```

Two F4c negative controls remain important:

- `077_vV1wc`, 18-24 s: 55/60 observations rejected; no position emitted.
- `105_Qf1wc`, 10-16 s: 58/60 observations rejected; no position emitted.

### Evidence limitation

The five videos are useful product demonstrations but are not a formal
position-accuracy benchmark. GAPS has no visual classical-position labels.
Apart from the user-verified 104 move and the explicitly reviewed 031 barre
semantics, reclassifications are sanity evidence rather than accuracy ground
truth. Threshold tuning against these five outputs alone risks making the
display look better without making it more correct.

## 4. Remaining accuracy risks

### 4.1 Index-dominant inference

The stable position is still driven mainly by one index observation. This is
brittle when the index is hovering, partially occluded, angled, temporarily
mis-landmarked, or not the finger currently supplying the clearest position
evidence. F4d fixes the known barre geometry but is intentionally narrow.

### 4.2 Weak measurement foundation

There is no labeled public position set with stable holds, shifts, lighting,
camera angles, and technique categories. Without it, the system cannot report
position precision, coverage, false-lock rate, or shift delay with defensible
numbers.

### 4.3 Permissive confidence and single-source agreement

The temporal estimator accepts vision confidence as low as `0.05`. The emitted
confidence does not yet explicitly measure agreement among multiple fingers,
hand-pose quality, fret-map age, or geometry innovation. The product should
prefer `uncertain` over a confident wrong label.

### 4.4 Geometry between detector passes

The detector runs at 2 Hz. Between detector passes the current chain reuses the
last homography; it does not perform a true per-frame optical-flow update.
Fresh fret centers replace the stored map when available, with no explicit
innovation gate against a temporally implausible map. Camera movement or a bad
detection can therefore create drift or stale geometry.

### 4.5 Frame-count-based shift behavior

Position switching requires five agreeing frames. That is about 500 ms at
10 FPS but roughly 230 ms at 21.5 FPS. Accuracy and perceived lag therefore
depend on runtime frame rate.

### 4.6 MediaPipe dropout

When the palm is hidden behind the neck or motion blur is high, MediaPipe may
lose landmarks. The existing F6 hand-box by fret-zone IoU fallback is intended
for this case, but it needs a dataset download and potentially a training run.
It should only be opened if measured dropout is the dominant remaining error.

## 5. Recommended roadmap, in order

## 5.1 F4e-A: build a labeled public position benchmark first

This is the recommended next bounded iteration. It changes measurement, not
product behavior.

Create a public-only annotation set containing stable windows and shifts. Aim
for coverage of:

- Positions I, III, V, VII, and IX.
- Single notes, ordinary chord shapes, and barres.
- Full-neck and closer framing.
- At least two lighting/contrast conditions and several neck angles.
- Clean holds, deliberate shifts, occlusions, off-neck hands, and invalid
  boundary geometry.

Suggested annotation fields:

```json
{
  "source": "public source stem",
  "start_s": 12.0,
  "end_s": 14.5,
  "state": "stable",
  "position": 2,
  "technique": "chord",
  "visibility": "full_neck",
  "confidence": "verified",
  "notes": "visual reason for label"
}
```

Label transition intervals as `shifting` instead of forcing a position. Keep
uncertain ranges out of the scored test portion.

Report at least:

- **Displayed-position precision:** correct displayed frames / all frames that
  display a position.
- **Coverage:** frames displaying a position / all scorable stable frames.
- **False-lock rate:** wrong locked frames / all scorable stable frames.
- **Shift latency:** time from stable arrival at the new position to correct
  lock.
- **Dropout recovery:** time from valid hand return to correct lock.
- Results split by note/chord/barre, position, visibility, and lighting when
  sample size allows.

Freeze a development portion for choosing rules and a separate test portion
for the final report. Do not tune and report on the same windows.

Data-hygiene rule: user/private recordings are never metric, training, or
threshold-tuning data. Public footage may be labeled and scored. A private
debug clip requires explicit per-clip approval and can only reproduce a
functional bug; it is never committed or used for reported accuracy.

## 5.2 F4e-B: replace index-only locking with a multi-finger pose solver

Once a baseline is frozen, use the whole fretting-hand pose to score candidate
positions rather than letting one landmark dominate.

Proposed mechanics:

1. Retain MCP, PIP, DIP, and tip landmarks for index, middle, ring, and pinky,
   including landmark visibility/depth information available from MediaPipe.
2. Classify per-finger evidence:
   - curled/likely fretting: exact fingertip wire cell;
   - extended across-neck index: existing barre-axis semantics;
   - hovering or geometrically inconsistent: low weight or no vote;
   - boundary-clipped/off-neck: invalid, preserving F4c.
3. Score candidate positions from the set of plausible finger contacts and
   normal hand-span/order constraints. Do not assume the minimum observed fret
   is always the position.
4. Require enough independent agreement before emitting a candidate. Preserve
   each raw per-finger contact for the replay overlay and diagnostics.
5. Pass a solver confidence based on pose quality and finger agreement into
   the temporal estimator.

No new dependency or training run should be necessary for this iteration.
Synthetic tests should cover ordinary chords, barres, stretches, hovering
index fingers, partial occlusion, reversed neck orientation, high frets, and
conflicting landmarks. Re-run the frozen benchmark and both F4c negative
controls.

## 5.3 Calibrate abstention and confidence

Accuracy should be improved partly by refusing weak observations, not by
guessing.

Build confidence from explicit factors:

- homography/fret-map confidence and age;
- on-neck landmark support;
- per-finger landmark/pose quality;
- agreement among valid fingers;
- consistency between the contact solver and the coarse hand anchor;
- temporal agreement and recent geometry stability.

Generate a precision-versus-coverage curve on the development split, choose a
threshold before opening the test split, and preserve `uncertain`, `shifting`,
and `lost` as honest output states. The live L2 gate remains at least 90% of
the 15 prescribed holds correct; displayed-position precision should be
reported separately from coverage.

## 5.4 Strengthen temporal fretboard geometry

After the hand solver, address geometry drift:

- Reject abrupt homography or fret-map innovations unless repeated.
- Track neck corners or stable neck features between 2 Hz YOLO detections with
  existing OpenCV facilities.
- Age geometry confidence and force reacquisition when tracking support
  disappears.
- Locally refine expected wire locations in the rectified neck using the
  calibrated rule-of-18 map as a prior and temporal aggregation.
- Do not replace the current detector with standalone Canny/Hough; prior
  project evidence found that route lighting-fragile.

## 5.5 Make temporal locking time-based

Replace fixed frame counts with elapsed stable time so behavior is consistent
across frame rates. Add motion/change-point evidence so rapid shifts enter
`shifting` promptly and stable arrival can lock after a defined duration.
Preserve the isolated-jump guard from F4b.

Measure both correctness and shift latency on the frozen benchmark. A faster
label is not an improvement if it increases false locks during motion.

## 5.6 Optional one-hold session calibration

An optional short Position-I hold can estimate a per-session residual offset
and validate neck direction before normal use. This trades a little setup
friction for robustness across guitar shapes, lenses, and camera placement.
Keep it optional and bounded; it must not conceal invalid geometry or replace
the automatic fret map.

## 5.7 F6 learned fallback only if dropout dominates

If labeled and live evidence shows that MediaPipe loss, rather than wrong
contact semantics, is the principal remaining problem, open F6:

- Detect a coarse hand box and intersect it with fret-zone boxes.
- Use the already identified public `ghaleb/guitar-fretboard` Hand + Zone1-12
  labels as the likely starting point.
- A Roboflow download/account and any model training require explicit user
  approval before starting.
- Report this primarily as a coverage/dropout improvement; verify that it does
  not increase false locks.

## 6. Should a dedicated fret CV model be trained now?

**Recommendation: no, not yet.** The existing detector already supplies fret
wire observations and fits them to physical rule-of-18 geometry. The failures
addressed so far were coordinate conversion, invalid-hand acceptance, and
finger-contact semantics—not clear evidence that wire detection is the main
bottleneck.

A learned fret/nut keypoint model becomes justified only if the labeled error
breakdown shows repeated fret-map error while the hand pose is otherwise
correct. That route also requires an approved dataset/download/training run and
should be compared against the cheaper temporal/local refinement first.

## 7. Recommended decision for the next session

Do not start by changing thresholds. The highest-leverage sequence is:

1. Explicitly approve and execute **F4e-A**, the public labeled benchmark.
2. Freeze its development/test split and baseline report.
3. Explicitly approve and execute **F4e-B**, the multi-finger pose solver.
4. Use the error breakdown to choose confidence, geometry, temporal, or F6
   work rather than implementing all options speculatively.

Suggested next-session request:

```text
Read docs/prompts/fretcam-loop.md and
docs/plans/2026-07-22-fretcam-accuracy-improvement-handoff.md. I explicitly
approve one newly bounded F4e-A iteration: create and baseline the public-only
labeled position benchmark described in the handoff. Do not change product
inference, download data, train a model, add dependencies, or touch tabvision/.
```

After F4e-A is complete and reviewed, a separate request can approve F4e-B.

## 8. Verification commands and useful replay windows

From the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path 'fretcam\src').Path + ';' + (Resolve-Path 'tabvision').Path

ruff check fretcam/src fretcam/tests
ruff format --check fretcam/src fretcam/tests

@'
import sys
sys.path.append(r'C:\Users\patri\AppData\Local\Programs\Python\Python312\Lib\site-packages')
import pytest
raise SystemExit(pytest.main(['-q', 'fretcam/tests']))
'@ | .\fretcam\.venv\Scripts\python.exe -
```

Replay form:

```powershell
.\fretcam\.venv\Scripts\python.exe -m fretcam.replay_position `
  --clip 104_xf1wc --start 12 --duration 7 --sample-fps 10 `
  --output <output.mp4> --still <still.png>
```

Known windows:

```text
023_Ypswc  2-8 s
031_vpswc  2-8 s
104_xf1wc  12-19 s
178_SM1wc  16-22 s
238_My1wc  16-22 s
077_vV1wc  18-24 s  (negative control)
105_Qf1wc  10-16 s  (negative control)
```

## 9. Live gates still pending

The headless work does not replace live acceptance:

- L1 remains pending for neck lock, drift, throughput, latency feel, lighting,
  and first impressions.
- F5 remains blocked on the L1 report under the existing loop.
- L2 remains blocked on F5 and requires the full 15-hold protocol:
  Positions I/III/V/VII/IX by note/chord/barre, plus shifts, occlusion recovery,
  and two lighting conditions.
- F8 remains blocked until L2 passes; any TabVision integration is a separate
  program requiring explicit sign-off.

The new F4e-A/F4e-B work described here must be explicitly approved as newly
bounded work if it is performed before L1.
