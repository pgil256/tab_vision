# FretCam live accuracy phase 2

Date: 2026-07-23
Branch: `codex/fretcam-accuracy-phase-2`

This phase completes the six requested live-accuracy workstreams without a
new model, dependency, download, training run, saved browser frame, or change
inside `tabvision/`.

## What changed

### Neck-guided hand acquisition

- A locked fretboard immediately opens an enlarged and upscaled full-neck hand
  search; it no longer waits for two successful full-frame hand detections.
- Full-frame and neck searches alternate until acquisition. Once acquired, a
  narrower last-known-hand crop handles the fast path while periodic
  full-frame refreshes preserve recovery.
- Every returned hand is evaluated. Selection uses neck overlap, landmark
  geometry, confidence, and temporal continuity rather than discarding all but
  a MediaPipe handedness choice.
- Variable crops use MediaPipe IMAGE mode. The stable full-frame stream uses
  timestamped VIDEO mode, avoiding invalid tracking state across changing crop
  coordinate systems.

### Temporal landmarks

- MediaPipe refreshes run at a drift-free 10 Hz deadline cadence.
- Lucas-Kanade optical flow fills intervening frames.
- A One Euro filter smooths all 21 landmarks.
- Finger-length, joint-angle, detector-innovation, and optical-flow-innovation
  checks reject inconsistent chains while allowing coherent fast hand shifts.
- A strong rejected or occluded finger may be held for at most 180 ms.
- Every finger carries its own quality, source, freshness, and anatomical
  score.

### Physical contact evidence

- Contacts use the distal finger pad, with separate string and fret hypotheses
  for index, middle, ring, and pinky.
- Visibility and pressing state are separate. Curl, depth, neck proximity,
  motion, and per-finger track quality contribute to hover-versus-press
  evidence.
- Partial and multi-finger barres are represented explicitly. Non-index barres
  remain press-gated and bounded to nearby strings.
- Chord-shape compatibility can support or reduce confidence but cannot
  override contradictory visible fingers.

### Upper-neck geometry and calibration

- Rectified image gradients refine fret-wire support, string curves, nut, and
  body-joint boundaries.
- Body-joint inference supports frets 10–19 and penalizes candidate maps that
  leave observed fret-wire peaks unexplained.
- The nonlinear rule-of-18 map is maintained through upper positions. Invalid
  or crossing string fits are rejected, and sudden homography corrections are
  bounded and smoothed.
- Curved string fits compensate contact mapping for residual lens and rolling
  perspective when all six strings are trustworthy.
- Calibration may use Position I alone or a two-point Position I plus V/IX
  scale-and-offset fit. The fitted axis drives contacts, fret ticks,
  confidence, and the solver—not only the displayed number.

### Live-path regression coverage

- `fretcam.live_position_benchmark` replays the frozen public labels through a
  real localhost uvicorn/FastAPI WebSocket.
- The bounded matrix covers 2/5/10/20 FPS, JPEG quality 50/72/90,
  320/480/640 px inputs, six lighting modes, temporary occlusion, and camera
  motion.
- Reports preserve the complete confidence-factor vector, raw blockers,
  geometry freshness, accuracy, and transport timing.

### Expanded local development evidence

- `fretcam-local-eval` validates machine-local manifests for Positions I–XII,
  notes, chords, barres, stretches, slides, close/full-neck views, both player
  handednesses, lighting, instruments, sleeves, backgrounds, and per-finger
  visibility/pressing/string/fret labels.
- Repository policy permits only verified public-licensed or reproducibly
  synthetic sources. Private, user-recorded, and consented third-party media
  are rejected in every evaluation, tuning, training, and evidence role.
- Media and manifests remain outside the repository and separate from the
  browser's no-recording workflow.

## Verification

The final frozen development replay produced:

| Metric | Phase-2 result | Prior checkpoint |
|---|---:|---:|
| Displayed-position precision | 77/77 (1.000) | 76/80 (0.950) |
| Stable-frame coverage | 72/161 (0.447) | 71/161 (0.441) |
| Stable false locks | 0/161 | 0/161 |
| Negative-control displays | 0/120 | 0/120 |

The replay initially exposed unsupported body-joint axes and a picking-hand
false acquisition. Those failures were reproduced and isolated on the
development split. Nonlinear axes now require independent string evidence and
survive routine detector refreshes without sawtooth resets; a longitudinal
wrist-versus-fingertip check rejects the picking-hand geometry. The accepted
run retains the new correct Position-V and Position-VII detections while
abstaining on the uncertain Position-III chord.

Verification also passed:

- 172 FretCam tests and 5 parameterized subtests;
- Ruff checks and formatting for every changed Python file;
- JavaScript syntax and Git whitespace checks;
- a 20 FPS public-media probe with exact 10 Hz MediaPipe updates, intervening
  optical-flow frames, hover/press gating, and no false displayed position;
- rendered local-browser inspection with meaningful page content, the expected
  live status and calibration controls, no error overlay, and no console
  errors;
- interaction checks for the Position V/IX two-point anchor, player
  handedness, and preview mirroring.

The source-disjoint held-out split was not opened during this phase. The JSON
replay artifact remains machine-local under
`~/.tabvision/cache/fretcam_artifacts/phase2-dev-accepted.json` and is not
committed.
