# FretCam-loop state
last_updated: 2026-07-22
current_branch: fretcam/f3-position-estimator

Loop protocol: `docs/prompts/fretcam-loop.md`. Design:
`docs/plans/2026-07-22-fretcam-live-position-hud-design.md`.

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| F1 | scaffold (`fretcam/` FastAPI+WS+page) | passed | WS test 1/1; 517 B ×100: median 0.536 ms, p95 0.903 ms | — | — |
| F2 | detection chain (OBB→homography→hand→anchor) | closed-negative | 2/3 clips; `027_Zpswc` 0/56 plausible anchors; detector/hand median 123.733/51.324 ms | preserve evidence; do not tune past gate | gate required ≥3 clips |
| F2b | calibrated fret-axis geometry fix + original F2 rerun | passed | 3/3 clips; centers 12.000/2.756/9.381; total median 122.504 ms | — | — |
| F3 | position estimator (smoothing/hysteresis) | passed | 19 tests; first lock 0.4 s; 52/60 GAPS frames locked; estimator median 0.0402 ms | — | — |
| F4 | HUD + guidance + latency | open | ≥10 FPS, ≤150 ms | wire F2b+F3 into WS/browser HUD; measure A4 | — |
| L1 | live test 1 (Pat: A1+A4) | blocked | — | — | F4 |
| F5 | fix round + full checklist | blocked | — | — | L1 |
| L2 | full §6 acceptance (Pat) | blocked | A2 ≥90% of holds | — | F5 |
| F6 | IoU fallback (TapToTab mechanism) | conditional | — | needs ghaleb dataset → STOP first | opens on L2 fail |
| F7 | GAPS anchor probe (cache-only, fill-in) | open (rerun required) | old x×24 result banked: 0.247; superseded by F2b geometry correction | rerun with calibrated/fret-12 mapping | — |
| F8 | M4 bridge verdict | blocked | target > 38.76% @60 s (assisted) | — | L2 pass + corrected F7 |

**Build path is active.** F3 passes its synthetic and public-replay checks; F4
is next. F7 must later be rerun because it used the corrected bug's old
`canonical_x × 24` conversion.

## Standing constraints (from the loop prompt — do not relax silently)
- No edits inside `tabvision/`, SPEC, or §8. FretCam is quarantined.
- Private recordings: never in training/eval/label roles; debug clips only
  with per-clip approval, no metrics, never committed.
- Pre-approved deps: fastapi, uvicorn[standard], websockets + existing
  `tabvision.video.*` imports. Anything else stops the iteration.
- Training runs and Roboflow downloads: STOP for approval.

## Questions for Pat
- None.

## Live-test log (newest first)
- None yet.

## Iteration log (newest first)
- 2026-07-22 — F3 passed — five-frame EMA/hysteresis estimator, Roman position,
  open-safe window, transition/dropout states, and temporal confidence shipped.
  Nineteen tests passed. Public `031_vpswc` replay locked after 0.4 s, stayed
  locked for 52/60 frames with four held dropouts, and added 0.0402 ms median
  estimator latency; the diagnostic overlay still passed visual inspection.
- 2026-07-22 — F2b passed — root cause was the adapter ignoring the calibrated
  fret map and mapping the unit-neck body joint to fret 24. Orientation-aware
  fret-map interpolation plus the existing rule-of-18 fret-12 fallback changed
  original-gate centers to 12.000/2.756/9.381 and passed 3/3 without changing
  clips or thresholds. Twelve tests passed; total warm-path median 122.504 ms.
- 2026-07-22 — F7 closed-negative — on 1,566 audio-wrong ambiguous notes with
  cached anchors, the gold fret fell in the fixed FretCam window 387 times
  (0.247; Wilson 95% CI 0.226–0.269), below A14's 0.285 comparator and the
  0.382 anchor marginal. Audio prior parity was 0.782 vs 0.778; no inference,
  downloads, training, or TabVision package edits.
- 2026-07-22 — F2 closed-negative — 2/3 GAPS clips passed; `027_Zpswc`
  produced 0 plausible anchors in 56 samples at 2 Hz. Detector/hand/total
  median latency 123.733/51.324/174.607 ms (p95 total 252.166 ms; cold max
  8580.824 ms). Six headless tests passed; no threshold/clip substitution.
- 2026-07-22 — F1 passed — FastAPI/WebSocket echo scaffold and browser FPS/RTT
  page shipped; synthetic JPEG test passed; loopback median/p95 0.536/0.903 ms.
- 2026-07-22 — loop created (prompt + this state file). No code yet.
