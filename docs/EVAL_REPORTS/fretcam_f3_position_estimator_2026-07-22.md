# FretCam F3: temporal position estimator

**Date:** 2026-07-22
**Status:** **PASS — ready for F4 HUD integration.**

## Implementation

The F3 estimator consumes F2b's index-fingertip fret coordinate (falling back
to the calibrated hand centroid when the index landmark is absent) and emits a
Roman position label, the fixed `[N-1,N+4] union {0}` window, confidence, and a
temporal state.

Fixed temporal policy:

- EMA alpha 0.35 for the displayed index coordinate;
- five consecutive frames to acquire or change a position;
- 0.15-fret boundary slack to prevent floor-boundary jitter from flapping;
- ten-frame temporal-agreement window;
- five-frame dropout hold, then `No hand` and a fresh five-frame reacquisition;
- `Shifting...` with no active fretted window while a new position is pending;
- confidence = F2b vision confidence (board/hand/span) times temporal agreement.

Open fret 0 is present in every locked/held position window. During acquiring,
shifting, and lost states the estimator exposes only `(0,)`, preventing a stale
fretted-position prior from escaping a transition.

## Headless verification

Nineteen FretCam tests passed. F3-specific synthetic trajectories cover:

- exact five-frame initial lock;
- I-to-IX shift with no intermediate position labels;
- alternating jitter around a fret boundary without a label change;
- five held dropout frames, loss, and five-frame reacquisition;
- EMA behavior, Roman labels through XXIV, and open-string windows;
- monotonic timestamp enforcement.

Ruff check and formatting checks passed on all changed files.

## Public GAPS replay

Command: `cd fretcam; .venv/Scripts/python.exe -m fretcam.replay_position`

Input: public cached `031_vpswc`, 2.0–8.0 s, sampled at 10 FPS (60 frames).

| measure | result |
|---|---:|
| first lock delay | 0.4 s |
| acquiring | 4 frames |
| locked | 52 frames |
| short-dropout holding | 4 frames |
| lost / shifting | 0 frames |
| locked label | Position I |
| estimator latency median / p95 / max | 0.0402 / 0.0567 / 0.1311 ms |

The generated diagnostic MP4 and first-locked still are machine-local under
`~/.tabvision/cache/fretcam_artifacts/` and are not committed. Visual inspection
of the still passed: `Position I`, the `{0,1,2,3,4,5}` window, confidence, and
index marker are legible; the marker lies on the visible fretting index finger.
This is a geometry/temporal sanity check only—GAPS has no visual position ground
truth, so it is not an A2 accuracy claim.

## Verdict

F3's temporal contract is implemented and headlessly verified. Proceed to F4
to wire the estimator into the WebSocket response and browser HUD, add framing
guidance, and measure end-to-end throughput before scheduling L1.
