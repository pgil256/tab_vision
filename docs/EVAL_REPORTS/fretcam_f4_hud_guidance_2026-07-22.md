# FretCam F4 HUD + guidance report — 2026-07-22

## Verdict

**PASS F4.** The default local app now runs the F2b detection chain and F3
position estimator, returns a JSON HUD result per in-memory JPEG, and renders
the required neck polygon, fret ticks, hand/index markers, Roman position,
open-safe window, confidence bar, and grounded framing guidance. A conservative
warm-path public-cache loopback met A4's throughput and latency targets.

This is a build/transport verdict, not live-camera A1 or position-accuracy
evidence. L1 remains a required human test.

## Implementation

- One stateful processor is prewarmed during server startup, reset for each
  WebSocket session, and serialized to the prototype's single live client.
- Browser capture keeps one frame in flight. The response contains frame
  dimensions, the complete F2b detection result, the F3 position estimate,
  guidance, and measured server time. Frames are never persisted.
- Guidance uses only visible signals: missing neck, geometry touching a 3%
  image-edge margin, weak board confidence, missing hand, and estimator state.
  It does not claim glare unless board confidence is weak.
- F1's byte-echo mode remains opt-in only for its historical transport test;
  the shipped app and `/health` default to `mode: hud`.

## Headless A4 measurement

Command:

```powershell
cd fretcam
.venv\Scripts\python -m fretcam.benchmark_hud --rounds 30 --warmup 10
```

Method: real Uvicorn + localhost WebSocket, one JPEG in flight, public cached
`031_vpswc`, 640 px maximum dimension, JPEG quality 72, full YOLO/MediaPipe
chain with the board detector scheduled at 2 Hz. Ten frames preconditioned the
temporal state; the 30 measured frames excluded one-time model initialization.
The latency gate uses p95, not the median.

| Metric | Result | Target |
|---|---:|---:|
| End-to-end throughput | **21.512 FPS** | ≥10 FPS |
| End-to-end latency median | **39.450 ms** | ≤150 ms |
| End-to-end latency p95 | **120.752 ms** | ≤150 ms |
| End-to-end latency max | 140.359 ms | diagnostic |
| Server latency median / p95 | 35.669 / 117.035 ms | diagnostic |
| Neck locked | 30 / 30 measured frames | diagnostic |
| Detector frames | 3 / 30 | 2 Hz scheduler |
| Median JPEG payload | 29,072 bytes | diagnostic |

## Verification

- `25 passed` across transport, detection geometry, estimator trajectories,
  guidance precedence, JPEG processing, JSON response, prewarm/reset/cleanup.
- `ruff check fretcam/src fretcam/tests`: pass.
- `node --check fretcam/src/fretcam/static/client.js`: pass.
- Isolated browser smoke check on port 8766: meaningful page content, all
  seven key HUD/control elements present, no framework error overlay, and no
  browser warnings/errors. Camera permission was intentionally not requested;
  live rendering is L1.

## Caveat and next action

The public replay establishes end-to-end budget and output-contract continuity,
not live framing behavior, tick alignment, lock time, or perceived lag. Pat runs
L1 (A1 + A4 + first impressions) next using the checklist in
`docs/fretcam-loop-state.md`.
