# FretCam-loop state
last_updated: 2026-07-22
current_branch: fretcam/f2-detection-chain

Loop protocol: `docs/prompts/fretcam-loop.md`. Design:
`docs/plans/2026-07-22-fretcam-live-position-hud-design.md`.

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| F1 | scaffold (`fretcam/` FastAPI+WS+page) | passed | WS test 1/1; 517 B ×100: median 0.536 ms, p95 0.903 ms | — | — |
| F2 | detection chain (OBB→homography→hand→anchor) | closed-negative | 2/3 clips; `027_Zpswc` 0/56 plausible anchors; detector/hand median 123.733/51.324 ms | preserve evidence; do not tune past gate | gate required ≥3 clips |
| F3 | position estimator (smoothing/hysteresis) | blocked | — | — | F2 closed-negative |
| F4 | HUD + guidance + latency | blocked | ≥10 FPS, ≤150 ms | — | F3 |
| L1 | live test 1 (Pat: A1+A4) | blocked | — | — | F4 |
| F5 | fix round + full checklist | blocked | — | — | L1 |
| L2 | full §6 acceptance (Pat) | blocked | A2 ≥90% of holds | — | F5 |
| F6 | IoU fallback (TapToTab mechanism) | conditional | — | needs ghaleb dataset → STOP first | opens on L2 fail |
| F7 | GAPS anchor probe (cache-only, fill-in) | open (fill-in; awaiting Pat) | compare vs 0.285 anti-enrichment / 0.778 audio prior | window scorer over cached fingerings + banked lattice | Pat direction after F2 negative |
| F8 | M4 bridge verdict | blocked | target > 38.76% @60 s (assisted) | — | L2 + F7 |

**Build path paused at F2 closed-negative.** F7 remains technically independent,
but the loop stops for Pat's direction before selecting another item.

## Standing constraints (from the loop prompt — do not relax silently)
- No edits inside `tabvision/`, SPEC, or §8. FretCam is quarantined.
- Private recordings: never in training/eval/label roles; debug clips only
  with per-clip approval, no metrics, never committed.
- Pre-approved deps: fastapi, uvicorn[standard], websockets + existing
  `tabvision.video.*` imports. Anything else stops the iteration.
- Training runs and Roboflow downloads: STOP for approval.

## Questions for Pat
- Should the next iteration run the independent cache-only F7 probe while the
  FretCam build path remains paused at the failed F2 gate?

## Live-test log (newest first)
- None yet.

## Iteration log (newest first)
- 2026-07-22 — F2 closed-negative — 2/3 GAPS clips passed; `027_Zpswc`
  produced 0 plausible anchors in 56 samples at 2 Hz. Detector/hand/total
  median latency 123.733/51.324/174.607 ms (p95 total 252.166 ms; cold max
  8580.824 ms). Six headless tests passed; no threshold/clip substitution.
- 2026-07-22 — F1 passed — FastAPI/WebSocket echo scaffold and browser FPS/RTT
  page shipped; synthetic JPEG test passed; loopback median/p95 0.536/0.903 ms.
- 2026-07-22 — loop created (prompt + this state file). No code yet.
