# FretCam-loop state
last_updated: 2026-07-22
current_branch: fretcam/f4e-a-position-benchmark

Loop protocol: `docs/prompts/fretcam-loop.md`. Design:
`docs/plans/2026-07-22-fretcam-live-position-hud-design.md`.

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| F1 | scaffold (`fretcam/` FastAPI+WS+page) | passed | WS test 1/1; 517 B ×100: median 0.536 ms, p95 0.903 ms | — | — |
| F2 | detection chain (OBB→homography→hand→anchor) | closed-negative | 2/3 clips; `027_Zpswc` 0/56 plausible anchors; detector/hand median 123.733/51.324 ms | preserve evidence; do not tune past gate | gate required ≥3 clips |
| F2b | calibrated fret-axis geometry fix + original F2 rerun | passed | 3/3 clips; centers 12.000/2.756/9.381; total median 122.504 ms | — | — |
| F3 | position estimator (smoothing/hysteresis) | passed | 19 tests; first lock 0.4 s; 52/60 GAPS frames locked; estimator median 0.0402 ms | — | — |
| F4 | HUD + guidance + latency | passed | 25 tests; 21.512 FPS; E2E 39.450 ms median / 120.752 ms p95 | — | — |
| F4b | physical fret numbering + spike-safe position lock | passed | 33 tests; public II→VI lock in 0.4 s; only II/VI locked | preserve evidence; L1 still pending | — |
| F4c | reject off-neck hands + clipped geometry before lock | passed | 38 tests; 5 distinct sources; 2/2 false-lock clips now emit no position | preserve evidence; L1 still pending | — |
| F4d | wire-cell + barre contact semantics | passed | 44 tests; `031` contact I with tip-x 1.83; verified `104` II→VI preserved | preserve evidence; L1 still pending | — |
| F4e-A | frozen public position benchmark | passed | 16 sequences/12 sources; F4d baseline precision 55/69, stable coverage 60/276, false locks 10/276 | review report; F4e-B needs separate approval | — |
| L1 | live test 1 (Pat: A1+A4) | awaiting Pat | headless A4 pass; live A1/A4 pending | run checklist below and paste report | — |
| F5 | fix round + full checklist | blocked | — | — | L1 |
| L2 | full §6 acceptance (Pat) | blocked | A2 ≥90% of holds | — | F5 |
| F6 | IoU fallback (TapToTab mechanism) | conditional | — | needs ghaleb dataset → STOP first | opens on L2 fail |
| F7 | GAPS anchor probe (cache-only, fill-in) | completed-positive | corrected 1195/1566 = 0.763 (CI 0.741–0.783); +0.478 vs 0.285; old 0.247 preserved as superseded | preserve fixed result; no tuning | — |
| F8 | M4 bridge verdict | blocked | F7 positive; target >38.76% @60 s (assisted) | after L2 pass, synthesize and STOP before integration | L2 pass |

**Live checkpoint.** F4e-A freezes the first source-disjoint public position
benchmark without changing F4d inference. Across 450 samples from 16 sequences
and 12 sources, overall displayed precision is 55/69 (0.797), stable coverage
is 60/276 (0.217), and stable false locks are 10/276 (0.036). Held-out stable
false locks are 0/115, but coverage is only 8/115 and four displays occur on
an invalid crossfade. No shift origin was freshly locked and neither dropout
produced a measured recovery, so coverage/observation support is the dominant
baseline limitation. L1 is still Pat-only and must run before F5. F4e-B, if
desired before L1, requires a separate explicit approval; F8 remains blocked
on an L2 pass and separate integration sign-off.

## Standing constraints (from the loop prompt — do not relax silently)
- No edits inside `tabvision/`, SPEC, or §8. FretCam is quarantined.
- Private recordings: never in training/eval/label roles; debug clips only
  with per-clip approval, no metrics, never committed.
- Pre-approved deps: fastapi, uvicorn[standard], websockets + existing
  `tabvision.video.*` imports. Anything else stops the iteration.
- Training runs and Roboflow downloads: STOP for approval.

## Questions for Pat
- Review the F4e-A report. Approve F4e-B separately if you want the bounded
  multi-finger pose solver before L1; otherwise run L1 with the checklist below
  and paste the filled report.

## Live-test log (newest first)
- None yet.

## Iteration log (newest first)
- 2026-07-22 — F4e-A passed — froze 16 public-only labeled sequences from 12
  source-disjoint GAPS videos and baselined unchanged F4d inference over 450
  samples. Overall displayed precision is 55/69, stable coverage 60/276, and
  stable false locks 10/276; held-out coverage is 8/115 with 0/115 stable false
  locks, while four invalid-scene displays reduce held-out precision to 8/12.
  All three shifts lacked a fresh origin lock; one dropout origin did not lock
  and one recovery was censored. The benchmark defaults to dev-only, and the
  test split is frozen after this single baseline opening.
- 2026-07-22 — F4d passed — position locking now uses calibrated wire-cell
  containment; a confirmed extended index spanning ≥70% across the neck uses
  its PIP/DIP/tip contact axis and a local-width behind-wire deadband. The raw
  fingertip coordinate remains diagnostic-only. Forty-four tests and Ruff
  passed. Seven-source replay reclassified `031` from the tip-biased II to
  predominantly I, preserved `104` II→VI exactly, and retained zero positions
  on both F4c wrong-hand clips.
- 2026-07-22 — F4c passed — a hand must place at least three fingertips plus
  its index observation on the canonical neck, calibrated coordinates must lie
  inside the outer fret-cell boundaries, and every failure becomes a dropout
  before position locking. Thirty-eight tests and Ruff passed. Across five
  distinct public clips, the three known-valid sources retained output while
  `077_vV1wc` and `105_Qf1wc` rejected 55/60 and 58/60 frames respectively and
  emitted no position.
- 2026-07-22 — F4b passed — the calibrated cell index is now normalized to
  one-based physical fret numbering; nearest-fret locking uses 0.4-fret
  sub-cell slack, and isolated >10-fret landmark spikes are held unless a
  second frame confirms the move. Thirty-three tests and Ruff passed. The
  70-frame public full-neck replay locked only Position II and VI, reached VI
  0.4 s after stable arrival, and added 0.0399 ms median estimator latency.
- 2026-07-22 — corrected F7 completed-positive — with the exact clean-12,
  cached frames, A14 audio decoder, window, and timestamp protocol preserved,
  replacing `canonical_x × 24` with F2b calibration/fret-12 mapping changed the
  primary from 0.247 to 1195/1566 = 0.763 (95% CI 0.741–0.783). This is +0.478
  versus 0.285 and +0.048 versus the corrected 0.715 marginal; gold-only rescue
  was 0.408 versus 0.163 wrong-choice-only. Twenty-eight tests and Ruff passed.
- 2026-07-22 — F4 passed — the default WebSocket now runs a prewarmed F2b+F3
  processor and the browser renders neck/fret/hand/position/confidence/guidance
  overlays. Twenty-five tests, Ruff, JavaScript syntax, and browser smoke passed.
  A real public-cache localhost run reached 21.512 FPS with 39.450 ms median and
  120.752 ms p95 end-to-end latency; all 30 measured frames retained neck lock.
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

## LIVE TEST 1 checklist (Pat; A1 + A4, ≤3 min)

1. Run `cd fretcam; .venv\Scripts\fretcam --port 8766`, wait for startup to
   finish, then open `http://127.0.0.1:8766` and select **Start camera**.
2. In normal light, frame the full neck and time green-quad/fret-tick lock
   (target ≤3 s); play normally for 20 s and note tick drift or hand dropouts.
3. Repeat the framing/20 s check in a second, dimmer or differently angled
   light; follow the guidance line once if it asks for a framing correction.
4. Record HUD FPS and End-to-end for the final 10 s (targets ≥10 FPS and
   ≤150 ms), perceived lag, CPU fan, and the three biggest visual annoyances.
5. Paste: `A1 lock: PASS/FAIL — <times, drift, two-light notes>`; `A4 feel:
   <fps>, <E2E ms>, <ok/laggy>, fan <ok/loud>`; `first impressions: 1)… 2)…
   3)…`; `verdict: proceed / fix first: …`.
