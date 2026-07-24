# FretCam-loop state
last_updated: 2026-07-24
current_branch: codex/fretcam-accuracy-phase-2

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
| F4f | explicit live browser fretboard + hand-position display | passed | glowing green neck border; dedicated live status cards; browser HUD live with 0 console errors; 56 tests | preserve; complete | — |
| L1 | live test 1 (Pat: A1+A4) | accepted for now (informal) | Pat: "Im telling you its fine for now."; no formal numeric A4 claim | no further action unless Pat reopens it | — |
| F5 | accuracy/performance/product overhaul | passed | 106 tests; dev precision 76/80; coverage 71/161; stable false locks 0/161; negative displays 0/120 | preserve and run L2 only if Pat reopens formal acceptance | — |
| F5b | neck/landmark/contact/upper-neck accuracy phase | passed | 172 tests + 5 subtests; dev precision 77/77; coverage 72/161; stable false locks 0/161; negative displays 0/120 | preserve; do not open held-out split | — |
| F5c | exact trace + pose identity + adaptive search + failure marker | passed | 223 tests + 5 subtests; dev precision 67/67; coverage 67/161; stable false locks 0/161; negative displays 0/120 | preserve diagnostics privacy; do not open held-out split | — |
| L2 | full §6 acceptance (Pat) | blocked | A2 ≥90% of holds | — | F5 |
| F6 | IoU fallback (TapToTab mechanism) | conditional | — | needs ghaleb dataset → STOP first | opens on L2 fail |
| F7 | GAPS anchor probe (cache-only, fill-in) | completed-positive | corrected 1195/1566 = 0.763 (CI 0.741–0.783); +0.478 vs 0.285; old 0.247 preserved as superseded | preserve fixed result; no tuning | — |
| F8 | M4 bridge verdict | implemented-opt-in | production-aligned causal cache proxy 0.800111→0.800665; +6 net / 0.28% relative error reduction; full suites 932 TabVision + 240 FretCam pass | preserve explicit rollback; run controlled-live + frozen real-audio gate before promotion | L2 + held-out promotion evidence |

**Live checkpoint.** F5c is complete. The final dev-only frozen benchmark
reached displayed precision 67/67 (1.000), stable coverage 67/161 (0.416),
stable false locks 0/161, and negative-control displays 0/120. The small
coverage reduction from F5b is intentional abstention after a long evidence
gap; it removed four stale Position-II displays during a labeled shift. The
source-disjoint test split was not opened during F5c. A real WebSocket replay
of the longer Position-I sequence displayed 34/34 correctly with no false
lock at 10.001 effective FPS and 60.0 ms median end-to-end. The rendered
local-browser check found the expected live HUD and
opt-in accuracy tools, no error overlay, and no console warnings or errors.

Pat's live observation is positive and accepted for now. The final direction
was: "Im telling you its fine for now." No numeric FPS/E2E value is inferred,
and no formal A4 threshold claim is made. Do not request more L1 measurements
unless Pat reopens the gate.

On 2026-07-24 Pat explicitly requested the FretCam/audio integration. The M4
bridge now exists as `--video-backend fretcam` without changing §8. It uses
only stabilized coarse position windows on the demux media clock, with causal
pre-onset selection, open/capo support, and a default-policy one-nat cap; the legacy
per-string posterior is excluded on this route. The corrected-cache causal
proxy is directionally positive but small (+6/10,821 assignment-scored), so `legacy`
remains the default and L2/held-out promotion evidence is still required.

## Standing constraints (from the loop prompt — do not relax silently)
- The 2026-07-24 user request lifted quarantine only for the bounded M4 bridge
  inside `tabvision/`; §8 remains immutable and further exact-string/video
  expansion still requires its own gate.
- Private recordings: never in training/eval/label roles; debug clips only
  with per-clip approval, no metrics, never committed.
- Pre-approved deps: fastapi, uvicorn[standard], websockets + existing
  `tabvision.video.*` imports. Anything else stops the iteration.
- Training runs and Roboflow downloads: STOP for approval.

## Questions for Pat
- None. Await Pat's next direction. F4e-B still requires separate approval.

## Live-test log (newest first)
- 2026-07-22 — L1 disposition, verbatim: "Im telling you its fine for now."
  Record the current live behavior as informally accepted; do not infer or
  claim numeric A4 performance.
- 2026-07-22 — L1 numeric follow-up, verbatim: "2-8". Units and ordering are
  not explicit, so this is not yet classified as an A4 pass or failure.
- 2026-07-22 — L1 follow-up, verbatim: "No crazy dropouts. alternate lighting
  worked. fps ee2e numbers were visible and low." Lighting and dropout evidence
  are positive; the A4 result remains ambiguous until FPS and E2E are separated
  or approximated.
- 2026-07-22 — L1 follow-up, verbatim: "Time seemed reasonably low." This is
  recorded as a positive timing impression, not a numeric lock or latency
  result.
- 2026-07-22 — L1 preliminary, verbatim: "It seems like it works ok." No lock
  time, drift/dropout observation, FPS, end-to-end latency, or two-light result
  was supplied, so the formal L1 status remains open.

## Iteration log (newest first)
- 2026-07-24 — F8 implemented as explicit opt-in — added a synchronous
  media-clock FretCam adapter and a causal bounded position-window prior.
  `locked`/`holding` evidence at confidence ≥0.20 supports exactly
  `{open/capo} ∪ [N-1,N+4]` in the 150 ms lookback ending onset-30 ms, capped at one
  nat. The route excludes legacy `FrameFingering` evidence and retains
  `legacy` as default rollback. The production-aligned corrected-cache proxy
  moved exact position accuracy 0.800111→0.800665 (+6 net). Full suites passed
  932 TabVision and 240 FretCam tests. No held-out split opened.
- 2026-07-24 — F5c passed — added explicit bounded exact-packet traces with
  validated offline frame comparison, robust whole-hand pose/identity
  tracking, estimator-driven 15/5 Hz hand-search scheduling with at most two
  detector calls per frame, and a private two-second failure marker carrying
  expected position/fingers. Long-gap reacquisition now prevents stale labels,
  while initial candidate replacement no longer poisons a later stable lock.
  The final dev-only result is 67/67 displayed precision, 67/161 stable
  coverage, 0/161 stable false locks, and 0/120 negative displays. Two hundred
  twenty-three tests plus five parameterized subtests, Ruff, JavaScript syntax,
  exact-trace integrity checks, real-WebSocket replay, and rendered-browser
  verification passed; the held-out split remained closed.
- 2026-07-23 — F5b passed — added immediate neck-guided all-hand acquisition,
  timestamped VIDEO-mode refreshes with per-joint optical flow/One Euro
  tracking, physical finger-pad/hover/press/barre evidence, independently
  gated nonlinear upper-neck geometry, two-point Position I + V/IX
  calibration, a real-WebSocket accuracy matrix, and public/synthetic-only
  local finger-label tooling. Production replay found and fixed unsupported
  body-joint axes and a picking-hand boundary false lock. The accepted
  dev-only result is 77/77 displayed precision, 72/161 stable coverage, 0/161
  stable false locks, and 0/120 negative displays. One hundred seventy-two
  tests plus five parameterized subtests, Ruff, JavaScript syntax, and
  rendered-browser checks passed; the held-out split remained closed.
- 2026-07-23 — F5 passed — replaced index-dominant locking with a
  multi-finger/contact solver, time-weighted confidence and elapsed-time
  hysteresis; added asynchronous YOLO plus optical board tracking and
  implausible-geometry rejection; capped/adapted inference work; and shipped
  camera, handedness, mirror, calibration, why-not-locked, diagnostics, and
  stale-border controls. The final dev benchmark reached 76/80 displayed
  precision, 71/161 coverage, 0/161 stable false locks, and 0/120 negative
  displays. One hundred six tests, Ruff, JavaScript syntax, and desktop/mobile
  live-browser verification passed.
- 2026-07-22 — F4f passed — the browser now gives the fretboard and hand
  position their own persistent live readouts, and draws a thicker glowing
  green border around the detected neck. Start/stop and acquiring/locked/held
  states update live. Browser verification also found and fixed an empty-state
  overlay that could remain above live video; the verified page reached HUD
  LIVE with the overlay hidden and no console errors.
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
