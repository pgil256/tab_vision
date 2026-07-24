# FretCam live accuracy phase 3

Date: 2026-07-24
Branch: `codex/fretcam-accuracy-phase-2`

This phase completes the four requested follow-up workstreams: exact live-path
reproduction, constrained whole-hand tracking, adaptive hand-search scheduling,
and an explicit local failure marker. It adds no model, dependency, download,
training run, normal-browser recording, or change inside `tabvision/`.

## What changed

### Exact browser-packet trace and offline comparison

- Normal camera and echo sockets construct no capture object and retain no
  packet. Capture begins only after **Start exact comparison trace** is
  selected.
- Trace start resets live inference and records the clean session prefix. The
  non-rolling buffer preserves at most 10 seconds, 120 JPEG packets, or 24 MB;
  once a bound is reached, later packets are ignored rather than replacing the
  prefix.
- The browser sends each binary JPEG before its context metadata, so capture
  instrumentation does not delay the measured frame. The server joins context
  to the processed frame by a per-socket sequence number.
- Saved manifests contain exact JPEG hashes, generated relative packet paths,
  browser source/inference dimensions, JPEG quality, packet size, browser and
  processor timestamps, the complete HUD response, and nonvolatile detector
  decisions. They contain no camera identifier or absolute local path.
- The loader verifies monotonic sequence numbers, hashes, exact byte counts,
  JPEG dimensions, HUD dimensions, capture bounds, policy, and package type
  before replay.
- Missing, malformed, repeated, or byte-mismatched browser context aborts and
  clears the unsaved capture. A frozen trace still consumes later context
  packets cleanly, so reaching a cap never prevents the preserved prefix from
  being saved.
- Offline replay applies recorded player controls before the final reset and
  compares position evidence, blockers, contacts, geometry, detector
  scheduling, hand-search attempts, and pose decisions frame by frame.
  Volatile latency is deliberately excluded. Asynchronous divergence is
  reported rather than hidden.
- Handedness and calibration changes are blocked while a trace is active.
  Cancel, disconnect, or camera restart clears unsaved bytes. A failed disk
  write leaves the buffer available for a retry.

### Whole-hand constrained pose and identity tracking

- Palm pose uses a robust similarity fit over the wrist and four metacarpal
  anchors, with outlier-resistant residuals for translation, rotation, and
  scale.
- Chirality, palm orientation, finger proportions, joint geometry, detector
  innovation, and temporal continuity are evaluated separately. A fresh,
  hard-incompatible candidate cannot win merely through detector confidence or
  box overlap.
- Thumb-only observations cannot replace a fretting-hand track. All four
  fretting fingers carry independent quality, anatomical consistency, source,
  and freshness.
- Coherent pose velocity can bridge at most 180 ms of blur or occlusion.
  Predicted pose quality remains below the contact threshold, so pose-only
  continuity can guide reacquisition but can never assert a pressed finger.
- Pose identity remains available for 350 ms, independently of the 180 ms
  prediction/contact lifetime. The hard wrong-hand veto therefore remains
  active across the healthy 5 Hz detector interval.
- Resolution changes rescale both tracker and detection-chain state, including
  last-hand crops and motion references.

### Adaptive hand detector and search schedule

- Acquiring, shifting, and recovery states request hand refreshes at up to
  15 Hz. A healthy lock backs off to 5 Hz while optical flow carries
  intervening landmarks.
- The previous estimator state, blockers, geometry freshness, pose quality,
  and track age feed the next frame's schedule.
- A board-pending frame does not spend work on MediaPipe. The due request is
  retained for the first usable fretboard frame.
- Full-neck, last-known-hand, recovery-neck, and periodic full-video searches
  are ordered from live state. Periodic identity checks jointly score the full
  frame and narrow crop instead of blindly replacing the better result.
- No frame can perform more than two hand-detector calls. Diagnostics report
  every actual attempt, consumed asynchronous result, acceptance decision,
  schedule mode, interval, and selected search source.
- Extractor signatures are inspected before inference. An internal
  `TypeError` is propagated after one call rather than being mistaken for a
  legacy signature and retried invisibly.
- The public frame benchmark and exact replay use the same one-frame feedback
  helper as the browser path, preventing a benchmark-only fixed cadence.

### Explicit local failure marker

- **Enable 2-second failure buffer** starts a rolling in-memory buffer capped
  at 24 packets / 6 MB. It is off after every load, reconnect, camera restart,
  and disconnect.
- A save requires a second explicit action plus the user's expected Position
  I-XII (or unknown), the fingers actually pressing, and an optional
  240-character note.
- Failure packages use the same exact-packet integrity checks as traces but
  are a separate private diagnostic type. The trace comparator, public
  benchmark, local evaluation set, threshold tuning, training, and release
  evidence all reject them.
- Capture controls are accepted only from the exact loopback origin serving the
  WebSocket. Diagnostic roots inside Git repositories or through symlinked
  paths are rejected.

## Accuracy safeguards

- A hard pose mismatch vetoes the wrong hand while the tracked identity is
  fresh.
- After more than 250 ms without evidence, the same numeric position must
  satisfy a fresh 250 ms stability interval; the old label cannot immediately
  reappear during a shift.
- Before the first lock, a replaced boundary candidate starts a new acquisition
  history. Abandoned VI evidence therefore cannot suppress a later stable VII
  candidate after the full lock interval.
- Curled and visible fingers still require per-finger quality and physical
  press evidence. Pose continuity and chord compatibility do not bypass that
  gate.

## Verification

The final public development replay produced:

| Metric | Phase-3 result | Phase-2 checkpoint |
|---|---:|---:|
| Displayed-position precision | 67/67 (1.000) | 77/77 (1.000) |
| Stable-frame coverage | 67/161 (0.416) | 72/161 (0.447) |
| Stable false locks | 0/161 | 0/161 |
| Negative-control displays | 0/120 | 0/120 |
| Valid stable observations | 109/161 (0.677) | not recorded here |

The five-frame coverage reduction is intentional abstention from the new
long-gap stale-lock safeguard. Development replay briefly exposed four old
Position-II displays during a labeled shift; the safeguard removed all four.
The boundary-acquisition correction then recovered two safe Position-VII
displays without a wrong or negative-scene lock.

A final real localhost WebSocket replay of the longer Position-I barre
sequence displayed 34/34 positions correctly, covered 34/54 stable frames, and
produced zero false locks. It sustained 10.001 effective FPS with 60.0 ms
median end-to-end and 55.9 ms median server time; p95 end-to-end was 161.4 ms.

Verification also passed:

- 223 FretCam tests, one skipped test, and five parameterized subtests;
- focused pose, hard-identity, thumb-only, resolution-change, pose-only
  contact, asynchronous-board, detector-call-bound, trace-integrity,
  disk-retry, exact-origin, replay-order, and evidence-gap regressions;
- Ruff checks and formatting for every changed Python file;
- JavaScript syntax and Git whitespace checks;
- a paced 10 FPS, quality-72, 640 px public-media replay through the real
  uvicorn/FastAPI WebSocket;
- rendered in-app-browser inspection with the expected live HUD, capture tools
  collapsed and off, camera-dependent controls disabled, no error overlay, and
  no console warnings or errors.

The source-disjoint held-out split was not opened. Machine-local benchmark,
trace, and failure artifacts were not committed.
