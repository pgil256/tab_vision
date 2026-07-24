# FretCam loop — operating prompt for Claude sessions

Start a session with: "Read docs/prompts/fretcam-loop.md and execute one
iteration." Headless: `claude -p "$(cat docs/prompts/fretcam-loop.md)"`.
After a live test, start the session with your observations pasted in —
recording them is the iteration's first job.

---

You are working in the TabVision repo. Mission, sustained across sessions:
build and validate **FretCam** (live webcam fretboard + position HUD) per
`docs/plans/2026-07-22-fretcam-live-position-hud-design.md`, one bounded item
per iteration, ending in an M4 bridge verdict (evidence for/against wiring
position anchors into TabVision fusion). You have no memory between sessions —
`docs/fretcam-loop-state.md` is your memory.

**Two iteration types.** `F*` items are BUILD/PROBE work you do. `L*` items
are LIVE TESTS only Pat can run (guitar + webcam). You never mark an `L*`
item done — Pat's filled report does. If the topmost open item is an `L*`
awaiting Pat, pick the next open unblocked `F*` instead (F7 is designed as
fill-in work).

## Startup (every iteration, in order)

1. If Pat's message contains live-test observations, record them verbatim
   into the state file's live-test log and update affected queue items FIRST.
2. Read `CLAUDE.md`, the FretCam design doc (§2, §3, §6, §7 minimum), then
   `docs/fretcam-loop-state.md` (create from the template below if missing).
3. Check `docs/DECISIONS.md` entries newer than `last_updated` — work may
   have happened outside the loop.
4. Pick the topmost open unblocked item. One item per iteration.

## Work protocol

- **FretCam is quarantined.** All code in `fretcam/` at repo top level, own
  `pyproject.toml`/venv; imports `tabvision` as a library. **Zero edits inside
  `tabvision/`, `SPEC.md`, or §8 contracts** — M4 integration, if it ever
  happens, is a separate program with its own gates.
- **Headless verification first.** Every build item ships with what CAN be
  verified without a human: unit tests (anchor math, hysteresis, projection),
  replay against the local GAPS mp4 cache (`~/.tabvision/cache/gaps_video/`,
  NC, precedent: chunk-5/6), synthetic fixture frames/trajectories, and
  measured per-stage latency on this laptop. Live tests are for what only a
  live camera can show — not a substitute for tests you skipped.
- **Data hygiene (hard rule).** Private/user recordings: never in any
  training, eval, or label role. A Pat-recorded debug clip may be used ONLY
  with explicit per-clip approval, only to reproduce a functional bug (crash,
  tracking dropout), never to produce a reported metric or tune a threshold,
  never committed, deleted after. Accuracy evidence = live-test reports (§6
  protocol, subjective pass/fail) or public footage.
- **Pre-approved deps** (fretcam venv only): `fastapi`, `uvicorn[standard]`,
  `websockets`, plus anything `tabvision.video.*` already imports
  (mediapipe, ultralytics, opencv, numpy). Anything else → STOP.
- **Timebox.** One item per iteration; if something will exceed ~2 h,
  checkpoint, record how to resume, end the iteration.
- **Banked negatives are wins.** A failed gate gets written down with numbers
  and the item goes `closed-negative`. Do not iterate past a failed gate
  hoping. If §6 fails after the F6 fallback, the side quest ends honestly
  (house rule 7) with a DECISIONS entry.

## STOP and ask (end the iteration with the question) when:

- Anything costs money, or needs an account/download (Roboflow export of the
  ghaleb dataset needs a free account — still ask first).
- Any model (re)training run, even free Colab.
- A new dependency beyond the pre-approved list.
- Using any Pat-recorded clip (per-clip, see hygiene rule).
- Anything would touch `tabvision/` package code, SPEC, or `auto` behavior.
- An `L*` item is next — schedule it with Pat; print the checklist.
- A result contradicts the design doc's assumptions (e.g. the OBB detector
  can't find the neck in webcam-style frames at all) — replan, don't improvise
  a new architecture mid-iteration.

## End of iteration (always)

1. Update `docs/fretcam-loop-state.md`: statuses, key numbers, exact next
   action, `last_updated`.
2. Commit on a work branch cut from `main`, one per item
   (`fretcam/f2-detection-chain`). Do not push or merge unasked.
3. Print ≤10 lines: item, verdict (pass/fail/blocked/in-progress), key
   numbers, files, single next action. If the next item is an `L*`, print
   its checklist verbatim as the last thing.

## Work queue (details in design doc §3, §6, §7)

- **F1 — scaffold.** `fretcam/` package: FastAPI + WS server skeleton,
  static page (getUserMedia → canvas → WS frames), echo-mode roundtrip,
  browser FPS meter. Verify: WS roundtrip unit test with synthetic JPEG;
  measured loopback latency in the state file.
- **F2 — detection chain.** `process_frame()`: OBB checkpoint
  (`~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`) → homography +
  fret map (`tabvision.video.fretboard`) → MediaPipe hand →
  `neck_anchor.compute_neck_anchor`. Verify on GAPS cached frames + fixtures;
  record per-stage ms. Detector at 1–2 Hz + tracking between; hands every
  frame. Gate: neck lock + plausible center_fret on ≥ 3 distinct GAPS clips.
- **F3 — position estimator.** Index-fret → Roman position + window
  [N−1,N+4] ∪ {0}; EMA + k≈5 hysteresis; "shifting…" state; confidence per
  design §2. Verify: unit tests on synthetic trajectories (shifts, dropouts,
  jitter); replay a GAPS clip and eyeball-check the overlay video artifact.
- **F4 — HUD + guidance.** Neck polygon + fret ticks + hand marker +
  position label + confidence bar + framing guidance line; end-to-end FPS/
  latency measured. Output: LIVE TEST 1 checklist. Then STOP for Pat.
- **L1 — live test 1 (Pat).** A1 lock + A4 throughput + first impressions.
  Fill the report template into the state file.
- **F5 — fix round.** Address L1's top annoyances; re-verify headlessly;
  output LIVE TEST 2 checklist (full §6: A1–A4). STOP for Pat.
- **L2 — full acceptance (Pat).** §6 protocol: 5-s holds at I/III/V/VII/IX
  × note/chord/barre, shift runs, occlusion recovery, two lightings.
- **F6 — fallback (conditional; open only if L2 fails on hand dropouts).**
  Hand-bbox × fret-zone IoU (TapToTab mechanism) — needs the ghaleb
  dataset and possibly detector retraining → STOP for approval first.
  Re-run L2 after. Second failure closes the quest.
- **F7 — GAPS anchor probe (independent; fill-in work).** Cache-only:
  cached per-frame fingerings + homographies → centroid fret-window;
  score **P(gold fret ∈ window | audio wrong)** on the banked ambiguous
  lattice, compare against A14's 0.285 anti-enrichment baseline and the
  0.778 audio prior. Pure offline, $0. A negative does NOT close FretCam
  (wrong capture contract) — it calibrates M4 expectations. Report to
  `docs/EVAL_REPORTS/fretcam_gaps_anchor_probe_<date>.md`.
- **F8 — M4 bridge verdict (blocked by L2 pass + F7 report).** Synthesize:
  design the assisted-review A/B (anchors re-rank the review queue; target
  > 38.76% wrong-position reduction @60 s, assisted metric, reported
  separately). STOP before writing any integration code — that's a new
  program with user sign-off, not a loop item.

## Live-test report template (Pat fills; ≤3 min)

```markdown
### L<n> — <date>
setup: <camera position / lighting / guitar>   duration: <min>
A1 lock: PASS/FAIL — <lock time, drift, notes>
A2 positions: <x>/15 correct (I/III/V/VII/IX × note/chord/barre) — worst: <...>
A3 shifts: PASS/FAIL — <label lag, "shifting…" behavior, occlusion recovery s>
A4 feel: <fps shown> fps, latency feel <ok/laggy>, CPU fan <ok/loud>
top annoyances: 1) <...> 2) <...> 3) <...>
verdict: <proceed / fix first: ...>
```

## State file template (`docs/fretcam-loop-state.md`)

```markdown
# FretCam-loop state
last_updated: <date>
current_branch: <branch or none>

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| F1 | scaffold | open | — | create fretcam/ | — |
| F2 | detection chain | open | — | — | F1 |
| F3 | position estimator | open | — | — | F2 |
| F4 | HUD + guidance | open | — | — | F3 |
| L1 | live test 1 | blocked | — | — | F4 |
| F5 | fix round | blocked | — | — | L1 |
| L2 | full acceptance | blocked | — | — | F5 |
| F6 | IoU fallback | conditional | — | — | L2 fail |
| F7 | GAPS anchor probe | open (fill-in) | vs 0.285 / 0.778 | — | — |
| F8 | M4 bridge verdict | blocked | — | — | L2 + F7 |

## Questions for Pat
- <none>

## Live-test log (newest first)
- <none yet>

## Iteration log (newest first)
- <date> — <item> — <one-line outcome>
```
