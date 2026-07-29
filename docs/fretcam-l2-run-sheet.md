# FretCam L2 — controlled-live acceptance run sheet

The §6 protocol from `docs/plans/2026-07-22-fretcam-live-position-hud-design.md`,
turned into something runnable in one sitting. **L2 is the gate that blocks the
FretCam promotion path and the C2 live assisted-review A/B** — both are waiting
on this, not on compute.

Budget: ~15 minutes with the guitar, ~3 minutes to fill in the report.

Live inference only — nothing is recorded or persisted, so this creates no
private-recording artifact and does not touch the training/eval ban.

## 0. Environment (already prepared 2026-07-27 — no setup needed)

The camera is read by the **browser** (`getUserMedia` in `static/client.js`),
which sends JPEG frames over a WebSocket; the server never opens a camera
device. So the server runs in WSL where the models already live, and Chrome on
Windows supplies the camera. No Windows Python is required.

> That matters because Windows Python here is 3.13, and **MediaPipe publishes no
> cp313 wheel for any 0.10.x release** (verified against the PyPI API — the
> `mediapipe>=0.10,<1` pin in `fretcam/pyproject.toml` cannot be satisfied on
> 3.13). A native Windows install would have required installing Python 3.12
> first. The browser-camera architecture sidesteps that entirely.

Prepared and verified on 2026-07-27:

- `fastapi`, `uvicorn[standard]`, `websockets` installed into `tabvision/.venv`
  (the venv already had mediapipe / opencv / ultralytics / numpy);
- YOLO checkpoint present at `~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`;
- MediaPipe `hand_landmarker.task` present at `~/.mediapipe/models/`;
- server smoke-tested: `HTTP 200`, `<title>FretCam</title>`, camera + calibration
  controls served.

**Re-verified 2026-07-28, and the server is already running.** Beyond the HTTP
check, the live path was exercised end to end this time: `/health` returns
`{"status":"ok","mode":"hud"}`, and a synthetic JPEG pushed over `ws://…/ws`
came back as a HUD JSON payload in 0.35 s on the first frame (model prewarm)
then 0.10 s / 0.06 s. So the WebSocket, decode and inference path are known-good
before you pick up the guitar — if nothing appears, suspect the *camera
permission*, not the server. The WSL IP is still `172.24.194.6`.

> **CPU note (updated 2026-07-28):** Phase D completed and its extraction no
> longer runs — the contention that invalidated attempt 1's A4 is gone. Check
> `uptime` shows a low load before starting anyway; if anything heavy is
> running, pause it first, or treat a failing A4 as inconclusive rather than
> a FAIL — per §5 an A4 failure is an environment verdict, not an accuracy
> one.

Model prewarm runs automatically on connect (`app.py:149` → `processor.warmup`),
so A1 measures camera acquisition rather than a one-time model load — this is
the F4 fix for the 8.5 s cold start.

## 1. Start

**As of 2026-07-28 the server is already up on port 8765** — skip to opening
Chrome. Check it is still alive with:

```bash
wsl -d Ubuntu -- bash -c "curl -s http://127.0.0.1:8765/health"
```

If that returns nothing, start it again:

```bash
wsl -d Ubuntu -- bash -c "cd /home/gilhooleyp/projects/tab_vision/tabvision && .venv/bin/fretcam --host 0.0.0.0 --port 8765"
```

Note `.venv/bin/fretcam` — the console script. `python -m fretcam.cli` exits
silently with status 0 because `cli.py` has no `__main__` guard.

Then open **Chrome on Windows** at <http://127.0.0.1:8765> and click
**Start camera**. If WSL localhost forwarding misbehaves, use
<http://172.24.194.6:8765> (both are treated as secure origins, so the camera
permission prompt still appears).

Set **right/left-handed player mode** below the preview before starting.

## 2. What to expect before you judge it

The frozen dev benchmark (F5c) reads **precision 1.000 at coverage 0.416** —
the build is tuned to abstain rather than guess. So the expected failure mode is
**no readout**, not a wrong readout. A hold with no position displayed counts as
a miss for A2, because the bar is "readout correct in ≥90% of holds."

Two changes landed after attempt 1 (2026-07-28):

- **Re-acquire board button** (first button in the settings row). If the
  overlay ever shows a garbage non-rectangular quad — the known failure when
  the guitar enters the frame mid-session — click it and re-frame the neck
  instead of restarting the camera. It clears tracking and the position
  estimator; handedness and session calibration are preserved.
- **Honest fingertip guidance.** When a hand is visible but fewer than 3
  fretting fingertips are on the neck, the HUD now says so explicitly
  ("place 3 or more fingertips on the neck to lock") instead of the generic
  hand message. Single-finger holds may not lock **by design** (the F4c
  validity gate). Score them per the protocol regardless — a no-readout hold
  is still a miss — and note in the report how many misses were
  single-finger cells, so the protocol/build collision recorded in
  `docs/fretcam-loop-state.md` gets its number.

Known weak spots from the F4d category breakdown, listed so you can report
against them rather than rediscover them: **full-neck framing** had far lower
coverage than close framing (0.092 vs 0.473), **mixed lighting** raised false
locks, and **chords** were weaker than single notes. F5/F5b/F5c addressed these
(false locks went to 0), but they are the places to watch.

## 3. The protocol

### A1 — lock (2 min)
Frame the neck. The overlay must lock (fret ticks visibly on the frets) **within
3 s**, and hold through normal playing motion. Repeat under a **second lighting
condition**.

- [ ] locks ≤ 3 s, lighting 1 — time: ____
- [ ] locks ≤ 3 s, lighting 2 — time: ____
- [ ] holds through playing motion (no drift off the frets)

### A2 — position accuracy (6 min) — the main event
Five-second holds. **Bar: ≥ 90% correct (≥ 14/15), and never off by more than
one position.**

| position | single note | chord | barre |
|---|---|---|---|
| I | ☐ | ☐ | ☐ |
| III | ☐ | ☐ | ☐ |
| V | ☐ | ☐ | ☐ |
| VII | ☐ | ☐ | ☐ |
| IX | ☐ | ☐ | ☐ |

Mark ✓ correct, ✗ wrong (note what it read), — no readout (counts as a miss).
Tally: ____ / 15. Worst case seen: ____________

### A3 — shifts and occlusion (3 min)
Run I → V → IX.

- [ ] shows "shifting…" during the move (not a stale or wrong number)
- [ ] correct label **≤ 500 ms** after arrival — worst observed: ____ ms
- [ ] recovers from **full hand occlusion ≤ 1 s** — observed: ____ s

### A4 — throughput (1 min)
Read the on-page FPS/latency figures.

- [ ] **≥ 10 FPS** end-to-end — observed: ____
- [ ] readout latency **≤ 150 ms** — observed: ____
- [ ] CPU/fan acceptable

## 4. Report (paste into `docs/fretcam-loop-state.md`)

```markdown
### L2 — 2026-__-__
setup: <camera position / lighting / guitar>   duration: <min>
A1 lock: PASS/FAIL — <lock time, drift, notes>
A2 positions: <x>/15 correct (I/III/V/VII/IX × note/chord/barre) — worst: <...>
A3 shifts: PASS/FAIL — <label lag, "shifting…" behavior, occlusion recovery s>
A4 feel: <fps shown> fps, latency feel <ok/laggy>, CPU fan <ok/loud>
top annoyances: 1) <...> 2) <...> 3) <...>
verdict: <proceed / fix first: ...>
```

## 5. What each outcome unlocks

**Pass** → F8's promotion path opens (still needs a larger frozen evaluation
before `fretcam` could ever become the default), and **C2** — the live
assisted-review A/B, where anchors re-rank the review queue against the shipped
38.76% wrong-position reduction — becomes runnable.

**Fail on A2/A3** → opens **F6**, the hand-bbox × fret-zone IoU fallback
(the TapToTab mechanism). Its dataset dependency is now cleared:
`ghaleb/guitar-fretboard` is verified **CC BY 4.0** (384 images, `Hand` +
`Zone1..Zone12`). Per the loop rules F6 still needs explicit approval before
acquisition, and a second failure closes the side quest with an honest negative.

**Fail on A1/A4** → environment or throughput problem, not an accuracy verdict;
re-run rather than routing to F6.
