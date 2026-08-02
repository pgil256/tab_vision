# Handoff — video evidence closeout, 2026-07-28

Successor to `docs/HANDOFF-2026-07-27-video-evidence.md`. Read that one for how
the video track got here; read this one for what happened to it.

**One-line summary:** three of the four open video lines were tested to
completion and **all three failed**, each banked with a mechanism rather than a
shrug. The fourth (C1) was unblocked. Nothing about the shipping pipeline
changed, and nothing should — the honest state is that the geometry we already
have beats every alternative tried today.

**Spend:** roughly **$2** total on Modal L4 (two E2 runs, one Gate 1 run). No
other paid resource was touched.

---

## 1. Results banked

| line | verdict | number |
|---|---|---|
| **Phase D** — learned string resolver retrain | **FAIL** | val 6-way **0.2919** vs a 0.45 bar |
| **Phase E2** — learned fret keypoints | **FAIL** | **0.6305** vs the OBB fit's 0.7195 |
| **Wire-sparse calibration gate** | **FAIL** | LOO **−0.0043** vs ungated |
| **C1** — assisted-review ranker | **unblocked** | tolerance ±25 rows, cause localised |

Reports: `docs/EVAL_REPORTS/phaseD_gate1_2026-07-28.md`,
`e2_fret_keypoints_2026-07-28.md`, `wire_sparse_calibration_gate_2026-07-28.md`.
Decision records: five new entries in `docs/DECISIONS.md`.

### The three negatives, and what each actually refutes

**Phase D refutes its own documented root cause.** WS4 plateaued at ~0.30 and
carried a confident diagnosis — the whole-neck crop starves the model, and
onset-frame labels are misaligned — plus a committed fix (`--hand-tight`,
`--sustain`). Both were executed for the first time, everything else frozen, on
159,381 train / 22,556 val crops from 241 clips. The plateau did not move
(0.2919). **Crop framing and label alignment are not what limits that model.**
The signature is overfitting — val peaks at epoch 3 while train loss keeps
falling 1.62 → 0.34 — so the negative is scoped to this configuration, not to
learned string resolution generally.

**E2 refutes "learned keypoints beat the RANSAC fit".** A `yolo11n-pose` model
trained to pose mAP50 0.7399 (fret class 0.854) scores **0.089 worse** than the
OBB consensus fit it was meant to replace. It wins the wire-sparse subset
(0.7222 vs 0.6603) but two of those four clips have it firing at 0.000, so the
gain rests on one clip.

**The wire-sparse gate refutes a lever this session itself proposed.** Per-clip
fire rate correlates strongly with calibration benefit (Spearman **+0.797**), but
calibration is never *harmful* — it is net-positive even in the low-fire half
(+0.072). Exactly one clip of twelve has a negative delta. The E2 §6 observation
was an outlier dragging a four-clip average, not a threshold effect.

**Phase A's +0.151 calibration gain is unaffected and reinforced** by all of
this: calibration helps on 11 of 12 clips, by up to +0.385.

## 2. Method — the parts worth copying

**Pre-register before running, and commit it first.** The gate experiment's
design was committed as `a8f5f2e` *before* any number existed. That is not
ceremony: the obvious version of that experiment is **circular**. The wire-sparse
subset was *defined* as "fires below 0.50" and the harm was *observed* inside it,
so gating at 0.50 and scoring the same clips reduces to an arithmetic identity
that comes out positive whether or not fire rate carries information. Leave-one-
clip-out — threshold fitted only on the other eleven — tests generalisation
instead, and it failed. Had the circular version been run it would have
"confirmed" the lever and shipped a useless gate.

**Fix one variable.** The E2 A/B shares the cached homography (never re-fit),
`fit_fret_map`, the nut anchoring, the canonical window, `_MIN_WIRES` and the
frames. Only the *source* of wire positions differs. Two asymmetries were found
and corrected before the verdict was trusted:

- the keypoint arm first ran at detection conf 0.25 while Phase A's OBB pass ran
  at **0.10**. The cache path now encodes the floor so the two can never be
  silently interchanged.
- Phase A's crop pass **dedupes** detections before they reach
  `calibrate_fret_xs`; the keypoint arm had no equivalent. Measured: median
  minimum adjacent canonical gap 0.003 vs 0.024. Adding dedupe moved the keypoint
  arm 0.5745 → 0.6305.

That second fix landed **after a FAIL was already visible**, so both numbers are
reported and the report says plainly that the fix is justified by a mechanism
diagnostic and by symmetry with what the other arm already receives — not by the
metric moving. The verdict is identical either way.

**Diagnose the mechanism before believing a number.** Every negative here has a
mechanism: the keypoint arm's problem was `fit_fret_map` rejecting duplicated
wires (not detection — its wire counts match OBB); the gate's problem is visible
in the LOO picks, which gate a *helpful* clip and miss the harmful one; Phase D's
problem is an overfitting curve, not a flat one.

**Do not convert a failure into a search.** Fire rate was fixed in advance as
*the* statistic. Homography confidence, inlier counts, fit RMS and per-frame
gating are all plausible successors and were all deliberately left untried;
trying them after fire rate failed is exactly the fishing the pre-registration
existed to prevent. Same for Phase D: the epoch-3 peak points at
regularisation/capacity, and that is recorded as needing its own pre-registration
rather than run as an open hyperparameter sweep.

## 3. Corrections made to earlier claims

- **The 2026-07-27 handoff was wrong about Phase D.** It recorded the blocker as
  "CPU, then Gate 1". In fact the "252/270 clips acquired" figure was **video
  only** — the train split had no gold annotations and no reference audio, so the
  extraction returned 0 crops in 3.6 s and could never have produced a training
  crop. A warning block now heads that document.
- **The E2 report's §6 framing was corrected the same day.** "On wire-sparse
  clips calibration is net-harmful" is accurate about that subset but invited
  reading sparseness as the *cause*. It is one clip. A warning box now heads that
  section and points at the refutation.
- **`118_VD1wc` is diagnosed, not just flagged.** Three measurements singled it
  out (Phase A's largest regression, the only clip calibration harms, the clip
  where the keypoint model sees nothing). Cause: **an extreme foreshortening
  camera angle** — the neck is found fine (hconf 0.878) but perspective compresses
  the wires to 2.99/frame against a healthy 14–23, the nut is detected in 4% of
  frames, and with neither anchor the nut-side test degenerates to a coin flip
  (0.60 vs a decisive 0.01–0.38). About half the fitted maps come out **end-for-end
  reversed** — fret 0 at canonical 0.961 descending to fret 24 at 0.029. `063_bV1wc`
  is the instructive contrast: it sees even *less*, fires 0.000, and is therefore
  harmless. **Seeing nothing is safe; seeing a little is not.**

## 4. Changes to the tree

New:

| file | purpose |
|---|---|
| `scripts/acquire/datasets.py` (`gaps-annotations`) | fetch GAPS musicxml/midi/syncpoints/audio from the HF mirror, split-filtered |
| `scripts/train/yolo_fret_keypoints_modal.py` | E2 pose fine-tune on Modal L4 |
| `scripts/eval/e2_keypoint_cache.py` | cache keypoints over the Phase A frames |
| `scripts/eval/e2_fret_registration_ab.py` | the three-arm E2 A/B |
| `scripts/eval/wire_sparse_gate_ab.py` | LOO test of the calibration gate |
| `scripts/eval/diag_118_pathology.py` | per-clip geometry dump |
| `tests/unit/test_string_assignment_phase6_rows.py` | 7 tests pinning the row tolerance |

Modified:

- `scripts/eval/string_assignment_phase6.py` — `EXPECTED_DEV_ROWS` /
  `DEV_ROW_TOLERANCE = 25` (= floor of 0.05%; 26 was rejected at 0.0509%).
- `docs/HANDOFF-2026-07-27-video-evidence.md`, `docs/fretcam-l2-run-sheet.md`,
  `docs/DECISIONS.md`.

Suite: **1149 passed, 3 skipped**; ruff clean.

### Two traps caught before they cost anything

- **`guitar-fret-6pt` declares `flip_idx: [0,1,2,3,4,5]`** — an identity mapping.
  Mirroring reverses the six string intersections, so under ultralytics' default
  `fliplr=0.5` roughly half of every epoch would have carried transposed string
  labels. `fliplr` is pinned to **0.0**.
- **Roboflow's `data.yaml` uses `train: ../train/images`**, which resolves
  *outside* the dataset dir. The runner rewrites it with absolute paths so a
  missing split fails loudly.

## 5. Still open

- **L2 — attempt 1 was run (by a concurrent session) and needs a clean re-run.**
  Log: `docs/fretcam-loop-state.md`, commit `afd8d6a`. Outcome: A1 pass on
  protocol, **A2 ≈0/15 → fail as run**, A3 not evaluable, A4 **inconclusive**.
  Two findings matter more than the score:
  - **A protocol/build collision.** The F4c ≥3-fretting-fingertip validity gate
    rejects the single-note column of the A2 grid **by design**. F4c postdates
    the §6 protocol and was never re-checked against it, so part of that grid was
    unscoreable before anyone picked up a guitar. Recorded, not re-scored.
  - **A board re-acquisition defect**, load-independent and outside the protocol:
    bringing the guitar into frame *after* the session starts yields a degenerate
    quad with no recovery and no reset control. Every frozen benchmark clip
    starts with the board already in frame, so this path had never been
    exercised live.

  A4 was contaminated because the Phase D extraction was not paused — the exact
  confound `docs/fretcam-l2-run-sheet.md` now warns about, and the reason
  `~/phaseD_pause.sh` exists. **That caveat is now moot: Phase D is finished, so
  a re-run gets the whole box.** Fix the reset control and the guidance text
  first (the "no hand available" message conflates no-detection with
  fingertip-gate rejection while the tracking dots are visibly correct), then
  re-run clean. Only if A2 still fails clean does F6 routing open — and note F6's
  hand-bbox × fret-zone mechanism needs no fingertips and matches the observed
  failure mode exactly.
- **C1 is unblocked but not run.** `string_assignment_phase6.py` now loads the
  regenerated table; producing the actual 38.76% comparison is a multi-hour CPU
  run that has not been done. Results from it must be reported as **"near-exact,
  4/51,130 rows drifted, all in `03_Rock3-148-C_comp`"** — never as the frozen
  2026-07-15 comparison.
- The seven `string_assignment_phase*_2026-07-27.*` artifacts remain
  **deliberately untracked** until phase6 actually runs.

## 6. Environment notes that cost time

- **WSL restarted twice mid-session** (14:36 and ~16:54), killing the extraction,
  the Gate 1 waiter and the FretCam server each time. `setsid nohup` survives a
  parent shell exiting but **not** the WSL VM going down. The extraction is
  manifest-resumable at clip granularity, so each interruption cost at most one
  clip — it finished all 241 clips with no manual intervention after the
  supervisor was in place.
- **The supervisor is two layers, and it has to be.** A systemd *user* timer
  inside WSL (no sudo needed) handles process death; a Windows-side keeper is
  required for WSL itself dying, because nothing inside WSL can recover from
  that. Windows Task Scheduler registration is **blocked** in this environment
  (`Register-ScheduledTask` → access denied), so the Startup folder was used.
- ⚠️ **A `Type=oneshot` systemd service tears down its whole cgroup when
  `ExecStart` exits** and will kill the long-running jobs it just launched.
  Observed: FretCam started cleanly, served `/health`, and died seconds later on
  three consecutive supervisor runs. `KillMode=process` fixes it. This was caught
  only by testing the *automatic* path — the manual invocation had passed.
- `pgrep -f` / `pkill -f` **match their own wrapper command line**. A `pkill -f
  ".venv/bin/fretcam"` issued from a shell whose command line contains that
  string kills the shell. Kill by port (`fuser -k 8765/tcp`) or by a pattern the
  caller does not contain.
- Heredocs and `$(...)` **do not survive** the Windows→WSL→bash quoting chain in
  this setup: `<<"EOF"` loses its quoting and expands at write time, and `>=`
  inside a double-quoted string gets read as redirection (it created a file
  literally named `=4:>7} {obb`). Write scripts to disk with a file tool instead.
- The GAPS pickled dataclasses record `__main__` as their module when the builder
  runs under `python -m`, so those caches only load where that name happens to be
  imported. `_run_clip` grew an `arms` parameter so the gate experiment does not
  need the keypoint cache at all.

## 7. What NOT to do

- **Do not read E2's 0.7399 as an E2 result.** That is the model's score on its
  own validation split. The go bar is beating `calibrate.py` on wire-sparse
  clips, and it does not.
- **Do not re-run the wire-sparse gate at T=0.50 on clean-12.** It is circular
  and will "pass". See the pre-registration §1.
- **Do not treat Phase D's negative as "learned string resolution is
  impossible".** One backbone, one seed, one schedule, and an overfitting curve.
- **Do not build on `118_VD1wc`.** It is one clip with a known camera-geometry
  pathology, not evidence for a general rule.
- Everything the 2026-07-27 handoff §6 said to avoid still applies — in
  particular, **do not lower `min_clip_coverage`**.
