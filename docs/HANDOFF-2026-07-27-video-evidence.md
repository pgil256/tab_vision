# Handoff — video evidence work, 2026-07-27

Read this first if you are picking up the video track cold. It records what was
done, what it proved, what it disproved, what broke, and what is waiting on a
decision. Everything below was measured on this machine unless marked otherwise.

**One-line summary:** the video channel got substantially better and still
cannot contribute to Tab F1, because it remains below the audio prior it would
displace. The binding constraint is now known to be fretboard geometry, and the
cheapest attack on it is a dataset acquired and license-cleared today.

---

## 1. Why this work happened

The question was how to get more out of the computer-vision side. The repo's own
history said video had been measured and largely refuted, so the first step was
reading what had already been tried (`docs/NARRATIVE.md`, ~139 DECISIONS
entries, the chunk-5/6 eval reports) rather than re-proposing dead ends.

That reading surfaced one lever that was named in the records but never pulled:
the chunk-6 analysis found **~68% of ambiguous notes sat on clips where the
detector found ~0 fret wires**, because the cached footage was 360p. With no
wires detected, `calibrate_fret_xs` cannot fit, and `compute_fingering` silently
falls back to a **uniform** fret partition — which is physically wrong, since
real frets follow `d_n = L(1 − 2^(−n/12))` and bunch up toward the body. So the
rule-of-18 correction that had been built (chunk-6 WS1) was mostly inert.

Plan: `docs/plans/2026-07-27-video-evidence-roadmap-design.md` (Phases A–E,
approved 2026-07-27 with pre-registered decision thresholds).

## 2. What was done — Phase A

720p re-acquire + a second detector pass on the zoomed neck crop at confidence
0.10, measured against a same-day 360p control.

**Full results:** `docs/EVAL_REPORTS/phaseA_720p_cache_rebuild_2026-07-27.md`.

| | 360p control | 720p + crop |
|---|---:|---:|
| clips where the detector sees ~0 frets | 8 / 12 | **1 / 12** |
| share of ambiguous notes on those clips | 0.650 | **0.081** |
| ambiguous-note string accuracy (best orientation) | 0.568 | **0.720** |
| same, uniform partition (the control) | 0.543 | 0.536 |
| **gated Tab F1** | 0.8147 | **0.8147** |
| **ungated Tab F1** | — | **0.6142** |

Nine clips improved, three regressed. Biggest gain `142_GD1wc` +0.347; biggest
loss `118_VD1wc` −0.150.

### What this proves

**The detection wall was real and is now essentially gone.** That was the stated
blocker and it is removed.

**Resolution alone does nothing.** The uniform-partition row moved −0.007. The
entire +0.151 came from the calibration being *able to fit*. This was verified
directly rather than inferred: a diagnostic calls `calibrate_fret_xs` per cached
frame and reports the true fit rate, which rises in lockstep (`027` 36% → 88%,
`043` 0% → 85%). The fit rate also stays strictly below the "≥4 wires" share
everywhere, so the RANSAC consensus check is observably rejecting bad wire sets
rather than rubber-stamping them.

### What it disproves — the part that matters

**Tab F1 did not move.** Gated it is identical (+0.0000 on 12/12, because
measured coverage is 0.48–0.52 against a 0.71 threshold). Ungated it is *worse*
(−0.2006, violated on 10/12 clips), and even with a perfect gold-chosen
orientation it is still −0.051.

The reason: the audio playability prior resolves strings at **0.778**. Video is
now at 0.720 — much closer than the 0.568 it was, but still **below**. Replacing
a better prior with a worse one costs more than it gains. **The coverage gate has
been protecting Tab F1, not obstructing it.** Do not loosen it.

⚠️ **The 0.720 is a best-orientation figure** (the diagnostic picks the flip that
maximises gold accuracy). The deployable auto-orientation number is ≈**0.689**,
so the real gap to audio is ≈0.089. The banked 0.574 baseline uses the same
convention, so the +0.151 delta is valid; only the absolute level is lower than
the headline suggests.

## 3. Things that were wrong and are now corrected

**A claim I made mid-session and then refuted.** The ungated table showed
auto-orientation at 0.6142 vs a 0.7635 best-orientation ceiling, and I wrote that
orientation selection was the dominant lever. Testing it
(`scripts/eval/phasea_orientation_diag.py`) showed the four orientation scores
are **well separated** (median relative spread 0.545), not tied — the selector is
confidently wrong on 7/12 clips, and costs only **0.031** of string accuracy. The
0.149 appears in Tab F1 only because ungated fusion multiplies a mirrored
posterior across every note. Orientation is worth fixing but is **not** a route
to a Tab F1 gain.

**A bug I introduced.** Demoting yt-dlp's format 18 for the high-resolution path
let it choose by bitrate, and for `142_GD1wc` it chose **AV1** — which this
OpenCV build decodes as **zero frames** while ffprobe reports a healthy 1280×720
stream. The probe then wrote an *empty cache*, which scored 0.000 and would have
been silently reused forever. Fixed two ways: the selector now pins
`vcodec^=avc1`, and `_raw_cv_cache` raises rather than persisting an empty cache.
A later audit of all 264 cached 720p files found **263 h264, 1 vp9, 0 AV1**.

**A banked conclusion that was wrong.** `n3_ranker_build_2026-07-23.md` recorded
the assisted-review comparison as "blocked (verified)" on three grounds. All
three fail: the missing file is a **git-ignored, reproducible output** (not a
lost pipeline stage), phase4 and phase6 provenance record the **identical**
`event_ids_sha256`, and phase6 loads **no** timbre checkpoint. GuitarSet was on
disk the whole time at `~/mir_datasets/guitarset` — outside
`$TABVISION_DATA_ROOT`, which is why it read as missing. A correction note is at
the top of that report.

## 4. State of each track

| track | state | blocked on |
|---|---|---|
| **Phase A** (720p + crop) | **done, banked** | — |
| **Gate re-derivation (WS5)** | authorized by the decision tree, but **re-scoped** — do not lower the coverage threshold | evidence quality, not the gate |
| **C1** (assisted-review ranker) | regenerated, **4 rows short** | a tolerance decision (see §5) |
| **Phase D** (string-model retrain) | code + 11 tests landed; 252/270 train clips acquired | CPU, then Gate 1 |
| **Phase E2** (fret keypoints) | data acquired + license-verified | **~$0.40 training spend** |
| **L2** (live camera test) | fully prepared, server verified | 15 min with a guitar |

### C1 detail

`phase0` and `phase1` ran end to end (written under `--date 2026-07-27` so the
seven **tracked** 2026-07-15 artifacts were not overwritten — do not re-run with
the old date). Output note tables are the right size but not bit-identical
(sha256 `7d460bb7…` vs `6f067585…`, `c09c467a…` vs `541220a6…`), and hold
**51,126** development rows per condition against the **51,130** asserted at
`string_assignment_phase6.py:145`. Observed failure:
`RuntimeError: expected 51,130 development rows per condition, got 51126`.

That is the predicted toolchain drift (torch 2.12→2.11, Windows→Linux, different
ffmpeg resampler) failing loudly by design. The uncommitted regenerated reports
sit untracked in `docs/EVAL_REPORTS/string_assignment_phase*_2026-07-27.*`; they
were left uncommitted deliberately, since the run cannot serve the exact
comparison and would invite confusion with the canonical 2026-07-15 set.

### Phase E2 detail — the recommended next move

`s-workspace-y3mjn/guitar-fret-6pt`, **CC BY 4.0** (verified twice: the live
Roboflow page and the downloaded `data.yaml`), 926 images (710/144/72), already
at `~/.tabvision/data/datasets/roboflow-s-workspace-y3mjn-guitar-fret-6pt-v1`.

`kpt_shape: [6, 3]`, classes `fret` / `nut`. Reading the coordinates shows the
six points per instance are **the wire's intersections with the six strings** —
so the labels give the string axis and the fret axis *together*. That is exactly
the lattice `calibrate.py` currently reconstructs by RANSAC-fitting rule-of-18 to
noisy box centres, and Phase A proved that reconstruction is the binding
constraint. Attribution is owed to **both** b101 and s-workspace-y3mjn (the image
set is almost certainly a re-annotation — identical 926 count and split);
recorded in LICENSES.md.

No pretrained checkpoint exists anywhere for this task (re-verified today: HF
search for "fretboard" returns an empty list), so this is a training run.

## 5. Decisions waiting on the user

1. **C1 tolerance.** Relax the 51,130 assertion to a small tolerance (e.g.
   ±0.05%) and report "near-exact, 4/51,130 rows drifted"? The drift is 0.008%
   and cannot move a wrong-reduction metric under player-held nested OOF — but
   `string_assignment_phase6.py` is provenance-pinned, so loosening it is a
   deliberate call, not an implementation detail. **Recommended: yes.**
2. **Phase E2 spend** (~$0.40 Modal L4, or a slow local CPU run). **Recommended:
   yes** — it attacks the measured bottleneck.
3. **L2** whenever recording is possible — `docs/fretcam-l2-run-sheet.md`.

## 6. What NOT to do

- **Do not lower `min_clip_coverage`.** Measured: it costs 0.05–0.20 Tab F1.
- **Do not treat orientation selection as a Tab F1 win.** Worth 0.031 of string
  accuracy; perfect orientation still leaves ungated video net-negative.
- **Do not re-run phase0/phase1 with `--date 2026-07-15`** — it overwrites seven
  tracked provenance artifacts.
- **Do not revive confidence-keyed routing** without fresh pre-registration; it
  is a recorded do-not-retry (A14) and the new evidence quality does not by
  itself overturn it.
- **Do not chase resolution further.** The uniform-partition control shows pixels
  alone buy nothing.

## 7. Environment notes that cost time

- GuitarSet lives at `~/mir_datasets/guitarset` (360 wav + 360 jams), **not**
  under `$TABVISION_DATA_ROOT`. These scripts need `--data-home`.
- The GAPS Zenodo archive uses **different stem names** than `scan_gaps` expects;
  the HF mirror `xavriley/GAPS` matches the documented layout. Clean-12
  musicxml/midi/syncpoints/audio and the splits CSV were taken from there.
- FretCam's live HUD needs **no Windows Python**: the browser supplies the camera
  via `getUserMedia` and ships JPEGs over the WebSocket, so the server runs in
  WSL. This matters because Windows Python here is 3.13 and **MediaPipe publishes
  no cp313 wheel** for any 0.10.x release.
- `python -m fretcam.cli` exits silently (no `__main__` guard). Use the console
  script `.venv/bin/fretcam`.
- The 360p arm reproduces the banked aggregates closely (uniform 0.543 vs 0.544,
  calibrated 0.568 vs 0.574, gated Tab F1 0.8147 vs 0.8148) but **per-clip**
  agreement is loose (`212` −0.266, `294` −0.137) because YouTube re-encodes.
  Within-report deltas are valid; single-clip comparisons against the June column
  are not.

## 8. New tooling added

| tool | purpose |
|---|---|
| `scripts/eval/phasea_fret_wall.py` | detection-wall statistic, calibration **fit rate**, mean homography confidence, from any cache |
| `scripts/eval/phasea_report.py` | emits the full comparison tables in one command |
| `scripts/eval/phasea_orientation_diag.py` | why the orientation selector misfires |
| `scripts/viz/overlay_crop_detect.py` | renders crop-pass detections on real frames (the F2b guard rail) |
| `tests/unit/test_crop_detect.py` (15) | crop↔full-frame coordinate round trips |
| `tests/unit/test_phased_extraction.py` (11) | sustain-window clamping, hand-tight crop geometry |

Full suite: **1142 passed, 3 skipped**.
