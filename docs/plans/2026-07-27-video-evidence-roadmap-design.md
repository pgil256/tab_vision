# Video evidence roadmap — cache rebuild through the neck-cam gate

**Date:** 2026-07-27 · **Status:** awaiting sign-off · **Author:** roadmap
follow-up to the 2026-07-24 F8 bridge verdict (DECISIONS 2026-07-24) and the
2026-07-21 ROI deep-dive.

This is a **strategy + sequencing document** with a full implementation plan
for Phase A only. Phases B–E are decision records with pre-registered gates;
each gets its own implementation plan once its predecessor's evidence exists.

Nothing here changes SPEC targets (the video-assisted 0.94 / 0.86 / 0.85
figures stay retired per DECISIONS 2026-07-02 / 2026-07-06), touches §8
contracts, or promotes `--video-backend fretcam` (still gated on L2 + a larger
frozen eval per DECISIONS 2026-07-24).

---

## 1. Baselines this plan is measured against

| Quantity | Value | Source |
|---|---:|---|
| WS1 leading indicator (ambiguous-note string accuracy, clean-12, best-orientation) | **0.574** (chunk-5 baseline 0.543) | `v1_1_gaps_string_diag`, chunk-6 WS1 |
| Audio playability prior on the same notes | **0.778** | chunk-6 capstone 2026-06-29 |
| Zero-fret wall: share of ambiguous notes on clips where YOLO finds ~0 fret OBBs at conf 0.25 (360p cache) | **~68%** | chunk-6 WS3 analysis |
| Gated clean-12 Tab F1, gold audio, audio-only | **0.8148** | `v1_1_gaps_video_chain_2026-06-22` |
| FretCam bridge, source-disjoint-10 / clean-12 | **+0.000836 [0.000000, 0.001994]** / **−0.000155** | `fretcam_end_to_end` F8, DECISIONS 2026-07-24 |
| Shipped assisted-review level | **38.76%** wrong-position reduction @60 s (AUC 0.7127) | Phase 6, `string_assignment_phase7_2026-07-16` |
| Exact 38.76% re-comparison | **blocked** (PHASE1_NOTES row provenance not on disk) | `n3_ranker_build_2026-07-23` |
| FretCam HUD (F5c) | precision 1.000, stable coverage **0.416**, dev split | `fretcam/docs`, fretcam-loop-state |

Ordering constraint recorded in the repo: the controlled-live assisted-review
A/B proceeds **only after FretCam passes L2** (DECISIONS 2026-07-22, F8 queue
rule in `docs/prompts/fretcam-loop.md`). This plan therefore runs L2 (Phase B)
before the live A/B (Phase C2). The offline ranker track (C1) has no such
dependency.

## 2. What this plan deliberately does NOT do

Recorded do-not-retry / refuted levers stay closed:

- confidence-keyed routing of CV string evidence (A14 do-not-retry);
- more frame-voting on the current evidence (systematic error, not noise);
- nut-axis re-anchor as-is (WS2 measured 0.574 → 0.547);
- chord-targeted video (audio is better on chord members, 0.819 vs 0.542);
- private/user recordings in any training or eval role (LICENSES.md:76,
  DECISIONS 2026-06-11). Phase B's live protocol is live inference with
  nothing persisted, per the design §6 wording. Phase C2 uses the **assisted
  metric**, reported separately from automatic Tab F1, per the 2026-07-20
  posture — no eval-role violation.

## 3. Licensing gate (first)

| Artifact | Role here | License | Status |
|---|---|---|---|
| GAPS media (YouTube-linked) | 720p cache re-acquire, eval substrate | CC-BY-NC-SA-4.0 — "Do not commit or redistribute media" (LICENSES.md:72) | ✅ NC acceptable under the 2026-07-20 personal posture; cache-only, never committed |
| Roboflow `b101/guitar-3` | existing detector training data | CC-BY-4.0 | ✅ already in LICENSES.md |
| ultralytics YOLO11 | detector runtime | AGPL-3.0 | ✅ deliberately accepted (existing) |
| Modal L4 (Phase D) | paid fine-tune | n/a | ⛔ **STOP for spend approval** per operating rule 8 |

**Phase E / F6 data — verified 2026-07-27** (against the live Roboflow pages and
the HF/GitHub APIs; slugs corrected, several differ from the names carried in
earlier docs):

| Artifact | Role | Verified license | Status |
|---|---|---|---|
| `b101/guitar-3` | **already in the shipping detector** | `CC BY 4.0` | ✅ re-confirmed against the live page; the shipping path is clean |
| `s-workspace-y3mjn/guitar-fret-6pt` | **Phase E2 keypoints** | `CC BY 4.0` | ✅ **the E2 unlock** — 926 images, *keypoint* detection, `nut`/`fret`. Created April 2026, so it postdates the 2026-06 survey that found nothing. Almost certainly a re-annotation of `b101/guitar-3` (identical count/vocabulary): legally clean, but attribute **both** |
| `ghaleb/guitar-fretboard` | F6 IoU fallback | `CC BY 4.0` | ✅ 384 images, `Hand` + `Zone1..Zone12` — matches the F6 mechanism |
| `bandsucore/guitar-neck-detection-suhgk` | optional neck data | `CC BY 4.0` | ✅ 1,001 images, single `neck` class |
| `my-workspace-xslxf/guitar-neck-chords-yrnmt` | optional | `CC BY 4.0` | ⚠️ **trap-type (a) risk** — a re-export banner (`ChordDetection - v11`) where the CC-BY tag was set by a re-uploader who may not have held the rights. Same shape as the SynthTab trap. Avoid or attribute cautiously |
| `soen357/fretboard` | — | **gone** | ⛔ workspace now reports `0 projects · 0 images`; search results are stale cache. Drop from all plans |
| `joaomarcoscrs/guitar-chords-daewp` | *was* the assumed E2 source | `CC BY 4.0` | ⛔ **premise was wrong** — 151 images (not ~343) and its keypoints are `hand`/`guitar-neck`/`strings`, **not fret or inlay points**. Does not serve E2 |
| `shamakg/string-fret-guitar` (HF) | candidate checkpoint | **none** | ⛔ unlabeled weights blob, no card, no license — all-rights-reserved |
| `AlbertMitjans/chord-detection` | reference | **none** (`"license": null`) | ⛔ read-only reference, never a dependency (unchanged) |

Three findings that change Phase E:

1. **The pretrained-checkpoint gap is unchanged** — HF `search=fretboard` returns
   a literally empty list; no downloadable fret-keypoint checkpoint exists
   anywhere. E2 means *training* one, not fetching one.
2. **The dataset gap has closed.** `guitar-fret-6pt` gives permissive 6-point
   fret/nut keypoints. This is the concrete substrate for the "learned keypoints
   beat OBB wires" idea, on CC-BY data, with no NC entanglement.
3. **No inlay-dot dataset exists — confirmed, not assumed.** Roboflow and HF
   searches return nothing guitar-related for inlay/position markers. Inlay
   annotation is genuinely novel work, so E2 should open on `guitar-fret-6pt`'s
   fret-wire keypoints (same rectification geometry, different landmark) and
   treat inlay dots as a later increment requiring self-annotation.

**Synthetic route (E, optional).** Poly Haven is CC0 but has **no guitar** (only
a ukulele — four strings, wrong scale length). Sketchfab CC0 for "guitar"
returns one asset and it is a fish. Usable meshes exist under **CC-BY**, but the
license is per-model and NC variants sit under the same search UI, so each must
be checked individually. The stronger engineering point: a downloaded mesh's
fret spacing is *decorative*, so for "perfect string/fret labels" the fretboard
should be generated procedurally from scale length and `d_n = L(1 − 2^(−n/12))`,
using a CC-BY body mesh only for surrounding context — which also removes most
of the licensing exposure, since the labeled geometry becomes ours.

No new dependency enters the default pipeline in Phases A–C. Phase A is pure
re-measurement with existing tools.

---

## 4. Phase A — 720p / conf-0.10 / crop-then-detect cache rebuild (implementation plan)

**Claim under test:** the geometry chain's string evidence is
detection-limited, not logic-limited. ~68% of ambiguous notes sit on clips
where no fret wires are detected at 360p/conf 0.25, so the WS1 rule-of-18
lever cannot fire. 720p acquire + lower fret confidence + a second detector
pass on the zoomed neck crop attack exactly that wall. All three levers are
already named in the chunk-6 / WS3 records; none has been run.

### A1 — acquire clean-12 at 720p

- Plumb `--max-height` (default 360) through `scripts/acquire/gaps_video.py`
  `main()` into the existing `_download(..., max_height=...)` parameter.
- Download the clean-12 at `--max-height 720` into a **separate cache dir**
  `~/.tabvision/cache/gaps_video_720/` so every pre-existing result stays
  reproducible from the untouched 360p cache.
- Re-run `--offsets --offsets-json` against the 720p files. Offsets are
  audio-derived and must match the 360p values to sub-frame precision; any
  per-clip delta > 1 video frame (42 ms) is a stop-and-investigate.
- Record the *actual* delivered height per clip (some sources may cap below
  720p) in the eval report.

### A2 — crop-then-detect fret pass

The hook already exists: `fretboard/keypoint.py` accepts and discards
`guitar_box` ("reserved for crop-then-detect"). Implementation lives in the
chain probe's CV pass (`scripts/eval/v1_1_gaps_video_chain_probe.py`), not in
the shipping pipeline:

1. Full-frame pass at conf 0.25 (unchanged) → neck OBB.
2. Axis-aligned bounding rect of the neck OBB corners + 12% pad → crop;
   upscale with `cv2.INTER_CUBIC` so the crop's long edge ≥ 1280 px.
3. Second YOLO pass on the crop at **conf 0.10**; map fret/nut OBB centers
   and extents back to full-frame coordinates; merge with the full-frame
   detections (dedupe by center distance < half the local fret pitch).
4. Neck acceptance stays at ≥ 0.25 — only the *fret/nut* floor drops, and the
   `calibrate.py` RANSAC consensus fit is the robustness backstop against the
   extra false wires that conf 0.10 will admit.

Guard rails (the F2b lesson — the last big video swing was a coordinate bug):
unit tests for the crop↔full-frame round-trip including rotated OBBs, plus
`scripts/viz/overlay_fretboard.py` renders on 3 clips **before** any cache
build, eyeballed for wire alignment.

### A3 — rebuild the rich cache

Run the chain probe's CV pass over the 720p media at conf 0.10 into
`~/.tabvision/cache/gaps_video_chain_720/`. The cache naming scheme is already
conf-keyed (`{stem}.rawcv.c0.10.pkl` via `gaps_cv_cache.rawcv_cache_path`),
and the separate directory isolates resolution. The 360p `rawcv.c0.25.pkl`
caches are never overwritten.

### A4 — measure (in order, cache-only after A3)

1. **Zero-fret wall:** share of ambiguous notes on clips with ~0 usable fret
   OBBs (the WS3 statistic), 360p-vs-720p.
2. **Leading indicator:** `v1_1_gaps_string_diag` with the WS1 calibrated
   fret map on the new cache. Baseline 0.574.
3. **Lagging indicator:** the chain probe's gated + ungated Tab F1 A/B on
   clean-12 gold audio. The gated 12/12 no-regression property must hold.

### A5 — pre-registered decision tree

| Outcome (leading indicator) | Action |
|---|---|
| < 0.60 | Bank the negative (house rule 7). Resolution/conf/crop are not the wall; the geometry line stops here. Phase D still proceeds on its own gates. |
| 0.60–0.65 **and** zero-fret share < 30% | One further lever allowed (conf 0.05 floor or per-clip multi-scale), then a final call. No further tuning beyond that. |
| ≥ 0.65 | Proceed to **WS5 gate re-derivation**: re-derive the 0.71 coverage gate on clean-12 only, frozen recipe, then one confirmation run on source-disjoint-10. |
| any gated clean-12 regression | The gate stays as-is; video stays suppressed by default regardless of the indicator. |

The chunk-6 target of **≥ 0.75** (stated in the diag's docstring) remains the
bar for "competitive with the 0.778 audio prior." Source-disjoint-10 is not
touched until clean-12 passes; test-22 only for a single final frozen confirm.

**Cost:** 1–2 days; $0 GPU (YOLO11n inference is CPU-viable); a few hundred
MB of cached media. **Deliverables:** eval report
`docs/EVAL_REPORTS/phaseA_720p_cache_rebuild_<date>.md`, DECISIONS entry,
new unit tests for the crop pass.

---

## 5. Phase B — L2 controlled-live gate (user-run; prerequisite for C2)

The §6 protocol from `docs/plans/2026-07-22-fretcam-live-position-hud-design.md`,
unchanged:

- **A1** overlay locked ≤ 3 s, holds through playing motion, two lightings.
- **A2** 5-s holds at I/III/V/VII/IX × note/chord/barre (15 holds): ≥ 90%
  correct, never off by > 1 position.
- **A3** I→V→IX shift runs: "shifting…" during motion, correct ≤ 500 ms after
  arrival; occlusion recovery ≤ 1 s.
- **A4** ≥ 10 FPS end-to-end, readout latency ≤ 150 ms.

Prep (assistant, before Pat's session): prewarmed server, per-hold checklist
in the loop's report-template format, timer script. Output: the ≤3-min L2
template filled into `docs/fretcam-loop-state.md` + a DECISIONS entry.

Fail path per the loop rules: A2/A3 failure opens **F6** (hand-bbox ×
fret-zone IoU, TapToTab mechanism) — which needs the `ghaleb` dataset
(license verification first) and a STOP for approval — then L2 re-runs once.
A second failure closes the quest, recorded honestly.

L2 costs no money, only one live session. Everything downstream of FretCam
(C2, any future default promotion) is formally blocked on it.

---

## 6. Phase C — assisted-review A/B

Two tracks with different blockers. The review-queue module
(`tabvision/tabvision/eval/review_queue.py`) is SHA-pinned by the Phase 6/7
provenance records — all work here is additive (a versioned ranker artifact),
never a restructure.

**C1 — offline ranker upgrade. Feasibility resolved 2026-07-27: NOT blocked.**

`n3_ranker_build_2026-07-23.md` records the exact 38.76% comparison as
"blocked (verified)" on three grounds. All three are wrong:

| n3 claim | Reality (verified 2026-07-27) |
|---|---|
| `PHASE1_NOTES` is "a re-decode stage **not on disk** and not cheaply regenerable" | It is a **git-ignored CSV output** — `.gitignore:76-77` covers both the phase0 and phase1 `*_notes.csv`. A fresh clone loses it *by design*; phase0's own report calls it "generated locally and git-ignored because it is **reproducible**". Regenerating it means running a script, not reconstructing a lost stage |
| Row order differs — "the **event-id SHA differs**" | The provenance files record the **identical** hash: phase4 and phase6 both carry `event_ids_sha256 = 17b7d3b3a7da24f82de778fffc84cff73ee012c2c10d80fd82dc9727020fce3c`. The 43,080-vs-35,959 gap is just all-splits vs dev-OOF-ambiguous — same set, same order, same wrong rate (0.34517 vs n3's 0.3452) |
| Three features need "the Phase 4 timbre model" | Phase 6 loads **no checkpoint** — it re-runs the model in-process (`string_assignment_phase6.py:279`, `run_phase4_oof`). Recorded cost 240 s |

Verified on this machine: **GuitarSet is present and intact** — 360
`audio_mono-mic/*.wav` + 360 `annotation/*.jams`, 1.2 GB, at
`~/mir_datasets/guitarset`. It is **not** under `$TABVISION_DATA_ROOT`, so these
scripts need `--data-home ~/mir_datasets/guitarset`; that path difference is why
it read as missing. All 300 dev WAVs sha256-match the `audio_manifest` in the
phase4 provenance, 300/300. The decode checkpoint `guitar-gaps.pth` is already in
the HF cache.

**Cost: ~5.6 h unattended CPU, zero downloads, ~930 MB disk** — dominated by the
phase0 event decode (~2.7 h) and the phase3 posterior cache (~2.3 h; recorded rate
0.92 s compute per second of audio per checkpoint). The full C1 deliverable (ten
features **+ physics**) additionally needs the `q6_full_dev_cache` ensemble JSONs,
taking it to roughly **9.5 h**; the first 5.6 h independently restores the 38.76%
baseline and de-risks the rest.

Bit-identity is not guaranteed (torch 2.12→2.11, numpy 2.4.6→2.4.4,
Windows→Linux, different ffmpeg resampler) but is **cheaply checkable** against the
recorded `541220a6…` (phase1) and `6f067585…` (phase0) hashes — and a drifted
decode fails *loudly*: the 51,130-row assertion (phase6:145), the 35,959-row
assertion (phase4:167), the row-order assertion (phase6:266), and the `onset_s`
tolerance check (phase6:319) all guard it.

⚠️ **Silent-failure landmine to fix before running the fallback.**
`n3_ranker_build.py:151-153` does `if not cache.is_file(): continue` on the
per-track ensemble JSON. With that cache absent it reports physics firing on
0/35,959 notes and both arms identical — **no error raised**. Any C1 run must
assert non-empty physics coverage rather than trust a silent zero.

Then: the exact Phase 6 replay protocol (2 s/note, gold-in-top-3 correctable,
reduction @10/30/60 s) on the full ten-feature ranker **plus the three physics
features** N3 validated (+0.0514 @60 s marginal). Gate: beat **38.76% @60 s**;
report @10/@30 alongside. GuitarSet has no video, so no FretCam features enter C1.

**Scheduling:** deliberately not started — it is CPU-bound and would contend with
Phase A's cache builds. Queue it once Phase A's arms complete.

**C2 — live FretCam-anchored A/B (blocked by an L2 pass).** Design §7.3b(b):
personal recording sessions with FretCam running; anchors re-rank the review
queue and the C-key candidate ordering; measure wrong-position reduction
@60 s vs the shipped 38.76% as the **assisted metric**, reported separately
from automatic Tab F1. Per the F8 queue rule this is a **new program**: the
A/B *design* is the deliverable of this phase, and a STOP for sign-off comes
before any integration code.

---

## 7. Phase D — WS4 learned string resolver retrain (STOP: paid run)

The banked WS4 negative (net −0.117 Tab F1; val 6-way accuracy plateau ~0.30
vs 0.167 chance) has a documented root cause — "the whole-neck crop starves
the model" plus onset-frame label alignment noise — and a documented,
committed-but-unauthorized fix. Changes vs the banked run, everything else
frozen (clip-disjoint split, peak-ratio alignment filter, **no flips**):

1. **Hand-tight crops** from the cached `HandSample` landmarks (hand bbox
   × 1.6 pad, minimum 160 px source extent) instead of the whole-neck crop,
   in `scripts/train/extract_string_dataset.py`.
2. **Sustain-window sampling**: label frames from
   `[onset + 80 ms, min(onset + 400 ms, offset − 40 ms)]` instead of the
   onset frame.
3. Prefer the 720p cache from Phase A as the frame source (sequencing
   benefit; not a hard dependency).

Pre-registered gates:

- **Gate 1 (local, $0 beyond extraction):** clip-disjoint val 6-way accuracy
  ≥ **0.45**. Below that, bank the negative; no pipeline A/B, no further
  spend.
- **Gate 2:** cache-only gated clean-12 A/B — no Tab F1 regression **and**
  leading-indicator improvement over the then-current geometry level. Anything
  less stays an eval-only artifact, exactly like WS4 v1.

**Spend:** extraction is local; the Modal L4 fine-tune is in the ~$0.40–$5
band of the prior runs. Per operating rule 8, the training launch waits for
explicit approval even after this doc is signed off.

**Status 2026-07-27 — code landed, data acquired, execution queued.**

- `scripts/train/extract_string_dataset.py` gained `--hand-tight` (square crop
  around the fretting hand's landmark span, pad 1.6×, floored at 160 px, falling
  back to the clip neck rect on hand dropout) and `--sustain` (frame sampled from
  `[onset+80 ms, min(onset+400 ms, next_onset−40 ms)]`, clamped so it never
  borrows a frame from the following note). Both default **off**, so prior
  extractions reproduce bit-identically. 11 unit tests in
  `tests/unit/test_phased_extraction.py`.
- GAPS **train split** acquired at 720p: **252 of 270 clips** (18 failed — a mix
  of transient 403s and permanently unavailable uploads). That matches the 251
  clips the banked WS4 run used, so the corpus is not the limiting factor.
  Train-only by construction: `read_split_stems` filters on the metadata CSV, so
  the clean-12 / test-22 eval clips cannot leak into a training manifest.
- Codec audit across all 264 cached 720p files: **263 H.264, 1 VP9, zero AV1**.
  This matters because AV1 decodes as *zero frames* in this OpenCV build while
  ffprobe reports a healthy stream — the failure that produced a spurious 0.000
  in Phase A before the `vcodec^=avc1` fix.
- **Not started:** extraction (~150k crops with MediaPipe per frame) is heavily
  CPU-bound and would contend with the C1 decode currently running. Queue it
  after C1, then Gate 1 (clip-disjoint val 6-way accuracy ≥ 0.45) decides whether
  any spend is justified.

---

## 8. Phase E — neck-cam program gate (STOP: commitment decision)

The formally recorded reopen condition for exact-string video is "a changed
capture contract (user-owned fixed neck-cam), which is a different product."
Phase E opens **only on explicit opt-in** — it is a product commitment, not an
increment. Before any product code, two cheap evidence spikes, both runnable
against the Phase A cache:

- **E1 — string-vibration sensing ($0, cache-only):** per-string rectified
  strips (the `geometry_refinement` string curves), temporal intensity
  spectrum per string, scored as "did this string sound" AUC vs gold on
  clean-12 clips where strings are visibly resolved. Pre-registered go bar:
  AUC ≥ 0.70 on that visible subset. This is a genuinely different physical
  signal — the optical analogue of GuitarSet's hex pickup — and the kind of
  "materially better position solver" NARRATIVE.md names as the only thing
  that reopens video.
- **E2 — learned fret keypoints (re-scoped 2026-07-27 by the §3 verification):**
  the original framing was inlay dots at 3/5/7/9/12, which anchor the fret map
  absolutely and would eliminate the first-visible-wire (`k0`) search class of
  error behind F2b. **No public dataset annotates inlay markers** — that is now
  confirmed, so inlay work means self-annotation and moves to a later increment.
  E2 instead opens on `s-workspace-y3mjn/guitar-fret-6pt` (CC BY 4.0, 926
  images, 6-point `nut`/`fret` keypoints): same rectification geometry, a
  landmark that public data actually supplies, and a direct test of "learned
  keypoints beat OBB wire detection" — which Phase A has independently shown to
  be the binding constraint. Go bar: keypoint-derived fret registration beats
  `calibrate.py`'s consensus fit on wire-sparse clips. Note there is **no
  pretrained checkpoint to fetch**; this is a training run and needs its own
  spend approval.

  **Data acquired and inspected 2026-07-27.** `guitar-fret-6pt` v1 is in place at
  `~/.tabvision/data/datasets/roboflow-s-workspace-y3mjn-guitar-fret-6pt-v1`
  (710/144/72 = 926 images; `data.yaml` self-declares `license: CC BY 4.0`,
  matching the page). The label format is more useful than the name suggests:
  `kpt_shape: [6, 3]` with classes `fret` / `nut`, and reading the coordinates
  shows the six points per instance are **the wire's intersections with the six
  strings** — successive y at near-constant x along each fret.

  That means the export supplies the **string axis and the fret axis together**,
  which is exactly what `calibrate.py` currently has to *reconstruct* by
  RANSAC-fitting rule-of-18 to noisy OBB centres. Phase A established that this
  reconstruction is the binding constraint (fit rate drives the whole +0.151, and
  its partial-evidence failures cause the `118`-class regressions), so a model
  that predicts the lattice directly attacks the measured bottleneck rather than
  a hypothesised one.

  **Next step needs approval:** training is a YOLOv8-pose fine-tune (no
  checkpoint exists to fetch). `yolo11n-pose` on 710 images is plausibly
  CPU-feasible here but slow; the Modal L4 route is the ~$0.40-class run already
  used for the detector. **STOP for spend sign-off before launching**, per
  operating rule 8.

Data policy is fixed in advance: user footage is **inference-only**; training
uses public/synthetic data exclusively. Revisiting the private-recordings ban
for a consented self-captured corpus is a separate, explicit user decision —
flagged here, not assumed.

## 9. Risks

| Risk | Mitigation |
|---|---|
| yt-dlp format drift / sources capped below 720p | record delivered height per clip; the A5 tree conditions on the measured zero-fret share, not the nominal resolution |
| crop-then-detect coordinate bugs (the F2b class) | round-trip unit tests + overlay renders on 3 clips before any cache build |
| clean-12 over-fitting from repeated dev looks | pre-registered thresholds (A5, D gates, E bars); source-disjoint-10 touched once per line, after a dev pass |
| `PHASE1_NOTES` regeneration infeasible | C1 is time-boxed; the honest fallback is the N3 marginal-delta framing, clearly labeled as not the 38.76% comparison |
| conf 0.10 floods `calibrate.py` with false wires | RANSAC inlier-consensus fit is the designed backstop; the diag's bias histograms will show any systematic pull |
| L2 needs Pat's live time | prep reduces the session to ~15 minutes; template capped at 3 minutes |

## 10. Decision asks

1. **Phase A** as specified (720p, conf 0.10, crop-then-detect, A5 tree)?
2. **Ordering** — L2 (B) ahead of the live A/B (C2), per the recorded gates;
   C1 and D can interleave freely?
3. **Phase D spend** — pre-approve the Modal run contingent on Gate 1, or
   hold the STOP until Gate 1 numbers exist?
4. **Phase E** — opt in now, defer until A–D evidence lands, or drop?
