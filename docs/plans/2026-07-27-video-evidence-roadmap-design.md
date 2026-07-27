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
| `joaomarcoscrs/guitar-chords`, `ghaleb/guitar-fretboard`, other Phase-E data | keypoints / IoU fallback | **UNVERIFIED** | ⛔ fetch and quote the actual LICENSE string verbatim before any acquisition (design-plan gate 1). `AlbertMitjans/chord-detection` has **no license file** — read-only reference, never a dependency |

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

**C1 — offline ranker upgrade (unblocked now).** Regenerate the Phase 6 row
provenance (the `PHASE1_NOTES` re-decode that `n3_ranker_build_2026-07-23`
found missing), time-boxed; if regeneration is infeasible, say so and stop C1.
With provenance restored, run the exact Phase 6 replay protocol (2 s/note,
gold-in-top-3 correctable, reduction @10/30/60 s) on the full ten-feature
ranker **plus the three physics features** N3 already validated (+0.0514 @60 s
marginal on the self-contained ranker). Gate: beat **38.76% @60 s** on the
exact protocol; report @10/@30 alongside. GuitarSet has no video, so no
FretCam features enter C1.

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
- **E2 — inlay-dot keypoints:** fret markers at 3/5/7/9/12 anchor the fret
  map absolutely, eliminating the first-visible-wire (`k0`) search class of
  error behind F2b. Data: synthetic renders + the one permissive keypoint
  source, license verified first (§3). Go bar: absolute fret-index anchoring
  beats `calibrate.py`'s consensus fit on wire-sparse clips.

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
