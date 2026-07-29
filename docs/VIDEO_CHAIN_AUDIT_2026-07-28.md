# Video chain audit — 2026-07-28

**Status:** point-in-time audit of `main` @ `32e78af`. Written for coding agents
picking up video-related work. Read this before touching anything under
`tabvision/tabvision/video/`, `fretcam/`, `tabvision/tabvision/fusion/`
(vision paths), or any video eval script.

**One-paragraph verdict:** per-note string identification from casual video is
a **closed line** — refuted four independent ways against an audio prior it
cannot beat. The video chain was *also* pursued incorrectly in specific,
documented ways that cost roughly six weeks and nearly banked a wrong
refutation. Two video roles remain genuinely untested and open: coarse
position windows feeding **segment-level** decoding, and the FretCam live HUD
as a product (decided by the L2 gate, not Tab F1). Do not restart per-note
video work without reading §6 and §7.

---

## 1. What the video chain was for, in plain terms

A guitar note's pitch does not determine where it was played: the same pitch
exists on up to six (string, fret) positions. Audio recovers pitch well;
tab requires position. Video of the fretting hand was the designated sensor
for position (SPEC §3.2), with stretch targets of 0.94 single-line /
0.86 strummed / 0.85 chord-instance Tab F1 — all "video-assisted."

The prize was confirmed repeatedly by **oracle** tests (inject perfect string
knowledge, measure the ceiling):

| oracle test | result | report |
|---|---|---|
| GuitarSet player 05, aggregate | 0.657 → **0.986** | `v1_1_oracle_string_probe_2026-06-03.md` |
| Kaggle UT-Austin, 24 clips | 0.42 → **1.00** | `v1_1_dataset_search_2026-06-03.md` |
| GAPS clean-12, gold audio | 0.8148 → **0.9726** | `v1_1_gaps_video_chain_2026-06-22.md` |
| Fret-zone-per-1s window (audio-side oracle) | +0.2756 macro Tab F1 | `string_assignment_phase0_2026-07-27.md` |

The lever is real. The chain never approached it.

## 2. What actually happened (timeline with the load-bearing numbers)

### Era 1 — build (2026-05)
YOLO-OBB detector fine-tuned; neck mAP50 **0.995** — the **only** Phase 3/4/5
video acceptance gate ever run. The preflight gate (≥9/10), homography gate
(≤5 px vs hand-clicked fret intersections), and hand-tracking gate (≥0.75
top-1 on 100 labeled frames) were deferred 2026-05-05 and removed from v1 on
2026-05-07. **Zero hand labels and zero homography ground truth ever existed.**

### Era 2 — string resolution on real footage (2026-06)
- Real chain on the Kaggle rig: audio-only 0.4243 → **0.5453** (+0.121, gold
  pitch) after fixing two defects the missing gates would have caught: hand
  *selection* inverted 100% of the time (mirror assumption), homography
  orientation inverted (a clip went 0.96 → 0.17 uncorrected).
- Transfer to GAPS clean-12: **no-op** (0.8148 → 0.8148 under every gate and
  orientation setting; ungated it *hurts* 10–11/12 clips by −0.15 to −0.20).
- WS1 rule-of-18 calibration fixed a systematic geometry bug (uniform fret
  partition vs physical wire spacing) — channel 0.544 → 0.574, Tab F1 flat.
- WS4 learned string resolver (ResNet-18, 153k crops, 251 GAPS clips):
  **−0.117 Tab F1**, val accuracy plateau ~0.30 (chance 0.167). Banked.
- **The capstone (2026-06-29), the single decisive measurement:** on the same
  ambiguous notes, the plain audio playability prior scores **0.778**; the
  best video channel scores **0.574**. Fusion *adds* the video term to a cost
  that already contains the audio prior, so any nonzero weight pulls toward
  the worse source. This is the bar video must beat, and it was articulated
  ~2 months after it could have been.

### Era 3 — claims retired (2026-07-02/06)
0.94 / 0.86 / 0.85 all retired (user-approved). A14 measured
P(video right | audio wrong) = **0.285** — "video is anti-informative."
⚠️ That figure was later shown to be geometry-bug-tainted (see §4); the
retirements stand on the capstone and end-to-end evidence, but SPEC §1.4.1
does not yet note the invalidated justification.

### Era 4 — FretCam (2026-07-22 → 24)
Pivot from exact-string to bounded **position windows** (Roman-numeral hand
position, `{open/capo} ∪ [N−1, N+4]`), a resurrection of the 2026-05-07
hand-neck-anchor idea. F2's initial 2/3 failure and F7's 0.247
anti-enrichment both reversed when a coordinate bug was fixed (adapter
ignored the calibrated fret map; unit-neck joint mapped to fret 24):
F7 corrected = **0.763 [0.741, 0.783]**, though honest enrichment over its
own window marginal is only **+0.048**. HUD quality (F5c): displayed
precision **1.000** at stable coverage **0.416** on dev, 0 false locks.
Bridge into fusion (F8): **+0.000836 [0.000000, 0.001994]** on ten
source-disjoint clips, **−0.000155** on clean-12. Shipped **opt-in**
(`--video-backend fretcam`); promotion requires the L2 controlled-live gate
plus a larger frozen eval.

### Era 5 — the roadmap (2026-07-27/28)
- Phase A (720p + crop-then-detect): detection wall collapsed (8/12 → 1/12
  blind clips); channel 0.568 → **0.720** best-orientation (+0.151), but
  deployable auto-orientation ≈ **0.689**; uniform control flat (−0.007);
  **gated Tab F1 +0.0000 on 12/12**. "The gate has been protecting Tab F1,
  not obstructing it."
- E2 learned fret keypoints: **FAIL** — 0.6305 vs OBB 0.7195 pooled (−0.089).
- Wire-sparse calibration gate: pre-registered (`a8f5f2e`), **refuted** by
  leave-one-clip-out (−0.0043); harm is one outlier clip, not a threshold.
- `118_VD1wc` diagnosed: extreme foreshortening → ~3 wires/frame, nut in 4%
  of frames, orientation coin-flip, half the fitted maps end-for-end
  reversed. "Sees nothing is safe; sees a little is not."
- Phase D (WS4 retrain, hand-tight crops): extraction ran ~18 h; Gate 1 bar
  is ≥0.45 clip-disjoint val accuracy; the 30-clip smoke sat at 0.3135.
- **L2 — the 15-minute live gate — remains not run** as of this audit.

**Pattern: five consecutive channel improvements (WS1, F7-corrected, Phase A,
E2's subset win, the wire-sparse hypothesis) moved gated Tab F1 by exactly
+0.0000.** The channel is worse than the prior it would displace; the gate is
doing its job.

## 3. Verdict: lost cause, or pursued incorrectly?

Both, and the parts separate cleanly:

1. **Per-note (string, fret) from casual/in-the-wild video: lost cause under
   current constraints.** Four independent refutations: legacy
   `FrameFingering` path (quarantined), WS2 nut-axis (−0.027 channel), WS4
   learned (−0.117 Tab F1), E2 keypoints (−0.089 channel). The audio prior
   at 0.778 vs deployable video at ~0.689 is a structural gap, and casual
   footage adds foreshortening/orientation pathologies on top. The recorded
   reopen condition is a **changed capture contract** (user-owned fixed
   neck-cam — a different product), or Phase E1's string-vibration signal
   clearing AUC ≥ 0.70. Correct process would not have saved this goal — it
   would have reached the same verdict ~6 weeks sooner and several GPU runs
   cheaper.
2. **The pursuit was also incorrect** in identifiable ways (§5): skipped
   component gates invited a coordinate-bug era; the program chased an
   aspirational 0.94 instead of the break-even question; the eval surface
   cannot show a video win by construction; integration targeted the wrong
   level (per-note emission cost).
3. **Not lost causes, never actually tested:** (a) position windows feeding
   segment-level decoding (§6), (b) the FretCam HUD as a live product,
   decided by L2 (§7).

## 4. The two reversed refutations (why they matter)

Both major "video doesn't work" results that later flipped were **coordinate
bugs**, not signal absence:
- F2's 2/3 gate failure → F2b: adapter used a uniform canonical-x conversion
  with `max_fret=24` instead of the calibrated fret map → 3/3 pass on
  unchanged clips.
- F7's 0.247 anti-enrichment → 0.763 after the same one-variable fix. A14's
  banked 0.285 (used to justify retiring 0.86/0.85) shares the geometry and
  is tainted; NARRATIVE.md: "Video is not anti-informative. It was
  mis-projected."

The Phase 3/4 gates that were skipped in May (≤5 px homography check, 100
labeled fingertip frames) measure exactly this geometry, directly. Their
absence converted every downstream eval into an *indirect* test of geometry,
and indirect tests cannot distinguish "no signal" from "signal read upside
down." That is the mechanism by which the project nearly closed a live lever
permanently.

## 5. Findings (what future work must not repeat)

1. **Never skip component-level ground truth to save labeling effort.** The
   ~100–300 frames of labels avoided in May cost two months of confounded
   end-to-end evals. If a subsystem's output has a geometry, measure the
   geometry.
2. **State the break-even bar before building.** Any new evidence channel
   competes with the 0.778 audio playability prior (per-note) — or, at
   segment level, with whatever the decoder already extracts. Write the bar
   into the pre-registration.
3. **The headline metric cannot see video.** GuitarSet has no video, and the
   eval harness hard-zeroes vision (`lambda_vision=0.0` in
   `tabvision/eval/guitarset_audio.py`, `eval/string_assignment.py`,
   `fusion/context_reranker.py`). GAPS clean-12 is the only offline video
   surface and its audio baseline (0.8148) is strong. Private/user
   recordings are banned from eval. Consequence: **offline evals can bank
   negatives but can never demonstrate video's product value; only the live
   gates (L2, C2) can.** Budget accordingly.
4. **Don't report a gated arm the gate provably zeroes.** Coverage measured
   0.48–0.52 against the 0.71 gate on every clean-12 run; "gated Tab F1"
   rows were structurally +0.0000 for a month. Standing rule: do **not**
   lower `min_clip_coverage` (measured cost 0.05–0.20 Tab F1).
5. **Conditional probabilities ship with their marginals, adjacently.**
   (A14's 0.285 without its marginal; F7 repeated the shape.)
6. **"Sees a little" is worse than "sees nothing."** Weak evidence must
   become *no* evidence (abstention), not low-confidence evidence: 118
   (fires 28%, −0.129) vs 063 (fires 0%, +0.000); the chunk-2 geometric
   detector's full-frame quad at confidence 0.85 is the same failure shape.
7. **Parity between eval harness and production is a correctness property.**
   See §8.

**Recorded do-not-retry list** (do not re-propose without new evidence):
confidence-keyed routing of CV string evidence (A14 closure); lowering
`min_clip_coverage`; nut-axis re-anchor as-is (WS2); more frame-voting;
chord-targeted video; post-hoc statistic fishing over the wire-sparse
failure (homography confidence / inlier counts / fit RMS gating each need
their own pre-registration; explicitly untested: the orientation-determinacy
precondition for accepting a fitted map).

## 6. The one live technical hypothesis: segment-level position windows

`string_assignment_phase0_2026-07-27.md` (GuitarSet player 05) quantifies
the remaining string-assignment headroom as **temporal**: knowing one fret
zone per 1 s window is worth **+0.2756 macro Tab F1** (oracle); per 4 s
joint window **+0.1446**. `string_assignment_phase1_2026-07-27.md` then
shows a rule-based audio-only segment decoder captures almost none of it
(**+0.0017** confirmation aggregate). The signal exists and audio alone did
not extract it.

A coarse, high-precision, abstaining position window — exactly what FretCam
emits (precision 1.000 at coverage 0.416 on dev) — is that signal type. The
current bridge injects it **per-note** into `emission_cost` capped at 1 nat,
where it competes with the 0.778 prior and measured +0.000836. The untested
integration is **segment-level**: use windows to anchor or veto segment
hypotheses where the decoder is at a fork. Under that framing video does not
need to out-predict the audio prior note-by-note; it needs to break
segment-level ties. This is the only framing under which video's measured
strengths match the job. Caveat: it cannot be evaluated on GuitarSet (no
video); the honest surfaces are GAPS (offline) and C2-style assisted-review
A/Bs (live). Any attempt must be pre-registered with a break-even bar.

## 7. Current component status (as of 2026-07-28)

| component | status |
|---|---|
| `--video-backend legacy` (exact-string `FrameFingering` → emission cost) | default in CLI, but **quarantined by policy** and zeroed in all reported metrics; see §8 defect |
| `--video-backend fretcam` (position windows → capped fret prior) | opt-in; +0.000836 [0, 0.002]; promotion blocked on L2 + larger frozen eval |
| WS1 rule-of-18 calibration (`video/fretboard/calibrate.py`) | eval-harness only, never threaded to production (deliberate — capstone decision) |
| WS4 learned string model / E2 keypoint model | banked negatives, not shipped |
| `fusion/vision_evidence.py` (orientation/voting/gating) | **unwired** — no production caller |
| FretCam HUD (`fretcam/`, 22k LOC) | quarantined prototype; 4 files CI-gated; L2 run sheet ready (`docs/fretcam-l2-run-sheet.md`), **not run** |
| Phase 3 preflight / homography gates; Phase 4 hand gates | never run; removed from v1 2026-05-07 |
| Phase E1 (string-vibration sensing, AUC ≥ 0.70 go bar) | never run |

## 8. Live defect found by this audit (fix in flight)

Production `pipeline.py` applies raw per-frame `FrameFingering` evidence at
`lambda_vision=1.0` (weighted only by homography confidence) with **none** of
the chunk-3 protections — `choose_orientation`, `combine_fingerings`,
`gate_fingering_to_audio`, and the 0.71 coverage rule exist only in
`fusion/vision_evidence.py` + eval scripts, with no production caller
(verified by grep 2026-07-28). The nearest measured configuration (ungated)
scored **−0.15 to −0.20** on GAPS and produced mirrored-orientation failures
on the Kaggle rig. Every published Tab F1 number uses `lambda_vision=0.0`,
so the shipped default matches no reported metric. A fix session was started
2026-07-28 (wire the protections in, or default to audio-only until L2
passes). If you are reading this later, verify which resolution landed
before reasoning about default-path behavior.

## 9. Recommended order of work

1. Confirm the §8 parity fix landed.
2. **Run L2** (~15 min with a guitar; `docs/fretcam-l2-run-sheet.md`). It
   gates F8 promotion and C2, and is the highest-information measurement
   available at any price.
3. Read Phase D Gate 1 with a pre-committed stop (bar ≥0.45; smoke was
   0.3135 at 30 clips — consistent with the WS4 failure plateau).
4. Design (pre-registered) the segment-level position-window integration
   (§6) before any further per-note video work.
5. Hygiene: fix `CLAUDE.md`'s stale "v1.1 video stretch (0.94/0.86)"
   pointer; note in SPEC §1.4.1 that A14's 0.285 justification was later
   bug-invalidated (retirements stand on the capstone + e2e evidence);
   correct the inverted `video/fretboard/__init__.py` docstring
   (keypoint is primary, geometric is the unwired fallback); remove ~10 MB
   of committed `debug_*.jpg` / `frame_*.png` at repo root; retire or wire
   `vision_evidence.py`; update `tabvision/pyproject.toml`'s
   "audio + vision fusion" description.

## 10. Primary sources

Chronology and decisions: `docs/NARRATIVE.md`, `docs/DECISIONS.md`,
`SPEC.md` §1.4.1/§7/§8, `AUDIT.md`, `docs/fretcam-loop-state.md`,
`docs/plans/2026-07-27-video-evidence-roadmap-design.md`,
`docs/HANDOFF-2026-07-27-video-evidence.md`.
Key eval reports (all under `docs/EVAL_REPORTS/`):
`v1_1_oracle_string_probe_2026-06-03.md`, `v1_1_chunk2_cv_chain_2026-06-10.md`,
`v1_1_chunk3_real_video_robustness_2026-06-11.md`,
`v1_1_gaps_video_chain_2026-06-22.md`, `v1_1_gaps_chunk6_ws1_2026-06-25.md`
(incl. the 2026-06-29 capstone), `v1_1_gaps_ws4_learned_2026-06-29.md`,
`a14_video_complementarity_2026-07-06.md`, the `fretcam_f*`/`fretcam_e2e_*`
series (2026-07-22/24), `fretcam_audio_bridge_fixed_policy_2026-07-24.md`,
`phaseA_720p_cache_rebuild_2026-07-27.md`, `e2_fret_keypoints_2026-07-28.md`,
`wire_sparse_calibration_gate_2026-07-28.md`,
`string_assignment_phase0_2026-07-27.md`,
`string_assignment_phase1_2026-07-27.md`.
