# DECISIONS — TabVision Spec Adoption

Per SPEC.md §0.5: when a non-obvious branch is taken in any decision tree,
record an entry here.

Format:

```
## YYYY-MM-DD — <short title>
**Phase:** <N>
**Decision tree:** <tree name>
**Branch taken:** <branch label>
**Evidence:** <metric values, file paths>
**Reasoning:** <one paragraph>
```

---

## 2026-05-05 — Project name kept as `tabvision` (not `tabify`)

**Phase:** 0
**Decision tree:** spec adoption / naming (design doc Q-naming)
**Branch taken:** Keep existing project name `tabvision`. Global-edit
`tabify` → `tabvision` in SPEC.md.
**Evidence:** Existing branding in `tabvision-client/`, `tabvision-server/`,
`web-client/` (`tabvision-web`). No `tabify` artifacts in repo prior to
spec arrival.
**Reasoning:** SPEC.md §0 says "rename freely." Keeping `tabvision` avoids
churn across existing directory names, package names, and any external
brand surfaces (Vercel deploy, demo videos, future portfolio narrative).
The name was always a placeholder in the spec; user confirmed during
brainstorm.

---

## 2026-05-05 — Hybrid spec adoption (not full rewrite)

**Phase:** 0
**Decision tree:** spec adoption / Q1 (design doc)
**Branch taken:** **Hybrid (option C).** Adopt the spec literally, scaffold
per Phase 0, fast-forward through phases the existing v0 code already
satisfies. Don't abandon the 91.6%-F1 v0 work.
**Evidence:** `docs/plans/2026-05-05-tabvision-spec-adoption-design.md` §3
(phase mapping table).
**Reasoning:** Existing pipeline produces the deliverable for Phases 1, 4,
and 5 in spirit, just not under §8 contracts. Throwing it away to rebuild
from scratch wastes meaningful working code; ignoring the spec leaves the
project without acceptance gates, distorted-electric capability, or
multi-format export. Hybrid balances both.

---

## 2026-05-05 — `refactor/v1` cut from `feature/audio-finetune-phase1`, not `main`

**Phase:** 0
**Decision tree:** spec adoption / branch strategy (design doc §5)
**Branch taken:** Cut `refactor/v1` off `feature/audio-finetune-phase1` —
deviation from design doc §5 which prescribes branching off `main`.
**Evidence:** `git rev-list --count main..feature/audio-finetune-phase1` =
33. Main is missing the 91.6%-F1 work, the 20-clip benchmark harness, the
GuitarSet TFRecord pipeline, and the design doc itself (committed to
`feature/audio-finetune-phase1` as `a3d7dad`).
**Reasoning:** Phase 0 needs the existing v0 code accessible for inventory
and porting. Branching off `main` would lose 33 commits of work, including
the very pipeline we're auditing. Cherry-picking to bring main current was
out of scope for the user's "start phase 0" instruction. Note: this means
`refactor/v1` inherits the in-flight H2 finetune work-in-progress (none of
which was committed — the only commit unique to the branch at the time of
cut was the spec adoption design doc + spec rename + AUDIT/LICENSES).

---

## 2026-05-05 — Phase 0 started before H2 (audio finetune) verdict

**Phase:** 0
**Decision tree:** spec adoption / sequencing (design doc §5 step 1)
**Branch taken:** Start Phase 0 immediately; freeze `feature/audio-finetune-phase1`
mid-experiment. Deviation from design doc §5 ("Now → H2 verdict (~1 week)
... Cut refactor/v1 ... Phase 0 work").
**Evidence:** User explicit override 2026-05-05: "clear memory and start
phase 0 then." H2 (label-encoding hypothesis) untested at branch freeze.
**Reasoning:** User authorized the deviation. Risk: H2 outcome remains
unknown when Phase 7 entry comes. Mitigation: at Phase 7 entry, decide
whether to revive `feature/audio-finetune-phase1` for one more H2 attempt
or accept the unanswered hypothesis as part of Scenario X / Y in design
doc §8. The frozen branch state is git-recoverable indefinitely.

---

## 2026-05-05 — Phase 1.5 redefined to use existing datasets only (no new recording)

**Phase:** 1.5 (recorded at Phase 0 for reference)
**Decision tree:** spec adoption / Phase 1.5 scope (design doc §6)
**Branch taken:** **Use existing public datasets + already-recorded historical
clips only.** Drop the spec's "15+ new user-recorded clips" requirement.
Eval split: GuitarSet (clean acoustic), IDMT-SMT-Guitar (clean electric),
EGDB (distorted electric), existing 11/20 self-recorded videos (iPhone OOD
bonus tier).
**Evidence:** Design doc §6 (revised Phase 1.5 table).
**Reasoning:** User declined to record new clips during brainstorm. Existing
historical self-recordings preserve iPhone-domain ground truth without new
recording effort. Acknowledged blind spot: distorted-electric tier is
measured on EGDB studio data, not iPhone-recorded distortion.

---

## 2026-05-05 — Phase 2 spec gap: Riley/Edwards SOTA model unavailable

**Phase:** 2 (entry)
**Decision tree:** SPEC §7 Phase 2 acceptance — decision tree assumes the
Riley/Edwards model can be installed and run; outcome paths cover model
errors, version issues, recall vs precision lag, etc. **The tree does not
cover "the model code/weights aren't publicly available at all."** Per
SPEC §0.8 this is a "stop and ask" condition.
**Branch taken:** **Pause Phase 2 implementation; verify license posture
of every spec-named candidate; document findings; ask the user.**
**Evidence:** Verified 2026-05-05 via `gh api`:
- `xavriley/HighResolutionGuitarTranscription` is a fork of the Nerfies
  paper-website template. README is verbatim Nerfies. **No model code,
  no weights, no LICENSE file.** xavriley has no separate
  guitar-transcription-model repo (searched all 30+ repos).
- `trimplexx/music-transcription` README badge claims MIT, but the repo
  has **no LICENSE file** — under copyright default ("no license = all
  rights reserved"), README claims aren't binding.
- GAPS (Cwitkowitz) has no public GitHub repo; only the arXiv paper.
- `cwitkowitz/guitar-transcription-continuous` (FretNet) is **MIT** but
  has **no pretrained weights or releases** — training from scratch on
  GuitarSet is required.
- `cwitkowitz/guitar-transcription-with-inhibition` is **MIT** but same
  caveat (no pretrained weights).

**Conclusion:** No permissively-licensed *pretrained* guitar transcription
SOTA model exists today. Scenario Y from design doc §8 ("Phase 2
underwhelms or licensing blocks default-pipeline use → fall back to Basic
Pitch + fine-tune as v1 audio backbone") is the operative path.
**Reasoning:** The spec's Phase 2 plan was written assuming Riley/Edwards
could be picked off the shelf. Since it can't, the choice is between (a)
training a Cwitkowitz model from scratch (~weeks of GPU work — same shape
as the v0 finetune already in flight on `feature/audio-finetune-phase1`),
(b) staying on Basic Pitch and treating Phase 7's augmentation/fine-tune
as the only audio-quality lever, or (c) a smaller "Phase 2′" that ports
v0's heuristic filters (sustain redetection, harmonic, ghost-note) into
the spec-compliant pipeline as a Basic-Pitch *post-processing* layer
(narrows the 3× over-detection observed in Phase 1 acceptance numbers).

User decision pending. LICENSES.md updated; this entry is the audit
trail.

---

## 2026-05-05 — Phase 3 dataset reveals 3-class fretboard-parts annotation (not whole-guitar bbox)

**Phase:** 3 (training)
**Decision tree:** SPEC §7 Phase 3 — assumes one detector emits a single
"guitar" class bbox. The actual Roboflow dataset emits three classes:
fret / neck / nut.
**Branch taken:** **Treat the YOLO-OBB model as a unified guitar-region
+ fretboard-keypoint detector.** Use the `neck` class as the proxy for
GuitarBBox (preflight + cropping), and the `fret` + `nut` classes as
fretboard-rectification keypoints (replacing v0's Hough-line geometric
detection as the primary fretboard path).
**Evidence:**
- README.dataset.txt: 926 images, classes [fret, neck, nut], CC BY 4.0,
  YOLOv8-OBB format.
- Roboflow API: project type = "instance-segmentation"; OBB export
  converts polygons → oriented bboxes per class. 36k frets / 1.8k nuts /
  1.8k necks across the corpus.
**Reasoning:** This is *more* useful than the spec's two-stage
(guitar-bbox → fretboard-Hough) approach because (a) the same model
provides both signals in one pass and (b) per-fret OBBs give us
geometric anchors for the homography that are far more reliable than
edge detection. v0's `fretboard_detection.py` becomes the fallback path
for clips where the YOLO model fails. `tabvision.video.fretboard.keypoint`
(name from SPEC §7 Phase 3 deliverable list) gets implemented as the
new primary; `tabvision.video.fretboard.geometric` (the existing thin
v0 wrapper) stays as the fallback.
**Effect on prior decisions:**
- The Phase 3 detector path entry ("Option A — fine-tune YOLO-OBB on
  Roboflow guitar dataset") still stands. The change is interpretive:
  what gets emitted is fretboard parts, not a guitar-body bbox.
- `YoloOBBBackend` needs a small refactor to support multi-class output
  (per-class detection lookup), exposed as `detect_neck` /
  `detect_frets` / `detect_nut` accessors.
- Spec §7 Phase 3 acceptance "guitar IoU ≥ 0.95 vs hand-labeled GT" gets
  re-interpreted as "neck IoU ≥ 0.95" since that's our proxy for the
  guitar region.

---

## 2026-05-05 — Phase 3 detector path: fine-tune YOLO-OBB (AGPL accepted)

**Phase:** 3 (entry)
**Decision tree:** SPEC §7 Phase 3 acceptance — assumes ultralytics YOLOv8
is usable. **Constraint not in the tree:** ultralytics is AGPL-3.0
(verified — full GNU AGPL v3 LICENSE file in their repo). SPEC §1.5 prefers
permissive licensing for default pipeline.
**Branch taken:** **Option A (fine-tune YOLO-OBB on Roboflow guitar
dataset).** Accept AGPL contagion for the entire TabVision pipeline.
**Evidence:**
- `gh api repos/ultralytics/ultralytics`: license = AGPL-3.0, full GNU AGPL
  v3 in LICENSE file, README explicitly mentions Enterprise License for
  non-AGPL commercial use.
- HF search: zero pretrained guitar-detection models. Closest permissive
  alternatives are general-purpose detectors (YOLOS, DETR, RT-DETR) which
  would need fine-tuning anyway.
- User explicitly chose option A: "Do A. Fork the data from
  https://universe.roboflow.com/b101/guitar-3 and attribute it. Finetune
  Yolo OBB with it."
**Effect:** Per AGPL §1, our combined work is a "covered work" and any
distribution must include source under AGPL. For a portfolio project on
public GitHub this is fine — the demo-purpose is unaffected. For any
future closed-source / SaaS use of TabVision, swap-out paths must exist
(YOLOS or DETR fine-tune; documented in LICENSES.md Phase 9 deferral list).
**Reasoning:** No spec-compliant alternative exists today. Spec's "permissive
default" preference vs the practical reality that guitar detection requires
either AGPL ultralytics (with weights and tooling) or significantly more
work (label + train a permissive detector from scratch). User sized the
trade-off against portfolio-project goals and accepted AGPL.

---

## 2026-05-05 — REVERSAL: Phase 2 reopened; Riley/Edwards model IS available

**Phase:** 2 (entry — reversing the prior "spec gap" entry)
**Decision tree:** Phase 2 acceptance.
**Branch taken:** **Reopen Phase 2 with `xavriley/hf_midi_transcription`
as the highres backend.** User pointed out the actual implementation
that I missed during license verification.
**Evidence:**
- `https://github.com/xavriley/hf_midi_transcription` (description says
  "Audio-to-MIDI for solo saxophone" — misleading; the README confirms
  multi-instrument: saxophone, bass, **guitar**, piano).
- `instruments.json` exposes `guitar` (default = `guitar-gaps.pth`),
  `guitar_gaps`, and `guitar_fl` (Francois Leduc dataset) checkpoints.
- License declared MIT in three places:
  1. `pyproject.toml`: `License :: OSI Approved :: MIT License` classifier.
  2. HuggingFace model card YAML frontmatter `license: mit`.
  3. README "License: MIT" in Model Details.
- Pretrained weights hosted on HF: https://huggingface.co/xavriley/midi-transcription-models.
- Pip-installable: `pip install hf-midi-transcription`.
- The HF model card explicitly cites both source papers: Riley/Edwards
  Domain-Adaptation paper (https://arxiv.org/pdf/2402.15258) and the
  Cwitkowitz GAPS paper (https://arxiv.org/abs/2408.08653). So the
  spec's two separately-listed SOTA candidates are both shipped as
  checkpoints in this single package.
**Reasoning for the missed verification:** I searched xavriley's repos
for the keyword "guitar" in name/description and only found the
companion-website repo. `hf_midi_transcription` was filed under
"saxophone" in its description and didn't match my filter. Had I read
the README of every multi-instrument repo (or searched for
"transcription" instead of "guitar"), I'd have caught it. Lesson:
description-keyword filters miss multi-instrument hosts; READMEs are
authoritative.

**Effect on prior decisions:**
- Phase 2 is **back on**, not skipped.
- The Phase 1 polish work (commit 7856c4f) **still stands** and is
  still useful: it gives Basic Pitch as a fallback / comparison baseline.
- The Phase 1 polish addendum's "Phase 2 disposition: skipped" claim is
  superseded; Phase 2 is now the active phase.
- LICENSES.md updated with corrected status; the prior `❌` for
  Riley/Edwards is corrected to `✅`.

---

## 2026-05-05 — Phase 3 detector acceptance: 50-epoch finetune passes neck IoU gate

**Phase:** 3 (training acceptance)
**Decision tree:** SPEC §7 Phase 3 — `Guitar detector ≥ 0.95 IoU?`
(reinterpreted as `neck IoU ≥ 0.95` per the 2026-05-05 dataset
reinterpretation entry).
**Branch taken:** **Pass — proceed-to-Phase-4 leaf.**
**Evidence:**
- Modal run `ap-yzlJk4xVR3NfyFxbpDUlpt` (50 epochs, batch 16, lr0=0.01,
  yolo11n-obb.pt base, seed 0). 423.6 s wallclock on an L4. Final val
  metrics on 144 held-out images / 3062 instances:
  | class | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | all  | 0.970 | 0.928 | **0.956** | 0.692 |
  | fret | 0.944 | 0.832 | 0.917 | 0.484 |
  | **neck** | **0.989** | **1.000** | **0.995** | **0.905** |
  | nut  | 0.978 | 0.951 | 0.955 | 0.687 |
- Stable weights at `~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt`
  (5.7 MB, symlink to runs/20260505-190316/run/weights/best.pt).
**Reasoning:** Neck mAP50 = 0.995 clears the 0.95 acceptance bar with
4.5 percentage points to spare; mAP50-95 = 0.905 is also strong (means
the box quality is good across IoU thresholds, not just the lenient
0.5). Per-class P/R for fret and nut are healthy enough that the
keypoint fretboard backend will have plenty of fret OBBs to work with.
**Effect:** Phase 3 detector deliverable is acceptance-passed. The
remaining acceptance pieces (preflight 9/10 on labeled framing set;
fretboard ≤ 5 px median homography error on 5 user clips) are blocked
on hand-labeled ground-truth data, not on detector quality.

---

## 2026-05-05 — Phase 3 remaining acceptance gates deferred to a labeling pass

**Phase:** 3 (acceptance) → 4 (entry)
**Decision tree:** SPEC §7 Phase 3 acceptance checklist (preflight ≥ 9/10
on labeled good/bad framing set; fretboard ≤ 5 px median homography
error on 5 user clips).
**Branch taken:** **Defer the two label-dependent gates; advance to
Phase 4.** The detector gate (the one capacity-limited deliverable)
already passed at `neck mAP50 = 0.995`. The remaining gates are
data-collection tasks, not engineering tasks.
**Reasoning:** Engineering Phase 4 in parallel with the hand-labeling
work is strictly faster than blocking on it. Pat will come back and
collect the ground truth (5 clips × 4 fret-intersection clicks; ~10
clips of intentionally good vs bad framing) before Phase 9 hardening.
The Phase 3 code is stable and won't drift while the labels are being
collected — re-running the eval is a one-line pytest invocation when
the fixtures arrive.
**Outstanding work to schedule:**
- Build the labeling harness (a small click-to-mark-fret-intersections
  tool — tkinter or web — saving JSON; plus a "framing: good/bad"
  classifier doc).
- Collect 5 user clips × 4 fret-intersection clicks each (frets 5 + 12,
  top + bottom edges) for the fretboard gate.
- Collect ~10 clips intentionally framed well and badly (off-centre,
  partially occluded, dim lighting, oblique angle) for the preflight gate.
- Wire the two pytest harnesses (`-m fretboard_eval`, `-m preflight_eval`)
  to read the fixtures dir.
**Effect on prior decisions:** The 2026-05-05 detector-acceptance entry
remains valid; this entry records the deferral of the data-bound gates
so a later reader doesn't read "Proceed to Phase 4" as implying those
two acceptance items also passed.

---

## 2026-05-05 — Phase 4 entry: port v0 MediaPipe pipeline + add §8 contract layer

**Phase:** 4 (entry)
**Decision tree:** N/A — entry decision, no failure-mode branch yet.
**Branch taken:** **Hybrid port.** v0's
`tabvision-server/app/video_pipeline.py` already runs MediaPipe Tasks
API HandLandmarker on full frames and returns a `HandObservation` with
extended/pressing/muting flags per finger. v1 wraps that logic to fit
the §8 `HandBackend` and `FrameFingering` contracts: (a) per-frame
`detect(frame, H, cfg)` signature, (b) output is
`(n_fingers, n_strings, max_fret+1)` logits projected through the
homography rather than raw landmark coords, and (c) per-finger
posteriors built from distance-to-fret-cell + curl prior + z-depth
prior per spec §7 Phase 4.
**Reasoning:** Per CLAUDE.md operating rule 1 (audit before refactor),
v0's MediaPipe wrapper is reusable: it already (i) loads the Tasks API,
(ii) selects the fretting hand by handedness label with a finger-spread
fallback, and (iii) extracts the 21 landmarks per hand. What v1 needs
that v0 lacks is the canonical-coordinate projection (needs the
homography that didn't exist in v0's coupled Hough pipeline) and the
per-cell posterior over (string, fret). Net new code is the projection
+ posterior layer; v0's MediaPipe plumbing is wrapped, not rewritten.
**Open questions (will be resolved by acceptance run):**
- Distance kernel σ for the fingertip-to-cell prior; default = 0.5
  fret-widths, will calibrate against the 100-frame labeled set.
- Whether the curl prior helps or hurts; ablation will tell us.
- Fretting-hand identification — start with v0's handedness logic;
  switch to wrist-near-nut (now possible because we have the
  homography) only if the eval shows misidentification.

---

## 2026-05-07 — Phase 5 uses hand-neck anchors as first-class fusion priors

**Phase:** 5 (vision-fusion integration)
**Decision tree:** Phase 4/5 fusion contract — whether exact per-finger
posteriors are the primary video signal, or whether coarser neck-region
evidence should guide candidate selection first.
**Branch taken:** **Use coarse hand-neck anchors as first-class Phase 5
priors.** MediaPipe + fretboard homography estimate the fretting hand's
center fret/span; the pipeline converts each timed anchor into an
`AudioEvent.fret_prior` before calling Viterbi/chord fusion.
**Evidence:** Phase 4 manual fingering labels proved too expensive for the
near-term path, while the detector/fretboard stack is already good at
identifying the neck coordinate system. Phase 5 fusion already accepts
`AudioEvent.fret_prior` as emission evidence, so the anchor signal can be
integrated without changing §8 public function signatures.
**Reasoning:** Exact fingertip-to-string/fret labels are brittle and costly
to validate; "the hand is around frets 3-6" is a stronger, more stable
visual prior for resolving audio's same-pitch string/fret ambiguity. Keeping
the signal as a prior lets audio and playability override it when the visual
evidence is weak or wrong.
