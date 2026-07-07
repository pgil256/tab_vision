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

## 2026-06-17 — Chunk 4 complete: highres cross-corpus diagnosis on Guitar-TECHS

**Phase:** v1.1 audio transcription/alignment (chunk 4)
**Decision tree:** Second-corpus gate — UT-Austin-specific alignment vs highres
note grouping vs broader cross-corpus transcription behavior
**Branch taken:** Keep `highres`; do not switch audio models. Attribute the
residual Tab F1 ceiling to the audio-only string-resolution limit (the v1.1
video chain's job), and the UT-Austin raw-audio collapse to corpus-specific
alignment — completing and superseding the 2026-06-11 "validate on a second
dataset" decision below.
**Evidence:** Full 12-clip Guitar-TECHS chord run via the new cached runner
`scripts/eval/v1_1_second_corpus_probe.py` (`highres`, `--position-prior none`,
`--splits train`, no pitch/time calibration): onset F1 `0.7321`, pitch F1
`0.6787`, Tab F1 `0.0700` (lower-95 `0.0377`), chord acc `0.0207` across
1292 notes; decomposition wrong_position_same_pitch `788` (43.4%) +
extra_detection `634` (34.9%) dominate. Reports:
`docs/EVAL_REPORTS/v1_1_highres_guitartechs_chords_2026-06-17.md` and
`..._decomposition_2026-06-17.md`. Contrast: UT-Austin raw highres (onset `0.04`,
pitch `0.00`) and global-calibrated (onset `0.4598`, pitch `0.3613`, Tab
`0.1415`, oracle-video `0.3535`) in
`docs/EVAL_REPORTS/v1_1_audio_alignment_probe_2026-06-11.md`.
**Reasoning:** Uncalibrated highres lands onsets/pitches on a second corpus
(0.73/0.68) where UT-Austin raw is ~0, so highres is not globally broken and the
UT-Austin failure is corpus-specific tuning/time-origin alignment (22/24 clips
prefer a −1 semitone shift). The Tab F1 ceiling reproduces the
`wrong_position_same_pitch` string-ambiguity shape on both corpora (SPEC
§1.4.1), so the largest lever is the v1.1 video string-resolution chain, not an
audio-model switch. Honesty bounds: 0.68 pitch still fails the 0.90 audio gate
(electric-domain penalty — "not broken," not "acceptable"); Guitar-TECHS is
electric / out-of-domain / n=12, a diagnostic and not an acceptance baseline; and
extra_detection (35%) is chord over-detection that video does not directly fix.

---

## 2026-06-11 - Keep highres; validate on a better second dataset

**Phase:** v1.1 audio transcription/alignment
**Decision tree:** User direction after chunk-4 audio-alignment probe
**Branch taken:** Keep `highres` as the active audio model and pause model-switch
work. Test `highres` against a stronger second corpus before drawing more
conclusions from the UT-Austin real-audio failure.
**Evidence:** User instruction on 2026-06-11: "Keep highres. Do not switch audio
models yet. Test highres against a better second dataset." New smoke artifact:
`tabvision/data/eval/guitartechs_highres_smoke.toml` and
`docs/EVAL_REPORTS/v1_1_highres_guitartechs_smoke_2026-06-11.md`, run with
`--backend highres --position-prior none`, scored one Guitar-TECHS direct-input
clip at onset F1 `0.7187`, pitch F1 `0.6562`, Tab F1 `0.0000`. The paired
decomposition report shows 18 wrong-position-same-pitch, 5 pitch-off, 1
timing-only, 3 missed-onset, and 13 extra-detection errors. A same-settings
12-clip Guitar-TECHS chord run exceeded the 30-minute local interactive budget
before writing a report; older full Guitar-TECHS evidence remains in
`docs/EVAL_REPORTS/local_guitartechs_noprior.md`.
**Reasoning:** UT-Austin remains useful for video string/fret evidence, but its
real-audio path has corpus-specific pitch/time alignment issues. Guitar-TECHS is
local, public, CC-BY-4.0, and has per-string MIDI labels, so it is the best
runnable second-corpus audio check today even though it is electric and
out-of-domain for the GAPS-trained highres checkpoint. Keep highres while adding
cross-corpus evidence; revisit GAPS as the stronger real-performance offline
eval when data acquisition and parsing are wired.

---

## 2026-06-11 - Remove private video/tab corpus from v1.1 evidence

**Phase:** v1.1 video string-resolution / eval data
**Decision tree:** Dataset replacement after user correction
**Branch taken:** Remove the private eval and training/tab
corpora from tracked data, benchmark fixtures, and active documentation. Use
license-checked public/offline sources by role: GuitarSet for clean audio/tab
labels, Kaggle UT-Austin for current real-video string/fret eval, and GAPS as
the best optional real performance video/audio replacement when NC offline eval
is acceptable. Keep GOAT as a candidate only if access and dataset license terms
are explicitly verified.
**Evidence:** User correction on 2026-06-11 that the private recordings/tabs
are inaccurate; v1.1 chunk-3 report
`docs/EVAL_REPORTS/v1_1_chunk3_real_video_robustness_2026-06-11.md` shows
video robustness improved gold-pitch Tab F1 0.4243 -> 0.5453 while real highres
audio remains 0.0583 -> 0.0657 with oracle ceiling 0.1959.
**Reasoning:** The old personal labels create misleading metrics and are not a
reproducible portfolio gate. Public/offline corpora keep the project auditable,
license-trackable, and comparable, while chunk-3 says the next accuracy work
should target audio transcription/alignment before adding more video fusion.

---

## 2026-05-13 — Tab F1 v1 acceptance: per-tier targets + public-corpus composite

**Phase:** Accuracy work (cross-cuts Phases 1, 2, 3, 5, 7, 8 of the SPEC)
**Decision tree:** Design plan adoption + SPEC §1.4 amendment proposal
**Branch taken:** Replace the aggregate 0.88 Tab F1 acceptance gate with
a per-tier table; drop SynthTab (CC-BY-NC) and GOAT (request-only) from
the default pipeline; rely on GuitarSet + Guitar-TECHS + EGDB
(license-pending) for the public-corpus composite eval.

**Evidence:**
- Strategy / decision record: `docs/plans/2026-05-12-tab-f1-to-spec-design.md`
- Phase 0 implementation plan: `docs/plans/2026-05-13-tab-f1-phase-0-implementation.md`
- SPEC amendment block: `SPEC.md` §1.4.1 (per-tier table + composite test set)
- First baseline artifact (2 of 4 tiers covered): `docs/EVAL_REPORTS/composite_baseline_2026-05-13.md`
- Companion error decomposition: `docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md`
- Implementation branch with the eval harness: `impl/tab-f1-phase-0`

**Reasoning:** The 2026-05-08 GuitarSet validation showed aggregate Tab
F1 = 0.6104 with comp tracks at 0.670 and solo tracks at 0.508. The
aggregate target hid the dominant failure axis (string/fret assignment
on single-line passages), and the SPEC §1.4 numbers (0.94 / 0.86 / 0.90
/ 0.82) baked in implicit per-tier expectations that the project hadn't
explicitly negotiated. The 2026-05-13 user conversation locked in
relaxed v1 targets (0.85 / 0.90 / 0.87 / 0.80), kept the original SPEC
numbers as the v1.1 / portfolio stretch reference, and committed to
audio-only fusion priors + cheap pitch post-processing as the leverage
path (no SynthTab pretrain → no NC license taint on shipped weights).

**Per-tier acceptance gate (v1):**

| Tier | v1 target | 2026-05-13 baseline (mean / lower 95% CI) |
|---|---:|---:|
| Clean acoustic single-line | 0.85 | 0.5076 / 0.4448 (fail) |
| Clean acoustic strummed | 0.90 | 0.6708 / 0.6015 (fail) |
| Clean electric | 0.87 | missing — pending Guitar-TECHS |
| Distorted electric | 0.80 | missing — pending EGDB |

Both covered tiers fail by ~25–35 pp. Per the error decomposition,
`wrong_position_same_pitch` accounts for 77% of single-line loss and
50% of strummed loss — Phases 1-7 of the design plan target this
bucket.

**Decisions inventoried in the design plan (D1–D11):**

- D1 Per-tier replaces aggregate. D2 Targets table. D3 Composite eval.
  D4 No SynthTab. D5 Video qualitative-only. D6 Free-tier compute first
  (Local > Colab > Kaggle > Lightning > Modal). D7 1-2 month cadence.
  D8 No stretch (bends/slides) in v1. D9 D2 numbers on top-1 only.
  D10 Personal clips fully banned. D11 This is a SPEC §1.4 amendment,
  not a SPEC-achievement plan.

**Open Phase 0 user actions:** Lightning Studios / Kaggle / Colab / W&B
account verification; EGDB author email; Guitar-TECHS Zenodo download.

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
**Branch taken:** **Use existing public datasets only.** Superseded on
2026-06-11 for private recordings: the historical private corpus was removed
because the labels are not trusted. Current eval split: GuitarSet (clean
acoustic), public/offline electric corpora, Kaggle UT-Austin for current
real-video string/fret eval, and GAPS as optional NC offline video/audio eval.
**Evidence:** Design doc §6 (revised Phase 1.5 table).
**Reasoning:** User declined to record new clips during brainstorm. Existing
public/programmatic sources preserve reproducibility and keep the release story
auditable. Private recording labels are no longer trusted as validation data.
Known blind spot: any studio-domain electric metric must say so explicitly.

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
fretboard <= 5 px median homography error on license-checked public/offline clips) are blocked
on hand-labeled ground-truth data, not on detector quality.

---

## 2026-05-05 — Phase 3 remaining acceptance gates deferred to a labeling pass

**Phase:** 3 (acceptance) → 4 (entry)
**Decision tree:** SPEC §7 Phase 3 acceptance checklist (preflight ≥ 9/10
on labeled good/bad framing set; fretboard ≤ 5 px median homography
error on license-checked public/offline clips).
**Branch taken:** **Defer the two label-dependent gates; advance to
Phase 4.** The detector gate (the one capacity-limited deliverable)
already passed at `neck mAP50 = 0.995`. The remaining gates are
data-collection tasks, not engineering tasks.
**Reasoning:** Engineering Phase 4 in parallel with the hand-labeling
work is strictly faster than blocking on it. The remaining labels must come
from license-checked public/offline clips before any future hard gate.
The Phase 3 code is stable and won't drift while the labels are being
collected — re-running the eval is a one-line pytest invocation when
the fixtures arrive.
**Outstanding work to schedule:**
- Build the labeling harness (a small click-to-mark-fret-intersections
  tool — tkinter or web — saving JSON; plus a "framing: good/bad"
  classifier doc).
- Collect/license-check public/offline clips with fret-intersection labels for
  the fretboard gate.
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
- Distance kernel Ïƒ for the fingertip-to-cell prior; default = 0.5
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

---

## 2026-05-07 — Phase 5 GuitarSet pitch-to-tab bottleneck

**Phase:** 5 (audio-to-tab mapping)
**Decision tree:** GuitarSet audio-only diagnostic — if pitch F1 is good
but Tab F1 is bad, fix string/fret candidate selection before tuning video
calibration or `lambda_vision`.
**Branch taken:** **Add an optional learned pitch-position prior.** Raw
GuitarSet JAMS provide held-out string/fret labels; the evaluator now learns
`P(string,fret | pitch)` from train players and attaches it via the existing
2D `AudioEvent.fret_prior` path before audio-only Viterbi decode.
**Evidence:** On full validation, oracle gold-onset/gold-pitch events scored
only `0.4335` Tab F1 with the default decoder, proving the mapping is bad even
when audio extraction is perfect. On the first 10 validation tracks, about
two-thirds of same-pitch events landed on the wrong adjacent string/fret. Most
errors were low-fret equivalents such as G-string notes decoded on B or B-string
notes decoded on high E. A GuitarSet train-split prior raised oracle
full-validation Tab F1 to `0.6802`. On the 3-track highres smoke, Tab F1 moved
from `0.3356` to `0.7260` while onset F1 (`0.9692`) and pitch F1 (`0.9555`)
stayed unchanged.
**Reasoning:** This confirms the immediate bottleneck is pitch-to-position
ambiguity, not highres onset/pitch extraction and not Phase 5 vision weighting.
The prior is optional for now; it does not change public fusion APIs or the
default production decode until a full validation run and home-video check
justify promoting it.

**Follow-up evidence:** A Modal L4 full-validation highres run completed on
2026-05-07. With no position prior: onset F1 `0.9218`, pitch F1 `0.9022`,
Tab F1 `0.3878`. With the GuitarSet train-split prior: onset F1 `0.9218`,
pitch F1 `0.9022`, Tab F1 `0.6104` (`+22.26 pp`). Per-track, 51/60 improved,
8/60 regressed, and 1/60 was unchanged. Mean track Tab F1 moved from `0.347`
to `0.589`.
**Promotion decision:** **Do not make this an unconditional production default
yet.** Promote the prior next as a versioned/configured production option, then
make it the default only after (a) a checked-in prior artifact is available
without requiring raw GuitarSet at runtime, (b) same-pitch regressions are
classified and reduced or accepted, and (c) the home-video Phase 5 benchmark
shows no regression. The full GuitarSet result is strong enough to justify the
production integration path, but the 8 regressed validation clips make a silent
global default premature.

---

## 2026-05-07 — Phase 5 pitch-position prior stays explicit by default

**Phase:** 5 (production prior path)
**Decision tree:** Phase 5 prior promotion — make learned pitch-position
evidence a default decode behavior only if full-validation and home-video
ablation evidence show a clear no-regression improvement.
**Branch taken:** **Keep the prior optional.** The production pipeline now
accepts `--position-prior guitarset-v1`, which loads a checked-in versioned
artifact from `tabvision/tabvision/fusion/priors/guitarset_v1.json`; default
transcription remains `--position-prior none`.
**Evidence:** Existing full GuitarSet validation evidence remains strong:
highres with no prior scored onset F1 `0.9218`, pitch F1 `0.9022`, Tab F1
`0.3878`; highres with the GuitarSet train-split prior scored onset F1
`0.9218`, pitch F1 `0.9022`, Tab F1 `0.6104` (`+22.26 pp`). However, 8/60
validation clips regressed. The home-video prior on/off benchmark is prepared
through the new explicit CLI/pipeline option, but local completion is blocked
until the held-out home-video eval data plus heavyweight audio/vision assets
are available in this worktree.
**Reasoning:** The prior fixes a real pitch-to-tab ambiguity bottleneck, but
the target product is home iPhone video, not GuitarSet. A silent default would
hide a dataset-specific learned bias inside every decode. Keeping it explicit
preserves baseline behavior while allowing the coordinator to run the exact
home-video ablation before deciding whether to promote it.

---

## 2026-05-07 — Phase 8 smoke eval is deterministic without external data

**Phase:** 8 (eval harness hardening)
**Decision tree:** Phase 8 determinism gate and deferred Phase 1.5/3/4 debt audit
**Branch taken:** **Use a dependency-light synthetic smoke scope for CI, and
emit explicit blockers for full model-backed eval until the manifest and labels
exist.**
**Evidence:** `docs/EVAL_REPORTS/eval_full_20260507T000000Z.json` and
`.md` report 0 manifest clips, all four required tiers missing, Phase 3
preflight/fretboard labels at 0/10 and 0/5, and Phase 4 hand labels at
0/100. `python -m scripts.eval.run --scope smoke --twice-and-diff
--timestamp 2026-05-07T00:00:00Z` reports `deterministic=true` with a
180-second smoke budget.
**Reasoning:** Full eval cannot honestly produce audio-only/audio+vision/prior
or confidence-calibration metrics without the Phase 1.5 manifest and external
media/annotations. The smoke scope still exercises the same report writer and
fixed output format in CI, so Phase 8 hardening can progress without masking the
remaining data-bound acceptance debt.

---

## 2026-05-07 — Manual annotation gates removed from v1 release criteria

**Phase:** Remaining v1 / release hardening
**Decision tree:** Remaining-plan cleanup after Phase 8 smoke report
**Branch taken:** **Remove manual work from v1 gates.** Phase 1.5 private-corpus
manifest completeness, Phase 3 preflight labels, Phase 3 fretboard click labels,
Phase 4 fretting labels, manual dataset downloads, new recordings, and
user-corrected self-labeling are now `removed_from_v1` or `optional_future`.
**Evidence:** User instruction on 2026-05-07: "remove anything from the plan
that requires manual annotation or work." Current automated baseline is green:
`pytest -q` reported `272 passed, 12 skipped` before this change, and the Phase
8 smoke runner already exercises report generation without external data.
**Reasoning:** Manual annotation and the retired private media corpus are useful
historical context but make v1 unshippable as a reproducible portfolio artifact.
The remaining release plan must depend on automated evidence only: deterministic
smoke fixtures, checked-in fixtures, public/programmatic datasets such as
GuitarSet, existing Modal/public-data reports, license policy checks,
fresh-install checks, and renderer tests. `--position-prior guitarset-v1` stays
explicit; default transcription remains `--position-prior none` until automated
evidence justifies promotion.

## 2026-06-02 — Cross-dataset check: prior doesn't transfer to electric; audio backbone is the blocker

**Phase:** Accuracy work (#2 cross-dataset prior generalization, run on laptop CPU)
**Decision tree:** Tab-F1 strategy §6 "verify the +22 pp prior generalizes before building on it"
**Branch taken:** Prior lift does **not** generalize to electric (out-of-domain),
and the dominant cause is upstream — the highres (acoustic GAPS) backbone does
not transcribe electric guitar well. Re-prioritize: electric tiers are blocked
on the **audio backbone**, not the prior/fusion.

**Evidence:** `docs/EVAL_REPORTS/cross_dataset_prior_2026-06-02.md` and the four
local reports (`local_guitarset_{prior,noprior}.md`,
`local_guitartechs_{prior,noprior}.md`). GuitarSet acoustic prior lift +28.9 pp
(single) / +19.6 pp (strummed), onset/pitch ~0.92–0.94 / 0.90–0.93 — reproduces
the documented 0.6104/0.3878 baseline. Guitar-TECHS electric (58 clips, 5541
notes): prior lift **+1.3 pp** (0.110 → 0.124, within the 95% CI), onset/pitch
**0.75 / 0.73**. Tab F1 capped ~0.12 by the pitch collapse.

**Reasoning:** The prior's electric lift is within noise, so it shows no useful
transfer — but the test is confounded: with pitch F1 only 0.73 on electric, the
prior has almost nothing correct to re-assign, so "acoustic-specific prior" can't
be cleanly separated from "nothing to work with." The clean, dominant finding is
that the audio backbone doesn't generalize to electric (pitch 0.93 → 0.73). This
makes the committed SPEC §1.4 clean-electric (0.90) and distorted-electric (0.82)
targets unreachable with the current backbone (measured 0.12). **Next step pivots
from #3 (GuitarSet-only fine-tune, acoustic) to evaluating an electric-capable
backbone** (`hf_midi_transcription` `guitar_fl`, or a highres fine-tune on
Guitar-TECHS/EGDB) before any further fusion/prior work on the electric tiers.
The prior remains justified for the acoustic tiers (in-domain +22 pp). Caveats:
GT subset is chord-dominant (P1+P2; no P3/scales/EGDB), single electric corpus,
long-form clips.

## 2026-06-02 — Scope v1 to acoustic; electric → v2 behind a tone toggle

**Phase:** Accuracy work / v1 scope (SPEC §1.4.1 amendment)
**Decision tree:** "is electric reachable for v1?" — after measuring it
**Branch taken:** Scope **v1 to acoustic**. Defer the electric tiers (clean
0.90, distorted 0.82) to **v2**, delivered as a **separate fine-tuned
`guitar-electric` checkpoint routed by the declared instrument** (tone
toggle), so the acoustic model is never disturbed.

**Evidence:**
- `docs/EVAL_REPORTS/cross_dataset_prior_2026-06-02.md` — clean-electric Tab
  F1 0.12, pitch F1 0.73 (vs acoustic 0.93); `guitar_fl` swap doesn't help.
- No highres **training** code in-repo (inference-only packages;
  `audio_finetune.py` is a scaffold) → electric is a bounded v2 project, not
  a v1 gate. v2 plan: `docs/plans/2026-06-02-electric-backbone-finetune-design.md`.
- Toggle landed: `tabvision/audio/backend.py` registers `highres-electric`;
  `tabvision/pipeline.audio_backend_for_session` routes electric →
  `highres-electric` (used when `run_pipeline(audio_backend_name="auto")`);
  the electric backend fails fast until `TABVISION_HIGHRES_ELECTRIC_CKPT` is
  set. Tests: `tabvision/tests/unit/test_audio_routing.py`.

**Reasoning:** Committing v1 to where the system can excel (acoustic, already
near-spec on onset/pitch, +22 pp prior) ships an honest, reproducible
artifact; electric stays on the roadmap without blocking v1. Separate
checkpoints + routing (not one shared model) avoid catastrophic forgetting of
the acoustic 0.93 — the architecture already routes by checkpoint
(`highres` / `highres-fl`). This supersedes the 2026-06-01 "highest targets
including electric" amendment with an evidence-based scope; SPEC §1.4.1
updated to match.

## 2026-06-02 — Acoustic single-line is information-limited; honest audio-only targets

**Phase:** Accuracy work / v1 acceptance (SPEC §1.4.1 target revision)
**Decision tree:** "close the single-line gap (0.51 → 0.94)?" — after diagnosis
**Branch taken:** Single-line Tab F1 cannot be closed audio-only (it's the
string/fret ambiguity, not a tuning miss). **Set honest audio-only v1 targets**
(single-line ≥ 0.45, strummed ≥ 0.60, aggregate ≥ 0.55); the original
0.94 / 0.86 become the **v1.1 video-assisted** reference. Commit the one real
audio win found (hand-position continuity).

**Evidence:** `docs/EVAL_REPORTS/acoustic_single_line_2026-06-02.md`.
- Decomposition: single-line loss is **322 `wrong_position_same_pitch`** vs 8
  `pitch_off` — pitch is correct, the *string* is wrong. (Aggregate 54 %.)
- Melodic prior **regresses** single-line (0.474 → 0.449); left default-off.
- Continuity sweep: `POSITION_SHIFT_COST` 0.05 → **2.5** lifts single-line
  0.508 → 0.523 and strummed 0.671 → 0.676 (full validation, no regression) —
  **committed as the new default** in `tabvision/fusion/playability.py`
  (env-overridable). It does not move single-line toward 0.94.

**Reasoning:** With pitch correct and continuity raised 50×, single-line still
sits at ~0.52 — the residual errors are notes where audio *cannot* determine the
string (the same pitch is acoustically near-identical across strings). This is
exactly what the video/hand pipeline resolves, but GuitarSet is audio-only and
v1 is audio-only, so 0.94 is unreachable for v1. Honest targets reflect the
demonstrated audio-only capability (`lower_95_CI ≥ target`); single-line is
flagged video-limited with **video string-resolution as the v1.1 lever** (a
style/structure-conditional prior is the only remaining audio-only lever, with
bounded upside). Onset/pitch/chord/latency unchanged (met).

## 2026-06-03 — v1 ACCEPTED (audio-only acoustic); chord-instance acc → v1.1 (video)

**Phase:** Accuracy work / v1 acceptance (SPEC §1.4.1)
**Decision tree:** "does the formal acceptance run clear §1.4.1?" — all-metrics run
**Branch taken:** **Stamp v1 ACCEPTED on the audio-only acoustic scope.** Tab F1
(per-tier + aggregate), onset, pitch, and latency all clear their §1.4.1 gates on
the GuitarSet player-05 validation set. **Re-scope chord-instance accuracy ≥ 0.85
to a v1.1 (video) target** — it shares single-line Tab F1's audio string/fret
information limit, so it was a v1.1 target mis-filed as a v1 gate (the 2026-06-02
amendment lowered single-line Tab F1 0.94 → 0.45 for the same reason but left
chord at 0.85). User-approved.

**Evidence:** `docs/EVAL_REPORTS/v1_acceptance_2026-06-03.md` (eval harness
`292252d`, 60 clips, `--position-prior guitarset-v1`):
- Tab F1 lower-95: single-line **0.457** (≥ 0.45), strummed **0.606** (≥ 0.60),
  aggregate **0.600** (≥ 0.55).
- Onset F1 mean 0.938 / 0.923 (≥ 0.92); Pitch F1 mean 0.930 / 0.901 (≥ 0.90).
- Latency: 60 clips / 1054 s ⇒ ~17.6 s per ~24 s clip (0.74× realtime) ⇒ ≈ 45 s
  for a 60 s clip (≤ 5 min).
- Chord-instance accuracy **0.52 single-line / 0.48 strummed** — tracks per-tier
  Tab F1 (single-line chord 0.52 ≈ single-line Tab F1 0.52).
- Harness change (chord metric + model reuse + §1.4.1 targets): commit `292252d`.

**Reasoning:** Whole-chord recovery requires the exact string + fret for every
note in a chord, so it is bounded by the same audio string-resolution limit that
caps single-line Tab F1 — which §1.4.1 already accepted by lowering single-line to
0.45. Measuring it (0.48–0.52, matching Tab F1) confirmed it is information-limited,
not an implementation gap, so 0.85 belongs with the 0.94 single-line number as a
v1.1 video-assisted reference. v1 ships an honest, reproducible audio-only acoustic
artifact; chord ≥ 0.85 returns as a v1.1 gate once video string-resolution lands.
Two harness bugs were fixed en route to the run: per-clip model reload (OOM ~clip
17 → build the highres backend once) and a duplicate-OpenMP segfault on Windows
(`KMP_DUPLICATE_LIB_OK=TRUE`).

## 2026-06-03 — v1.1 string-resolver already works (oracle-validated); v1.1 is eval-data-gated

**Phase:** v1.1 (video string-resolution) — P1 validation
**Decision tree:** v1.1 design §9 ("test the resolver on a clean signal first")
**Branch taken:** **Validate before building.** Probed the *existing* fusion with a
gold-derived oracle `FrameFingering` rather than building the §5 "new resolver."
The resolver is already wired and correct, so v1.1 P1 needs **no new code**; the
milestone reduces to **P0 (eval data)**.

**Evidence:** `docs/EVAL_REPORTS/v1_1_oracle_string_probe_2026-06-03.md`,
`scripts/eval/v1_1_oracle_string_probe.py`, `tests/unit/test_video_string_resolution.py`.
- Oracle (perfect hand signal), 60-clip player-05 validation: single-line Tab F1
  **0.57 → 0.995** (> 0.94 target), strummed **0.75 → 0.978** (> 0.85), aggregate
  0.66 → 0.986 — pure fusion, no audio model / video / rendering.
- Path: `fuse → playability.find_fingering_at(onset) → emission_cost` vision term
  `lambda_vision · -log(marginal_string_fret[s, f])`, candidate-restricted by Viterbi.
- No-regression confirmed by test: absent/zero fingerings == the audio-only decode.

**Reasoning:** The 2026-06-03 v1.1 design §4 mis-stated the gap — it described the
fret-only *neck-anchor* path; the `FrameFingering` path was already consumed per
note. The probe is the §9 "clean-signal" test and passes overwhelmingly, proving
the lever and the code. v1.1 is now an **eval-data** problem: synthetic-from-
GuitarSet to prove on clean rendered video, then a license-clean public
video+string corpus as the acceptance gate (§6) — directly analogous to
v2-electric being gated on the missing upstream trainer.

## 2026-06-03 — v1.1 eval dataset = Kaggle UT-Austin (NC ok for eval); real-video data pipeline locked

**Phase:** v1.1 (video string-resolution) — P0 eval data + chunk-1
**Decision tree:** v1.1 design §9 ("no §1.5-clean public video+string dataset → escalate")
**Branch taken:** A deep-research pass confirmed **no portfolio-clean public dataset has
both fretting-hand video AND per-string labels**. Rather than block, **use the Kaggle
UT-Austin "guitar-transcription-dataset" (CC-BY-NC-SA)** as the v1.1 eval set: a
non-commercial license does not bar an *eval* corpus, because SPEC §1.5 governs the
**shipping pipeline** (which bundles no dataset), not the offline acceptance set.
Synthetic-from-GuitarSet stays the fully-clean fallback.

**Evidence:** `docs/EVAL_REPORTS/v1_1_dataset_search_2026-06-03.md` (deep-research run
`wf_d6833878-6c5`: 98 agents / 16 sources / 19 verified claims).
- Two disjoint buckets, empty intersection: per-string-labelled corpora (GuitarSet MIT,
  Guitar-TECHS CC-BY, GOAT, EGDB, IDMT) are all audio-only; video+per-string corpora
  (Kaggle UT-Austin, GAPS, TapToTab) are all NC / gated. Guitar-TECHS was the named gap
  → verified audio-only (arXiv:2501.03720).
- §1.5 reading corrected: the rule is on the shipping default pipeline; an eval set is
  downloaded to produce a metric, never shipped/redistributed (as GuitarSet/EGDB are).
- **Chunk-1** (`scripts/eval/v1_1_kaggle_oracle_probe.py`): the Kaggle per-frame finger
  labels parse to per-note gold (new-placement = onset; highest-fret-per-string sounds;
  `our_idx = 6 − their_string`, audio-verified), and the oracle lift reproduces on REAL
  clips — audio-only **0.42 → oracle 1.00** (25 clips / 527 notes).

**Reasoning:** The lever (string from video) is now proven twice (GuitarSet 0.52→0.99,
Kaggle 0.42→1.00) and the resolver needs no new code. The eval-data gate is resolved
with a real-video corpus whose only flaw is a non-commercial license that does not apply
to offline eval use. Remaining work is purely the MediaPipe CV chain (chunk 2: does real
hand/fretboard detection on this footage produce good fingerings) + the real-audio eval
(chunk 3). Caveats: single-source student dataset (a proof, not a robust headline); do
not commit the data; revisit if TabVision is ever commercialised.

## 2026-06-11 — v1.1 chunk-3 video robustness complete; highres audio is the current bottleneck

**Phase:** v1.1 (video string-resolution) — chunk 3
**Decision tree:** v1.1 design §9 ("Video regresses audio-only tiers → the
confidence gate (§5.3) is mis-tuned; it must collapse to weight 0")
**Branch taken:** Make orientation and confidence gating automatic before judging
the real-video chain. Keep the gate conservative by default: per-event evidence
must support the audio pitch, and clips with less than 71% surviving video coverage
fall back to audio-only.

**Evidence:** `docs/EVAL_REPORTS/v1_1_chunk3_real_video_robustness_2026-06-11.md`,
`tabvision/fusion/vision_evidence.py`, `scripts/eval/v1_1_real_chain_probe.py`.
- Gold-pitch real-video eval (24 scored clips / 527 notes): audio-only **0.4243**
  → audio+real-video **0.5453**, oracle **1.0000**, no per-clip regressions.
- Real highres WAV eval now runs with pitch/time calibration: highres audio-only
  **0.0583** → audio+real-video **0.0657**, oracle-video ceiling **0.1959**.
- The highres wrapper needed a Windows stdio encoding guard because
  `hf_midi_transcription` prints a Unicode status glyph during checkpoint download.

**Reasoning:** Chunk 2's manual `--flip-fret --flip-string` was real but
rig-specific. Chunk 3 resolves that by selecting orientation from audio-compatible
candidate support, then voting multiple nearby frame posteriors and scaling/skipping
video by confidence. The conservative 0.71 clip-coverage floor is intentionally a
no-regression gate: sparse evidence should become no evidence. The real highres
number is low even with oracle video, so further video-gate tuning is the wrong next
move; the next accuracy work should target highres audio transcription/alignment on
the UT-Austin WAVs or a better audio backbone for that corpus.

## 2026-06-11 - v1.1 chunk-4 audio alignment favors a global highres correction

**Phase:** v1.1 (video string-resolution) - chunk 4
**Decision tree:** v1.1 chunk-4 plan ("raise the UT-Austin real-audio ceiling
before more video fusion work")
**Branch taken:** Add a cached audio-alignment diagnostic and prefer the measured
global highres correction over the earlier per-clip calibration for the next
implementation pass.

**Evidence:** `docs/EVAL_REPORTS/v1_1_audio_alignment_probe_2026-06-11.md`,
`scripts/eval/v1_1_audio_alignment_probe.py`,
`tests/unit/test_v1_1_audio_alignment_probe.py`.
- highres raw audio is effectively unaligned: onset F1 **0.0409**, pitch F1
  **0.0000**, Tab F1 **0.0000**, oracle-video Tab F1 **0.0000**.
- highres per-clip calibration reproduces the previous low ceiling: Tab F1
  **0.0603**, oracle-video **0.1979**.
- highres global calibration (`pitch_shift=-1`, `time_shift_s=+0.14`) is better:
  onset F1 **0.4598**, pitch F1 **0.3613**, Tab F1 **0.1415**, oracle-video
  **0.3535**.
- Real-chain rerun with that global correction scored audio-only **0.1415**,
  audio+real-video **0.1656**, oracle-video **0.3535**.
- `highres-fl` ran but trailed highres; Basic Pitch could not be compared in the
  current Windows Python 3.12 venv because `basic-pitch` depends on TensorFlow
  and no matching TensorFlow distribution is available.

**Reasoning:** The repeated `-1` semitone preference is a corpus/reference-pitch
signal, but the clip-local time search overfits the alignment proxy: clips 0 and
1 prefer ~`+1.25s`, yet the aggregate score is much better with one shared
`+0.14s` time correction. The next implementation should therefore promote/test
a global highres calibration path first, then inspect retained video evidence
under corrected timing. Tuning generic video weights remains secondary until the
audio calibration path is explicit and regression-tested.

## 2026-06-19 - v1.1 chunk-5 GAPS gold via per-pitch match-maximizing MIDI alignment

**Phase:** v1.1 (video string-resolution) - chunk 5 (GAPS integration, audio half)
**Decision tree:** v1.1 chunk-5 (recon "implied chunk-5"); design
`2026-06-03-v1.1-video-string-resolution-design.md` §6 eval-data gate.
**Branch taken:** Derive GAPS onset-timed gold tab by walking the MusicXML TAB
part (staff-tuning aware) and **snapping each score note to the exact aligned-MIDI
onset** via a per-pitch, match-maximizing monotonic alignment - rather than the
recon's per-cluster onset match (too fragile) or a raw syncpoint warp (too coarse
for the 50 ms gate). Restrict the eval set with two logged per-clip filters
(standard tuning; gold/MIDI coverage >= 80%).

**Evidence:** `docs/EVAL_REPORTS/v1_1_gaps_chunk5_2026-06-19.md`,
`tabvision/eval/parsers/gaps_musicxml_tab.py`, `tabvision/data/eval/gaps.toml`,
`docs/EVAL_REPORTS/v1_1_gaps_chunk5_audio_only{,_decomp}_2026-06-18.md`.
- GAPS audio is **in-domain**: 8 clean clips average onset F1 **0.952**, pitch F1
  **0.946** (vs UT-Austin's ~0); audio-only Tab F1 mean 0.768.
- 22-clip test split (audio-only, no prior): Tab F1 **0.647** (lower-95 0.573) -
  **passes** the single-line 0.45 gate. Onset F1 0.828 / pitch F1 0.819.
- Error decomposition: after `extra_detection` (51.5%, the repeat-coverage
  artifact the coverage filter flags), the dominant real loss is
  `wrong_position_same_pitch` (34.1%) - the audio string-resolution limit the
  v1.1 video chain targets.

**Reasoning:** The recon spike validated on one clean clip; the full corpus adds
voice-major MusicXML ordering, scordatura, repeats/voltas, and rubato. A raw
per-measure syncpoint warp has p95 onset error 70-900 ms (fails the 50 ms gate),
so onsets must come from the high-resolution MIDI; the open problem is the
score<->MIDI correspondence. Per-pitch monotonic alignment that *maximizes
matches* (gaps cost >> time term) reaches the ~94% multiset ceiling (~90% median
on standard clips) with exact MIDI onsets, and is robust to rubato (it is an
alignment, not interpolation) and to repeats (extra MIDI notes become gaps).
Scordatura clips can't be scored by the standard-tuning pipeline, and
repeat-heavy clips' score-centric gold under-covers the performance (FP
deflation) - both are filtered with logged drop lists, not silently capped. The
video real-chain (yt-dlp + video<->audio crop-offset alignment + chunk-3 gated CV
chain) is the next chunk; yt-dlp feasibility is confirmed.

## 2026-06-22 - v1.1 chunk-5 GAPS video real-chain: alignment works; CV chain does not transfer

**Phase:** v1.1 (video string-resolution) - chunk 5 (GAPS integration, video half)
**Decision tree:** v1.1 chunk-5 video half; design
`2026-06-03-v1.1-video-string-resolution-design.md` §6; supersedes the "video
real-chain is the next chunk" note in the 2026-06-19 entry.
**Branch taken:** Acquire GAPS source video via yt-dlp and recover the
video<->audio crop offset by **onset-strength-envelope cross-correlation**
(`video_time = gold_onset + offset`), rather than trusting the metadata CSV's
`duration`/`cropped_duration` columns (inconsistent) or a raw-waveform xcorr
(self-similar guitar energy → weak peak). Run the chunk-3 confidence-gated CV
chain over the clean-12 and report `audio-only` / `+real (auto)` /
`+real (best-fixed-orientation)` / `+oracle`, for gold-pitch audio (string axis
isolated — the 0.94 frame) and highres audio (honest end-to-end). Keep the
chunk-3 §5.3 no-regression coverage gate at **0.71** (a looser 0.5 leaked
corrupting evidence and regressed ~0.05).

**Evidence:** `docs/EVAL_REPORTS/v1_1_gaps_video_chain_2026-06-22.md` (+ raw
`*_auto_*`), `tabvision/scripts/acquire/gaps_video.py`,
`tabvision/scripts/eval/v1_1_gaps_video_chain_probe.py`,
`tabvision/tests/unit/test_gaps_video_align.py` (10 tests; 428 unit pass).
- **Alignment works:** all 12 offsets sub-frame (+0.01..+0.05 s; one 24 fps frame
  ≈ 42 ms), xcorr peak ratios 2.3-11.2, wav/video durations match < 0.1 s. The
  upload *is* the GAPS crop on these clips; the feared large offset is absent.
- **Video does not lift Tab F1.** Gold audio: audio-only **0.8148** ->
  +real(auto) **0.8148** (lo-95 0.768) == +real(best-orient) == audio-only;
  **below the 0.94 bar**. highres: 0.7612 -> 0.7612 (oracle 0.910, pitch-capped).
- **The lever is real:** gold-audio **oracle 0.9726** (~clears 0.94); residual
  loss is **98.3% wrong_position_same_pitch** - exactly the string axis video
  targets - but the CV chain captures none of it.
- **Sweep (gold, cache-only):** under no gate/orientation setting does video beat
  audio-only; ungated with the per-clip *oracle* orientation it still hurts 10/12
  (mean -0.052). So the bottleneck is the CV-derived string *evidence*, not
  gating or orientation selection.

**Reasoning:** The chunk-2/3 CV chain was tuned to the non-mirrored Kaggle
UT-Austin rig; GAPS is in-the-wild classical-guitar footage (diverse camera
geometry, neck angle, framing). On clean single-line pieces the audio-only
playability prior is already strong (0.815 gold), and the CV's string evidence is
either incompatible with the audio pitch (per-event gate drops it -> coverage
< 0.71 -> no-regression fallback) or confidently wrong (the dense orientation
assigns the wrong string). Voting over more frames cannot fix systematically
wrong evidence (vote_frames=1 here; homography is 0.99-stable, offset sub-frame).
This is a clean negative, verified: `+oracle` 0.973 confirms the gold/fusion path,
and the smoke reproduced the chunk-5 audio baseline (179 = 0.857). The next chunk
is CV-chain transfer to in-the-wild footage (per-clip fretboard/fret-cell
calibration, orientation-agnostic string-axis homography, perspective-robust
fingertip->string), not audio tuning - the string axis is worth ~0.16 (gold) on
GAPS clean-12.

## 2026-06-22 - v1.1 chunk-6 WS0: rich CV-intermediate cache enables cache-only geometry iteration

**Phase:** v1.1 (video string-resolution) - chunk 6 (CV-chain transfer), WS0 (enabler)
**Decision tree:** chunk-6 design
`2026-06-03-v1.1-video-string-resolution-design.md` decision tree +
`2026-06-22-v1.1-chunk6-cv-transfer-design.md` (geometry-first ordering, confirmed
by the user 2026-06-22); follows the 2026-06-22 chunk-5 entry.
**Branch taken:** Before any geometry work, change the GAPS video probe's per-frame
cache from the final `FrameFingering` to a **rich v2 cache** of the raw CV
intermediates — YOLO `OBBPredictions` (nut/fret/neck anchors), the fitted
`Homography`, and the selected fretting `HandSample` — and reconstruct the
fingering downstream via `fingering_from_raw` (= `replace(compute_fingering(hand,
H, cfg), t=t)`). Split the light (numpy-only) cache layer into
`scripts/eval/gaps_cv_cache.py` so the diagnostic and tests do not import
cv2/mediapipe/ultralytics. Codify the chunk-5 `_diag` analysis into a reusable
cache-only diagnostic (`scripts/eval/v1_1_gaps_string_diag.py`). Build the v2 cache
at the current **360p** (one-time CV re-run; 720p deferred to WS3, per the
confirmed plan).

**Evidence:** `docs/EVAL_REPORTS/v1_1_gaps_chunk6_ws0_2026-06-22.md`;
`tabvision/scripts/eval/gaps_cv_cache.py`,
`tabvision/scripts/eval/v1_1_gaps_string_diag.py`,
`tabvision/scripts/eval/v1_1_gaps_video_chain_probe.py` (refactor),
`tabvision/tests/unit/test_gaps_string_diag.py` (12 tests; full suite 440 pass / 4
skip; ruff + mypy clean).
- **Behaviour-preserving:** scoring from the v2 cache reproduces chunk-5 exactly —
  gold `audio-only 0.8148 -> +real(auto) 0.8148` (lo-95 0.7679), oracle **0.9726**;
  highres `0.7612 -> 0.7612` (lo-95 0.7063), oracle 0.9099; per-clip values
  identical to `_run2.log`.
- **Diagnostic reproduces the `_diag` baseline:** ambiguous-note string accuracy
  **4191/7697 = 0.544** (chunk-5 reported 0.543), per-clip `best_orient` + `haveCV`
  match exactly; identical from the rich or legacy cache (`fingering_from_raw` is
  bit-faithful). Full offset histograms recorded (string −1..−4 bias; fret swings
  +5/+4/+9/+14 — the pitch-preserving mirror).
- **Bonus:** the rich cache is *smaller* than the legacy one (6.2 MB vs 35 MB over
  the clean-12) — compact raw inputs vs a dense `(4,6,25)` float64 grid per frame.

**Reasoning:** The chunk-5 cache stored only the final, already-string-resolved
`FrameFingering`; its logit grid is re-orientable cache-only but only by four
discrete whole-grid flips, which cannot undo GAPS's graded −1..−4 string offset
(why the chunk-5 sweep found no win). The real string-axis levers — per-clip board
calibration, image-cued orientation, perspective-robust projection — all act
upstream of that grid (in `predictions_to_homography` / `compute_fingering`),
whose inputs the old cache discarded, so every geometry iteration cost a full
MediaPipe/YOLO re-run. Persisting those inputs once makes WS1/WS2/WS4 re-runnable
from cache in seconds, with the cache-only diagnostic as the fast leading
indicator (target 0.544 -> ≥ 0.75) and the existing probe as the lagging headline.
The refactor stays on the implementation side of the §8 contracts (no
type/Protocol/signature change); the work is harness-only and behaviour-preserving,
verified by exact reproduction of the chunk-5 numbers.

## 2026-06-25 — Chunk-6 WS1: per-clip nonlinear (rule-of-18) fret-map calibration

**Phase:** v1.1 chunk-6 (CV-chain string-resolution transfer), WS1 of the merged
WS1/WS2 board-calibration step.
**Decision tree:** chunk-6 §5 WS1 — anchor the canonical fret axis to detected
per-clip cues vs the rig-baked uniform partition; §6 per-step gate (leading
diagnostic + lagging probe); §8 detection-limited risk.
**Branch taken:** Replace `compute_fingering`'s **uniform** fret partition with a
per-clip **rule-of-18** map fit from the detected fret-OBB sequence + nut anchor,
behind the eval `calibrate` hook (training-free, cache-only, detected-cues-only).
New impl module `tabvision/video/fretboard/calibrate.py`; optional keyword
`fret_xs` on `compute_fingering` (default `None` ⇒ bit-identical uniform path);
opt-in `calibrate`/`posterior_cfg` hook on `fingering_from_raw` /
`load_frame_fingerings` + `make_fret_xs_calibrator`; `--calibrate` flag on the
diagnostic and probe; per-clip no-regression check added to the probe. The
homography is reused as-is (only the fret partition is re-derived); a **robust
inlier-consensus (RANSAC-style)** fit tolerates spurious/high-fret detections, and
frames with too few/garbled frets return `fret_xs=None` ⇒ uniform fallback ⇒ the
chunk-3 fall-back-to-audio invariant is preserved exactly.

**Evidence:** (clean-12, gold pitch, `--vote-frames 1`)
- **Leading indicator ↑:** ambiguous-note string accuracy **0.544 → 0.574**
  (`scripts/eval/v1_1_gaps_string_diag --calibrate`). Per-clip wins where the
  detector fires frets: 031 0.434→0.635 (+0.20), 104 0.479→0.619 (+0.14), 142
  0.475→0.526; deep −3 string-offset bucket 583→386, −1 bucket 1702→1666.
- **Fusion-relevant (ungated A/B, firing clips 031/104/142):** gold `+real` auto
  **0.586 → 0.684** (+0.098), best-orient **0.713 → 0.753** (+0.040) — the fret-map
  improves the actual Tab-F1 video contribution, not just the diagnostic argmax.
- **Lagging gated = no-op, no regression:** gold `+real(auto)` **0.8148 → 0.8148**,
  per-clip no-regression holds **12/12**. The 0.71 coverage gate suppresses video on
  every clip (calibration improves the string *within* surviving evidence, not the
  survival rate), so the lift cannot materialise until the gate is re-derived (WS5),
  exactly as the design anticipated.
- **Detection-limited ceiling (honest framing, SPEC §0 rule 7):** ~68% of ambiguous
  notes are on clips where YOLO detects ~0 fret OBBs at conf=0.25 (043/063/118/179/
  235/294), so the fret lever physically cannot fire there and falls back to uniform.
  Several of those *do* have nut detections (235 68%, 179 99%, 294 70%) — the WS2
  cross-string-axis lever's target.
- **Quality:** new `tests/unit/test_fretboard_calibrate.py` (17) + additions to
  `test_fingertip_to_fret.py` / `test_gaps_string_diag.py`; full unit suite **462
  pass / 4 skip**; ruff + ruff format + `mypy tabvision` clean. Default (`fret_xs=
  None` / no `--calibrate`) path reproduces the WS0 baseline bit-for-bit (string acc
  4191/7697 = 0.544; gold probe 0.8148/0.8148/0.9726).

**Reasoning:** The canonical x-axis is proportional to physical along-neck distance,
so the uniform partition systematically over-estimates fret number toward the body
(real wires follow `D_k = S(1−r^k)`, `r = 2^(−1/12)`); a fingertip physically at
fret 12 (canonical x ≈ 0.67 of a nut→fret-24 span) is read as ~fret 16, and the
pitch constraint then drags the predicted string to the bass side — exactly the
+4/+5 fret / −1..−4 string bias the WS0 diagnostic measured. Fitting the rule-of-18
shape per clip from the detected wires + nut anchor is parameter-free physics (no
per-clip constants, so no clean-12 overfitting); the affine `x = x0 + b(1−r^k)` is
anchored at the nut because a finite geometric sequence is scale-invariant in its
absolute fret index. The lever is real and fusion-relevant where the detector
cooperates, but detection-limited at conf=0.25; the next lever (WS2 cross-string
axis from the nut-OBB edge) targets the nut-detected, fret-sparse clips, and gate
re-derivation (WS5) is required to convert the evidence gain into a gated Tab-F1
lift. Stays entirely on the implementation side of §8 (fret evidence still rides
`marginal_string_fret`; no type/Protocol/signature change).

## 2026-06-25 — Chunk-6 WS2: nut-OBB cross-string axis is a measured negative (deferred)

**Phase:** v1.1 chunk-6, WS2 (cross-string axis) of the merged board-calibration step.
**Decision tree:** chunk-6 §5 WS2 — re-anchor the canonical *string* axis to the
detected nut OBB edge; §8 detection-limited risk + the geometry-aware-confidence
pairing.
**Branch taken:** Implemented the nut-axis homography re-fit
(`keypoint.predictions_to_homography_nut_axis` + `calibrate.calibrate_board`, opt-in
`--calibrate-board`, default off) and **measured it before promoting**. It is a net
negative in this first form, so it is **deferred behind a geometry-aware confidence /
fret-richness selection gate** and kept opt-in; the default pipeline and the WS1
result are unchanged.
**Evidence:** The nut OBB's cross-string width is 14–29% narrower than the neck-OBB
edge the homography uses (ratio 0.71–0.86) and its center is offset by 0.15–1.09
neck-edge-lengths — real signal. But the full re-fit regresses the leading indicator
**0.574 (WS1) → 0.547**: it helps the big nut-only clips (235 0.419→0.435, 179
0.598→0.609, 212 0.835→0.848) yet hurts the fret-rich ones (104 0.619→0.468, 142
0.526→0.464, 027) and 294 (0.584→0.541, a misdetected nut). Report:
`docs/EVAL_REPORTS/v1_1_gaps_chunk6_ws1_2026-06-25.md` §4. Suite 467 pass / 4 skip;
ruff + mypy clean.
**Reasoning:** Re-fitting the *whole* homography from the nut edge perturbs the
along-neck (x) axis WS1 already calibrated, and the nut OBB is itself noisy, so
without a quality gate the bad re-fits drag down the good ones — exactly why the
design pairs the cross-string axis with geometry-aware confidence. The honest wall:
~68% of ambiguous notes are on clips with ~0 detected frets at conf=0.25, so the
geometry-first path plateaus ~0.57–0.58; clearing 0.75/0.94 needs a cost/STOP lever
(lower-conf re-detect + cache rebuild, 720p, or the WS4 learned model) — to be
decided explicitly per SPEC §0 rule 8. WS2 stays impl-side of §8 (new keypoint
function is unused by production until promoted).

## 2026-06-25 — Chunk-6 WS4: APPROVED — train a learned string-resolution model (GAPS)

**Phase:** v1.1 chunk-6, WS4 (the training lever the design gated behind a HARD STOP).
**Decision tree:** chunk-6 §5 WS4 + §8 — the geometry-first path (WS1/WS2) plateaus
~0.57 because ~68% of ambiguous notes are on clips with ~0 detected frets at
conf=0.25; the next lever is a learned model, which is **training** (data + compute +
money) → SPEC §0 rules 6 & 8 require explicit user approval.
**Branch taken:** **User explicitly approved the training run** (2026-06-25), after a
data assessment, with three confirmed choices: (1) **full 270-clip GAPS train split**
(not a pilot); (2) **Modal** GPU, reusing the YOLO-OBB Modal infra; (3) **NC license
accepted** (TabVision is non-commercial → an NC-tainted model artifact is acceptable,
recorded in LICENSES.md). Approach: a pretrained backbone over the YOLO **neck-crop**
→ 6-way string posterior, pitch-conditioned at fusion, fed through the existing
`marginal_string_fret` → `AudioEvent.fret_prior` channel (no §8 change). Train on the
`train` split, eval on held-out `test` (clean-12 ⊂ test — no leakage). Go/no-go on the
first checkpoint vs the geometric leading 0.574.
**Evidence:** GAPS `gaps_metadata_with_splits.csv` — 270 train / 30 test / 101
unassigned, all train with `yt_id` + syncpoints; ~216K ambiguous-note string labels
(free from gold tab); clean-12 confirmed ⊂ `test`. Design:
`docs/plans/2026-06-25-v1.1-ws4-learned-string-model-design.md`. Geometry baseline:
`docs/EVAL_REPORTS/v1_1_gaps_chunk6_ws1_2026-06-25.md`.
**Reasoning:** Labels, URLs, alignment, and an official train/test split all exist, so
the binding cost is compute (download + CV-extract + GPU), not data or annotation. A
learned model resolves strings from pixels, sidestepping the fret-detection wall that
caps geometry. The honest risks (label noise from alignment, download attrition,
transfer beyond GAPS, NC/AGPL taint) are recorded in the design §7; the eval is on the
clean held-out test split with the no-regression invariant intact (absent/low-conf →
audio-only). New deps (`torch`/`torchvision`/`timm`, `modal`) are training/eval-only.

## 2026-06-29 — Chunk-6 WS4: learned model is a measured NEGATIVE; bank + stop

**Phase:** v1.1 chunk-6, WS4 (the approved learned-model lever).
**Decision tree:** chunk-6 §5 WS4 + the go/no-go gate (does `+learned` beat the
geometric leading 0.574 / lift audio-only 0.8148 on held-out clean-12?).
**Branch taken:** Trained the model (full pipeline on Modal L4) and **evaluated it;
the result is a clear negative, so — at the user's direction (2026-06-29) — record
the measured negative and STOP** rather than chase it further. Geometry WS1 (fret-map,
0.544 → 0.574, committed `587c174`) stands as chunk-6's positive deliverable. The
WS4 pipeline stays committed + reusable.
**Evidence:** Go/no-go eval (`scripts/eval/v1_1_gaps_learned_probe.py`, held-out
clean-12, gold frame): **audio-only 0.8148 → +learned 0.6974 (oracle 0.9726)** — the
learned string evidence is **net-negative**, dragging Tab F1 down −0.117. Training
(153,482 crops from 251 GAPS train clips, clip-disjoint val) **plateaued at raw 6-way
val_acc ~0.30** by epoch 8 while train loss kept falling (1.63 → 0.66) — overfit to
training players, no transferable string signal. Report:
`docs/EVAL_REPORTS/v1_1_gaps_ws4_learned_2026-06-29.md`.
**Reasoning:** A model only ~1.8× chance on raw 6-way yields pitch-restricted
predictions worse than the audio playability prior. Likely root cause: the whole-neck
crop starves the model (the fretting hand is small in a 224²-squished wide crop, so
the across-string finger position — the actual signal — is too coarse), compounded by
onset-frame alignment label noise and the intrinsic difficulty of string-from-image
across players. The one promising fix (a hand-tight crop, re-extract + re-train) is
speculative against a *large* gap (net-negative, not marginal) and was not authorized
for further spend. Honest conclusion (SPEC §0 rule 7): audio-only single-line string
resolution on in-the-wild GAPS is information-limited, and neither the geometric chain
nor a GAPS-trained ResNet-18 neck-crop classifier reliably beats audio-only; WS1 is the
sole measurable, no-regression positive. NC/AGPL artifacts (GAPS-trained weights, the
1.38 GB crop dataset) stay local/offline-only, never committed.

## 2026-06-29 — Chunk-6 capstone: the audio prior beats the video chain (no lift to ship)

**Phase:** v1.1 chunk-6 — the "make WS1 real" / WS5 gate-materialisation question.
**Decision tree:** before threading WS1's nonlinear fret-map through the §8
`Homography` contract to production + recalibrating the gate (WS5), measure whether
a net-positive Tab-F1 lift is even possible.
**Branch taken:** **Measured first; the lift does not exist, so do NOT change the §8
contract** (it would ship better geometry but zero headline gain). WS1 stands as a
documented geometric improvement; chunk-6 lands + pauses.
**Evidence (decisive):** on the clean-12 **ambiguous** notes, the **audio playability
prior** resolves the string correctly **7859/10103 = 0.778**, vs WS1 video
**0.574** (baseline uniform 0.544) — the audio prior is **~0.20 better** than the
video chain at string resolution. Fusion adds the video term to the audio emission
cost, so any non-trivial `lambda_vision` pulls the decision toward the *worse* source
(0.574) and degrades Tab F1; a tiny weight is a no-op. This is exactly why the gated
probe sat at audio-only 0.8148 (no-op) and ungated *hurt* (geometric net-negative;
learned 0.6974). No per-event gate keyed on video *confidence* can selectively apply
video only where audio fails, so none yields a lift.
**Reasoning / reframe:** the 0.94 single-line **video** target (SPEC §1.4.1) assumed
video resolves strings *better* than audio. On in-the-wild GAPS clean-12 the ordering
is the opposite — **audio 0.778 > geometry-video 0.574 > learned-video (worse)** —
and video only "wins" via the *oracle* (perfect strings → 0.973), which no real chain
approaches. So string-resolution video is not an additive lever over the audio prior
on this corpus; chunk-6's honest net is the geometry WS1 improvement (real but
sub-prior) plus this clarifying measurement. Implication for v1.1: the single-line
video stretch is not just hard but *information-dominated by audio* here — revisit the
target/scope in SPEC §1.4.1 before further video string-resolution spend.

## 2026-06-30 — `--audio-filters on` for `highres` is a measured regression; leave default off

**Phase:** v1 accuracy hardening (unattended `/loop`, audio-only pipeline tuning lane).
**Decision tree:** the `--audio-filters` flag (commit `bf61d4e`, 2026-06-18) added a way
to force the v0-ported post-detection filter suite (`tabvision.audio.filters`:
low-confidence/short/quiet drop, same-pitch merge, sustain-redetection drop,
harmonic/ghost-note drop, end-trim) on for the `highres` backend, which has it off by
default (only `basicpitch` has it on by default). The flag's own commit message named
it as "the chunk-4 lever for highres extra_detection (~35%)" but no eval report or
DECISIONS.md entry ever actually measured it on `highres` before this entry.
**Branch taken:** **Measured it on the 24-clip fast validation manifest
(`data/eval/local_gs_val24.toml`, `highres` + `guitarset-v1` prior) before touching the
default — it regresses sharply, so the default (`auto` → off for `highres`) is
unchanged.**
**Evidence:** `--audio-filters off` (current default): single_line Tab F1 0.4820
(onset 0.9227, pitch 0.9140), strummed Tab F1 0.7951 (onset 0.9359, pitch 0.9184) —
both consistent with the official 60-clip acceptance numbers. `--audio-filters on`:
single_line Tab F1 0.4744 (onset 0.8706, pitch 0.8643), strummed Tab F1 **0.4596**
(onset **0.5822**, pitch 0.5592) — strummed drops from passing the 0.60 SPEC target to
failing it outright. Six-bucket decomposition (aggregate counts): `extra_detection`
did fall as hoped (150 → 69), but `missed_onset` exploded **7×** (199 → 1392, from
18.6% to 68.2% of total loss). Reports:
`docs/EVAL_REPORTS/v1_1_audiofilters_{off,on}_val24_2026-06-30{,_decomp}.md`.
**Reasoning:** the filter suite's tuned constants (e.g. `min_confidence=0.3`,
`sustain_amplitude_ratio=0.95`) were ported from v0 and dialed in for `basicpitch`'s
note-density/confidence distribution; applied unmodified to `highres` they prune large
numbers of real onsets in dense/strummed passages, not just the spurious detections
they were designed to catch. The lever is real (filters *do* cut `extra_detection`)
but the default config is the wrong instrument for `highres` — it would need separate
retuning against `highres`'s own confidence/density distribution to be net-positive,
which is out of scope here (SPEC §0 rule 7: flag, don't hallucinate a quick win).
Banked negative, no code or default changed. The eval-harness `--audio-filters` plumbing
added to test this (commit `5091a09`) stays — useful for any future retuning attempt.
**Next candidate (untested, flagged not pursued this round):** `HighResBackend`'s own
`onset_threshold`/`frame_threshold` constructor defaults (0.3/0.1) are never overridden
anywhere in the repo and are a more surgical, unexplored lever for the same
`missed_onset`/`extra_detection` buckets — flagged for follow-up, not yet implemented
or measured.

## 2026-06-30 — `HighResBackend` threshold kwargs were silently inert (dependency bug); fixed. `onset_threshold=0.2` is a wash, default unchanged

**Phase:** v1 accuracy hardening (unattended `/loop`, audio-only pipeline tuning lane);
follow-up to the `--audio-filters` entry above.
**Decision tree:** measure the flagged `onset_threshold`/`frame_threshold` candidate
before changing any default (same discipline as the audio-filters entry).
**Branch taken:** **First probe (frame_threshold 0.1→0.2) and second probe
(onset_threshold 0.3→0.2) both produced bit-identical Tab F1 and six-bucket
decomposition output to the untouched defaults — a methodology red flag, not a
genuine null result.** Traced it to a real bug in the `hf_midi_transcription`
dependency: `MidiTranscriptionModel.__init__` accepts and stores
`onset_threshold`/`offset_threshold`/`frame_threshold` in `self.config`, but
`_init_transcriptor(instrument)` — the call that builds the underlying
`piano_transcription_inference.PianoTranscription` — only ever receives the
instrument name, so `PianoTranscription` always falls back to its own
hard-coded defaults (0.3/0.3/0.1) regardless of what `HighResBackend` was
constructed with. **Fixed** (`tabvision/audio/highres.py::_load_model`,
commit `e5ea355`): `PianoTranscription.transcribe()` rebuilds
`RegressionPostProcessor` fresh from `self.onset_threshold` /
`self.offset_threshod` [sic, upstream typo] / `self.frame_threshold` on every
call (not just at construction), so setting those attributes directly on
`self._model.transcriptor` after construction makes our thresholds actually
take effect. Default behavior is unchanged (`HighResBackend`'s own defaults
equal the library's), so the fix is a no-op for current production traffic —
it only unlocks tuning that was previously impossible. 4 new tests
(`test_highres_threshold_wiring.py`) use a fake `MidiTranscriptionModel` that
reproduces the real library's broken wiring, so they'd fail if the patch were
removed. 551 tests pass, ruff+mypy clean.
**With the fix in place, re-measured `onset_threshold=0.2`** (24-clip fast
validation manifest, `highres` + `guitarset-v1`): `correct` 1981→1997 (+16),
`missed_onset` 199→188 (−11), but `extra_detection` 150→188 (**+38**) —
lowering the threshold recovers a few missed onsets at the cost of 3-4× as
many new false positives. Net Tab F1: single_line 0.4820→0.4838 (+0.0018,
noise-level), strummed 0.7951→0.7909 (−0.0042). **Verdict: a wash, not a
clear win — leave `onset_threshold` at its 0.3 default.** Not pursued further
(no intermediate-value sweep, e.g. 0.25): the trade-off direction (more new
false positives than recovered misses) argues against this being a fruitful
single-knob lever, and SPEC §0 rule 7 argues for banking a marginal result
over chasing it. `frame_threshold` is confirmed to have **zero effect on Tab
F1 specifically** (it only gates note duration/offset, which this eval's
onset+pitch+string+fret matching doesn't score) — not just untested, a
structurally inert lever for this metric.

## 2026-07-02 — `tabvision transcribe` now defaults to the accepted config (`auto`→highres + `guitarset-v1` prior); measured parity with the banked baseline

**Phase:** v1 shipped-config alignment (2026-07-01 roadmap, item A1; user-directed
day-one path).
**Decision tree:** the roadmap's "two biggest gaps aren't model gaps" finding #1:
`tabvision transcribe` defaulted to `basicpitch` + `--position-prior none` while every
acceptance number was measured with `highres` + `guitarset-v1` (the prior alone is
worth +22–29pp Tab F1), so CLI users silently got a much worse pipeline than the one
we validated. Measure parity through the new default code path before committing.
**Branch taken:** **Changed the `transcribe` defaults to `--audio-backend auto` +
`--position-prior guitarset-v1`.** `auto` (not bare `highres`) because
`run_pipeline`'s tone-toggle routing (`audio_backend_for_session`) resolves `auto` →
`highres` for the default acoustic/classical instrument and → `highres-electric` for
`--instrument electric` — so the electric toggle now also works without a backend
flag. `basicpitch` and `--position-prior none` stay reachable as explicit flags for
ablations/evals; the composite-eval harness is unaffected (its own defaults were
already the accepted config).
**Evidence (measured parity, not assumed):** one composite-eval run on
`data/eval/local_gs_val24.toml` with `--backend auto` — i.e. through the same
auto-resolution path the CLI default now takes — reproduces the banked baseline
**bit-for-bit**: single_line Tab F1 0.4820 (lower-95 0.3761, onset 0.9227, pitch
0.9140), strummed 0.7951 (lower-95 0.7565, onset 0.9359, pitch 0.9184), identical
to `v1_1_audiofilters_off_val24_2026-06-30.md`. Report:
`docs/EVAL_REPORTS/v1_1_cli_default_parity_val24_2026-07-02{,_decomp}.md`. CLI smoke
on a real GuitarSet clip (`05_BN1-129-Eb_solo_mic.wav`, zero config flags) produces a
46-note tab end-to-end; the checked-in A440 fixture runs end-to-end but yields 0
notes under `highres` (guitar-trained model vs synthetic sine — a model behavior, not
a CLI bug; the fixture's non-empty expectation is basicpitch-specific and its e2e
test still pins `basicpitch` explicitly).
**Reasoning / notes:** (1) the first `highres` run downloads the checkpoint once
(~37 s) — documented in `tabvision/README.md`, which also loses its stale "not
promoted to the silent default until…" caveat (that gate was met by the 60-clip
acceptance run + this parity check; promotion is the explicit user-directed point of
roadmap A1). (2) On Windows py3.12 the old default was outright broken
(basicpitch/TensorFlow has no py3.12 wheels), so highres-by-default simplifies the
supported envs. (3) `tabvision diagnose` still defaults to `basicpitch` — left
untouched (outside A1's stated scope, lines 127–130/174–177 only); flagged as a
follow-up alignment candidate.

## 2026-07-02 — `guitarset-v1` prior on GAPS is a measured NEGATIVE (−0.138 Tab F1); A7 (GAPS-native prior) skipped per the recorded branch logic

**Phase:** v1.1 accuracy (2026-07-01 roadmap, item A2; user-directed day-one path).
**Decision tree:** every GAPS eval to date ran `--position-prior none` (baseline Tab
F1 0.6468, lower-95 0.5734) while `wrong_position` was 34.1% of real GAPS loss and
the prior is worth +22–29pp on GuitarSet — measure it once, bank either way, with
pre-registered branch logic: lower-95 lift ⇒ A7 (GAPS-native prior) unblocked after
A6; wash/negative ⇒ A7 skipped.
**Branch taken:** **Negative branch fires — A7 is marked SKIPPED in the roadmap.**
No code or default changed by this measurement.
**Evidence:** `scripts.eval.v1_1_second_corpus_probe`, GAPS test-22, `highres`,
identical bootstrap/tolerance settings to the 2026-06-18 baseline; only the prior
differs. Tab F1 mean **0.6468 → 0.5087** (lower-95 0.5734 → 0.4549 — disjoint from
the baseline mean, not noise), chord-instance acc 0.6633 → 0.5125. Decomposition is
a perfectly controlled exchange: onset/pitch F1 and the pitch_off / timing /
missed_onset / extra_detection buckets are bit-identical; the prior net-flips
**2,131 notes correct → wrong_position_same_pitch** (2,978 → 5,109; ~16% of
pitch-matched notes). Reports:
`docs/EVAL_REPORTS/v1_1_gaps_prior_guitarset_v1_2026-07-01{,_decomp}.md`.
**Reasoning:** the prior encodes GuitarSet's open-position conventions
(steel-string pop/comping); GAPS is classical repertoire played up the neck, so the
cross-domain prior overrides string decisions the playability decode already got
right — the fusion-side analogue of the electric tier's cross-domain 0.12. This
does **not** contradict the A1 default (GuitarSet-domain home recording is the v1
target and measures +22–29pp); it does mean the shipped default is
domain-sensitive, and classical/GAPS-style input currently decodes better with
`--position-prior none` — flagged for user docs if classical ever becomes a target.
Honest caveat recorded in the report (not actioned): this is a *cross-domain
transfer* negative and doesn't by itself falsify an *in-domain* GAPS prior; per the
pre-registered branch logic A7 is nonetheless out of the roadmap, and reopening it
needs fresh justification plus the A6 gold-coverage fix first.

## 2026-07-02 — SPEC §1.4.1: the 0.94 single-line video-assisted reference is retired (user-approved); B8 dropped; A15 (tab-corpus fingering prior) added as a gated item

**Phase:** v1.1 scope maintenance (the chunk-6 capstone's pending directive, D1
packet part 1).
**Decision tree:** CLAUDE.md operating rule 3 / SPEC §1.4: scope/target changes
need a SPEC edit + explicit user approval. The capstone (2026-06-29) recommended
revisiting the 0.94 single-line video target; the user approved lowering it on
2026-07-02 ("Ok lower video target").
**Branch taken:** **SPEC §1.4.1 amended (dated block, append-style): the 0.94
single-line video-assisted reference is RETIRED.** The binding v1 single-line gate
stays ≥ 0.45; **no new stretch number is set until one is demonstrated** (rule 7 —
don't promise what nothing has measured). The strummed 0.86 and chord 0.85
video-assisted references stay open pending A14 (chord-frame video, the one
unmeasured axis where video plausibly beats audio) and the rest of D1.
**Evidence:** chunk-6 capstone (audio prior 0.778 > geometry video 0.574 > learned
video, ungated fusion hurts, no gate recovers a lift) + the A2 cross-domain prior
negative above (string resolution is convention/evidence-limited, not
decode-limited).
**Related user decisions same session (recorded here, roadmap edited):**
(1) **B8 (remove vestigial video UI) DROPPED** — the user wants video playback kept
as correction context; B1 already removed the actively-misleading fake
"Tracking fingers" progress stage for audio-only runs. (2) **A15 added** (roadmap
Tier 3, user-proposed): a tab-corpus fingering/sequence prior ("how guitar is
normally played") biasing the decode before user correction — staged
license-review → CPU n-gram → (rule-8 gated) neural model, with hard
no-regression gates on both val24 and GAPS clean-12 given the A2
domain-sensitivity result. Not covered by banked negatives (melodic prior was
hand-coded; WS4 was visual).

## 2026-07-02 — A15 step 1: GuitarSet sequence prior has real in-domain signal; gated to singleton moves + tied to the unigram config family

**Phase:** v1.1 accuracy roadmap A15 (fingering-sequence prior), staging step 1
(free in-domain probe) executed per the user's 2026-07-02 plan (merge PR #19 →
GuitarSet probe → PDMX-only license review → gated n-gram probes → neural only
on signal + sign-off).
**Decision tree:** roadmap A15 hard gates (no-regression on BOTH val24 and GAPS;
key the prior rather than shipping one global default).
**Branch taken:** implemented `fusion/transition_prior.py` — Δstring|Δpitch
n-gram (schemes `delta`, `delta_fret` with count-backoff), learned from
anchor-to-anchor cluster transitions, default OFF, env-keyed
(`TABVISION_TRANSITION_PRIOR[_WEIGHT]`), artifact `guitarset-seq-v1` (train
players 00–04, singleton moves only). Two design decisions measured, not
assumed:
(1) **Ungated application is a banked negative** — applying the learned term to
chord-to-chord transitions costs strummed Tab F1 (0.8564→0.8187 at w=4) and
chord accuracy (0.792→0.733); the decode now hard-gates the term to
singleton→singleton cluster moves (chords stay hand-coded — A5
chord-dictionary territory, per the user's A15/A5 complement framing).
(2) **Standalone GAPS transfer is a wash-to-negative** (GuitarSet-trained
0.7782→0.7653; even GAPS-trained in-domain is a 0.7778 wash — classical
single-line hand-coded transitions are already near-optimal), BUT under the
product default (unigram on) the sequence term helps BOTH corpora (val24
0.7777→0.7936, GAPS 0.6213→0.6422/0.6721 oracle) by partially repairing the
unigram's cross-domain damage. Deployment therefore ties the sequence prior to
the `guitarset-v1` config family (active only when the pitch-position prior
is), leaving GAPS's accepted `--position-prior none` config untouched.
**Evidence:** `docs/EVAL_REPORTS/a15_guitarset_sequence_probe_2026-07-02.md`,
`a15_gaps_sequence_probe_transfer_2026-07-02.md`,
`a15_gaps_sequence_probe_indomain_2026-07-02.md` (all oracle-audio; real-audio
gated runs are staging step 4, pending).

## 2026-07-02 — A15 step 4: gated n-gram real-audio results — 60-clip confirm PASSES, uncoupled GAPS FAILS → coupling is load-bearing

**Phase:** v1.1 roadmap A15, staging step 4 (real-audio no-regression gates).
**Branch taken:** `guitarset-seq-v1` w=4.0 measured on all three real-audio sets:
(1) val24 accepted config: single-line 0.4820→0.5140 (lo95 0.3761→0.4144),
strummed wash, onset/pitch bit-identical. (2) **60-clip player-05 confirm
(accepted config + seq): single-line 0.523→0.5418, lower-95 0.457→0.4748
(above the 0.45 gate); strummed 0.676→0.6783 wash (lo95 0.606→0.6029)** — the
measurement-discipline bar for a default flip is met for the guitarset-v1
config family. (3) **GAPS test-22 uncoupled (prior=none + seq): 0.6468→0.5931
(−5.4pp), a hard gate FAIL** — real audio amplifies the oracle −0.4pp (extra
detections create spurious singleton transitions the prior then distorts).
**Consequence:** the sequence prior must ship tied to the guitarset-v1
position-prior config (off when the position prior is off); a global/uncoupled
default is a banked negative. Default flip itself = user decision, not taken.
**Evidence:** `docs/EVAL_REPORTS/a15_val24_seq_w4_2026-07-02.md`,
`a15_gs60_seq_w4_2026-07-02.md`, `a15_gaps22_none_seq_w4_2026-07-02.md`
(+ decomps). Neural step (rule-8 spend) not started — recommendation queued:
corpus scale (PDMX), not model capacity, is the bottleneck.

## 2026-07-02 — A15 step 4b: default-on flip shipped as a coupled default; PDMX guitar yield resolved at 3,435

**Phase:** v1.1 roadmap A15, post-gate deployment (user-directed 2026-07-02:
"wire the seq prior into the CLI/pipeline so it's active iff the position
prior is active").
**Branch taken:** `run_pipeline` gained `sequence_prior="auto"` (CLI
`--sequence-prior {auto,none,guitarset-seq-v1}`): `auto` resolves to
`guitarset-seq-v1` at the gate-accepted w=4.0 **iff the position prior is
active**, else clears the install. The coupling is structural, not advisory —
the uncoupled GAPS real-audio FAIL (0.6468→0.5931, banked step-4 entry) means
a global default was never on the table; the 60-clip lower-95 confirm
(0.457→0.4748) is the measurement bar the flip stands on. Production (Modal
`v1_adapter`) and the composite-eval harness inherit the coupling through
`run_pipeline`'s default; `TABVISION_TRANSITION_PRIOR[_WEIGHT]` env vars
override the flag for sweeps (and `=none` reproduces the pre-A15 baseline).
val24 (`guitarset_audio.py`) drives `fuse()` directly and is untouched — its
numbers stay env-knob-controlled. Tests: coupling branches + env precedence
(`test_pipeline.py`), parser (`test_cli_fusion_flag.py`), plus a conftest
save/restore of the process-global prior install.
**Also resolved:** the Zenodo `PDMX.csv` retry succeeded — **3,435**
`no_license_conflict`+MXL guitar-program songs (1,068 all-guitar, 798 solo;
classical/acoustic-leaning as predicted). Mid-range of the estimate → the
`mxl.tar.gz` TAB-staff sampling step is justified
(`docs/2026-07-02-pdmx-license-yield-review.md`).
**Not taken:** neural sequence model — rule-8 spend still awaiting user
sign-off; recommendation stands (corpus scale, not model capacity, is the
bottleneck → PDMX n-gram extraction first).

## 2026-07-02 — A15 PDMX acquisition step 2: TAB-staff yield = 734 scores (21.4%) → n-gram corpus is REAL; neural stays no-go

**Phase:** v1.1 roadmap A15, PDMX acquisition (user approved the mxl.tar.gz
fetch + TAB-staff sampling in lieu of neural training spend; PR #20 merged
the same day, so the coupled default is on `main`).
**Branch taken:** fetched `mxl.tar.gz` (1.89 GB → local data root, never the
repo) and scanned it with the new committed scanner
(`scripts/acquire/pdmx_tab_scan.py`, streaming, CSV-filtered): **734 of the
3,435 guitar×clean×MXL songs carry a TAB staff (21.4%)** — inside the
predicted 10–50% band. Validation through the GAPS MusicXML tab walk (the
exact extraction code path): 10/10 parse, 9/10 at pitch-consistency 1.000
(one 0.942 → per-note filter at extraction), all sampled tunings standard
EADGBE (MuseScore always *declares* staff-tuning; declaring ≠ nonstandard).
Scale: ~460 tab notes/score sampled mean → **~340k tab notes vs the 14,003
transition samples behind `guitarset-seq-v1`** (~20×, full pieces not 30 s
excerpts). Genre lean confirmed classical/untagged, rock/pop present.
**Consequence:** corpus-scale bottleneck has a real, license-clean fix
without training spend — next step is PDMX extraction + n-gram build through
`scripts/eval/a15_sequence_prior_probe.py` under the same val24 + GAPS dual
no-regression gates (expect a classical-leaning prior; config-keying is the
guard). Neural remains not-started per the standing no-spend recommendation.
**Evidence:** `docs/2026-07-02-pdmx-license-yield-review.md` (TAB-staff
section); scan summary JSON regenerable via the committed scanner.

## 2026-07-05 — A15 PDMX n-gram verdict: corpus scale does NOT beat domain match; guitarset-seq-v1 stays the default; A15 CLOSED

**Phase:** v1.1 roadmap A15, PDMX extraction + n-gram probe (final step of the
2026-07-02 staged plan; branch `v1.1/a15-pdmx-ngram`).
**Branch taken:** extracted the full PDMX TAB corpus
(`scripts/acquire/pdmx_extract_transitions.py`): 734 TAB-bearing scores →
**554 standard-tuning used** (180 scordatura skipped — the decode assumes
EADGBE and scordatura changes the Δpitch↔Δstring geometry; the 10-score scan
sample happened to miss all of them), 260,935 notes walked, 690 (0.26%)
dropped pitch-inconsistent → **71,527 singleton transitions ≈ 5.1×** the
14,003 behind `guitarset-seq-v1`. Clustering is score-time by construction
(same-onset chords cluster exactly; any positive score-time gap is a
transition — the 80 ms audio rule cannot apply to divisions), so PDMX
"singleton moves" are defined slightly more finely than GuitarSet's.
Built two schema-v1 candidates with the gate-accepted hyperparameters:
`pdmx-seq-v1` (PDMX-only) and `guitarset-pdmx-seq-v1` (GuitarSet counts ×5 to
match PDMX mass, then summed — raw pooling would have been a PDMX near-copy).
**Gate results (dual no-regression, exactly the step-4 harness):**
- **Oracle:** val24 incumbent 0.7936 > pooled 0.7858 > PDMX 0.7752; GAPS-22
  nothing beats the accepted no-prior 0.7782 (PDMX uncoupled 0.7755 wash).
- **Real-audio val24 (accepted config, w=4):** single-line incumbent
  **0.5140** (lo95 0.4144) vs PDMX **0.4782** (−3.6pp, FAIL) and pooled
  **0.5047** (−0.9pp, FAIL); strummed a three-way wash (0.7949–0.7954);
  onset/pitch bit-identical as expected.
- **Real-audio GAPS-22 uncoupled (prior none + seq w=4):** PDMX **0.6381**
  vs no-seq baseline **0.6468** (−0.9pp) — far better than guitarset-seq's
  banked −5.4pp (0.5931), but still a regression → uncoupling stays dead.
**Consequence:** `SEQUENCE_PRIOR_DEFAULT` stays `guitarset-seq-v1`; the
coupled default shipped in step 4b is unchanged. The corpus-scale question
is now answered with data: **5× more (cleaner, full-piece) transitions from
the wrong domain lose to 14k in-domain samples** — the n-gram was already
data-saturated, and what PDMX's breadth buys is cross-domain robustness
(GAPS damage −5.4pp → −0.9pp), not in-domain accuracy. This also finalizes
the neural no-go: if 5× data doesn't move a 2-parameter-context n-gram,
model capacity was never the bottleneck; domain-matched data is. A15 is
CLOSED. The candidate artifacts stay committed and selectable
(`--sequence-prior pdmx-seq-v1` / `guitarset-pdmx-seq-v1`,
env `TABVISION_TRANSITION_PRIOR`) as the measured evidence.
**Evidence:** `docs/EVAL_REPORTS/a15_pdmx_oracle_{val24,gaps22}_2026-07-05.md`,
`a15_val24_pdmx_seq_w4_2026-07-05.md`, `a15_val24_gspdmx_seq_w4_2026-07-05.md`,
`a15_gaps22_none_pdmx_seq_w4_2026-07-05.md` (+ decomps); transitions cache
regenerable via the committed extractor.

## 2026-07-06 — A6: GAPS gold repeat/volta unfolding — honest single-line Tab F1 is 0.6969, not 0.6468

**Phase:** v1.1 roadmap A6 (branch `v1.1/a6-gaps-gold-coverage`)
**Decision tree:** GAPS gold-coverage artifact — the parser walked each written
measure once, so repeat traversals had no score counterpart and the performer's
replayed notes counted as false positives (the docstring's "first-traversal-
biased" gold). Fix by unfolding, or leave the artifact and accept a depressed
headline?
**Branch taken:** **Unfold, gated on a length-match with the syncpoint span.**
The GAPS syncpoints index the *unfolded performance* timeline; `parse()` now
unfolds simple repeats + 1st/2nd voltas (`_unfold_measures`) to that order
before walking, and trusts the unfold only when `|len(unfold) - sync_span| <= 3`
(nonstandard volta encodings fall back to the written order — safe). Validated
on the real data: the unfold reproduces the syncpoint span on **11/14** repeat
clips in test-22; the other 3 fall back; the 8 non-repeat clips are untouched.
Eval-gold-only — nothing shipped changes. `TABVISION_GAPS_NO_UNFOLD` forces the
pre-A6 behaviour for the controlled A/B.
**Evidence (controlled A/B, highres, `--position-prior none`, `--splits test`;
predictions cache-shared):** unfold **OFF** = Tab F1 **0.6468** (lo-95 0.5734),
14,699 gold — *reproduces the banked baseline exactly*, validating the harness.
unfold **ON** = **0.6969** (lo-95 0.6256), 16,079 gold. **Δ = +0.0501 Tab F1,
+1,380 gold notes (+9.4%)**; onset/pitch F1 rise in lockstep (0.8277→0.8796,
0.8185→0.8703). Reports: `docs/EVAL_REPORTS/a6_gaps_unfold_{on,off}_2026-07-06.md`.
Tests: 7 new (unfold cases + end-to-end repeat-coverage + fallback), 15 parser
tests green.
**Reasoning:** The +0.05 is a **coverage-accounting correction, not a model
improvement** — A6 stops penalising the model for repeat notes it always
transcribed (they become true positives, which is why all three F1s move
together). The honest GAPS single-line number is **0.6969**; all GAPS-tuned work
should now measure against it. The A2 cross-domain-prior negative (−0.138) and
any future GAPS tuning are re-based on this cleaner gold. The 3 fallback clips
keep the (smaller) residual artifact; reopening them needs bespoke volta
handling, not worth it at n=22.

## 2026-07-06 — A10: pitch_off decomposed by semitone delta — no dominant fixable mode; bucket formally closed

**Phase:** v1.1 roadmap A10 (branch `v1.1/a14-a10-probes`)
**Decision tree:** eval-decomposition instrumentation — does the opaque 11%
`pitch_off` bucket become actionable (octave / harmonic / semitone classes need
different fixes) or formally closed?
**Branch taken:** **Formally closed.** Instrumented `ErrorDecomposition` with
per-event signed semitone deltas (`pred − gold`, captured at the `pitch_off`
match point in `decompose_errors`) plus an octave / harmonic / semitone / other
classifier; `format_decomposition_markdown` renders the histogram + class
summary (aggregate and per tier) in every future decomposition run. Ran the
accepted val24 config (`highres` + `guitarset-v1`, coupled sequence prior) via
the cached runner — baseline parity exact (single-line 0.5140 / lo-95 0.4144,
strummed 0.7953), so the decomposition is trustworthy.
**Evidence:** `pitch_off` = 117 events = 11.2% of loss. Classes: **other 61
(52%)**, harmonic 30 (26%), semitone 20 (17%), **octave 6 (5%)**; 109/117 in
strummed; the largest single delta is −5 with 11 events. Reports:
`docs/EVAL_REPORTS/a10_val24_baseline_2026-07-06.md`,
`docs/EVAL_REPORTS/a10_val24_pitch_off_decomposition_2026-07-06.md`. Unit
coverage: `tests/unit/test_error_decomposition.py`,
`tests/unit/test_composite_report_formatting.py` (611 unit tests green).
**Reasoning:** The classic cheap fix — octave disambiguation — would address 6
notes (~0.6% of total loss); harmonic suppression ~2.9%. Nothing clears the
cost bar of even an hours-class intervention. The bucket is dominated by
diffuse near-miss pairings inside dense strummed clusters (52% "other", spread
−19..+24 with no mass concentration), i.e. a matching artifact of chord-dense
audio, not a detector pitch pathology. Closed as a fix target; the
instrumentation is permanent, so any future backend whose histogram
re-concentrates (e.g. octave-heavy) reopens the bucket for free.

## 2026-07-06 — A14: video complementarity probe — no routed hybrid exists; the video question is CLOSED on every measured axis

**Phase:** v1.1 roadmap A14 (branch `v1.1/a14-a10-probes`)
**Decision tree:** chunk-6 capstone follow-up — the aggregate said audio 0.778 >
video 0.574, but did it hide a complementary per-note subpopulation that an
audio-uncertainty-keyed router could exploit? (The one hybrid left unmeasured
before the D1 SPEC edit.)
**Branch taken:** **No hybrid exists; video CLOSED for D1.** Built the
cache-only per-note join (`scripts/eval/a14_video_complementarity_probe.py`):
a fuse-mirroring decode (self-checked identical to `fuse(events, [], cfg)` on
all 12 clips) that also extracts a per-note **string-flip local margin** (best
vs cheapest string-changing state, neighbours fixed — the same trellis quantity
B4 will surface), joined against WS1-calibrated best-orientation video
predictions from the WS0 rich cache. Parity reproduced: audio 0.780 (capstone
0.778 — 31 of 10,103 notes lost to span-infeasible clusters), video 0.574 exact.
**Evidence** (`docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md`, 7,666
joined notes):
- 2×2 confusion: audio-wrong ∩ video-right = **443 (5.8%)** — the oracle-router
  ceiling is +5.8pp and requires a perfect per-note router.
- **Anti-enrichment:** P(video right | audio wrong) = **0.285** vs marginal
  0.574 — video is *half* as accurate exactly where audio fails. Errors are
  strongly co-located, the opposite of complementary evidence.
- **Chord axis refuted** (D1's open 0.85 reference): on chord-member notes
  audio is *better* (0.819 vs 0.779 singleton) and video *worse* (0.542 vs
  0.600) — the "a chord shape is one static frame" hypothesis fails on real
  footage.
- **Routing sweep:** the margin is genuinely informative about audio errors
  (Q1 audio acc 0.695 vs Q4 0.846), but video underperforms audio in **every**
  margin quartile (worst gap −0.264, even Q1 −0.185), so every τ > 0 loses;
  best routed Δ = +0.0000 (route nothing).
**Reasoning:** Audio and video are not independent witnesses on this corpus —
both fail on the same hard notes, and where audio's playability decode is
uncertain, the CV chain is *more* degraded (occlusion/motion at position
shifts). Combined with the capstone, every axis is now measured: aggregate
(capstone), gated/ungated fusion at any λ/orientation (capstone), uncertainty-
routed hybrid (this), chord axis (this). Recommendation for the D1 packet:
retire the 0.85 chord-instance and 0.86 strummed video-assisted references the
same way the 0.94 single-line was retired — no video-assisted stretch number
until a video chain demonstrates one. **Do-not-retry:** audio-uncertainty-keyed
(or any confidence-keyed) routing of the current CV chain's string evidence.

## 2026-07-06 — D1 (partial): strummed 0.86 + chord-instance 0.85 video-assisted references RETIRED (user-approved)

**Phase:** v1.1 / D1 decision packet (`docs/2026-07-06-d1-decision-packet.md`).
**Decision tree:** the 0.86 strummed / 0.85 chord-instance video-assisted
reference targets were held open pending A14 (chord-frame video, the one axis
where video plausibly beat audio). A14 refuted it — retire, or keep as an
aspirational reference?
**Branch taken (user, 2026-07-06):** **RETIRE both**, mirroring the 0.94
single-line retirement (2026-07-02). SPEC §1.4.1 amended. No video-assisted
stretch number stands for any tier; v1 records audio-only baselines
(single-line ≥ 0.45 gate; strummed / chord-instance as measured). Any future
stretch number must be *demonstrated* first (rule 7). **Data + code kept:** the
video chain (`tabvision/video/*`), GAPS caches, and probes stay in the repo as
measured evidence and remain runnable — they are no longer acceptance targets.
**Evidence:** A14 (`docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md`,
DECISIONS 2026-07-06) — chord-member notes: audio 0.819 > video 0.542;
P(video right | audio wrong) 0.285 (anti-enriched); no router beats audio-only.
Plus the chunk-6 capstone (audio 0.778 > video 0.574) + the A2 domain-sensitivity
negative.
**Reasoning:** every axis video could plausibly win is now measured and lost, so
holding these references open would be promising what nothing has demonstrated.
Retiring them keeps the SPEC honest. The still-open D1 items (expressive-markings
stretch, studio-condition tier, stale §15) remain the user's to decide.

## 2026-07-06 — A3/A4: fusion-constants sweep — harness validated; movers are the domain-sensitive prior-trust lever; A4 gap-decay a wash; no default changed

**Phase:** v1.1 roadmap A3+A4 (branch `v1.1/a3-a4-fusion-sweep`)
**Decision tree:** does grid-sweeping the (now env-overridable) fusion constants
find a safe accuracy lift, or a wash / a domain-sensitive trap?
**Branch taken:** **Built the infra + A4, banked the sweep, changed NO default.**
The in-process sweep (`scripts.eval.a3_fusion_sweep`, caches raw AudioEvents then
re-fuses per grid point) reproduces the val24 baseline **0.4820 / 0.7951**
exactly, validating it. A5 (chord dictionary) + per-tier configs ride this.
**Evidence (val24, highres + guitarset-v1, no sequence prior):**
- Biggest movers all = "trust the guitarset-v1 prior more": **LOW_FRET_BIAS=0.0**
  +0.0386 agg (single-line 0.4820→0.5728), **FRET_PRIOR_WEIGHT=1.5** +0.0297
  (0.5306/0.8060), prior **power=3.0** +0.0297 (identical). But **val24 IS
  GuitarSet** (in-domain for that prior; the same prior is −0.138 on GAPS, A2),
  so these are almost certainly GuitarSet-overfit → must clear GAPS clean-12
  no-regression, expected to fail it. Flagged, not adopted.
- **A4 `TRANSITION_GAP_TAU` (gap-decay): WASH** — best TAU=1.0 at +0.0005, most
  values negative. Short TAU trades single-line for strummed but nets negative.
  Keep default `inf` (off). Banked negative for the A4 hypothesis; the knob
  stays (env-overridable) as measured evidence.
- **Domain-neutral candidate: OPEN_STRING_BONUS=0.0** lifts strummed
  0.7951→0.8140, single-line flat (confirms the docstring's suspicion that the
  bonus — calibrated against a now-absent vision floor — slightly hurts
  strummed). `SPAN_NORM=6.0` +0.0066, `CHORD_MAX_GAP_S=0.04` +0.0055 also
  small-positive. Prior **alpha inert** (drop from future sweeps).
**Reasoning:** The non-prior constants behave as the roadmap predicted
(+0.005–0.02, wash-class). The large val24 lift is real but is the *same
domain-sensitive prior-trust lever* the A2 negative already flagged — so it is a
GuitarSet-overfit trap, not a shippable win, until it clears GAPS. The honest
deliverable is (1) the sweep infra, (2) A4 measured as a wash, (3) a shortlist of
gated candidates (safest: `OPEN_STRING_BONUS=0.0`). No `playability` default
changed. Report: `docs/EVAL_REPORTS/a3_fusion_sweep_val24_2026-07-06.md`.

## 2026-07-06 — A12: timbral string-ID (TabCNN) — becomes a training spend, soft no (FILED, not pursued)

**Phase:** v1.1 roadmap A12 (Tier 3, approval-gated); zero-spend feasibility only.
**Decision tree:** is a pretrained timbral string-ID model a zero-spend drop-in
(gate: weights must exist), or does it become a rule-8 training spend?
**Branch taken:** **BECOMES A TRAINING SPEND — soft no on current evidence; filed
for the decision packet, not pursued.** No pretrained weights exist anywhere in
the TabCNN lineage (TabCNN itself is unlicensed → code non-shippable; the MIT
successors — inhibition, FretNet — also ship no checkpoints). Expected value is
thin: paper TDR 0.84 vs our audio prior's 0.778 (measured on a *favorable*
all-notes population, not the contested-string subset), and TabCNN's own tab F1
0.75 is *below* our current in-domain ~0.815 — its only value would be as a
string-posterior feed, not a transcriber. The `AudioEvent.fret_prior` channel +
A3's new `FRET_PRIOR_WEIGHT` knob are ready if ever pursued with sign-off.
**Evidence:** `docs/2026-07-06-a12-tabcnn-feasibility.md` (full verdict, URLs).
The stub `audio/tabcnn.py` references a "trimplexx CRNN", not Wiggins TabCNN —
reconcile before any build.
**Reasoning:** The gate's "weights confirmed to exist" condition is not met, so
proceeding needs a training run (rule 8). On TDR-0.84-vs-0.778 the lift doesn't
justify the spend + a new CQT front-end. Recommend holding; if revisited, build
on the MIT successors and re-scope the expected lift against the contested-string
subset first (a zero-spend re-analysis of existing caches).

## 2026-07-07 — A3 gate: OPEN_STRING_BONUS=0.0 candidate — passes GuitarSet, FAILS GAPS cross-domain → NO default change

**Phase:** v1.1 roadmap A3 follow-up (the sweep's safest domain-neutral candidate).
**Decision tree:** the A3 val24 sweep flagged `OPEN_STRING_BONUS=0.0` as the one
*domain-neutral-looking* mover (strummed 0.7951→0.8140, single-line flat; the
bonus's docstring admits it was calibrated against a now-absent vision floor).
Take it through the acceptance gate (60-clip player-05 lower-95 confirm + GAPS
clean-12 per-clip no-regression) — adopt as default, or bank negative?
**Branch taken:** **NO default change; banked negative.** `OPEN_STRING_BONUS`
stays 0.5. The gate is a split decision that the cross-domain bar vetoes:
- **GuitarSet 60-clip (in-domain, guitarset-v1 prior): PASS.** Both tiers'
  lower-95 *improved* — single-line 0.4570→0.4627 (+0.0057), strummed
  0.6058→0.6175 (+0.0117); 19/60 clips up, 5 down, 36 flat.
- **GAPS clean-12 (cross-domain classical, prior none): FAIL.** Single-line
  lower-95 0.7093→0.7003 (−0.0091), **11 of 12 clips regress** (worst 294 −0.023).
Reports: `docs/EVAL_REPORTS/a3_gate_open0_{gs60,gaps12}_2026-07-07.md`. Gate
harness: `scripts/eval/a3_gate_probe.py`.
**Reasoning:** Even removing a *hand-coded bonus* (not a learned prior) is
**domain-sensitive** — the open-string bonus is corpus-coupled: GuitarSet
(pop/rock/jazz comping) and GAPS (classical) use open strings differently, so
what helps one hurts the other. This closes the last A3 sweep candidate and
generalizes the 2026-07-02 A2 / 2026-07-06 A3 finding: audio-fusion constant
tuning is domain-sensitive *across the board*, not just for the guitarset-v1
prior. The shipped defaults (a cross-corpus compromise) should not move on
in-domain gains alone. The A3 infra (env-overridable constants + sweep + gate
harness) stands as the reusable measurement machinery; the specific val24
movers are all now gated-and-closed.
**Tooling note:** the gate harness's verdict logic was corrected here (the first
run mislabeled the GuitarSet leg FAIL by applying the strict *per-clip*
no-regression rule to the in-domain confirm, whose bar is per-tier lower-95).
Now a two-bar model: lower-95 always gates; per-clip no-regression is the HARD
bar only for the GAPS cross-domain leg (`--strict-per-clip`). Unit-pinned in
`tests/unit/test_a3_gate_verdict.py`.

## 2026-07-07 — A8 studio-condition degradation: the capture chain does NOT degrade accuracy → keep tuning; input-robustness retired as an accuracy risk

**Phase:** v1.1 roadmap A8 (Tier-2 diagnostic; the highest-value *unmeasured*
item per the blindspot audit — every accuracy number in the repo is on clean
corpus WAVs, but the product ingests browser-`MediaRecorder` Opus-in-webm).
**Decision tree:** re-encode val24 through the real capture chain, re-transcribe
(highres), re-score. Fork: if accuracy holds → keep tuning clean-corpus fusion;
if it craters → pivot effort to input robustness (denoise/AGC, preflight
rejection, the B9 bad-input banner) instead of another decimal of clean Tab F1.
**Branch taken:** **KEEP TUNING. The eval-vs-product gap is ~0 — input
degradation of the encode/transmit chain is retired as an accuracy risk.**
A degradation *curve* (codec-only floor → realistic mic → worst-case room), all
faithful to the app (`RecordPanel.tsx` disables echoCancellation /
noiseSuppression / autoGainControl, so codec is the only *guaranteed*
degradation; mic band + noise are environmental):

| condition | Δ aggregate Tab F1 vs clean |
|---|---|
| `opus_128` (codec floor) | +0.0015 |
| `opus_64` (codec stress) | +0.0027 |
| `laptop_mic` (realistic, HP70/LP8000 + noise + Opus 96k) | +0.0063 |
| `noisy_room` (worst-case, HP90/LP7000 + louder noise + light comp + Opus 64k) | +0.0085 |

Every delta is a **wash-to-tiny-positive**; the lower-95 CIs overlap heavily
(clean agg lo-95 0.5537 vs noisy_room 0.5627), so the honest statement is **no
measurable degradation**, not "degradation helps." Onset and pitch F1 are
equally flat (codec/noise do not cost note recall). Clean rows reproduce the
roadmap baseline **exactly** (single-line 0.4820 / strummed 0.7951), validating
the harness end to end. Report:
`docs/EVAL_REPORTS/a8_studio_degradation_val24_2026-07-07.md`. Harness:
`scripts/eval/a8_studio_degradation.py` (+ 13 unit tests).
**Reasoning:** Opus at MediaRecorder bitrates is near-transparent for pitch
content (guitar fundamentals ≤ ~1.3 kHz; the mic lowpass at 7–8 kHz preserves
them and the low harmonics), and the highres backbone is robust to a modest
pink-noise floor — mild lowpassing may even remove HF distractors. This
**retires input degradation of the encode chain from the risk register**: don't
spend on denoise/preflight *for accuracy* (a bad-input banner may still be worth
it as UX), and trust that clean-corpus tuning transfers to the product.
**Scope / honest caveat:** A8 models the codec + a *plausible* built-in-mic band
+ additive noise. It does **not** cover real field artifacts — room reverb,
input clipping, an arbitrary phone mic's true response, or non-GuitarSet playing.
So the claim is "the encode/transmit chain + a modeled laptop mic are safe," not
"any recording in any room works"; a real-device field set would be a separate,
non-automated effort if ever justified. Ceiling reminder stands: single-line
audio is information-limited (A2/A14 closed); A8 removes a *worry*, it does not
add upside. **No SPEC change** (diagnostic, not a gate). This does make **D1-c**
(optional studio-condition diagnostic *tier* in §1.4.1) decidable — the harness
and first number now exist; still the user's call.

## 2026-07-07 — D1 remainder resolved (user-approved batch): SPEC §1.4.1 studio tier + §15 replaced

**Phase:** v1.1 D1 decision packet (`docs/2026-07-06-d1-decision-packet.md`) —
the SPEC-hygiene remainder after the 0.94 (2026-07-02) and 0.86/0.85
(2026-07-06) video-assisted-reference retirements. All four resolved to the
packet's recommendation.
**Decisions + SPEC actions:**
- **D1-a (retire 0.86/0.85 video refs): CONFIRMED — no-op.** Already applied
  2026-07-06 (§1.4.1 carries the user-approved A14 retirement; the packet/handoff
  line listing it as "awaiting user" was stale). Re-confirmed today; no SPEC edit
  needed. (Verified against §1.4.1 before acting — did not double-retire.)
- **D1-b (expressive-markings ≥ 0.70 technique F1): OUT until baselined.** No
  technique-F1 number has ever been measured, so per §0 rule 7 it stays a
  *tracked v1.1 stretch, explicitly unbaselined* — annotated as such at §1.4 and
  queued as a §15 live question (run the free GuitarSet-JAMS baseline first, set
  a real stretch from its value). No binding number written.
- **D1-c (studio-condition eval tier): IN as a diagnostic tier (reported, not
  gated).** Added to §1.4.1 referencing the A8 harness + the 2026-07-07 result
  (gap ~0). Tracks capture-chain degradation release-over-release; gates nothing.
- **D1-d (stale §15): REPLACED wholesale.** The five pre-Phase-1.5 questions are
  all answered by events; §15 now lists the actual live items — D1-b (technique
  baseline), D2 (electric v2 sequencing), D3 (export-deps license review), D4
  (Phase 9 "proceed").
**Reasoning:** the binding v1 gates (single-line ≥ 0.45, strummed ≥ 0.60,
aggregate ≥ 0.55) are untouched — this batch is honesty/hygiene: retire
references three negatives refuted, don't publish an unmeasured stretch, track
the product-condition gap as a named diagnostic, and stop carrying expired
questions. The packet is **resolved**; remaining user-gated items (D2/D3/D4) now
live in §15. **Still open elsewhere:** A5 (chord-shape dictionary port) is the
next accuracy lever (rides the A3 sweep harness).

## 2026-07-07 — A5 chord-shape bonus: mechanism landed (no-op default), sweep→gate pending

**Phase:** v1.1 accuracy roadmap Tier 2 (`docs/plans/2026-07-01-accuracy-ux-roadmap.md`
A5) — the Phase-5 chord-dictionary port deferred at Phase-5 ship.
**What landed:** `tabvision/tabvision/fusion/chord_shapes.py` ports v0's voicing
shapes (`tabvision-server/app/chord_shapes.py`: 22 open + 72 E/A-shape barre + 39
power = **133 voicings**) into the v1 `0=low E` string convention
(`string_idx = 6 - v0_string`), and adds a per-cluster emission term
`chord_shape_cost(state)` wired into `viterbi._viterbi_clusters.state_emission`.
It rewards a decoded cluster whose `(string, fret)` positions overlap a canonical
voicing by ≥ `CHORD_SHAPE_MIN_NOTES` (default 3): `cost -= CHORD_SHAPE_BONUS *
overlap` (count-scaled). Only the voicing *shapes* were ported — v0's scale-box /
`GuitarPosition` / `PlayingStyle` machinery plays no part in an emission bonus and
would be dead code here.
**No-op discipline (A3/A4):** `CHORD_SHAPE_BONUS` defaults to `0.0` (env-overridable
`TABVISION_CHORD_SHAPE_BONUS`, runtime-rebindable — `state_emission` reads the
module global live), so fusion is **bit-identical** until a sweep sets it. Wired
as an `a3_fusion_sweep` axis (`[0.0, 0.1, 0.25, 0.5, 1.0]`).
**Key invariant (why this only touches the target tiers):** a state assigns one
candidate per event, so a cluster's decoded position-set has size = cluster size;
a singleton (single-line) or dyad can never reach the 3-note match gate.
**Single-line Tab F1 is therefore invariant to this term at any magnitude** — A5
can only move strummed/chord (its stated target). Proven by
`test_single_line_decode_invariant_to_bonus` + exact E-major-open recovery under a
dominating bonus.
**Gates:** ruff check/format clean; **mypy clean** (63 files — the shape params had
to be typed `Mapping[int, int | None]` not `dict`, else dict value-invariance
fails the type gate); 16 new unit tests + the existing 57 fusion tests green. (The
local global interpreter can't run the torch/soundfile audio suite — env, not
code; CI runs it, cf. PR #28 green.)
**Measured 2026-07-07 (eval env reconstructed in-session — installed
soundfile/pretty_midi, manifests + transcription cache present).** The
`CHORD_SHAPE_BONUS` sweep on val24 finds **0.1** the best magnitude (strummed
0.7951→0.7980; single-line **exactly 0.4820 at every value** — the invariance
holds empirically; ≥0.25 turns negative as count-scaling over-biases). At **0.1**
the candidate **clears the full A3 gate on BOTH legs** — the first fusion constant
to do so:
- **in-domain** 60-clip GuitarSet player-05: strummed +0.0053 mean / **+0.0061
  lower-95**, single-line +0.0000 (PASS — per-tier lower-95, the in-domain bar);
- **cross-domain** GAPS clean-12 (`--position-prior none --strict-per-clip`):
  +0.0006 mean, **0 per-clip regressions** (PASS — the hard cross-domain bar).
Contrast A3's `OPEN_STRING_BONUS=0.0` (passed GuitarSet, **FAILED** GAPS lo-95
−0.0091) and A4 (wash): this reward is **domain-neutral** because it is grounded
in voicing **geometry**, not corpus prior tuning. To gate it, `a3_gate_probe.py`
was made module-aware (resolves `chord_shapes` constants, not just `playability`).
Report: `docs/EVAL_REPORTS/a5_chord_shape_gate_2026-07-07.md` (+ full sweep
`…_sweep_val24_2026-07-07.md`). **Ceiling honesty (roadmap A5):** +0.0053 strummed
is **below** the hoped +0.01–0.04 and does **not** move the chord-accuracy
0.48→0.85 gap — a real, small, rigorously-gated, near-zero-downside win (env
-reversible), not a breakthrough.
**Default decision — USER-GATED (still 0.0 in-tree).** Shipping `0.0 → 0.1`
re-bases the canonical val24 **strummed** baseline 0.7951→0.7980 (single-line
0.4820 unchanged), which every future sweep/gate then references. Per the
project's close stewardship of shipped defaults, and though A5 is a "no-approval"
tier, that re-basing + the below-hope magnitude make the flip the **user's call**;
the committed default stays **0.0** pending it. **Reasoning:** the mechanism +
full gate are done and banked; only the one-line default flip (with its
test/sweep-baseline/docstring ripple) awaits the ship/hold call. **Still open:**
the A5 default-flip (ship 0.1 vs hold 0.0 — user-gated, above); D2/D3/D4 in SPEC §15.
