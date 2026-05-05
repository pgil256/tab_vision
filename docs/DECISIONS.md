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
