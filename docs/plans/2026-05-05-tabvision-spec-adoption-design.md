# TabVision Spec Adoption — Design

**Date:** 2026-05-05
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — pending Phase 0 sign-off
**Spec source:** `TAB_SPEC_UPDATE.md` (renamed → `SPEC.md` as part of Phase 0)

## 1. Context

`TAB_SPEC_UPDATE.md` is a v1.0 canonical spec for a guitar-tab transcription
pipeline. It is a substantial reframing of the existing TabVision project, not
an incremental change:

- **Architecture:** Electron + Flask client/server → Python CLI tool.
- **Module boundaries:** strict layering with immutable §8 contracts
  (`tabvision.{demux,audio,video.{guitar,fretboard,hand},preflight,fusion,render}`).
- **Scope expansion:** distorted electric + classical, `--style`
  (strumming/fingerstyle/mixed), `--capo 0–7`, preflight tool with framing
  feedback, four export formats (ASCII / GP5 / MusicXML / MIDI).
- **Audio strategy:** Phase 1 Basic Pitch baseline → Phase 2 Riley/Edwards
  High-Res swap → Phase 7 augmentation + self-labeling. Fine-tuning becomes
  augmentation-driven, not the headline strategy.
- **Strict 10-phase gating** with acceptance tests, decision trees, and a
  pre-code Phase 0 audit (`AUDIT.md`).
- **Eval reset:** four difficulty tiers, Tab F1 ≥ 0.88 aggregate.

The current project has working code (91.6% Exact F1 on the 11-video set) and
an in-flight Phase 1 audio finetune on `feature/audio-finetune-phase1`. This
doc maps the spec onto that reality: what to keep, what to port, what to
build, and in what order.

## 2. Locked decisions

| # | Decision | Choice |
|---|---|---|
| Q1 | Adoption posture | **Hybrid** — adopt spec, scaffold per Phase 0, fast-forward through phases the existing code already satisfies. |
| Q2 | Location of new package | **Parallel directory** at repo root. New `tabvision/` package alongside existing `tabvision-server/` + `tabvision-client/`. |
| Q3 | In-flight finetune work | **Finish-then-pivot.** Run H2 (label-encoding) experiment to conclusion on `feature/audio-finetune-phase1`, document the outcome, freeze the branch, then start spec adoption on `refactor/v1`. |
| Q4 | Fate of `tabvision-client/` (Electron) | **Freeze in place.** Keep desktop app + Flask server functional as a v0 demo. New CLI is the spec-compliant artifact; v1 acceptance is measured against the CLI only. Repurposing Electron as a thin CLI consumer is the likely Phase-9-or-v1.1 follow-up. |
| Naming | Project / package name | Keep `tabvision`. Spec global-edit `tabify` → `tabvision`. CLI command stays `tabvision transcribe ...`. |

## 3. Phase mapping

**Fast-forward criterion.** A phase is **fast-forwarded** if its existing
implementation (a) functionally produces the deliverable, (b) can be wrapped
to match the §8 contract without rewrites, and (c) has at least one passing
test. Otherwise: **Port** (move + reshape) or **Build** (new work).

| Phase | Spec deliverable | Status | Action |
|---|---|---|---|
| **0** | Audit + scaffold + LICENSES.md | none | **Build** — first thing we do |
| **1** | Basic Pitch end-to-end → ASCII | `audio_pipeline.py` works | **Port** — wrap behind `tabvision.audio.basicpitch` + `AudioBackend` protocol |
| **1.5** | Annotation tool + 15+ clips / 4 tiers | partial; **revised below** | **Build** annotator; **curate** existing + public datasets — no new self-recording |
| **2** | Riley/Edwards SOTA swap | none | **Build** — net-new |
| **3** | Guitar detect + preflight + fretboard | partial; fretboard exists, no YOLO, no preflight | **Build** YOLO + preflight; **Port** existing fretboard as fallback |
| **4** | MediaPipe finger posteriors | MediaPipe wired | **Port** — wrap as `tabvision.video.hand` |
| **5** | Viterbi fusion + chord-aware | fusion engine has chord/melodic logic | **Port** — partial rewrite of `fusion_engine.py` to fit spec's `playability.py` + `viterbi.py` split |
| **6** | ASCII + GP5 + MusicXML + MIDI | ASCII only | **Build** — 3 new exporters |
| **7** | Augmentation + self-labeling | finetune branch in flight | **Fold** finetune outcome into Phase 7; **Build** augmentation + self-labeling on top |
| **8** | Deterministic eval harness, CI smoke | harness exists, not deterministic/CI-bound | **Port + harden** |
| **9** | Polish, demo, license verify | none | **Build** |

**Net:** Port × 3 (1, 4, 5), Partial × 2 (3, 8), Build × 5 (0, 1.5, 2, 6, 9),
Fold-in × 1 (7).

## 4. Repo layout

```
tab_vision/                         (repo root, name unchanged)
├── tabvision/                      ← NEW: spec-compliant package + CLI
│   ├── pyproject.toml
│   ├── tabvision/
│   │   ├── __init__.py
│   │   ├── types.py                (§8 dataclasses)
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── demux/
│   │   ├── audio/
│   │   ├── video/{guitar,fretboard,hand}/
│   │   ├── preflight/
│   │   ├── fusion/
│   │   ├── render/
│   │   └── cli.py
│   ├── tests/{unit,integration,eval}/
│   ├── scripts/{acquire,train,eval,augment,annotate}/
│   └── data/{fixtures,eval}/
├── tabvision-server/               ← FROZEN v0 backend (Flask). No further dev.
├── tabvision-client/               ← FROZEN v0 desktop UI (Electron). No further dev.
├── docs/
│   ├── plans/                      (existing convention; this doc lives here)
│   ├── HISTORY.md                  (context from v0)
│   ├── DECISIONS.md                (decision-tree branches taken)
│   ├── EVAL_REPORTS/               (per-phase eval reports)
│   ├── DEMO/                       (Phase 9 portfolio assets)
│   └── NARRATIVE.md                (project story for portfolio)
├── AUDIT.md                        ← Phase 0 deliverable
├── LICENSES.md                     ← Phase 0 deliverable
├── SPEC.md                         (renamed from TAB_SPEC_UPDATE.md)
└── CLAUDE.md                       (updated to reference new layout)
```

**Key choices:**

- **No `legacy/` directory.** Spec's `legacy/` semantically means "delete at
  Phase 9," but per Q4 the desktop app stays alive into v1.1. So
  `tabvision-server/` + `tabvision-client/` keep their current paths, just
  frozen.
- **Package nesting** `tabvision/tabvision/` is the standard Python pattern
  (outer dir = project, inner dir = importable package). Keeps `pyproject.toml`
  separate from existing `tabvision-server/requirements.txt`.
- **Two `pyproject.toml`s.** New package has its own; existing
  `tabvision-server/` `requirements.txt` is untouched.

## 5. Sequencing & branch strategy

**Branches:**

- `main` — production-ish baseline (unchanged).
- `feature/audio-finetune-phase1` (active) — H2 runs to conclusion, then
  **frozen**. Whatever checkpoint (if any) lands the gate gets folded into
  Phase 7 later.
- `refactor/v1` (new) — branched off `main`. Single long-running branch for
  spec adoption. Squash-merged to `main` only when a phase passes its
  acceptance gate. Per-phase sub-branches optional.

**Sequence:**

1. **Now → H2 verdict (~1 week).** Finish H2. Document outcome.
2. **H2 verdict → Phase 0 done (~1 week).** Cut `refactor/v1` off `main`.
   Run §2.1 user interview → `AUDIT.md`. License map → `LICENSES.md`.
   Scaffold `tabvision/tabvision/` skeleton. CI green on trivial test.
   **Gate:** user signs off on AUDIT + LICENSES.
3. **Phase 0 → Phase 1 port (~1 week).** Wrap audio pipeline behind
   `tabvision.audio.basicpitch`. Stub `tabvision.video`. Port
   `app/fusion_engine.py` audio-only path. Port ASCII export. CLI runs
   end-to-end. Eval harness reports first numbers. **Gate:** pipeline runs.
4. **Phase 1.5 (~1 week).** Curate existing + public datasets per Section 6.
   No new recording.
5. **Phases 2 → 9 sequentially**, each gated by §9.3 acceptance.
   No fixed dates past Phase 0.

**Cadence:** one phase merged to `main` at a time, design-doc-then-execute
via the existing `docs/plans/` convention. Per spec §10, cut scope, not
quality, if time pressure hits.

## 6. Phase 1.5 — revised eval set strategy

**Constraint:** no new self-recording. Use existing public datasets +
already-recorded historical clips.

| Tier (§1.4) | Source | Notes |
|---|---|---|
| Clean acoustic single-line | GuitarSet held-out player(s) | Already JAMS-annotated, hexaphonic ground truth |
| Clean acoustic strummed | GuitarSet "comp" excerpts | Same source, comp/chord-mode subset |
| Clean electric | IDMT-SMT-Guitar held-out | Registration needed; mostly monophonic |
| Distorted electric | EGDB held-out | Multi-amp, ground-truth tab |
| **Bonus tier: iPhone OOD** | Existing 11/20 self-recorded videos | Already-annotated; **not a v1 acceptance gate**, but a tracked metric so we know iPhone-domain regressions when they happen |

**Phase 1.5 deliverables (revised):**

1. `tabvision/scripts/annotate/cli.py` — annotator. Useful for converting +
   validating existing annotations and for any future ad-hoc annotation. **No
   new recordings required.**
2. **Manifest** `tabvision/data/eval/manifest.toml` listing each clip,
   source, tier, ground-truth path, train/val/test split.
3. **Conversion scripts** `tabvision/scripts/eval/convert_*.py` for any
   non-JAMS source → JAMS.
4. **License + redistribution audit** for each public dataset.

**Phase 1.5 acceptance gate (revised):** manifest exists, all tiers
represented, `pytest -m eval` runs to completion on the manifest, baseline
metrics committed.

**Acknowledged blind spot:** distorted-electric §1.4 target is measured on
EGDB (studio domain), not iPhone-recorded distortion. Documented here so
Phase 7's tier-target claim is honest.

## 7. Phase 0 audit content

Phase 0 gate = `AUDIT.md` + `LICENSES.md` + scaffold + green CI. Most audit
content is discoverable from existing artifacts (`CLAUDE.md`, memory, git
log, `docs/plans/`), so the actual interactive interview is short.

**§2.1 Step-0 interview answers (pre-filled):**

| Q | Pre-filled answer | Confidence |
|---|---|---|
| 1. Where is the project? | `/home/gilhooleyp/projects/tab_vision`, single repo, contains backend + Electron client | High |
| 2. State? | Backend works end-to-end; 91.6% Exact F1 on 11-video set; mid-experiment on Phase 1 audio finetune (H2 untested) | High |
| 3. Prior approach? | Audio (Basic Pitch) + vision (MediaPipe + geometric fretboard) + heuristic fusion. Worked: melodic segment correction, string compactness, 2-pass anchoring. Didn't ship: video hand anchor, learned-fusion LightGBM (NO_SHIP), Phase 0 RMS truncation (no-op on iPhone audio) | High |
| 4. Datasets/weights/annotations to preserve? | 11 + 20 self-recorded videos + ground-truth, GuitarSet TFRecords at `tabvision-server/tools/outputs/tfrecords/guitarset/splits/`, fine-tuned checkpoints (TBD H2 outcome), benchmark results | High |
| 5. Branches with abandoned approaches worth revisiting? | `agent-farm-improvements` (learned-fusion), `feature/audio-finetune` (Phase 0 RMS) — both `NO_SHIP` | High |
| 6. Don't touch? | `tabvision-client/` and `tabvision-server/` per Q4 — frozen, not deleted | High |
| 7. Python version + tooling? | **NEEDS VERIFICATION** during Phase 0 — read existing `requirements.txt` + venv | Medium |

**`AUDIT.md` structure:**

1. **User interview** — table above.
2. **Inventory** — `find tabvision-server/app -type f` annotated, plus
   `tabvision-client` tree.
3. **What works** — pipeline runs end-to-end; cite test files + benchmark
   JSON + memory metrics.
4. **What's broken** — known: video-4 audio detection issue, fundamental
   arpeggio position problem, no distorted-electric capability.
5. **What's unknown** — capo support paths, untested code (run coverage to
   find).
6. **Reusable artifacts** — `audio_pipeline.py`, `fusion_engine.py` logic,
   eval scripts, GuitarSet TFRecords, ground-truth annotations, plan docs.
7. **Baseline metrics** — copy 11-video table from memory + run one fresh
   end-to-end clip through current pipeline to verify reproducibility.

**`LICENSES.md` initial map:**

| Component | License | Default-pipeline OK? |
|---|---|---|
| Basic Pitch | Apache 2.0 | ✅ |
| MediaPipe Hands | Apache 2.0 | ✅ |
| OpenCV | Apache 2.0 | ✅ |
| ffmpeg-python | Apache 2.0 | ✅ |
| Riley/Edwards High-Res | research weights — verify | ⚠️ check before Phase 2 swap |
| trimplexx CRNN | verify | ⚠️ |
| YOLOv8n (ultralytics) | AGPL — caution for portfolio | ⚠️ pin to weights-only path or pick alternative |
| GuitarSet | CC-BY-4.0 | ✅ for training; no redistribution issue |
| IDMT-SMT-Guitar | research-use, registration | ⚠️ training only, not redistributed |
| EGDB | verify | ⚠️ |

Two-flag risks for portfolio: Riley/Edwards research license (Phase 2) and
YOLOv8 / ultralytics AGPL (Phase 3). Resolve before each phase commits.

## 8. Phase 7 integration — folding in finetune work

Spec Phase 7 is broader than what `feature/audio-finetune-phase1` covers.
Phase 7 wants:

- (a) audio augmentation (DadaGP → SFZ soundfonts → parametric EQ + IR +
  noise + codec re-encoding),
- (b) video augmentation,
- (c) fine-tuning recipes for audio + hand,
- (d) self-labeling loop,
- (e) ablation report.

Current finetune work covers a slice of (c) — Basic Pitch fine-tune on
GuitarSet. **No augmentation, no self-labeling.**

**Two scenarios for H2 fold-in (resolved at Phase 7 entry, not now):**

**Scenario X — Phase 2 (Riley/Edwards swap) hits its ≥5pp Pitch F1 lift over
Basic Pitch.** The Path-2 fine-tune of Basic Pitch becomes mostly historical:
Riley/Edwards is the new audio backbone, and Phase 7's fine-tuning recipe
targets *that* model. The H2 result and TFRecord pipeline still inform the
recipe (data loaders, augmentation hooks reusable), but the H2 checkpoint
itself is shelved.

**Scenario Y — Phase 2 underwhelms or Riley/Edwards licensing blocks
default-pipeline use.** Falls back to Basic Pitch + fine-tune as the audio
backbone. H2's outcome becomes load-bearing: positive ⇒ ship the H2
checkpoint as `tabvision/models/basic_pitch_guitar.pth`; negative ⇒ Phase 7's
augmentation work has to lift Basic Pitch from what H2 couldn't.

**Preserve unconditionally:**

- GuitarSet TFRecord splits at `tabvision-server/tools/outputs/tfrecords/guitarset/splits/`.
- `tabvision-server/app/training/load_pretrained.py` (verified equivalent
  2026-04-29).
- `tabvision-server/tools/finetune_basic_pitch_smoke.py` — working
  training-loop reference.
- `EVAL_REPORTS` from H2 → feeds Phase 7's ablation report.

**Conditional on Scenario Y:**

- The H2 checkpoint itself.
- The decision to make Basic Pitch the v1 audio backbone vs. Riley/Edwards.

**Spec deviation noted:** Phase 0 RMS truncation logic (memory:
`project_phase0_rms_activity_end.md`) is a no-op on iPhone audio. Spec Phase
7's "drop low-RMS extras" idea would need the different mechanism the
original Path-2 plan called out (Basic Pitch's last-pitched-note timestamp).

**Phase 7 acceptance gate** stays as written in the spec — all §1.4
aggregate + per-tier targets met. The fine-tune is one input, not a
deliverable.

## 9. Risks specific to this hybrid

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Port drift.** Wrapping existing modules to fit §8 contracts may surface refactors that turn "Port" into "Build." Phase 1 timeline could blow up. | Treat §8 contracts as the gate. If porting > 1 week per module, downgrade to "Build" and note in `DECISIONS.md`. |
| 2 | **License blocker at Phase 2 / 3.** Riley/Edwards research weights or ultralytics AGPL could block default-pipeline use. | LICENSES.md flags both `⚠️` at Phase 0. Resolve before each phase commits. Have a fallback (GAPS for audio, alternative detector for guitar). |
| 3 | **Eval tier coverage gap.** Existing 11/20 videos are clean-acoustic-leaning. Distorted-electric tier exists only on EGDB (studio domain). | Document explicitly: distorted-electric §1.4 target measured on EGDB held-out, not iPhone. iPhone-domain distortion is acknowledged blind spot. |
| 4 | **Frozen rot.** `tabvision-server/` + `tabvision-client/` get stale (deps drift, Python version moves). Demo silently breaks. | Phase-0 deliverable: capture current dep versions in `tabvision-server/FROZEN.md` + `make demo` smoke test that runs the v0 pipeline once on a fixture. CI runs nightly. |
| 5 | **H2 outcome drags.** "Finish-then-pivot" assumes ~1 week. If it bleeds, Phase 0 starts late. | Hard timebox: by **2026-05-12**, document H2 status. If inconclusive, freeze branch with current findings and proceed to Phase 0 anyway. |
| 6 | **Spec contract bit-rot.** §8 dataclasses get stale as implementation needs surface. | Per spec §0.3: signatures are immutable within a phase; changes require explicit user approval + spec update. |

## 10. Immediate next actions

1. **Now.** Commit this design doc to `feature/audio-finetune-phase1`.
2. **Today–next week.** Run H2 (label-encoding hypothesis) experiment to
   conclusion. Hard deadline: **2026-05-12**.
3. **2026-05-12 (or H2 verdict, whichever first).** Write
   `docs/EVAL_REPORTS/audio-finetune-phase1-final.md`. Freeze branch.
4. **Same day.** Cut `refactor/v1` off `main`. Update `CLAUDE.md` to
   reference the spec adoption doc + new layout.
5. **Phase 0 work** (≤ 1 week): write `AUDIT.md` (start from §7
   pre-fills), `LICENSES.md` (start from §7 table), rename
   `TAB_SPEC_UPDATE.md` → `SPEC.md` with `tabify` → `tabvision` global edit,
   scaffold `tabvision/tabvision/` skeleton + `pyproject.toml` + CI.
   **Gate:** user signs off on AUDIT + LICENSES.
6. **Phase 1 port** begins. New design doc when we get there if scope
   warrants it.

## 11. Open questions (deferred)

- **Phase 2 audio backbone choice** — Riley/Edwards vs. GAPS vs. trimplexx
  tabcnn. Phase 2 will A/B them.
- **Annotator UX** — CLI vs. simple GUI (spec §15 Q3). Defer to Phase 1.5
  entry.
- **Preflight strictness default** — lenient vs. strict (spec §15 Q4).
  Defer to Phase 3 entry.
- **Repo-root `vercel.json`** — flag in `AUDIT.md`; verify whether it
  signals an aborted/incomplete deploy plan.
- **Repurposing Electron as CLI consumer** (Q4 follow-up) — defer to
  Phase 9 / v1.1.

## 12. Cross-references

- `TAB_SPEC_UPDATE.md` — canonical spec; renamed to `SPEC.md` in Phase 0.
- `docs/plans/2026-04-24-audio-backbone-finetune-design.md` — Path 2
  (Basic Pitch fine-tune) plan. Outcome feeds Phase 7 per §8 above.
- `docs/plans/2026-04-24-learned-fusion-design.md` — shelved; revisit after
  spec Phase 5 fusion gate.
- `docs/plans/2026-04-23-training-video-pipeline-refinement-design.md` —
  context for Phase 3 fretboard work.
- `docs/plans/2026-04-23-video-hand-anchor-design.md` — context for
  Phase 4 hand work; `use_video_hand_anchor` defaulted off (memory:
  `project_video_hand_anchor.md`).
- `CLAUDE.md` — to be updated post-Phase-0 with new module references.

---

*This doc is the design output of a brainstorming session on 2026-05-05.
Proceed to Phase 0 only after sign-off on AUDIT + LICENSES.*
