# Tab F1 v1 acceptance — Strategy & Decision Record

**Date:** 2026-05-12 (revised 2026-05-13 per PR #10 review)
**Author:** Patrick (brainstormed with Claude)
**Status:** v3 — strategy / decision-record only; **not** an implementation plan
**Scope note:** This is a **SPEC §1.4 amendment proposal** plus
              strategy. Implementation detail lives in companion docs.
**Companions:**
- `SPEC.md` §1.4.1 (the amendment table; committed in the same change set)
- `docs/plans/2026-05-13-tab-f1-phase-0-implementation.md` (Phase 0 impl)
- Later phase impl plans (write after Phase 0 evidence)
**Replaces:** v1 + v2 (2026-05-12 single-aggregate-target drafts; both
              had load-bearing license errors and stale path references
              and have been superseded by this rewrite).

## 0. License gate (must clear before any compute spend)

Per SPEC §1.5 the **shipping default pipeline** must be portfolio-clean.
NC-licensed material is acceptable in research/experiment configurations
that are NOT shipped. Each resource is verified 2026-05-13:

| Resource | License | Portfolio-default usable? | Source / verification |
|---|---|---|---|
| GuitarSet | CC-BY-4.0 | **yes** | https://zenodo.org/records/3371780 |
| Guitar-TECHS | CC-BY-4.0 | **yes** | arXiv:2501.03720 §4 distribution |
| EGDB | none on repo — **author email pending** | **gated** | https://ss12f32v.github.io/Guitar-Transcription/ (LICENSES.md ⚠️) |
| GOAT | request-only, research-only | **no — DROPPED** | arXiv:2509.22655 §4.2 *"made available by request to better control its use for research purposes only"* |
| SynthTab dataset | **CC-BY-NC-4.0** | **no — DROPPED** | github.com/yongyizang/SynthTab README *"SynthTab is released with CC BY-NC 4.0 license"* |
| SynthTab rendering code | CC-BY-4.0 | n/a (we're not redistributing the code) | repo `LICENSE` file |
| DadaGP | access-by-email research-only; underlying GP tabs derive from copyrighted songs | **research/dev only** — NOT in default path | github.com/dada-bots/dadaGP README; underlying tab copyright unsettled |
| Basic Pitch | Apache-2.0 | yes (Phase 1 pitch ensemble) | github.com/spotify/basic-pitch |
| highres (xavriley) | MIT | yes — current production audio backend | github.com/xavriley/hf_midi_transcription |
| MediaPipe Hands | Apache-2.0 | yes — video pipeline | per LICENSES.md |
| YOLO-OBB (ultralytics) | AGPL-3.0 (accepted per DECISIONS.md) | yes (portfolio is AGPL-OK) | per LICENSES.md |
| Free amp/cab IRs | varies (most free-public) | yes for default if redistribution terms allow; verify per-pack | Modern Music Solutions Declassified, Djammincabs |

**Drops vs v2 plan:**
- **SynthTab dropped** because the dataset is CC-BY-NC-4.0; pretraining
  the shipping audio backend on it taints derived weights (SynthTab paper
  treats trained models as derivative work). Distillation as a laundering
  step is rejected — both legally murky and explicitly out of bounds
  per the 2026-05-13 review.
- **GOAT dropped** because it's request-only research-only. Cannot
  evaluate a public portfolio against it.

**Hard rule:** any phase that depends on a "gated" or "no" row must
produce evidence that the gate cleared (e.g., a written reply from the
EGDB author) BEFORE that phase ships. No conditional commits, no
"we'll-figure-it-out-later" merges.

## 1. Decisions

These supersede the v2 D1–D10 set. Append to `docs/DECISIONS.md` per
SPEC §0.5 once the plan is approved.

| # | Decision | Rationale |
|---|---|---|
| D1 | Tab F1 evaluated **per tier**, not as a single aggregate. SPEC §1.4 aggregate 0.88 is retired. | Aggregate hides the real failure mode (string/fret assignment on solo lines). |
| D2 | Per-tier v1 acceptance targets: **0.85 / 0.90 / 0.87 / 0.80** for clean acoustic single-line / strummed / clean electric / distorted electric. | User-stated floor (0.80) and strummed (≥ 0.90); middle tiers proposed and accepted. Original SPEC numbers (0.94 / 0.86 / 0.90 / 0.82) become the v1.1 / portfolio stretch reference. |
| D3 | Eval set is a **multi-source public-corpus composite**: GuitarSet + Guitar-TECHS + EGDB (license-pending) + qualifying synthetic. Personal videos banned. GOAT dropped. SynthTab dropped from default path. | Per-tier evaluation requires per-tier sources; portfolio constraint excludes NC and request-only data from the shipping path. |
| D4 | **No SynthTab in the default pipeline.** Audio-side lift comes from priors + cheap pitch post-processing + GuitarSet fine-tune. DadaGP-derived synthetic remains acceptable for **internal training/dev only** if it's never shipped. | SynthTab CC-BY-NC-4.0 taints derived weights; SPEC §1.5 bars NC from default. |
| D5 | **No quantitative video-gate.** Video pipeline ships as a qualitative feature; per-tier Tab F1 measured audio-only. | No public dataset has video + per-note string/fret labels (verified 2026-05-12). |
| D6 | **Free-tier compute first.** Order per CLAUDE.md operating rule 6 and SPEC §6.3: **Local CPU > Colab > Kaggle > Lightning Studios > Modal**. Modal is the last resort. | Project rule, plus Lightning's 22 GPU-hr/month free tier covers any fine-tune we'd plausibly run. |
| D7 | **1-2 month cadence.** No fixed deadline. | User-stated. |
| D8 | Stretch goals (bends / slides / hammer-ons / pull-offs) **out of scope** for v1. | SPEC §1.4 already marks them v1.1. |
| D9 | Top-K acceptable as an editor UX feature; the D2 numbers are on **top-1 only**. | User-stated. |
| D10 | Personal training clips off the table entirely — not as accuracy gate, not as dev set, not as label source. | User-stated. |
| D11 | This document is a **SPEC §1.4 amendment**, not a SPEC-achievement plan. Land the SPEC.md update (§1.4.1) in the same change set. | Honest framing of relaxed targets; reviewer's approval bar. |

## 2. Goal & framing

**v1 acceptance:** hit the D2 per-tier Tab F1 targets on the D3
public-corpus composite eval set within 1-2 months on free-tier
compute, with the existing v1 pipeline (no §8 contract changes).

**Stretch / portfolio reference:** the original SPEC §1.4 numbers
(0.94 / 0.86 / 0.90 / 0.82). If we hit them, that's the portfolio
narrative; v1 acceptance does not require them.

**Out of v1 acceptance:** quantitative video-fusion Tab F1
improvement claim (no public dataset for it; tracked as qualitative
only).

## 3. Current evidence

GuitarSet validation, 60 tracks, 8715 gold notes, 2026-05-08
production candidate (highres + `guitarset-v1` prior, audio-only):

| Metric | Current | Status |
|---|---:|---|
| Onset F1 (50 ms) | 0.9218 | passes SPEC §1.4 ≥ 0.92 |
| Pitch F1 (50 ms) | 0.9022 | passes SPEC §1.4 ≥ 0.90 |
| Tab F1 aggregate (retired) | 0.6104 | — |
| Tab F1, comp subset | 0.670 mean | — |
| Tab F1, solo subset | 0.508 mean | — |

The 27 pp gap to the **retired** 0.88 aggregate target is almost
entirely string/fret assignment on single-line passages. Audio is at
spec; only fusion-side assignment is short. This frames the per-tier
work: **strummed (chord context) is closest to its target; single-line
needs the most lift.**

**Coverage gap:** GuitarSet covers only the clean acoustic tiers.
Clean-electric and distorted-electric have **no current measurement**
on a public corpus and must be acquired in Phase 0.

## 4. Resource inventory

### 4.1 Datasets (default-pipeline path only)

| Source | License | Modality | Labels | Tier coverage |
|---|---|---|---|---|
| GuitarSet (on-disk) | CC-BY-4.0 | audio (hex + DI) | JAMS (string + fret + pitch) | clean acoustic single-line, strummed |
| Guitar-TECHS (acquire) | CC-BY-4.0 | audio (multi-mic + DI) | 6-track per-string MIDI | clean acoustic single-line, clean electric |
| EGDB (acquire, license pending) | none on repo — author email required | audio (DI + 5 amp sims) | GuitarPro tabs + aligned MIDI | clean electric, distorted electric |
| Free IR-augmented GuitarSet | CC-BY-4.0 (with IR pack licenses verified) | derived audio | inherited string + fret | distorted electric (fallback if EGDB blocks) |

### 4.2 Datasets (research / dev only — NEVER in the default pipeline)

| Source | License | Use |
|---|---|---|
| DadaGP | access-by-email, research-only | possible internal-training augmentation; not shipped, not redistributed |
| SynthTab | CC-BY-NC-4.0 | reference only; not a substrate for any shipped weight |

### 4.3 Compute accounts (free-tier first, per D6 order)

| Account | Free allowance | Use |
|---|---|---|
| Local CPU | 6 cores WSL2 | eval runs, prior training, cheap post-processing experiments |
| Colab | ~12 hr/day with limits | quick experiments, prior sweeps |
| Kaggle | ~30 GPU-hr/week T4 | longer sweeps, baseline checks |
| Lightning Studios | 22 GPU-hr/month | any fine-tune work, batched in one monthly window |
| W&B | unlimited (academic) | experiment tracking — required before any GPU job |
| Hugging Face Hub | unlimited public | weight / checkpoint hosting |
| Modal | pay-per-use | **production smoke retests only**; never default training |

### 4.4 Code already on `main`

- `tabvision.audio.*` — production pitch backends (highres, basicpitch).
- `tabvision.fusion.{viterbi,chord,playability,position_prior,neck_prior,melodic_prior}` — Phase 5 shipped.
- `tabvision.video.{guitar,fretboard,hand}` — Phase 4 shipped.
- `tabvision.pipeline.run_pipeline` — production-facing orchestrator.
- `tabvision.eval.{manifest,metrics,runner,guitarset_audio}` — eval scaffolding with `REQUIRED_TIERS = ("clean_acoustic_single_line", "clean_acoustic_strummed", "clean_electric", "distorted_electric")` already encoded ([tabvision/tabvision/eval/manifest.py](tabvision/tabvision/eval/manifest.py)).
- `tabvision-server/{modal_app.py, app/v1_adapter.py}` — Modal production adapter.

### 4.5 What's been tried (lessons carried forward)

| Attempt | Outcome | Lesson |
|---|---|---|
| Learned-fusion LightGBM ranker (2026-04-29) | +0.3 pp LOOCV vs +5 pp gate; **-27.8 pp** regression on training-17 | Catastrophic single-fold regression with small data. **Re-try only with strict per-fold regression guard AND with video features actually populated**, which the apr-29 run lacked. |
| Basic Pitch fine-tune (2026-04-30) | Superseded by highres backend swap | Fine-tune infra reusable; ceiling lift now lives in highres post-processing and possibly a GuitarSet-only highres fine-tune. |
| Melodic prior | Regresses aggregate by 1.15 pp | Helps solo, hurts comp. Needs solo-density gating. |
| Position prior `guitarset-v1` | +22 pp Tab F1 | Per-pitch tabular priors are the largest cheap intervention. Style/structure-conditional priors are the natural extension. |

## 5. Composite eval policy

Each tier in the composite eval set must satisfy these rules. The
manifest schema (`tabvision/tabvision/eval/manifest.py`) already
encodes tier names and required clip fields; the Phase 0 impl plan
extends it for source-specific annotation paths and CI reporting.

**Per-tier minimums:**
- Each of the four required tiers: **≥ 20 clips** and **≥ 500 gold
  notes**. Below this the bootstrap CI is too wide to claim acceptance.
- Total composite: ≥ 80 clips, ≥ 2,000 notes.

**Split policy:**
- GuitarSet: held-out **by player** (player 05 = validation; this is
  the existing convention from `guitarset_audio_eval.py`). No
  train/test leak at player level.
- Guitar-TECHS: split by **performer** if performer metadata is
  available; else by clip with a deterministic seed.
- EGDB: split by **source track** (the 240 clean DIs); amp-sim
  renders of the same track go to the same split. Required to avoid
  amp-render leakage.

**Source weighting:**
- Per-tier metrics are reported **un-weighted across sources within a
  tier** (every clip has equal weight). The strategic question "is
  GuitarSet over-represented in clean acoustic" gets a separate
  per-source breakdown in the report; the headline number is the
  un-weighted clip mean.

**Leakage rules:**
- No clip used for prior training (`guitarset-v1` etc.) appears in
  evaluation. Currently `guitarset-v1` is trained on GuitarSet train
  split, evaluated on player 05 — compliant.
- Fine-tune sets must be disjoint from eval sets by player / performer.
- DadaGP-derived synthetic, if used, is training-only and never
  appears in the eval manifest.

**Confidence intervals:**
- Every per-tier number reported with a **95% bootstrap CI** over
  clips (resample clips with replacement, recompute the tier-mean,
  10 000 resamples). The acceptance test is `lower_95_CI ≥ target`,
  not just `mean ≥ target` — this disciplines small-sample wishful
  thinking.

**Parsers:**
- One parser per source, named by the annotation format (not the
  source). Phase 0 ships: `guitarset_jams`, `guitar_techs_midi`,
  `egdb_gp`. Each parser converts source-native annotations into the
  §8 `TabEvent` dataclass list. Round-trip parser tests required.

## 6. Phase outline (high-level only)

Each phase has a goal + acceptance bar here. **Per-phase implementation
plans** (exact files / tests / commands / acceptance outputs) are
written **separately**, one phase at a time, only after the prior
phase's evidence justifies starting it.

- **Phase 0 — Foundation.** Per-tier baselines + error decomposition on
  the composite eval. Acquire Guitar-TECHS; send EGDB email; verify free
  compute accounts. **No production code changes.** Acceptance: per-tier
  baseline numbers exist for ≥ 3 of 4 tiers with bootstrap CIs;
  per-tier 7-bucket error breakdown exists. [Companion:
  `2026-05-13-tab-f1-phase-0-implementation.md`.]
- **Phase 1 — Pitch ceiling lift (cheap moves).** Voicing/silence gate
  + peak-picking + Basic Pitch pitch-only ensemble. Acceptance: Pitch
  F1 ≥ 0.93 on GuitarSet validation, no Onset F1 regression > 1 pp.
- **Phase 2 — Highres fine-tune on GuitarSet only.** Lightning
  free-tier; ~3 GPU-hr. **No SynthTab pretrain.** Acceptance: Pitch F1
  ≥ 0.94, no Onset regression > 1 pp; cross-dataset sanity ≥ 0.90 on
  Guitar-TECHS held-out.
- **Phase 3 — Style/structure-conditional priors.** Leave-one-player-out
  CV with hard regression guard. Acceptance: solo Tab F1 +2 pp vs
  `guitarset-v1`, no per-bucket regression > 1 pp on comp, no fold
  regression > 3 pp.
- **Phase 4 — UI-field audit (capo/tuning/instrument/tone/style).**
  Unit tests confirm each field propagates into a pipeline decision.
- **Phase 5 — Learned fusion v2.** Re-attempt with proper features
  (chord-context, prior-values, playability-cost, video-when-on).
  Acceptance: +3 pp mean Tab F1, no per-fold regression > 3 pp,
  margin-fallback to structured search baked in.
- **Phase 6 — Video pipeline qualitative integration.** Enable
  `TABVISION_VIDEO_ENABLED=true` in dev with a runtime quality gate.
  Acceptance: video on/off does not regress audio-only metrics by > 0.5 pp.
- **Phase 7 — Solo-gated melodic prior.** Acceptance: solo +3 pp,
  comp ±1 pp.
- **Phase 8 — Tier shortfall recovery.** Only if a tier still misses
  its D2 target. Per-tier tactics (chord-shape templates for strummed,
  IR-augmentation for distorted, etc.).
- **Phase 9 — Final eval + DECISIONS.md update + SPEC.md PR.**

Sequencing: 0 → 1 → 2 in series; 3–7 parallelizable after 2; 8 only
on shortfall; 9 closes. Total wall-clock estimate: **4-6 weeks
engineering** + ~1 week EGDB-email turnaround.

## 7. Risks

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | EGDB license never resolves | medium | Phase 8 fallback: free-IR-augmented GuitarSet for distorted-electric tier; explicitly flagged as synthesized in reports. |
| R2 | Guitar-TECHS clips don't span all promised tiers (some clean-electric tracks may be missing) | low-medium | Phase 0 acceptance only requires ≥ 3 of 4 tiers; distorted-electric can wait on EGDB. |
| R3 | GuitarSet-only fine-tune (Phase 2) over-fits player 05's adjacent training distribution | medium | Cross-dataset sanity on Guitar-TECHS held-out; abort if Guitar-TECHS regresses > 5 pp. |
| R4 | Per-tier composite has too few clips for statistical significance | medium | D2 acceptance requires `lower_95_CI ≥ target`, not mean. Per-tier minimum 20 clips / 500 notes (§5). |
| R5 | Phase 5 learned fusion reproduces apr-29 single-fold catastrophe | medium | Strict per-fold regression guard + margin fallback. Decision tree pivots to Phase 7 if it triggers. |
| R6 | LICENSES.md updates required for Guitar-TECHS / EGDB / IR packs | certain | Update in Phase 0 alongside acquisition. |
| R7 | Free-tier monthly compute allowance exhausted before Phase 2 + 5 retries | low | Phase 2 ≈ 3 GPU-hr; Phase 5 is CPU. Combined < 10 hr/month, well inside Lightning's 22 hr cap. |
| R8 | Synthetic data (DadaGP) inadvertently ends up in shipped weights via training/eval pipeline cross-contamination | low | Synthetic clips never appear in `tabvision/data/eval/manifest.toml`; an explicit assert in Phase 0 manifest validator rejects any synthetic-source clip in the default eval set. |

## 8. Out of scope

- Personal training clips (D10).
- SynthTab in any shipped configuration (D4).
- GOAT (license).
- Aggregate Tab F1 ≥ 0.88 as an acceptance gate (D1).
- Stretch v1.1 (bends / slides / hammer-ons) per D8.
- Quantitative video-gate (D5).
- Top-K UI optimization — UI work is separate; D2 applies to top-1.
- §8 contract changes — no SPEC §8 signature edits in this plan.
- Modal as a default training surface (D6).

## 9. Open questions (do not gate the plan)

- EGDB author reply timing — assumed ~1 week.
- Whether Guitar-TECHS subdivides cleanly into "clean acoustic" vs
  "clean electric" subsets at clip-level metadata, or whether we'll
  need to inspect waveforms.
- Whether free IR pack licenses (Modern Music Solutions, Djammincabs)
  permit redistribution of derived audio in evaluation reports.
  Phase 8 fallback only.

## 10. Companion docs in this PR

- `SPEC.md` — §1.4.1 amendment block (per-tier targets + composite test set).
- `CLAUDE.md` — active-branch update (`main`, not `refactor/v1`).
- `docs/plans/2026-05-13-tab-f1-phase-0-implementation.md` — Phase 0
  implementation: exact files, tests, commands, acceptance outputs.

Later phase implementation plans (`docs/plans/2026-05-NN-tab-f1-phase-N-implementation.md`)
will be written one phase at a time, only after the prior phase's
evidence is in.
