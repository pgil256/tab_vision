# Tab F1 → Per-Tier Targets — Design

**Date:** 2026-05-12
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — pending sign-off
**Spec source:** `SPEC.md` §1.4 (per-tier table), §5 Phase 5, §8 contracts, §1.5 hard constraints, §6.3 free compute accounts.
**Branch:** to be cut off `refactor/v1` once approved.
**Depends on:** `docs/plans/2026-05-06-phase5-fusion-design.md`, `docs/plans/2026-05-06-video-pipeline-integration-design.md`, `docs/EVAL_REPORTS/guitarset_accuracy_boost_2026-05-08.md`.
**Replaces:** earlier 2026-05-12 single-aggregate-target draft (never committed).

## 0. Decisions taken on 2026-05-12

These were locked in during the planning conversation; record them in
`docs/DECISIONS.md` per SPEC §0.5 once the plan is approved.

| # | Decision | Rationale |
|---|---|---|
| D1 | Tab F1 evaluated **per tier**, not as a single aggregate. SPEC §1.4 aggregate 0.88 is retired. | Aggregate hides the real failure mode (string/fret assignment on solo lines). Per-tier targets force the conversation onto the right axis. |
| D2 | Per-tier numeric targets (table below). | Strummed raised from SPEC 0.86 → 0.90; distorted-electric floor lowered 0.82 → 0.80. Middle tiers relaxed to reflect the gap between current 0.61 and any realistic ceiling. |
| D3 | Eval set is a **multi-source composite**: GuitarSet + Guitar-TECHS + GOAT + EGDB (pending license) + synthetic. Personal videos banned from any role. | GuitarSet alone gives one player, one genre cluster, no electric/distorted. Per-tier evaluation requires per-tier sources. |
| D4 | **SynthTab** pretrain → real-data fine-tune is the audio-side plan. No DIY DadaGP synthesis unless SynthTab proves insufficient. | SynthTab (CC-BY-4.0, ~6,700 h with string/fret labels) pre-empts the engineering cost of building a renderer. Literature (SynthTab paper, High-Res Domain Adaptation arXiv:2402.15258) shows pretrain+fine-tune lifts cross-dataset generalization. |
| D5 | **No quantitative video-gate.** Video pipeline ships as a qualitative feature. Production runs audio-only; video is opt-in. | No public dataset has synchronized guitar video + per-note string/fret labels. Confirmed via 2026-05-12 research pass (see §3.1). |
| D6 | **Free-tier compute first.** Order: local CPU > Lightning Studios free (22 GPU-hr/mo) > Kaggle (30 hr/wk T4) > Colab > Modal. | Per CLAUDE.md operating rule 6 and SPEC §6.3 §1.5 hard constraint. The earlier $30-80 fine-tune estimate was Modal pricing; free tier fits a highres fine-tune comfortably. |
| D7 | **1-2 month cadence.** No fixed deadline. | User-stated. |
| D8 | Stretch goals (bends / slides / hammer-ons / pull-offs) **out of scope** for v1; SPEC §1.4 already marks them v1.1. | Confirmed in conversation. |
| D9 | Top-K is acceptable as an editor UX feature but the **0.80 floor and per-tier targets apply to the top-1 prediction**. | User-stated. |
| D10 | Personal training clips (the 20-video set) **off the table entirely** — not as accuracy gate, not as dev set, not as label source. | User-stated. |

### Per-tier Tab F1 targets (D2)

| Tier | SPEC §1.4 | This plan |
|---|---:|---:|
| Clean acoustic single-line | 0.94 | **0.85** |
| Clean acoustic strummed | 0.86 | **0.90** |
| Clean electric | 0.90 | **0.87** |
| Distorted electric | 0.82 | **0.80** |

All on the multi-source composite test set (D3). Top-1 prediction only.
Onset F1 (≥ 0.92) and Pitch F1 (≥ 0.90) from SPEC §1.4 remain unchanged
— audio already clears them on GuitarSet.

## 1. Goal

Hit the D2 per-tier Tab F1 targets on the D3 composite eval set within
1-2 months using free-tier compute, while keeping the production system
the SPEC §8 contract-conformant v1 pipeline.

## 2. Current evidence

GuitarSet validation, 60 tracks, 8715 gold notes, 2026-05-08 production
candidate (highres + `guitarset-v1` prior, no video, no melodic prior):

| Metric | Current | SPEC | Status |
|---|---:|---:|---|
| Onset F1 (50 ms) | 0.9218 | ≥ 0.92 | pass |
| Pitch F1 (50 ms) | 0.9022 | ≥ 0.90 | pass |
| Tab F1 aggregate | 0.6104 | ≥ 0.88 (deprecated) | retired metric |

Per-track distribution (2026-05-12 diagnostic):

- Tab F1 mean **0.589**, median 0.620, min 0.166, max 0.933
- Comp tracks (n=30) mean **0.670**; solo tracks (n=30) mean **0.508**
- Worst 10 tracks: 7 are solos. Best 5: 4 are comps.
- Tab/Pitch ratio: comp 0.744, **solo 0.546** — solos lose 45% of
  pitch-correct notes to wrong string/fret assignment.

**Bottleneck is string/fret assignment on single-line passages where
chord-cluster context is absent.** Audio is essentially at spec; only the
Tab F1 numbers are red, and only on the solo regime.

The single-tier mapping of GuitarSet is "clean acoustic strummed" for
comp tracks and "clean acoustic single-line" for solo tracks. The
electric and distorted-electric tiers (D2) have no current measurement
and must be acquired (D3).

## 3. Resource inventory

### 3.1 Datasets

Verified by the 2026-05-12 research pass. Italics = on-disk now;
**bold** = to acquire.

| Source | License | Modality | Labels | Size | Tier coverage |
|---|---|---|---|---|---|
| *GuitarSet* | CC-BY-4.0 | audio (hex + DI) | JAMS (string + fret + pitch) | 3 h, 6 players | clean acoustic single-line, strummed |
| **Guitar-TECHS** | CC-BY-4.0 | audio (multi-mic + DI) | 6-track MIDI per string | 5h12m | clean acoustic single-line, clean electric |
| **GOAT** | CC-BY-4.0 | DI audio | tablature | 5.9 h | clean electric |
| **EGDB** | None on repo — **email author for portfolio-use permission** | audio (DI + 5 amp sims, ~6 renders) | GuitarPro tabs + aligned MIDI (string + fret) | ~12 h synthesized | clean electric, distorted electric |
| **SynthTab** | CC-BY-4.0 | synthesized audio | string + fret + onset | ~6,700 h | all four tiers (pretrain only) |
| GAPS | CC-BY-NC-SA | YouTube video links + MIDI pitch | pitch-only, **not tab** | 14 h | reject — non-commercial taint |
| DadaGP | research-access (email) | symbolic GP files | tab natively | 26,181 files | fallback synthesis source if SynthTab insufficient |
| ~~The 20 personal clips~~ | n/a | n/a | n/a | n/a | **banned** (D10) |

**Confirmed gap:** no public dataset combines guitar video with per-note
string+fret labels. This is the load-bearing finding behind D5.

### 3.2 Compute

| Account | Free allowance | Status | Use |
|---|---|---|---|
| Lightning Studios | 22 GPU-hr/month | Phase 0 setup | SynthTab pretrain, highres fine-tune |
| Kaggle | ~30 GPU-hr/week T4 | Phase 0 setup | overflow for long sweeps |
| Colab | ~12 hr/day with limits | Phase 0 setup | quick experiments |
| W&B | unlimited (academic) | Phase 0 setup | experiment tracking |
| HuggingFace Hub | unlimited public | already used | weights / checkpoints |
| Modal | pay-per-use | already used | production smoke retests only |
| Local CPU | 6 cores WSL2 | available | eval, priors, light tuning |

Per CLAUDE.md operating rule 6: Local > Colab > Kaggle > Lightning >
Modal. Modal is the resort, not the default.

### 3.3 Code already in tree

- `tabvision.audio.highres` — production pitch backend, 0.92 / 0.90 on GuitarSet.
- `tabvision.fusion.position_prior` — `guitarset-v1` prior, +22pp Tab F1.
- `tabvision.fusion.{viterbi,chord,playability,neck_prior,melodic_prior}` — Phase 5 shipped, cluster Viterbi + chord state enumeration + playability emission/transition costs.
- `tabvision.video.{guitar,fretboard,hand}` — Phase 4 shipped (1603 LOC).
- `tabvision.pipeline.run_pipeline` — composes all of the above; production runs through it via `tabvision-server/app/v1_adapter.py`.
- `tabvision-server/tools/eval_basic_pitch_baseline.py` + `tabvision/scripts/eval/guitarset_audio_eval.py` — current evaluation harness; needs extension for multi-source composite.
- `tabvision-server/tools/outputs/errors-2026-04-28_185743.md` — apr-28 error-decomposition methodology (proven on personal clips); port the same 7-bucket harness to the composite eval set.

### 3.4 What has been tried (lessons)

| Attempt | Date | Outcome | Lesson |
|---|---|---|---|
| Learned-fusion LightGBM ranker | 2026-04-29 | +0.3pp LOOCV vs +5pp gate; **catastrophic −27.8pp regression on training-17** | Small data + over-fit on one held-out group. **Critically: video features were `null` on every row** (`audio_only=True`) — so this wasn't actually a test of learned-fusion-with-video. Re-attempt with proper feature instrumentation is justified. |
| Basic Pitch fine-tune on GuitarSet | 2026-04-29/30 | Did happen; superseded by highres backend swap before final integration | Fine-tune infrastructure is reusable for highres; SynthTab pretrain is the missing first step. |
| Melodic prior | current | Regresses aggregate Tab F1 from 0.6104 to 0.5989 | Helps solo, hurts comp. Needs solo-density gating, not a flat enable. |
| Position prior `guitarset-v1` | 2026-05-08 | +22pp Tab F1 vs no prior | Per-pitch tabular priors are the largest-leverage cheap intervention. Style/structure-conditional versions are the natural extension. |
| Phase 5 cluster Viterbi + chord enumeration | 2026-05-06 | Shipped, drives current production | The audio-only structured search is already well-tuned. Further gain needs either better priors or different evidence (which video can't provide on the eval set). |

## 4. Plan

10 phases. Phases 0–2 sequential; 3–8 parallelizable. Decision tree
inside each phase determines whether to continue, branch, or escalate.

### Phase 0 — Foundation (parallel, no compute, 1 week wall-clock)

**Goal:** assemble the evidence base + accounts the rest of the plan
depends on. No production code changes; setup only.

Concurrent tracks:

- **0A. Acquisition.**
  - [user] Email EGDB author (`f08946011@ntu.edu.tw`) for written
    portfolio-use permission. Template draft in §10.
  - [code] Download Guitar-TECHS and GOAT (both CC-BY-4.0, no email).
  - [code] Sample SynthTab to a 500-clip pilot subset (~50 h). Full
    download deferred until Phase 2.
- **0B. Compute accounts.**
  - [user] Lightning Studios, Kaggle, Colab, W&B sign-ups per SPEC §6.3.
  - [code] Verify each by running a hello-world (W&B init + a GPU
    `nvidia-smi` job on each platform).
- **0C. Eval harness extension.**
  - [code] Build `tabvision/scripts/eval/composite_eval.py`. Reads a
    manifest TOML (per-clip tier label + source + audio path + tab
    annotation path) and runs the same `guitarset_audio_eval.py` logic
    across all sources. Outputs per-tier Tab F1, per-source CSVs, and a
    consolidated Markdown report.
  - Manifest schema follows the placeholder in
    `tabvision/data/eval/manifest.toml`. Tier label is one of
    `clean_acoustic_single_line`, `clean_acoustic_strummed`,
    `clean_electric`, `distorted_electric`.
- **0D. Error decomposition.**
  - [code] Port `tools/error_analysis.py` (apr-28 7-bucket harness)
    from personal-clip input to the composite eval set. Output:
    `docs/EVAL_REPORTS/error_decomposition_<date>.md` with per-tier
    bucket counts.
- **0E. Baseline measurement.**
  - [code] Run `composite_eval.py` against the current production
    pipeline. Get the per-tier numbers. These are the Phase 1+
    starting points.

**Phase 0 acceptance gate:**
- Per-tier Tab F1 baseline numbers exist for at least 3 of the 4 tiers
  (distorted electric is EGDB-dependent; deferred OK).
- Per-tier 7-bucket error decomposition exists.
- All free-tier compute accounts verified.
- EGDB email sent.
- No production code changes.

**Decision tree:**
- If baseline already hits some tier (e.g., strummed at 0.92) → drop
  that tier from later phases' work.
- If pitch-side metrics regress vs the 2026-05-08 GuitarSet numbers on
  the composite set → STOP and investigate before any further work.
  The composite eval should not change audio-side numbers on GuitarSet.

### Phase 1 — Pitch ceiling lift, cheap moves (local CPU, 2-3 days)

**Goal:** Pitch F1 from 0.915 → ≥ 0.93 on GuitarSet validation, without
training. Gives Tab F1 mathematical headroom regardless of fusion-side
work.

Moves, in order:

1. **Voicing/silence gate** on highres pitch posteriors. Tune the
   joint onset+pitch confidence threshold. Trade some recall for
   precision; expect net F1 gain.
2. **Onset peak-picking adjustment.** The 50 ms tolerance is generous;
   misaligned within-tolerance peaks still produce pitch mis-reads.
   Improve peak localization → tighter onset match → higher pitch TP
   count.
3. **Basic Pitch pitch-only ensemble.** Run Basic Pitch alongside
   highres. Use Basic Pitch's pitch output (not onset) as a tiebreaker
   on pitch-disagreement events; downweight (or drop) events where the
   two backends disagree on pitch. SPEC §6.1 path; LICENSES.md
   confirms Basic Pitch is Apache-2.0 default-pipeline-safe.

**Phase 1 acceptance:**
- Pitch F1 ≥ 0.93 on GuitarSet validation.
- Onset F1 ≥ 0.92 (no regression).
- Aggregate Tab F1 ≥ 0.62 (no regression beyond mathematical
  pitch-improvement bound).

**Decision tree:**
- 0.93 met → continue.
- 0.92–0.93 → continue; Phase 2 fine-tune still useful as ceiling lift.
- < 0.92 → diagnose. Could be a threshold-sweep artifact rather than a
  real regression. Inspect on 3-5 representative tracks before
  escalating.

### Phase 2 — SynthTab pretrain + highres fine-tune (Lightning, 1 week)

**Goal:** Pitch F1 ≥ 0.94 on GuitarSet validation. Lift the audio
ceiling beyond what threshold-tuning alone can do.

**Approach.** Per the SynthTab paper (ICASSP 2024) and arXiv:2402.15258,
the proven recipe is: pretrain on synthetic, fine-tune on real.

- **Pretrain corpus:** SynthTab 500-clip pilot (Phase 0). Full set
  (~6,700 h) is overkill at this stage and won't fit in the free tier
  monthly budget.
- **Pretrain head:** the highres model's pitch+onset head. Backbone
  frozen for the pretrain phase to avoid catastrophic forgetting on
  the spectral feature extractor.
- **Fine-tune:** GuitarSet train split (4 players, 240 tracks ≈ 2 h),
  unfrozen, 5-10 epochs with early stopping on Pitch F1.
- **Compute:** Lightning Studios free tier (22 GPU-hr/month). Estimate:
  pretrain ~6 GPU-hr, fine-tune ~3 GPU-hr. Buffer for re-runs ~5 GPU-hr.
  Stays inside the monthly allowance.

**Phase 2 acceptance:**
- GuitarSet validation Pitch F1 ≥ 0.94.
- No Onset F1 regression > 1 pp.
- Cross-dataset sanity: on Guitar-TECHS (held out from training),
  Pitch F1 ≥ 0.90 (no catastrophic transfer loss).

**Decision tree:**
- Met all three → continue. New `audio_backend = "highres-synthtab"`
  becomes the candidate for production replacement.
- GuitarSet met, Guitar-TECHS regresses > 5 pp → over-fit on the
  pretrain distribution. Reduce pretrain epochs, increase fine-tune
  weight, retry once.
- GuitarSet ≤ 0.93 → SynthTab pretrain didn't transfer; abandon
  Phase 2 and revisit with the actual diagnostic (Pitch P/R curves)
  before any further training spend.

### Phase 3 — Style/structure-conditional priors (local CPU, 3 days)

**Goal:** lift Tab F1 on solos via finer-grained per-pitch position
priors. Expected +1 to +5 pp on solo subsets.

- **Buckets:** {bn, jazz, funk, rock, ss} × {solo, comp} = 10 priors.
  GuitarSet's `style` field gives the genre axis directly; structure
  axis derived from cluster-singleton density (already computable in
  fusion).
- **Train:** GuitarSet train split (players 00, 01, 02, 03, 04).
  Per-bucket Laplace-smoothed counts. Empty cells fall back to
  `guitarset-v1`.
- **Validate:** leave-one-player-out CV (not LOOCV per-clip — too
  small). Primary metric: per-bucket Tab F1 delta vs `guitarset-v1`
  baseline on the held-out player.
- **Risk:** the apr-29 learned-fusion attempt failed with one
  catastrophic regression. Same class of risk here — small data,
  bucketing on 4 training players. **Hard regression guard:** abort
  the bucket if any cross-validation fold regresses by > 3 pp.

**Phase 3 acceptance:**
- Mean Tab F1 over solo buckets: +2 pp vs `guitarset-v1` baseline.
- No bucket regresses by > 1 pp on comp.
- No cross-validation fold regresses by > 3 pp on any bucket.

**Decision tree:**
- Met → ship the prior set, expose `position_prior = "guitarset-styled-v1"`.
- Solo gain < 2 pp → drop the structure axis, ship style-only.
- Any bucket fails the regression guard → drop that bucket only;
  fall back to `guitarset-v1` for it. Don't kill the whole experiment
  on one bad bucket.

### Phase 4 — Style+structure-aware capo/tuning audit (local, 1 day)

**Goal:** verify the capo / instrument / tone / style fields from the
upload UI are actually flowing into prior selection and playability
weights as designed.

- **Trace:** unit-test that with `capo_fret = 5` the position prior
  shifts correctly (frets 0-19 become frets 5-24).
- **Smoke:** run a known capo-3 clip from GuitarSet (if any exist)
  and confirm the output tab is rendered against the capo.
- **Audit playability:** confirm `instrument = electric` doesn't apply
  the open-string bonus differently when it shouldn't, etc.

Small phase; mostly a correctness-check before later phases compound
any bugs here.

**Phase 4 acceptance:**
- All upload-form fields measurably affect at least one pipeline
  decision per a unit test.
- No silent fallback to defaults on any field.

### Phase 5 — Learned fusion v2 (local, 3-5 days)

**Goal:** the 2026-04-24 plan's learned-fusion approach, redone with
proper feature instrumentation. **Per-pitch + chord-context ranker**,
not the audio-only ranker that flat-lined at +0.3 pp in 2026-04-29.

**Why this can work this time:** the apr-29 attempt's per-candidate
features were limited (no fusion-prior values, no neck-anchor values
because video was off, no chord-cluster context). With Phase 5
shipping the structured search already, those values are now exposed
and can be features.

**Per-candidate features:**
- `pitch`, `confidence`, `duration`, `amplitude` (audio).
- `position_prior_log_prob`, `melodic_prior_log_prob`,
  `neck_prior_log_prob` (fusion priors at this candidate).
- `cluster_size`, `cluster_span`, `is_singleton`, `singleton_density_2s`
  (chord context).
- `emission_cost`, `transition_cost_to_prev` (playability).
- `cand_string`, `cand_fret`, `is_open`, `is_low_position` (identity).
- `style`, `instrument`, `tone` (from session config; flow-from-UI
  audited in Phase 4).

**Training:** GuitarSet train split, leave-one-player-out CV. LightGBM
`lambdarank` with hard regression guard at -3 pp per held-out player.

**Phase 5 acceptance:**
- Mean Tab F1 across all held-out players: +3 pp vs Phase 3-or-earlier
  baseline.
- No held-out player regresses by > 3 pp.
- Margin-based fallback to structured-search pick when learned-fusion
  margin is below a threshold (mitigates OOD behavior in production).

**Decision tree:**
- Met → ship behind a flag, default off, with the margin fallback.
  Default-on requires a separate review pass with at least one week of
  production smoke clean.
- Per-player regression > 3 pp on any fold → the apr-29 failure mode
  repeats. Stop Phase 5 and pivot to Phase 7 instead.

### Phase 6 — Video pipeline qualitative integration (1-2 days)

Goal: re-enable the video stack in production for users whose uploads
have usable video, without claiming any quantitative Tab F1 improvement.
**No video accuracy gate** (D5).

- Flip `TABVISION_VIDEO_ENABLED=true` in `tabvision-server/modal_app.py`
  in dev.
- Verify pipeline runs end-to-end on at least one synthetic
  fretboard-rendered clip (Phase 6A) and the qualitative output is
  reasonable.
- Add a runtime quality gate (the one the v1_adapter currently fakes):
  reject video evidence when `handDetectionRate < 0.3` or
  `fretboardDetectionConfidence < 0.5`. Diagnostics in result JSON.
- Production smoke: end-to-end on the existing `test_a440.mp4` (audio
  ceiling) and one real-world iPhone clip (qualitative inspection only,
  not gated).

**Phase 6A — Synthetic fretboard video** (optional, 2-3 days):
- Render a procedurally-generated fretboard animation (Blender or
  pyrender) against SynthTab audio. Synchronized by-construction.
- Use for video-pipeline smoke + regression tests (does turning video
  on/off change anything?), NOT for accuracy claims.

**Phase 6 acceptance:**
- Video enable in dev does not regress GuitarSet audio-only Pitch /
  Onset / Tab F1 metrics (delta within ±0.5 pp).
- At least one synthetic clip produces a non-empty `fingerings` list
  in the result.
- Production smoke clean.

**Decision tree:**
- Audio-only metrics regress when video enabled → video is making
  things worse on no-video-content clips. Add a fail-fast that
  disables video output when `videoObservationCount == 0`, retry.
- No regression but no positive signal either → ship video as opt-in,
  default off. Revisit when a public video+tab dataset emerges.
- Positive signal on some clips → ship default-on with the quality
  gate.

### Phase 7 — Solo-gated melodic prior (local, 2 days)

**Goal:** re-enable the existing melodic prior in the regime where it
helps (solo passages) without re-introducing the comp regression that
caused the current ship-disable.

- Gate the melodic prior on rolling-window singleton density: apply
  only when ≥ 80% of clusters in the last 2 seconds are singletons.
- Re-tune the 35/65 prior-blend ratio currently hard-coded in
  `tabvision/tabvision/fusion/melodic_prior.py:64`.

**Phase 7 acceptance:**
- Solo subset Tab F1 +3 pp vs Phase 3 baseline.
- Comp subset Tab F1 within ±1 pp.
- No per-track regression > 3 pp.

### Phase 8 — Tier shortfall recovery (as needed, 1-2 weeks)

Triggered only if a tier still misses its D2 target after Phases 1-7.

- **Distorted electric < 0.80:**
  - If EGDB acquired: oversample EGDB distorted variants in Phase 2
    fine-tune; re-run.
  - If EGDB blocked: synthesize a distorted training subset via
    SynthTab clean audio + free IR pack convolution (Modern Music
    Solutions Declassified, Djammincabs).
- **Clean acoustic single-line < 0.85:**
  - Re-tune Phase 7 melodic-prior strength on the single-line subset.
  - If still short: add a position-shift smoothing prior (events
    within < 200 ms shouldn't span > 5 frets unless audio amplitude
    suggests a deliberate slide).
- **Clean acoustic strummed < 0.90:**
  - Chord-shape template prior: for each detected chord cluster,
    boost candidate fingerings that match a curated set of 30-50
    common guitar chord shapes (port from
    `tabvision-server/app/chord_shapes.py`, 790 LOC).
- **Clean electric < 0.87:**
  - Likely co-resolves with one of the above. Investigate per-tier
    error decomposition before adding tier-specific work.

### Phase 9 — Final eval + documentation

- Run `composite_eval.py` with full per-tier table.
- Write `docs/EVAL_REPORTS/per_tier_acceptance_<date>.md`.
- Update `docs/DECISIONS.md` with each Dn entry actually taken.
- Final SPEC §1.4 amendment proposal: tier table replaces aggregate
  target. Land as a SPEC PR.

## 5. Sequencing

```
Phase 0 (parallel setup)  [week 1]
    ↓
Phase 1 (pitch ceiling cheap)  [week 1]
    ↓
Phase 2 (SynthTab + fine-tune)  [week 2]
    ↓
┌────────────────────────────────────────┐
│ Phase 3 (style priors)          [w3]   │
│ Phase 4 (UI fields audit)       [w3]   │  parallel
│ Phase 5 (learned fusion v2)     [w3-4] │
│ Phase 6 (video qualitative)     [w3]   │
│ Phase 7 (solo melodic prior)    [w3]   │
└────────────────────────────────────────┘
    ↓
Phase 8 (tier recovery)          [w5-6 as needed]
    ↓
Phase 9 (final eval + docs)      [w6]
```

Total wall-clock: **4-6 weeks engineering**, plus 1-2 weeks waiting
time on the EGDB email if it gates Phase 8 distorted-electric work.

## 6. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | SynthTab pretrain doesn't transfer to real audio (domain gap) | medium | Literature shows pretrain+fine-tune works (SynthTab paper, arXiv:2402.15258). Smoke on Guitar-TECHS held-out before committing to full pretrain spend. |
| R2 | EGDB license never resolves | low-medium | Author replies are usually fast; if blocked, synthetic IR-based distorted electric via Phase 8 fallback. |
| R3 | SynthTab labels are noisy (DadaGP human-transcribed varies in quality) | medium | Use SynthTab as pretrain only, never as eval gate. Phase 0 spot-check a 50-clip random sample. |
| R4 | Per-tier composite eval set has too few clips per tier for statistical significance | medium | Bootstrap 95% CIs in all per-tier reports. State the CI explicitly when reporting against the D2 target. |
| R5 | Video pipeline degrades audio-only metrics when enabled | low | Quality gate in Phase 6 + audio-only fallback. Phase 6 acceptance explicitly checks this. |
| R6 | Phase 5 learned-fusion reproduces the apr-29 single-fold catastrophe | medium | Hard regression guard per-fold + margin fallback to structured search. Phase 5 decision tree pivots to Phase 7 if it triggers. |
| R7 | Free-tier compute monthly allowance insufficient for Phase 2 + 8 retries | low | Lightning 22 hr/mo + Kaggle 30 hr/wk + Colab is ~150 GPU-hr/mo combined; Phase 2 needs ~14 hr. Plenty of buffer. |
| R8 | LICENSES.md needs updates for Guitar-TECHS, GOAT, SynthTab, EGDB | certain | Update in Phase 0. Each is CC-BY-4.0 (or pending in EGDB's case); attribution must appear in README and any blog. |

## 7. Out of scope

- Personal training clips (D10).
- Single-aggregate Tab F1 ≥ 0.88 (D1).
- Stretch v1.1 (bends/slides/hammer-ons) per D8.
- Quantitative video-gate (D5). Video ships qualitative-only.
- Top-K UX surface — UI work is separate. D2 targets apply to top-1.
- New SPEC §8 contracts — none of these phases changes signatures.
- Real-money compute except for production smoke retests on Modal.

## 8. Phase 0 user actions (the things only you can do)

1. Sign up / verify free-tier compute accounts:
   - Lightning Studios (https://lightning.ai)
   - Kaggle (https://kaggle.com)
   - Colab (https://colab.research.google.com)
   - Weights & Biases (https://wandb.ai, free academic tier)
2. Email the EGDB author for portfolio-use written permission.
   Template:

   > Subject: TabVision portfolio project — request to use EGDB
   >
   > Dr. Chen,
   >
   > I'm a developer building TabVision, a portfolio guitar
   > transcription project (public GitHub repo, blog post, recorded
   > demo). I would like to use EGDB as the distorted-electric
   > evaluation tier of my multi-source test set, and cite your
   > ICASSP 2022 paper. The repo has no LICENSE file, so I'm asking
   > for written permission to use EGDB in this portfolio context,
   > including reporting evaluation metrics computed on it.
   >
   > Thank you,
   > Patrick Gilhooley

3. Confirm or push back on the D2 per-tier targets (table in §0).
4. Approve the plan; I cut a branch from `refactor/v1` and start
   Phase 0E (the baseline measurement, since 0A and 0B are blocked on
   the above).

## 9. Things still genuinely unresolved

These can be answered in flight; don't gate the plan on them.

- The exact size of the SynthTab pilot (500 clips is a guess; the
  right number is "smallest subset that produces a fine-tune gain"
  and emerges from Phase 2's first run).
- Whether Phase 4 finds any actual capo/tuning regressions worth
  fixing, or if it's a 30-minute box-tick.
- Phase 6A: whether procedural fretboard rendering is 2 days or 2
  weeks of work. Defer until we know whether Phase 6 alone is enough.

## 10. Open invitation to redirect

This plan favors free compute over fast iteration; SynthTab over DIY
synthesis; per-tier targets over single-aggregate; audio-only gates
over speculative video-gate construction. If any of those defaults are
wrong for what you actually want, say so before Phase 0 starts —
backtracking from Phase 3 is expensive.
