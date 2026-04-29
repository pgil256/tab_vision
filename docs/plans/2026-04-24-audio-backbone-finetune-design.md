# Path 2 — Fine-tune Basic Pitch on Guitar Audio

**Date:** 2026-04-24 (refreshed 2026-04-29)
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — actively executing on `feature/audio-finetune-phase1`
**Originally superseded:** `2026-04-24-learned-fusion-design.md`. That plan was attempted on `agent-farm-improvements` 2026-04-29 and **did not ship** (LightGBM ranker LOOCV +0.3pp vs +5pp gate). See `tools/outputs/position_selector_report-2026-04-29.md`.

## 0. 2026-04-29 refresh — corrected baselines

§1's "audio is 91% of loss" framing relied on a harness bug that produced misaligned bucketing (no `_find_best_time_offset`, no muted-X separation). The corrected harness (`agent-farm-improvements` commit `fa3ca0f`, output `tools/outputs/errors-2026-04-28_185743.md`) gives different numbers:

| Bucket (recoverable loss) | Original (apr-24) | Corrected (apr-28) | Δ |
|---|---:|---:|---:|
| extra_detection | 246 (39.4%) | 83 (27.9%) | -163 |
| missed_onset | 170 (27.2%) | 74 (24.8%) | -96 |
| pitch_off | 153 (24.5%) | 31 (10.4%) | -122 |
| wrong_position_same_pitch | 55 (8.8%) | 105 (35.2%) | +50 |
| timing_only | 7 (1.1%) | 5 (1.7%) | -2 |

Net implications:

- **Audio-side total share went from 91% → 63%** of recoverable loss. Audio is still the larger fight, but not by an order of magnitude.
- **Mean exact F1 baseline shifts from 0.43 to ~0.51** (222 correct out of 437 pitched GT — see `position_dataset.md`). This is the reference point for the ship gate below.
- **Phase 0 in §6 already failed against the buggy harness** (`feature/audio-finetune` commit `382ed03`, memory: `project_phase0_rms_activity_end.md`). End-of-playing RMS truncation is a no-op on iPhone audio; per-clip amplitude floor only marginally helped. Phase 0 is shelved as designed; if we want extras truncation, it needs a different mechanism (e.g., last-pitched-note timestamp from Basic Pitch itself).
- **Ship gate revised** (was: training-set mean exact F1 ≥ 0.60, pitch F1 ≥ 0.88):
  - **Training-set mean exact F1 ≥ 0.60** (still ≈ +9 pp on the corrected baseline; original was +17 pp on the buggy baseline).
  - **Training-set mean pitch F1 ≥ 0.88** (corrected baseline pitch F1 from `errors-2026-04-28_185743.md`: pitch_tp = 222 + 105 = 327, pitch_fn = 31 + 74 + 5 = 110, pitch precision/recall ≈ 0.74/0.75. So +13 pp on pitch F1 stays roughly accurate.).
  - Worst-video regression > -0.03 forbidden — unchanged.
  - Original 11-video suite stays > 0.91 — unchanged.
- **Expected gains in §2 (50% pitch_off reduction, 30% extras, 20% missed)** translate to: 16 pitch_off + 25 extras + 15 missed = ~56 events recovered out of 215 wrong (-26% of loss). Still meaningful, much smaller than the +30-40% recovery the original plan implied.

The 5-week timeline and architecture are unchanged. The §7 phasing (Week 1 = de-risk, Week 5 = ship gate) is still the right structure.

## 1. Context

The Step-0 error analysis (originally `tools/outputs/errors-2026-04-24_131204.md`, refreshed in `tools/outputs/errors-2026-04-28_185743.md`) broke the 20-video training set into seven buckets. The original framing said audio was the bottleneck; the corrected framing in §0 above tempers that — see §0 for the refreshed numbers. Original §1 numbers preserved below for traceability:

| Bucket | Count | Share of exact-F1 loss |
|---|---:|---:|
| extra_detection | 246 | 39.4% |
| missed_onset | 170 | 27.2% |
| pitch_off | 153 | 24.5% |
| wrong_position_same_pitch | 55 | 8.8% |
| timing_only | 7 | 1.1% |

Position selection (the target of the shelved learned-fusion plan) is only 8.8% of loss. Three audio-side buckets together are 91%. Of those, the largest lever is replacing Basic Pitch — a general-purpose polyphonic pitch model — with a version fine-tuned on guitar audio specifically.

Separate finding: **67.5% of `extra_detection` events fall after GT_end + 1s.** This is independent of the model and addressable with cheap heuristics (Phase 0, below).

## 2. Hypothesis

The dominant loss is that Basic Pitch — the pitch backbone — is genuinely wrong on iPhone-quality guitar audio a lot of the time (pitch F1 75%, 153 pitch_off events in 20 videos). Fine-tuning Basic Pitch on guitar-specific labeled audio will:

- Reduce `pitch_off` by at least half.
- Reduce `extra_detection` by ~30% (better pitch → fewer spurious detections carrying wrong pitches).
- Reduce `missed_onset` by ~20% (sharper onset head).

Target: training-set mean exact F1 from 43% to ≥ 65%, with training-set mean pitch F1 from 75% to ≥ 88%.

## 3. Why this path vs alternatives

Three paths were considered:

- **Path 1: heuristic iteration + small learned components.** Ceiling ~75–85% F1, diminishing returns, fusion engine keeps growing. Rejected as primary, but *Phase 0 borrows from it* for the free-lunch extras fix.
- **Path 2: fine-tune a guitar-specific audio backbone.** Ceiling ~85–92% F1, data problem is tractable via GuitarSet, architecture is already proven. **Chosen.**
- **Path 3: end-to-end audio+video→tab transformer.** Ceiling ~92–97%, but 10× the data is needed and 100× the compute. Deferred until Path 2 ships and we know whether we're still audio-limited.

Within Path 2, two backbones were considered:

- **Fine-tune Basic Pitch** (2M params, TCN, CPU-runnable). Pretrained on GuitarSet+MedleyDB+MAESTRO+MAPS, output interface matches our pipeline. Weak docs on training are the main risk.
- **Small transformer from scratch / pretrained-encoder + head.** Architecturally cleaner, extensible, but 3 hours of GuitarSet is far too little for a cold-start transformer. Kept as the fallback if the Basic Pitch training loop can't be revived in a week.

## 4. Architecture (end state)

Three separable components:

1. **Audio transcription model** — fine-tuned Basic Pitch. In: audio. Out: `(onset_time, midi_pitch, confidence, duration)` per note. Shipped as `app/models/basic_pitch_guitar.pth`, gated by `FusionConfig.use_finetuned_audio`, default off until the regression gate passes.
2. **Video position model** — unchanged in this plan. MediaPipe + fretboard geometry → candidate `(string, fret)` per onset. Replacing this is a separate future plan.
3. **Fusion** — unchanged in this plan. The existing engine consumes `(pitch, candidate positions)` and produces tab notes. With a stronger audio model feeding it, many of today's heuristic filters (sustain-redetection, harmonic filter) can be reviewed and possibly simplified in a follow-up.

Deployment target: Vercel + Next.js frontend, backend TBD (likely FastAPI/Flask on Railway or Fly, with GPU inference on Modal/Replicate if CPU proves too slow). This plan is deployment-agnostic — the fine-tuned checkpoint runs on the existing backend.

## 5. Dataset — Tier A only, initially

- **GuitarSet** (NYU MARL): 360 30-second clips, 6 players, hexaphonic pickup, JAMS annotations with MIDI + (string, fret). ~3 hours aligned audio. Sufficient for a first fine-tune.
- Split 80/20 **by player** (not clip) so no player identity leaks into evaluation.
- Our 20-video training set is the out-of-distribution test — GuitarSet is studio, ours is iPhone.

**Not in scope for this plan:**

- Tier B (render Guitar Pro files to synthetic audio). Add only if Tier A caps below the ship gate.
- Tier C (YouTube + community tab alignment). Deferred indefinitely.

## 6. Phase 0 — free-lunch audio fixes (parallel, ~2 days)

Three cheap fixes that don't depend on fine-tuning. Ship these first to reduce noise in the baseline.

- **End-of-playing truncation.** Drop detected notes with `onset_time > audio_activity_end + 0.5s`, where `audio_activity_end` is the last time the RMS envelope exceeds a silence threshold for 250ms continuously. Target: remove ~66% of today's extras (166 of 246). Expected +6 to +8 pp mean exact F1.
- **Amplitude floor.** Drop detected notes with amplitude below a percentile threshold (e.g. 15th percentile of all detections in the clip). Cheap, tuned per-clip, probably removes a few more extras.
- **Sustain-redetection audit.** Revisit the existing filter — the error analysis will show how many remaining extras are same-pitch same-string sustain re-triggers.

These also clean up the baseline we measure fine-tuning against. Ship them behind `FusionConfig.use_end_of_playing=True` default-on.

## 7. Phase 1 — fine-tune execution (five weeks)

### Week 1 — data loader + training loop (risk phase, timeboxed)

- Download GuitarSet, parse one `.jams` file manually to understand the schema.
- Write `app/training/guitarset_dataset.py` — PyTorch Dataset emitting the exact tensor shapes Basic Pitch's training loop expects.
- Fork Basic Pitch's training repo. Overfit on a 5-clip subset to confirm gradients flow and loss decreases.

**Bailout gate — Friday of week 1.** If a single training epoch cannot be run cleanly on 5 clips, pivot to the middle path: pretrained audio encoder (HuBERT / MusicFM, frozen) + a small onset+pitch head trained from scratch. Do *not* spend week 2 grinding on Basic Pitch's training loop if week 1 didn't crack it.

### Week 2 — baseline + first fine-tune

- Split GuitarSet 80/20 by player.
- Evaluate vanilla Basic Pitch on the held-out 20% — record note F1, frame F1, onset P/R. This is our research baseline.
- First fine-tune: unfreeze all weights, lr=1e-4, batch=8, 20 epochs, Adam, standard Basic Pitch losses (BCE on onset + CE on pitch). Rent an A10G (~8 hours, $3–5).
- Evaluate on the held-out split. First Δ vs vanilla.

### Week 3 — OOD evaluation + augmentation

- Run the fine-tuned model on our 20 training videos.
- Re-run the error analysis harness (`tools/error_analysis.py`) against the new audio output.
- Compare buckets: `missed_onset` recovered? `pitch_off` reduced? `extra_detection` reduced?
- If the 20-video gain is < half the GuitarSet gain, add SpecAugment + room-impulse-response convolution + pink noise to training. Retrain.

**Go/no-go at end of week 3:**

| 20-video pitch F1 gain | Decision |
|---|---|
| ≥ +10 pp (75 → ≥ 85) | Continue to week 4–5. |
| +5 to +10 pp | Continue, and add Tier B synthetic data next. |
| < +5 pp | Audio-alone ceiling reached. Pivot focus to the string/fret head + source separation. |

### Week 4 — hyperparameter sweep

- Grid over lr, batch, epochs, augmentation strength. ~10 runs total.
- Pick best by held-out GuitarSet note F1, breaking ties on our-20-video exact F1.

### Week 5 — integration + ship

- Export checkpoint to `app/models/basic_pitch_guitar.pth`.
- Add `FusionConfig.use_finetuned_audio: bool = False` in `app/fusion_engine.py`.
- Wire a loader in `app/audio_pipeline.py` that uses the fine-tuned checkpoint when the flag is on, stock Basic Pitch otherwise.
- Add a regression test: run one checked-in clip end-to-end, assert mean exact F1 on training-01 exceeds the vanilla-Basic-Pitch baseline by at least the week-3 delta.
- Flip the default to `True` in a follow-up PR after watching it for a few days.

## 8. Ship gate (cross-phase)

To consider Path 2 shipped:

| Metric | Current | Ship gate |
|---|---|---|
| Training-set mean exact F1 | 0.43 | ≥ 0.60 |
| Training-set mean pitch F1 | 0.75 | ≥ 0.88 |
| Worst-video regression on exact F1 | n/a | > -0.03 forbidden |
| Original 11-video suite mean exact F1 | 0.92 | > 0.91 (no regression on the "easy" suite) |
| Per-video inference time | baseline | ≤ 2× (we can spend more on CPU; no GPU required for shipping) |

Phase 0 alone is not expected to hit this gate — its contribution is ~+6–8 pp exact F1. The remaining ~+11–15 pp is the fine-tune's job.

## 9. Relationship to other plans

- **`2026-04-24-learned-fusion-design.md`** — shelved. Its target (position selection) is 8.8% of loss. Revisit after Path 2 ships, because cleaner audio may shrink that share further or leave it as a real remaining gap worth attacking.
- **`2026-04-24-error-analysis-harness.md`** — completed. The harness is the measurement tool for every checkpoint we take from here on.

## 10. Risks

- **Basic Pitch training loop is undocumented.** Bounded by the week-1 timebox + bailout to middle path.
- **GuitarSet was partially in Basic Pitch's pretraining.** The fine-tune gain on GuitarSet's held-out split may be small; the real signal is the OOD test on our 20 videos. Trust week-3 numbers over week-2 numbers.
- **Domain gap (studio vs. iPhone) bigger than expected.** Mitigation: augmentation in week 3. If still too big, add Tier B synthetic data.
- **Pitch gets better but extras stay high.** Means the extras are not pitch-driven (harmonics, resonance, past-end). Phase 0 catches the past-end subset; remaining extras become a separate heuristic pass.
- **String/fret head becomes the new bottleneck.** If pitch F1 goes from 75% to 92% but exact F1 only moves from 43% to 55%, the gap is position selection, and the shelved learned-fusion plan comes off the shelf.
- **We pick a wrong augmentation and regress on the clean GuitarSet set.** Always run both evaluations (held-out + our 20) and don't commit an augmentation that regresses the clean set by more than 2 pp.

## 11. Open questions

- Should we keep CREPE as a monophonic second opinion for pitch_off in low-polyphony passages? Worth testing in week 5 as a cheap additional source of signal, independent of the fine-tune.
- Where does inference live at deploy time — CPU on the existing backend, or GPU on Modal/Replicate? Defer until week 5, when we know the model's wall-clock cost.
- When the fine-tuned audio also emits stronger string-hints (Basic Pitch's per-pitch confidence curve differs per-string in practice), is that signal strong enough to shrink the fusion engine? Measure in week 5, act in a follow-up.

## 12. Out of scope for this plan (ranked for later)

1. **Video position CNN.** Small model trained on labeled frame crops → `(string_probs, fret_probs)` per onset. Only pursue if Path 2 leaves a position gap ≥ 10 pp.
2. **Tier B synthetic data.** Render Guitar Pro files via soundfonts, add to training. Revisit if week 3 go/no-go lands in the middle band.
3. **Source separation prefix.** Demucs/Spleeter stems to isolate guitar when accompaniment is present. Out of scope while the benchmark is clean solo guitar.
4. **Learned fusion (the shelved plan).** Revisit in full after Path 2 ships and re-measure error buckets.
5. **Path 3 end-to-end.** Only if Path 2 plateaus well below 90% F1 and we have resources to collect 10× more data.
