# Learned Fusion for Position Selection — Design

**Date:** 2026-04-24
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed

## 1. Problem

On the 20-video training set:

- **Exact F1: 43%** (pitch+string+fret+time all correct)
- **Pitch F1: 75%** (pitch correct)
- **Position accuracy: 54%** (string+fret correct given correct pitch)

The largest loss is the **pitch→exact gap of ~32 points**: the audio pipeline finds the right pitch, but the fusion engine picks the wrong (string, fret). The original 11-video suite is at 91.6% exact F1, so the gap is both real and video-specific — heuristics over-fit to the easier set.

The fusion engine (`app/fusion_engine.py`, ~2,000 lines) is now a stack of hand-tuned rules. Recent experiments (video hand anchor, fused handedness filter) produce null or negative results. Diminishing returns from heuristics.

## 2. Hypothesis

Position selection is the right intervention point and is learnable from existing data.

Reasons:

- It is by far the largest attributable loss (32pp).
- Every position decision has rich context the heuristic only partially uses (candidate list, hand anchor, prior position, chord membership, video evidence, chord-shape fit).
- Labels exist: for every GT note whose pitch we recover, we know the correct (string, fret).
- Runtime cost of a small model per onset is negligible (<1 ms) — does not violate the speed axis.

## 3. Non-goals

- No new audio backend (Basic Pitch stays; pitch F1 of 75% is a later fight).
- No end-to-end model.
- No change to pre-filter / post-filter / chord-size limiting / anchor math.
- No retraining of MediaPipe or fretboard detection.
- No new training data collection. We work with the 20 existing videos.

## 4. Plan

Six steps. Each is independently checkpointable.

### 4.1 Step 0 — Error analysis harness (~0.5 day)

Build `tabvision-server/tools/error_analysis.py`. For each of the 20 videos:

1. Run pipeline in a fresh subprocess (avoids batch drift per `project_benchmark_drift`).
2. Match detections to GT using existing code in `evaluate_transcription.py`.
3. Bucket every aligned pair into:
   - `correct` — string and fret match
   - `wrong_position_same_pitch` — right pitch, wrong (string, fret)
   - `pitch_off` — detected pitch ≠ GT pitch (audio-side loss, not ours to fix here)
   - `missed_onset` — GT note has no detection
   - `extra_detection` — detection has no matching GT
   - `timing_only` — match outside time_tolerance but otherwise correct
   - `chord_split` — one GT chord → multiple time-adjacent detections

Outputs:

- `tools/outputs/errors-YYYY-MM-DD.csv` — per-note rows with all context fields.
- `tools/outputs/errors-YYYY-MM-DD.md` — per-video and aggregate counts + percentages.

**Decision gate:** if `wrong_position_same_pitch` is not the dominant bucket (≥40% of exact-F1 loss), stop and reconsider. The rest of the plan assumes it is.

### 4.2 Step 1 — Feature instrumentation (~1 day)

The fusion engine already computes everything we need, but it throws it away. Add a `FeatureSink` that, at each position-selection decision, emits one row per candidate (string, fret) containing:

**Per-candidate features:**

- `cand_string` (1–6), `cand_fret` (0–24)
- `dist_anchor_fret`, `dist_anchor_string` (signed, to nearest chord anchor)
- `dist_prev_fret`, `dist_prev_string` (signed, to previous played note)
- `in_chord_region` (bool), `chord_n_notes`, `chord_string_span`
- `chord_shape_score` — lookup via `chord_shapes.py`
- `video_finger_over_fret` — MediaPipe-derived likelihood (0 if no video)
- `video_hand_anchor_fret` — projected fret from hand, when available
- `heuristic_score` — the score the current engine would assign
- `is_heuristic_pick` (bool) — what the heuristic chose, for later calibration

**Per-event features (shared across candidates):**

- `onset_time`, `amplitude`, `basicpitch_confidence`
- `num_candidates`, `is_chord` (>=2 simultaneous), `chord_size`
- `prev_position_string`, `prev_position_fret`, `seconds_since_prev`

Emission is gated behind `FusionConfig.emit_position_features`, default off. Zero runtime cost when flag is off.

### 4.3 Step 2 — Labeled dataset construction (~0.5 day)

`tools/build_position_dataset.py`:

1. Run fusion with `emit_position_features=True` on all 20 videos.
2. For each emitted event, align the detected onset+pitch to GT using step 0's matcher.
3. If pitch aligns and GT has a (string, fret): mark the candidate whose (string, fret) matches GT as `label=1`, all others `label=0`. Drop events with no GT match.
4. Save as `tools/outputs/position_dataset.parquet` with ~2,000–4,000 candidate-rows.

Include `video_id` as a group key so we can do leave-one-video-out CV.

### 4.4 Step 3 — Model training (~1 day)

Use **LightGBM in `lambdarank` mode** (pairwise ranking). Each event is a query; candidates are documents; label is 1 for the correct position, 0 otherwise. Ranker naturally handles variable candidate lists and class imbalance.

- Small model: ~50 trees, depth 4, lr 0.1. We have ~3k rows.
- Leave-one-video-out cross-validation. Primary metric: per-event top-1 accuracy on held-out video.
- Save model to `app/models/position_selector.lgb` and feature schema to `app/models/position_selector_features.json`.
- `tools/train_position_selector.py` handles dataset load, CV, fit, dump, and writes a Markdown report.

**Fallback:** if LightGBM ranker is flakey on small data, switch to binary classifier over (event-candidate) with argmax within event. Decision at training time, not design time.

### 4.5 Step 4 — Flag-gated integration (~0.5 day)

In `fusion_engine.py`:

- Add `FusionConfig.use_learned_position_selector: bool = False`.
- When true, replace the current argmax-over-heuristic-score with an argmax over model scores, using the same candidate list and features emitted in step 1.
- If model's top score margin over the heuristic pick is below a tunable `learned_margin_threshold`, fall back to the heuristic. This limits damage when the model is out-of-distribution.
- No other heuristic is changed (pre-filter, chord limiter, post-filter, melodic-segment correction, slide correction all stay).

### 4.6 Step 5 — Validation & decision (~0.5 day)

Use `ab_anchor.py` subprocess-per-video to A/B heuristic vs learned on all 20 videos. Run the model trained leave-one-out per video (the one that did *not* see it in training).

**Ship gate:**

- Mean exact F1 improvement **≥ +5 points** across the 20 training videos.
- **No single video regresses by more than 3 points** exact F1.
- Original 11-video suite: mean exact F1 does not drop by more than 1 point.
- Runtime per video does not increase by more than 50 ms.

If it ships, flip the default to `True` in a separate PR once we've watched it for a few days.

If it misses the gate, the error-analysis CSV tells us exactly which events the model still gets wrong — the data becomes our next set of priorities (stronger features, different candidate generation, or a different model architecture).

## 5. Success criteria (summary)

| Metric | Current | Ship gate |
|---|---|---|
| Training-set exact F1 (mean) | 0.43 | ≥ 0.48 |
| Training-set worst-video regression | n/a | > -0.03 forbidden |
| Original 11-video exact F1 (mean) | 0.92 | > 0.91 |
| Per-video runtime overhead | 0 | < 50 ms |

## 6. Risks & mitigations

- **Data too small (~3k rows, 20 groups).** LightGBM + lambdarank is conservative; leave-one-out CV keeps us honest. If overfitting, drop features, lower tree count.
- **Feature leakage.** `is_heuristic_pick` and `heuristic_score` are included intentionally — we want the model to *beat* the heuristic, not clone it. Inspect SHAP / gain for over-reliance; regularize or drop if it becomes a parrot.
- **Benchmark drift.** All evaluation uses subprocess-per-video. No full-batch `run_benchmarks.py`.
- **OOD at inference.** Margin-based fallback to the heuristic bounds downside.
- **Labeling noise from imperfect pitch alignment.** Restrict labels to events with high-confidence pitch match (tight time + ±0 semitone). Keep the dataset small and clean rather than large and noisy.

## 7. Open questions

- Do we want a separate model for chord events vs single-note events? Probably not in v1 — let the `is_chord` feature handle it, revisit if CV shows bimodal error.
- Is step 0's bucketing sufficient, or do we need a sub-category for "position wrong because anchor was wrong"? Start simple; drill in only if step 0 says so.

## 8. Out of scope for this plan (next bets)

Ranked by expected value if the learned fusion ships:

1. Audio-side pitch F1 (currently 75%). CREPE or Basic Pitch v2 for monophonic runs; evaluate on pitch F1 alone before touching fusion.
2. Direct-string-from-video CNN (small hand-crop classifier). Adds a strong feature for the learned fusion.
3. Onset-from-video for timing-only misses.
