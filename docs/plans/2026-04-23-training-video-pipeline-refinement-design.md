# Training Video Pipeline Refinement

**Date**: 2026-04-23
**Goal**: 85%+ avg exact F1 across 20 training videos
**Primary benchmark**: 20 training videos (replace original 11 as primary)
**Secondary benchmark**: Original 11 videos (regression gate only)

## Context

20 training videos recorded on electric guitar with ground-truth ASCII tabs.
Categories: position ambiguity (01-05), chord varieties (06-10), single-note passages (11-15), edge cases (16-20).

Training videos are 90-95% accurate to tabs (notes mostly correct, timing approximate). Ground-truth tabs will be corrected via hybrid approach: run benchmark, flag biggest discrepancies, only fix tabs where pipeline is clearly right and tab is wrong.

## Phase 1: Audio-Only Baseline & Diagnostics

1. Run `python run_benchmarks.py --save training_baseline --verbose` on all 20 training videos
2. Extract per-video: pitch F1, exact F1, position accuracy. The pitch-to-exact gap reveals how much is pitch detection vs position assignment.
3. Aggregate by category (01-05, 06-10, 11-15, 16-20) to identify weakest areas
4. Generate discrepancy report for ground-truth review: list specific mismatches where pipeline output diverges from tabs

**Deliverable**: Diagnostic summary with per-video and per-category accuracy, plus discrepancy report for manual tab correction.

## Phase 2: Audio-Only Optimization Loop

Iterate until 85%+ or diminishing returns:

### A. Pick target
Prioritize by:
1. High pitch F1, low exact F1 (position errors, fixable in fusion engine)
2. Low pitch F1 (pitch detection issues, audio pipeline tuning)
3. Note count ratio far from 1.0 (too many or too few notes detected)

### B. Diagnose failure
Run problem video with `--verbose`. Look for:
- Wrong string, right pitch (position scoring weights)
- Phantom notes (harmonic/sustain filter thresholds)
- Missing notes (filters too aggressive)
- Chord fragmentation (onset detection / grouping)

### C. Apply fix
Either parameter tune (thresholds, weights) or algorithm fix (new correction pass, modify existing one).

### D. Verify
Re-run full benchmark. Accept only if:
- Target video improves
- No regression >2% on any other training video
- Original 11 videos stay stable

### E. Save checkpoint
`python run_benchmarks.py --save tuning_vN` after each accepted change.

## Phase 3: Video Fusion

**Entry criteria**: Audio-only avg F1 has plateaued, remaining errors dominated by position ambiguity.

1. Enable video: flip `audio_only` to `false` in index.json defaults for training videos
2. Assess fretboard detection on electric guitar: edge detection, Hough transform, neck geometry assumptions may need tuning
3. Tune video for position resolution: focus on training 01-05 (position ambiguity set), check if video observations help or hurt
4. Final regression check on original 11 videos with video enabled

**Deliverable**: Final benchmark at 85%+ avg exact F1, saved as named checkpoint.
