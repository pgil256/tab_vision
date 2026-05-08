# Training Video Pipeline Refinement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reach 85%+ avg exact F1 on 20 training videos by running a diagnostic benchmark, then iterating with parameter tuning and targeted algorithm fixes.

**Architecture:** Three-phase approach: (1) establish audio-only baseline on all 20 training videos, build diagnostic tooling; (2) iterative optimization loop analyzing error patterns and applying fixes with regression protection; (3) enable video fusion to resolve remaining position ambiguity. The existing `run_benchmarks.py` and `evaluate_transcription.py` handle scoring and diffs; we add a diagnostic analysis script on top.

**Tech Stack:** Python, Basic Pitch, existing fusion engine (`tabvision-server/app/fusion_engine.py`), existing audio pipeline (`tabvision-server/app/audio_pipeline.py`), existing benchmark runner (`tabvision-server/run_benchmarks.py`)

---

## Task 1: Run Audio-Only Baseline on All 20 Training Videos

**Files:**
- Read: `tabvision-server/run_benchmarks.py`
- Read: `tabvision-server/tests/fixtures/benchmarks/index.json`
- Output: `tabvision-server/tests/fixtures/benchmarks/results/training_baseline.json`

**Step 1: Verify all 20 videos are accessible**

Run from `tabvision-server/`:
```bash
cd tabvision-server && python -c "
import json, os
with open('tests/fixtures/benchmarks/index.json') as f:
    data = json.load(f)
repo = os.path.dirname(os.path.abspath('.'))
missing = []
for bm in data['benchmarks']:
    vp = os.path.join('..', bm['video_path'])
    gp = bm['ground_truth_path']
    if not os.path.isabs(gp):
        gp = os.path.join('..', gp)
    if not os.path.exists(vp):
        missing.append(f\"{bm['id']}: video {vp}\")
    if not os.path.exists(gp):
        missing.append(f\"{bm['id']}: gt {gp}\")
if missing:
    print('MISSING:')
    for m in missing:
        print(f'  {m}')
else:
    print(f'All {len(data[\"benchmarks\"])} benchmarks have video + ground truth files')
"
```

Expected: "All 21 benchmarks have video + ground truth files" (20 training + 1 sample-video)

**Step 2: Run the full benchmark suite and save as baseline**

Run from `tabvision-server/`:
```bash
cd tabvision-server && python run_benchmarks.py --save training_baseline --audio-only
```

Expected: Summary table printed for all 21 benchmarks. Results saved to `tests/fixtures/benchmarks/results/training_baseline.json`.

**Step 3: Record the baseline numbers**

Copy the summary table output. Key numbers to capture:
- Per-video: exact F1, pitch F1, position accuracy, note count ratio
- Average exact F1 across training-01 through training-20
- Average exact F1 for the original sample-video

**Step 4: Commit baseline**

```bash
git add tests/fixtures/benchmarks/results/training_baseline.json
git commit -m "benchmark: save audio-only baseline for 20 training videos"
```

---

## Task 2: Build Diagnostic Analysis Script

**Files:**
- Create: `tabvision-server/analyze_benchmarks.py`
- Test: manual run (script is a diagnostic tool, not production code)

**Step 1: Write the diagnostic script**

Create `tabvision-server/analyze_benchmarks.py` that reads a benchmark results JSON and produces:

1. **Per-category breakdown**: Average exact F1, pitch F1, position accuracy grouped by:
   - Position Ambiguity (training-01 to training-05)
   - Chord Varieties (training-06 to training-10)
   - Single-Note Passages (training-11 to training-15)
   - Edge Cases (training-16 to training-20)

2. **Error classification per video**: For each video, classify the dominant error:
   - "pitch_detection" if pitch F1 < 0.7
   - "position_assignment" if pitch F1 >= 0.7 but exact F1 < pitch F1 * 0.7
   - "note_count" if detected/gt ratio is < 0.7 or > 1.5
   - "mixed" otherwise

3. **Discrepancy report**: For each video, list up to 10 near-misses with:
   - Expected string+fret vs detected string+fret
   - MIDI note
   - Whether it's a position error or pitch error

4. **Priority ranking**: Sort videos by "improvement potential" = pitch F1 - exact F1 (biggest gap = most position errors to fix)

Usage:
```bash
python analyze_benchmarks.py tests/fixtures/benchmarks/results/training_baseline.json
```

**Step 2: Run analysis on baseline results**

```bash
cd tabvision-server && python analyze_benchmarks.py tests/fixtures/benchmarks/results/training_baseline.json
```

Expected: Categorized report showing which categories and videos to target first.

**Step 3: Commit**

```bash
git add analyze_benchmarks.py
git commit -m "feat: add benchmark diagnostic analysis script"
```

---

## Task 3: Review Discrepancy Report and Fix Ground-Truth Tabs

**Files:**
- Modify: `test-data/training-tabs/training-XX-tabs.txt` (whichever need correction)

**Step 1: Present discrepancy report to user**

After running the analysis script, present the discrepancy report for user review. For each flagged mismatch, show:
- The video ID and timestamp
- What the pipeline detected (string + fret)
- What the ground-truth tab says (string + fret)
- The MIDI note (to verify pitch correctness)

**Step 2: User decides which tabs to correct**

User reviews and identifies cases where they actually played something different from the written tab. These tabs get corrected.

**Step 3: Re-run benchmark with corrected tabs**

```bash
cd tabvision-server && python run_benchmarks.py --save post_gt_correction --audio-only
python run_benchmarks.py --diff training_baseline
```

Expected: Some videos improve simply from ground-truth correction. Diff shows which changed.

**Step 4: Commit corrected tabs**

```bash
git add test-data/training-tabs/training-*-tabs.txt
git commit -m "fix: correct ground-truth tabs based on discrepancy analysis"
```

---

## Task 4: Optimization Cycle — Iteration Template

This task repeats for each optimization cycle. Each cycle targets one error pattern.

**Files:**
- Modify: `tabvision-server/app/audio_pipeline.py` (for pitch detection issues)
- Modify: `tabvision-server/app/fusion_engine.py` (for position assignment issues)
- Test: `tabvision-server/tests/test_note_filtering.py` or new test file
- Output: `tabvision-server/tests/fixtures/benchmarks/results/tuning_vN.json`

**Step 1: Identify target from analysis**

Pick the video or category with the lowest exact F1 where pitch F1 is reasonable (>0.7). Run that specific video with verbose output:

```bash
cd tabvision-server && python run_benchmarks.py --id training-XX --verbose --audio-only
```

Examine the near-misses to identify the dominant error pattern.

**Step 2: Write a failing test (if algorithm change)**

If the fix is an algorithm change (not just a parameter tweak), write a test in `tabvision-server/tests/` that captures the expected behavior:

```python
def test_specific_error_pattern():
    # Setup: create notes that trigger the error pattern
    # Assert: the fix produces correct output
```

Run to verify it fails:
```bash
cd tabvision-server && pytest tests/test_SPECIFIC.py::test_specific_error_pattern -v
```

**Step 3: Apply the fix**

Either:
- **Parameter change**: Modify the config dataclass defaults in `audio_pipeline.py:38` (`AudioAnalysisConfig`) or `fusion_engine.py:57` (`FusionConfig`)
- **Algorithm change**: Modify the relevant function in `fusion_engine.py` (position selection at line 705, chord optimization, melodic correction at line 1235, slide correction at line 1019, pre-filter at line 173, post-filter at line 336)

**Step 4: Verify test passes (if applicable)**

```bash
cd tabvision-server && pytest tests/test_SPECIFIC.py::test_specific_error_pattern -v
```

**Step 5: Run full benchmark and diff**

```bash
cd tabvision-server && python run_benchmarks.py --save tuning_vN --audio-only
python run_benchmarks.py --diff tuning_vPREV
```

Accept if:
- Target video/category improved
- No regression >2% on any other training video
- Run sample-video separately to check original benchmark: `python run_benchmarks.py --id sample-video --audio-only`

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(accuracy): describe specific improvement"
```

**Repeat** from Step 1 with the next worst-performing video/category.

---

## Task 5: Enable Video Fusion (Phase 3)

**Entry criteria:** Audio-only avg F1 has plateaued. Remaining errors are dominated by position ambiguity (high pitch F1, lower exact F1).

**Files:**
- Modify: `tabvision-server/tests/fixtures/benchmarks/index.json` (flip audio_only default)
- Possibly modify: `tabvision-server/app/fretboard_detection.py` (electric guitar tuning)
- Possibly modify: `tabvision-server/app/video_pipeline.py` (electric guitar tuning)

**Step 1: Run video-enabled benchmark on a few test videos**

Start with 2-3 videos from the position ambiguity category:
```bash
cd tabvision-server && python run_benchmarks.py --id training-01
cd tabvision-server && python run_benchmarks.py --id training-03
cd tabvision-server && python run_benchmarks.py --id training-05
```

Note: without `--audio-only` flag, these will use the default from index.json. Temporarily edit index.json to set `audio_only: false` for just these videos, or modify the entries individually.

**Step 2: Check fretboard detection quality**

Look at the output for "fretboard detection confidence" and "video detection rate". If these are low on electric guitar, the video pipeline needs tuning before it can help.

**Step 3: Tune video pipeline for electric guitar**

If detection quality is poor, adjust parameters in:
- `tabvision-server/app/fretboard_detection.py`: edge detection thresholds, Hough transform params
- `tabvision-server/app/video_pipeline.py`: MediaPipe confidence thresholds, finger mapping

**Step 4: Enable video for all training videos**

Once video detection is reliable:
```json
// index.json defaults
"audio_only": false
```

**Step 5: Run full benchmark with video**

```bash
cd tabvision-server && python run_benchmarks.py --save final_with_video
python run_benchmarks.py --diff tuning_vLAST
```

**Step 6: Regression check on original videos**

```bash
cd tabvision-server && python run_benchmarks.py --id sample-video
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(accuracy): enable video fusion for training benchmarks"
```

---

## Task 6: Final Validation and Cleanup

**Step 1: Run final full benchmark**

```bash
cd tabvision-server && python run_benchmarks.py --save final --verbose
```

**Step 2: Verify 85%+ target**

Check the average exact F1 across training-01 through training-20. Must be >= 0.85.

**Step 3: Update baseline minimums in index.json**

For each training video, set `f1_min` baselines based on achieved scores (with some margin):
```json
"baselines": {
    "f1_min": <achieved_f1 - 0.05>,
    "pitch_f1_min": <achieved_pitch_f1 - 0.05>
}
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(accuracy): achieve 85%+ avg F1 on training video benchmark"
```
