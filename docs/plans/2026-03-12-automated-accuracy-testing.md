# Automated Accuracy Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build automated testing infrastructure to catch accuracy regressions without manual review — pytest regression tests, multi-video benchmark suite, and end-to-end server integration tests.

**Architecture:** Three layers: (A) pytest unit tests that run `evaluate_transcription` logic against known inputs and assert F1/precision/recall thresholds, (B) a benchmark runner script that evaluates all video/tab pairs in a `benchmarks/` directory and outputs a comparison report, (C) an integration test that POSTs a video to the Flask server, polls for completion, and evaluates the returned TabDocument. Ground truth tab files live alongside test videos in `tests/fixtures/benchmarks/`.

**Tech Stack:** pytest, Flask test client, existing `evaluate_transcription.py` functions

---

### Task 1: Create benchmark fixtures directory and ground truth index

**Files:**
- Create: `tabvision-server/tests/fixtures/benchmarks/index.json`
- Create: `tabvision-server/tests/fixtures/benchmarks/README.md`

**Step 1: Create benchmark index file**

The index maps video files to their ground truth tabs, expected metrics baselines, and metadata. Videos are referenced by path relative to the repo root (since they're large and gitignored).

```json
{
  "benchmarks": [
    {
      "id": "sample-video",
      "video_path": "sample-video.mp4",
      "ground_truth_path": "tests/fixtures/benchmarks/sample-video-tabs.txt",
      "video_duration": 13.28,
      "description": "Simple fingerpicking passage, open chords and barre shapes",
      "baselines": {
        "exact_f1_min": 0.88,
        "exact_precision_min": 0.85,
        "exact_recall_min": 0.85,
        "pitch_f1_min": 0.90,
        "max_detected_notes": 50,
        "min_detected_notes": 30
      }
    }
  ],
  "defaults": {
    "time_tolerance": 0.6,
    "audio_only": true
  }
}
```

**Step 2: Copy ground truth tab file into fixtures**

```bash
cp /home/gilhooleyp/projects/tab_vision/sample-video-tabs.txt \
   tabvision-server/tests/fixtures/benchmarks/sample-video-tabs.txt
```

**Step 3: Commit**

```bash
git add tabvision-server/tests/fixtures/benchmarks/
git commit -m "feat: add benchmark fixtures directory with ground truth index"
```

---

### Task 2: Refactor evaluate_transcription for import use

**Files:**
- Modify: `tabvision-server/evaluate_transcription.py`

The existing `evaluate_transcription.py` already has good importable functions (`parse_ground_truth_tabs`, `evaluate_accuracy`, `run_transcription`, `get_video_duration`). We need one small addition: a helper that loads a benchmark entry and runs evaluation, returning metrics — so tests and the benchmark script can share logic.

**Step 1: Write the failing test for the helper**

Create `tabvision-server/tests/test_evaluation.py`:

```python
"""Tests for evaluation helpers."""
import pytest
from evaluate_transcription import (
    parse_ground_truth_tabs,
    evaluate_accuracy,
    EvalMetrics,
)
from app.fusion_engine import TabNote


class TestParseGroundTruth:
    def test_parses_simple_tab(self):
        tabs = (
            "e|---5---|\n"
            "B|-------|\n"
            "G|-------|\n"
            "D|-------|\n"
            "A|-------|\n"
            "E|-------|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 1
        assert notes[0]['string'] == 1
        assert notes[0]['fret'] == 5

    def test_parses_chord(self):
        tabs = (
            "e|0--|\n"
            "B|1--|\n"
            "G|0--|\n"
            "D|2--|\n"
            "A|3--|\n"
            "E|---|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 5
        # All notes at same beat position
        beats = {n['beat'] for n in notes}
        assert len(beats) == 1

    def test_parses_two_digit_fret(self):
        tabs = (
            "e|--12--|\n"
            "B|------|\n"
            "G|------|\n"
            "D|------|\n"
            "A|------|\n"
            "E|------|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 1
        assert notes[0]['fret'] == 12

    def test_bar_lines_dont_add_time(self):
        """Bar lines are visual separators, not time markers."""
        tabs = (
            "e|--5--|--7--|\n"
            "B|-----|-----|\n"
            "G|-----|-----|\n"
            "D|-----|-----|\n"
            "A|-----|-----|\n"
            "E|-----|-----|\n"
        )
        notes = parse_ground_truth_tabs(tabs)
        assert len(notes) == 2
        # Second note should be 4 dashes later (1.0 beat), not reset by bar line
        assert notes[1]['beat'] - notes[0]['beat'] == pytest.approx(1.0, abs=0.01)


class TestEvaluateAccuracy:
    def test_perfect_match(self):
        detected = [TabNote(timestamp=1.0, string=1, fret=5, confidence=0.9)]
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        detected[0].midi_note = 69
        metrics = evaluate_accuracy(detected, ground_truth, time_tolerance=0.5, video_duration=4.0)
        assert metrics.exact_f1 == 1.0

    def test_no_detections(self):
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        metrics = evaluate_accuracy([], ground_truth, time_tolerance=0.5, video_duration=4.0)
        assert metrics.exact_recall == 0.0
        assert metrics.exact_fn == 1

    def test_false_positive(self):
        detected = [TabNote(timestamp=1.0, string=1, fret=5, confidence=0.9)]
        detected[0].midi_note = 69
        metrics = evaluate_accuracy(detected, [], time_tolerance=0.5, video_duration=4.0)
        assert metrics.exact_precision == 0.0
        assert metrics.exact_fp == 1

    def test_wrong_position_same_pitch(self):
        """Same MIDI note but different string/fret should be pitch match, not exact."""
        detected = [TabNote(timestamp=1.0, string=2, fret=10, confidence=0.9)]
        detected[0].midi_note = 69  # A4
        ground_truth = [{'string': 1, 'fret': 5, 'beat': 1.0, 'midi_note': 69}]
        metrics = evaluate_accuracy(detected, ground_truth, time_tolerance=0.5, video_duration=4.0)
        assert metrics.exact_tp == 0
        assert metrics.pitch_tp == 1
```

**Step 2: Run tests to verify they pass**

```bash
cd tabvision-server && python -m pytest tests/test_evaluation.py -v
```

These should all pass since they test existing functions. If `TabNote` constructor differs, adjust accordingly.

**Step 3: Commit**

```bash
git add tests/test_evaluation.py
git commit -m "test: add unit tests for evaluation helpers"
```

---

### Task 3: Pytest regression test with threshold assertions

**Files:**
- Create: `tabvision-server/tests/test_regression.py`

This is the core automated accuracy test. It runs the full transcription pipeline against each benchmark video and asserts metrics meet baseline thresholds. Marked `slow` so it can be skipped in fast CI.

**Step 1: Write the regression test**

```python
"""Regression tests for transcription accuracy.

These tests run the full pipeline against benchmark videos with known ground truth
and assert that accuracy metrics meet minimum thresholds. They catch regressions
when pipeline code changes.

Run with: pytest tests/test_regression.py -v
Skip in fast CI with: pytest -m "not slow"
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate_transcription import (
    parse_ground_truth_tabs,
    evaluate_accuracy,
    run_transcription,
    get_video_duration,
    EvalMetrics,
)

# Path to benchmark fixtures
BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'benchmarks')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def load_benchmark_index():
    """Load benchmark index and return list of benchmark configs."""
    index_path = os.path.join(BENCHMARKS_DIR, 'index.json')
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        data = json.load(f)
    return data.get('benchmarks', []), data.get('defaults', {})


def get_benchmark_ids():
    """Get benchmark IDs for pytest parametrize."""
    index_path = os.path.join(BENCHMARKS_DIR, 'index.json')
    if not os.path.exists(index_path):
        return []
    with open(index_path) as f:
        data = json.load(f)
    return [b['id'] for b in data.get('benchmarks', [])]


def run_benchmark(benchmark: dict, defaults: dict) -> tuple[EvalMetrics, list]:
    """Run a single benchmark and return (metrics, tab_notes)."""
    video_path = os.path.join(REPO_ROOT, benchmark['video_path'])
    gt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        benchmark['ground_truth_path']
    )

    if not os.path.exists(video_path):
        pytest.skip(f"Video not found: {video_path}")

    # Load ground truth
    with open(gt_path) as f:
        ground_truth = parse_ground_truth_tabs(f.read())

    # Run pipeline
    audio_only = benchmark.get('audio_only', defaults.get('audio_only', True))
    tab_notes = run_transcription(video_path, audio_only=audio_only)

    # Evaluate
    video_duration = benchmark.get('video_duration') or get_video_duration(video_path)
    time_tolerance = benchmark.get('time_tolerance', defaults.get('time_tolerance', 0.6))
    metrics = evaluate_accuracy(
        tab_notes, ground_truth,
        time_tolerance=time_tolerance,
        video_duration=video_duration,
    )
    return metrics, tab_notes


@pytest.mark.slow
class TestAccuracyRegression:
    """Accuracy regression tests against benchmark videos."""

    @pytest.fixture(scope="class")
    def benchmark_data(self):
        """Load benchmarks once for all tests in this class."""
        benchmarks, defaults = load_benchmark_index()
        if not benchmarks:
            pytest.skip("No benchmarks defined in index.json")

        results = {}
        for bm in benchmarks:
            video_path = os.path.join(REPO_ROOT, bm['video_path'])
            if not os.path.exists(video_path):
                continue
            metrics, tab_notes = run_benchmark(bm, defaults)
            results[bm['id']] = {
                'metrics': metrics,
                'tab_notes': tab_notes,
                'benchmark': bm,
            }
        return results

    def test_exact_f1_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_f1 = baseline.get('exact_f1_min', 0.0)
            assert metrics.exact_f1 >= min_f1, (
                f"[{bm_id}] Exact F1 regressed: {metrics.exact_f1:.3f} < {min_f1} "
                f"(TP={metrics.exact_tp}, FP={metrics.exact_fp}, FN={metrics.exact_fn})"
            )

    def test_precision_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_prec = baseline.get('exact_precision_min', 0.0)
            assert metrics.exact_precision >= min_prec, (
                f"[{bm_id}] Precision regressed: {metrics.exact_precision:.3f} < {min_prec}"
            )

    def test_recall_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_rec = baseline.get('exact_recall_min', 0.0)
            assert metrics.exact_recall >= min_rec, (
                f"[{bm_id}] Recall regressed: {metrics.exact_recall:.3f} < {min_rec}"
            )

    def test_pitch_f1_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_f1 = baseline.get('pitch_f1_min', 0.0)
            assert metrics.pitch_f1 >= min_f1, (
                f"[{bm_id}] Pitch F1 regressed: {metrics.pitch_f1:.3f} < {min_f1}"
            )

    def test_note_count_in_range(self, benchmark_data):
        """Detect note explosion or collapse."""
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            n_detected = len(data['tab_notes'])
            max_notes = baseline.get('max_detected_notes', 999)
            min_notes = baseline.get('min_detected_notes', 0)
            assert min_notes <= n_detected <= max_notes, (
                f"[{bm_id}] Note count out of range: {n_detected} "
                f"(expected {min_notes}-{max_notes})"
            )

    def test_no_f1_regression_summary(self, benchmark_data):
        """Print a summary table after all benchmarks (always passes)."""
        print("\n" + "=" * 70)
        print("REGRESSION TEST SUMMARY")
        print("=" * 70)
        print(f"{'Benchmark':<20} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Notes':>8}")
        print("-" * 70)
        for bm_id, data in benchmark_data.items():
            m = data['metrics']
            print(
                f"{bm_id:<20} {m.exact_f1:>8.1%} {m.exact_precision:>8.1%} "
                f"{m.exact_recall:>8.1%} {len(data['tab_notes']):>8}"
            )
        print("=" * 70)
```

**Step 2: Add slow marker to pytest config**

Add to `tabvision-server/pytest.ini` (or `pyproject.toml` if it exists):

```ini
[pytest]
markers =
    slow: marks tests as slow (run full pipeline against video files)
```

**Step 3: Run regression test**

```bash
cd tabvision-server && python -m pytest tests/test_regression.py -v -s
```

Expected: All tests pass with current metrics (F1~91.4%).

**Step 4: Verify skipping slow tests works**

```bash
python -m pytest tests/ -v -m "not slow"
```

Expected: Regression tests are skipped, other tests run.

**Step 5: Commit**

```bash
git add tests/test_regression.py pytest.ini
git commit -m "feat: add pytest regression tests with accuracy threshold assertions"
```

---

### Task 4: Multi-video benchmark runner script

**Files:**
- Create: `tabvision-server/run_benchmarks.py`

A standalone script that runs all benchmarks, prints a comparison table, and optionally saves/diffs against a baseline JSON file. Useful for A/B comparisons when tuning parameters.

**Step 1: Write the benchmark runner**

```python
"""Run all benchmarks and report accuracy metrics.

Usage:
    python run_benchmarks.py                    # Run all, print table
    python run_benchmarks.py --save baseline    # Save results as baseline
    python run_benchmarks.py --diff baseline    # Compare against saved baseline
    python run_benchmarks.py --audio-only       # Force audio-only mode
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from evaluate_transcription import (
    parse_ground_truth_tabs,
    evaluate_accuracy,
    run_transcription,
    get_video_duration,
    print_metrics,
)

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
RESULTS_DIR = os.path.join('tests', 'fixtures', 'benchmarks', 'results')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_index():
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    return data['benchmarks'], data.get('defaults', {})


def run_all_benchmarks(audio_only_override=None, verbose=False):
    benchmarks, defaults = load_index()
    results = {}

    for bm in benchmarks:
        video_path = os.path.join(REPO_ROOT, bm['video_path'])
        gt_path = os.path.join(bm['ground_truth_path'])

        if not os.path.exists(video_path):
            print(f"SKIP {bm['id']}: video not found at {video_path}")
            continue

        print(f"\n--- Running benchmark: {bm['id']} ---")

        with open(gt_path) as f:
            ground_truth = parse_ground_truth_tabs(f.read())

        audio_only = audio_only_override if audio_only_override is not None else \
            bm.get('audio_only', defaults.get('audio_only', True))
        tab_notes = run_transcription(video_path, audio_only=audio_only)

        video_duration = bm.get('video_duration') or get_video_duration(video_path)
        time_tolerance = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))

        metrics = evaluate_accuracy(
            tab_notes, ground_truth,
            time_tolerance=time_tolerance,
            video_duration=video_duration,
        )

        results[bm['id']] = {
            'metrics': metrics.to_dict(),
            'note_count': len(tab_notes),
            'ground_truth_count': len(ground_truth),
        }

        if verbose:
            print_metrics(metrics, label=bm['id'])

    return results


def print_summary_table(results):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<20} {'ExactF1':>8} {'Prec':>8} {'Rec':>8} "
          f"{'PitchF1':>8} {'PosAcc':>8} {'Notes':>8}")
    print("-" * 80)
    for bm_id, res in results.items():
        m = res['metrics']
        print(f"{bm_id:<20} {m['exact']['f1']:>8.1%} {m['exact']['precision']:>8.1%} "
              f"{m['exact']['recall']:>8.1%} {m['pitch']['f1']:>8.1%} "
              f"{m['position']['accuracy']:>8.1%} {res['note_count']:>8}")
    print("=" * 80)


def print_diff(current, baseline):
    print("\n" + "=" * 80)
    print("BENCHMARK DIFF (current vs baseline)")
    print("=" * 80)
    print(f"{'Benchmark':<20} {'ExactF1':>10} {'Prec':>10} {'Rec':>10} {'Notes':>10}")
    print("-" * 80)

    any_regression = False
    for bm_id, cur in current.items():
        if bm_id not in baseline:
            print(f"{bm_id:<20} {'NEW':>10}")
            continue

        base = baseline[bm_id]
        cm, bm = cur['metrics']['exact'], base['metrics']['exact']

        f1_diff = cm['f1'] - bm['f1']
        prec_diff = cm['precision'] - bm['precision']
        rec_diff = cm['recall'] - bm['recall']
        note_diff = cur['note_count'] - base['note_count']

        def fmt(val):
            sign = "+" if val >= 0 else ""
            marker = " !!" if val < -0.02 else ""
            return f"{sign}{val:>.1%}{marker}"

        def fmt_int(val):
            sign = "+" if val >= 0 else ""
            return f"{sign}{val}"

        print(f"{bm_id:<20} {fmt(f1_diff):>10} {fmt(prec_diff):>10} "
              f"{fmt(rec_diff):>10} {fmt_int(note_diff):>10}")

        if f1_diff < -0.02:
            any_regression = True

    print("=" * 80)
    if any_regression:
        print("WARNING: F1 regression detected (>2% drop marked with !!)")
    else:
        print("OK: No significant regressions detected")


def main():
    parser = argparse.ArgumentParser(description='Run transcription benchmarks')
    parser.add_argument('--save', type=str, help='Save results with this label')
    parser.add_argument('--diff', type=str, help='Diff against saved results with this label')
    parser.add_argument('--audio-only', action='store_true', help='Force audio-only mode')
    parser.add_argument('--verbose', action='store_true', help='Print detailed metrics per benchmark')
    args = parser.parse_args()

    audio_only = True if args.audio_only else None
    results = run_all_benchmarks(audio_only_override=audio_only, verbose=args.verbose)
    print_summary_table(results)

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"{args.save}.json")
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {save_path}")

    if args.diff:
        diff_path = os.path.join(RESULTS_DIR, f"{args.diff}.json")
        if not os.path.exists(diff_path):
            print(f"ERROR: Baseline not found: {diff_path}")
            sys.exit(1)
        with open(diff_path) as f:
            baseline = json.load(f)['results']
        print_diff(results, baseline)


if __name__ == '__main__':
    main()
```

**Step 2: Test the benchmark runner**

```bash
cd tabvision-server && python run_benchmarks.py --verbose --save initial
```

Expected: Runs sample-video benchmark, prints metrics, saves to `tests/fixtures/benchmarks/results/initial.json`.

**Step 3: Test diff mode**

```bash
python run_benchmarks.py --diff initial
```

Expected: Shows +0.0% across all metrics (comparing against itself).

**Step 4: Commit**

```bash
git add run_benchmarks.py
git commit -m "feat: add multi-video benchmark runner with save/diff support"
```

---

### Task 5: End-to-end server integration test

**Files:**
- Create: `tabvision-server/tests/test_e2e_accuracy.py`

This test POSTs a real video to the Flask test client, polls until complete, then evaluates the returned TabDocument against ground truth. Tests the full HTTP path including serialization.

**Step 1: Write the e2e test**

```python
"""End-to-end accuracy test via Flask server.

Posts a video to POST /jobs, polls GET /jobs/:id until complete,
fetches GET /jobs/:id/result, and evaluates against ground truth.

Run with: pytest tests/test_e2e_accuracy.py -v -s
"""
import json
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app
from evaluate_transcription import (
    parse_ground_truth_tabs,
    evaluate_accuracy,
    EvalMetrics,
)
from app.fusion_engine import TabNote
from app.guitar_mapping import STANDARD_TUNING

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'benchmarks')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _tab_document_to_tab_notes(tab_document: dict) -> list[TabNote]:
    """Convert a TabDocument JSON (from server response) to TabNote objects."""
    tab_notes = []
    for note_data in tab_document.get('notes', []):
        fret = note_data['fret']
        string = note_data['string']
        note = TabNote(
            timestamp=note_data['timestamp'],
            string=string,
            fret=fret,
            confidence=note_data.get('confidence', 0.5),
        )
        # Compute MIDI note
        if fret != 'X' and fret != 'x':
            open_midi = STANDARD_TUNING.get(string)
            if open_midi is not None:
                note.midi_note = open_midi + int(fret)
        tab_notes.append(note)
    return tab_notes


@pytest.fixture
def app_client(tmp_path):
    app = create_app()
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.test_client() as client:
        yield client


def load_benchmark_index():
    index_path = os.path.join(BENCHMARKS_DIR, 'index.json')
    with open(index_path) as f:
        return json.load(f)


@pytest.mark.slow
@pytest.mark.e2e
class TestEndToEndAccuracy:
    """Test full server pipeline: upload -> process -> evaluate."""

    def test_server_transcription_accuracy(self, app_client):
        """POST video, poll for completion, evaluate result against ground truth."""
        data = load_benchmark_index()
        benchmarks = data['benchmarks']
        defaults = data.get('defaults', {})

        for bm in benchmarks:
            video_path = os.path.join(REPO_ROOT, bm['video_path'])
            if not os.path.exists(video_path):
                pytest.skip(f"Video not found: {video_path}")

            gt_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                bm['ground_truth_path']
            )
            with open(gt_path) as f:
                ground_truth = parse_ground_truth_tabs(f.read())

            # POST video
            with open(video_path, 'rb') as vf:
                response = app_client.post('/jobs', data={
                    'video': (vf, os.path.basename(video_path)),
                    'capo_fret': '0',
                }, content_type='multipart/form-data')

            assert response.status_code == 201, f"Upload failed: {response.get_json()}"
            job_id = response.get_json()['job_id']

            # Poll for completion (timeout after 120s)
            deadline = time.time() + 120
            while time.time() < deadline:
                status_resp = app_client.get(f'/jobs/{job_id}')
                assert status_resp.status_code == 200
                status = status_resp.get_json()['status']

                if status == 'completed':
                    break
                elif status == 'failed':
                    pytest.fail(f"[{bm['id']}] Job failed: {status_resp.get_json()}")

                time.sleep(1)
            else:
                pytest.fail(f"[{bm['id']}] Job timed out after 120s")

            # Fetch result
            result_resp = app_client.get(f'/jobs/{job_id}/result')
            assert result_resp.status_code == 200
            tab_document = result_resp.get_json()

            # Convert to TabNotes and evaluate
            tab_notes = _tab_document_to_tab_notes(tab_document)
            video_duration = bm.get('video_duration') or 13.28
            time_tolerance = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))

            metrics = evaluate_accuracy(
                tab_notes, ground_truth,
                time_tolerance=time_tolerance,
                video_duration=video_duration,
            )

            baselines = bm.get('baselines', {})
            print(f"\n[{bm['id']}] E2E Results: "
                  f"F1={metrics.exact_f1:.1%} P={metrics.exact_precision:.1%} "
                  f"R={metrics.exact_recall:.1%} ({len(tab_notes)} notes)")

            # Assert baselines (with 5% slack for server-vs-direct differences)
            slack = 0.05
            min_f1 = baselines.get('exact_f1_min', 0.0) - slack
            assert metrics.exact_f1 >= min_f1, (
                f"[{bm['id']}] E2E F1 too low: {metrics.exact_f1:.3f} < {min_f1:.3f}"
            )
```

**Step 2: Add e2e marker to pytest config**

Update `tabvision-server/pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (run full pipeline against video files)
    e2e: marks end-to-end tests (require running server)
```

**Step 3: Run the e2e test**

```bash
cd tabvision-server && python -m pytest tests/test_e2e_accuracy.py -v -s
```

Expected: Video uploads, job processes, result is evaluated, F1 meets threshold.

Note: The Flask test client runs in the same process, but `process_job` is launched in a Thread via the route handler. The poll loop waits for completion.

**Step 4: Commit**

```bash
git add tests/test_e2e_accuracy.py pytest.ini
git commit -m "feat: add end-to-end server accuracy integration test"
```

---

### Task 6: Add convenience test commands

**Files:**
- Modify: `tabvision-server/pytest.ini`

**Step 1: Final pytest.ini**

```ini
[pytest]
markers =
    slow: marks tests as slow (run full pipeline against video files)
    e2e: marks end-to-end tests (require running server)
testpaths = tests
```

**Step 2: Verify all test tiers work**

```bash
# Fast unit tests only (skip slow/e2e)
cd tabvision-server && python -m pytest tests/ -v -m "not slow"

# Regression tests only
python -m pytest tests/test_regression.py -v -s

# E2E only
python -m pytest tests/test_e2e_accuracy.py -v -s

# Everything
python -m pytest tests/ -v -s
```

**Step 3: Final commit**

```bash
git add pytest.ini
git commit -m "chore: finalize pytest config with test tier markers"
```
