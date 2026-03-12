"""Regression tests for transcription accuracy.

These tests run the full pipeline against benchmark videos with known ground truth
and assert that accuracy metrics meet minimum thresholds. They catch regressions
when pipeline code changes.

Run with: pytest tests/test_regression.py -v -s
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
        return [], {}
    with open(index_path) as f:
        data = json.load(f)
    return data.get('benchmarks', []), data.get('defaults', {})


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
            min_f1 = baseline.get('f1_min', 0.0)
            assert metrics.exact_f1 >= min_f1, (
                f"[{bm_id}] Exact F1 regressed: {metrics.exact_f1:.3f} < {min_f1} "
                f"(TP={metrics.exact_tp}, FP={metrics.exact_fp}, FN={metrics.exact_fn})"
            )

    def test_precision_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_prec = baseline.get('precision_min', 0.0)
            assert metrics.exact_precision >= min_prec, (
                f"[{bm_id}] Precision regressed: {metrics.exact_precision:.3f} < {min_prec}"
            )

    def test_recall_above_baseline(self, benchmark_data):
        for bm_id, data in benchmark_data.items():
            baseline = data['benchmark']['baselines']
            metrics = data['metrics']
            min_rec = baseline.get('recall_min', 0.0)
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

    def test_regression_summary(self, benchmark_data):
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
