"""Run all benchmarks and report accuracy metrics.

Usage:
    python run_benchmarks.py                    # Run all, print table
    python run_benchmarks.py --save baseline    # Save results as baseline
    python run_benchmarks.py --diff baseline    # Compare against saved baseline
    python run_benchmarks.py --audio-only       # Force audio-only mode
    python run_benchmarks.py --id sample-video  # Run single benchmark
    python run_benchmarks.py --verbose          # Print per-note detail
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
    EvalMetrics,
)

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
RESULTS_DIR = os.path.join('tests', 'fixtures', 'benchmarks', 'results')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_index():
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    return data['benchmarks'], data.get('defaults', {})


def run_single_benchmark(bm: dict, defaults: dict,
                         audio_only_override=None,
                         verbose=False) -> dict | None:
    """Run one benchmark entry. Returns result dict or None if video missing."""
    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    gt_path = bm['ground_truth_path']

    # Ground truth path may be absolute or relative to repo root
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(REPO_ROOT, gt_path)

    if not os.path.exists(video_path):
        print(f"  SKIP {bm['id']}: video not found at {video_path}")
        return None

    if not os.path.exists(gt_path):
        print(f"  SKIP {bm['id']}: ground truth not found at {gt_path}")
        return None

    with open(gt_path) as f:
        ground_truth = parse_ground_truth_tabs(f.read())

    audio_only = audio_only_override if audio_only_override is not None else \
        bm.get('audio_only', defaults.get('audio_only', True))

    tab_notes = run_transcription(video_path, audio_only=audio_only)

    video_duration = bm.get('video_duration') or get_video_duration(video_path)
    time_tolerance = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))

    bpm = bm.get('bpm') or defaults.get('bpm')
    metrics = evaluate_accuracy(
        tab_notes, ground_truth,
        time_tolerance=time_tolerance,
        video_duration=video_duration,
        bpm=bpm,
        auto_align=True,
    )

    if verbose:
        print_metrics(metrics, label=bm['id'])

    return {
        'metrics': metrics.to_dict(),
        'note_count': len(tab_notes),
        'ground_truth_count': len(ground_truth),
        'metrics_obj': metrics,
    }


def run_all_benchmarks(audio_only_override=None, verbose=False,
                       filter_id=None) -> dict:
    """Run all benchmarks, return results dict keyed by benchmark id."""
    benchmarks, defaults = load_index()
    results = {}

    for bm in benchmarks:
        if filter_id and bm['id'] != filter_id:
            continue
        print(f"\n--- {bm['id']} ({bm.get('description', '')}) ---")
        result = run_single_benchmark(bm, defaults,
                                      audio_only_override=audio_only_override,
                                      verbose=verbose)
        if result:
            results[bm['id']] = result

    return results


def print_summary_table(results: dict):
    """Print a formatted summary table of all benchmark results."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Benchmark':<22} {'ExactF1':>8} {'Prec':>8} {'Rec':>8} "
          f"{'PitchF1':>8} {'PosAcc':>8} {'Det':>6} {'GT':>6}")
    print("-" * 90)

    f1_vals = []
    for bm_id, res in results.items():
        m = res['metrics']
        exact = m['exact']
        pitch = m['pitch']
        pos = m['position']
        f1_vals.append(exact['f1'])
        print(f"{bm_id:<22} {exact['f1']:>8.1%} {exact['precision']:>8.1%} "
              f"{exact['recall']:>8.1%} {pitch['f1']:>8.1%} "
              f"{pos['accuracy']:>8.1%} {res['note_count']:>6} {res['ground_truth_count']:>6}")

    print("=" * 90)
    if f1_vals:
        avg_f1 = sum(f1_vals) / len(f1_vals)
        min_f1 = min(f1_vals)
        max_f1 = max(f1_vals)
        print(f"{'AVERAGE':<22} {avg_f1:>8.1%}   "
              f"(min {min_f1:.1%}  max {max_f1:.1%}  n={len(f1_vals)})")
    print("=" * 90)


def print_diff(current: dict, baseline: dict):
    """Print a diff table comparing current results to a saved baseline."""
    print("\n" + "=" * 80)
    print("BENCHMARK DIFF  (current vs baseline)")
    print("=" * 80)
    print(f"{'Benchmark':<22} {'ExactF1':>10} {'Prec':>10} {'Rec':>10} {'Notes':>10}")
    print("-" * 80)

    any_regression = False
    for bm_id, cur in current.items():
        if bm_id not in baseline:
            print(f"{bm_id:<22} {'NEW':>10}")
            continue

        base = baseline[bm_id]
        cm, bm_m = cur['metrics']['exact'], base['metrics']['exact']
        f1_d = cm['f1'] - bm_m['f1']
        p_d = cm['precision'] - bm_m['precision']
        r_d = cm['recall'] - bm_m['recall']
        note_d = cur['note_count'] - base['note_count']

        def fmt(val):
            sign = "+" if val >= 0 else ""
            marker = " !!" if val < -0.02 else ("  +" if val > 0.02 else "")
            return f"{sign}{val:.1%}{marker}"

        print(f"{bm_id:<22} {fmt(f1_d):>10} {fmt(p_d):>10} {fmt(r_d):>10} "
              f"{('+' if note_d >= 0 else '') + str(note_d):>10}")
        if f1_d < -0.02:
            any_regression = True

    print("=" * 80)
    if any_regression:
        print("WARNING: F1 regression detected (>2% drop marked with !!)")
    else:
        print("OK: No significant regressions detected")


def main():
    parser = argparse.ArgumentParser(description='Run transcription benchmarks')
    parser.add_argument('--save', type=str, help='Save results with this label')
    parser.add_argument('--diff', type=str,
                        help='Diff against saved results with this label')
    parser.add_argument('--audio-only', action='store_true',
                        help='Force audio-only mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed metrics per benchmark')
    parser.add_argument('--id', type=str,
                        help='Run only a specific benchmark by id')
    args = parser.parse_args()

    audio_only = True if args.audio_only else None
    results = run_all_benchmarks(
        audio_only_override=audio_only,
        verbose=args.verbose,
        filter_id=args.id,
    )
    print_summary_table(results)

    # Strip non-serialisable metrics_obj before saving
    serialisable = {k: {kk: vv for kk, vv in v.items() if kk != 'metrics_obj'}
                    for k, v in results.items()}

    if args.save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"{args.save}.json")
        with open(save_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': serialisable,
            }, f, indent=2)
        print(f"\nResults saved to {save_path}")

    if args.diff:
        diff_path = os.path.join(RESULTS_DIR, f"{args.diff}.json")
        if not os.path.exists(diff_path):
            print(f"ERROR: Baseline not found: {diff_path}")
            sys.exit(1)
        with open(diff_path) as f:
            baseline = json.load(f)['results']
        print_diff(serialisable, baseline)


if __name__ == '__main__':
    main()
