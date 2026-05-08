"""Diagnostic analysis of benchmark results.

Usage:
    python analyze_benchmarks.py <results_json_path>
"""
import json
import sys

CATEGORIES = {
    "Position Ambiguity": ["training-01", "training-02", "training-03", "training-04", "training-05"],
    "Chord Varieties": ["training-06", "training-07", "training-08", "training-09", "training-10"],
    "Single-Note Passages": ["training-11", "training-12", "training-13", "training-14", "training-15"],
    "Edge Cases": ["training-16", "training-17", "training-18", "training-19", "training-20"],
}


def classify_error(metrics):
    """Classify dominant error type for a video."""
    pitch_f1 = metrics["pitch"]["f1"]
    exact_f1 = metrics["exact"]["f1"]
    ratio = metrics["counts"]["ratio"]

    if ratio < 0.7 or ratio > 1.5:
        return "note_count"
    if pitch_f1 < 0.7:
        return "pitch_detection"
    if exact_f1 < pitch_f1 * 0.7:
        return "position_assignment"
    return "mixed"


def print_category_breakdown(results):
    print("=" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("=" * 80)
    print()
    print(f"{'Category':<25} {'Exact F1':>10} {'Pitch F1':>10} {'Pos Acc':>10} {'Videos':>8}")
    print("-" * 65)

    all_exact, all_pitch, all_pos = [], [], []
    for cat_name, video_ids in CATEGORIES.items():
        exact_f1s, pitch_f1s, pos_accs = [], [], []
        for vid_id in video_ids:
            if vid_id not in results:
                continue
            m = results[vid_id]["metrics"]
            exact_f1s.append(m["exact"]["f1"])
            pitch_f1s.append(m["pitch"]["f1"])
            pos_accs.append(m["position"]["accuracy"])

        if not exact_f1s:
            continue

        avg_exact = sum(exact_f1s) / len(exact_f1s)
        avg_pitch = sum(pitch_f1s) / len(pitch_f1s)
        avg_pos = sum(pos_accs) / len(pos_accs)
        all_exact.extend(exact_f1s)
        all_pitch.extend(pitch_f1s)
        all_pos.extend(pos_accs)

        print(f"{cat_name:<25} {avg_exact:>9.1%} {avg_pitch:>9.1%} {avg_pos:>9.1%} {len(exact_f1s):>8}")

    print("-" * 65)
    if all_exact:
        print(f"{'OVERALL':<25} {sum(all_exact)/len(all_exact):>9.1%} "
              f"{sum(all_pitch)/len(all_pitch):>9.1%} "
              f"{sum(all_pos)/len(all_pos):>9.1%} {len(all_exact):>8}")
    print()


def print_error_classification(results):
    print("=" * 80)
    print("ERROR CLASSIFICATION PER VIDEO")
    print("=" * 80)
    print()
    print(f"{'Video':<15} {'Exact F1':>10} {'Pitch F1':>10} {'Ratio':>8} {'Error Type':<22}")
    print("-" * 67)

    counts = {}
    for vid_id in sorted(results.keys()):
        m = results[vid_id]["metrics"]
        err = classify_error(m)
        counts[err] = counts.get(err, 0) + 1
        print(f"{vid_id:<15} {m['exact']['f1']:>9.1%} {m['pitch']['f1']:>9.1%} "
              f"{m['counts']['ratio']:>7.2f} {err:<22}")

    print()
    print("Error type summary:")
    for err_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {err_type}: {count} videos")
    print()


def print_near_miss_report(results):
    print("=" * 80)
    print("NEAR-MISS DISCREPANCY REPORT (up to 10 per video)")
    print("=" * 80)

    for vid_id in sorted(results.keys()):
        m = results[vid_id]["metrics"]
        near_misses = m.get("near_misses", [])
        if not near_misses:
            continue

        print(f"\n--- {vid_id} (exact F1={m['exact']['f1']:.1%}, "
              f"{len(near_misses)} near misses) ---")
        print(f"  {'#':<4} {'Expected':>12} {'Detected':>12} {'MIDI':>6} {'Type':<16}")

        # Count error types for this video
        type_counts = {}
        for nm in near_misses:
            t = nm["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        for i, nm in enumerate(near_misses[:10]):
            gt = f"s{nm['gt_string']}f{nm['gt_fret']}"
            det = f"s{nm['det_string']}f{nm['det_fret']}"
            print(f"  {i+1:<4} {gt:>12} {det:>12} {nm['midi_note']:>6} {nm['type']:<16}")

        remaining = len(near_misses) - 10
        if remaining > 0:
            print(f"  ... and {remaining} more")

        summary_parts = [f"{count} {t}" for t, count in sorted(type_counts.items(), key=lambda x: -x[1])]
        print(f"  Summary: {', '.join(summary_parts)}")


def print_priority_ranking(results):
    print()
    print("=" * 80)
    print("PRIORITY RANKING (improvement potential = pitch F1 - exact F1)")
    print("=" * 80)
    print()
    print("Videos with high pitch F1 but low exact F1 have the most position errors")
    print("to fix -- the pitch detector found the right notes, but they were placed")
    print("on the wrong string/fret.")
    print()
    print(f"{'Rank':<6} {'Video':<15} {'Pitch F1':>10} {'Exact F1':>10} {'Gap':>10} {'Error Type':<22}")
    print("-" * 75)

    ranked = []
    for vid_id, vid_data in results.items():
        m = vid_data["metrics"]
        gap = m["pitch"]["f1"] - m["exact"]["f1"]
        ranked.append((vid_id, m, gap))

    ranked.sort(key=lambda x: -x[2])

    for rank, (vid_id, m, gap) in enumerate(ranked, 1):
        err = classify_error(m)
        print(f"{rank:<6} {vid_id:<15} {m['pitch']['f1']:>9.1%} {m['exact']['f1']:>9.1%} "
              f"{gap:>9.1%} {err:<22}")

    print()
    print("Top 5 targets for position assignment improvements:")
    for vid_id, m, gap in ranked[:5]:
        pos_correct = m["position"]["correct"]
        pos_total = m["position"]["total"]
        pos_wrong = pos_total - pos_correct
        print(f"  {vid_id}: {pos_wrong}/{pos_total} pitch-matched notes on wrong position "
              f"(pos accuracy {m['position']['accuracy']:.0%})")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_benchmarks.py <results_json_path>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Analyzing {len(results)} benchmark results from {data.get('timestamp', 'unknown')}")
    print()

    print_category_breakdown(results)
    print_error_classification(results)
    print_near_miss_report(results)
    print_priority_ranking(results)


if __name__ == "__main__":
    main()
