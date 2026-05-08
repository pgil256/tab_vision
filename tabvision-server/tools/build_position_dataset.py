"""Step 2 of the learned-fusion plan
(docs/plans/2026-04-24-learned-fusion-design.md §4.3).

Reads a position-features JSONL produced by tools/dump_position_features.py,
aligns each event to ground truth, emits a labeled parquet dataset for
training the position selector in Step 3.

A row = one candidate. label=1 iff the candidate's (string, fret) matches
the GT note that aligns to the event's (onset_time, midi_note). Events
without a matching GT note are dropped — they're extras / hallucinations
and have no correct candidate to learn against.

Group key: video_id (for leave-one-video-out CV in Step 3).

Usage (from tabvision-server/):
  python tools/build_position_dataset.py
  python tools/build_position_dataset.py --features tools/outputs/position-features-2026-04-29_093154.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

THIS_FILE = os.path.abspath(__file__)
TOOLS_DIR = os.path.dirname(THIS_FILE)
SERVER_DIR = os.path.dirname(TOOLS_DIR)
REPO_ROOT = os.path.dirname(SERVER_DIR)
BENCHMARKS_DIR = os.path.join(SERVER_DIR, 'tests', 'fixtures', 'benchmarks')
OUTPUTS_DIR = os.path.join(TOOLS_DIR, 'outputs')

sys.path.insert(0, SERVER_DIR)


def load_index() -> tuple[dict, dict]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    bm_by_id = {b['id']: b for b in data['benchmarks']}
    return bm_by_id, data.get('defaults', {})


def load_events_grouped(jsonl_path: str) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            by_video.setdefault(ev['video_id'], []).append(ev)
    return by_video


class _DetectedStub:
    """Minimal duck-type for evaluate_transcription._find_best_time_offset."""
    __slots__ = ('timestamp', 'midi_note')

    def __init__(self, t: float, m: int) -> None:
        self.timestamp = t
        self.midi_note = m


def gt_with_aligned_times(bm: dict, defaults: dict,
                          events: list[dict]) -> tuple[list[dict], float]:
    """Load GT, compute beat_to_time, auto-align to detected onsets."""
    from evaluate_transcription import (
        parse_ground_truth_tabs, _find_best_time_offset, get_video_duration,
    )

    gt_path = bm['ground_truth_path']
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(REPO_ROOT, gt_path)
    with open(gt_path) as f:
        gt = parse_ground_truth_tabs(f.read())

    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    video_duration = bm.get('video_duration') or get_video_duration(video_path)
    bpm = bm.get('bpm') or defaults.get('bpm')
    tol = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))

    if not gt:
        return [], tol

    total_beats = max(n['beat'] for n in gt)
    if bpm and bpm > 0:
        beat_to_time = 60.0 / bpm
    else:
        beat_to_time = video_duration / total_beats if total_beats else 1.0

    offset = 0.0
    if bpm and bpm > 0 and events:
        stubs = [_DetectedStub(ev['onset_time'], ev['midi_note']) for ev in events]
        offset = _find_best_time_offset(
            stubs,
            [n['beat'] for n in gt],
            [n.get('midi_note') for n in gt],
            beat_to_time, video_duration, tol,
        )

    aligned = [
        {**n, 'time': n['beat'] * beat_to_time + offset}
        for n in gt
    ]
    return aligned, tol


def _match_event_to_gt(ev: dict, gt_with_time: list[dict],
                       matched_gt: set[int], tol: float) -> Optional[int]:
    """Closest unmatched GT note within tol with matching MIDI."""
    best_gi = None
    best_dt = float('inf')
    for gi, gt in enumerate(gt_with_time):
        if gi in matched_gt:
            continue
        if gt.get('midi_note') is None:
            continue
        if gt['midi_note'] != ev['midi_note']:
            continue
        dt = abs(gt['time'] - ev['onset_time'])
        if dt > tol or dt >= best_dt:
            continue
        best_gi, best_dt = gi, dt
    return best_gi


def label_events(video_id: str, events: list[dict],
                 gt_with_time: list[dict], tol: float) -> list[dict]:
    """Greedy 1:1 match each event to a GT note; emit labeled candidate rows."""
    rows: list[dict] = []
    matched_gt: set[int] = set()
    events_sorted = sorted(events, key=lambda e: e['onset_time'])
    for ev in events_sorted:
        gi = _match_event_to_gt(ev, gt_with_time, matched_gt, tol)
        if gi is None:
            continue
        matched_gt.add(gi)
        gt = gt_with_time[gi]
        gt_fret = gt['fret'] if gt['fret'] != 'X' else None
        for c in ev['candidates']:
            label = int(
                c['cand_string'] == gt['string']
                and c['cand_fret'] == gt['fret']
            )
            rows.append({
                'video_id': video_id,
                'event_id': ev['event_id'],
                'onset_time': ev['onset_time'],
                'midi_note': ev['midi_note'],
                'amplitude': ev['amplitude'],
                'basicpitch_confidence': ev['basicpitch_confidence'],
                'is_chord': ev['is_chord'],
                'chord_size': ev['chord_size'],
                'chord_string_span': ev['chord_string_span'],
                'num_candidates': ev['num_candidates'],
                'prev_position_string': ev['prev_position_string'],
                'prev_position_fret': ev['prev_position_fret'],
                'seconds_since_prev': ev['seconds_since_prev'],
                'hand_anchor_fret': ev['hand_anchor_fret'],
                'video_hand_anchor_fret': ev['video_hand_anchor_fret'],
                'selected_string': ev['selected_string'],
                'selected_fret': ev['selected_fret'],
                'cand_string': c['cand_string'],
                'cand_fret': c['cand_fret'],
                'dist_anchor_fret': c['dist_anchor_fret'],
                'dist_anchor_string': c['dist_anchor_string'],
                'dist_prev_fret': c['dist_prev_fret'],
                'dist_prev_string': c['dist_prev_string'],
                'heuristic_score': c['heuristic_score'],
                'is_heuristic_pick': c['is_heuristic_pick'],
                'gt_string': gt['string'],
                'gt_fret': gt_fret,
                'label': label,
            })
    return rows


def _latest_features_jsonl() -> Optional[str]:
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    candidates = sorted(
        f for f in os.listdir(OUTPUTS_DIR)
        if f.startswith('position-features-') and f.endswith('.jsonl')
    )
    if not candidates:
        return None
    return os.path.join(OUTPUTS_DIR, candidates[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features',
                    help='input position-features JSONL (default: most recent)')
    ap.add_argument('--output',
                    default=os.path.join(OUTPUTS_DIR, 'position_dataset.parquet'),
                    help='output parquet path')
    args = ap.parse_args()

    if args.features is None:
        args.features = _latest_features_jsonl()
        if args.features is None:
            print('no position-features-*.jsonl found in tools/outputs/', file=sys.stderr)
            return 1
        print(f'using features file: {args.features}', file=sys.stderr)

    bm_by_id, defaults = load_index()
    events_by_video = load_events_grouped(args.features)

    all_rows: list[dict] = []
    matched_event_count = 0
    unmatched_event_count = 0
    label_pos = 0

    for vid in sorted(events_by_video):
        events = events_by_video[vid]
        bm = bm_by_id.get(vid)
        if bm is None:
            print(f'[{vid}] no benchmark entry; skipping', file=sys.stderr)
            continue
        gt_with_time, tol = gt_with_aligned_times(bm, defaults, events)
        rows = label_events(vid, events, gt_with_time, tol)
        labeled_event_ids = {r['event_id'] for r in rows}
        matched_event_count += len(labeled_event_ids)
        unmatched_event_count += len(events) - len(labeled_event_ids)
        label_pos += sum(r['label'] for r in rows)
        all_rows.extend(rows)
        with_label_1 = sum(1 for eid in labeled_event_ids
                           if any(r['event_id'] == eid and r['label'] == 1
                                  for r in rows))
        print(f'[{vid}] labeled events: {len(labeled_event_ids)}/{len(events)}, '
              f'with-correct-candidate: {with_label_1}, rows: {len(rows)}',
              file=sys.stderr)

    print(file=sys.stderr)
    print(f'Total labeled events:   {matched_event_count}', file=sys.stderr)
    print(f'Total unmatched events: {unmatched_event_count}', file=sys.stderr)
    print(f'Total candidate rows:   {len(all_rows)}', file=sys.stderr)
    print(f'Positive-label rows:    {label_pos} '
          f'({100.0 * label_pos / len(all_rows):.1f}% of rows)' if all_rows else 'no rows',
          file=sys.stderr)

    if not all_rows:
        return 1

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_parquet(args.output, index=False)
    csv_path = args.output.replace('.parquet', '.csv')
    df.to_csv(csv_path, index=False)
    print(f'wrote {args.output}', file=sys.stderr)
    print(f'wrote {csv_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
