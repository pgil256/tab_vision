"""Subprocess-per-video error-bucket harness.

Parent: python tools/error_analysis.py [--id training-01]
  -> runs each video in a fresh subprocess (--worker), aggregates.

Worker: python tools/error_analysis.py --worker --id <bm_id>
  -> loads ground truth + transcribes + classifies, prints one JSON line.

Outputs:
  tools/outputs/errors-YYYY-MM-DD_HHMMSS.csv  -- per-event rows
  tools/outputs/errors-YYYY-MM-DD_HHMMSS.md   -- aggregate + per-video table

Originally from feature/audio-finetune. Ported with the auto-align +
muted_undetectable fixes living in app/error_analyzer.py.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime

# Ensure tabvision-server/ is on sys.path so `app.*` + evaluate_transcription import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
OUTPUTS_DIR = os.path.join('tools', 'outputs')
# __file__ = tabvision-server/tools/error_analysis.py
# dirname^1 = tabvision-server/tools, dirname^2 = tabvision-server,
# dirname^3 = repo root (where test-data/ lives).
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_training_ids() -> list[str]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    return [b['id'] for b in data['benchmarks']
            if b['id'].startswith('training-')]


def load_benchmark(bm_id: str) -> tuple[dict, dict]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    for b in data['benchmarks']:
        if b['id'] == bm_id:
            return b, data.get('defaults', {})
    raise SystemExit(f'unknown benchmark id: {bm_id}')


def run_worker(bm_id: str) -> dict:
    """In-process: run transcription + classify. Returns {video_id, events[]}."""
    from evaluate_transcription import (
        parse_ground_truth_tabs, run_transcription, get_video_duration,
    )
    from app.error_analyzer import classify_events

    bm, defaults = load_benchmark(bm_id)

    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    gt_path = bm['ground_truth_path']
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(REPO_ROOT, gt_path)

    with open(gt_path) as f:
        ground_truth = parse_ground_truth_tabs(f.read())

    audio_only = bm.get('audio_only', defaults.get('audio_only', True))
    detected = run_transcription(video_path, audio_only=audio_only)

    duration = bm.get('video_duration') or get_video_duration(video_path)
    tol = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))
    bpm = bm.get('bpm') or defaults.get('bpm')

    events = classify_events(detected, ground_truth,
                             time_tolerance=tol, bpm=bpm,
                             video_duration=duration, video_id=bm_id)
    return {'video_id': bm_id, 'events': [dataclasses.asdict(e) for e in events]}


def dispatch_subprocess(bm_id: str) -> dict:
    """Parent: run a fresh subprocess for this video."""
    r = subprocess.run(
        [sys.executable, __file__, '--worker', '--id', bm_id],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0:
        return {'video_id': bm_id, 'error': r.stderr[-500:]}
    line = next(l for l in reversed(r.stdout.splitlines()) if l.strip())
    return json.loads(line)


def write_markdown(path: str, per_video: dict[str, dict[str, int]]):
    from app.error_analyzer import BUCKETS
    agg = {b: 0 for b in BUCKETS}
    rows = []
    for vid in sorted(per_video):
        counts = per_video[vid]
        for b in BUCKETS:
            agg[b] += counts.get(b, 0)
        total = sum(counts.values()) or 1
        rows.append((vid, counts, total))

    total_agg = sum(agg.values()) or 1
    n_correct = agg.get('correct', 0)
    n_muted = agg.get('muted_undetectable', 0)
    # Recoverable loss: pitched GT events that didn't match exactly,
    # plus detection-side extras.
    pitched_gt = (sum(agg.get(b, 0) for b in
                      ('correct', 'wrong_position_same_pitch',
                       'pitch_off', 'timing_only', 'missed_onset')))
    recoverable_loss = pitched_gt - n_correct + agg.get('extra_detection', 0)

    with open(path, 'w') as f:
        f.write(f'# Error analysis — {datetime.now().isoformat(timespec="seconds")}\n\n')
        f.write('## Aggregate\n\n')
        f.write('| bucket | count | share | share-of-loss |\n')
        f.write('|---|---:|---:|---:|\n')
        for b in BUCKETS:
            share = agg[b] / total_agg if total_agg else 0
            if b in ('correct', 'muted_undetectable'):
                loss_share = '—'
            else:
                loss_share = (f'{agg[b] / recoverable_loss:.1%}'
                              if recoverable_loss else '—')
            f.write(f'| {b} | {agg[b]} | {share:.1%} | {loss_share} |\n')
        f.write(f'\nTotal classified events: **{total_agg}** '
                f'(pitched GT: {pitched_gt}, muted/X: {n_muted}, '
                f'recoverable loss: {recoverable_loss})\n\n')

        f.write('## Per-video\n\n')
        header = '| video | ' + ' | '.join(BUCKETS) + ' | total |\n'
        sep = '|---' * (len(BUCKETS) + 2) + '|\n'
        f.write(header)
        f.write(sep)
        for vid, counts, total in rows:
            cells = ' | '.join(str(counts.get(b, 0)) for b in BUCKETS)
            f.write(f'| {vid} | {cells} | {total} |\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true',
                    help='run one video in-process and emit JSON to stdout')
    ap.add_argument('--id', help='benchmark id')
    args = ap.parse_args()

    if args.worker:
        if not args.id:
            ap.error('--worker requires --id')
        print(json.dumps(run_worker(args.id)))
        return

    ids = [args.id] if args.id else load_training_ids()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    csv_path = os.path.join(OUTPUTS_DIR, f'errors-{stamp}.csv')
    md_path = os.path.join(OUTPUTS_DIR, f'errors-{stamp}.md')

    all_rows: list[dict] = []
    per_video: dict[str, dict[str, int]] = {}

    for bm_id in ids:
        print(f'[{bm_id}] running...', flush=True)
        res = dispatch_subprocess(bm_id)
        if 'error' in res:
            print(f'[{bm_id}] ERROR: {res["error"]}', flush=True)
            continue
        all_rows.extend(res['events'])
        counts: dict[str, int] = {}
        for e in res['events']:
            counts[e['bucket']] = counts.get(e['bucket'], 0) + 1
        per_video[bm_id] = counts
        print(f'[{bm_id}] {counts}', flush=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f'wrote {csv_path} ({len(all_rows)} rows)')

    write_markdown(md_path, per_video)
    print(f'wrote {md_path}')


if __name__ == '__main__':
    main()
