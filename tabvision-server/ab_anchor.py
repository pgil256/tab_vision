"""A/B comparison runner — each video runs in a fresh subprocess to avoid batch drift.

The full-batch `run_benchmarks.py` run accumulates state across 20 videos and
produces different numbers per-video than running the same video in isolation.
This script launches each video in its own subprocess so comparisons are clean.

Usage:
    python ab_anchor.py              # all training videos
    python ab_anchor.py --id training-09  # single video
"""
import argparse
import json
import os
import re
import subprocess
import sys

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
BENCHMARK_RUNNER = 'run_benchmarks.py'


def load_training_ids() -> list[str]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        return [b['id'] for b in json.load(f)['benchmarks']
                if b['id'].startswith('training-')]


def run_one(bm_id: str, anchor_on: bool) -> dict:
    """Run one benchmark in a fresh subprocess, return parsed metrics."""
    cmd = [
        sys.executable, BENCHMARK_RUNNER,
        '--with-video', '--id', bm_id,
    ]
    if anchor_on:
        cmd.append('--use-video-hand-anchor')

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {'error': r.stderr[:500]}

    for line in r.stdout.splitlines():
        m = re.match(rf'^{re.escape(bm_id)}\s+(\S+)%\s+(\S+)%\s+(\S+)%\s+(\S+)%\s+(\S+)%', line)
        if m:
            return {
                'exact_f1': float(m.group(1)) / 100.0,
                'precision': float(m.group(2)) / 100.0,
                'recall': float(m.group(3)) / 100.0,
                'pitch_f1': float(m.group(4)) / 100.0,
                'pos_acc': float(m.group(5)) / 100.0,
            }
    return {'error': 'metrics line not found'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='Run only this benchmark id')
    args = parser.parse_args()

    ids = [args.id] if args.id else load_training_ids()

    print(f"{'ID':<16}  {'Off exact / pos':<22}  {'On exact / pos':<22}  Δexact   Δpos")
    print('-' * 86)
    deltas_e, deltas_p = [], []
    for bm_id in ids:
        r_off = run_one(bm_id, anchor_on=False)
        r_on = run_one(bm_id, anchor_on=True)
        if 'error' in r_off or 'error' in r_on:
            print(f'{bm_id:<16}  ERROR: {r_off.get("error") or r_on.get("error")}')
            continue
        de = r_on['exact_f1'] - r_off['exact_f1']
        dp = r_on['pos_acc'] - r_off['pos_acc']
        deltas_e.append(de); deltas_p.append(dp)
        mark_e = ' ++' if de > 0.02 else (' !!' if de < -0.02 else '   ')
        mark_p = ' ++' if dp > 0.02 else (' !!' if dp < -0.02 else '   ')
        print(f'{bm_id:<16}  '
              f'{r_off["exact_f1"]:>6.3f} / {r_off["pos_acc"]:>6.3f}      '
              f'{r_on["exact_f1"]:>6.3f} / {r_on["pos_acc"]:>6.3f}      '
              f'{de:+.3f}{mark_e}  {dp:+.3f}{mark_p}')

    if deltas_e:
        print('-' * 86)
        print(f'{"AVG DELTA":<16}  {"":<22}  {"":<22}  {sum(deltas_e)/len(deltas_e):+.3f}'
              f'    {sum(deltas_p)/len(deltas_p):+.3f}')


if __name__ == '__main__':
    main()
