"""Sanity / dataset driver for position-selection feature emission.

Step 1 follow-up: runs the existing pipeline end-to-end on every
training-* benchmark with FusionConfig.emit_position_features=True and
dumps every PositionDecision to a single JSONL file. Subprocess-per-video
mirrors tools/error_analysis.py so we don't get batch drift.

Usage (from tabvision-server/):
  python tools/dump_position_features.py            # all training-*
  python tools/dump_position_features.py --id training-09
  python tools/dump_position_features.py --worker --id training-09
                                                    # internal subprocess mode
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime

THIS_FILE = os.path.abspath(__file__)
TOOLS_DIR = os.path.dirname(THIS_FILE)
SERVER_DIR = os.path.dirname(TOOLS_DIR)
REPO_ROOT = os.path.dirname(SERVER_DIR)
BENCHMARKS_DIR = os.path.join(SERVER_DIR, 'tests', 'fixtures', 'benchmarks')
OUTPUTS_DIR = os.path.join(TOOLS_DIR, 'outputs')

sys.path.insert(0, SERVER_DIR)

import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def load_benchmark(bm_id: str) -> tuple[dict, dict]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    for b in data['benchmarks']:
        if b['id'] == bm_id:
            return b, data.get('defaults', {})
    raise SystemExit(f'unknown benchmark id: {bm_id}')


def load_training_ids() -> list[str]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        data = json.load(f)
    return [b['id'] for b in data['benchmarks']
            if b['id'].startswith('training-')]


def run_worker(bm_id: str) -> dict:
    """Run pipeline with feature emission on. Return events as plain dicts."""
    from evaluate_transcription import run_transcription
    from app.fusion_engine import FusionConfig, PositionDecision  # noqa: F401

    bm, defaults = load_benchmark(bm_id)
    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    audio_only = bm.get('audio_only', defaults.get('audio_only', True))

    config = FusionConfig()
    config.emit_position_features = True
    config._feature_events = []

    run_transcription(video_path, audio_only=audio_only, fusion_config=config)

    events = [dataclasses.asdict(e) for e in config._feature_events]
    return {'video_id': bm_id, 'events': events}


def dispatch_subprocess(bm_id: str) -> dict:
    r = subprocess.run(
        [sys.executable, THIS_FILE, '--worker', '--id', bm_id],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0:
        return {'video_id': bm_id, 'error': r.stderr[-500:]}
    line = next(l for l in reversed(r.stdout.splitlines()) if l.strip())
    return json.loads(line)


def summarize(per_video: dict[str, list[dict]]) -> str:
    lines: list[str] = []
    n_events_total = 0
    n_cands_total = 0
    chord_count = 0
    video_anchor_count = 0
    cand_per_event: list[int] = []
    selected_eq_heuristic = 0
    selected_total = 0
    seconds_since_seen = 0

    for vid in sorted(per_video):
        evs = per_video[vid]
        n_events_total += len(evs)
        for ev in evs:
            n_cands_total += len(ev['candidates'])
            cand_per_event.append(len(ev['candidates']))
            if ev['is_chord']:
                chord_count += 1
            if ev['video_hand_anchor_fret'] is not None:
                video_anchor_count += 1
            if ev['seconds_since_prev'] is not None:
                seconds_since_seen += 1
            picked = next(
                (c for c in ev['candidates'] if c['is_heuristic_pick']), None
            )
            if picked is not None:
                selected_total += 1
                if (ev['selected_string'] == picked['cand_string']
                        and ev['selected_fret'] == picked['cand_fret']):
                    selected_eq_heuristic += 1

    lines.append(f'Videos with events: {len(per_video)}')
    lines.append(f'Total events: {n_events_total}')
    lines.append(f'Total candidate rows: {n_cands_total}')
    if cand_per_event:
        lines.append(
            f'Candidates per event: min={min(cand_per_event)} '
            f'max={max(cand_per_event)} '
            f'mean={sum(cand_per_event) / len(cand_per_event):.1f}'
        )
    lines.append(f'Chord events: {chord_count} '
                 '(expected 0 — chord path not instrumented yet)')
    lines.append(f'Events with video_hand_anchor: {video_anchor_count}')
    lines.append(f'Events with seconds_since_prev set: {seconds_since_seen}')
    if selected_total:
        agree_pct = 100.0 * selected_eq_heuristic / selected_total
        lines.append(
            f'selected matches heuristic pick: {selected_eq_heuristic}/{selected_total} '
            f'({agree_pct:.1f}%) — disagreement = video override'
        )
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', help='one benchmark id; default: all training-*')
    ap.add_argument('--worker', action='store_true',
                    help='internal: run one video, emit JSON to stdout')
    args = ap.parse_args()

    if args.worker:
        if not args.id:
            ap.error('--worker requires --id')
        print(json.dumps(run_worker(args.id)))
        return

    ids = [args.id] if args.id else load_training_ids()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    jsonl_path = os.path.join(OUTPUTS_DIR, f'position-features-{stamp}.jsonl')
    summary_path = os.path.join(OUTPUTS_DIR, f'position-features-{stamp}.md')

    per_video: dict[str, list[dict]] = {}
    with open(jsonl_path, 'w') as out:
        for bm_id in ids:
            print(f'[{bm_id}] running...', flush=True)
            res = dispatch_subprocess(bm_id)
            if 'error' in res:
                print(f'[{bm_id}] ERROR: {res["error"]}', flush=True)
                continue
            evs = res.get('events', [])
            per_video[bm_id] = evs
            for ev in evs:
                out.write(json.dumps({**ev, 'video_id': bm_id}) + '\n')
            print(f'[{bm_id}] {len(evs)} events', flush=True)

    summary = summarize(per_video)
    print('\n' + summary)
    with open(summary_path, 'w') as f:
        f.write(f'# Position features dump — {datetime.now().isoformat(timespec="seconds")}\n\n')
        f.write('```\n' + summary + '\n```\n\n')
        f.write('## Per-video event counts\n\n')
        f.write('| video | events |\n|---|---:|\n')
        for vid in sorted(per_video):
            f.write(f'| {vid} | {len(per_video[vid])} |\n')

    print(f'\nwrote {jsonl_path}')
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
