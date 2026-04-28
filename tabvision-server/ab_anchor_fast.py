"""Fast A/B comparison — subprocess per video, but each subprocess runs the
expensive pipeline ONCE and fusion TWICE (flag off and flag on).

Compared to naive subprocess-per-config:
  - 20 videos × 2 configs = 40 subprocess runs × (pipeline + fusion)  ≈ 80 min
  - This script: 20 subprocesses × (pipeline + 2 fusions)             ≈ 40 min

No batch drift (each video is a fresh subprocess) and no wasted pipeline reruns.

Usage:
    python ab_anchor_fast.py                # all training videos
    python ab_anchor_fast.py --id training-09
"""
import argparse
import json
import os
import subprocess
import sys

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_training_ids() -> list[str]:
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        return [b['id'] for b in json.load(f)['benchmarks']
                if b['id'].startswith('training-')]


# This is the worker script executed by each subprocess — runs pipeline once,
# fusion twice, prints both results as JSON.
WORKER = r'''
import json
import os
import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app.audio_pipeline import AudioAnalysisConfig, analyze_pitch, detect_note_onsets
from app.fretboard_detection import detect_fretboard_from_video
from app.fusion_engine import FusionConfig, fuse_audio_only, fuse_audio_video
from app.video_pipeline import VideoAnalysisConfig, analyze_video_at_timestamps
from evaluate_transcription import (
    evaluate_accuracy, extract_audio, get_video_duration, parse_ground_truth_tabs,
)

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
REPO_ROOT = os.path.dirname(os.path.abspath(os.getcwd()))


def main(bm_id):
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        index = json.load(f)
    bm = next(b for b in index['benchmarks'] if b['id'] == bm_id)
    defaults = index.get('defaults', {})

    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    gt_path = bm['ground_truth_path']
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(REPO_ROOT, gt_path)

    with open(gt_path) as f:
        ground_truth = parse_ground_truth_tabs(f.read())

    # Expensive: audio + video analysis once.
    audio_path = '/tmp/ab_fast_audio.wav'
    try:
        extract_audio(video_path, audio_path)
        detected_notes = analyze_pitch(audio_path, AudioAnalysisConfig())
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    timestamps = detect_note_onsets(detected_notes)
    fretboard = None
    video_observations = {}
    if timestamps:
        fretboard = detect_fretboard_from_video(video_path, num_sample_frames=5)
        if fretboard:
            video_observations = analyze_video_at_timestamps(
                video_path, timestamps, VideoAnalysisConfig()
            )

    def run_fusion(cfg):
        dr = len(video_observations) / len(timestamps) if timestamps else 0
        if fretboard and fretboard.detection_confidence > 0.5 and dr > 0.2:
            return fuse_audio_video(detected_notes, video_observations,
                                    fretboard, capo_fret=0, config=cfg)
        return fuse_audio_only(detected_notes, capo_fret=0, config=cfg)

    video_duration = bm.get('video_duration') or get_video_duration(video_path)
    time_tol = bm.get('time_tolerance', defaults.get('time_tolerance', 0.6))
    bpm = bm.get('bpm') or defaults.get('bpm')

    out = {}
    for label, cfg in [('off', FusionConfig(use_video_hand_anchor=False)),
                       ('on',  FusionConfig(use_video_hand_anchor=True))]:
        notes = run_fusion(cfg)
        m = evaluate_accuracy(
            notes, ground_truth,
            time_tolerance=time_tol,
            video_duration=video_duration,
            bpm=bpm,
            auto_align=True,
        ).to_dict()
        out[label] = {
            'exact_f1': m['exact']['f1'],
            'pitch_f1': m['pitch']['f1'],
            'pos_acc':  m['position']['accuracy'],
            'note_count': len(notes),
        }

    print('@@AB_RESULT@@' + json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1])
'''


def run_video(bm_id: str) -> dict | None:
    """Invoke the worker in a fresh subprocess; parse its JSON output."""
    cmd = [sys.executable, '-c', WORKER, bm_id]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return {'error': r.stderr[-500:]}
    for line in r.stdout.splitlines():
        if line.startswith('@@AB_RESULT@@'):
            return json.loads(line[len('@@AB_RESULT@@'):])
    return {'error': 'no result marker in output'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='Run only this benchmark id')
    args = parser.parse_args()

    ids = [args.id] if args.id else load_training_ids()

    print(f"{'ID':<16}  {'Off exact/pos':<20}  {'On exact/pos':<20}  Δexact   Δpos")
    print('-' * 82)
    deltas_e, deltas_p = [], []
    for bm_id in ids:
        print(f'  [{bm_id}]', end=' ', flush=True)
        r = run_video(bm_id)
        if not r or 'error' in r:
            print(f'ERROR: {r.get("error", "no result")[:200] if r else "timed out"}')
            continue
        off, on = r['off'], r['on']
        de = on['exact_f1'] - off['exact_f1']
        dp = on['pos_acc'] - off['pos_acc']
        deltas_e.append(de); deltas_p.append(dp)
        mark_e = ' ++' if de > 0.02 else (' !!' if de < -0.02 else '   ')
        mark_p = ' ++' if dp > 0.02 else (' !!' if dp < -0.02 else '   ')
        print(f'{off["exact_f1"]:>6.3f}/{off["pos_acc"]:>6.3f}    '
              f'{on["exact_f1"]:>6.3f}/{on["pos_acc"]:>6.3f}    '
              f'{de:+.3f}{mark_e}  {dp:+.3f}{mark_p}')

    if deltas_e:
        print('-' * 82)
        print(f'{"AVG DELTA":<16}  {"":<20}  {"":<20}  '
              f'{sum(deltas_e)/len(deltas_e):+.3f}    '
              f'{sum(deltas_p)/len(deltas_p):+.3f}')


if __name__ == '__main__':
    main()
