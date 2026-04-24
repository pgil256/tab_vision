"""Layer 2 diagnostic: anchor quality vs. ground-truth frets.

For each benchmark video, runs the video pipeline, builds the hand-anchor
timeline, and compares the anchor's value at each ground-truth note's
timestamp against the note's fret. Reports MAE, coverage, and |err| ≤ 2
fraction per video.

This is intentionally a read-only diagnostic — it does NOT change tabs
produced by fusion. It only measures whether the video-derived anchor is
accurate enough to be worth using as the primary hand-position signal.

Usage:
    python debug_hand_anchor.py                      # all benchmarks
    python debug_hand_anchor.py --id training-01     # single benchmark
    python debug_hand_anchor.py --id-prefix training # all training-*
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app.audio_pipeline import analyze_pitch, AudioAnalysisConfig, detect_note_onsets
from app.fretboard_detection import detect_fretboard_from_video
from app.hand_anchor import build_hand_position_timeline, get_hand_anchor_at
from app.video_pipeline import analyze_video_at_timestamps, VideoAnalysisConfig
from evaluate_transcription import (
    extract_audio, parse_ground_truth_tabs, get_video_duration,
    _find_best_time_offset,
)

BENCHMARKS_DIR = os.path.join('tests', 'fixtures', 'benchmarks')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_index():
    with open(os.path.join(BENCHMARKS_DIR, 'index.json')) as f:
        return json.load(f)['benchmarks']


def diagnose_one(bm: dict, verbose: bool = False) -> dict | None:
    """Run the diagnostic for one benchmark, returning per-video stats."""
    video_path = os.path.join(REPO_ROOT, bm['video_path'])
    gt_path = bm['ground_truth_path']
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(REPO_ROOT, gt_path)

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        return None

    with open(gt_path) as f:
        ground_truth = parse_ground_truth_tabs(f.read())
    if not ground_truth:
        return None

    # Audio — we need onset timestamps to sample video.
    audio_path = '/tmp/anchor_audio.wav'
    try:
        extract_audio(video_path, audio_path)
        detected = analyze_pitch(audio_path, AudioAnalysisConfig())
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    timestamps = detect_note_onsets(detected)
    if not timestamps:
        return None

    fretboard = detect_fretboard_from_video(video_path, num_sample_frames=5)
    if not fretboard:
        return {'skipped': 'no fretboard'}

    video_observations = analyze_video_at_timestamps(
        video_path, timestamps, VideoAnalysisConfig()
    )
    if not video_observations:
        return {'skipped': 'no video observations'}

    timeline = build_hand_position_timeline(video_observations, fretboard)
    if not timeline:
        return {
            'skipped': 'empty timeline',
            'observation_count': len(video_observations),
            'fretboard_confidence': fretboard.detection_confidence,
        }

    if verbose:
        print(f"  fretboard: starting_fret={fretboard.starting_fret} "
              f"frets_detected={len(fretboard.fret_positions)} "
              f"actual_fret_numbers={fretboard.actual_fret_numbers} "
              f"conf={fretboard.detection_confidence:.2f}")
        print(f"  timeline head:")
        for p in timeline[:5]:
            print(f"    t={p.timestamp:.2f} fret={p.anchor_fret:.2f} conf={p.confidence:.2f}")

    # GT notes are dicts with `beat` — convert to timestamps via bpm + auto-align.
    bpm = bm.get('bpm')
    video_duration = bm.get('video_duration') or get_video_duration(video_path)
    total_beats = max(n['beat'] for n in ground_truth)
    if bpm and bpm > 0:
        beat_to_time = 60.0 / bpm
    else:
        beat_to_time = video_duration / total_beats if total_beats > 0 else 1
    time_offset = 0.0
    if bpm and bpm > 0 and detected:
        # _find_best_time_offset expects objects with .timestamp — adapt DetectedNotes
        class _Adapter:
            __slots__ = ('timestamp', 'midi_note')
            def __init__(self, t, m):
                self.timestamp = t
                self.midi_note = m
        adapters = [_Adapter(n.start_time, n.midi_note) for n in detected]
        time_offset = _find_best_time_offset(
            adapters,
            [n['beat'] for n in ground_truth],
            [n.get('midi_note') for n in ground_truth],
            beat_to_time, video_duration, 0.5,
        )

    errors: list[float] = []
    covered = 0
    total_pitched = 0
    for gt in ground_truth:
        fret = gt.get('fret')
        if fret is None or fret == 'X':
            continue
        total_pitched += 1
        ts = gt['beat'] * beat_to_time + time_offset
        anchor_fret, conf = get_hand_anchor_at(timeline, ts, max_gap=0.3)
        if anchor_fret is None:
            continue
        covered += 1
        errors.append(abs(anchor_fret - float(fret)))

    if not errors:
        return {
            'coverage': 0.0,
            'total_pitched': total_pitched,
            'note': 'no anchor overlap with GT timestamps',
            'timeline_points': len(timeline),
        }

    mae = sum(errors) / len(errors)
    within_2 = sum(1 for e in errors if e <= 2) / len(errors)
    within_4 = sum(1 for e in errors if e <= 4) / len(errors)

    return {
        'mae': mae,
        'within_2': within_2,
        'within_4': within_4,
        'coverage': covered / total_pitched if total_pitched else 0.0,
        'timeline_points': len(timeline),
        'total_pitched': total_pitched,
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose hand-anchor quality')
    parser.add_argument('--id', type=str, help='Run only this benchmark id')
    parser.add_argument('--id-prefix', type=str,
                        help='Run benchmarks whose id starts with this prefix')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    benchmarks = load_index()
    if args.id:
        benchmarks = [b for b in benchmarks if b['id'] == args.id]
    elif args.id_prefix:
        benchmarks = [b for b in benchmarks if b['id'].startswith(args.id_prefix)]

    results: dict[str, dict] = {}
    for bm in benchmarks:
        print(f"\n--- {bm['id']} ---")
        try:
            r = diagnose_one(bm, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[bm['id']] = {'error': str(e)}
            continue
        if r is None:
            print("  SKIP: missing video/GT")
            continue
        if 'skipped' in r:
            print(f"  SKIP: {r['skipped']}")
            results[bm['id']] = r
            continue
        results[bm['id']] = r
        print(f"  timeline points: {r['timeline_points']}")
        print(f"  coverage:        {r['coverage']:.1%} ({r.get('total_pitched', 0)} pitched GT notes)")
        if 'mae' in r:
            print(f"  anchor MAE:      {r['mae']:.2f} frets")
            print(f"  within 2 frets:  {r['within_2']:.1%}")
            print(f"  within 4 frets:  {r['within_4']:.1%}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':<22} {'points':>8} {'cov':>8} {'MAE':>8} {'±2':>8} {'±4':>8}")
    print("-" * 70)
    maes, covs, w2s = [], [], []
    for bm_id, r in results.items():
        if 'mae' in r:
            print(f"{bm_id:<22} {r['timeline_points']:>8} {r['coverage']:>7.1%} "
                  f"{r['mae']:>8.2f} {r['within_2']:>7.1%} {r['within_4']:>7.1%}")
            maes.append(r['mae'])
            covs.append(r['coverage'])
            w2s.append(r['within_2'])
        else:
            print(f"{bm_id:<22}   {r.get('skipped', r.get('error', 'no data'))}")
    print("-" * 70)
    if maes:
        print(f"{'AVERAGE':<22} {'':>8} {sum(covs)/len(covs):>7.1%} "
              f"{sum(maes)/len(maes):>8.2f} {sum(w2s)/len(w2s):>7.1%}  n={len(maes)}")


if __name__ == '__main__':
    main()
