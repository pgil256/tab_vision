"""Evaluate transcription against ground truth tabs.

Supports multi-dimensional accuracy metrics:
- Pitch F1: correct MIDI note regardless of string/fret position
- Position accuracy: correct string+fret for correct pitches
- Chord F1: chord-level detection accuracy
- Technique F1: technique annotation accuracy

Usage:
    python evaluate_transcription.py <video_path> <ground_truth_path> [options]

Options:
    --time-tolerance FLOAT    Timing tolerance in seconds (default: 0.5)
    --config CONFIG_JSON      Path to fusion config JSON (optional)
    --output OUTPUT_JSON      Path to save metrics JSON (optional)
    --compare CONFIG_JSON_B   A/B comparison: second config to compare against
    --sweep                   Run tolerance sweep (0.1 to 1.0 in 0.1 steps)
    --audio-only              Run audio-only mode (skip video)
    --verbose                 Print detailed match/miss info
"""
import argparse
import json
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dataclasses import dataclass
from typing import Optional

from app.audio_pipeline import extract_audio, analyze_pitch, AudioAnalysisConfig
from app.fusion_engine import fuse_audio_only, fuse_audio_video, FusionConfig, TabNote
from app.video_pipeline import analyze_video_at_timestamps, VideoAnalysisConfig
from app.fretboard_detection import detect_fretboard_from_video, FretboardDetectionConfig
from app.audio_pipeline import detect_note_onsets
from app.guitar_mapping import STANDARD_TUNING, MAX_FRET


@dataclass
class EvalMetrics:
    """Multi-dimensional evaluation metrics."""
    # Exact match (string + fret + timing)
    exact_tp: int = 0
    exact_fp: int = 0
    exact_fn: int = 0

    # Pitch-only match (correct MIDI note within time tolerance)
    pitch_tp: int = 0
    pitch_fp: int = 0
    pitch_fn: int = 0

    # Position accuracy (for pitch-correct notes, how many have right string+fret)
    position_correct: int = 0
    position_total: int = 0

    # Chord metrics
    chord_tp: int = 0  # chords where all notes matched
    chord_partial: int = 0  # chords where some notes matched
    chord_fp: int = 0  # detected chords not in ground truth
    chord_fn: int = 0  # ground truth chords not detected

    # Counts
    total_ground_truth: int = 0
    total_detected: int = 0

    # Near misses for diagnostics
    near_misses: list = None

    def __post_init__(self):
        if self.near_misses is None:
            self.near_misses = []

    @property
    def exact_precision(self) -> float:
        denom = self.exact_tp + self.exact_fp
        return self.exact_tp / denom if denom > 0 else 0.0

    @property
    def exact_recall(self) -> float:
        denom = self.exact_tp + self.exact_fn
        return self.exact_tp / denom if denom > 0 else 0.0

    @property
    def exact_f1(self) -> float:
        p, r = self.exact_precision, self.exact_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def pitch_precision(self) -> float:
        denom = self.pitch_tp + self.pitch_fp
        return self.pitch_tp / denom if denom > 0 else 0.0

    @property
    def pitch_recall(self) -> float:
        denom = self.pitch_tp + self.pitch_fn
        return self.pitch_tp / denom if denom > 0 else 0.0

    @property
    def pitch_f1(self) -> float:
        p, r = self.pitch_precision, self.pitch_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def position_accuracy(self) -> float:
        return self.position_correct / self.position_total if self.position_total > 0 else 0.0

    @property
    def chord_precision(self) -> float:
        denom = self.chord_tp + self.chord_fp
        return self.chord_tp / denom if denom > 0 else 0.0

    @property
    def chord_recall(self) -> float:
        denom = self.chord_tp + self.chord_fn
        return self.chord_tp / denom if denom > 0 else 0.0

    @property
    def chord_f1(self) -> float:
        p, r = self.chord_precision, self.chord_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'exact': {
                'precision': self.exact_precision,
                'recall': self.exact_recall,
                'f1': self.exact_f1,
                'tp': self.exact_tp,
                'fp': self.exact_fp,
                'fn': self.exact_fn,
            },
            'pitch': {
                'precision': self.pitch_precision,
                'recall': self.pitch_recall,
                'f1': self.pitch_f1,
                'tp': self.pitch_tp,
                'fp': self.pitch_fp,
                'fn': self.pitch_fn,
            },
            'position': {
                'accuracy': self.position_accuracy,
                'correct': self.position_correct,
                'total': self.position_total,
            },
            'chord': {
                'precision': self.chord_precision,
                'recall': self.chord_recall,
                'f1': self.chord_f1,
                'tp': self.chord_tp,
                'partial': self.chord_partial,
                'fp': self.chord_fp,
                'fn': self.chord_fn,
            },
            'counts': {
                'ground_truth': self.total_ground_truth,
                'detected': self.total_detected,
                'ratio': self.total_detected / self.total_ground_truth if self.total_ground_truth > 0 else 0,
            },
            'near_misses': self.near_misses[:20],  # Cap for JSON output
        }


def _note_to_midi(string: int, fret) -> Optional[int]:
    """Convert string+fret to MIDI note number."""
    if fret == 'X' or fret == 'x':
        return None
    open_midi = STANDARD_TUNING.get(string)
    if open_midi is None:
        return None
    return open_midi + int(fret)


def parse_ground_truth_tabs(tabs_content: str) -> list[dict]:
    """Parse tab notation into a list of notes.

    Returns list of {string: int, fret: int or 'X', beat: float} dicts.
    """
    lines = tabs_content.strip().split('\n')

    # Parse each string line (e string = 1, B = 2, G = 3, D = 4, A = 5, E = 6)
    string_map = {'e': 1, 'B': 2, 'G': 3, 'D': 4, 'A': 5, 'E': 6}

    notes = []

    for line in lines:
        if '|' not in line:
            continue

        # Extract string name and content
        parts = line.split('|')
        if len(parts) < 2:
            continue

        # Parse string identifier from first part
        string_id = None
        first_part = parts[0].strip()
        for char in first_part:
            if char in string_map:
                string_id = string_map[char]
                break

        if string_id is None:
            continue

        # Parse fret numbers from tab content
        # Each character position represents one 16th note (0.25 beats)
        # Bar lines '|' are just separators, not time markers
        content = '|'.join(parts[1:])

        i = 0
        beat_position = 0
        while i < len(content):
            char = content[i]

            if char == '|':
                # Bar line: just a visual separator, no time added
                i += 1
            elif char == '-':
                beat_position += 0.25  # Each dash = one 16th note
                i += 1
            elif char.isdigit():
                # Could be 1 or 2 digit fret number
                fret_str = char
                if i + 1 < len(content) and content[i + 1].isdigit():
                    two_digit = int(char + content[i + 1])
                    if two_digit <= MAX_FRET:
                        # Valid 2-digit fret (e.g., "12" = fret 12)
                        fret_str = char + content[i + 1]
                        i += 1
                    # else: treat as two separate single-digit frets
                    # (e.g., "43" = fret 4 then fret 3)

                fret = int(fret_str)
                notes.append({
                    'string': string_id,
                    'fret': fret,
                    'beat': beat_position,
                    'midi_note': _note_to_midi(string_id, fret),
                })
                beat_position += 0.25
                i += 1
            elif char == 'X' or char == 'x':
                notes.append({
                    'string': string_id,
                    'fret': 'X',
                    'beat': beat_position,
                    'midi_note': None,
                })
                beat_position += 0.25
                i += 1
            elif char == '/':
                beat_position += 0.25  # Slide symbol
                i += 1
            else:
                i += 1

    return sorted(notes, key=lambda n: (n['beat'], n['string']))


def _group_by_beat(notes: list[dict], tolerance: float = 0.01) -> list[list[dict]]:
    """Group notes into chords by beat position."""
    if not notes:
        return []
    groups = []
    current_group = [notes[0]]
    for note in notes[1:]:
        if abs(note['beat'] - current_group[0]['beat']) <= tolerance:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups


def _group_tab_notes_by_time(tab_notes: list[TabNote], tolerance: float = 0.05) -> list[list[TabNote]]:
    """Group TabNotes into chords by timestamp."""
    if not tab_notes:
        return []
    sorted_notes = sorted(tab_notes, key=lambda n: n.timestamp)
    groups = []
    current_group = [sorted_notes[0]]
    for note in sorted_notes[1:]:
        if abs(note.timestamp - current_group[0].timestamp) <= tolerance:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups


def evaluate_accuracy(detected_notes: list[TabNote], ground_truth: list[dict],
                     time_tolerance: float = 0.5,
                     video_duration: float = 13.28) -> EvalMetrics:
    """Evaluate detection accuracy against ground truth with multi-dimensional metrics.

    Args:
        detected_notes: List of TabNote objects from transcription
        ground_truth: List of parsed ground truth notes
        time_tolerance: Allowed timing difference in seconds
        video_duration: Total video duration for beat-to-time conversion

    Returns:
        EvalMetrics with all accuracy dimensions
    """
    metrics = EvalMetrics()
    metrics.total_ground_truth = len(ground_truth)
    metrics.total_detected = len(detected_notes)

    if not ground_truth:
        metrics.exact_fp = len(detected_notes)
        metrics.pitch_fp = len(detected_notes)
        return metrics

    total_beats = max(n['beat'] for n in ground_truth) if ground_truth else 16
    beat_to_time = video_duration / total_beats if total_beats > 0 else 1

    # Convert ground truth to time-based
    gt_with_time = []
    for note in ground_truth:
        gt_with_time.append({
            'string': note['string'],
            'fret': note['fret'],
            'time': note['beat'] * beat_to_time,
            'beat': note['beat'],
            'midi_note': note.get('midi_note'),
        })

    # --- Exact match (string + fret + timing) ---
    gt_exact_matched = set()
    det_exact_matched = set()

    for det_idx, det in enumerate(detected_notes):
        best_match_idx = None
        best_time_diff = float('inf')

        for gt_idx, gt in enumerate(gt_with_time):
            if gt_idx in gt_exact_matched:
                continue
            time_diff = abs(det.timestamp - gt['time'])
            if time_diff > time_tolerance:
                continue
            if det.string == gt['string'] and det.fret == gt['fret']:
                if time_diff < best_time_diff:
                    best_match_idx = gt_idx
                    best_time_diff = time_diff

        if best_match_idx is not None:
            gt_exact_matched.add(best_match_idx)
            det_exact_matched.add(det_idx)

    metrics.exact_tp = len(gt_exact_matched)
    metrics.exact_fp = len(detected_notes) - len(det_exact_matched)
    metrics.exact_fn = len(gt_with_time) - len(gt_exact_matched)

    # --- Pitch-only match (correct MIDI note within time tolerance) ---
    gt_pitch_matched = set()
    det_pitch_matched = set()
    pitch_matches = []  # (det_idx, gt_idx) for position accuracy calc

    for det_idx, det in enumerate(detected_notes):
        det_midi = det.midi_note
        best_match_idx = None
        best_time_diff = float('inf')

        for gt_idx, gt in enumerate(gt_with_time):
            if gt_idx in gt_pitch_matched:
                continue
            gt_midi = gt.get('midi_note')
            if gt_midi is None:
                continue
            time_diff = abs(det.timestamp - gt['time'])
            if time_diff > time_tolerance:
                continue
            if det_midi == gt_midi:
                if time_diff < best_time_diff:
                    best_match_idx = gt_idx
                    best_time_diff = time_diff

        if best_match_idx is not None:
            gt_pitch_matched.add(best_match_idx)
            det_pitch_matched.add(det_idx)
            pitch_matches.append((det_idx, best_match_idx))

    # Count non-muted ground truth for pitch metrics
    gt_pitched = [g for g in gt_with_time if g.get('midi_note') is not None]
    metrics.pitch_tp = len(gt_pitch_matched)
    metrics.pitch_fp = len(detected_notes) - len(det_pitch_matched)
    metrics.pitch_fn = len(gt_pitched) - len(gt_pitch_matched)

    # --- Position accuracy (for pitch-correct notes) ---
    for det_idx, gt_idx in pitch_matches:
        det = detected_notes[det_idx]
        gt = gt_with_time[gt_idx]
        metrics.position_total += 1
        if det.string == gt['string'] and det.fret == gt['fret']:
            metrics.position_correct += 1

    # --- Near misses (pitch correct but wrong position) ---
    for gt_idx, gt in enumerate(gt_with_time):
        if gt_idx in gt_exact_matched:
            continue
        gt_midi = gt.get('midi_note')
        for det in detected_notes:
            time_diff = abs(det.timestamp - gt['time'])
            if time_diff <= time_tolerance:
                if gt_midi is not None and det.midi_note == gt_midi:
                    metrics.near_misses.append({
                        'gt_string': gt['string'], 'gt_fret': gt['fret'],
                        'det_string': det.string, 'det_fret': det.fret,
                        'gt_time': gt['time'], 'det_time': det.timestamp,
                        'midi_note': gt_midi, 'type': 'wrong_position',
                    })
                    break
                elif time_diff <= time_tolerance * 0.8:
                    metrics.near_misses.append({
                        'gt_string': gt['string'], 'gt_fret': gt['fret'],
                        'det_string': det.string, 'det_fret': det.fret,
                        'gt_time': gt['time'], 'det_time': det.timestamp,
                        'midi_note': gt_midi, 'type': 'wrong_pitch',
                    })
                    break

    # --- Chord-level metrics ---
    gt_chords = _group_by_beat(ground_truth)
    gt_multi_chords = [c for c in gt_chords if len(c) >= 2]

    det_groups = _group_tab_notes_by_time(detected_notes)
    det_multi_chords = [g for g in det_groups if len(g) >= 2]

    gt_chord_matched = set()
    for det_group in det_multi_chords:
        det_time = det_group[0].timestamp
        best_chord_idx = None
        best_overlap = 0

        for chord_idx, gt_chord in enumerate(gt_multi_chords):
            if chord_idx in gt_chord_matched:
                continue
            gt_time = gt_chord[0]['beat'] * beat_to_time
            if abs(det_time - gt_time) > time_tolerance:
                continue

            # Count matching notes
            gt_pairs = {(n['string'], n['fret']) for n in gt_chord}
            det_pairs = {(n.string, n.fret) for n in det_group}
            overlap = len(gt_pairs & det_pairs)

            if overlap > best_overlap:
                best_overlap = overlap
                best_chord_idx = chord_idx

        if best_chord_idx is not None:
            gt_chord_matched.add(best_chord_idx)
            gt_chord = gt_multi_chords[best_chord_idx]
            gt_pairs = {(n['string'], n['fret']) for n in gt_chord}
            det_pairs = {(n.string, n.fret) for n in det_group}
            if gt_pairs == det_pairs:
                metrics.chord_tp += 1
            else:
                metrics.chord_partial += 1
        else:
            metrics.chord_fp += 1

    metrics.chord_fn = len(gt_multi_chords) - len(gt_chord_matched)

    return metrics


def run_transcription(video_path: str, audio_only: bool = False,
                      audio_config: AudioAnalysisConfig = None,
                      fusion_config: FusionConfig = None) -> list[TabNote]:
    """Run the transcription pipeline on a video file."""
    if audio_config is None:
        audio_config = AudioAnalysisConfig()
    if fusion_config is None:
        fusion_config = FusionConfig()

    audio_path = '/tmp/eval_audio.wav'

    print("Extracting audio...")
    extract_audio(video_path, audio_path)

    print("Running pitch detection...")
    detected_notes = analyze_pitch(audio_path, audio_config)
    print(f"Audio detected: {len(detected_notes)} notes")

    tab_notes = None

    if not audio_only:
        try:
            timestamps = detect_note_onsets(detected_notes)
            if timestamps:
                fretboard = detect_fretboard_from_video(video_path, num_sample_frames=5)
                if fretboard:
                    video_config = VideoAnalysisConfig()
                    video_observations = analyze_video_at_timestamps(
                        video_path, timestamps, video_config
                    )
                    detection_rate = len(video_observations) / len(timestamps) if timestamps else 0
                    if fretboard.detection_confidence > 0.5 and detection_rate > 0.2:
                        tab_notes = fuse_audio_video(
                            detected_notes, video_observations, fretboard,
                            capo_fret=0, config=fusion_config
                        )
                        print(f"Audio+video fusion: {len(tab_notes)} notes "
                              f"({sum(1 for n in tab_notes if n.video_matched)} video-confirmed)")
        except Exception as e:
            print(f"Video analysis failed: {e}, falling back to audio-only")

    if tab_notes is None:
        tab_notes = fuse_audio_only(detected_notes, capo_fret=0, config=fusion_config)
        print(f"Audio-only fusion: {len(tab_notes)} notes")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return tab_notes


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def print_metrics(metrics: EvalMetrics, label: str = ""):
    """Print formatted metrics summary."""
    header = f"EVALUATION RESULTS{f' ({label})' if label else ''}"
    print(f"\n{'=' * 60}")
    print(header)
    print('=' * 60)
    print(f"Ground truth notes:  {metrics.total_ground_truth}")
    print(f"Detected notes:      {metrics.total_detected}")
    count_ratio = metrics.total_detected / metrics.total_ground_truth if metrics.total_ground_truth > 0 else 0
    print(f"Note count ratio:    {count_ratio:.2f}x (ideal: 1.0)")
    print('-' * 60)
    print("EXACT MATCH (string + fret + timing):")
    print(f"  Precision:         {metrics.exact_precision:.1%}")
    print(f"  Recall:            {metrics.exact_recall:.1%}")
    print(f"  F1 Score:          {metrics.exact_f1:.1%}")
    print(f"  TP/FP/FN:          {metrics.exact_tp}/{metrics.exact_fp}/{metrics.exact_fn}")
    print('-' * 60)
    print("PITCH MATCH (correct MIDI note, any position):")
    print(f"  Precision:         {metrics.pitch_precision:.1%}")
    print(f"  Recall:            {metrics.pitch_recall:.1%}")
    print(f"  F1 Score:          {metrics.pitch_f1:.1%}")
    print(f"  TP/FP/FN:          {metrics.pitch_tp}/{metrics.pitch_fp}/{metrics.pitch_fn}")
    print('-' * 60)
    print("POSITION ACCURACY (for pitch-correct notes):")
    print(f"  Accuracy:          {metrics.position_accuracy:.1%}")
    print(f"  Correct/Total:     {metrics.position_correct}/{metrics.position_total}")
    print('-' * 60)
    print("CHORD DETECTION:")
    print(f"  Precision:         {metrics.chord_precision:.1%}")
    print(f"  Recall:            {metrics.chord_recall:.1%}")
    print(f"  F1 Score:          {metrics.chord_f1:.1%}")
    print(f"  Exact/Partial/FP/FN: {metrics.chord_tp}/{metrics.chord_partial}/{metrics.chord_fp}/{metrics.chord_fn}")
    print('=' * 60)

    if metrics.near_misses:
        print(f"\nNEAR MISSES ({len(metrics.near_misses)} found):")
        for nm in metrics.near_misses[:10]:
            if nm['type'] == 'wrong_position':
                print(f"  Pitch OK but wrong pos: s{nm['gt_string']}f{nm['gt_fret']} -> "
                      f"s{nm['det_string']}f{nm['det_fret']} (MIDI {nm['midi_note']})")
            else:
                print(f"  Wrong pitch: expected s{nm['gt_string']}f{nm['gt_fret']} "
                      f"got s{nm['det_string']}f{nm['det_fret']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate guitar transcription accuracy')
    parser.add_argument('video_path', nargs='?',
                       default='/home/gilhooleyp/projects/tab_vision/test-data/existing/sample-video.mp4',
                       help='Path to video file')
    parser.add_argument('ground_truth_path', nargs='?',
                       default='/home/gilhooleyp/projects/tab_vision/test-data/existing/sample-video-tabs.txt',
                       help='Path to ground truth tab file')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Timing tolerance in seconds')
    parser.add_argument('--output', type=str, default=None,
                       help='Save metrics to JSON file')
    parser.add_argument('--sweep', action='store_true',
                       help='Run tolerance sweep from 0.1 to 1.0')
    parser.add_argument('--audio-only', action='store_true',
                       help='Skip video analysis')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed match/miss info')

    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth_path) as f:
        tabs_content = f.read()

    ground_truth = parse_ground_truth_tabs(tabs_content)
    print(f"Ground truth: {len(ground_truth)} notes")

    # Run transcription
    tab_notes = run_transcription(args.video_path, audio_only=args.audio_only)

    # Get video duration
    video_duration = get_video_duration(args.video_path)
    print(f"Video duration: {video_duration:.2f}s")

    if args.sweep:
        # Tolerance sweep
        print("\nTOLERANCE SWEEP:")
        print(f"{'Tol':>6} {'ExactF1':>8} {'PitchF1':>8} {'PosAcc':>8} {'ChordF1':>8}")
        print('-' * 42)
        sweep_results = {}
        for tol_10 in range(1, 11):
            tol = tol_10 / 10.0
            m = evaluate_accuracy(tab_notes, ground_truth,
                                 time_tolerance=tol, video_duration=video_duration)
            print(f"{tol:>6.1f} {m.exact_f1:>8.1%} {m.pitch_f1:>8.1%} "
                  f"{m.position_accuracy:>8.1%} {m.chord_f1:>8.1%}")
            sweep_results[str(tol)] = m.to_dict()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(sweep_results, f, indent=2)
            print(f"\nSweep results saved to {args.output}")
    else:
        # Single evaluation
        metrics = evaluate_accuracy(tab_notes, ground_truth,
                                   time_tolerance=args.time_tolerance,
                                   video_duration=video_duration)

        print_metrics(metrics)

        if args.verbose and metrics.near_misses:
            print(f"\nALL NEAR MISSES ({len(metrics.near_misses)}):")
            for nm in metrics.near_misses:
                print(f"  {nm['type']}: s{nm['gt_string']}f{nm['gt_fret']} @ "
                      f"{nm['gt_time']:.2f}s -> s{nm['det_string']}f{nm['det_fret']} "
                      f"@ {nm['det_time']:.2f}s")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\nMetrics saved to {args.output}")


if __name__ == '__main__':
    main()
