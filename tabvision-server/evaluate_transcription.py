"""Evaluate transcription against ground truth tabs."""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from app.audio_pipeline import extract_audio, analyze_pitch, AudioAnalysisConfig
from app.fusion_engine import fuse_audio_only, FusionConfig, TabNote


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
                    fret_str += content[i + 1]
                    i += 1

                fret = int(fret_str)
                notes.append({
                    'string': string_id,
                    'fret': fret,
                    'beat': beat_position,
                })
                beat_position += 0.25
                i += 1
            elif char == 'X' or char == 'x':
                notes.append({
                    'string': string_id,
                    'fret': 'X',
                    'beat': beat_position,
                })
                beat_position += 0.25
                i += 1
            elif char == '/':
                beat_position += 0.25  # Slide symbol
                i += 1
            else:
                i += 1

    return sorted(notes, key=lambda n: (n['beat'], n['string']))


def evaluate_accuracy(detected_notes: list, ground_truth: list,
                     time_tolerance: float = 0.5,
                     video_duration: float = 13.28) -> dict:
    """Evaluate detection accuracy against ground truth.

    Args:
        detected_notes: List of TabNote objects from transcription
        ground_truth: List of parsed ground truth notes
        time_tolerance: Allowed timing difference in seconds
        video_duration: Total video duration for beat-to-time conversion

    Returns:
        Dictionary with accuracy metrics
    """
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
        })

    # Match detected notes to ground truth (greedy, closest time first)
    matches = []
    false_positives = []

    gt_matched = set()

    for det in detected_notes:
        best_match = None
        best_match_idx = None
        best_time_diff = float('inf')

        for i, gt in enumerate(gt_with_time):
            if i in gt_matched:
                continue

            time_diff = abs(det.timestamp - gt['time'])
            if time_diff > time_tolerance:
                continue

            # Check string and fret (exact match)
            if det.string == gt['string'] and det.fret == gt['fret']:
                if time_diff < best_time_diff:
                    best_match = gt
                    best_match_idx = i
                    best_time_diff = time_diff

        if best_match:
            matches.append((det, best_match))
            gt_matched.add(best_match_idx)
        else:
            false_positives.append(det)

    # Calculate metrics
    true_positives = len(matches)
    false_negatives = len(gt_with_time) - len(gt_matched)

    precision = true_positives / (true_positives + len(false_positives)) if (true_positives + len(false_positives)) > 0 else 0
    recall = true_positives / len(gt_with_time) if gt_with_time else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / len(gt_with_time) if gt_with_time else 0

    # Populate near-misses: detected notes that match timing but wrong string/fret
    near_misses = []
    for i, gt in enumerate(gt_with_time):
        if i in gt_matched:
            continue
        for det in detected_notes:
            time_diff = abs(det.timestamp - gt['time'])
            if time_diff <= time_tolerance:
                near_misses.append({
                    'gt': gt,
                    'det_string': det.string,
                    'det_fret': det.fret,
                    'det_time': det.timestamp,
                    'time_diff': time_diff,
                })

    return {
        'total_ground_truth': len(gt_with_time),
        'total_detected': len(detected_notes),
        'true_positives': true_positives,
        'false_positives': len(false_positives),
        'false_negatives': false_negatives,
        'near_misses': near_misses,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy * 100,
        'note_count_ratio': len(detected_notes) / len(gt_with_time) if gt_with_time else 0,
        'matches': matches,
        'gt_matched': gt_matched,
        'gt_with_time': gt_with_time,
    }


def main():
    # Load ground truth
    with open('/home/gilhooleyp/projects/tab_vision/sample-video-tabs.txt') as f:
        tabs_content = f.read()

    ground_truth = parse_ground_truth_tabs(tabs_content)
    print(f"Ground truth: {len(ground_truth)} notes")

    # Run audio pipeline
    video_path = '/home/gilhooleyp/projects/tab_vision/sample-video.mp4'
    audio_path = '/tmp/eval_audio.wav'

    print("\nExtracting audio...")
    extract_audio(video_path, audio_path)

    print("Running pitch detection...")
    audio_config = AudioAnalysisConfig()
    detected_notes = analyze_pitch(audio_path, audio_config)
    print(f"Audio detected: {len(detected_notes)} notes")

    print("Running fusion (audio-only)...")
    fusion_config = FusionConfig()
    tab_notes = fuse_audio_only(detected_notes, capo_fret=0, config=fusion_config)
    print(f"Tab notes after fusion: {len(tab_notes)}")

    # Get video duration
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
        capture_output=True, text=True
    )
    video_duration = float(result.stdout.strip())
    print(f"Video duration: {video_duration:.2f}s")

    # Evaluate
    metrics = evaluate_accuracy(tab_notes, ground_truth,
                               time_tolerance=0.6, video_duration=video_duration)

    # Detailed analysis
    total_beats = max(n['beat'] for n in ground_truth)
    beat_to_time = video_duration / total_beats

    print("\nMATCHES:")
    for det, gt in metrics['matches']:
        print(f"  s{gt['string']}f{gt['fret']} @ beat={gt['beat']:.1f} "
              f"(gt={gt['time']:.2f}s, det={det.timestamp:.2f}s)")

    print("\nMISSED (false negatives):")
    for i, gt in enumerate(metrics['gt_with_time']):
        if i not in metrics['gt_matched']:
            # Find closest detection
            closest = None
            closest_dist = 999
            for det in tab_notes:
                dist = abs(det.timestamp - gt['time'])
                if dist < closest_dist:
                    closest = det
                    closest_dist = dist
            if closest and closest_dist < 1.0:
                print(f"  s{gt['string']}f{gt['fret']} @ beat={gt['beat']:.1f} "
                      f"(t={gt['time']:.2f}s) | nearest: s{closest.string}f{closest.fret} "
                      f"@ {closest.timestamp:.2f}s (dt={closest_dist:.2f})")
            else:
                print(f"  s{gt['string']}f{gt['fret']} @ beat={gt['beat']:.1f} "
                      f"(t={gt['time']:.2f}s) | no detection nearby")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Ground truth notes:  {metrics['total_ground_truth']}")
    print(f"Detected notes:      {metrics['total_detected']}")
    print(f"Note count ratio:    {metrics['note_count_ratio']:.2f}x (ideal: 1.0)")
    print(f"True positives:      {metrics['true_positives']}")
    print(f"False positives:     {metrics['false_positives']}")
    print(f"False negatives:     {metrics['false_negatives']}")
    print("-" * 50)
    print(f"Precision:           {metrics['precision']:.1%}")
    print(f"Recall:              {metrics['recall']:.1%}")
    print(f"F1 Score:            {metrics['f1_score']:.1%}")
    print("=" * 50)

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return metrics


if __name__ == '__main__':
    main()
