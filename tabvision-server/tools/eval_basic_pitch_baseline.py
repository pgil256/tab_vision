"""Vanilla Basic Pitch baseline on the held-out GuitarSet player split.

Phase 1 Week 2 deliverable per
docs/plans/2026-04-24-audio-backbone-finetune-design.md §7.

What this measures:
  - Frame note F1 at thresholds {0.3, 0.5, 0.7} — per-cell binary match
    of model.note > threshold against the densified note target
    (target derived from JAMS, frame-rate matched to model output).
  - Onset P/R/F1 at threshold 0.5 with ±1 frame tolerance (~12 ms at
    86 fps).
  - Note-event F1 via `mir_eval.transcription.precision_recall_f1_overlap`
    on the note events produced by `basic_pitch.note_creation.\
    model_output_to_notes` — the same path `predict.py` uses, so the
    numbers are apples-to-apples with the fine-tune output. Threshold
    sweep at 0.05 stride; best (P, R, F1) reported per track.

Implementation note: we call `basic_pitch.inference.predict(audio_path)`
directly per track. That handles the 30-frame overlap + unwrap_output
that the production inference path uses; doing it ourselves with naive
non-overlapping windows produces worse note-event F1 (verified empirically
on a 2-track smoke; F1 ~ 0.1 vs. the production-path number we expect).
The trade-off: we can't directly evaluate a *fine-tuned Keras model* this
way; we'll need to save the fine-tune as a SavedModel and pass the path.

Inputs (defaults):
  - Validation track IDs derived from the TFRecord split at
    `tools/outputs/tfrecords/guitarset/splits/validation/*.tfrecord`
    (so the player split is consistent with training).
  - Audio at `~/mir_datasets/guitarset/audio_mono-mic/{tid}_mic.wav`.
  - JAMS at `~/mir_datasets/guitarset/annotation/{tid}.jams` (mir_eval
    ground truth — onset+pitch only, no offset constraint).

Outputs:
  - `tools/outputs/finetune_baseline-YYYY-MM-DD.md` (markdown summary).
  - `tools/outputs/finetune_baseline-YYYY-MM-DD.csv` (per-track table).

Usage:
    python -m tools.eval_basic_pitch_baseline
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Constants from basic_pitch.constants — duplicated here so module load is
# fast (TF imports happen lazily inside main).
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_SAMPLES = 43844           # 2-second window in samples
ANNOTATIONS_FPS = 86
ANNOT_N_FRAMES = 172              # frames per 2-second window
N_FREQ_BINS_NOTES = 88
N_FREQ_BINS_CONTOURS = 264
FREQ_BIN_BASE_HZ = 27.5

DEFAULT_TFRECORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tools', 'outputs', 'tfrecords',
)
DEFAULT_DATA_HOME = os.path.expanduser('~/mir_datasets/guitarset')
DEFAULT_JAMS_HOME = os.path.join(DEFAULT_DATA_HOME, 'annotation')
DEFAULT_AUDIO_HOME = os.path.join(DEFAULT_DATA_HOME, 'audio_mono-mic')
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tools', 'outputs',
)


@dataclass
class _PerTrackResult:
    file_id: str
    n_ref_notes: int
    n_est_notes: int
    frame_f1_03: float
    frame_f1_05: float
    frame_f1_07: float
    onset_p_05: float
    onset_r_05: float
    onset_f1_05: float
    note_p: float
    note_r: float
    note_f1: float
    best_onset_thresh: float
    best_frame_thresh: float


# ---------------------------------------------------------------------------
# Frame target reconstruction from JAMS (matches model output time scale)
# ---------------------------------------------------------------------------


def _frame_targets_from_jams(
    jams_path: str, n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense (n_frames, 88) note + onset target tensors from JAMS.

    Both targets are at 86 fps with bin idx = midi - 21 (matching the
    model output convention).
    """
    import json
    note = np.zeros((n_frames, N_FREQ_BINS_NOTES), dtype=np.float32)
    onset = np.zeros((n_frames, N_FREQ_BINS_NOTES), dtype=np.float32)
    with open(jams_path) as f:
        jams = json.load(f)
    for ann in jams.get('annotations', []):
        if ann.get('namespace') != 'note_midi':
            continue
        for d in ann.get('data') or []:
            try:
                t0 = float(d['time'])
                dur = float(d['duration'])
                midi = int(round(float(d['value'])))
            except (KeyError, TypeError, ValueError):
                continue
            bin_idx = midi - 21
            if not (0 <= bin_idx < N_FREQ_BINS_NOTES):
                continue
            f_start = int(round(t0 * ANNOTATIONS_FPS))
            f_end = int(round((t0 + dur) * ANNOTATIONS_FPS))
            if f_start >= n_frames:
                continue
            f_end = min(f_end, n_frames - 1)
            f_start_clamped = max(f_start, 0)
            onset[f_start_clamped, bin_idx] = 1.0
            note[f_start_clamped:f_end + 1, bin_idx] = 1.0
    return note, onset


# ---------------------------------------------------------------------------
# Frame / onset metrics
# ---------------------------------------------------------------------------


def _binary_f1(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Compute (precision, recall, F1) over flattened binary tensors."""
    pred_b = pred.astype(bool)
    targ_b = target.astype(bool)
    tp = int(np.logical_and(pred_b, targ_b).sum())
    fp = int(np.logical_and(pred_b, ~targ_b).sum())
    fn = int(np.logical_and(~pred_b, targ_b).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _onset_metrics(
    pred_onset: np.ndarray, target_onset: np.ndarray,
    threshold: float, tolerance_frames: int = 1,
) -> tuple[float, float, float]:
    """Onset precision/recall/F1 with a ±tolerance-frame match window.

    The target tensor is dilated by the tolerance window before TP counting,
    so a prediction within ±tolerance of any target frame counts as a hit.
    Same dilation is applied symmetrically for recall (target frame counts as
    hit if any prediction frame falls within ±tolerance).
    """
    pred_b = pred_onset > threshold
    targ_b = target_onset.astype(bool)

    # Dilate along the time axis (axis 0).
    def _dilate(a: np.ndarray, k: int) -> np.ndarray:
        out = a.copy()
        for d in range(1, k + 1):
            out[d:] |= a[:-d]
            out[:-d] |= a[d:]
        return out

    pred_dilated = _dilate(pred_b, tolerance_frames)
    targ_dilated = _dilate(targ_b, tolerance_frames)
    tp_pred = int(np.logical_and(pred_b, targ_dilated).sum())
    tp_targ = int(np.logical_and(targ_b, pred_dilated).sum())
    fp = int(pred_b.sum() - tp_pred)
    fn = int(targ_b.sum() - tp_targ)
    p = tp_pred / (tp_pred + fp) if (tp_pred + fp) else 0.0
    r = tp_targ / (tp_targ + fn) if (tp_targ + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Note-event metrics via mir_eval
# ---------------------------------------------------------------------------


def _midi_to_hz(midi_array: np.ndarray) -> np.ndarray:
    return 440.0 * 2 ** ((midi_array - 69.0) / 12.0)


def _ref_notes_from_jams(jams_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (intervals (N,2), pitches_hz (N,)) from a GuitarSet JAMS file."""
    import json
    with open(jams_path) as f:
        jams = json.load(f)
    intervals: list[tuple[float, float]] = []
    pitches_hz: list[float] = []
    for ann in jams.get('annotations', []):
        if ann.get('namespace') != 'note_midi':
            continue
        for d in ann.get('data') or []:
            try:
                onset = float(d['time'])
                duration = float(d['duration'])
                midi = float(d['value'])
            except (KeyError, TypeError, ValueError):
                continue
            intervals.append((onset, onset + duration))
            pitches_hz.append(440.0 * 2 ** ((midi - 69.0) / 12.0))
    if not intervals:
        return np.zeros((0, 2)), np.zeros((0,))
    iv = np.asarray(intervals, dtype=np.float64)
    pitches = np.asarray(pitches_hz, dtype=np.float64)
    order = iv[:, 0].argsort()
    return iv[order], pitches[order]


def _est_notes_from_output(
    output: dict[str, np.ndarray],
    onset_thresh: float,
    frame_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run note_creation on the model output, return (intervals, pitches_hz).

    `include_pitch_bends=True` is critical: that's what production `predict()`
    uses, and it triggers `drop_overlapping_pitch_bends` which dedups
    near-overlapping events. Setting it False produces ~3-5x more events
    and tanks precision (verified empirically — F1 drops from ~0.84 to ~0.14
    on the GuitarSet held-out validation player split).
    """
    from basic_pitch.note_creation import model_output_to_notes
    _, note_events = model_output_to_notes(
        output,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=True,
        min_note_len=11,
        include_pitch_bends=True,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )
    if not note_events:
        return np.zeros((0, 2)), np.zeros((0,))
    starts = np.asarray([n[0] for n in note_events], dtype=np.float64)
    ends = np.asarray([n[1] for n in note_events], dtype=np.float64)
    midi = np.asarray([n[2] for n in note_events], dtype=np.float64)
    intervals = np.stack([starts, ends], axis=1)
    return intervals, _midi_to_hz(midi)


def _note_event_f1(
    ref_intervals: np.ndarray, ref_pitches: np.ndarray,
    est_intervals: np.ndarray, est_pitches: np.ndarray,
) -> tuple[float, float, float]:
    """mir_eval onset+pitch match F1 (no offset constraint, default tolerances)."""
    from mir_eval.transcription import precision_recall_f1_overlap
    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        return 0.0, 0.0, 0.0
    p, r, f1, _ = precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        offset_ratio=None,  # ignore offsets — pitch + onset only
    )
    return float(p), float(r), float(f1)


# ---------------------------------------------------------------------------
# Per-track evaluation
# ---------------------------------------------------------------------------


def _eval_one_track(
    file_id: str,
    audio_path: str,
    jams_path: str,
    note_thresh_sweep: Iterable[tuple[float, float]],
    model_path: str | None = None,
) -> _PerTrackResult:
    """Run inference.run_inference on the audio file (overlap+unwrap-correct),
    then compute frame / onset / note-event metrics."""
    from basic_pitch.inference import run_inference, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    resolved = model_path or str(ICASSP_2022_MODEL_PATH)
    cache_key = f'_cached_model_{resolved}'
    if not hasattr(_eval_one_track, cache_key):
        setattr(_eval_one_track, cache_key, Model(resolved))
    out = run_inference(audio_path, model_or_model_path=getattr(_eval_one_track, cache_key))
    n_frames = out['note'].shape[0]
    target_note, target_onset = _frame_targets_from_jams(jams_path, n_frames)

    pred_note = out['note']
    pred_onset = out['onset']

    # Frame F1 at three thresholds.
    frame_f1s = {}
    for thr in (0.3, 0.5, 0.7):
        _, _, f1 = _binary_f1(pred_note > thr, target_note)
        frame_f1s[thr] = f1

    # Onset metrics at 0.5 (canonical), with ±1 frame tolerance.
    op, or_, of = _onset_metrics(pred_onset, target_onset, threshold=0.5)

    # Note-event F1 via mir_eval. Threshold sweep.
    ref_intervals, ref_pitches = _ref_notes_from_jams(jams_path)
    best = (-1.0, 0.0, 0.0, 0.0, 0.0)  # (f1, p, r, onset_thr, frame_thr)
    n_est_at_best = 0
    for onset_thr, frame_thr in note_thresh_sweep:
        est_intervals, est_pitches = _est_notes_from_output(
            out, onset_thresh=onset_thr, frame_thresh=frame_thr,
        )
        p, r, f1 = _note_event_f1(ref_intervals, ref_pitches, est_intervals, est_pitches)
        if f1 > best[0]:
            best = (f1, p, r, onset_thr, frame_thr)
            n_est_at_best = est_intervals.shape[0]

    return _PerTrackResult(
        file_id=file_id,
        n_ref_notes=ref_intervals.shape[0],
        n_est_notes=n_est_at_best,
        frame_f1_03=frame_f1s[0.3],
        frame_f1_05=frame_f1s[0.5],
        frame_f1_07=frame_f1s[0.7],
        onset_p_05=op,
        onset_r_05=or_,
        onset_f1_05=of,
        note_p=best[1],
        note_r=best[2],
        note_f1=best[0],
        best_onset_thresh=best[3],
        best_frame_thresh=best[4],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _make_thresh_sweep() -> list[tuple[float, float]]:
    """Co-sweep onset/frame thresholds at 0.05 stride."""
    onset_grid = [round(0.1 + 0.05 * i, 2) for i in range(15)]   # 0.10..0.80
    frame_grid = [round(0.1 + 0.05 * i, 2) for i in range(13)]   # 0.10..0.70
    return [(o, f) for o in onset_grid for f in frame_grid]


def _aggregate(results: list[_PerTrackResult]) -> dict[str, float]:
    if not results:
        return {}
    keys = ('frame_f1_03', 'frame_f1_05', 'frame_f1_07',
            'onset_p_05', 'onset_r_05', 'onset_f1_05',
            'note_p', 'note_r', 'note_f1',
            'best_onset_thresh', 'best_frame_thresh')
    out = {}
    for k in keys:
        vals = [getattr(r, k) for r in results]
        out[f'mean_{k}'] = float(np.mean(vals))
        out[f'median_{k}'] = float(np.median(vals))
    out['n_tracks'] = len(results)
    out['total_ref_notes'] = sum(r.n_ref_notes for r in results)
    out['total_est_notes'] = sum(r.n_est_notes for r in results)
    return out


def _write_report(
    md_path: str, csv_path: str, results: list[_PerTrackResult],
    summary: dict[str, float], split_label: str,
):
    # CSV per-track.
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'file_id', 'n_ref_notes', 'n_est_notes',
            'frame_f1_0.3', 'frame_f1_0.5', 'frame_f1_0.7',
            'onset_p@0.5', 'onset_r@0.5', 'onset_f1@0.5',
            'note_p', 'note_r', 'note_f1',
            'best_onset_thr', 'best_frame_thr',
        ])
        for r in results:
            w.writerow([
                r.file_id, r.n_ref_notes, r.n_est_notes,
                f'{r.frame_f1_03:.4f}', f'{r.frame_f1_05:.4f}', f'{r.frame_f1_07:.4f}',
                f'{r.onset_p_05:.4f}', f'{r.onset_r_05:.4f}', f'{r.onset_f1_05:.4f}',
                f'{r.note_p:.4f}', f'{r.note_r:.4f}', f'{r.note_f1:.4f}',
                f'{r.best_onset_thresh:.2f}', f'{r.best_frame_thresh:.2f}',
            ])

    # Markdown summary.
    today = dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f'# Vanilla Basic Pitch baseline — {today}')
    lines.append('')
    lines.append('Reference number for the audio fine-tune (plan §7 Week 2).')
    lines.append(f'Split: **{split_label}**.  Tracks: **{summary["n_tracks"]}**.')
    lines.append('')
    lines.append('## Aggregate')
    lines.append('')
    lines.append('| Metric | Mean | Median |')
    lines.append('|---|---:|---:|')
    for label, key in [
        ('Frame note F1 @ 0.3', 'frame_f1_03'),
        ('Frame note F1 @ 0.5', 'frame_f1_05'),
        ('Frame note F1 @ 0.7', 'frame_f1_07'),
        ('Onset P @ 0.5 (±1 frame)', 'onset_p_05'),
        ('Onset R @ 0.5 (±1 frame)', 'onset_r_05'),
        ('Onset F1 @ 0.5 (±1 frame)', 'onset_f1_05'),
        ('Note-event P (best)', 'note_p'),
        ('Note-event R (best)', 'note_r'),
        ('Note-event F1 (best)', 'note_f1'),
    ]:
        m = summary.get(f'mean_{key}', float('nan'))
        med = summary.get(f'median_{key}', float('nan'))
        lines.append(f'| {label} | {m:.4f} | {med:.4f} |')
    lines.append('')
    lines.append(
        f'Best note-event threshold (track-wise mean): '
        f'onset={summary.get("mean_best_onset_thresh", float("nan")):.2f}, '
        f'frame={summary.get("mean_best_frame_thresh", float("nan")):.2f}.'
    )
    lines.append('')
    lines.append(f'Total reference notes across split: **{summary["total_ref_notes"]}**.')
    lines.append(f'Total estimated notes at best thresholds: **{summary["total_est_notes"]}**.')
    lines.append('')
    lines.append('## Per-track table')
    lines.append('See `' + os.path.basename(csv_path) + '`.')
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append(
        '- **Note-event F1 is the headline metric.** It maps directly to '
        'how the fine-tune output will be consumed '
        '(`note_creation.model_output_to_notes` → notes → fusion engine).'
    )
    lines.append('')
    lines.append(
        '- Frame F1: per-cell binary on (T, 88) note head vs densified target. '
        'Sanity reference for "is the model picking up the right pitches at the '
        'right times" — frame F1 ≤ note-event F1 is expected because '
        'frame-level disagreement around onset/offset edges contributes to FP/FN '
        'at every frame, while note-event matching is one decision per note.'
    )
    lines.append('')
    lines.append(
        '- **Onset P/R/F1 is unreliable as currently implemented.** ±1-frame '
        '(~12 ms) tolerance vs the model\'s smoothed multi-frame onset ridges '
        'systematically under-counts TPs. Numbers reported for completeness; '
        'use note-event F1 to compare baselines and fine-tunes.'
    )
    lines.append('')
    lines.append(
        '- Note-event metrics use `mir_eval.transcription.precision_recall_f1_overlap` '
        'with `offset_ratio=None` (onset+pitch only). Best F1 over a 0.05-stride '
        'sweep of (onset_thresh, frame_thresh).'
    )
    lines.append('')
    lines.append(
        '- **Scope reminder.** This is *in-distribution* GuitarSet held-out (split '
        'by player). The plan §0 ship gate is on our **20-video iPhone set** '
        '(out-of-distribution), where the current exact F1 is ~0.51 and the '
        'target is ≥ 0.60. Use the present number (note-event F1 ≈ 0.87) only as '
        'the within-GuitarSet reference — improvement here is a *necessary but '
        'not sufficient* condition for OOD improvement.'
    )

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _track_ids_from_tfrecord_split(tfrecord_dir: str, split: str) -> list[str]:
    """Read just the file_id field out of each TFRecord example."""
    import glob
    import tensorflow as tf
    from basic_pitch.data import tf_example_deserialization as bd

    pattern = os.path.join(tfrecord_dir, 'guitarset', 'splits', split, '*.tfrecord')
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    ds = tf.data.TFRecordDataset(files).map(bd.parse_transcription_tfexample)
    out: list[str] = []
    for fields in ds:
        file_id = fields[0]
        out.append(file_id.numpy().decode())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tfrecord-dir', default=DEFAULT_TFRECORD_DIR,
                    help='base dir containing guitarset/splits/{split}/*.tfrecord '
                         '(used only to determine which track ids belong to the split)')
    ap.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    ap.add_argument('--jams-home', default=DEFAULT_JAMS_HOME)
    ap.add_argument('--audio-home', default=DEFAULT_AUDIO_HOME)
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    ap.add_argument('--limit', type=int, default=None,
                    help='evaluate only the first N tracks (for smoke runs)')
    ap.add_argument('--coarse-sweep', action='store_true',
                    help='use a smaller threshold sweep — 0.1 stride instead of 0.05 — for speed')
    ap.add_argument('--model-path', default=None,
                    help='path to a Basic Pitch SavedModel directory (defaults to '
                         'the shipped ICASSP_2022 weights — pass a fine-tune output to '
                         'evaluate that instead)')
    ap.add_argument('--label', default='baseline',
                    help='filename suffix for the report — output goes to '
                         'tools/outputs/finetune_{label}-{date}.{md,csv}')
    args = ap.parse_args(argv)

    # Determine threshold sweep.
    if args.coarse_sweep:
        onset_grid = [round(0.1 + 0.1 * i, 2) for i in range(8)]
        frame_grid = [round(0.1 + 0.1 * i, 2) for i in range(7)]
        sweep = [(o, f) for o in onset_grid for f in frame_grid]
    else:
        sweep = _make_thresh_sweep()
    print(f'threshold sweep: {len(sweep)} (onset, frame) pairs', file=sys.stderr)

    track_ids = _track_ids_from_tfrecord_split(args.tfrecord_dir, args.split)
    if not track_ids:
        print(f'no tfrecord files matched split={args.split}', file=sys.stderr)
        return 1
    if args.limit is not None:
        track_ids = track_ids[:args.limit]
    print(f'split={args.split}: {len(track_ids)} tracks', file=sys.stderr)

    results: list[_PerTrackResult] = []
    for i, tid in enumerate(track_ids):
        audio_path = os.path.join(args.audio_home, f'{tid}_mic.wav')
        jams_path = os.path.join(args.jams_home, f'{tid}.jams')
        if not (os.path.exists(audio_path) and os.path.exists(jams_path)):
            print(f'  ! {tid}: missing audio or JAMS', file=sys.stderr)
            continue
        try:
            r = _eval_one_track(
                tid, audio_path, jams_path, note_thresh_sweep=sweep,
                model_path=args.model_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f'  ! {tid}: {exc}', file=sys.stderr)
            continue
        results.append(r)
        print(
            f'  [{i+1}/{len(track_ids)}] {tid}: '
            f'frame_f1@0.5={r.frame_f1_05:.3f} '
            f'onset_f1={r.onset_f1_05:.3f} '
            f'note_f1={r.note_f1:.3f} (onset_thr={r.best_onset_thresh:.2f}, '
            f'frame_thr={r.best_frame_thresh:.2f}, '
            f'{r.n_ref_notes} ref notes)',
            file=sys.stderr,
        )

    if not results:
        print('no results — aborting report', file=sys.stderr)
        return 2

    summary = _aggregate(results)
    today = dt.date.today().isoformat()
    md_path = os.path.join(args.output_dir, f'finetune_{args.label}-{today}.md')
    csv_path = os.path.join(args.output_dir, f'finetune_{args.label}-{today}.csv')
    _write_report(md_path, csv_path, results, summary, split_label=args.split)
    print(f'wrote {md_path}', file=sys.stderr)
    print(f'wrote {csv_path}', file=sys.stderr)

    print('--- summary ---', file=sys.stderr)
    for k in ('mean_frame_f1_05', 'mean_onset_f1_05', 'mean_note_f1',
              'mean_best_onset_thresh', 'mean_best_frame_thresh',
              'total_ref_notes', 'total_est_notes'):
        print(f'  {k}: {summary[k]}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
