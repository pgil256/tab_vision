"""Build basic_pitch-compatible TFRecords from sideloaded GuitarSet data.

Phase 1 Week 1 deliverable per docs/plans/2026-04-24-audio-backbone-finetune-design.md §7.

Why this exists: Spotify's `basic_pitch.data.datasets.guitarset` Beam pipeline
relies on `mirdata` to read JAMS + WAV pairs, and `mirdata.initialize('guitarset')`
needs an index JSON hosted on Zenodo. Zenodo is unreachable from our network
(see memory `project_audio_finetune_phase1_status.md`), so we generate the
TFRecords ourselves from the HuggingFace-sideloaded data
(`tools/sideload_guitarset_from_hf.py`).

The TFRecord schema matches Spotify's exactly — see
`basic_pitch.data.tf_example_serialization` and
`basic_pitch.data.tf_example_deserialization.parse_transcription_tfexample`.
That means the output drops in to `basic_pitch.train` without further glue.

Output layout matches `basic_pitch.data.tf_example_deserialization.sample_datasets`:

    {output_dir}/guitarset/splits/train/*.tfrecord
    {output_dir}/guitarset/splits/validation/*.tfrecord
    {output_dir}/guitarset/splits/test/*.tfrecord

Usage:
    python -m tools.build_guitarset_tfrecords --limit 5
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf

# basic_pitch constants — duplicated here to keep the script free of TF imports
# until serialization time. See basic_pitch.constants for the source of truth.
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_CHANNELS = 1
ANNOTATIONS_FPS = 86
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS
N_FREQ_BINS_NOTES = 88
N_FREQ_BINS_CONTOURS = 264
FREQ_BIN_BASE_HZ = 27.5  # MIDI 21 (A0)
NOTES_BIN_PER_SEMITONE = 1
CONTOURS_BIN_PER_SEMITONE = 3

DEFAULT_DATA_HOME = os.path.expanduser('~/mir_datasets/guitarset')
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tools', 'outputs', 'tfrecords',
)


@dataclass
class _NoteEvent:
    onset_s: float
    duration_s: float
    midi_int: int


@dataclass
class _ContourSample:
    time_s: float
    frequency_hz: float
    confidence: float


# --- JAMS parsing -----------------------------------------------------------


def _parse_jams(jams_path: str) -> tuple[list[_NoteEvent], list[_ContourSample]]:
    with open(jams_path) as f:
        jams = json.load(f)
    notes: list[_NoteEvent] = []
    contours: list[_ContourSample] = []
    for ann in jams.get('annotations', []):
        ns = ann.get('namespace')
        data = ann.get('data')
        if ns == 'note_midi':
            for d in data or []:
                try:
                    midi = int(round(float(d['value'])))
                    notes.append(_NoteEvent(
                        onset_s=float(d['time']),
                        duration_s=float(d['duration']),
                        midi_int=midi,
                    ))
                except (KeyError, TypeError, ValueError):
                    continue
        elif ns == 'pitch_contour':
            # JAMS dense format: dict with parallel time/value/confidence lists.
            if not isinstance(data, dict):
                continue
            times = data.get('time') or []
            values = data.get('value') or []
            confs = data.get('confidence') or [None] * len(times)
            for i in range(len(times)):
                v = values[i] if i < len(values) else None
                if not isinstance(v, dict) or not v.get('voiced'):
                    continue
                freq = v.get('frequency')
                if freq is None or freq <= 0:
                    continue
                conf = confs[i] if i < len(confs) and confs[i] is not None else 1.0
                contours.append(_ContourSample(
                    time_s=float(times[i]),
                    frequency_hz=float(freq),
                    confidence=float(conf),
                ))
    return notes, contours


# --- bin/frame mapping ------------------------------------------------------


def _freq_to_note_bin(freq_hz: float) -> int | None:
    if freq_hz <= 0:
        return None
    bin_idx = int(round(12 * math.log2(freq_hz / FREQ_BIN_BASE_HZ)))
    return bin_idx if 0 <= bin_idx < N_FREQ_BINS_NOTES else None


def _freq_to_contour_bin(freq_hz: float) -> int | None:
    if freq_hz <= 0:
        return None
    bin_idx = int(round(36 * math.log2(freq_hz / FREQ_BIN_BASE_HZ)))
    return bin_idx if 0 <= bin_idx < N_FREQ_BINS_CONTOURS else None


def _midi_to_note_bin(midi_int: int) -> int | None:
    bin_idx = midi_int - 21
    return bin_idx if 0 <= bin_idx < N_FREQ_BINS_NOTES else None


# --- sparse array build ----------------------------------------------------


def _build_sparse_arrays(
    notes: Sequence[_NoteEvent],
    contours: Sequence[_ContourSample],
    n_time_frames: int,
) -> tuple[
    list[tuple[int, int]], list[float],
    list[tuple[int, int]], list[float],
    list[tuple[int, int]], list[float],
]:
    """Return (notes_idx, notes_val, onsets_idx, onsets_val, contours_idx, contours_val)."""
    notes_idx: list[tuple[int, int]] = []
    notes_val: list[float] = []
    onsets_idx: list[tuple[int, int]] = []
    onsets_val: list[float] = []
    notes_seen: set[tuple[int, int]] = set()
    onsets_seen: set[tuple[int, int]] = set()
    for n in notes:
        bin_idx = _midi_to_note_bin(n.midi_int)
        if bin_idx is None:
            continue
        f_start = int(round(n.onset_s / ANNOTATION_HOP))
        f_end = int(round((n.onset_s + n.duration_s) / ANNOTATION_HOP))
        if f_start >= n_time_frames:
            continue
        f_end = min(f_end, n_time_frames - 1)
        f_start_clamped = max(f_start, 0)
        on_key = (f_start_clamped, bin_idx)
        if on_key not in onsets_seen:
            onsets_seen.add(on_key)
            onsets_idx.append(on_key)
            onsets_val.append(1.0)
        for f in range(f_start_clamped, f_end + 1):
            key = (f, bin_idx)
            if key not in notes_seen:
                notes_seen.add(key)
                notes_idx.append(key)
                notes_val.append(1.0)

    contours_idx: list[tuple[int, int]] = []
    contours_val: list[float] = []
    contours_seen: set[tuple[int, int]] = set()
    for c in contours:
        f = int(round(c.time_s / ANNOTATION_HOP))
        if f < 0 or f >= n_time_frames:
            continue
        bin_idx = _freq_to_contour_bin(c.frequency_hz)
        if bin_idx is None:
            continue
        key = (f, bin_idx)
        if key in contours_seen:
            continue
        contours_seen.add(key)
        contours_idx.append(key)
        contours_val.append(max(0.0, min(1.0, c.confidence)))

    notes_idx.sort()
    onsets_idx_sorted = sorted(range(len(onsets_idx)), key=lambda i: onsets_idx[i])
    onsets_idx = [onsets_idx[i] for i in onsets_idx_sorted]
    onsets_val = [onsets_val[i] for i in onsets_idx_sorted]
    contours_perm = sorted(range(len(contours_idx)), key=lambda i: contours_idx[i])
    contours_idx = [contours_idx[i] for i in contours_perm]
    contours_val = [contours_val[i] for i in contours_perm]
    return notes_idx, notes_val, onsets_idx, onsets_val, contours_idx, contours_val


# --- audio loading ----------------------------------------------------------


def _load_resample_to_wav_bytes(audio_path: str) -> tuple[bytes, float]:
    """Load WAV, downmix to mono, resample to 22050, return WAV-encoded bytes + duration."""
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    duration_s = len(y) / float(AUDIO_SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(buf, y, AUDIO_SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return buf.getvalue(), duration_s


# --- TFRecord serialization -------------------------------------------------


def _build_tf_example(
    file_id: str,
    source: str,
    encoded_wav: bytes,
    notes_idx: Sequence[tuple[int, int]],
    notes_val: Sequence[float],
    onsets_idx: Sequence[tuple[int, int]],
    onsets_val: Sequence[float],
    contours_idx: Sequence[tuple[int, int]],
    contours_val: Sequence[float],
    notes_onsets_shape: tuple[int, int],
    contours_shape: tuple[int, int],
):
    import tensorflow as tf  # local import to keep module load fast

    def _bytes_feature(v):
        if isinstance(v, tf.Tensor):
            v = v.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _serialize_int64(arr):
        return tf.io.serialize_tensor(np.asarray(arr, dtype=np.int64))

    def _serialize_float32(arr):
        return tf.io.serialize_tensor(np.asarray(arr, dtype=np.float32))

    notes_idx_np = np.asarray(notes_idx, dtype=np.int64) if notes_idx else np.zeros((0, 2), np.int64)
    onsets_idx_np = np.asarray(onsets_idx, dtype=np.int64) if onsets_idx else np.zeros((0, 2), np.int64)
    contours_idx_np = np.asarray(contours_idx, dtype=np.int64) if contours_idx else np.zeros((0, 2), np.int64)

    feature = {
        'file_id': _bytes_feature(file_id.encode('utf-8')),
        'source': _bytes_feature(source.encode('utf-8')),
        'audio_wav': _bytes_feature(encoded_wav),
        'notes_indices': _bytes_feature(_serialize_int64(notes_idx_np).numpy()),
        'notes_values': _bytes_feature(_serialize_float32(notes_val).numpy()),
        'onsets_indices': _bytes_feature(_serialize_int64(onsets_idx_np).numpy()),
        'onsets_values': _bytes_feature(_serialize_float32(onsets_val).numpy()),
        'contours_indices': _bytes_feature(_serialize_int64(contours_idx_np).numpy()),
        'contours_values': _bytes_feature(_serialize_float32(contours_val).numpy()),
        'notes_onsets_shape': _bytes_feature(_serialize_int64(notes_onsets_shape).numpy()),
        'contours_shape': _bytes_feature(_serialize_int64(contours_shape).numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# --- track iteration --------------------------------------------------------


def _list_track_ids(data_home: str) -> list[str]:
    ann_dir = os.path.join(data_home, 'annotation')
    if not os.path.isdir(ann_dir):
        return []
    return sorted(
        name[:-len('.jams')]
        for name in os.listdir(ann_dir)
        if name.endswith('.jams')
    )


def _audio_path_for(track_id: str, data_home: str) -> str:
    return os.path.join(data_home, 'audio_mono-mic', f'{track_id}_mic.wav')


def _process_track(track_id: str, data_home: str, source_name: str = 'guitarset'):
    audio_path = _audio_path_for(track_id, data_home)
    jams_path = os.path.join(data_home, 'annotation', f'{track_id}.jams')
    encoded_wav, duration_s = _load_resample_to_wav_bytes(audio_path)
    notes, contours = _parse_jams(jams_path)
    n_time_frames = int(math.ceil(duration_s / ANNOTATION_HOP)) + 1
    notes_idx, notes_val, onsets_idx, onsets_val, contours_idx, contours_val = (
        _build_sparse_arrays(notes, contours, n_time_frames)
    )
    return _build_tf_example(
        file_id=track_id,
        source=source_name,
        encoded_wav=encoded_wav,
        notes_idx=notes_idx,
        notes_val=notes_val,
        onsets_idx=onsets_idx,
        onsets_val=onsets_val,
        contours_idx=contours_idx,
        contours_val=contours_val,
        notes_onsets_shape=(n_time_frames, N_FREQ_BINS_NOTES),
        contours_shape=(n_time_frames, N_FREQ_BINS_CONTOURS),
    )


# --- splitting --------------------------------------------------------------


def _player_id_from_track(track_id: str) -> str:
    """GuitarSet track_ids start with `NN_…` where NN is player index 00-05.

    Splitting by player is the plan §5 requirement so player identity does
    not leak between train and validation.
    """
    return track_id.split('_', 1)[0]


def _split_tracks(
    track_ids: Sequence[str],
    train_pct: float,
    validation_pct: float,
    *,
    per_track: bool = False,
) -> dict[str, list[str]]:
    """Split tracks into train/validation/test.

    Default: by-player (plan §5 — no player identity leakage). Use
    `per_track=True` for smoke tests where we feed only a handful of clips
    and need each split non-empty.
    """
    if per_track:
        n = len(track_ids)
        n_train = max(1, int(round(n * train_pct)))
        n_val = max(1, int(round(n * validation_pct)))
        if n_train + n_val > n:
            n_val = max(1, n - n_train)
        return {
            'train': list(track_ids[:n_train]),
            'validation': list(track_ids[n_train:n_train + n_val]),
            'test': list(track_ids[n_train + n_val:]),
        }

    by_player: dict[str, list[str]] = {}
    for tid in track_ids:
        by_player.setdefault(_player_id_from_track(tid), []).append(tid)

    players = sorted(by_player.keys())
    n_train = max(1, int(round(len(players) * train_pct)))
    n_val = max(1, int(round(len(players) * validation_pct)))
    if n_train + n_val > len(players):
        n_val = max(1, len(players) - n_train)
    train_players = players[:n_train]
    val_players = players[n_train:n_train + n_val]
    test_players = players[n_train + n_val:]

    out = {'train': [], 'validation': [], 'test': []}
    for p in train_players:
        out['train'].extend(by_player[p])
    for p in val_players:
        out['validation'].extend(by_player[p])
    for p in test_players:
        out['test'].extend(by_player[p])
    return out


# --- main -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-home', default=DEFAULT_DATA_HOME)
    ap.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                    help='base directory for tfrecord output (the script appends '
                         'guitarset/splits/{split}/00.tfrecord)')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only the first N tracks (smoke test)')
    ap.add_argument('--train-pct', type=float, default=0.7)
    ap.add_argument('--validation-pct', type=float, default=0.2)
    ap.add_argument('--per-track-split', action='store_true',
                    help='split per-track instead of per-player (use for smoke tests with few tracks)')
    args = ap.parse_args(argv)

    track_ids = _list_track_ids(args.data_home)
    if not track_ids:
        print(f'no tracks under {args.data_home}', file=sys.stderr)
        return 1
    if args.limit is not None:
        track_ids = track_ids[:args.limit]
    print(f'processing {len(track_ids)} tracks', file=sys.stderr)

    splits = _split_tracks(
        track_ids, args.train_pct, args.validation_pct,
        per_track=args.per_track_split,
    )
    for k, v in splits.items():
        print(f'  split {k}: {len(v)} tracks ({v[:3]}{"..." if len(v) > 3 else ""})',
              file=sys.stderr)

    import tensorflow as tf  # noqa

    out_root = os.path.join(args.output_dir, 'guitarset', 'splits')
    written = 0
    for split_name, tids in splits.items():
        if not tids:
            continue
        split_dir = os.path.join(out_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, '00.tfrecord')
        with tf.io.TFRecordWriter(out_path) as w:
            for tid in tids:
                try:
                    ex = _process_track(tid, args.data_home)
                except Exception as exc:  # noqa: BLE001
                    print(f'  ! {tid}: {exc}', file=sys.stderr)
                    continue
                w.write(ex.SerializeToString())
                written += 1
                print(f'  wrote {split_name}/{tid}', file=sys.stderr)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  -> {out_path} ({size_mb:.1f} MB)', file=sys.stderr)

    print(f'done: {written} examples', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
