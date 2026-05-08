"""GuitarSet dataset loader for the audio-finetune Phase 1.

Phase 1 Week 1 deliverable per
docs/plans/2026-04-24-audio-backbone-finetune-design.md §7. Reads JAMS
files directly from disk and returns (onset, duration, midi, string, fret)
notes — no mirdata dependency at parse time, so the parser works equally
well whether the data was sideloaded from HuggingFace
(`tools/sideload_guitarset_from_hf.py`) or downloaded from Zenodo via
mirdata.

Not a full PyTorch / TF Dataset yet — Spotify's basic_pitch ships its own
TFRecord pipeline + training loop (see `basic_pitch.data.datasets.guitarset`
and `basic_pitch.train`), so we'll feed those rather than build our own
training loop. This module stays useful for evaluation, debugging, and
sanity-checking the JAMS schema.

Usage:
    python -m app.training.guitarset_dataset --download
    python -m app.training.guitarset_dataset --parse-one
    python -m app.training.guitarset_dataset --schema-stats
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# Default location mirdata uses; override via --data-home or DATA_HOME env.
DEFAULT_DATA_HOME = os.environ.get(
    'GUITARSET_HOME', os.path.expanduser('~/mir_datasets/guitarset'),
)


@dataclass
class GuitarSetNote:
    """One note from a GuitarSet JAMS file."""
    onset: float       # seconds
    duration: float    # seconds
    midi_note: int     # rounded from JAMS pitch_midi
    string: int        # 1=high E, 6=low E (mapped from JAMS string_index 0-5)
    fret: int          # rounded from JAMS fret


_OPEN_STRING_MIDI_BY_DATA_SOURCE = {
    # GuitarSet JAMS convention (matches mirdata's
    # _GUITAR_STRINGS = ["E", "A", "D", "G", "B", "e"]): data_source "0"
    # is the low-E string, "5" is the high-E string.
    '0': 40,  # low E
    '1': 45,  # A
    '2': 50,  # D
    '3': 55,  # G
    '4': 59,  # B
    '5': 64,  # high E
}
# Tab convention used by the rest of the pipeline: string 1 = high E,
# string 6 = low E.
_STRING_ID_BY_DATA_SOURCE = {
    '0': 6, '1': 5, '2': 4, '3': 3, '4': 2, '5': 1,
}


def download_partition(partition: str = 'audio_mic',
                       data_home: str = DEFAULT_DATA_HOME,
                       force_overwrite: bool = False) -> None:
    """Download a GuitarSet partition via mirdata (Zenodo).

    For environments where Zenodo is unreachable, use
    `tools/sideload_guitarset_from_hf.py` to pull from the
    `taohu/guitarset` HuggingFace mirror instead.
    """
    import mirdata
    gs = mirdata.initialize('guitarset', data_home=data_home)
    print(f'Downloading partitions: annotations + {partition} -> {data_home}',
          file=sys.stderr)
    gs.download(
        partial_download=['annotations', partition],
        force_overwrite=force_overwrite,
        cleanup=True,
    )


def list_track_ids(data_home: str = DEFAULT_DATA_HOME) -> list[str]:
    """List track_ids by globbing the annotation directory."""
    annotation_dir = os.path.join(data_home, 'annotation')
    if not os.path.isdir(annotation_dir):
        return []
    track_ids = []
    for name in os.listdir(annotation_dir):
        if name.endswith('.jams'):
            track_ids.append(name[:-len('.jams')])
    return sorted(track_ids)


def load_track_notes(track_id: str,
                     data_home: str = DEFAULT_DATA_HOME) -> list[GuitarSetNote]:
    """Parse one track's JAMS annotation directly into GuitarSetNote rows.

    GuitarSet JAMS schema: 6 `note_midi` annotations per track, one per
    string, distinguished by `annotation_metadata.data_source` ("0" = high
    E ... "5" = low E). Fret = round(MIDI) - open-string MIDI.
    """
    import json

    jams_path = os.path.join(data_home, 'annotation', f'{track_id}.jams')
    if not os.path.exists(jams_path):
        return []
    with open(jams_path) as f:
        jams = json.load(f)

    out: list[GuitarSetNote] = []
    for ann in jams.get('annotations', []):
        if ann.get('namespace') != 'note_midi':
            continue
        ds = ann.get('annotation_metadata', {}).get('data_source')
        if ds not in _OPEN_STRING_MIDI_BY_DATA_SOURCE:
            continue
        open_midi = _OPEN_STRING_MIDI_BY_DATA_SOURCE[ds]
        string_id = _STRING_ID_BY_DATA_SOURCE[ds]
        for d in ann.get('data', []) or []:
            try:
                onset = float(d['time'])
                duration = float(d['duration'])
                midi_int = int(round(float(d['value'])))
            except (KeyError, TypeError, ValueError):
                continue
            fret = midi_int - open_midi
            if fret < 0 or fret > 24:
                continue
            out.append(GuitarSetNote(
                onset=onset, duration=duration, midi_note=midi_int,
                string=string_id, fret=fret,
            ))
    out.sort(key=lambda n: (n.onset, n.string))
    return out


def schema_stats(track_ids: Optional[list[str]] = None,
                 data_home: str = DEFAULT_DATA_HOME, max_tracks: int = 10) -> dict:
    """Summary stats over a sample of tracks (used for the de-risk check)."""
    if track_ids is None:
        track_ids = list_track_ids(data_home)
    sample = track_ids[:max_tracks]

    n_notes_per_track: list[int] = []
    midi_counter: Counter = Counter()
    fret_counter: Counter = Counter()
    string_counter: Counter = Counter()
    durations: list[float] = []
    for tid in sample:
        notes = load_track_notes(tid, data_home)
        n_notes_per_track.append(len(notes))
        for n in notes:
            midi_counter[n.midi_note] += 1
            fret_counter[n.fret] += 1
            string_counter[n.string] += 1
            durations.append(n.duration)

    return {
        'tracks_sampled': len(sample),
        'total_notes': sum(n_notes_per_track),
        'mean_notes_per_track': (sum(n_notes_per_track) / len(n_notes_per_track)
                                 if n_notes_per_track else 0),
        'midi_min': min(midi_counter, default=None),
        'midi_max': max(midi_counter, default=None),
        'fret_distribution': dict(sorted(fret_counter.items())),
        'string_distribution': dict(sorted(string_counter.items())),
        'mean_note_duration': sum(durations) / len(durations) if durations else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-home', default=DEFAULT_DATA_HOME)
    ap.add_argument('--download', action='store_true',
                    help='download annotations + audio_mic partitions')
    ap.add_argument('--partition', default='audio_mic',
                    choices=['audio_mic', 'audio_mix',
                             'audio_hex_debleeded', 'audio_hex_original'])
    ap.add_argument('--parse-one', action='store_true',
                    help='parse one track to verify the JAMS schema')
    ap.add_argument('--schema-stats', action='store_true',
                    help='compute summary stats over the first 10 tracks')
    ap.add_argument('--max-tracks', type=int, default=10)
    args = ap.parse_args()

    if args.download:
        download_partition(args.partition, data_home=args.data_home)
        print('done', file=sys.stderr)
        return 0

    track_ids = list_track_ids(args.data_home)
    if not track_ids:
        print('no tracks found — run with --download first', file=sys.stderr)
        return 1
    print(f'GuitarSet tracks: {len(track_ids)} '
          f'(first: {track_ids[0]})', file=sys.stderr)

    if args.parse_one:
        notes = load_track_notes(track_ids[0], args.data_home)
        print(f'\n{track_ids[0]}: {len(notes)} notes')
        for n in notes[:8]:
            print(f'  t={n.onset:.3f}s dur={n.duration:.3f}s '
                  f'midi={n.midi_note} string={n.string} fret={n.fret}')
        return 0

    if args.schema_stats:
        stats = schema_stats(track_ids, args.data_home, args.max_tracks)
        for k, v in stats.items():
            print(f'  {k}: {v}')
        return 0

    print('nothing to do — pass --download / --parse-one / --schema-stats',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
