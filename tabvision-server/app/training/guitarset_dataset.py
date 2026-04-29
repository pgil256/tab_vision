"""GuitarSet dataset loader for the audio-finetune Phase 1.

Phase 1 Week 1 deliverable per
docs/plans/2026-04-24-audio-backbone-finetune-design.md §7. Uses mirdata
to handle download + caching, exposes a thin per-track parser that returns
mono audio + (onset, midi, string, fret, end) annotations from JAMS.

Not a full PyTorch Dataset yet — that arrives once the Basic Pitch training
loop is forked and its expected tensor shapes are known. This file is the
data-side de-risk: confirm the JAMS schema is what we think it is and that
mirdata reliably loads it before we commit to the training-loop fork.

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


def _initialize(data_home: str = DEFAULT_DATA_HOME):
    import mirdata
    return mirdata.initialize('guitarset', data_home=data_home)


def download_partition(partition: str = 'audio_mic',
                       data_home: str = DEFAULT_DATA_HOME,
                       force_overwrite: bool = False) -> None:
    """Download a GuitarSet partition (audio + annotations).

    Partitions:
      annotations         — JAMS files only (~6 MB)
      audio_mic           — mono mic recording (~580 MB)
      audio_hex_debleeded — hex pickup, per-string (~1.4 GB)
      audio_hex_original  — hex pickup, raw (~1.4 GB)
      audio_mix           — mono mixed pickup (~580 MB)

    For Basic Pitch fine-tuning the audio_mic + annotations partitions are
    sufficient (Basic Pitch is mono-input).
    """
    gs = _initialize(data_home)
    print(f'Downloading partitions: annotations + {partition} -> {data_home}',
          file=sys.stderr)
    gs.download(
        partial_download=['annotations', partition],
        force_overwrite=force_overwrite,
        cleanup=True,
    )


def list_track_ids(data_home: str = DEFAULT_DATA_HOME) -> list[str]:
    """Return all GuitarSet track_ids (e.g., '00_BN1-129-Eb_solo')."""
    gs = _initialize(data_home)
    return sorted(gs.track_ids)


def load_track_notes(track_id: str,
                     data_home: str = DEFAULT_DATA_HOME) -> list[GuitarSetNote]:
    """Parse one track's JAMS annotation into a list of GuitarSetNote.

    GuitarSet JAMS schema: each track has 6 `note_midi` annotations (one per
    string). Each annotation contains intervals + labeled MIDI pitch values.
    Fret is derived from the MIDI - open-string MIDI for that string.
    """
    gs = _initialize(data_home)
    track = gs.track(track_id)

    # mirdata exposes `notes_all` which merges all 6 string annotations.
    notes_all = track.notes_all
    if notes_all is None:
        return []

    # Per-string: mirdata exposes `notes` as dict[string_idx, notes_obj] with
    # string indices 0-5 (1=high E in mirdata convention; we'll re-map).
    # Use that to recover string + fret per note.
    per_string = track.notes
    out: list[GuitarSetNote] = []
    if per_string is None:
        return []
    open_string_midi_by_idx = {0: 64, 1: 59, 2: 55, 3: 50, 4: 45, 5: 40}
    string_id_by_idx = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    for s_idx, ns in per_string.items():
        if ns is None:
            continue
        intervals = ns.intervals
        pitches = ns.pitches  # MIDI numbers (may be float)
        for (start, end), midi in zip(intervals, pitches):
            midi_int = int(round(midi))
            fret = midi_int - open_string_midi_by_idx[s_idx]
            if fret < 0 or fret > 24:
                # Skip clearly impossible mapping (likely string-index mismatch)
                continue
            out.append(GuitarSetNote(
                onset=float(start),
                duration=float(end - start),
                midi_note=midi_int,
                string=string_id_by_idx[s_idx],
                fret=fret,
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
