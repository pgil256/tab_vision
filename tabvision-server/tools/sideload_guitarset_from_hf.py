"""Sideload GuitarSet from the taohu/guitarset HuggingFace mirror.

Workaround for environments where Zenodo (CERN) is unreachable. Downloads
the HF parquet shards directly via huggingface_hub, then writes each row's
raw WAV bytes + JAMS string to disk in mirdata's expected layout:

    <data_home>/annotation/<track_id>.jams
    <data_home>/audio_mono-mic/<track_id>_mic.wav

Bypasses the `datasets` library so we avoid pulling in torch / torchcodec
just to decode audio that's already a WAV byte stream.

Usage (from tabvision-server/):
    python tools/sideload_guitarset_from_hf.py
    python tools/sideload_guitarset_from_hf.py --data-home /custom/path
    python tools/sideload_guitarset_from_hf.py --max-tracks 5
                                                    # smoke-test 5 tracks
    python tools/sideload_guitarset_from_hf.py --include-mix
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_DATA_HOME = os.environ.get(
    'GUITARSET_HOME', os.path.expanduser('~/mir_datasets/guitarset'),
)
HF_REPO = 'taohu/guitarset'
HF_REPO_TYPE = 'dataset'
SHARD_FILES = [f'data/train-{i:05d}-of-00005.parquet' for i in range(5)]


def _download_shard(shard_filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO, filename=shard_filename, repo_type=HF_REPO_TYPE,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-home', default=DEFAULT_DATA_HOME)
    ap.add_argument('--max-tracks', type=int, default=None,
                    help='only sideload the first N tracks (for smoke testing)')
    ap.add_argument('--include-mix', action='store_true',
                    help='also write audio_mix WAVs')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing files')
    args = ap.parse_args()

    annotation_dir = os.path.join(args.data_home, 'annotation')
    audio_mic_dir = os.path.join(args.data_home, 'audio_mono-mic')
    audio_mix_dir = os.path.join(args.data_home, 'audio_mono-pickup_mix')
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(audio_mic_dir, exist_ok=True)
    if args.include_mix:
        os.makedirs(audio_mix_dir, exist_ok=True)

    import pyarrow.parquet as pq

    columns = ['track_id', 'jams', 'audio_mic']
    if args.include_mix:
        columns.append('audio_mix')

    n_processed = 0
    for shard_path in SHARD_FILES:
        print(f'Downloading {shard_path} from {HF_REPO} ...',
              file=sys.stderr, flush=True)
        local = _download_shard(shard_path)
        table = pq.read_table(local, columns=columns)
        print(f'  shard rows: {len(table)}', file=sys.stderr)

        for row in table.to_pylist():
            if args.max_tracks is not None and n_processed >= args.max_tracks:
                break
            track_id = row['track_id']

            jams_path = os.path.join(annotation_dir, f'{track_id}.jams')
            if args.force or not os.path.exists(jams_path):
                with open(jams_path, 'w') as f:
                    f.write(row['jams'])

            mic_path = os.path.join(audio_mic_dir, f'{track_id}_mic.wav')
            if args.force or not os.path.exists(mic_path):
                with open(mic_path, 'wb') as f:
                    f.write(row['audio_mic']['bytes'])

            if args.include_mix:
                mix_path = os.path.join(audio_mix_dir, f'{track_id}_mix.wav')
                if args.force or not os.path.exists(mix_path):
                    with open(mix_path, 'wb') as f:
                        f.write(row['audio_mix']['bytes'])

            n_processed += 1
            if n_processed % 25 == 0:
                print(f'  {n_processed} tracks done',
                      file=sys.stderr, flush=True)

        if args.max_tracks is not None and n_processed >= args.max_tracks:
            break

    print(f'\nDone. {n_processed} tracks sideloaded into {args.data_home}',
          file=sys.stderr)
    print(f'  {annotation_dir}', file=sys.stderr)
    print(f'  {audio_mic_dir}', file=sys.stderr)
    if args.include_mix:
        print(f'  {audio_mix_dir}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
