"""Accuracy-loop Q2 (ROI deep-dive §3.2) — SynthTab symbolic corpus extraction.

S1a consumed this same substrate as *counts* — per-pitch position marginals
and singleton transitions — and closed CI-negative on every arm
(`s1a_synthtab_priors_2026-07-20.md`). The diagnosis recorded there was that
counts discard the thing that carries the signal: **sequence context**. The
Phase 0 segment gate measures that context at +0.1446 ambiguous top-1, and
the two 2024-25 papers that own this task (MIDI-to-Tab, Fretting-Transformer)
get it from a masked-string model pretrained on symbolic tabs at scale.

This script extracts what such a model needs: per-track **note sequences**
(pitch, string, fret, onset), not counts. Parsing is delegated to S1a's
``_track_events`` so the corpus is the same substrate the banked S1a numbers
came from — same tempo map, same standard-tuning filter, same SynthTab
``string_index`` 1=high-E → repo 0=low-E flip, same 24-fret bound.

Output is a single compressed ``.npz`` of concatenated int arrays plus
per-track offsets, so the trainer can memory-map windows without re-reading
a 1.1 GB zip. Nothing here trains, registers, or touches the pipeline.

SynthTab is CC-BY-NC-4.0 (LICENSES.md); any artifact derived from this
corpus inherits NC and must be labeled before registration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.eval.build_synthtab_v1_prior import (
    ACOUSTIC_PROGRAMS,
    GUITAR_PROGRAMS,
    _read_ppq,
    _track_events,
)

SCHEMA_VERSION = 1
DEFAULT_ZIP = "all_jams_midi_V2_60000_tracks.zip"


def extract_corpus(
    *,
    zip_path: Path,
    variant: str,
    max_tracks: int = 0,
    progress_every: int = 5000,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Walk the SynthTab archive and collect per-track note sequences."""
    programs = ACOUSTIC_PROGRAMS if variant == "acoustic" else GUITAR_PROGRAMS

    onsets_ms: list[np.ndarray] = []
    pitches: list[np.ndarray] = []
    strings: list[np.ndarray] = []
    frets: list[np.ndarray] = []
    track_lengths: list[int] = []
    track_programs: list[int] = []

    scanned = eligible = parsed = 0
    skipped_program = skipped_nonstandard = skipped_unreadable = 0
    program_histogram: Counter[int] = Counter()
    started = time.perf_counter()

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        jams_names = [name for name in names if name.endswith(".jams")]
        members_by_dir: dict[str, list[str]] = {}
        for name in names:
            members_by_dir.setdefault(name.rsplit("/", 1)[0], []).append(name)

        for jams_name in jams_names:
            if max_tracks and parsed >= max_tracks:
                break
            scanned += 1
            try:
                raw = json.loads(archive.read(jams_name))
            except (ValueError, OSError, zipfile.BadZipFile):
                skipped_unreadable += 1
                continue
            program = raw.get("sandbox", {}).get("instrument")
            if not isinstance(program, int) or program not in programs:
                skipped_program += 1
                continue
            eligible += 1
            track_dir = jams_name.rsplit("/", 1)[0]
            ppq = _read_ppq(archive, track_dir, members_by_dir.get(track_dir, []))
            events = _track_events(raw, ppq)
            if events is None:
                skipped_nonstandard += 1
                continue

            parsed += 1
            program_histogram[program] += 1
            onsets_ms.append(
                np.fromiter(
                    (round(event.onset_s * 1000.0) for event in events),
                    dtype=np.int32,
                    count=len(events),
                )
            )
            pitches.append(
                np.fromiter((event.pitch_midi for event in events), np.int16, len(events))
            )
            strings.append(
                np.fromiter((event.string_idx for event in events), np.int8, len(events))
            )
            frets.append(np.fromiter((event.fret for event in events), np.int8, len(events)))
            track_lengths.append(len(events))
            track_programs.append(program)

            if progress_every and parsed % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  parsed={parsed} scanned={scanned} "
                    f"notes={sum(track_lengths)} ({elapsed:.0f}s)",
                    flush=True,
                )

    if not parsed:
        raise SystemExit(f"no eligible {variant} tracks parsed from {zip_path}")

    lengths = np.asarray(track_lengths, dtype=np.int64)
    arrays = {
        "onset_ms": np.concatenate(onsets_ms),
        "pitch": np.concatenate(pitches),
        "string": np.concatenate(strings),
        "fret": np.concatenate(frets),
        # Exclusive-scan offsets: track i is [offset[i], offset[i+1]).
        "track_offset": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "track_program": np.asarray(track_programs, dtype=np.int16),
    }
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "source_zip": zip_path.name,
        "tracks_scanned": scanned,
        "tracks_eligible": eligible,
        "tracks_parsed": parsed,
        "notes": int(lengths.sum()),
        "skipped_program": skipped_program,
        "skipped_nonstandard_tuning": skipped_nonstandard,
        "skipped_unreadable": skipped_unreadable,
        "program_histogram": dict(sorted(program_histogram.items())),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return arrays, metadata


def characterize(arrays: dict[str, np.ndarray], cluster_gap_ms: int = 80) -> dict[str, object]:
    """Corpus statistics that decide the model's tokenization and window.

    The headline is ``ambiguous_note_share``: the fraction of notes whose
    pitch is playable at more than one position under standard tuning and a
    24-fret bound. That is the slice the Phase 0 lattice scores, so it is
    what the pretraining objective has to be dense in — a corpus that is
    mostly unambiguous notes would teach the model very little about the
    decision the decoder actually gets wrong.
    """
    offsets = arrays["track_offset"]
    lengths = np.diff(offsets)
    pitch = arrays["pitch"]
    onset = arrays["onset_ms"]

    # Positions playable per pitch under standard tuning, 0-24 frets.
    open_midi = np.asarray([40, 45, 50, 55, 59, 64], dtype=np.int16)
    positions_for_pitch: dict[int, int] = {}
    for value in range(int(pitch.min()), int(pitch.max()) + 1):
        frets = value - open_midi
        positions_for_pitch[value] = int(np.count_nonzero((frets >= 0) & (frets <= 24)))
    positions = np.asarray([positions_for_pitch[int(value)] for value in pitch])

    # Cluster (chord) structure, using the decode's own 80 ms grouping.
    cluster_sizes: list[int] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        track_onsets = onset[start:end]
        if track_onsets.size == 0:
            continue
        boundaries = np.flatnonzero(np.diff(track_onsets) > cluster_gap_ms) + 1
        cluster_sizes.extend(int(size) for size in np.diff([0, *boundaries, track_onsets.size]))

    sizes = np.asarray(cluster_sizes) if cluster_sizes else np.zeros(1, dtype=np.int64)
    return {
        "tracks": int(lengths.size),
        "notes": int(lengths.sum()),
        "notes_per_track_median": float(np.median(lengths)),
        "notes_per_track_p90": float(np.percentile(lengths, 90)),
        "pitch_range": [int(pitch.min()), int(pitch.max())],
        "ambiguous_note_share": float(np.mean(positions > 1)),
        "mean_positions_per_note": float(positions.mean()),
        "string_histogram": {
            index: int(count)
            for index, count in enumerate(
                np.bincount(arrays["string"].astype(np.int64), minlength=6)
            )
        },
        "fret_zero_share": float(np.mean(arrays["fret"] == 0)),
        "clusters": int(sizes.size),
        "polyphonic_cluster_share": float(np.mean(sizes > 1)),
        "mean_cluster_size": float(sizes.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=None)
    parser.add_argument("--variant", choices=("acoustic", "all"), default="all")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    zip_path = args.zip_path or (data_root / "datasets" / "synthtab" / DEFAULT_ZIP)
    if not zip_path.is_file():
        raise SystemExit(f"SynthTab archive not found: {zip_path}")
    output = args.output or (data_root / "models" / "s1b_symbolic" / f"synthtab_{args.variant}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"extracting {args.variant} tracks from {zip_path.name}", flush=True)
    arrays, metadata = extract_corpus(
        zip_path=zip_path, variant=args.variant, max_tracks=args.max_tracks
    )
    stats = characterize(arrays)
    metadata["statistics"] = stats

    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            onset_ms=arrays["onset_ms"],
            pitch=arrays["pitch"],
            string=arrays["string"],
            fret=arrays["fret"],
            track_offset=arrays["track_offset"],
            track_program=arrays["track_program"],
        )
    metadata["output"] = str(output)
    metadata["output_bytes"] = output.stat().st_size
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
