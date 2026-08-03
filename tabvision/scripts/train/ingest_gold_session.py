"""Ingest one gold-tab practice session into the local video-training corpus.

Usage (from the ``tabvision/`` package directory):

    python -m scripts.train.ingest_gold_session take.mp4 take.tab.json \
        [--prior-store ~/.tabvision/personal/labels.jsonl]

The tab file lists exactly what was played, in order, in tab convention
(string 1 = high E): ``{"notes": [{"string": 6, "fret": 3}, ...]}``.
The audio backend stamps each gold note with its performed onset (pitch-
sequence alignment; exact matches only), then labelled JPEG frames are
extracted around every onset into::

    <data-root>/personal/video_corpus/<video-stem>-<hash8>/

Re-ingesting the same (video, tab) pair overwrites the same session
directory — idempotent. ``--prior-store`` additionally appends the labels
to the personal-prior store; that append is NOT deduplicated, so pass it
only on first ingest of a session.

Posture: SPEC §1.5 carve-out (2026-08-02, widened for gold-tab sessions).
Everything written is local-only: never shipped, never a default, never in
eval corpora or published figures.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from tabvision.personal.alignment import align_gold_notes
from tabvision.personal.gold_tab import load_gold_tab
from tabvision.personal.video_corpus import ingest_frames, matches_to_personal_labels
from tabvision.types import SessionConfig


def _default_corpus_root() -> Path:
    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))
    return data_root / "personal" / "video_corpus"


def _session_id(video: Path, tab: Path) -> str:
    digest = hashlib.sha256()
    with video.open("rb") as handle:
        digest.update(handle.read(1_048_576))
    digest.update(tab.read_bytes())
    return f"{video.stem}-{digest.hexdigest()[:8]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="the recorded take (video with audio)")
    parser.add_argument("tab", type=Path, help="gold tab JSON for exactly this take")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="corpus root (default <TABVISION_DATA_ROOT>/personal/video_corpus)",
    )
    parser.add_argument(
        "--prior-store",
        type=Path,
        default=None,
        help=(
            "also append the aligned labels to this personal-prior JSONL store "
            "(source 'gold-tab'); not deduplicated across re-ingests"
        ),
    )
    parser.add_argument(
        "--audio-backend",
        default="auto",
        help="audio backend for onset stamping (default auto — the tone toggle)",
    )
    parser.add_argument(
        "--min-match",
        type=float,
        default=0.7,
        help=(
            "refuse the session when fewer than this fraction of gold notes "
            "align (default 0.7) — a diverging take is refused, not salvaged"
        ),
    )
    args = parser.parse_args(argv)

    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if not args.tab.is_file():
        parser.error(f"tab not found: {args.tab}")

    gold_notes = load_gold_tab(args.tab)

    from tabvision.demux import demux
    from tabvision.pipeline import _make_audio_backend, audio_backend_for_session

    session = SessionConfig()
    backend_name = args.audio_backend
    if backend_name == "auto":
        backend_name = audio_backend_for_session(session)
    backend = _make_audio_backend(backend_name)

    print(f"demuxing {args.video} ...", file=sys.stderr)
    demuxed = demux(args.video)
    print(f"transcribing audio with {backend_name} ...", file=sys.stderr)
    audio_events = backend.transcribe(demuxed.wav, demuxed.sample_rate, session)

    result = align_gold_notes(list(audio_events), gold_notes)
    print(
        f"aligned {len(result.matches)}/{result.gold_count} gold notes "
        f"({result.matched_fraction:.0%}) against {result.event_count} audio events",
        file=sys.stderr,
    )
    if result.matched_fraction < args.min_match:
        print(
            f"REFUSED: matched fraction {result.matched_fraction:.0%} is below "
            f"--min-match {args.min_match:.0%}. Either the tab does not describe "
            "this take, or the take diverged from the tab. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    corpus_root = args.corpus_root or _default_corpus_root()
    session_dir = corpus_root / _session_id(args.video, args.tab)
    summary = ingest_frames(
        demuxed.frame_iterator,
        result.matches,
        session_dir,
        source_media=str(args.video),
    )

    if args.prior_store is not None:
        from tabvision.fusion.personal_prior import append_personal_labels

        labels = matches_to_personal_labels(result.matches)
        append_personal_labels(args.prior_store, labels, source_media=str(args.video))
        print(f"appended {len(labels)} gold-tab labels -> {args.prior_store}", file=sys.stderr)

    print(f"session:        {summary.session_dir}")
    print(f"notes aligned:  {summary.notes}")
    print(f"frames written: {summary.frames_written}")
    print(f"rows:           {summary.rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
