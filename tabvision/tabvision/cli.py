"""TabVision CLI entry point — see SPEC.md §3.3, §7 Phase 1.

Phase 1 deliverable: ``tabvision transcribe input.mov -o output.tab``.
Phase 3 will add ``tabvision check input.mov`` for preflight only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tabvision.errors import TabVisionError

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if args.version:
            from tabvision import __version__

            print(f"tabvision {__version__}")
            return 0
        parser.print_help()
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "transcribe":
            return _cmd_transcribe(args)
    except TabVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabvision")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=False)

    t = sub.add_parser("transcribe", help="transcribe a video to ASCII tab")
    t.add_argument("input", type=Path, help="input video file")
    t.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output .tab file; stdout if omitted",
    )
    t.add_argument(
        "--audio-backend",
        choices=["basicpitch", "highres", "highres-fl"],
        default="basicpitch",
        help=(
            "audio transcription backend. 'basicpitch' (Phase 1, Apache-2.0) "
            "is fast/CPU-only. 'highres' (Phase 2) wraps Riley/Edwards + "
            "Cwitkowitz GAPS via hf-midi-transcription (MIT) — needs torch + "
            "extras. 'highres-fl' uses the Francois Leduc checkpoint."
        ),
    )
    t.add_argument("--capo", type=int, default=0, help="capo fret (0-7)")
    t.add_argument(
        "--instrument",
        choices=["acoustic", "classical", "electric"],
        default="acoustic",
    )
    t.add_argument("--tone", choices=["clean", "distorted"], default="clean")
    t.add_argument(
        "--style",
        choices=["fingerstyle", "strumming", "mixed"],
        default="mixed",
    )

    return parser


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Phase 1 audio-only end-to-end (extends in Phase 2 with highres backend)."""
    from tabvision.demux import demux
    from tabvision.fusion import fuse
    from tabvision.render.ascii import render
    from tabvision.types import GuitarConfig, SessionConfig

    cfg = GuitarConfig(capo=args.capo)
    session = SessionConfig(
        instrument=args.instrument, tone=args.tone, style=args.style
    )

    logger.info("demuxing %s", args.input)
    demuxed = demux(args.input)
    logger.info(
        "audio: %d samples @ %d Hz (%.1fs); video: %.1f fps (%.1fs total)",
        demuxed.wav.size,
        demuxed.sample_rate,
        demuxed.wav.size / demuxed.sample_rate,
        demuxed.fps,
        demuxed.duration_s,
    )

    backend = _make_audio_backend(args.audio_backend)
    logger.info("transcribing audio with %s", backend.name)
    audio_events = backend.transcribe(demuxed.wav, demuxed.sample_rate, session)
    logger.info("audio backend produced %d events", len(audio_events))

    # Phase 1: video stubbed; pass empty fingerings → fusion takes audio-only path.
    fingerings: list = []
    tab_events = fuse(audio_events, fingerings, cfg, session)
    logger.info("fusion produced %d tab events", len(tab_events))

    output = render(tab_events, cfg)
    if args.output:
        args.output.write_text(output)
        logger.info("wrote %s", args.output)
    else:
        sys.stdout.write(output)

    return 0


def _make_audio_backend(name: str):
    from tabvision.audio.backend import make

    return make(name)


if __name__ == "__main__":
    sys.exit(main())
