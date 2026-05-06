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
        if args.command == "check":
            return _cmd_check(args)
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
        "--fusion-lambda-vision",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help=(
            "weight on vision evidence in fusion (default 1.0). 0.0 "
            "disables vision entirely (audio-only Viterbi); values >1 "
            "lean more heavily on the fingertip-to-fret posterior. "
            "See SPEC §5 / Phase-5 design doc §2."
        ),
    )
    t.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "disable the video stack entirely; transcribe audio-only. "
            "Equivalent to --fusion-lambda-vision 0 plus skipping the "
            "guitar / fretboard / hand backends."
        ),
    )
    t.add_argument(
        "--video-stride",
        type=int,
        default=3,
        metavar="N",
        help=(
            "run video backends on every Nth frame (default 3 — about "
            "10 fps effective from a 30 fps source). Lower = more "
            "vision evidence + slower; higher = faster + more sparse."
        ),
    )
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
    pf = t.add_mutually_exclusive_group()
    pf.add_argument(
        "--strict",
        action="store_true",
        help="abort on any preflight warn/fail finding (default: lenient — abort only on fail)",
    )
    pf.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip preflight entirely (Phase 3 escape hatch)",
    )

    c = sub.add_parser(
        "check",
        help="run preflight on a clip and print the report (Phase 3)",
    )
    c.add_argument("input", type=Path, help="input video file")
    c.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any warn finding (default: only fail-severity exits non-zero)",
    )

    return parser


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Run the full transcription pipeline (demux → audio + video → fuse → render).

    Phase 5 onward: video stack is wired through ``tabvision.pipeline.run_pipeline``.
    See SPEC.md §3.1 and ``docs/plans/2026-05-06-video-pipeline-integration-design.md``.
    """
    from tabvision.pipeline import run_pipeline
    from tabvision.render.ascii import render
    from tabvision.types import GuitarConfig, SessionConfig

    cfg = GuitarConfig(capo=args.capo)
    session = SessionConfig(instrument=args.instrument, tone=args.tone, style=args.style)

    if not args.no_preflight:
        rc = _run_preflight_gate(args)
        if rc != 0:
            return rc

    tab_events = run_pipeline(
        args.input,
        audio_backend_name=args.audio_backend,
        lambda_vision=args.fusion_lambda_vision,
        video_stride=args.video_stride,
        video_enabled=not args.no_video,
        cfg=cfg,
        session=session,
    )
    logger.info("pipeline produced %d tab events", len(tab_events))

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


def _cmd_check(args: argparse.Namespace) -> int:
    """`tabvision check input.mov` — Phase 3 preflight only."""
    from tabvision.preflight import check, render

    report = check(args.input, strict=args.strict)
    sys.stdout.write(render(report))
    if not report.passed:
        return 1
    return 0


def _run_preflight_gate(args: argparse.Namespace) -> int:
    """Run preflight before transcription. Lenient by default."""
    from tabvision.preflight import check, render

    try:
        report = check(args.input, strict=args.strict)
    except Exception as exc:  # noqa: BLE001 — preflight should not block transcribe in degraded environments
        logger.warning("preflight skipped due to error: %s", exc)
        return 0

    has_fail = any(f.severity == "fail" for f in report.findings)
    if has_fail or (args.strict and not report.passed):
        sys.stderr.write(render(report))
        sys.stderr.write("Aborting transcription. Re-run with --no-preflight to bypass.\n")
        return 1
    if not report.passed:
        sys.stderr.write(render(report))
        sys.stderr.write("Continuing in lenient mode despite warnings.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
