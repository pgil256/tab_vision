"""TabVision CLI entry point — see SPEC.md §3.3, §7 Phase 1.

Phase 1 deliverable: ``tabvision transcribe input.mov -o output.tab``.
Phase 3 will add ``tabvision check input.mov`` for preflight only.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from tabvision.errors import InvalidInputError, TabVisionError

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
        if args.command == "diagnose":
            return _cmd_diagnose(args)
    except TabVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _capo_arg(value: str) -> int:
    """argparse type for ``--capo``: an integer fret in the documented 0-7 range.

    A negative or out-of-range capo silently corrupts the rendered tab (every
    pitch is shifted past the playable range), so reject it at the CLI boundary
    with a clear message instead of letting it flow into ``GuitarConfig``.
    """
    try:
        capo = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"capo must be an integer, got {value!r}") from None
    if not 0 <= capo <= 7:
        raise argparse.ArgumentTypeError(f"capo must be between 0 and 7, got {capo}")
    return capo


def _video_stride_arg(value: str) -> int:
    """argparse type for ``--video-stride``: an integer frame stride >= 1.

    The pipeline raises ``ValueError`` for ``stride < 1``, but only after demux
    and the audio backend have already run, and that error escapes the CLI's
    ``TabVisionError`` handler as a raw traceback. Reject it up front instead.
    """
    try:
        stride = int(value)
    except ValueError:
        msg = f"video-stride must be an integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if stride < 1:
        raise argparse.ArgumentTypeError(f"video-stride must be >= 1, got {stride}")
    return stride


def _lambda_vision_arg(value: str) -> float:
    """argparse type for ``--fusion-lambda-vision``: a float >= 0.

    A negative weight doesn't disable the vision term the way 0.0 does — it
    flips its sign in ``playability.emission_cost``, so the decoder rewards
    fingerings the vision model considers *unlikely*. That's a silent
    correctness bug (wrong tab, no error), so reject it at the CLI boundary.
    """
    try:
        lam = float(value)
    except ValueError:
        msg = f"fusion-lambda-vision must be a number, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if lam < 0.0:
        msg = (
            f"fusion-lambda-vision must be >= 0 (0 disables vision; negative inverts it), got {lam}"
        )
        raise argparse.ArgumentTypeError(msg)
    return lam


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabvision")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=False)

    t = sub.add_parser("transcribe", help="transcribe a video to tab")
    t.add_argument("input", type=Path, help="input video file")
    t.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file; stdout if omitted",
    )
    t.add_argument(
        "--format",
        choices=["ascii", "gp5", "musicxml", "midi"],
        default="ascii",
        help="render format (default: ascii)",
    )
    t.add_argument(
        "--audio-backend",
        choices=[
            "basicpitch",
            "highres",
            "highres-fl",
            "highres-ensemble",
            "highres-electric",
            "auto",
        ],
        default="auto",
        help=(
            "audio transcription backend. 'auto' (default) is the tone "
            "toggle: routes to 'highres-electric' when --instrument "
            "electric, else 'highres' — the accepted v1 config. 'highres' "
            "(Phase 2) wraps Riley/Edwards + Cwitkowitz GAPS via "
            "hf-midi-transcription (MIT) — needs torch + extras; first run "
            "downloads the checkpoint once (~37 s). 'highres-fl' uses the "
            "Francois Leduc checkpoint. 'highres-ensemble' is the registered "
            "Phase 3 clean-acoustic GAPS+FL selector; it is explicit until the "
            "Phase 7 rollout. 'basicpitch' (Phase 1, Apache-2.0) "
            "is the fast CPU-only baseline."
        ),
    )
    t.add_argument("--capo", type=_capo_arg, default=0, help="capo fret (0-7)")
    t.add_argument(
        "--fusion-lambda-vision",
        type=_lambda_vision_arg,
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
        type=_video_stride_arg,
        default=3,
        metavar="N",
        help=(
            "run video backends on every Nth frame (default 3 — about "
            "10 fps effective from a 30 fps source). Lower = more "
            "vision evidence + slower; higher = faster + more sparse."
        ),
    )
    t.add_argument(
        "--position-prior",
        choices=["auto", "none", "guitarset-v1"],
        default="auto",
        help=(
            "pitch-to-string/fret prior. 'auto' (default) uses the "
            "hash-verified GuitarSet artifact only in its validated clean "
            "acoustic, standard-tuning, capo-zero domain. 'none' disables it; "
            "an explicit artifact is for reproducible evaluation or rollback."
        ),
    )
    t.add_argument(
        "--sequence-prior",
        choices=["auto", "none", "guitarset-seq-v1"],
        default="auto",
        help=(
            "learned fingering-sequence prior on the decode's transitions "
            "(A15). 'auto' (default) couples it to --position-prior: active "
            "iff the position prior is, at the gate-accepted weight "
            "(single-line +3.2pp real-audio Tab F1, strummed unchanged). "
            "The coupling is mandatory — uncoupled use is a banked GAPS "
            "regression (DECISIONS.md 2026-07-02). 'none' disables; an "
            "explicit artifact name forces it on. The "
            "TABVISION_TRANSITION_PRIOR env var overrides this flag for "
            "sweeps."
        ),
    )
    t.add_argument(
        "--string-evidence",
        choices=["auto", "none", "guitarset-timbre-v1"],
        default="auto",
        help=(
            "timbral string classifier evidence. 'auto' uses a registered "
            "gate-passed model in its validated domain, otherwise degrades to "
            "neutral evidence; 'none' disables it."
        ),
    )
    t.add_argument(
        "--audio-filters",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "post-detection audio-event filtering (low-quality drop, same-pitch "
            "merge, sustain/harmonic artifact removal — see tabvision.audio.filters). "
            "'auto' (default) keeps each backend's built-in default (basicpitch on, "
            "highres off); 'on'/'off' force it. Use 'on' to curb highres "
            "over-detection."
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
    t.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "colour-grade the ASCII tab by confidence when printing to a "
            "terminal — green (high) / yellow (medium) / red (low). 'auto' "
            "(default) colours only an interactive TTY and honours NO_COLOR; "
            "file output (-o) is always plain. Ignored for non-ascii formats."
        ),
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

    d = sub.add_parser(
        "diagnose",
        help="write an HTML report with overlay/audio/tab/confidence sections",
    )
    d.add_argument("input", type=Path, help="input video file")
    d.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output .html report; defaults to <input>.diagnose.html",
    )
    d.add_argument(
        "--audio-backend",
        choices=[
            "basicpitch",
            "highres",
            "highres-fl",
            "highres-ensemble",
            "highres-electric",
            "auto",
        ],
        default="basicpitch",
        help="audio transcription backend used for the diagnostic decode",
    )
    d.add_argument("--capo", type=_capo_arg, default=0, help="capo fret (0-7)")
    d.add_argument(
        "--fusion-lambda-vision",
        type=_lambda_vision_arg,
        default=1.0,
        metavar="FLOAT",
        help="weight on vision evidence in fusion (default 1.0)",
    )
    d.add_argument(
        "--no-video",
        action="store_true",
        help="disable the video stack for the diagnostic decode",
    )
    d.add_argument(
        "--video-stride",
        type=_video_stride_arg,
        default=3,
        metavar="N",
        help="run video backends on every Nth frame (default 3)",
    )
    d.add_argument(
        "--instrument",
        choices=["acoustic", "classical", "electric"],
        default="acoustic",
    )
    d.add_argument("--tone", choices=["clean", "distorted"], default="clean")
    d.add_argument("--style", choices=["fingerstyle", "strumming", "mixed"], default="mixed")
    d.add_argument(
        "--audio-filters",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "post-detection audio-event filtering for the diagnostic decode. "
            "'auto' (default) keeps the backend default; 'on'/'off' force it."
        ),
    )
    d.add_argument(
        "--no-preflight",
        action="store_true",
        help="skip preflight section generation",
    )

    return parser


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Run the full transcription pipeline (demux → audio + video → fuse → render).

    Phase 5 onward: video stack is wired through ``tabvision.pipeline.run_pipeline``.
    See SPEC.md §3.1 and ``docs/plans/2026-05-06-video-pipeline-integration-design.md``.
    """
    from tabvision.pipeline import run_pipeline
    from tabvision.render import render
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
        position_prior=args.position_prior,
        sequence_prior=args.sequence_prior,
        string_evidence=args.string_evidence,
        audio_filters=_resolve_audio_filters(args.audio_filters),
        cfg=cfg,
        session=session,
    )
    logger.info("pipeline produced %d tab events", len(tab_events))

    output = render(tab_events, args.format, cfg)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
        logger.info("wrote %s", args.output)
    elif args.format == "ascii":
        if _should_color(args.color):
            from tabvision.render.ascii import render as render_ascii

            sys.stdout.write(render_ascii(tab_events, cfg, color=True))
        else:
            sys.stdout.write(output.decode("utf-8"))
    else:
        sys.stdout.buffer.write(output)

    return 0


def _should_color(choice: str) -> bool:
    """Whether to ANSI-colour the ascii tab written to stdout.

    ``always``/``never`` force it; ``auto`` (default) colours only an
    interactive terminal and honours the ``NO_COLOR`` convention. File output
    (``-o``) never reaches here, so written tabs stay plain and byte-stable.
    """
    if choice == "always":
        return True
    if choice == "never":
        return False
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _make_audio_backend(name: str):
    from tabvision.audio.backend import make

    return make(name)


def _resolve_audio_filters(choice: str) -> bool | None:
    """Map the ``--audio-filters`` CLI choice to a ``run_pipeline`` value.

    ``auto`` → ``None`` (keep each backend's built-in default); ``on`` → ``True``;
    ``off`` → ``False`` (explicit overrides).
    """
    if choice == "on":
        return True
    if choice == "off":
        return False
    return None


def _cmd_check(args: argparse.Namespace) -> int:
    """`tabvision check input.mov` — Phase 3 preflight only."""
    from tabvision.preflight import check, render

    report = check(args.input, strict=args.strict)
    sys.stdout.write(render(report))
    if not report.passed:
        return 1
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    """`tabvision diagnose input.mov` — Phase 9 HTML report."""
    from tabvision.diagnose import write_diagnose_report
    from tabvision.types import GuitarConfig, SessionConfig

    cfg = GuitarConfig(capo=args.capo)
    session = SessionConfig(instrument=args.instrument, tone=args.tone, style=args.style)
    output_path = args.output or args.input.with_suffix(args.input.suffix + ".diagnose.html")
    report_path = write_diagnose_report(
        args.input,
        output_path,
        audio_backend_name=args.audio_backend,
        lambda_vision=args.fusion_lambda_vision,
        video_stride=args.video_stride,
        video_enabled=not args.no_video,
        preflight_enabled=not args.no_preflight,
        audio_filters=_resolve_audio_filters(args.audio_filters),
        cfg=cfg,
        session=session,
    )
    print(f"wrote {report_path}")
    return 0


def _run_preflight_gate(args: argparse.Namespace) -> int:
    """Run preflight before transcription. Lenient by default."""
    from tabvision.preflight import check, render

    try:
        report = check(args.input, strict=args.strict)
    except InvalidInputError:
        # A missing/bad input file is a real, actionable error, not a degraded
        # environment — re-raise so main()'s TabVisionError handler prints it
        # once, cleanly, instead of logging a confusing "preflight skipped"
        # warning here and then hitting the same error again from demux().
        raise
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
