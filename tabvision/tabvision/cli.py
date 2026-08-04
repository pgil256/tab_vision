"""TabVision CLI entry point — see SPEC.md §3.3, §7 Phase 1.

Phase 1 deliverable: ``tabvision transcribe input.mov -o output.tab``.
Phase 3 will add ``tabvision check input.mov`` for preflight only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import TextIO

from tabvision.errors import InvalidInputError, TabVisionError

logger = logging.getLogger(__name__)

_TRANSCRIBE_PROGRESS_PCT = {
    "preflight": 0,
    "demux": 10,
    "model_load": 20,
    "audio_inference": 35,
    "video_analysis": 60,
    "decode": 80,
    "render": 90,
    "complete": 100,
}


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
            if args.json_output:
                json_stdout = sys.stdout
                with redirect_stdout(sys.stderr):
                    return _cmd_transcribe(args, json_stdout=json_stdout)
            return _cmd_transcribe(args)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "diagnose":
            return _cmd_diagnose(args)
        if args.command == "bank-gold":
            return _cmd_bank_gold(args)
        if args.command == "review-audio":
            return _cmd_review_audio(args)
        if args.command == "clean-audio":
            return _cmd_clean_audio(args)
    except TabVisionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


def _capo_arg(value: str) -> int:
    """argparse type for ``--capo``: an integer fret in the supported 0-12 range.

    A negative or out-of-range capo silently corrupts the rendered tab (every
    pitch is shifted past the playable range), so reject it at the CLI boundary
    with a clear message instead of letting it flow into ``GuitarConfig``.
    """
    try:
        capo = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"capo must be an integer, got {value!r}") from None
    if not 0 <= capo <= 12:
        raise argparse.ArgumentTypeError(f"capo must be between 0 and 12, got {capo}")
    return capo


_TUNING_PRESETS: dict[str, tuple[int, ...]] = {
    "standard": (40, 45, 50, 55, 59, 64),
    "drop-d": (38, 45, 50, 55, 59, 64),
    "eb-standard": (39, 44, 49, 54, 58, 63),
    "d-standard": (38, 43, 48, 53, 57, 62),
    "drop-c": (36, 43, 48, 53, 57, 62),
    "dadgad": (38, 45, 50, 55, 57, 62),
    "open-g": (38, 43, 50, 55, 59, 62),
}


def _unit_interval_arg(value: str) -> float:
    """Parse one normalized video-ROI coordinate."""
    try:
        coordinate = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"ROI coordinates must be numbers between 0 and 1, got {value!r}"
        ) from None
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise argparse.ArgumentTypeError(
            f"ROI coordinates must be between 0 and 1, got {coordinate}"
        )
    return coordinate


_POSITION_PRIOR_CHOICES = ("auto", "none", "guitarset-v1", "gaps-v1")


def _position_prior_arg(value: str) -> str:
    """argparse type for ``--position-prior``: a named choice or a .json path.

    The named set matches the registered artifacts. A ``.json`` path selects
    a local personal artifact (SPEC §1.5 carve-out, 2026-08-02); existence
    and schema are validated by the inference-policy resolver so parsing
    stays filesystem-free.
    """
    if value in _POSITION_PRIOR_CHOICES or value.lower().endswith(".json"):
        return value
    raise argparse.ArgumentTypeError(
        f"must be one of {', '.join(_POSITION_PRIOR_CHOICES)} or a path to a "
        "personal position-prior .json artifact"
    )


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
    if not math.isfinite(lam) or lam < 0.0:
        msg = (
            "fusion-lambda-vision must be finite and >= 0 "
            f"(0 disables vision; negative inverts it), got {lam}"
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
        "--json",
        action="store_true",
        dest="json_output",
        help=(
            "write a machine-readable result envelope to stdout; requires "
            "--output so stdout is reserved for JSON"
        ),
    )
    t.add_argument(
        "--editor-output",
        type=Path,
        default=None,
        help=(
            "additively write the structured assisted-editor document, including "
            "Python-ranked pitch-preserving candidates"
        ),
    )
    t.add_argument(
        "--progress",
        action="store_true",
        help="write machine-readable 'PROGRESS <stage> <pct>' lines to stderr",
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
            "electric, to 'highres-ensemble' for clean acoustic (promoted "
            "2026-07-20: +0.021 aggregate Tab F1, better onset/pitch, ~2x "
            "audio inference time), else 'highres'. 'highres' "
            "(Phase 2) wraps Riley/Edwards + Cwitkowitz GAPS via "
            "hf-midi-transcription (MIT) — needs torch + extras; first run "
            "downloads the checkpoint once (~37 s). 'highres-fl' uses the "
            "Francois Leduc checkpoint. 'highres-ensemble' is the registered "
            "Phase 3 clean-acoustic GAPS+FL selector. 'basicpitch' "
            "(Phase 1, Apache-2.0) is the fast CPU-only baseline."
        ),
    )
    t.add_argument("--capo", type=_capo_arg, default=0, help="capo fret (0-12)")
    t.add_argument(
        "--tuning",
        choices=tuple(_TUNING_PRESETS),
        default="standard",
        help="guitar tuning preset (default: standard)",
    )
    t.add_argument(
        "--accuracy-mode",
        choices=["fast", "accurate"],
        default="accurate",
        help=(
            "speed/accuracy profile. 'fast' uses the lightweight Basic Pitch "
            "backend when --audio-backend is auto; 'accurate' keeps normal routing"
        ),
    )
    t.add_argument(
        "--roi",
        nargs=4,
        type=_unit_interval_arg,
        default=None,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help=(
            "optional normalized fretboard crop for video analysis, measured "
            "from the top-left of the frame"
        ),
    )
    t.add_argument(
        "--fusion-lambda-vision",
        type=_lambda_vision_arg,
        default=1.0,
        metavar="FLOAT",
        help=(
            "weight on vision evidence in fusion (default 1.0); only "
            "meaningful with --video. 0.0 disables vision entirely "
            "(audio-only Viterbi); values >1 lean more heavily on the "
            "legacy fingertip posterior. The FretCam input-odds "
            "contribution remains capped at the default decoder weights so "
            "it cannot overwhelm strong audio evidence. See SPEC §5 / "
            "Phase-5 design doc §2."
        ),
    )
    t_video = t.add_mutually_exclusive_group()
    t_video.add_argument(
        "--video",
        action="store_true",
        help=(
            "enable the video stack (opt-in since 2026-07-28 — the ungated "
            "legacy chain measured -0.15 to -0.20 aggregate Tab F1, so "
            "audio-only is the default; see DECISIONS.md). An explicit "
            "--video-backend also opts in."
        ),
    )
    t_video.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "force the audio-only default explicitly. Redundant since "
            "audio-only became the default (2026-07-28), but kept so "
            "existing callers such as the desktop shell keep working; it "
            "also wins over an explicit --video-backend."
        ),
    )
    t.add_argument(
        "--video-backend",
        choices=["legacy", "fretcam"],
        default=None,
        help=(
            "video analyzer used when the video stack is enabled; passing "
            "this flag implies --video. 'fretcam' uses the stabilized, "
            "media-clock-aligned playing-position window; 'legacy' (the "
            "default when enabled) is the ungated per-string fingertip "
            "posterior, kept for diagnostics and rollback"
        ),
    )
    t.add_argument(
        "--video-contact-evidence",
        action="store_true",
        help=(
            "EXPERIMENTAL: with --video-backend fretcam, also apply FretCam's "
            "per-finger (string, fret) contacts as a capped fusion prior. Off "
            "by default; see docs/EVAL_REPORTS/fretcam_contact_evidence_2026-07-25.md"
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
        type=_position_prior_arg,
        default="auto",
        help=(
            "pitch-to-string/fret prior. 'auto' (default) uses the "
            "hash-verified GuitarSet artifact in its validated clean "
            "acoustic, standard-tuning, capo-zero domain, and the "
            "GAPS-trained 'gaps-v1' for clean classical sessions "
            "(2026-07-20). 'none' disables it; "
            "an explicit artifact is for reproducible evaluation or rollback. "
            "A path to a personal .json artifact built by "
            "scripts/train/build_personal_prior.py is also accepted "
            "(SPEC §1.5 carve-out, 2026-08-02)."
        ),
    )
    t.add_argument(
        "--harvest-personal-labels",
        type=Path,
        default=None,
        metavar="STORE.jsonl",
        help=(
            "opt-in personal-prior label harvest (SPEC §1.5 carve-out, "
            "2026-08-02): append high-confidence (pitch, string, fret) labels "
            "— audio pitches joined against FretCam's locked position "
            "windows, kept only when exactly one playable candidate is "
            "consistent — to this JSONL store. Requires --video-backend "
            "fretcam and capo 0. Build the artifact with "
            "scripts/train/build_personal_prior.py, then use it via "
            "--position-prior."
        ),
    )
    t.add_argument(
        "--sequence-prior",
        choices=["auto", "none", "guitarset-seq-v1", "gaps-seq-v1"],
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
        choices=["auto", "none", "guitarset-timbre-v1", "acoustic-physics-v1"],
        default="auto",
        help=(
            "per-note string evidence. 'auto' (default) applies "
            "'acoustic-physics-v1' to clean steel-string acoustic in standard "
            "tuning at capo 0, and abstains elsewhere; 'none' disables it. "
            "The table is derived from published string specifications rather "
            "than fitted: it reads each note's inharmonicity to identify the "
            "string, abstaining per note when the partials are unreadable. "
            "Sealed player-05: +0.1006 Tab F1 [+0.0615, +0.1416]."
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

    b = sub.add_parser(
        "bank-gold",
        help="bank a corrected editor document as local personal training data",
    )
    b.add_argument("source", type=Path, help="original audio or video recording")
    b.add_argument("document", type=Path, help="corrected editor-document JSON")
    b.add_argument("--root", type=Path, required=True, help="local personal-data root")
    b.add_argument(
        "--no-prior",
        action="store_true",
        help="bank labelled video frames without appending position-prior labels",
    )

    review = sub.add_parser("review-audio", help="analyze a take for local review")
    review.add_argument("input", type=Path, help="audio or video recording")
    review.add_argument("--bins", type=int, default=600, help="waveform envelope bins")

    clean = sub.add_parser("clean-audio", help="render a trimmed and cleaned local WAV")
    clean.add_argument("input", type=Path, help="audio or video recording")
    clean.add_argument("output", type=Path, help="output WAV")
    clean.add_argument("--trim-start", type=float, default=0.0)
    clean.add_argument("--trim-end", type=float, default=None)
    clean.add_argument("--gain-db", type=float, default=0.0)
    clean.add_argument("--normalize", action="store_true")
    clean.add_argument("--highpass-hz", type=int, default=0)

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
    d.add_argument("--capo", type=_capo_arg, default=0, help="capo fret (0-12)")
    d.add_argument(
        "--fusion-lambda-vision",
        type=_lambda_vision_arg,
        default=1.0,
        metavar="FLOAT",
        help="weight on vision evidence in fusion (default 1.0); only meaningful with --video",
    )
    d_video = d.add_mutually_exclusive_group()
    d_video.add_argument(
        "--video",
        action="store_true",
        help="enable the video stack for the diagnostic decode (audio-only is the default)",
    )
    d_video.add_argument(
        "--no-video",
        action="store_true",
        help="force the audio-only default explicitly (kept for compatibility)",
    )
    d.add_argument(
        "--video-backend",
        choices=["legacy", "fretcam"],
        default=None,
        help="video analyzer for the diagnostic decode; passing this flag implies --video",
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


def _cmd_bank_gold(args: argparse.Namespace) -> int:
    from tabvision.personal.bank import bank_corrected_document

    summary = bank_corrected_document(
        args.source,
        args.document,
        root=args.root,
        bank_prior=not args.no_prior,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cmd_review_audio(args: argparse.Namespace) -> int:
    from tabvision.audio.review import analyze_take

    print(json.dumps(analyze_take(args.input, bins=args.bins), separators=(",", ":")))
    return 0


def _cmd_clean_audio(args: argparse.Namespace) -> int:
    from tabvision.audio.review import clean_take

    output = clean_take(
        args.input,
        args.output,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        gain_db=args.gain_db,
        normalize=args.normalize,
        highpass_hz=args.highpass_hz,
    )
    print(json.dumps({"output": str(output)}, separators=(",", ":")))
    return 0


def _cmd_transcribe(args: argparse.Namespace, *, json_stdout: TextIO | None = None) -> int:
    """Run the full transcription pipeline (demux → audio + video → fuse → render).

    Phase 5 onward: video stack is wired through ``tabvision.pipeline.run_pipeline``.
    See SPEC.md §3.1 and ``docs/plans/2026-05-06-video-pipeline-integration-design.md``.
    """
    from tabvision.pipeline import run_pipeline, run_pipeline_with_artifacts
    from tabvision.render import render
    from tabvision.types import GuitarConfig, SessionConfig

    if args.json_output and args.output is None:
        raise InvalidInputError("--json requires --output so stdout remains valid JSON")

    total_started = time.perf_counter()
    preflight_s = 0.0
    if args.roi is not None and (args.roi[0] >= args.roi[2] or args.roi[1] >= args.roi[3]):
        raise InvalidInputError("ROI requires left < right and top < bottom")
    cfg = GuitarConfig(capo=args.capo, tuning_midi=_TUNING_PRESETS[args.tuning])
    session = SessionConfig(instrument=args.instrument, tone=args.tone, style=args.style)

    def report_progress(stage: str) -> None:
        if args.progress:
            pct = _TRANSCRIBE_PROGRESS_PCT[stage]
            print(f"PROGRESS {stage} {pct}", file=sys.stderr, flush=True)

    if not args.no_preflight:
        report_progress("preflight")
        preflight_started = time.perf_counter()
        rc = _run_preflight_gate(args)
        preflight_s = time.perf_counter() - preflight_started
        if rc != 0:
            return rc

    pipeline_started = time.perf_counter()
    video_enabled, video_backend = _resolve_video_args(args)
    if args.harvest_personal_labels is not None:
        # The harvest joins audio pitches against FretCam windows, so both
        # chains must actually run; and the label store is capo-0 indexed
        # (see tabvision.fusion.personal_prior.harvest_position_labels).
        if not video_enabled or video_backend != "fretcam":
            print(
                "--harvest-personal-labels requires --video-backend fretcam",
                file=sys.stderr,
            )
            return 2
        if cfg.capo != 0:
            print(
                "--harvest-personal-labels requires capo 0; the label store is capo-0 indexed",
                file=sys.stderr,
            )
            return 2
    pipeline_kwargs = {
        "audio_backend_name": (
            "basicpitch"
            if args.accuracy_mode == "fast" and args.audio_backend == "auto"
            else args.audio_backend
        ),
        "lambda_vision": args.fusion_lambda_vision,
        "video_stride": args.video_stride,
        "video_enabled": video_enabled,
        "video_backend": video_backend,
        "video_roi": tuple(args.roi) if args.roi is not None else None,
        "contact_evidence": args.video_contact_evidence,
        "position_prior": args.position_prior,
        "sequence_prior": args.sequence_prior,
        "string_evidence": args.string_evidence,
        "audio_filters": _resolve_audio_filters(args.audio_filters),
        "cfg": cfg,
        "session": session,
        "progress_callback": report_progress if args.progress else None,
    }
    pipeline_result = None
    if args.editor_output is not None or args.harvest_personal_labels is not None:
        pipeline_result = run_pipeline_with_artifacts(args.input, **pipeline_kwargs)
        tab_events = list(pipeline_result.tab_events)
    else:
        tab_events = run_pipeline(args.input, **pipeline_kwargs)
    pipeline_s = time.perf_counter() - pipeline_started
    logger.info("pipeline produced %d tab events", len(tab_events))

    if args.harvest_personal_labels is not None and pipeline_result is not None:
        from tabvision.fusion.personal_prior import (
            append_personal_labels,
            harvest_position_labels,
        )

        harvested = harvest_position_labels(
            pipeline_result.audio_events,
            pipeline_result.position_observations,
            cfg,
        )
        append_personal_labels(
            args.harvest_personal_labels,
            harvested,
            source_media=str(args.input),
        )
        # stderr: stdout may be carrying the rendered tab or the JSON envelope.
        print(
            f"harvested {len(harvested)} personal labels from "
            f"{len(pipeline_result.position_observations)} position windows "
            f"-> {args.harvest_personal_labels}",
            file=sys.stderr,
        )

    report_progress("render")
    render_started = time.perf_counter()
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
    render_s = time.perf_counter() - render_started

    editor_path: str | None = None
    if args.editor_output is not None:
        from tabvision.assist.document import build_editor_document

        if pipeline_result is None:  # pragma: no cover - guarded by the branch above
            raise RuntimeError("editor output requires detailed pipeline artifacts")
        editor_document = build_editor_document(
            pipeline_result,
            cfg=cfg,
            source_path=args.input,
            video_enabled=video_enabled,
        )
        args.editor_output.parent.mkdir(parents=True, exist_ok=True)
        args.editor_output.write_text(
            json.dumps(editor_document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        editor_path = str(args.editor_output.resolve())

    if args.json_output:
        from tabvision.render.ascii import LOW_CONFIDENCE_THRESHOLD

        low_confidence_flags = [
            {
                "type": "low_confidence_note",
                "event_index": index,
                "onset_s": round(float(event.onset_s), 6),
                "confidence": round(float(event.confidence), 6),
            }
            for index, event in enumerate(tab_events)
            if event.confidence < LOW_CONFIDENCE_THRESHOLD
        ]
        envelope = {
            "status": "ok",
            "output_path": str(args.output.resolve()),
            "low_confidence_flags": low_confidence_flags,
            "timings": {
                "preflight_s": round(preflight_s, 6),
                "pipeline_s": round(pipeline_s, 6),
                "render_s": round(render_s, 6),
                "total_s": round(time.perf_counter() - total_started, 6),
            },
        }
        if editor_path is not None:
            envelope["editor_path"] = editor_path
        (json_stdout or sys.stdout).write(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True) + "\n"
        )

    report_progress("complete")
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


def _resolve_video_args(args: argparse.Namespace) -> tuple[bool, str]:
    """Map the video flags to ``(video_enabled, video_backend)``.

    Audio-only is the default (DECISIONS.md 2026-07-28): the ungated legacy
    chain measured −0.15 to −0.20 aggregate Tab F1 and no published figure
    uses video. ``--video`` opts in, and an explicit ``--video-backend``
    also opts in so pre-flip FretCam invocations keep their meaning.
    ``--no-video`` still forces audio-only (the desktop shell passes it)
    and wins over an explicit ``--video-backend``; argparse rejects
    combining it with ``--video``.
    """
    video_backend = args.video_backend if args.video_backend is not None else "legacy"
    video_enabled = not args.no_video and (args.video or args.video_backend is not None)
    if not video_enabled and args.fusion_lambda_vision not in (0.0, 1.0):
        logger.warning(
            "--fusion-lambda-vision has no effect while the video stack is "
            "disabled; pass --video to enable it"
        )
    return video_enabled, video_backend


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
    video_enabled, video_backend = _resolve_video_args(args)
    report_path = write_diagnose_report(
        args.input,
        output_path,
        audio_backend_name=args.audio_backend,
        lambda_vision=args.fusion_lambda_vision,
        video_stride=args.video_stride,
        video_enabled=video_enabled,
        video_backend=video_backend,
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
