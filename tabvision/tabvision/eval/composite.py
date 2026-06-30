"""Composite multi-source eval — Phase 0 per-tier baseline harness.

Reads a manifest (validated by :mod:`tabvision.eval.manifest`),
dispatches each clip's annotation through the registered parser,
runs a user-supplied predictor over the media, and aggregates per-tier
onset / pitch / tab F1 with bootstrap CIs plus the error-decomposition
buckets.

The predictor is **injected** so the harness is testable without the
heavy audio backend. Production usage wires up
:func:`tabvision.pipeline.run_pipeline` from the CLI; tests pass a
fake predictor for fast iteration.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tabvision.eval.bootstrap import BootstrapResult, bootstrap_ci
from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.eval.manifest import ManifestValidation, validate_manifest
from tabvision.eval.metrics import (
    ChordAccuracyResult,
    EventF1Result,
    TabF1Result,
    chord_instance_accuracy,
    event_f1,
    tab_f1,
)
from tabvision.eval.parsers import get_parser
from tabvision.types import AudioBackend, GuitarConfig, SessionConfig, TabEvent

Predictor = Callable[[Path, SessionConfig], list[TabEvent]]
"""``(media_path, session) -> list[TabEvent]``. The composite-eval harness
calls this once per non-train clip."""


@dataclass(frozen=True)
class ClipEvalResult:
    """Per-clip metrics + error decomposition."""

    clip_id: str
    tier: str
    source: str
    n_gold: int
    n_predicted: int
    onset: EventF1Result
    pitch: EventF1Result
    tab: TabF1Result
    chord: ChordAccuracyResult
    errors: ErrorDecomposition


@dataclass(frozen=True)
class TierReport:
    """Aggregate metrics for one tier — bootstrap CI on each F1."""

    tier: str
    n_clips: int
    n_gold_total: int
    onset_f1: BootstrapResult
    pitch_f1: BootstrapResult
    tab_f1: BootstrapResult
    chord_accuracy: BootstrapResult
    errors: ErrorDecomposition  # summed across clips in this tier


@dataclass(frozen=True)
class CompositeReport:
    """Top-level composite-eval result."""

    manifest_path: str
    manifest_validation: ManifestValidation
    per_clip: list[ClipEvalResult]
    tiers: Mapping[str, TierReport]
    bootstrap_n: int
    bootstrap_seed: int
    onset_tolerance_s: float

    def tab_f1_acceptance(self, targets: Mapping[str, float]) -> dict[str, str]:
        """Compute the pass/gap/fail status per tier vs ``targets``.

        Status semantics per design plan §5:
        - ``"pass"``: ``lower_95_CI >= target`` (the official acceptance bar)
        - ``"gap"``: ``mean >= target > lower_95_CI``
        - ``"fail"``: ``mean < target``
        - ``"missing"``: tier has no clips in this report
        """
        statuses: dict[str, str] = {}
        for tier, target in targets.items():
            report = self.tiers.get(tier)
            if report is None:
                statuses[tier] = "missing"
                continue
            mean = report.tab_f1.statistic
            lower = report.tab_f1.lower
            if lower >= target:
                statuses[tier] = "pass"
            elif mean >= target:
                statuses[tier] = "gap"
            else:
                statuses[tier] = "fail"
        return statuses


DEFAULT_EVAL_SPLITS: tuple[str, ...] = ("validation", "test")
"""Splits included in composite eval by default. ``train`` is excluded."""


def run_composite_eval(
    manifest_path: str | Path,
    *,
    predictor: Predictor,
    media_root: str | Path | None = None,
    annotation_root: str | Path | None = None,
    splits: tuple[str, ...] = DEFAULT_EVAL_SPLITS,
    cfg: GuitarConfig | None = None,
    onset_tolerance_s: float = 0.05,
    bootstrap_n: int = 10_000,
    bootstrap_seed: int = 42,
) -> CompositeReport:
    """Per-clip eval, then per-tier aggregation with bootstrap CIs.

    Raises ``ValueError`` if the manifest fails validation (fail-severity
    issues from :func:`validate_manifest`). Train-split clips are
    skipped by default; pass ``splits=("train",)`` to evaluate on them
    (useful for diagnosing training-set fit).
    """
    manifest_path = Path(manifest_path)
    validation = validate_manifest(manifest_path)
    if not validation.passed:
        fail_messages = [i.message for i in validation.items if i.severity == "fail"]
        raise ValueError(f"Manifest {manifest_path} has fail-severity issues: {fail_messages}")

    if cfg is None:
        cfg = GuitarConfig()

    payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or []

    per_clip: list[ClipEvalResult] = []
    for clip in clips:
        if clip["split"] not in splits:
            continue

        media_path = _resolve_path(clip["media_path"], media_root)
        annotation_path = _resolve_path(clip["annotation_path"], annotation_root)

        parser = get_parser(clip["annotation_format"])
        gold = parser(annotation_path, cfg)

        session = _session_from_clip(clip)
        predicted = predictor(media_path, session)

        per_clip.append(
            ClipEvalResult(
                clip_id=clip["id"],
                tier=clip["tier"],
                source=clip["source"],
                n_gold=len(gold),
                n_predicted=len(predicted),
                onset=event_f1(
                    predicted, gold, match_pitch=False, onset_tolerance_s=onset_tolerance_s
                ),
                pitch=event_f1(
                    predicted, gold, match_pitch=True, onset_tolerance_s=onset_tolerance_s
                ),
                tab=tab_f1(predicted, gold, onset_tolerance_s=onset_tolerance_s),
                chord=chord_instance_accuracy(predicted, gold),
                errors=decompose_errors(predicted, gold, onset_tolerance_s=onset_tolerance_s),
            )
        )

    tiers = _aggregate_per_tier(
        per_clip,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
    )

    return CompositeReport(
        manifest_path=str(manifest_path),
        manifest_validation=validation,
        per_clip=per_clip,
        tiers=tiers,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        onset_tolerance_s=onset_tolerance_s,
    )


def _aggregate_per_tier(
    per_clip: list[ClipEvalResult],
    *,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> dict[str, TierReport]:
    by_tier: dict[str, list[ClipEvalResult]] = {}
    for result in per_clip:
        by_tier.setdefault(result.tier, []).append(result)

    reports: dict[str, TierReport] = {}
    for tier, results in by_tier.items():
        onset_f1s = [r.onset.f1 for r in results]
        pitch_f1s = [r.pitch.f1 for r in results]
        tab_f1s = [r.tab.f1 for r in results]
        chord_accs = [r.chord.accuracy for r in results]
        reports[tier] = TierReport(
            tier=tier,
            n_clips=len(results),
            n_gold_total=sum(r.n_gold for r in results),
            onset_f1=bootstrap_ci(onset_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
            pitch_f1=bootstrap_ci(pitch_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
            tab_f1=bootstrap_ci(tab_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
            chord_accuracy=bootstrap_ci(chord_accs, n_bootstrap=bootstrap_n, seed=bootstrap_seed),
            errors=aggregate_decompositions(r.errors for r in results),
        )
    return reports


def _resolve_path(path_str: str, root: str | Path | None) -> Path:
    """Expand ``$TABVISION_DATA_ROOT`` and apply optional override.

    ``root`` (function arg) takes precedence over the env var.
    """
    expanded = path_str
    if "$TABVISION_DATA_ROOT" in path_str:
        resolved_root: str | None
        if root is not None:
            resolved_root = str(root)
        else:
            resolved_root = os.environ.get("TABVISION_DATA_ROOT")
        if not resolved_root:
            raise ValueError(
                f"Path {path_str!r} contains $TABVISION_DATA_ROOT but neither "
                f"the env var nor the function arg is set"
            )
        expanded = path_str.replace("$TABVISION_DATA_ROOT", resolved_root)
    return Path(expanded).expanduser()


def _session_from_clip(clip: dict[str, object]) -> SessionConfig:
    """Map manifest clip metadata to a :class:`SessionConfig`.

    Phase 0 defaults all clips to acoustic / clean / mixed. Per-clip
    instrument / tone / style fields can be added to the manifest
    schema in a later phase.
    """
    del clip  # unused in Phase 0
    return SessionConfig()


DEFAULT_TIER_TARGETS: Mapping[str, float] = {
    "clean_acoustic_single_line": 0.45,
    "clean_acoustic_strummed": 0.60,
    "clean_electric": 0.90,
    "distorted_electric": 0.82,
}
"""Per-tier Tab F1 acceptance targets.

Acoustic tiers use the v1 honest audio-only gates from SPEC §1.4.1
(2026-06-02): single-line >= 0.45, strummed >= 0.60. Single-line is
information-limited from audio (string/fret ambiguity), so the original
0.94 is the v1.1 video-assisted reference, not a v1 gate. Electric tiers
are deferred to v2; their numbers here are the SPEC §1.4 stretch reference
and do not gate the acoustic v1 acceptance (they are "missing" in an
acoustic-only run).
"""


def format_baseline_markdown(
    report: CompositeReport,
    *,
    targets: Mapping[str, float] = DEFAULT_TIER_TARGETS,
    backend_label: str = "<unset>",
    position_prior_label: str = "<unset>",
    eval_harness_sha: str = "<unset>",
    title: str = "Composite per-tier baseline",
) -> str:
    """Render a Phase 0 per-tier baseline report as Markdown.

    Output format follows
    ``docs/plans/2026-05-13-tab-f1-phase-0-implementation.md`` §4.1.
    """
    statuses = report.tab_f1_acceptance(targets)
    lines: list[str] = [f"# {title}", ""]

    lines.append("## Per-tier results")
    lines.append("")
    header_cells = [
        "Tier",
        "Clips",
        "Gold notes",
        "Tab F1 mean",
        "Tab F1 lower-95",
        "Target",
        "Status",
        "Onset F1",
        "Pitch F1",
    ]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for tier, target in targets.items():
        tier_report = report.tiers.get(tier)
        if tier_report is None:
            lines.append(f"| {tier} | 0 | 0 | — | — | {target:.2f} | missing | — | — |")
            continue
        tab_mean = tier_report.tab_f1.statistic
        tab_lo = tier_report.tab_f1.lower
        onset_mean = tier_report.onset_f1.statistic
        pitch_mean = tier_report.pitch_f1.statistic
        lines.append(
            f"| {tier} | {tier_report.n_clips} | {tier_report.n_gold_total} | "
            f"{tab_mean:.4f} | {tab_lo:.4f} | {target:.2f} | {statuses[tier]} | "
            f"{onset_mean:.4f} | {pitch_mean:.4f} |"
        )
    lines.append("")

    lines.append("## Chord-instance accuracy")
    lines.append("")
    lines.append(
        "Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 "
        "video-assisted target; audio-only is string-resolution-limited, like "
        "single-line Tab F1 (SPEC §1.4.1)."
    )
    lines.append("")
    lines.append("| Tier | Clips | Chord acc mean | Lower-95 |")
    lines.append("|---|---:|---:|---:|")
    for tier in targets:
        tier_report = report.tiers.get(tier)
        if tier_report is None:
            lines.append(f"| {tier} | 0 | — | — |")
            continue
        lines.append(
            f"| {tier} | {tier_report.n_clips} | "
            f"{tier_report.chord_accuracy.statistic:.4f} | "
            f"{tier_report.chord_accuracy.lower:.4f} |"
        )
    lines.append("")

    lines.append("## Per-source breakdown")
    lines.append("")
    lines.append("| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |")
    lines.append("|---|---|---:|---:|---:|---:|")
    grouped: dict[tuple[str, str], list[ClipEvalResult]] = {}
    for clip in report.per_clip:
        grouped.setdefault((clip.tier, clip.source), []).append(clip)
    for (tier, source), clips in sorted(grouped.items()):
        tab_mean = sum(c.tab.f1 for c in clips) / len(clips)
        onset_mean = sum(c.onset.f1 for c in clips) / len(clips)
        pitch_mean = sum(c.pitch.f1 for c in clips) / len(clips)
        lines.append(
            f"| {tier} | {source} | {len(clips)} | "
            f"{tab_mean:.4f} | {onset_mean:.4f} | {pitch_mean:.4f} |"
        )
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- Manifest: `{report.manifest_path}`")
    lines.append(f"- Audio backend: `{backend_label}`")
    lines.append(f"- Position prior: `{position_prior_label}`")
    lines.append(f"- Eval-harness SHA: `{eval_harness_sha}`")
    lines.append(f"- Onset tolerance: {report.onset_tolerance_s * 1000:.0f} ms")
    lines.append(
        f"- Bootstrap: N={report.bootstrap_n:,}, seed={report.bootstrap_seed}, "
        f"95% percentile interval"
    )
    lines.append("- Acceptance gate: `lower_95_CI >= target` per design plan §5")
    lines.append("")

    return "\n".join(lines) + "\n"


def format_decomposition_markdown(
    report: CompositeReport,
    *,
    title: str = "Tab F1 error decomposition",
) -> str:
    """Render the per-tier six-bucket error decomposition.

    Six buckets are populated; the apr-28 ``muted_undetectable`` seventh
    bucket is deferred until the v1 contract carries a muted/X flag.
    """
    bucket_columns = (
        "correct",
        "wrong_position_same_pitch",
        "pitch_off",
        "timing_only",
        "missed_onset",
        "extra_detection",
    )
    lines: list[str] = [f"# {title}", ""]

    lines.append("## Aggregate (all tiers)")
    lines.append("")
    from tabvision.eval.error_decomposition import aggregate_decompositions

    overall = aggregate_decompositions(c.errors for c in report.per_clip)
    lines.append("| Bucket | Count | Share of loss |")
    lines.append("|---|---:|---:|")
    shares = overall.share_of_loss()
    for col in bucket_columns:
        count = getattr(overall, col)
        if col == "correct":
            lines.append(f"| {col} | {count} | — |")
        else:
            lines.append(f"| {col} | {count} | {shares[col] * 100:.1f}% |")
    lines.append("")

    lines.append("## Per-tier breakdown")
    lines.append("")
    header_cells = ["Tier"] + list(bucket_columns)
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    for tier_name in sorted(report.tiers):
        tier_report = report.tiers[tier_name]
        row = [tier_name]
        for col in bucket_columns:
            row.append(str(getattr(tier_report.errors, col)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines) + "\n"


def make_run_pipeline_predictor(
    *,
    audio_backend_name: str,
    position_prior: str | None,
    melodic_prior_enabled: bool = False,
    video_enabled: bool = False,
    audio_filters: bool | None = None,
) -> Predictor:
    """Wrap :func:`tabvision.pipeline.run_pipeline` for composite-eval use.

    Imports ``run_pipeline`` lazily so the composite-eval CLI's --help
    works without the audio-highres extras installed.

    ``audio_filters`` mirrors the ``tabvision`` CLI's ``--audio-filters``:
    ``None`` keeps each backend's built-in default (basicpitch on, highres
    off); ``True``/``False`` forces post-detection filtering on/off — see
    ``tabvision.audio.filters``.
    """
    from tabvision.audio.backend import make as make_audio_backend  # noqa: PLC0415
    from tabvision.pipeline import run_pipeline  # noqa: PLC0415

    # Build the audio backend ONCE and reuse it across every clip. The highres
    # backend caches its model on first transcribe; rebuilding it per clip (the
    # old behaviour) reloaded the ~0.5 GB checkpoint every clip — ~10x slower,
    # and the accumulation exhausted memory partway through a 60-clip run.
    # "auto" routes per session, so it can't be prebuilt; it falls back per-clip.
    shared_backend: AudioBackend | None = (
        None
        if audio_backend_name == "auto"
        else make_audio_backend(
            audio_backend_name,
            **({} if audio_filters is None else {"filter_config": audio_filters}),
        )
    )

    def predictor(media_path: Path, session: SessionConfig) -> list[TabEvent]:
        return run_pipeline(
            str(media_path),
            audio_backend=shared_backend,
            audio_backend_name=audio_backend_name,
            position_prior=position_prior,
            melodic_prior_enabled=melodic_prior_enabled,
            video_enabled=video_enabled,
            audio_filters=audio_filters,
            session=session,
        )

    return predictor


def _resolve_audio_filters(choice: str) -> bool | None:
    """Map the ``--audio-filters`` CLI choice to a backend-config override.

    Mirrors ``tabvision.cli._resolve_audio_filters``: ``auto`` keeps each
    backend's built-in default, ``on``/``off`` force it.
    """
    if choice == "on":
        return True
    if choice == "off":
        return False
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``tabvision-composite-eval``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="tabvision-composite-eval",
        description=("Run the v1 per-tier composite eval and write a Markdown report."),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", default="highres", help="audio backend name")
    parser.add_argument(
        "--position-prior",
        default="guitarset-v1",
        help='position prior name; pass "none" to disable',
    )
    parser.add_argument("--melodic-prior", action="store_true")
    parser.add_argument(
        "--enable-video",
        action="store_true",
        help="enable video stack (default: off — Phase 0 ships audio-only)",
    )
    parser.add_argument(
        "--audio-filters",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "post-detection audio-event filtering (see tabvision.audio.filters). "
            "'auto' (default) keeps each backend's built-in default (basicpitch "
            "on, highres off); 'on'/'off' force it."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--decomposition-output",
        type=Path,
        help=(
            "optional: write the six-bucket error decomposition "
            "(port of the apr-28 7-bucket harness; muted_undetectable deferred) "
            "to this file too"
        ),
    )
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--onset-tolerance-s", type=float, default=0.05)
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="comma-separated splits to include",
    )
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--eval-harness-sha", default="<unset>")

    args = parser.parse_args(argv)

    position_prior: str | None = args.position_prior
    if position_prior and position_prior.lower() == "none":
        position_prior = None

    predictor = make_run_pipeline_predictor(
        audio_backend_name=args.backend,
        position_prior=position_prior,
        melodic_prior_enabled=args.melodic_prior,
        video_enabled=args.enable_video,
        audio_filters=_resolve_audio_filters(args.audio_filters),
    )

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    report = run_composite_eval(
        args.manifest,
        predictor=predictor,
        media_root=args.media_root,
        annotation_root=args.annotation_root,
        splits=splits,
        onset_tolerance_s=args.onset_tolerance_s,
        bootstrap_n=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
    )

    baseline_md = format_baseline_markdown(
        report,
        backend_label=args.backend,
        position_prior_label=position_prior or "none",
        eval_harness_sha=args.eval_harness_sha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(baseline_md, encoding="utf-8")

    if args.decomposition_output:
        decomp_md = format_decomposition_markdown(report)
        args.decomposition_output.parent.mkdir(parents=True, exist_ok=True)
        args.decomposition_output.write_text(decomp_md, encoding="utf-8")

    return 0


__all__ = [
    "ClipEvalResult",
    "CompositeReport",
    "DEFAULT_EVAL_SPLITS",
    "DEFAULT_TIER_TARGETS",
    "Predictor",
    "TierReport",
    "format_baseline_markdown",
    "format_decomposition_markdown",
    "main",
    "make_run_pipeline_predictor",
    "run_composite_eval",
]


if __name__ == "__main__":
    raise SystemExit(main())
