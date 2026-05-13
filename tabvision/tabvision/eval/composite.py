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
    EventF1Result,
    TabF1Result,
    event_f1,
    tab_f1,
)
from tabvision.eval.parsers import get_parser
from tabvision.types import GuitarConfig, SessionConfig, TabEvent

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
        fail_messages = [
            i.message for i in validation.items if i.severity == "fail"
        ]
        raise ValueError(
            f"Manifest {manifest_path} has fail-severity issues: {fail_messages}"
        )

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
                errors=decompose_errors(
                    predicted, gold, onset_tolerance_s=onset_tolerance_s
                ),
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
        reports[tier] = TierReport(
            tier=tier,
            n_clips=len(results),
            n_gold_total=sum(r.n_gold for r in results),
            onset_f1=bootstrap_ci(
                onset_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed
            ),
            pitch_f1=bootstrap_ci(
                pitch_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed
            ),
            tab_f1=bootstrap_ci(
                tab_f1s, n_bootstrap=bootstrap_n, seed=bootstrap_seed
            ),
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


__all__ = [
    "ClipEvalResult",
    "CompositeReport",
    "DEFAULT_EVAL_SPLITS",
    "Predictor",
    "TierReport",
    "run_composite_eval",
]
