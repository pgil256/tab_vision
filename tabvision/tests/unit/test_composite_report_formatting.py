"""Smoke tests for the composite-eval markdown formatters (Phase 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabvision.eval.bootstrap import BootstrapResult
from tabvision.eval.composite import (
    DEFAULT_TIER_TARGETS,
    ClipEvalResult,
    CompositeReport,
    TierReport,
    format_baseline_markdown,
    format_decomposition_markdown,
)
from tabvision.eval.error_decomposition import ErrorDecomposition
from tabvision.eval.manifest import ManifestValidation
from tabvision.eval.metrics import EventF1Result, TabF1Result


def _bootstrap(value: float, lower: float, upper: float) -> BootstrapResult:
    return BootstrapResult(
        statistic=value,
        lower=lower,
        upper=upper,
        n_observations=20,
        n_bootstrap=10_000,
        confidence=0.95,
    )


def _event_f1(value: float) -> EventF1Result:
    return EventF1Result(
        precision=value,
        recall=value,
        f1=value,
        true_positives=10,
        false_positives=1,
        false_negatives=1,
    )


def _tab_f1(value: float) -> TabF1Result:
    return TabF1Result(
        precision=value,
        recall=value,
        f1=value,
        true_positives=10,
        false_positives=1,
        false_negatives=1,
    )


def _clip(tier: str, source: str, tab_value: float) -> ClipEvalResult:
    return ClipEvalResult(
        clip_id=f"{source}-{tier}-x",
        tier=tier,
        source=source,
        n_gold=12,
        n_predicted=11,
        onset=_event_f1(0.95),
        pitch=_event_f1(0.92),
        tab=_tab_f1(tab_value),
        errors=ErrorDecomposition(
            correct=10, wrong_position_same_pitch=1, missed_onset=1
        ),
    )


def _report(tmp_path: Path) -> CompositeReport:
    per_clip = [
        _clip("clean_acoustic_strummed", "GuitarSet", 0.92),
        _clip("clean_acoustic_strummed", "GuitarSet", 0.94),
        _clip("clean_acoustic_single_line", "GuitarSet", 0.62),
        _clip("clean_acoustic_single_line", "Guitar-TECHS", 0.71),
    ]
    tiers = {
        "clean_acoustic_strummed": TierReport(
            tier="clean_acoustic_strummed",
            n_clips=2,
            n_gold_total=24,
            onset_f1=_bootstrap(0.95, 0.93, 0.97),
            pitch_f1=_bootstrap(0.92, 0.90, 0.94),
            tab_f1=_bootstrap(0.93, 0.91, 0.95),
            errors=ErrorDecomposition(correct=20, wrong_position_same_pitch=2),
        ),
        "clean_acoustic_single_line": TierReport(
            tier="clean_acoustic_single_line",
            n_clips=2,
            n_gold_total=24,
            onset_f1=_bootstrap(0.95, 0.92, 0.98),
            pitch_f1=_bootstrap(0.92, 0.90, 0.95),
            tab_f1=_bootstrap(0.665, 0.55, 0.78),  # gap: mean > 0.85? no, fail
            errors=ErrorDecomposition(
                correct=10, wrong_position_same_pitch=10, missed_onset=4
            ),
        ),
    }
    validation = ManifestValidation(
        manifest_path=str(tmp_path / "manifest.toml"),
        passed=True,
        clip_count=4,
        clip_ids=["a", "b", "c", "d"],
        present_tiers=["clean_acoustic_single_line", "clean_acoustic_strummed"],
        missing_tiers=["clean_electric", "distorted_electric"],
        items=[],
    )
    return CompositeReport(
        manifest_path=str(tmp_path / "manifest.toml"),
        manifest_validation=validation,
        per_clip=per_clip,
        tiers=tiers,
        bootstrap_n=10_000,
        bootstrap_seed=42,
        onset_tolerance_s=0.05,
    )


def test_baseline_markdown_has_required_sections(tmp_path: Path) -> None:
    md = format_baseline_markdown(_report(tmp_path))

    assert "## Per-tier results" in md
    assert "## Per-source breakdown" in md
    assert "## Methodology" in md
    for tier in DEFAULT_TIER_TARGETS:
        assert tier in md


def test_baseline_markdown_status_column(tmp_path: Path) -> None:
    """The status column must categorise as pass / gap / fail / missing."""
    md = format_baseline_markdown(_report(tmp_path))

    # clean_acoustic_strummed: lower_95 = 0.91 >= 0.90 target → pass
    strum_row = next(
        line for line in md.split("\n") if line.startswith("| clean_acoustic_strummed")
    )
    assert "| pass |" in strum_row

    # clean_acoustic_single_line: mean=0.665 < 0.85 → fail
    single_row = next(
        line for line in md.split("\n") if line.startswith("| clean_acoustic_single_line")
    )
    assert "| fail |" in single_row

    # clean_electric: tier not in report → missing
    electric_row = next(line for line in md.split("\n") if line.startswith("| clean_electric"))
    assert "| missing |" in electric_row


def test_baseline_markdown_methodology_includes_settings(tmp_path: Path) -> None:
    md = format_baseline_markdown(
        _report(tmp_path),
        backend_label="highres",
        position_prior_label="guitarset-v1",
        eval_harness_sha="deadbeef",
    )
    assert "`highres`" in md
    assert "`guitarset-v1`" in md
    assert "`deadbeef`" in md
    assert "Bootstrap: N=10,000" in md
    assert "Onset tolerance: 50 ms" in md


def test_decomposition_markdown_has_aggregate_and_per_tier(tmp_path: Path) -> None:
    md = format_decomposition_markdown(_report(tmp_path))

    assert "## Aggregate (all tiers)" in md
    assert "## Per-tier breakdown" in md
    # Bucket names should appear in the aggregate table
    for bucket in (
        "correct",
        "wrong_position_same_pitch",
        "pitch_off",
        "timing_only",
        "missed_onset",
        "extra_detection",
    ):
        assert bucket in md


def test_decomposition_markdown_aggregates_per_clip(tmp_path: Path) -> None:
    """Aggregate row should sum per-clip decompositions, not duplicate per-tier."""
    md = format_decomposition_markdown(_report(tmp_path))
    # 4 clips × 10 correct each = 40
    aggregate_section = md.split("## Per-tier breakdown")[0]
    assert "| correct | 40 |" in aggregate_section


@pytest.mark.parametrize(
    "tier",
    list(DEFAULT_TIER_TARGETS),
)
def test_default_targets_cover_all_required_tiers(tier: str) -> None:
    assert tier in DEFAULT_TIER_TARGETS
    assert 0.0 < DEFAULT_TIER_TARGETS[tier] <= 1.0
