"""Unit tests for deterministic eval report generation."""

from __future__ import annotations

import json
from pathlib import Path

from tabvision.eval.runner import run_eval


def test_smoke_eval_writes_byte_identical_reports(tmp_path: Path) -> None:
    manifest = tmp_path / "missing-manifest.toml"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    first = run_eval(
        manifest_path=manifest,
        output_dir=out_a,
        scope="smoke",
        seed=7,
        timestamp="2026-05-07T00:00:00Z",
    )
    second = run_eval(
        manifest_path=manifest,
        output_dir=out_b,
        scope="smoke",
        seed=7,
        timestamp="2026-05-07T00:00:00Z",
    )

    assert first.json_bytes == second.json_bytes
    assert first.markdown == second.markdown
    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.markdown_path.read_text(encoding="utf-8") == second.markdown_path.read_text(
        encoding="utf-8"
    )
    assert "Smoke budget target: < 180 s" in first.markdown


def test_full_eval_reports_optional_manual_gates_and_automated_evidence_policy(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "clip-a"
tier = "clean_acoustic_single_line"
source = "GuitarSet"
split = "validation"
media_path = "$TABVISION_DATA_ROOT/guitarset/a.wav"
annotation_path = "$TABVISION_DATA_ROOT/guitarset/a.jams"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = run_eval(
        manifest_path=manifest,
        output_dir=tmp_path / "reports",
        scope="full",
        seed=0,
        timestamp="2026-05-07T00:00:00Z",
    )

    payload = json.loads(result.json_bytes)
    assert payload["scope"] == "full"
    assert payload["manifest"]["missing_tiers"] == [
        "clean_acoustic_strummed",
        "clean_electric",
        "distorted_electric",
    ]
    assert [row["variant"] for row in payload["ablations"]] == [
        "audio_only",
        "audio_vision",
        "audio_vision_prior",
    ]
    assert all(row["status"] == "optional_future" for row in payload["ablations"])
    assert all(row["status"] == "optional_future" for row in payload["tier_breakdown"][1:])
    assert payload["phase_debt"]["phase_3"]["preflight"]["status"] == "optional_future"
    assert payload["phase_debt"]["phase_4"]["hand"]["status"] == "optional_future"
    assert "Optional Manual Validation Gates" in result.markdown
    assert "removed_from_v1" in result.markdown
    assert "not a v1 release blocker" in result.markdown
