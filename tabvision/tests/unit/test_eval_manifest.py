"""Unit tests for eval manifest validation and debt summaries."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from tabvision.eval.manifest import REQUIRED_TIERS, validate_manifest


def test_missing_manifest_is_optional_for_v1_release(tmp_path: Path) -> None:
    missing = tmp_path / "manifest.toml"

    result = validate_manifest(missing)

    assert result.passed
    assert result.clip_count == 0
    assert result.missing_tiers == list(REQUIRED_TIERS)
    assert any(item.code == "MANIFEST_MISSING" and item.severity == "warn" for item in result.items)
    assert all(item.severity != "fail" for item in result.items)


def test_manifest_reports_missing_required_clip_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "clip-a"
tier = "clean_acoustic_single_line"
source = "GuitarSet"
split = "validation"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = validate_manifest(manifest)

    assert not result.passed
    assert result.clip_count == 1
    assert {item.code for item in result.items if item.clip_id == "clip-a"} >= {
        "MISSING_MEDIA_PATH",
        "MISSING_ANNOTATION_PATH",
    }


def test_manifest_validation_is_json_serializable_and_sorted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
[[clips]]
id = "b"
tier = "distorted_electric"
source = "EGDB"
split = "test"
media_path = "$TABVISION_DATA_ROOT/egdb/b.wav"
annotation_path = "$TABVISION_DATA_ROOT/egdb/b.jams"

[[clips]]
id = "a"
tier = "clean_acoustic_strummed"
source = "GuitarSet"
split = "validation"
media_path = "$TABVISION_DATA_ROOT/guitarset/a.wav"
annotation_path = "$TABVISION_DATA_ROOT/guitarset/a.jams"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first = validate_manifest(manifest).to_json_bytes()
    second = validate_manifest(manifest).to_json_bytes()

    assert first == second
    payload = json.loads(first)
    assert payload["clip_ids"] == ["a", "b"]
    assert payload["present_tiers"] == ["clean_acoustic_strummed", "distorted_electric"]
    assert payload["passed"] is True
    assert tomllib.loads(manifest.read_text(encoding="utf-8"))["clips"][0]["id"] == "b"
