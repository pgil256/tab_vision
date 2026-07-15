from __future__ import annotations

from tabvision.fusion.artifact_registry import (
    load_artifact_manifest,
    registered_artifact_names,
)


def test_registered_global_artifacts_are_hash_verified() -> None:
    position = load_artifact_manifest("guitarset-v1", expected_kind="position")
    sequence = load_artifact_manifest("guitarset-seq-v1", expected_kind="sequence")
    assert position.sha256 == "71f491cb7d377c163b5d08cbea69ebc7c47783059bf06eb727b9a66e8f7fd003"
    assert sequence.sha256 == "3c657db2891f6e22f4fae4c6c9025551b197218c7165779fa8712ce9f40f5e8e"
    assert position.compatible_sequence_prior == sequence.name
    assert sequence.compatible_position_prior == position.name


def test_registered_names_exclude_failed_candidates() -> None:
    assert registered_artifact_names("position") == ("guitarset-v1",)
    assert registered_artifact_names("sequence") == ("guitarset-seq-v1",)
