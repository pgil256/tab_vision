from __future__ import annotations

import hashlib
import json

import pytest

import tabvision.fusion.artifact_registry as registry
from tabvision.errors import ConfigurationError
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
    assert registered_artifact_names("assignment_context") == ()


def test_phase2_context_artifact_is_hash_verified_but_unregistered() -> None:
    candidate = load_artifact_manifest(
        "context-v1",
        expected_kind="assignment_context",
        require_registered=False,
    )
    assert candidate.sha256 == "9d0df2eba2c99f2271e2932e7b791cb9ebb228c6b1d44bb27099fcd66ce56fea"
    assert candidate.registered is False
    package_root = candidate.artifact_path.parents[2]
    code = candidate.payload["code"]
    assert (
        code["context_reranker_sha256"]
        == hashlib.sha256((package_root / "fusion/context_reranker.py").read_bytes()).hexdigest()
    )
    assert (
        code["evaluation_script_sha256"]
        == hashlib.sha256(
            (package_root.parent / "scripts/eval/string_assignment_phase2.py").read_bytes()
        ).hexdigest()
    )
    with pytest.raises(ConfigurationError, match="Close symbolic-context expansion"):
        load_artifact_manifest("context-v1", expected_kind="assignment_context")


def test_phase2_torchscript_artifact_loads_and_masks_candidates() -> None:
    torch = pytest.importorskip("torch")
    candidate = load_artifact_manifest(
        "context-v1",
        expected_kind="assignment_context",
        require_registered=False,
    )
    model = torch.jit.load(str(candidate.artifact_path))
    event = torch.zeros((1, 2, 33))
    features = torch.zeros((1, 2, 6, 16))
    mask = torch.tensor([[[True, True, False, False, False, False]] * 2])
    padding = torch.zeros((1, 2), dtype=torch.bool)

    logits = model(event, features, mask, padding)

    assert logits.shape == (1, 2, 6)
    assert torch.all(logits[..., 2:] == -1.0e9)


def test_corrupt_registered_artifact_is_rejected(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "broken.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "position",
        "artifact_version": "broken-v1",
        "artifact_file": artifact.name,
        "artifact_sha256": "0" * 64,
        "registered": True,
    }
    (tmp_path / "broken.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(registry, "_ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(registry, "_MANIFEST_FILES", {"broken-v1": "broken.manifest.json"})
    registry.load_artifact_manifest.cache_clear()
    try:
        with pytest.raises(ConfigurationError, match="hash mismatch"):
            registry.load_artifact_manifest("broken-v1", expected_kind="position")
    finally:
        registry.load_artifact_manifest.cache_clear()
