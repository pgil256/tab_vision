"""Focused tests for frozen TabCNN artifact provenance and acquisition."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.acquire.tabcnn_models import (
    TabCNNArtifactError,
    download_artifact,
    verify_artifact,
)
from tabvision.eval.tabcnn_artifacts import (
    DAFX_GUITARPROFX_ONNX,
    DAFX_OFFICIAL_CHECKPOINT,
    FROZEN_ARTIFACTS,
    SHARED_CQT,
    SYNTHTAB_TABCNN_SOURCE_SHA256,
    SYNTHTAB_X4,
    TabCNNArtifact,
    artifact_manifest_json_bytes,
    default_models_root,
)


def _artifact(
    payload: bytes, *, download_url: str | None = "https://example.test/model"
) -> TabCNNArtifact:
    return TabCNNArtifact(
        artifact_id="fixture_model",
        family_id="fixture-family",
        role="model",
        filename="fixture.bin",
        relative_path="tabcnn/fixture/fixture.bin",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        source_repository_url="https://example.test/repository",
        source_revision="a" * 40,
        source_path="fixture.bin",
        source_url="https://example.test/repository/blob/revision/fixture.bin",
        download_url=download_url,
        family_source_repository_url="https://example.test/family",
        family_source_revision="b" * 40,
        license_id="CC-BY-4.0",
        license_posture="test fixture",
        overlap_labels=("fixture:no-overlap",),
    )


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        content_range: str = "",
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_frozen_manifest_serialization_records_exact_identities(tmp_path: Path) -> None:
    encoded = artifact_manifest_json_bytes(models_root=tmp_path)
    assert encoded.endswith(b"\n")
    assert encoded == artifact_manifest_json_bytes(models_root=tmp_path)

    payload = json.loads(encoded)
    assert payload["schema_version"] == 2
    assert payload["protocol_frozen_on"] == "2026-07-29"
    by_id = {item["artifact_id"]: item for item in payload["artifacts"]}

    synth = by_id[SYNTHTAB_X4.artifact_id]
    assert synth["size_bytes"] == 52_573_995
    assert synth["sha256"] == ("a5a0812844edd1dd9540170d2bcadb543b83de2066bd18b18ac13d666d511318")
    assert synth["source"]["revision"] == "6136f79d04d8627f1fec57d31cd5667db9854bbc"
    assert synth["license"]["id"].startswith("LicenseRef-")

    dafx = by_id[DAFX_GUITARPROFX_ONNX.artifact_id]
    assert dafx["sha256"] == ("8d9ce59157bdab37fb4816d32d7f29f3da0cdbf3c7876707c819af4d1f88e6b7")
    assert dafx["source"]["revision"] == "c15524a6944febe68129d26c2a89eca455b5499d"
    assert dafx["family_source"]["revision"] == ("f50309ad06dc734ddae5e3a0eda756fca221e2e7")
    assert dafx["derived_from"]["digest"] == "ce168b2cd426f81a2a78499214e40605"
    assert dafx["derived_from"]["size_bytes"] == 3_345_122
    assert dafx["derived_from"]["locally_verified"] is False
    assert dafx["official_equivalence_verified"] is False
    assert "no export program" in dafx["official_equivalence_blocker"]
    runtime_contract = dafx["runtime_contract"]
    assert runtime_contract["input_layout"] == "float32[N, 192, 9, 1]"
    assert runtime_contract["output_layout"] == "float32[N, 6, 21] LogSoftmax"
    assert "class 0 is silence" in runtime_contract["class_order"]
    assert "per-clip min-max" in runtime_contract["front_end"]
    for field in ("class_order", "front_end"):
        assert "publisher-declared" in runtime_contract[field]
        assert "independently unverified" in runtime_contract[field]
        assert "descriptive-only" in runtime_contract[field]
    evidence = runtime_contract["evidence"]
    assert any("robust-guitar-tabs/code@f50309ad" in item for item in evidence)
    assert any("tabcnn-onnx@c15524a6" in item for item in evidence)
    assert any("native fret-0..19-then-silence" in item for item in evidence)

    synth_contract = synth["runtime_contract"]
    assert "checkpoint native: frets 0..19, then silence" in synth_contract["class_order"]
    assert "amplitude_to_db(ref=max) / 80 + 1" in synth_contract["front_end"]
    assert SYNTHTAB_TABCNN_SOURCE_SHA256 == (
        "f4dfd32f90f96e0fc7ea679751aa22df8f0f79e71a5ad2a4b9663e96b8f7d069"
    )
    assert any(SYNTHTAB_TABCNN_SOURCE_SHA256 in item for item in synth_contract["evidence"])
    assert synth["official_equivalence_verified"] is None
    assert synth["official_equivalence_blocker"] is None

    cqt = by_id[SHARED_CQT.artifact_id]
    assert cqt["size_bytes"] == 696_312
    assert cqt["sha256"] == ("4e5dfa1f10f76545a30cbfd3224431503dbad943b1def78624632284e6df597a")
    assert len(FROZEN_ARTIFACTS) == 3
    assert DAFX_OFFICIAL_CHECKPOINT.license_id == "CC-BY-4.0"


def test_default_models_root_uses_tabvision_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABVISION_DATA_ROOT", str(tmp_path))
    assert default_models_root() == tmp_path / "models"
    for artifact in FROZEN_ARTIFACTS:
        assert artifact.path_below(default_models_root()).is_relative_to(tmp_path / "models")


def test_partial_http_download_resumes_and_atomically_promotes(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    artifact = _artifact(payload)
    final = artifact.path_below(tmp_path)
    final.parent.mkdir(parents=True)
    final.with_name(f"{final.name}.part").write_bytes(payload[:4])
    ranges: list[str | None] = []

    def opener(request, timeout):
        del timeout
        ranges.append(request.get_header("Range"))
        return _Response(
            payload[4:],
            status=206,
            content_range=f"bytes 4-{len(payload) - 1}/{len(payload)}",
        )

    assert download_artifact(artifact, tmp_path, opener=opener) == "downloaded"
    assert ranges == ["bytes=4-"]
    assert final.read_bytes() == payload
    assert not final.with_name(f"{final.name}.part").exists()


def test_valid_existing_file_is_idempotent_without_network(tmp_path: Path) -> None:
    payload = b"already verified"
    artifact = _artifact(payload)
    final = artifact.path_below(tmp_path)
    final.parent.mkdir(parents=True)
    final.write_bytes(payload)

    def opener(request, timeout):
        del request, timeout
        raise AssertionError("verified existing artifact must not access the network")

    assert download_artifact(artifact, tmp_path, opener=opener) == "verified"


def test_corrupt_existing_final_is_never_overwritten(tmp_path: Path) -> None:
    artifact = _artifact(b"expected")
    final = artifact.path_below(tmp_path)
    final.parent.mkdir(parents=True)
    final.write_bytes(b"corrupt!")

    with pytest.raises(TabCNNArtifactError, match="refusing to overwrite"):
        download_artifact(artifact, tmp_path)
    assert final.read_bytes() == b"corrupt!"


def test_explicit_local_source_is_verified_and_copied(tmp_path: Path) -> None:
    payload = b"controlled local source"
    artifact = _artifact(payload, download_url=None)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    root = tmp_path / "models"

    assert download_artifact(artifact, root, source_path=source) == "downloaded"
    final = artifact.path_below(root)
    assert final.read_bytes() == payload
    assert verify_artifact(final, artifact) is None


def test_git_lfs_pointer_is_rejected_explicitly(tmp_path: Path) -> None:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 10\n"
    )
    artifact = _artifact(pointer)
    source = tmp_path / "pointer.bin"
    source.write_bytes(pointer)

    with pytest.raises(TabCNNArtifactError, match="Git-LFS pointer"):
        download_artifact(artifact, tmp_path / "models", source_path=source)
