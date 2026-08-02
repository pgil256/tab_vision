"""Frozen artifact identities for the TabCNN complementarity experiment.

This module contains metadata only. Importing it never creates directories,
opens model files, or accesses the network. Model bytes are evaluation-only
and live below ``$TABVISION_DATA_ROOT/models`` rather than in the repository.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

TABVISION_DATA_ROOT_ENV = "TABVISION_DATA_ROOT"
DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"
MANIFEST_SCHEMA_VERSION = 2
PROTOCOL_FROZEN_ON = "2026-07-29"

SYNTHTAB_TABCNN_SOURCE_SHA256 = "f4dfd32f90f96e0fc7ea679751aa22df8f0f79e71a5ad2a4b9663e96b8f7d069"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")

ArtifactRole = Literal["model", "front_end"]
DigestAlgorithm = Literal["md5", "sha256"]


@dataclass(frozen=True)
class RuntimeContract:
    """Source-backed runtime interpretation of one executable artifact."""

    input_layout: str
    output_layout: str
    class_order: str
    front_end: str
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        fields = (self.input_layout, self.output_layout, self.class_order, self.front_end)
        if any(not value.strip() for value in fields):
            raise ValueError("runtime contract fields must be non-empty")
        if not self.evidence or any(not item.strip() for item in self.evidence):
            raise ValueError("runtime contract must include non-empty source evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_layout": self.input_layout,
            "output_layout": self.output_layout,
            "class_order": self.class_order,
            "front_end": self.front_end,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class PublishedArtifactIdentity:
    """Upstream identity retained even when a derived transport is executed."""

    name: str
    record_url: str
    source_url: str
    size_bytes: int
    digest_algorithm: DigestAlgorithm
    digest: str
    license_id: str
    locally_verified: bool

    def __post_init__(self) -> None:
        expected = _MD5_RE if self.digest_algorithm == "md5" else _SHA256_RE
        if self.size_bytes <= 0:
            raise ValueError("published artifact size must be positive")
        if expected.fullmatch(self.digest) is None:
            raise ValueError(f"invalid {self.digest_algorithm} digest for {self.name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "record_url": self.record_url,
            "source_url": self.source_url,
            "size_bytes": self.size_bytes,
            "digest_algorithm": self.digest_algorithm,
            "digest": self.digest,
            "license_id": self.license_id,
            "locally_verified": self.locally_verified,
        }


@dataclass(frozen=True)
class TabCNNArtifact:
    """One executable model or front-end artifact with immutable provenance."""

    artifact_id: str
    family_id: str
    role: ArtifactRole
    filename: str
    relative_path: str
    size_bytes: int
    sha256: str
    source_repository_url: str
    source_revision: str
    source_path: str
    source_url: str
    download_url: str | None
    family_source_repository_url: str
    family_source_revision: str
    license_id: str
    license_posture: str
    overlap_labels: tuple[str, ...]
    runtime_contract: RuntimeContract | None = None
    derived_from: PublishedArtifactIdentity | None = None
    official_equivalence_verified: bool | None = None
    official_equivalence_blocker: str | None = None

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"relative_path must stay below the models root: {self.relative_path}")
        if relative.name != self.filename:
            raise ValueError("filename must match the final component of relative_path")
        if self.size_bytes <= 0:
            raise ValueError("artifact size must be positive")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError(f"invalid SHA-256 for {self.artifact_id}")
        if not self.overlap_labels:
            raise ValueError(f"{self.artifact_id} must declare overlap labels")
        if self.derived_from is None:
            if (
                self.official_equivalence_verified is not None
                or self.official_equivalence_blocker is not None
            ):
                raise ValueError(
                    "official equivalence fields require a published derived_from identity"
                )
        elif self.official_equivalence_verified is None:
            raise ValueError("derived artifacts must state official_equivalence_verified")
        elif self.official_equivalence_verified:
            if self.official_equivalence_blocker is not None:
                raise ValueError("verified official equivalence cannot have a blocker")
        elif not self.official_equivalence_blocker:
            raise ValueError("unverified official equivalence must state a blocker")

    def path_below(self, models_root: str | Path) -> Path:
        """Resolve this artifact below an explicit models root."""

        return Path(models_root).joinpath(*PurePosixPath(self.relative_path).parts)

    def to_dict(self, *, models_root: str | Path | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "family_id": self.family_id,
            "role": self.role,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "source": {
                "repository_url": self.source_repository_url,
                "revision": self.source_revision,
                "path": self.source_path,
                "url": self.source_url,
                "download_url": self.download_url,
            },
            "family_source": {
                "repository_url": self.family_source_repository_url,
                "revision": self.family_source_revision,
            },
            "license": {
                "id": self.license_id,
                "posture": self.license_posture,
            },
            "overlap_labels": list(self.overlap_labels),
            "runtime_contract": (
                self.runtime_contract.to_dict() if self.runtime_contract else None
            ),
            "derived_from": self.derived_from.to_dict() if self.derived_from else None,
            "official_equivalence_verified": self.official_equivalence_verified,
            "official_equivalence_blocker": self.official_equivalence_blocker,
        }
        if models_root is not None:
            payload["load_path"] = str(self.path_below(models_root))
        return payload


DAFX_OFFICIAL_CHECKPOINT = PublishedArtifactIdentity(
    name="best_TabCNN_tablature_trancription_model",
    record_url="https://zenodo.org/records/11406378",
    source_url=(
        "https://zenodo.org/records/11406378/files/"
        "best_TabCNN_tablature_trancription_model?download=1"
    ),
    size_bytes=3_345_122,
    digest_algorithm="md5",
    digest="ce168b2cd426f81a2a78499214e40605",
    license_id="CC-BY-4.0",
    # The identity comes from the published record. The local experiment ran
    # the separately SHA-256-pinned ONNX transport below.
    locally_verified=False,
)

SYNTHTAB_X4 = TabCNNArtifact(
    artifact_id="synthtab_pretrained_x4",
    family_id="synthtab-tabcnn-x4",
    role="model",
    filename="SynthTab-Pretrained.pt",
    relative_path="tabcnn/synthtab/SynthTab-Pretrained.pt",
    size_bytes=52_573_995,
    sha256="a5a0812844edd1dd9540170d2bcadb543b83de2066bd18b18ac13d666d511318",
    source_repository_url="https://github.com/yongyizang/SynthTab",
    source_revision="6136f79d04d8627f1fec57d31cd5667db9854bbc",
    source_path="demo_embedding/pretrained_models/SynthTab-Pretrained.pt",
    source_url=(
        "https://github.com/yongyizang/SynthTab/blob/"
        "6136f79d04d8627f1fec57d31cd5667db9854bbc/"
        "demo_embedding/pretrained_models/SynthTab-Pretrained.pt"
    ),
    download_url=(
        "https://raw.githubusercontent.com/yongyizang/SynthTab/"
        "6136f79d04d8627f1fec57d31cd5667db9854bbc/"
        "demo_embedding/pretrained_models/SynthTab-Pretrained.pt"
    ),
    family_source_repository_url="https://github.com/yongyizang/SynthTab",
    family_source_revision="6136f79d04d8627f1fec57d31cd5667db9854bbc",
    license_id="LicenseRef-SynthTab-Weights-NonCommercial-Ambiguous",
    license_posture=(
        "Internal evaluation only: the dataset is described as CC-BY-NC-4.0 "
        "while the repository is CC-BY; redistribution requires separate clearance."
    ),
    overlap_labels=(
        "training:synthetic-only",
        "guitarset:transfer-no-declared-overlap",
        "gaps:classical-transfer",
        "egset12:electric-transfer",
        "guitar-techs:independent-electric-transfer",
    ),
    runtime_contract=RuntimeContract(
        input_layout="float32[N, 192, 9, 1]",
        output_layout="float32[N, 6, 21] probabilities after local softmax and remap",
        class_order=(
            "checkpoint native: frets 0..19, then silence; runtime shared: "
            "silence, then frets 0..19"
        ),
        front_end=(
            "22,050 Hz mono; peak-normalized waveform; librosa CQT at hop 512, "
            "C1, 192 bins, 24 bins/octave; amplitude_to_db(ref=max) / 80 + 1"
        ),
        evidence=(
            "github.com/yongyizang/SynthTab@6136f79d:demo_embedding/"
            "exp_finetuning.py and demo_embedding/tabcnn.py; pinned tabcnn.py "
            f"SHA-256 {SYNTHTAB_TABCNN_SOURCE_SHA256}",
        ),
    ),
)

DAFX_GUITARPROFX_ONNX = TabCNNArtifact(
    artifact_id="dafx_guitarprofx_onnx",
    family_id="guitarprofx-tabcnn",
    role="model",
    filename="tabcnn-gpfx.onnx",
    relative_path="tabcnn/dafx/tabcnn-gpfx.onnx",
    size_bytes=3_339_568,
    sha256="8d9ce59157bdab37fb4816d32d7f29f3da0cdbf3c7876707c819af4d1f88e6b7",
    source_repository_url="https://huggingface.co/cstr/tabcnn-onnx",
    source_revision="c15524a6944febe68129d26c2a89eca455b5499d",
    source_path="tabcnn-gpfx.onnx",
    source_url=(
        "https://huggingface.co/cstr/tabcnn-onnx/blob/"
        "c15524a6944febe68129d26c2a89eca455b5499d/tabcnn-gpfx.onnx"
    ),
    download_url=(
        "https://huggingface.co/cstr/tabcnn-onnx/resolve/"
        "c15524a6944febe68129d26c2a89eca455b5499d/tabcnn-gpfx.onnx"
    ),
    family_source_repository_url="https://github.com/robust-guitar-tabs/code",
    family_source_revision="f50309ad06dc734ddae5e3a0eda756fca221e2e7",
    license_id="CC-BY-4.0",
    license_posture=(
        "Evaluation transport of the CC-BY-4.0 Zenodo checkpoint; official "
        "family source is CC0. Attribution and export provenance are required."
    ),
    overlap_labels=(
        "guitarset:development-overlap",
        "gaps:primary-classical-transfer",
        "egset12:published-reproduction",
        "guitar-techs:independent-electric-transfer",
    ),
    runtime_contract=RuntimeContract(
        input_layout="float32[N, 192, 9, 1]",
        output_layout="float32[N, 6, 21] LogSoftmax",
        class_order=(
            "publisher-declared for the executed ONNX transport; independently "
            "unverified and descriptive-only: class 0 is silence; class k is "
            "fret k-1 for k=1..20"
        ),
        front_end=(
            "publisher-declared for the executed ONNX transport; independently "
            "unverified and descriptive-only: 22,050 Hz mono; librosa CQT at "
            "hop 512, C1, 192 bins, 24 bins/octave; "
            "amplitude_to_db(ref=max), then per-clip min-max scaling to [0,1]"
        ),
        evidence=(
            "github.com/robust-guitar-tabs/code@f50309ad:AMT-Tools/examples/"
            "papers/guitarProFx.py, amt_tools/features/common.py, and "
            "amt_tools/models/common.py establish the native CQT geometry, "
            "dB/80+1 scaling, and native fret-0..19-then-silence order",
            "huggingface.co/cstr/tabcnn-onnx@c15524a6:README.md is the publisher "
            "declaration for the executed transport's per-clip min-max front end "
            "and output class roll; those interpretations are independently "
            "unverified and descriptive-only",
        ),
    ),
    derived_from=DAFX_OFFICIAL_CHECKPOINT,
    official_equivalence_verified=False,
    official_equivalence_blocker=(
        "The official 3,345,122-byte MD5-pinned checkpoint was unavailable locally, "
        "and the pinned ONNX repository contains no export program or cryptographic "
        "mapping from the official bytes to this ONNX SHA-256. The ONNX contract is "
        "verified as executed, but equivalence to the published checkpoint is not."
    ),
)

SHARED_CQT = TabCNNArtifact(
    artifact_id="tabcnn_shared_cqt",
    family_id="tabcnn-shared-front-end",
    role="front_end",
    filename="tabcnn-cqt.bin",
    relative_path="tabcnn/shared/tabcnn-cqt.bin",
    size_bytes=696_312,
    sha256="4e5dfa1f10f76545a30cbfd3224431503dbad943b1def78624632284e6df597a",
    source_repository_url="https://huggingface.co/cstr/tabcnn-onnx",
    source_revision="c15524a6944febe68129d26c2a89eca455b5499d",
    source_path="tabcnn-cqt.bin",
    source_url=(
        "https://huggingface.co/cstr/tabcnn-onnx/blob/"
        "c15524a6944febe68129d26c2a89eca455b5499d/tabcnn-cqt.bin"
    ),
    download_url=(
        "https://huggingface.co/cstr/tabcnn-onnx/resolve/"
        "c15524a6944febe68129d26c2a89eca455b5499d/tabcnn-cqt.bin"
    ),
    family_source_repository_url="https://huggingface.co/cstr/tabcnn-onnx",
    family_source_revision="c15524a6944febe68129d26c2a89eca455b5499d",
    license_id="CC-BY-4.0",
    license_posture=(
        "Hash-pinned evaluation reference for the executed librosa CQT geometry; "
        "the binary is not loaded for inference. Attribution is required."
    ),
    overlap_labels=("front-end:no-training-corpus-overlap",),
)

FROZEN_ARTIFACTS: tuple[TabCNNArtifact, ...] = (
    SYNTHTAB_X4,
    DAFX_GUITARPROFX_ONNX,
    SHARED_CQT,
)
_ARTIFACTS_BY_ID = {artifact.artifact_id: artifact for artifact in FROZEN_ARTIFACTS}


def data_root() -> Path:
    """Return the external data root without creating it."""

    return Path(os.environ.get(TABVISION_DATA_ROOT_ENV, DEFAULT_DATA_ROOT)).expanduser()


def default_models_root() -> Path:
    """Return the external model root without creating it."""

    return data_root() / "models"


def artifact_by_id(artifact_id: str) -> TabCNNArtifact:
    """Return one frozen artifact or raise a useful error."""

    try:
        return _ARTIFACTS_BY_ID[artifact_id]
    except KeyError as exc:
        choices = ", ".join(sorted(_ARTIFACTS_BY_ID))
        raise KeyError(f"unknown TabCNN artifact {artifact_id!r}; choose from {choices}") from exc


def artifact_manifest(
    *,
    artifacts: tuple[TabCNNArtifact, ...] = FROZEN_ARTIFACTS,
    models_root: str | Path | None = None,
) -> dict[str, object]:
    """Return the deterministic frozen provenance manifest."""

    root_label = str(models_root) if models_root is not None else "$TABVISION_DATA_ROOT/models"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_frozen_on": PROTOCOL_FROZEN_ON,
        "models_root": root_label,
        "artifacts": [
            artifact.to_dict(models_root=models_root)
            for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
        ],
    }


def artifact_manifest_json_bytes(
    *,
    artifacts: tuple[TabCNNArtifact, ...] = FROZEN_ARTIFACTS,
    models_root: str | Path | None = None,
) -> bytes:
    """Serialize the frozen manifest with stable ordering and a final newline."""

    return (
        json.dumps(
            artifact_manifest(artifacts=artifacts, models_root=models_root),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
