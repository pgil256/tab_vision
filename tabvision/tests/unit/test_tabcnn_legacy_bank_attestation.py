from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scripts.eval import tabcnn_complementarity_experiment as experiment
from tabvision.types import SessionConfig


def _runtime_paths(tmp_path: Path) -> experiment.RuntimePaths:
    return experiment.RuntimePaths(
        data_root=tmp_path,
        model_root=tmp_path / "models",
        cache_root=tmp_path / "cache",
        guitarset_root=tmp_path / "guitarset",
        gaps_root=tmp_path / "gaps",
        egset12_root=tmp_path / "egset12",
        guitar_techs_root=tmp_path / "guitar-techs",
        q6_dev_cache=tmp_path / "unused-dev.json",
        q6_player05_cache=tmp_path / "unused-player05.json",
        q6_gaps_cache=tmp_path / "unused-gaps.json",
        legacy_guitarset_cache=tmp_path / "unused-legacy.json",
    )


def _frozen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    row_overrides: dict[str, Any] | None = None,
    session_overrides: dict[str, Any] | None = None,
    reproduction_passed: bool = True,
) -> tuple[
    experiment.RuntimePaths,
    list[experiment.RawBankTarget],
    dict[str, Any],
    Path,
]:
    paths = _runtime_paths(tmp_path)
    paths.cache_root.mkdir(parents=True)
    audio_path = tmp_path / "fixture.wav"
    audio_path.write_bytes(b"frozen-audio")
    event_cache_path = tmp_path / "fixture.ensemble.json"
    event_cache_path.write_text("[]", encoding="utf-8")
    target = experiment.RawBankTarget(
        clip_id="guitarset/00_fixture_solo",
        track_id="00_fixture_solo",
        player="00",
        mode="solo",
        audio_path=audio_path,
        event_cache_path=event_cache_path,
    )

    backend_identity = "legacy-backend"
    checkpoint = {
        "filename": "guitar-gaps.pth",
        "sha256": "checkpoint-sha",
        "size_bytes": 17,
        "huggingface_revision": "frozen-revision",
    }
    ensemble = {"sha256": "ensemble-sha", "size_bytes": 19}
    published = {"baseline": 0.634, "shipped": 0.7346}
    observed = {"baseline": 0.6339, "shipped": 0.7345}
    tolerance = 0.0015
    frozen_identity: dict[str, Any] = {
        "ledger_sha256": "",
        "backend_identity_sha256": backend_identity,
        "evaluation_sha256": "evaluation-sha",
        "git_revision": "git-revision",
        "clips": 1,
        "published": published,
        "observed": observed,
        "tolerance": tolerance,
    }
    session = asdict(SessionConfig())
    session.update(session_overrides or {})
    row: dict[str, Any] = {
        "clip_id": target.clip_id,
        "audio_sha256": experiment.sha256_file(audio_path),
        "event_cache_sha256": experiment.sha256_file(event_cache_path),
        "backend_identity_sha256": backend_identity,
        "session": session,
        "origin": "generated",
    }
    row.update(row_overrides or {})
    reproduction = {
        "passed": reproduction_passed,
        "clips": 60,
        "player": "05",
        "observed": observed,
        "identity": {"published": published, "tolerance": tolerance},
    }
    ledger = {
        "format_version": 1,
        "split": "dev",
        "complete": True,
        "expected_clips": 1,
        "backend": {
            "identity_sha256": backend_identity,
            "code_revision": {
                "evaluation_sha256": frozen_identity["evaluation_sha256"],
                "git_revision": frozen_identity["git_revision"],
            },
            "checkpoints": [checkpoint],
            "ensemble_artifact": ensemble,
        },
        "clips": [row],
        "player05_control_reproduction": reproduction,
        "reproduction": dict(reproduction),
    }
    payload = (json.dumps(ledger, sort_keys=True) + "\n").encode()
    ledger_sha256 = hashlib.sha256(payload).hexdigest()
    frozen_identity["ledger_sha256"] = ledger_sha256
    ledger_path = paths.cache_root / "q6-dev-raw-event-bank.json"
    backup_path = paths.cache_root / (f"q6-dev-raw-event-bank-{ledger_sha256[:16]}.json")
    ledger_path.write_bytes(payload)
    backup_path.write_bytes(payload)

    current_backend = {
        "identity_sha256": "current-bank-kernel",
        "runtime": {"packages": {"torch": {"version": "test-version"}}},
        "checkpoints": [checkpoint],
        "ensemble_artifact": ensemble,
    }
    monkeypatch.setattr(
        experiment,
        "FROZEN_LEGACY_Q6_DEV_ATTESTATION",
        frozen_identity,
    )
    monkeypatch.setattr(experiment, "FROZEN_LEGACY_Q6_SOURCE_SHA256", {})
    monkeypatch.setattr(
        experiment,
        "FROZEN_LEGACY_Q6_RUNTIME_VERSIONS",
        {"torch": "test-version"},
    )
    return paths, [target], current_backend, ledger_path


def test_frozen_legacy_bank_accepts_exact_one_row_and_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, targets, current_backend, _ledger_path = _frozen_fixture(
        tmp_path,
        monkeypatch,
    )

    ledger, attestation = experiment._validate_frozen_legacy_q6_dev(
        paths,
        targets,
        current_backend,
    )

    assert ledger["clips"][0]["clip_id"] == targets[0].clip_id
    assert attestation["verified"] is True
    assert attestation["provenance"] == "migrated_legacy_q6"
    assert attestation["clips"] == 1
    assert attestation["current_bank_kernel_identity_sha256"] == "current-bank-kernel"


def test_frozen_legacy_bank_rejects_tampered_ledger_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, targets, current_backend, ledger_path = _frozen_fixture(
        tmp_path,
        monkeypatch,
    )
    ledger_path.write_bytes(ledger_path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="missing/tampered"):
        experiment._validate_frozen_legacy_q6_dev(
            paths,
            targets,
            current_backend,
        )


def test_frozen_legacy_bank_rejects_tampered_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, targets, current_backend, _ledger_path = _frozen_fixture(
        tmp_path,
        monkeypatch,
        row_overrides={"audio_sha256": "tampered-audio-sha"},
    )

    with pytest.raises(RuntimeError, match=r"row mismatch.*audio_sha256"):
        experiment._validate_frozen_legacy_q6_dev(
            paths,
            targets,
            current_backend,
        )


def test_frozen_legacy_bank_rejects_tampered_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, targets, current_backend, _ledger_path = _frozen_fixture(
        tmp_path,
        monkeypatch,
        session_overrides={"tone": "distorted"},
    )

    with pytest.raises(RuntimeError, match=r"row mismatch.*session"):
        experiment._validate_frozen_legacy_q6_dev(
            paths,
            targets,
            current_backend,
        )


def test_frozen_legacy_bank_rejects_failed_reproduction_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, targets, current_backend, _ledger_path = _frozen_fixture(
        tmp_path,
        monkeypatch,
        reproduction_passed=False,
    )

    with pytest.raises(
        RuntimeError,
        match="player05_control_reproduction receipt failed",
    ):
        experiment._validate_frozen_legacy_q6_dev(
            paths,
            targets,
            current_backend,
        )
