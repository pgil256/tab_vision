from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from tabvision.eval import highres_event_bank as bank
from tabvision.types import AudioEvent


def _source_paths(tmp_path: Path) -> tuple[tuple[str, Path], ...]:
    paths = []
    for logical_path in (
        "tabvision/audio/checkpoint_ensemble.py",
        "tabvision/audio/highres.py",
        "tabvision/audio/highres_ensemble.py",
        "tabvision/eval/highres_event_bank.py",
        "tabvision/types.py",
    ):
        path = tmp_path / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{logical_path}\n", encoding="utf-8")
        paths.append((logical_path, path))
    return tuple(paths)


def _artifact_tree(tmp_path: Path, label: str) -> tuple[Path, Path]:
    root = tmp_path / label / "snapshots"
    revision = root / "frozen-revision"
    revision.mkdir(parents=True)
    (revision / "guitar-gaps.pth").write_bytes(b"gaps-checkpoint")
    (revision / "guitar-fl.pth").write_bytes(b"fl-checkpoint")
    ensemble = tmp_path / label / "ensemble_v1.json"
    ensemble.write_bytes(b'{"registered":true}')
    return root, ensemble


def _fixed_runtime(version: str = "1") -> dict[str, Any]:
    return {
        "python": {"version": "3.12.3", "implementation": "CPython", "cache_tag": "cpython-312"},
        "cpu": {
            "platform": "Linux-test",
            "machine": "x86_64",
            "processor": "test-cpu",
            "model": "test-cpu",
        },
        "packages": {"torch": {"version": version}},
    }


def _configure_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_paths: tuple[tuple[str, Path], ...],
    checkpoint_root: Path,
    ensemble: Path,
    runtime: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(bank, "_bank_source_paths", lambda: source_paths)
    monkeypatch.setattr(bank, "_highres_checkpoint_snapshot_root", lambda: checkpoint_root)
    monkeypatch.setattr(bank, "DEFAULT_ENSEMBLE_ARTIFACT", ensemble)
    monkeypatch.setattr(bank, "_bank_runtime_identity", lambda: runtime or _fixed_runtime())


def test_bank_source_path_contract_excludes_runner_and_git() -> None:
    logical_paths = {logical_path for logical_path, _path in bank._bank_source_paths()}

    assert logical_paths == {
        "tabvision/audio/checkpoint_ensemble.py",
        "tabvision/audio/highres.py",
        "tabvision/audio/highres_ensemble.py",
        "tabvision/eval/highres_event_bank.py",
        "tabvision/types.py",
    }
    assert all("tabcnn_complementarity_experiment.py" not in path for path in logical_paths)
    assert "git_revision" not in bank.bank_source_revision()


def test_bank_source_revision_ignores_broad_runner_but_changes_for_relevant_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = _source_paths(tmp_path)
    runner = tmp_path / "scripts/eval/tabcnn_complementarity_experiment.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("gate = 1\n", encoding="utf-8")
    monkeypatch.setattr(bank, "_bank_source_paths", lambda: source_paths)

    original = bank.bank_source_revision()["source_sha256"]
    runner.write_text("gate = 2\n", encoding="utf-8")
    after_runner_change = bank.bank_source_revision()["source_sha256"]
    source_paths[1][1].write_text("changed highres source\n", encoding="utf-8")
    after_bank_change = bank.bank_source_revision()["source_sha256"]

    assert after_runner_change == original
    assert after_bank_change != original


def test_backend_identity_ignores_artifact_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = _source_paths(tmp_path)
    first_root, first_ensemble = _artifact_tree(tmp_path, "first")
    second_root, second_ensemble = _artifact_tree(tmp_path, "second")
    _configure_identity(
        monkeypatch,
        source_paths=source_paths,
        checkpoint_root=first_root,
        ensemble=first_ensemble,
    )
    first = bank.highres_bank_backend_identity()

    monkeypatch.setattr(bank, "_highres_checkpoint_snapshot_root", lambda: second_root)
    monkeypatch.setattr(bank, "DEFAULT_ENSEMBLE_ARTIFACT", second_ensemble)
    second = bank.highres_bank_backend_identity()

    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["locations"] != second["locations"]
    assert "git_revision" not in first


@pytest.mark.parametrize("artifact", ["checkpoint", "ensemble"])
def test_backend_identity_changes_for_relevant_artifact_bytes(
    artifact: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = _source_paths(tmp_path)
    checkpoint_root, ensemble = _artifact_tree(tmp_path, "artifacts")
    _configure_identity(
        monkeypatch,
        source_paths=source_paths,
        checkpoint_root=checkpoint_root,
        ensemble=ensemble,
    )
    original = bank.highres_bank_backend_identity()["identity_sha256"]

    if artifact == "checkpoint":
        (checkpoint_root / "frozen-revision/guitar-gaps.pth").write_bytes(b"changed checkpoint")
    else:
        ensemble.write_bytes(b'{"registered":false}')

    assert bank.highres_bank_backend_identity()["identity_sha256"] != original


def test_backend_identity_changes_for_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = _source_paths(tmp_path)
    checkpoint_root, ensemble = _artifact_tree(tmp_path, "artifacts")
    _configure_identity(
        monkeypatch,
        source_paths=source_paths,
        checkpoint_root=checkpoint_root,
        ensemble=ensemble,
        runtime=_fixed_runtime("1"),
    )
    original = bank.highres_bank_backend_identity()["identity_sha256"]

    monkeypatch.setattr(bank, "_bank_runtime_identity", lambda: _fixed_runtime("2"))

    assert bank.highres_bank_backend_identity()["identity_sha256"] != original


def test_runtime_contract_records_exact_required_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bank,
        "_distribution_evidence",
        lambda name: {
            "version": f"{name}-version",
            "record_sha256": f"{name}-record",
            "direct_url": {"present": True},
        },
    )

    runtime = bank._bank_runtime_identity()

    assert set(runtime["packages"]) == {
        "torch",
        "numpy",
        "scipy",
        "soundfile",
        "hf-midi-transcription",
        "piano-transcription-inference",
        "pretty-midi",
        "mido",
    }


def test_direct_url_evidence_excludes_absolute_url() -> None:
    evidence = bank._direct_url_evidence(
        json.dumps(
            {
                "url": "file:///private/install/location",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123",
                    "requested_revision": "main",
                },
            }
        )
    )

    assert evidence == {
        "present": True,
        "valid_json": True,
        "vcs_info": {
            "vcs": "git",
            "commit_id": "abc123",
            "requested_revision": "main",
        },
    }


def test_event_codec_matches_existing_bank_shape_and_onset_sort(tmp_path: Path) -> None:
    events = [
        AudioEvent(0.5, 0.7, 64, 0.75, 0.8, tags=("ensemble",)),
        AudioEvent(0.1, 0.2, 40, 0.5, 0.6),
    ]
    payload = bank.events_to_json(events)
    path = tmp_path / "events.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    loaded = bank.read_banked_events(path)

    assert payload == [
        {
            "onset_s": 0.5,
            "offset_s": 0.7,
            "pitch_midi": 64,
            "velocity": 0.75,
            "confidence": 0.8,
            "tags": ["ensemble"],
        },
        {
            "onset_s": 0.1,
            "offset_s": 0.2,
            "pitch_midi": 40,
            "velocity": 0.5,
            "confidence": 0.6,
            "tags": [],
        },
    ]
    assert [event.onset_s for event in loaded] == [0.1, 0.5]
    assert loaded[1].tags == ("ensemble",)


def test_load_mono_audio_preserves_mono_and_averages_stereo(tmp_path: Path) -> None:
    mono_path = tmp_path / "mono.wav"
    stereo_path = tmp_path / "stereo.wav"
    mono = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    stereo = np.asarray([[0.25, 0.75], [-0.5, 0.5], [0.75, 0.25]], dtype=np.float32)
    sf.write(mono_path, mono, 16_000, subtype="FLOAT")
    sf.write(stereo_path, stereo, 22_050, subtype="FLOAT")

    loaded_mono, mono_rate = bank.load_mono_audio(mono_path)
    loaded_stereo, stereo_rate = bank.load_mono_audio(stereo_path)

    assert mono_rate == 16_000
    assert stereo_rate == 22_050
    assert loaded_mono.dtype == np.float32
    assert loaded_stereo.dtype == np.float32
    np.testing.assert_array_equal(loaded_mono, mono)
    np.testing.assert_allclose(loaded_stereo, stereo.mean(axis=1), rtol=0.0, atol=1.0e-7)


def test_new_backend_is_cpu_offline_with_frozen_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()
    monkeypatch.setattr(bank, "_highres_checkpoint_records", lambda: ([], {}))

    def fake_backend(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(bank, "HighResEnsembleBackend", fake_backend)

    result = bank.new_highres_bank_backend()

    assert result is sentinel
    assert captured == {
        "device": "cpu",
        "batch_size": 8,
        "onset_threshold": 0.3,
        "offset_threshold": 0.3,
        "frame_threshold": 0.1,
    }
    assert bank.os.environ["HF_HUB_OFFLINE"] == "1"
    assert bank.os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
