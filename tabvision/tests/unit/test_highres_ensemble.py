from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tabvision.audio.highres_ensemble import HighResEnsembleBackend, load_ensemble_artifact
from tabvision.errors import BackendError
from tabvision.types import AudioEvent, SessionConfig


def _artifact(path: Path, *, registered: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registered": registered,
                "winner": "confidence_winner",
                "threshold": 0.5,
                "calibrators": {
                    "gaps": {"mean": [0.0], "scale": [1.0], "weights": [-2.0, 4.0]},
                    "fl": {"mean": [0.0], "scale": [1.0], "weights": [-2.0, 4.0]},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _event(onset: float, pitch: int, probability: float) -> AudioEvent:
    logits = np.full(128, -8.0, dtype=np.float32)
    logits[pitch] = np.log(probability / (1.0 - probability))
    return AudioEvent(onset, onset + 0.3, pitch, 0.7, 0.7, pitch_logits=logits)


def test_artifact_rejects_unregistered_and_invalid_calibration(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="not registered"):
        load_ensemble_artifact(_artifact(tmp_path / "unregistered.json", registered=False))

    broken = _artifact(tmp_path / "broken.json")
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["calibrators"]["gaps"]["scale"] = [0.0]
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BackendError, match="scale must be positive"):
        load_ensemble_artifact(broken)


def test_clean_acoustic_ensemble_runs_checkpoints_sequentially_and_preserves_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"active": 0, "max_active": 0, "closed": [], "calls": []}
    gaps_agreed = _event(0.0, 60, 0.1)
    gaps_wrong = _event(1.0, 64, 0.1)
    fl_agreed = _event(0.01, 60, 0.9)
    fl_winner = _event(1.01, 65, 0.9)

    class FakeHighResBackend:
        def __init__(self, *, checkpoint: str, include_pitch_logits: bool, **_kwargs: object):
            self.checkpoint = checkpoint
            self.include_pitch_logits = include_pitch_logits
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])

        def transcribe(
            self, _wav: np.ndarray, _sr: int, _session: SessionConfig
        ) -> tuple[AudioEvent, ...]:
            state["calls"].append(self.checkpoint)
            if self.checkpoint == "guitar_gaps":
                return gaps_agreed, gaps_wrong
            return fl_agreed, fl_winner

        def close(self) -> None:
            state["closed"].append(self.checkpoint)
            state["active"] -= 1

    monkeypatch.setattr(
        "tabvision.audio.highres_ensemble.HighResBackend",
        FakeHighResBackend,
    )
    backend = HighResEnsembleBackend(artifact_path=_artifact(tmp_path / "ensemble.json"))

    actual = backend.transcribe(np.zeros(10, dtype=np.float32), 16_000, SessionConfig())

    assert actual[0] is gaps_agreed
    assert actual[1] is fl_winner
    assert state == {
        "active": 0,
        "max_active": 1,
        "closed": ["guitar_gaps", "guitar_fl"],
        "calls": ["guitar_gaps", "guitar_fl"],
    }


def test_non_clean_session_falls_back_to_gaps_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    expected = _event(0.0, 60, 0.8)

    class FakeHighResBackend:
        def __init__(self, *, checkpoint: str, include_pitch_logits: bool, **_kwargs: object):
            calls.append((checkpoint, include_pitch_logits))

        def transcribe(
            self, _wav: np.ndarray, _sr: int, _session: SessionConfig
        ) -> tuple[AudioEvent, ...]:
            return (expected,)

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "tabvision.audio.highres_ensemble.HighResBackend",
        FakeHighResBackend,
    )
    backend = HighResEnsembleBackend(artifact_path=_artifact(tmp_path / "ensemble.json"))

    actual = backend.transcribe(
        np.zeros(10, dtype=np.float32),
        16_000,
        SessionConfig(instrument="classical"),
    )

    assert actual == (expected,)
    assert calls == [("guitar_gaps", False)]
