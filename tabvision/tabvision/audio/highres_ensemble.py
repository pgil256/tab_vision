"""Gate-passed GAPS + FL high-resolution ensemble for clean acoustic guitar."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tabvision.audio.checkpoint_ensemble import emitted_pitch_probability, select_events
from tabvision.audio.highres import HighResBackend
from tabvision.errors import BackendError
from tabvision.types import AudioEvent, SessionConfig

ENSEMBLE_SCHEMA_VERSION = 1
DEFAULT_ENSEMBLE_ARTIFACT = Path(__file__).with_name("ensemble_v1.json")

Source = Literal["gaps", "fl"]


@dataclass(frozen=True)
class CheckpointCalibration:
    """One fixed scalar posterior calibrator."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != (1,) or self.scale.shape != (1,) or self.weights.shape != (2,):
            raise ValueError(
                "checkpoint calibration shapes must be mean=(1), scale=(1), weights=(2)"
            )
        if any(np.any(~np.isfinite(value)) for value in (self.mean, self.scale, self.weights)):
            raise ValueError("checkpoint calibration values must be finite")
        if np.any(self.scale <= 0.0):
            raise ValueError("checkpoint calibration scale must be positive")

    def probability(self, emitted_probability: float) -> float:
        # Development features came from float16 posterior caches. Quantizing
        # this scalar preserves the evaluated decision boundary in live use.
        value = float(np.float16(emitted_probability))
        standardized = (value - float(self.mean[0])) / float(self.scale[0])
        logit = float(self.weights[0] + standardized * self.weights[1])
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)


@dataclass(frozen=True)
class EnsembleArtifact:
    threshold: float
    gaps: CheckpointCalibration
    fl: CheckpointCalibration


def load_ensemble_artifact(path: Path = DEFAULT_ENSEMBLE_ARTIFACT) -> EnsembleArtifact:
    """Load and validate the registered Phase 3 calibration artifact."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(f"highres ensemble artifact is unavailable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BackendError("highres ensemble artifact must be a JSON object")
    if payload.get("schema_version") != ENSEMBLE_SCHEMA_VERSION:
        raise BackendError("highres ensemble artifact schema is incompatible")
    if payload.get("registered") is not True or payload.get("winner") != "confidence_winner":
        raise BackendError("highres ensemble artifact is not registered")
    try:
        threshold = float(payload["threshold"])
        calibrators = payload["calibrators"]
        if not isinstance(calibrators, Mapping):
            raise TypeError("calibrators must be an object")
        gaps = _calibration(calibrators["gaps"])
        fl = _calibration(calibrators["fl"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"highres ensemble artifact is invalid: {exc}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise BackendError("highres ensemble threshold must be in [0, 1]")
    return EnsembleArtifact(threshold, gaps, fl)


def _calibration(raw: Any) -> CheckpointCalibration:
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint calibration must be an object")
    return CheckpointCalibration(
        mean=np.asarray(raw["mean"], dtype=np.float64),
        scale=np.asarray(raw["scale"], dtype=np.float64),
        weights=np.asarray(raw["weights"], dtype=np.float64),
    )


class HighResEnsembleBackend:
    """Sequential two-checkpoint backend with a frozen calibrated selector."""

    name = "highres-ensemble"

    def __init__(
        self,
        *,
        artifact_path: Path = DEFAULT_ENSEMBLE_ARTIFACT,
        **backend_kwargs: Any,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact = load_ensemble_artifact(self.artifact_path)
        self.backend_kwargs = dict(backend_kwargs)

    def transcribe(
        self,
        wav: np.ndarray,
        sr: int,
        session: SessionConfig,
    ) -> Sequence[AudioEvent]:
        # Phase 3 validation covers only clean acoustic guitar. Explicit use in
        # another session is a deterministic GAPS rollback, never an ensemble.
        if session.instrument != "acoustic" or session.tone != "clean":
            return self._transcribe_one("guitar_gaps", wav, sr, session, posteriors=False)

        gaps = self._transcribe_one("guitar_gaps", wav, sr, session, posteriors=True)
        fl = self._transcribe_one("guitar_fl", wav, sr, session, posteriors=True)
        calibrators = {"gaps": self.artifact.gaps, "fl": self.artifact.fl}

        def score(source: Source, _index: int, event: AudioEvent) -> float:
            return calibrators[source].probability(emitted_pitch_probability(event))

        return select_events(
            gaps,
            fl,
            score=score,
            threshold=self.artifact.threshold,
        )

    def _transcribe_one(
        self,
        checkpoint: str,
        wav: np.ndarray,
        sr: int,
        session: SessionConfig,
        *,
        posteriors: bool,
    ) -> tuple[AudioEvent, ...]:
        backend = HighResBackend(
            checkpoint=checkpoint,
            include_pitch_logits=posteriors,
            **self.backend_kwargs,
        )
        try:
            return tuple(backend.transcribe(wav, sr, session))
        finally:
            backend.close()
            del backend
            gc.collect()


__all__ = [
    "CheckpointCalibration",
    "DEFAULT_ENSEMBLE_ARTIFACT",
    "ENSEMBLE_SCHEMA_VERSION",
    "EnsembleArtifact",
    "HighResEnsembleBackend",
    "load_ensemble_artifact",
]
