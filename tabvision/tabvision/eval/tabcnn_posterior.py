"""Isolated TabCNN posterior inference for complementarity experiments.

This module deliberately stops at frame-level posterior generation and
event-level fret priors.  It does not create or mutate ``AudioEvent`` objects,
so the high-resolution audio backend remains the sole source of onsets and
pitches.

Both supported model families consume the same fixed feature representation:
22.05 kHz mono audio, a 192-bin CQT beginning at C1 with 24 bins per octave,
and centered nine-frame windows at a 512-sample hop.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import pickletools
import re
import sys
import threading
import types
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from tabvision.eval.tabcnn_artifacts import (
    SYNTHTAB_TABCNN_SOURCE_SHA256,
    SYNTHTAB_X4,
)

SAMPLE_RATE = 22_050
HOP_LENGTH = 512
FMIN_HZ = 32.70319566257483  # C1
N_BINS = 192
BINS_PER_OCTAVE = 24
WINDOW_FRAMES = 9
WINDOW_RADIUS = WINDOW_FRAMES // 2
N_STRINGS = 6
N_CLASSES = 21
SUPPORTED_MAX_FRET = 19
STANDARD_TUNING_MIDI = (40, 45, 50, 55, 59, 64)
DEFAULT_CHUNK_SIZE = 256
TORCH_INTRA_OP_THREADS = 1
TORCH_INTEROP_THREADS = 1
FeatureNormalization = Literal["synthtab", "guitarprofx"]
SYNTHTAB_NORMALIZATION: Literal["synthtab"] = "synthtab"
GUITARPROFX_NORMALIZATION: Literal["guitarprofx"] = "guitarprofx"
SYNTHTAB_CHECKPOINT_SHA256 = SYNTHTAB_X4.sha256
SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256 = (
    "e253a52ddf843c72b050f6870b4fb6ff7b5e7c5ec1b54b0e8f78c4bb0f4fd36b"
)
_ACTIVE_EPSILON = np.finfo(np.float64).eps

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
_SHIM_LOCK = threading.Lock()

_SYNTHTAB_PICKLE_GLOBALS = frozenset(
    {
        "__builtin__ set",
        "_codecs encode",
        "amt_tools.models.common SoftmaxGroups",
        "amt_tools.tools.instrument GuitarProfile",
        "collections OrderedDict",
        "numpy dtype",
        "numpy.core.multiarray scalar",
        "tabcnn TabCNN",
        "torch FloatStorage",
        "torch device",
        "torch._utils _rebuild_parameter",
        "torch._utils _rebuild_tensor_v2",
        "torch.nn.modules.activation ReLU",
        "torch.nn.modules.container Sequential",
        "torch.nn.modules.conv Conv2d",
        "torch.nn.modules.dropout Dropout",
        "torch.nn.modules.linear Linear",
        "torch.nn.modules.pooling MaxPool2d",
    }
)

_SYNTHTAB_STATE_SHAPES: Mapping[str, tuple[int, ...]] = {
    "conv.0.weight": (128, 1, 3, 3),
    "conv.0.bias": (128,),
    "conv.2.weight": (256, 128, 3, 3),
    "conv.2.bias": (256,),
    "conv.4.weight": (256, 256, 3, 3),
    "conv.4.bias": (256,),
    "dense.0.weight": (512, 23_808),
    "dense.0.bias": (512,),
    "dense.3.output_layer.weight": (126, 512),
    "dense.3.output_layer.bias": (126,),
}


@dataclass(frozen=True)
class CQTWindowBatch:
    """Fixed TabCNN inputs and their center-frame timestamps."""

    windows: np.ndarray
    times_s: np.ndarray


@dataclass(frozen=True)
class FramePosteriors:
    """Normalized probabilities in shared ``silence, fret 0..19`` order."""

    probabilities: np.ndarray
    times_s: np.ndarray


class PosteriorBackend(Protocol):
    """Common inference surface implemented by both checkpoint families."""

    feature_normalization: FeatureNormalization

    def predict_windows(
        self,
        windows: np.ndarray,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Return normalized ``(frames, 6, 21)`` shared-order probabilities."""


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a local file."""

    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: str | Path, *, expected_sha256: str) -> Path:
    """Reject Git-LFS pointers and require an exact, caller-pinned checksum."""

    checkpoint = Path(path).expanduser().resolve()
    expected = expected_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with checkpoint.open("rb") as handle:
        if handle.read(len(_GIT_LFS_HEADER)) == _GIT_LFS_HEADER:
            raise ValueError(f"{checkpoint} is a Git-LFS pointer, not model data")
    observed = sha256_file(checkpoint)
    if observed != expected:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {checkpoint}: "
            f"expected {expected}, observed {observed}"
        )
    return checkpoint


def cqt_windows(
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    normalization: FeatureNormalization = SYNTHTAB_NORMALIZATION,
) -> CQTWindowBatch:
    """Build deterministic centered TabCNN CQT windows.

    SynthTab peak-normalizes the waveform, then applies
    ``amplitude_to_db(ref=max) / 80 + 1``.  The executed, SHA-256-pinned
    GuitarProFX ONNX transport instead declares per-clip min-max scaling after
    the same dB conversion.  That transport contract is frozen at
    ``cstr/tabcnn-onnx@c15524a6``.

    The official native GuitarProFX family source at
    ``robust-guitar-tabs/code@f50309ad`` uses dB/80+1 and a last-class silence
    label.  Its published checkpoint could not be compared locally with the
    transport, so this module does not claim those bytes are export-equivalent.
    Four all-zero feature frames are padded on each side, so every CQT center
    frame has one nine-frame window.
    """

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"TabCNN requires {SAMPLE_RATE} Hz audio, got {sample_rate}")
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"expected a non-empty mono waveform, got shape {signal.shape}")
    if np.any(~np.isfinite(signal)):
        raise ValueError("waveform must contain only finite samples")

    if normalization not in (SYNTHTAB_NORMALIZATION, GUITARPROFX_NORMALIZATION):
        raise ValueError(f"unknown TabCNN feature normalization: {normalization!r}")
    if normalization == SYNTHTAB_NORMALIZATION:
        peak = float(np.max(np.abs(signal)))
        signal = signal if peak == 0.0 else signal / peak

    try:
        librosa = importlib.import_module("librosa")
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "TabCNN CQT extraction requires librosa; install the audio-baseline extra"
        ) from exc

    transform = librosa.cqt(
        signal,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN_HZ,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    magnitude = np.abs(transform)
    decibels = librosa.amplitude_to_db(magnitude, ref=np.max)
    if normalization == SYNTHTAB_NORMALIZATION:
        scaled = decibels / 80.0 + 1.0
    else:
        feature_min = float(np.min(decibels))
        feature_range = float(np.max(decibels)) - feature_min
        scaled = (decibels - feature_min) / (feature_range + 1.0e-9)
    features = np.asarray(scaled, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != N_BINS:
        raise RuntimeError(f"unexpected CQT shape {features.shape}")
    if np.any(~np.isfinite(features)):
        raise RuntimeError("CQT extraction produced non-finite features")

    padded = np.pad(features, ((0, 0), (WINDOW_RADIUS, WINDOW_RADIUS)))
    framed = np.lib.stride_tricks.sliding_window_view(
        padded,
        WINDOW_FRAMES,
        axis=1,
    )
    windows = np.ascontiguousarray(framed.transpose(1, 0, 2)[..., None])
    times_s = np.arange(features.shape[1], dtype=np.float64) * HOP_LENGTH / SAMPLE_RATE
    return CQTWindowBatch(windows=windows, times_s=times_s)


class DAFxTabCNNPosterior:
    """GuitarProFX TabCNN ONNX posterior backend.

    The ONNX graph must accept ``(N, 192, 9, 1)`` and return log-probabilities
    shaped ``(N, 6, 21)``.  Its native class order already matches the shared
    order: silence at 0 and fret ``k`` at class ``k + 1``.
    """

    feature_normalization: FeatureNormalization = GUITARPROFX_NORMALIZATION

    def __init__(self, session: Any) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("DAFx TabCNN ONNX must have exactly one input and one output")
        _validate_onnx_port(
            inputs[0],
            label="input",
            expected_tail=(N_BINS, WINDOW_FRAMES, 1),
        )
        _validate_onnx_port(
            outputs[0],
            label="output",
            expected_tail=(N_STRINGS, N_CLASSES),
        )
        self._session = session
        self._input_name = str(inputs[0].name)
        self._output_name = str(outputs[0].name)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> DAFxTabCNNPosterior:
        """Load a checksum-pinned ONNX file with deterministic CPU settings."""

        checkpoint = validate_checkpoint(path, expected_sha256=expected_sha256)
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "DAFx TabCNN inference requires the optional onnxruntime package"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        if hasattr(ort, "ExecutionMode"):
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(checkpoint),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return cls(session)

    def predict_log_probabilities(
        self,
        windows: np.ndarray,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Return the ONNX graph's shared-order log-probabilities."""

        values = _validate_windows(windows)
        size = _validate_chunk_size(chunk_size)
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), size):
            batch = values[start : start + size]
            raw = self._session.run(
                [self._output_name],
                {self._input_name: batch},
            )[0]
            output = np.asarray(raw, dtype=np.float32)
            expected_shape = (len(batch), N_STRINGS, N_CLASSES)
            if output.shape != expected_shape:
                raise RuntimeError(
                    f"DAFx TabCNN returned {output.shape}, expected {expected_shape}"
                )
            if np.any(~np.isfinite(output)):
                raise RuntimeError("DAFx TabCNN returned non-finite log-probabilities")
            _validate_log_probabilities(output)
            chunks.append(output)
        if not chunks:
            return np.empty((0, N_STRINGS, N_CLASSES), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(chunks, axis=0))

    def predict_windows(
        self,
        windows: np.ndarray,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Return normalized shared-order probabilities."""

        return _softmax(self.predict_log_probabilities(windows, chunk_size=chunk_size))


class SynthTabX4Posterior:
    """Sanitized local reimplementation of the official SynthTabx4 model."""

    feature_normalization: FeatureNormalization = SYNTHTAB_NORMALIZATION

    def __init__(self, clean_model: Any, torch_module: Any) -> None:
        self._model = clean_model
        self._torch = torch_module

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> SynthTabX4Posterior:
        """Load a reviewed legacy checkpoint and retain only copied tensors.

        ``torch.load(..., weights_only=False)`` is required by the legacy
        whole-model file.  To keep that narrow exception explicit, callers
        must supply a reviewed SHA-256, the pickle GLOBAL list is allowlisted,
        and only finite tensors with the exact TabCNNx4 shapes are copied into
        a new local ``nn.Module``.  Only the canonical SynthTab checkpoint
        digest is accepted.  The unpickled object is then discarded.
        """

        expected = expected_sha256.lower()
        if expected != SYNTHTAB_CHECKPOINT_SHA256:
            raise ValueError(
                "SynthTab expected_sha256 must equal the canonical checkpoint "
                f"SHA-256 {SYNTHTAB_CHECKPOINT_SHA256}"
            )
        checkpoint = validate_checkpoint(path, expected_sha256=SYNTHTAB_CHECKPOINT_SHA256)
        _validate_synthtab_pickle_globals(checkpoint)
        torch = _import_torch()
        _configure_torch_cpu_runtime(torch)
        state = _load_sanitized_synthtab_state(checkpoint, torch)
        model = _build_synthtab_x4_module(torch)
        clean_state = {
            "conv.0.weight": state["conv.0.weight"],
            "conv.0.bias": state["conv.0.bias"],
            "conv.2.weight": state["conv.2.weight"],
            "conv.2.bias": state["conv.2.bias"],
            "conv.4.weight": state["conv.4.weight"],
            "conv.4.bias": state["conv.4.bias"],
            "dense.0.weight": state["dense.0.weight"],
            "dense.0.bias": state["dense.0.bias"],
            "output.weight": state["dense.3.output_layer.weight"],
            "output.bias": state["dense.3.output_layer.bias"],
        }
        model.load_state_dict(clean_state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, torch)

    def predict_windows(
        self,
        windows: np.ndarray,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Return normalized probabilities remapped to shared class order."""

        values = _validate_windows(windows)
        size = _validate_chunk_size(chunk_size)
        chunks: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(values), size):
                batch = values[start : start + size]
                tensor = self._torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
                native_logits = self._model(tensor)
                native = native_logits.detach().cpu().numpy()
                chunks.append(remap_synthtab_probabilities(_softmax(native)))
        if not chunks:
            return np.empty((0, N_STRINGS, N_CLASSES), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(chunks, axis=0))


def remap_synthtab_probabilities(native_probabilities: np.ndarray) -> np.ndarray:
    """Remap ``fret 0..19, silence`` to ``silence, fret 0..19``."""

    native = np.asarray(native_probabilities, dtype=np.float32)
    if native.ndim != 3 or native.shape[1:] != (N_STRINGS, N_CLASSES):
        raise ValueError(
            "native SynthTab probabilities must have shape "
            f"(frames, {N_STRINGS}, {N_CLASSES}), got {native.shape}"
        )
    if np.any(~np.isfinite(native)) or np.any(native < 0.0):
        raise ValueError("native SynthTab probabilities must be finite and nonnegative")
    shared = np.empty_like(native)
    shared[..., 0] = native[..., SUPPORTED_MAX_FRET + 1]
    shared[..., 1:] = native[..., : SUPPORTED_MAX_FRET + 1]
    return np.ascontiguousarray(shared)


def infer_frame_posteriors(
    waveform: np.ndarray,
    backend: PosteriorBackend,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> FramePosteriors:
    """Extract features and run deterministic, sequential chunked inference."""

    features = cqt_windows(
        waveform,
        sample_rate=sample_rate,
        normalization=backend.feature_normalization,
    )
    probabilities = np.asarray(
        backend.predict_windows(features.windows, chunk_size=chunk_size),
        dtype=np.float32,
    )
    expected_shape = (len(features.times_s), N_STRINGS, N_CLASSES)
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"posterior backend returned {probabilities.shape}, expected {expected_shape}"
        )
    probabilities = _normalize_probabilities(probabilities)
    return FramePosteriors(
        probabilities=np.ascontiguousarray(probabilities),
        times_s=features.times_s.copy(),
    )


def event_fret_prior(
    pitch: int,
    frame_probs: np.ndarray,
    frame_times: np.ndarray,
    onset: float,
    *,
    max_fret: int = 24,
) -> np.ndarray | None:
    """Map one event to a pitch-compatible ``(6, max_fret + 1)`` prior.

    Temporal aggregation is fixed to the single nearest center frame.  An
    exact tie selects the earlier frame because ``numpy.argmin`` returns the
    first minimum.  Each supported pitch candidate uses its probability
    conditional on that string being active, then becomes a likelihood ratio
    relative to the supported-candidate mean.  Frets above 19 receive exactly
    neutral likelihood 1.0.  The model structurally abstains when fewer than
    two candidates are supported or their mean conditional probability is
    zero.  Inputs are never mutated.
    """

    if isinstance(pitch, bool) or not isinstance(pitch, (int, np.integer)):
        raise TypeError("pitch must be an integer MIDI note")
    if not math.isfinite(onset):
        raise ValueError("onset must be finite")
    if isinstance(max_fret, bool) or not isinstance(max_fret, int):
        raise TypeError("max_fret must be an integer")
    if max_fret < 0:
        raise ValueError("max_fret must be nonnegative")

    times = np.asarray(frame_times, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("frame_times must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("frame_times must be finite and nondecreasing")
    probabilities = np.asarray(frame_probs, dtype=np.float64)
    expected_shape = (len(times), N_STRINGS, N_CLASSES)
    if probabilities.shape != expected_shape:
        raise ValueError(f"frame_probs must have shape {expected_shape}")
    probabilities = _normalize_probabilities(probabilities)

    candidates = [
        (string_idx, int(pitch) - open_pitch)
        for string_idx, open_pitch in enumerate(STANDARD_TUNING_MIDI)
        if 0 <= int(pitch) - open_pitch <= max_fret
    ]
    if not candidates:
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.abs(times - float(onset))
    if not np.any(np.isfinite(distances)):
        return None
    frame_idx = int(np.argmin(distances))
    frame = probabilities[frame_idx]
    supported = [
        (string_idx, fret) for string_idx, fret in candidates if fret <= SUPPORTED_MAX_FRET
    ]
    if len(supported) < 2:
        return None

    conditional_active = np.asarray(
        [
            frame[string_idx, fret + 1] / max(1.0 - frame[string_idx, 0], _ACTIVE_EPSILON)
            for string_idx, fret in supported
        ],
        dtype=np.float64,
    )
    supported_mean = float(np.mean(conditional_active))
    if supported_mean <= 0.0:
        return None
    likelihood_ratios = {
        candidate: float(value / supported_mean)
        for candidate, value in zip(supported, conditional_active, strict=True)
    }

    prior = np.zeros((N_STRINGS, max_fret + 1), dtype=np.float64)
    for string_idx, fret in candidates:
        prior[string_idx, fret] = likelihood_ratios.get((string_idx, fret), 1.0)
    return prior


def posterior_sha256(frame_probs: np.ndarray, frame_times: np.ndarray) -> str:
    """Hash posteriors in a platform-stable dtype/shape representation."""

    probabilities = np.asarray(frame_probs)
    times = np.asarray(frame_times)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (N_STRINGS, N_CLASSES):
        raise ValueError(f"frame_probs must have shape (frames, {N_STRINGS}, {N_CLASSES})")
    if times.shape != (len(probabilities),):
        raise ValueError("frame_times must have one timestamp per posterior frame")
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("frame_probs must contain only finite values")
    if np.any(~np.isfinite(times)):
        raise ValueError("frame_times must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        canonical_probs = np.ascontiguousarray(probabilities, dtype="<f4")
    canonical_times = np.ascontiguousarray(times, dtype="<f8")
    if np.any(~np.isfinite(canonical_probs)):
        raise ValueError("frame_probs must remain finite when represented as float32")
    digest = hashlib.sha256(b"tabvision-tabcnn-posteriors-v1\0")
    digest.update(np.asarray(canonical_probs.shape, dtype="<i8").tobytes())
    digest.update(canonical_probs.tobytes())
    digest.update(np.asarray(canonical_times.shape, dtype="<i8").tobytes())
    digest.update(canonical_times.tobytes())
    return digest.hexdigest()


def _validate_windows(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    expected_tail = (N_BINS, WINDOW_FRAMES, 1)
    if values.ndim != 4 or values.shape[1:] != expected_tail:
        raise ValueError(f"expected windows shape (frames, {expected_tail}), got {values.shape}")
    if np.any(~np.isfinite(values)):
        raise ValueError("CQT windows must contain only finite values")
    return np.ascontiguousarray(values)


def _validate_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (N_STRINGS, N_CLASSES):
        raise ValueError(f"TabCNN outputs must have shape (frames, {N_STRINGS}, {N_CLASSES})")
    if np.any(~np.isfinite(values)):
        raise ValueError("TabCNN outputs must be finite")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.ascontiguousarray(probabilities, dtype=np.float32)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("posterior probabilities must be finite and nonnegative")
    totals = np.sum(values, axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("each string posterior must have positive mass")
    return values / totals


def _validate_onnx_port(
    port: Any,
    *,
    label: str,
    expected_tail: tuple[int, ...],
) -> None:
    name = getattr(port, "name", None)
    shape = getattr(port, "shape", None)
    tensor_type = getattr(port, "type", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"DAFx TabCNN ONNX {label} must have a non-empty name")
    if not isinstance(shape, (list, tuple)) or len(shape) != len(expected_tail) + 1:
        raise ValueError(
            f"DAFx TabCNN ONNX {label} must have rank {len(expected_tail) + 1}, got {shape!r}"
        )
    if tuple(shape[1:]) != expected_tail:
        raise ValueError(
            f"DAFx TabCNN ONNX {label} shape tail must be {expected_tail}, got {shape!r}"
        )
    batch_dimension = shape[0]
    if batch_dimension is not None and (
        not isinstance(batch_dimension, str) or not batch_dimension.strip()
    ):
        raise ValueError(
            f"DAFx TabCNN ONNX {label} batch dimension must be dynamic "
            f"(None or a non-empty symbolic name), got {batch_dimension!r}"
        )
    if tensor_type != "tensor(float)":
        raise ValueError(f"DAFx TabCNN ONNX {label} must be tensor(float), got {tensor_type!r}")


def _validate_log_probabilities(values: np.ndarray) -> None:
    maximum = np.max(values, axis=-1, keepdims=True)
    log_total = maximum + np.log(np.sum(np.exp(values - maximum), axis=-1, keepdims=True))
    if not np.allclose(log_total, 0.0, rtol=0.0, atol=2.0e-5):
        max_error = float(np.max(np.abs(log_total)))
        raise RuntimeError(
            "DAFx TabCNN output violates its LogSoftmax contract: "
            f"maximum log-sum-exp error {max_error:.8g}"
        )


def _configure_torch_cpu_runtime(torch: Any) -> None:
    """Pin the process-wide PyTorch CPU runtime before loading SynthTab."""

    try:
        torch.set_num_threads(TORCH_INTRA_OP_THREADS)
        if int(torch.get_num_interop_threads()) != TORCH_INTEROP_THREADS:
            torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "SynthTab requires a fresh PyTorch CPU runtime that can be pinned to "
            "one intra-op thread, one inter-op thread, and deterministic algorithms"
        ) from exc

    if (
        int(torch.get_num_threads()) != TORCH_INTRA_OP_THREADS
        or int(torch.get_num_interop_threads()) != TORCH_INTEROP_THREADS
        or not bool(torch.are_deterministic_algorithms_enabled())
    ):
        raise RuntimeError("PyTorch did not retain the frozen SynthTab CPU settings")


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "SynthTabx4 inference requires torch; install the audio-highres extra"
        ) from exc


def _build_synthtab_x4_module(torch: Any) -> Any:
    nn = torch.nn

    class CleanSynthTabX4(torch.nn.Module):
        """Local inference-only architecture; no amt-tools object is retained."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 128, (3, 3)),
                nn.ReLU(inplace=False),
                nn.Conv2d(128, 256, (3, 3)),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 256, (3, 3)),
                nn.ReLU(inplace=False),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.25),
            )
            self.dense = nn.Sequential(
                nn.Linear(23_808, 512),
                nn.ReLU(inplace=False),
                nn.Dropout(0.5),
            )
            self.output = nn.Linear(512, N_STRINGS * N_CLASSES)

        def forward(self, features: Any) -> Any:
            embedding = self.conv(features).flatten(1)
            return self.output(self.dense(embedding)).reshape(-1, N_STRINGS, N_CLASSES)

    return CleanSynthTabX4()


@contextlib.contextmanager
def _legacy_pickle_shims(torch: Any) -> Iterator[type[Any]]:
    """Temporarily expose only the three missing legacy checkpoint classes."""

    legacy_tabcnn = type("TabCNN", (torch.nn.Module,), {"__module__": "tabcnn"})
    legacy_softmax = type(
        "SoftmaxGroups",
        (torch.nn.Module,),
        {"__module__": "amt_tools.models.common"},
    )
    legacy_profile = type(
        "GuitarProfile",
        (),
        {"__module__": "amt_tools.tools.instrument"},
    )

    modules: dict[str, Any] = {
        "tabcnn": types.ModuleType("tabcnn"),
        "amt_tools": types.ModuleType("amt_tools"),
        "amt_tools.models": types.ModuleType("amt_tools.models"),
        "amt_tools.models.common": types.ModuleType("amt_tools.models.common"),
        "amt_tools.tools": types.ModuleType("amt_tools.tools"),
        "amt_tools.tools.instrument": types.ModuleType("amt_tools.tools.instrument"),
    }
    modules["tabcnn"].TabCNN = legacy_tabcnn
    modules["amt_tools.models.common"].SoftmaxGroups = legacy_softmax
    modules["amt_tools.tools.instrument"].GuitarProfile = legacy_profile
    modules["amt_tools"].models = modules["amt_tools.models"]
    modules["amt_tools"].tools = modules["amt_tools.tools"]
    modules["amt_tools.models"].common = modules["amt_tools.models.common"]
    modules["amt_tools.tools"].instrument = modules["amt_tools.tools.instrument"]

    missing = object()
    previous: dict[str, Any] = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield legacy_tabcnn
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _load_sanitized_synthtab_state(checkpoint: Path, torch: Any) -> dict[str, Any]:
    with _SHIM_LOCK, _legacy_pickle_shims(torch) as legacy_tabcnn:
        legacy = torch.load(
            checkpoint,
            weights_only=False,
            map_location="cpu",
        )
        if not isinstance(legacy, legacy_tabcnn):
            raise ValueError("checkpoint root is not the expected legacy tabcnn.TabCNN")
        raw_state = legacy.state_dict()
        if set(raw_state) != set(_SYNTHTAB_STATE_SHAPES):
            missing = sorted(set(_SYNTHTAB_STATE_SHAPES) - set(raw_state))
            extra = sorted(set(raw_state) - set(_SYNTHTAB_STATE_SHAPES))
            raise ValueError(f"unexpected SynthTabx4 state keys: missing={missing}, extra={extra}")
        sanitized: dict[str, Any] = {}
        for name, expected_shape in _SYNTHTAB_STATE_SHAPES.items():
            tensor = raw_state[name]
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
                shape = tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
                raise ValueError(
                    f"unexpected tensor for {name}: {shape}, expected {expected_shape}"
                )
            copied = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if not bool(torch.isfinite(copied).all()):
                raise ValueError(f"non-finite tensor in SynthTabx4 checkpoint: {name}")
            sanitized[name] = copied
        del raw_state
        del legacy
    return sanitized


def probe_synthtab_checkpoint_equivalence(path: str | Path) -> str:
    """Verify the clean graph against an independent pinned-source oracle.

    The nonzero input is exactly representable in float32.  The oracle spells
    out the operations from the SHA-256-pinned upstream ``tabcnn.py`` without
    calling the rebuilt module's layers.  The two graph executions, the
    independent oracle, and a frozen output digest must all agree exactly.
    """

    torch = _import_torch()
    backend = SynthTabX4Posterior.from_checkpoint(
        path,
        expected_sha256=SYNTHTAB_CHECKPOINT_SHA256,
    )
    values = np.arange(N_BINS * WINDOW_FRAMES, dtype=np.int32)
    fixture = (((values * 37) % 257) - 128).astype(np.float32) / 128.0
    windows = fixture.reshape(1, N_BINS, WINDOW_FRAMES, 1)
    features = torch.from_numpy(np.ascontiguousarray(windows.transpose(0, 3, 1, 2)))

    with torch.inference_mode():
        rebuilt_first = backend._model(features)
        rebuilt_second = backend._model(features)
        oracle = _synthtab_functional_oracle(
            features,
            backend._model.state_dict(),
            torch,
        )

    if not bool(torch.equal(rebuilt_first, rebuilt_second)):
        raise RuntimeError("SynthTab clean graph is not bitwise repeatable")
    if not bool(torch.equal(rebuilt_first, oracle)):
        maximum_error = float(torch.max(torch.abs(rebuilt_first - oracle)).item())
        raise RuntimeError(
            "SynthTab clean graph differs from the pinned-source functional oracle: "
            f"maximum absolute error {maximum_error:.8g}"
        )

    canonical = np.ascontiguousarray(
        rebuilt_first.detach().cpu().numpy(),
        dtype="<f4",
    )
    digest = hashlib.sha256(b"tabvision-synthtab-x4-equivalence-v1\0")
    digest.update(bytes.fromhex(SYNTHTAB_TABCNN_SOURCE_SHA256))
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    observed = digest.hexdigest()
    if observed != SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256:
        raise RuntimeError(
            "SynthTab equivalence output digest mismatch: "
            f"expected {SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256}, observed {observed}"
        )
    return observed


def _synthtab_functional_oracle(features: Any, state: Mapping[str, Any], torch: Any) -> Any:
    """Independently spell out SynthTab x4 inference from pinned upstream source."""

    functional = torch.nn.functional
    embedding = functional.conv2d(
        features,
        state["conv.0.weight"],
        state["conv.0.bias"],
    )
    embedding = functional.relu(embedding)
    embedding = functional.conv2d(
        embedding,
        state["conv.2.weight"],
        state["conv.2.bias"],
    )
    embedding = functional.relu(embedding)
    embedding = functional.conv2d(
        embedding,
        state["conv.4.weight"],
        state["conv.4.bias"],
    )
    embedding = functional.relu(embedding)
    embedding = functional.max_pool2d(embedding, (2, 2))
    embedding = embedding.flatten(1)
    embedding = functional.linear(
        embedding,
        state["dense.0.weight"],
        state["dense.0.bias"],
    )
    embedding = functional.relu(embedding)
    logits = functional.linear(
        embedding,
        state["output.weight"],
        state["output.bias"],
    )
    return logits.reshape(-1, N_STRINGS, N_CLASSES)


def _validate_synthtab_pickle_globals(checkpoint: Path) -> None:
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            pickle_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
            if len(pickle_names) != 1:
                raise ValueError("SynthTab checkpoint must contain exactly one data.pkl")
            payload = archive.read(pickle_names[0])
    except zipfile.BadZipFile as exc:
        raise ValueError("SynthTab checkpoint is not a valid torch zip archive") from exc

    operations = list(pickletools.genops(payload))
    forbidden = {
        "INST",
        "OBJ",
        "NEWOBJ_EX",
        "EXT1",
        "EXT2",
        "EXT4",
        "STACK_GLOBAL",
    }
    observed: set[str] = set()
    global_memo: dict[int, str] = {}
    for index, (opcode, argument, _) in enumerate(operations):
        if opcode.name in forbidden:
            raise ValueError(f"{opcode.name} is not permitted in the SynthTab checkpoint")
        if opcode.name == "GLOBAL":
            reference = str(argument)
            observed.add(reference)
            if index + 1 < len(operations):
                following, memo_key, _ = operations[index + 1]
                if following.name in {"BINPUT", "LONG_BINPUT", "PUT"}:
                    if memo_key is None:
                        raise ValueError("pickle GLOBAL memo key cannot be empty")
                    global_memo[int(memo_key)] = reference
        elif opcode.name == "NEWOBJ":
            _validate_synthtab_newobj(operations, index, global_memo)
    unexpected = sorted(observed - _SYNTHTAB_PICKLE_GLOBALS)
    if unexpected:
        raise ValueError(f"unexpected pickle GLOBAL references: {unexpected}")


def _validate_synthtab_newobj(
    operations: Sequence[tuple[Any, Any | None, int | None]],
    index: int,
    global_memo: Mapping[int, str],
) -> None:
    """Allow only canonical no-argument construction of allowlisted classes."""

    if index < 2 or operations[index - 1][0].name != "EMPTY_TUPLE":
        raise ValueError("NEWOBJ requires an empty argument tuple in SynthTab")
    class_index = index - 2
    while class_index >= 0 and operations[class_index][0].name in {
        "BINPUT",
        "LONG_BINPUT",
        "PUT",
    }:
        class_index -= 1
    if class_index < 0:
        raise ValueError("NEWOBJ class reference is missing in SynthTab")
    class_opcode, argument, _ = operations[class_index]
    if class_opcode.name == "GLOBAL":
        reference = str(argument)
    elif class_opcode.name in {"BINGET", "LONG_BINGET", "GET"}:
        reference = global_memo.get(int(argument), "") if argument is not None else ""
    else:
        reference = ""
    if reference not in _SYNTHTAB_PICKLE_GLOBALS:
        raise ValueError(
            "NEWOBJ class must be an allowlisted GLOBAL in SynthTab; "
            f"observed {reference or class_opcode.name!r}"
        )


__all__ = [
    "BINS_PER_OCTAVE",
    "CQTWindowBatch",
    "DAFxTabCNNPosterior",
    "DEFAULT_CHUNK_SIZE",
    "FMIN_HZ",
    "GUITARPROFX_NORMALIZATION",
    "FramePosteriors",
    "HOP_LENGTH",
    "N_BINS",
    "N_CLASSES",
    "N_STRINGS",
    "PosteriorBackend",
    "SAMPLE_RATE",
    "SUPPORTED_MAX_FRET",
    "SYNTHTAB_CHECKPOINT_SHA256",
    "SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256",
    "SYNTHTAB_NORMALIZATION",
    "SYNTHTAB_TABCNN_SOURCE_SHA256",
    "TORCH_INTEROP_THREADS",
    "TORCH_INTRA_OP_THREADS",
    "SynthTabX4Posterior",
    "WINDOW_FRAMES",
    "cqt_windows",
    "event_fret_prior",
    "infer_frame_posteriors",
    "posterior_sha256",
    "probe_synthtab_checkpoint_equivalence",
    "remap_synthtab_probabilities",
    "sha256_file",
    "validate_checkpoint",
]
