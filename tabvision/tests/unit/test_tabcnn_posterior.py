from __future__ import annotations

import hashlib
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tabvision.eval.tabcnn_artifacts import (
    SYNTHTAB_TABCNN_SOURCE_SHA256,
    SYNTHTAB_X4,
    default_models_root,
)
from tabvision.eval.tabcnn_posterior import (
    BINS_PER_OCTAVE,
    FMIN_HZ,
    GUITARPROFX_NORMALIZATION,
    HOP_LENGTH,
    N_BINS,
    N_CLASSES,
    N_STRINGS,
    SAMPLE_RATE,
    SYNTHTAB_CHECKPOINT_SHA256,
    SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256,
    SYNTHTAB_NORMALIZATION,
    TORCH_INTEROP_THREADS,
    TORCH_INTRA_OP_THREADS,
    WINDOW_FRAMES,
    DAFxTabCNNPosterior,
    SynthTabX4Posterior,
    _configure_torch_cpu_runtime,
    _validate_synthtab_pickle_globals,
    cqt_windows,
    event_fret_prior,
    posterior_sha256,
    probe_synthtab_checkpoint_equivalence,
    remap_synthtab_probabilities,
    sha256_file,
    validate_checkpoint,
)


class _Port:
    def __init__(
        self,
        name: str,
        shape: list[str | int],
        tensor_type: str = "tensor(float)",
    ) -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _FakeOnnxSession:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def get_inputs(self) -> list[_Port]:
        return [_Port("cqt", ["N", N_BINS, WINDOW_FRAMES, 1])]

    def get_outputs(self) -> list[_Port]:
        return [_Port("log_probs", ["N", N_STRINGS, N_CLASSES])]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        assert output_names == ["log_probs"]
        windows = inputs["cqt"]
        self.batch_sizes.append(len(windows))
        class_axis = np.arange(N_CLASSES, dtype=np.float32)
        logits = np.broadcast_to(
            class_axis,
            (len(windows), N_STRINGS, N_CLASSES),
        ).copy()
        logits *= windows[:, :1, WINDOW_FRAMES // 2, :1]
        maximum = np.max(logits, axis=-1, keepdims=True)
        normalizer = maximum + np.log(np.sum(np.exp(logits - maximum), axis=-1, keepdims=True))
        return [logits - normalizer]


def _normalized_frames(count: int) -> np.ndarray:
    probabilities = np.full(
        (count, N_STRINGS, N_CLASSES),
        1.0 / N_CLASSES,
        dtype=np.float64,
    )
    return probabilities


def test_cqt_windows_use_fixed_shape_scaling_and_timestamps() -> None:
    pytest.importorskip("librosa")
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    waveform = 0.2 * np.sin(2.0 * np.pi * 110.0 * time)

    first = cqt_windows(waveform)
    scaled = cqt_windows(4.0 * waveform)

    assert first.windows.shape[1:] == (N_BINS, WINDOW_FRAMES, 1)
    assert first.windows.shape[0] == len(first.times_s)
    assert first.times_s[0] == 0.0
    assert first.times_s[1] == pytest.approx(HOP_LENGTH / SAMPLE_RATE)
    assert np.min(first.windows) >= -1.0e-6
    assert np.max(first.windows) <= 1.0 + 1.0e-6
    np.testing.assert_allclose(first.windows, scaled.windows, rtol=0.0, atol=2.0e-6)
    assert SAMPLE_RATE == 22_050
    assert FMIN_HZ == pytest.approx(32.70319566257483)
    assert BINS_PER_OCTAVE == 24


def test_cqt_windows_use_exact_family_specific_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_waveforms: list[np.ndarray] = []
    decibels = np.broadcast_to(
        np.asarray([-40.0, -20.0, 0.0], dtype=np.float64),
        (N_BINS, 3),
    ).copy()

    def cqt(signal: np.ndarray, **kwargs: object) -> np.ndarray:
        captured_waveforms.append(signal.copy())
        assert kwargs == {
            "sr": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "fmin": FMIN_HZ,
            "n_bins": N_BINS,
            "bins_per_octave": BINS_PER_OCTAVE,
        }
        return np.ones((N_BINS, 3), dtype=np.complex64)

    def amplitude_to_db(magnitude: np.ndarray, *, ref: object) -> np.ndarray:
        assert magnitude.shape == (N_BINS, 3)
        assert ref is np.max
        return decibels.copy()

    fake_librosa = types.ModuleType("librosa")
    fake_librosa.cqt = cqt
    fake_librosa.amplitude_to_db = amplitude_to_db
    monkeypatch.setitem(__import__("sys").modules, "librosa", fake_librosa)
    waveform = np.asarray([0.25, -0.5], dtype=np.float32)

    synthtab = cqt_windows(waveform, normalization=SYNTHTAB_NORMALIZATION)
    guitarprofx = cqt_windows(waveform, normalization=GUITARPROFX_NORMALIZATION)

    np.testing.assert_array_equal(
        captured_waveforms[0],
        np.asarray([0.5, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(captured_waveforms[1], waveform)
    np.testing.assert_allclose(
        synthtab.windows[:, 0, WINDOW_FRAMES // 2, 0],
        np.asarray([0.5, 0.75, 1.0]),
    )
    np.testing.assert_allclose(
        guitarprofx.windows[:, 0, WINDOW_FRAMES // 2, 0],
        np.asarray([0.0, 0.5, 1.0]),
    )


def test_dafx_backend_preserves_shape_and_chunk_order() -> None:
    session = _FakeOnnxSession()
    backend = DAFxTabCNNPosterior(session)
    windows = np.zeros((5, N_BINS, WINDOW_FRAMES, 1), dtype=np.float32)
    windows[:, 0, WINDOW_FRAMES // 2, 0] = np.arange(5, dtype=np.float32)

    first = backend.predict_windows(windows, chunk_size=2)
    second = backend.predict_windows(windows, chunk_size=3)

    assert first.shape == (5, N_STRINGS, N_CLASSES)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=-1), 1.0, rtol=0.0, atol=1.0e-6)
    assert session.batch_sizes == [2, 2, 1, 3, 2]
    assert np.all(np.argmax(first[1], axis=-1) == N_CLASSES - 1)
    assert backend.feature_normalization == GUITARPROFX_NORMALIZATION


@pytest.mark.parametrize("fixed_port", ["input", "output"])
def test_dafx_backend_rejects_fixed_batch_dimensions(fixed_port: str) -> None:
    class FixedBatchSession(_FakeOnnxSession):
        def get_inputs(self) -> list[_Port]:
            batch: str | int = 1 if fixed_port == "input" else "N"
            return [_Port("cqt", [batch, N_BINS, WINDOW_FRAMES, 1])]

        def get_outputs(self) -> list[_Port]:
            batch: str | int = 1 if fixed_port == "output" else "N"
            return [_Port("log_probs", [batch, N_STRINGS, N_CLASSES])]

    with pytest.raises(ValueError, match="batch dimension must be dynamic"):
        DAFxTabCNNPosterior(FixedBatchSession())


def test_dafx_backend_accepts_none_batch_dimensions() -> None:
    class DynamicBatchSession(_FakeOnnxSession):
        def get_inputs(self) -> list[_Port]:
            return [_Port("cqt", [None, N_BINS, WINDOW_FRAMES, 1])]  # type: ignore[list-item]

        def get_outputs(self) -> list[_Port]:
            return [_Port("log_probs", [None, N_STRINGS, N_CLASSES])]  # type: ignore[list-item]

    assert isinstance(DAFxTabCNNPosterior(DynamicBatchSession()), DAFxTabCNNPosterior)


def test_synthtab_class_order_remaps_silence_and_frets() -> None:
    native = np.zeros((1, N_STRINGS, N_CLASSES), dtype=np.float32)
    native[:, :, 20] = 0.75  # native silence
    native[:, :, 0] = 0.20  # native open string
    native[:, :, 7] = 0.05  # native fret 7

    shared = remap_synthtab_probabilities(native)

    np.testing.assert_array_equal(shared[:, :, 0], native[:, :, 20])
    np.testing.assert_array_equal(shared[:, :, 1], native[:, :, 0])
    np.testing.assert_array_equal(shared[:, :, 8], native[:, :, 7])


def test_synthtab_backend_is_chunk_deterministic_and_shared_order() -> None:
    torch = pytest.importorskip("torch")

    class FakeCleanModel:
        def __call__(self, windows: object) -> object:
            logits = torch.zeros((len(windows), N_STRINGS, N_CLASSES))
            logits[..., 20] = 3.0  # native silence
            logits[..., 0] = 2.0  # native open string
            return logits

    backend = SynthTabX4Posterior(FakeCleanModel(), torch)
    windows = np.zeros((4, N_BINS, WINDOW_FRAMES, 1), dtype=np.float32)

    one_chunk = backend.predict_windows(windows, chunk_size=4)
    many_chunks = backend.predict_windows(windows, chunk_size=1)

    np.testing.assert_array_equal(one_chunk, many_chunks)
    assert np.all(one_chunk[..., 0] > one_chunk[..., 1])
    np.testing.assert_allclose(one_chunk.sum(axis=-1), 1.0, rtol=0.0, atol=1.0e-6)
    assert backend.feature_normalization == SYNTHTAB_NORMALIZATION


def test_synthtab_runtime_is_pinned_and_fails_closed() -> None:
    class FakeTorchRuntime:
        def __init__(self, *, reject_interop: bool = False) -> None:
            self.intra_threads = 8
            self.interop_threads = 8
            self.deterministic = False
            self.reject_interop = reject_interop

        def set_num_threads(self, value: int) -> None:
            self.intra_threads = value

        def get_num_threads(self) -> int:
            return self.intra_threads

        def set_num_interop_threads(self, value: int) -> None:
            if self.reject_interop:
                raise RuntimeError("interop already initialized")
            self.interop_threads = value

        def get_num_interop_threads(self) -> int:
            return self.interop_threads

        def use_deterministic_algorithms(self, enabled: bool) -> None:
            self.deterministic = enabled

        def are_deterministic_algorithms_enabled(self) -> bool:
            return self.deterministic

    runtime = FakeTorchRuntime()
    _configure_torch_cpu_runtime(runtime)

    assert runtime.intra_threads == TORCH_INTRA_OP_THREADS
    assert runtime.interop_threads == TORCH_INTEROP_THREADS
    assert runtime.deterministic is True

    with pytest.raises(RuntimeError, match="fresh PyTorch CPU runtime"):
        _configure_torch_cpu_runtime(FakeTorchRuntime(reject_interop=True))


def test_event_prior_uses_nearest_frame_and_earlier_tie() -> None:
    probabilities = _normalized_frames(2)
    probabilities[0, 5] = 0.01 / (N_CLASSES - 1)
    probabilities[0, 5, 0] = 0.01
    probabilities[0, 5, 1] = 0.98  # high-E open at the earlier frame
    probabilities[1, 5] = 0.01 / (N_CLASSES - 1)
    probabilities[1, 5, 0] = 0.01
    probabilities[1, 5, 1] = 0.001

    prior = event_fret_prior(
        64,
        probabilities,
        np.asarray([0.0, 2.0]),
        1.0,
    )

    assert prior is not None
    assert prior.shape == (N_STRINGS, 25)
    assert prior[5, 0] > prior[4, 5]


def test_event_prior_keeps_frets_20_to_24_neutral() -> None:
    probabilities = np.full(
        (1, N_STRINGS, N_CLASSES),
        0.5 / (N_CLASSES - 1),
        dtype=np.float64,
    )
    probabilities[..., 0] = 0.5

    prior = event_fret_prior(64, probabilities, np.asarray([0.0]), 0.0)

    assert prior is not None
    candidates = [(0, 24), (1, 19), (2, 14), (3, 9), (4, 5), (5, 0)]
    for string_idx, fret in candidates:
        assert prior[string_idx, fret] == pytest.approx(1.0)
    assert np.count_nonzero(prior) == len(candidates)


def test_event_prior_uses_conditional_active_likelihood_ratios_without_mutation() -> None:
    probabilities = np.zeros((1, N_STRINGS, N_CLASSES), dtype=np.float64)
    candidates = [(0, 24), (1, 19), (2, 14), (3, 9), (4, 5), (5, 0)]
    supported = candidates[1:]
    silences = np.asarray([0.90, 0.10, 0.70, 0.20, 0.50])
    conditional_values = np.asarray([0.80, 0.40, 0.20, 0.10, 0.50])

    # The unsupported low-E candidate looks almost silent.  An implementation
    # that substitutes its same-string mean would suppress fret 24.
    probabilities[0, 0, 0] = 0.999
    probabilities[0, 0, 1] = 0.001
    for (string_idx, fret), silence, conditional in zip(
        supported,
        silences,
        conditional_values,
        strict=True,
    ):
        active_mass = 1.0 - silence
        target_class = fret + 1
        filler_class = 1 if target_class != 1 else 2
        probabilities[0, string_idx, 0] = silence
        probabilities[0, string_idx, target_class] = active_mass * conditional
        probabilities[0, string_idx, filler_class] += active_mass * (1.0 - conditional)

    frame_times = np.asarray([0.25], dtype=np.float64)
    probabilities_before = probabilities.copy()
    frame_times_before = frame_times.copy()
    prior = event_fret_prior(64, probabilities, frame_times, 0.25)

    assert prior is not None
    expected_ratios = conditional_values / np.mean(conditional_values)
    for (string_idx, fret), expected in zip(supported, expected_ratios, strict=True):
        assert prior[string_idx, fret] == pytest.approx(expected)
    assert prior[0, 24] == 1.0
    assert np.count_nonzero(prior) == len(candidates)
    np.testing.assert_array_equal(probabilities, probabilities_before)
    np.testing.assert_array_equal(frame_times, frame_times_before)


def test_event_prior_structurally_abstains() -> None:
    probabilities = _normalized_frames(1)
    frame_times = np.asarray([0.0])

    # Only one pitch-compatible candidate is within the supported fret range.
    assert event_fret_prior(40, probabilities, frame_times, 0.0) is None

    # All supported candidate values are exactly zero.
    zero_supported = probabilities.copy()
    for string_idx, fret in [(1, 19), (2, 14), (3, 9), (4, 5), (5, 0)]:
        zero_supported[0, string_idx, fret + 1] = 0.0
    assert event_fret_prior(64, zero_supported, frame_times, 0.0) is None

    # No standard-tuning candidate is playable within the requested range.
    assert event_fret_prior(39, probabilities, frame_times, 0.0) is None


def test_event_prior_abstains_when_all_frame_distances_overflow() -> None:
    probabilities = _normalized_frames(2)
    frame_times = np.asarray([-1.0e308, -9.0e307])

    assert event_fret_prior(64, probabilities, frame_times, 1.0e308) is None


def test_posterior_hash_is_deterministic_and_dtype_stable() -> None:
    probabilities = _normalized_frames(2)
    times = np.asarray([0.0, HOP_LENGTH / SAMPLE_RATE])

    float64_digest = posterior_sha256(probabilities, times)
    float32_digest = posterior_sha256(probabilities.astype(np.float32), times)

    assert float64_digest == float32_digest
    assert float64_digest == posterior_sha256(probabilities.copy(), times.copy())
    assert len(float64_digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_probs", np.nan),
        ("frame_probs", np.inf),
        ("frame_times", np.nan),
        ("frame_times", np.inf),
    ],
)
def test_posterior_hash_rejects_nonfinite_inputs(field: str, value: float) -> None:
    probabilities = _normalized_frames(2)
    times = np.asarray([0.0, HOP_LENGTH / SAMPLE_RATE])
    if field == "frame_probs":
        probabilities[0, 0, 0] = value
    else:
        times[0] = value

    with pytest.raises(ValueError, match=f"{field} must contain only finite values"):
        posterior_sha256(probabilities, times)


def test_posterior_hash_rejects_float32_overflow() -> None:
    probabilities = _normalized_frames(1)
    probabilities[0, 0, 0] = np.finfo(np.float64).max

    with pytest.raises(ValueError, match="remain finite when represented as float32"):
        posterior_sha256(probabilities, np.asarray([0.0]))


def test_synthtab_loader_is_bound_to_canonical_checksum(tmp_path: Path) -> None:
    checkpoint = tmp_path / "not-the-reviewed-synthtab.pt"
    checkpoint.write_bytes(b"not the reviewed SynthTab checkpoint")

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        SynthTabX4Posterior.from_checkpoint(
            checkpoint,
            expected_sha256=SYNTHTAB_CHECKPOINT_SHA256,
        )
    with pytest.raises(ValueError, match="must equal the canonical checkpoint"):
        SynthTabX4Posterior.from_checkpoint(
            checkpoint,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("opcode_name", "payload"),
    [
        ("INST", b"(icollections\nOrderedDict\n."),
        ("OBJ", b"(ccollections\nOrderedDict\no."),
        (
            "NEWOBJ",
            b"\x80\x02ccollections\nOrderedDict\nK\x01\x85\x81.",
        ),
        (
            "NEWOBJ_EX",
            b"\x80\x04ccollections\nOrderedDict\n)}\x92.",
        ),
        ("EXT1", b"\x80\x02\x82\x01."),
        ("EXT2", b"\x80\x02\x83\x01\x00."),
        ("EXT4", b"\x80\x02\x84\x01\x00\x00\x00."),
    ],
)
def test_synthtab_pickle_rejects_unsafe_construction_opcodes(
    tmp_path: Path,
    opcode_name: str,
    payload: bytes,
) -> None:
    checkpoint = tmp_path / f"{opcode_name}.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("fixture/data.pkl", payload)

    with pytest.raises(ValueError, match=opcode_name):
        _validate_synthtab_pickle_globals(checkpoint)


def test_pinned_synthtab_checkpoint_matches_independent_oracle() -> None:
    checkpoint = SYNTHTAB_X4.path_below(default_models_root())
    if not checkpoint.is_file():
        pytest.skip(f"canonical external SynthTab checkpoint is absent: {checkpoint}")
    pytest.importorskip("torch")

    first = probe_synthtab_checkpoint_equivalence(checkpoint)
    second = probe_synthtab_checkpoint_equivalence(checkpoint)

    assert SYNTHTAB_CHECKPOINT_SHA256 == SYNTHTAB_X4.sha256
    assert SYNTHTAB_TABCNN_SOURCE_SHA256 == (
        "f4dfd32f90f96e0fc7ea679751aa22df8f0f79e71a5ad2a4b9663e96b8f7d069"
    )
    assert first == second == SYNTHTAB_EQUIVALENCE_GOLDEN_SHA256


def test_checkpoint_validation_rejects_pointer_and_bad_checksum(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.onnx"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
    )
    with pytest.raises(ValueError, match="Git-LFS pointer"):
        validate_checkpoint(pointer, expected_sha256="0" * 64)

    checkpoint = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"reviewed model bytes")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert sha256_file(checkpoint) == expected
    assert validate_checkpoint(checkpoint, expected_sha256=expected) == checkpoint.resolve()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_dafx_checkpoint_loader_imports_onnxruntime_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"small fake ONNX")
    expected = sha256_file(checkpoint)
    created: dict[str, object] = {}

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0
        execution_mode: object = None

    def inference_session(
        path: str,
        *,
        sess_options: SessionOptions,
        providers: list[str],
    ) -> _FakeOnnxSession:
        created["path"] = path
        created["options"] = sess_options
        created["providers"] = providers
        return _FakeOnnxSession()

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = SessionOptions
    fake_ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
    fake_ort.InferenceSession = inference_session
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", fake_ort)

    backend = DAFxTabCNNPosterior.from_checkpoint(
        checkpoint,
        expected_sha256=expected,
    )

    assert isinstance(backend, DAFxTabCNNPosterior)
    assert created["path"] == str(checkpoint.resolve())
    assert created["providers"] == ["CPUExecutionProvider"]
    options = created["options"]
    assert isinstance(options, SessionOptions)
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == "sequential"
