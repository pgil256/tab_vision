from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.eval import tabcnn_complementarity_experiment as experiment
from tabvision.eval.tabcnn_posterior import FramePosteriors
from tabvision.types import AudioEvent, GuitarConfig, TabEvent


class FakeBackend:
    feature_normalization = "synthtab"

    def predict_windows(self, windows: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
        del chunk_size
        probabilities = np.full((len(windows), 6, 21), 1.0 / 21.0, dtype=np.float32)
        return probabilities


def _model(tmp_path: Path, content: bytes = b"fake-model") -> experiment.ModelSpec:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(content)
    return experiment.ModelSpec(
        name="fake",
        checkpoint=checkpoint,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        family="unit-test",
        guitarset_overlap=False,
        frontend_normalization="synthtab",
        artifact_id="fake-artifact",
        source_revision="unit-test-revision",
        license_id="LicenseRef-Unit-Test",
        license_posture="unit-test evaluation only",
        evaluation_allowed=True,
        shipping_redistribution_allowed=False,
    )


def _clip(tmp_path: Path, *, event_cache_exists: bool = True) -> experiment.ExperimentClip:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake-wave")
    annotation = tmp_path / "clip.jams"
    annotation.write_text("{}", encoding="utf-8")
    event_cache = tmp_path / "clip.ensemble.json"
    if event_cache_exists:
        event_cache.write_text(
            json.dumps(
                [
                    {
                        "onset_s": 0.1,
                        "offset_s": 0.4,
                        "pitch_midi": 64,
                        "velocity": 0.7,
                        "confidence": 0.9,
                        "tags": ["banked"],
                    }
                ]
            ),
            encoding="utf-8",
        )
    return experiment.ExperimentClip(
        clip_id="egset12/01",
        corpus="egset12",
        source="EGSet12",
        split="test",
        tier="clean_electric",
        player=None,
        mode=None,
        audio_path=audio,
        annotation_path=annotation,
        annotation_format="egset12_jams",
        event_cache_path=event_cache,
        event_cache_strategy="experiment-bank",
    )


def _paths(tmp_path: Path) -> experiment.RuntimePaths:
    return experiment.RuntimePaths(
        data_root=tmp_path,
        model_root=tmp_path,
        cache_root=tmp_path / "cache",
        guitarset_root=tmp_path / "guitarset",
        gaps_root=tmp_path / "gaps",
        egset12_root=tmp_path / "egset12",
        guitar_techs_root=tmp_path / "guitar-techs",
        q6_dev_cache=tmp_path / "q6-dev",
        q6_player05_cache=tmp_path / "q6-player05",
        q6_gaps_cache=tmp_path / "q6-gaps",
        legacy_guitarset_cache=tmp_path / "legacy",
    )


def _computation() -> experiment.PosteriorComputation:
    frames = FramePosteriors(
        probabilities=np.full((2, 6, 21), 1.0 / 21.0, dtype=np.float32),
        times_s=np.asarray([0.0, 0.1], dtype=np.float64),
    )
    return experiment.PosteriorComputation(
        frames=frames,
        load_seconds=0.01,
        resample_seconds=0.02,
        cqt_seconds=0.03,
        inference_seconds=0.04,
        duration_s=1.0,
        original_sample_rate=22_050,
    )


def test_posterior_cache_key_invalidates_audio_model_frontend_and_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    revision = {"git_revision": "abc", "evaluation_sha256": "code-a"}
    first = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
    )

    clip.audio_path.write_bytes(b"changed-wave")
    changed_audio = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
    )
    assert changed_audio.key != first.key

    changed_code = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision={"git_revision": "abc", "evaluation_sha256": "code-b"},
    )
    assert changed_code.key != changed_audio.key

    monkeypatch.setattr(experiment, "FRONTEND_VERSION", "test-frontend-v2")
    changed_frontend = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision={"git_revision": "abc", "evaluation_sha256": "code-b"},
    )
    assert changed_frontend.key != changed_code.key

    second_spec = _model(tmp_path, b"changed-model")
    changed_model = experiment.posterior_cache_identity(
        clip.audio_path,
        second_spec,
        code_revision={"git_revision": "abc", "evaluation_sha256": "code-b"},
    )
    assert changed_model.key != changed_frontend.key


def test_atomic_posterior_cache_resumes_and_rejects_partial(
    tmp_path: Path,
) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    calls = 0

    def computer(
        audio_path: Path,
        backend: experiment.PosteriorBackend,
        *,
        chunk_size: int,
    ) -> experiment.PosteriorComputation:
        nonlocal calls
        del audio_path, backend, chunk_size
        calls += 1
        return _computation()

    revision = {"git_revision": "abc", "evaluation_sha256": "code"}
    first, resumed = experiment.ensure_posterior_cache(
        clip,
        spec,
        tmp_path / "cache",
        FakeBackend(),
        code_revision=revision,
        computer=computer,
    )
    assert not resumed
    assert first.path.is_file()
    assert calls == 1
    assert not list(first.path.parent.glob(f".{first.path.name}.*"))

    second, resumed = experiment.ensure_posterior_cache(
        clip,
        spec,
        tmp_path / "cache",
        FakeBackend(),
        code_revision=revision,
        computer=computer,
    )
    assert resumed
    assert calls == 1
    assert second.metadata["posterior_sha256"] == first.metadata["posterior_sha256"]

    identity = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
    )
    np.savez(first.path, probabilities=np.ones((1, 6, 21), dtype=np.float32))
    with pytest.raises(ValueError, match="incomplete posterior cache"):
        experiment.load_posterior_cache(first.path, identity)


def test_repeat_run_determinism_writes_verified_marker(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    revision = {"git_revision": "abc", "evaluation_sha256": "code"}

    def computer(
        audio_path: Path,
        backend: experiment.PosteriorBackend,
        *,
        chunk_size: int,
    ) -> experiment.PosteriorComputation:
        del audio_path, backend, chunk_size
        return _computation()

    cached, _resumed = experiment.ensure_posterior_cache(
        clip,
        spec,
        tmp_path / "cache",
        FakeBackend(),
        code_revision=revision,
        computer=computer,
    )
    marker = experiment.verify_posterior_determinism(
        clip,
        cached,
        FakeBackend(),
        chunk_size=16,
        computer=computer,
    )
    assert marker["verified"] is True
    assert experiment.posterior_determinism_status(cached)["verified"] is True


def test_mapping_preserves_event_fields_and_records_structural_coverage(
    tmp_path: Path,
) -> None:
    event = AudioEvent(
        onset_s=0.1,
        offset_s=0.5,
        pitch_midi=64,
        velocity=0.8,
        confidence=0.9,
        pitch_logits=np.asarray([0.1, 0.9]),
        tags=("raw",),
    )
    probabilities = np.full((2, 6, 21), 1.0 / 21.0, dtype=np.float32)
    cached = experiment.CachedPosterior(
        frames=FramePosteriors(
            probabilities=probabilities,
            times_s=np.asarray([0.0, 0.1]),
        ),
        metadata={},
        path=tmp_path / "unused.npz",
    )
    before = experiment._event_signature([event])
    priors, reasons, _seconds = experiment.map_posteriors_to_events(
        [event],
        cached,
        cfg=GuitarConfig(),
    )
    assert experiment._event_signature([event]) == before
    assert priors[0] is not None
    assert priors[0].shape == (6, 25)
    assert reasons == {
        "covered": 1,
        "unplayable_pitch": 0,
        "structural_abstention": 0,
        "unsupported_candidates": 1,
        "unsupported_non_neutral": 0,
    }


def test_none_event_prior_is_structural_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = AudioEvent(0.1, 0.5, 64, 0.8, 0.9)
    cached = experiment.CachedPosterior(
        frames=FramePosteriors(
            probabilities=np.full((2, 6, 21), 1.0 / 21.0, dtype=np.float32),
            times_s=np.asarray([0.0, 0.1]),
        ),
        metadata={},
        path=tmp_path / "unused.npz",
    )
    monkeypatch.setattr(experiment, "event_fret_prior", lambda *args, **kwargs: None)

    priors, reasons, _seconds = experiment.map_posteriors_to_events(
        [event],
        cached,
        cfg=GuitarConfig(),
    )

    assert priors == [None]
    assert reasons == {
        "covered": 0,
        "unplayable_pitch": 0,
        "structural_abstention": 1,
        "unsupported_candidates": 0,
        "unsupported_non_neutral": 0,
    }


def test_legacy_cache_is_validated_before_evaluation_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = replace(
        _clip(tmp_path),
        event_cache_strategy="legacy-a3-pending-reproduction",
    )
    calls: list[Path] = []

    def validate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        paths = args[0]
        del kwargs
        calls.append(paths.cache_root)
        return {"passed": True}

    class StopAfterValidationError(RuntimeError):
        pass

    def stop_before_scoring(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise StopAfterValidationError

    monkeypatch.setattr(experiment, "validate_legacy_guitarset_cache", validate)
    monkeypatch.setattr(
        experiment, "validate_event_bank_ledgers", lambda *_args, **_kwargs: {"verified": True}
    )
    monkeypatch.setattr(
        experiment,
        "load_cache_performance_receipt",
        lambda *_args, **_kwargs: ({"verified": True}, {clip.clip_id: {}}),
    )
    monkeypatch.setattr(experiment, "posterior_cache_identity", stop_before_scoring)
    paths = _paths(tmp_path)
    with pytest.raises(StopAfterValidationError):
        experiment.evaluate(
            [clip],
            [_model(tmp_path)],
            paths,
            allow_transcribe_missing=False,
        )
    assert calls == [paths.cache_root]


def test_runner_defaults_match_canonical_artifact_paths(tmp_path: Path) -> None:
    assert experiment.SYNTHTAB_X4.path_below(tmp_path) == (
        tmp_path / "tabcnn" / "synthtab" / "SynthTab-Pretrained.pt"
    )
    assert experiment.DAFX_GUITARPROFX_ONNX.path_below(tmp_path) == (
        tmp_path / "tabcnn" / "dafx" / "tabcnn-gpfx.onnx"
    )
    assert experiment.SHARED_CQT.path_below(tmp_path) == (
        tmp_path / "tabcnn" / "shared" / "tabcnn-cqt.bin"
    )


def test_manifest_records_eval_only_routing_and_local_artifacts(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    manifest = experiment.build_manifest([clip], [spec], _paths(tmp_path))
    assert manifest["evaluation_only"] is True
    assert manifest["fusion"]["current_plus_tabcnn"]["tabcnn_weight"] == 0.35
    assert manifest["routing"]["gaps"] == {
        "position": "gaps-v1",
        "sequence": "gaps-seq-v1",
        "physics": None,
    }
    assert manifest["routing"]["egset12"]["position"] is None
    assert manifest["clips"][0]["event_cache_present"] is True
    assert manifest["models"][0]["observed_sha256"] == spec.expected_sha256
    assert manifest["frontend"]["cqt_library"] == "librosa"
    assert manifest["frontend"]["cqt_library_version"] != ""
    assert manifest["reference_cqt_filterbank"]["used_as_runtime_input"] is False
    assert manifest["reference_cqt_filterbank"]["path"] == str(
        tmp_path / "tabcnn" / "shared" / "tabcnn-cqt.bin"
    )
    assert "not loaded" in manifest["reference_cqt_filterbank"]["note"]


def test_missing_event_cache_never_transcribes_implicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path, event_cache_exists=False)

    def unexpected_audio_load(_path: str | Path) -> tuple[np.ndarray, int]:
        raise AssertionError("audio load proves implicit transcription was attempted")

    monkeypatch.setattr(experiment, "load_mono_audio", unexpected_audio_load)
    provider = experiment.RawEventProvider(allow_transcribe_missing=False)
    with pytest.raises(FileNotFoundError, match="refuses implicit transcription"):
        provider.load(clip)
    assert provider._backend is None


@pytest.mark.parametrize("stage", ["manifest", "cache-posteriors", "evaluate", "all"])
def test_cli_exposes_all_frozen_stages(stage: str) -> None:
    args = experiment.build_parser().parse_args([stage, "--corpus", "gaps", "--model", "dafx"])
    assert args.stage == stage
    assert experiment._selection(args.corpus, experiment.CORPORA) == ["gaps"]
    assert experiment._selection(args.model, experiment.MODELS) == ["dafx"]


def test_guitarset_session_matches_published_rotation_harness(tmp_path: Path) -> None:
    clip = replace(
        _clip(tmp_path),
        clip_id="guitarset/00_Jazz1-200-B_solo",
        corpus="guitarset-dev",
        mode="solo",
    )

    session = experiment.session_for_clip(clip)

    assert session.instrument == "acoustic"
    assert session.tone == "clean"
    assert session.style == "mixed"


def test_corpus_routes_do_not_apply_acoustic_evidence_to_electric() -> None:
    assert experiment.ROUTING["guitarset-dev"]["physics"] == ("acoustic-physics-v1/partial_aware")
    assert experiment.ROUTING["gaps"]["physics"] is None
    for corpus in ("egset12", "guitar-techs"):
        assert experiment.ROUTING[corpus] == {
            "position": None,
            "sequence": None,
            "physics": None,
        }


def _write_guitarset_wav_bank(paths: experiment.RuntimePaths) -> None:
    """Create the WAV-only q6 inventory without relying on label discovery."""

    audio_root = paths.guitarset_root / "audio_mono-mic"
    annotation_root = paths.guitarset_root / "annotation"
    for player in ("00", "01", "02", "03", "04", "05"):
        for mode in ("solo", "comp"):
            for index in range(30):
                track_id = f"{player}_fixture_{index:02d}_{mode}"
                (audio_root / f"{track_id}_mic.wav").parent.mkdir(parents=True, exist_ok=True)
                (audio_root / f"{track_id}_mic.wav").write_bytes(b"wav")
                annotation_root.mkdir(parents=True, exist_ok=True)
                (annotation_root / f"{track_id}.jams").write_text("{}", encoding="utf-8")


def test_dev_guitarset_discovery_is_q6_wav_only_and_excludes_player04(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    _write_guitarset_wav_bank(paths)

    def labels_are_forbidden(*args: Any, **kwargs: Any) -> list[experiment.ClipEntry]:
        del args, kwargs
        raise AssertionError("development discovery must not scan GuitarSet labels")

    monkeypatch.setattr(experiment, "scan_guitarset", labels_are_forbidden)
    clips = experiment.discover_clips(paths, ["guitarset-dev"], force_q6=True)

    assert len(clips) == 300
    assert {clip.corpus for clip in clips} == {"guitarset-dev"}
    assert {clip.player for clip in clips} == {"00", "01", "02", "03", "05"}
    assert all(clip.event_cache_strategy.startswith("q6") for clip in clips)


def test_partial_egset_is_excluded_and_marked_blocked_unscored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    partial = experiment.ClipEntry(
        id="egset12/partial",
        tier="clean_electric",
        source="EGSet12",
        split="test",
        media_path=str(tmp_path / "partial.wav"),
        annotation_path=str(tmp_path / "partial.jams"),
        annotation_format="egset12_jams",
    )
    monkeypatch.setattr(experiment, "scan_egset12", lambda _root: [partial])
    status: dict[str, dict[str, Any]] = {}

    clips = experiment.discover_clips(paths, ["egset12"], corpus_status=status)

    assert clips == []
    assert status["egset12"] == {
        "status": "blocked_unscored",
        "expected_clips": 12,
        "discovered_clips": 0,
        "reason": "official digest-verified EGSet12 WAV/JAMS pairs are incomplete",
    }


def test_posterior_cache_identity_binds_runtime_and_chunk_size(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    revision = {"git_revision": "abc", "evaluation_sha256": "code"}
    baseline = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
        runtime={"python": "3.11", "packages": {"torch": "2.11.0"}},
        chunk_size=256,
    )
    changed_runtime = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
        runtime={"python": "3.12", "packages": {"torch": "2.11.0"}},
        chunk_size=256,
    )
    changed_chunk = experiment.posterior_cache_identity(
        clip.audio_path,
        spec,
        code_revision=revision,
        runtime={"python": "3.11", "packages": {"torch": "2.11.0"}},
        chunk_size=128,
    )

    assert changed_runtime.key != baseline.key
    assert changed_chunk.key != baseline.key


def test_posterior_cache_default_uses_frozen_generation_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path)
    spec = _model(tmp_path)
    monkeypatch.setattr(
        experiment,
        "evaluation_code_revision",
        lambda: {"git_revision": "new", "evaluation_sha256": "downstream-change"},
    )

    identity = experiment.posterior_cache_identity(clip.audio_path, spec)

    assert identity.metadata["code_revision"] == dict(
        experiment.FROZEN_POSTERIOR_GENERATION_REVISION
    )


def test_align_priors_projects_only_decoder_retained_events() -> None:
    first = np.asarray([[1.0]])
    second = np.asarray([[2.0]])
    raw_events = [
        AudioEvent(0.1, 0.2, 60, 0.8, 0.9),
        AudioEvent(0.3, 0.4, 64, 0.8, 0.9),
    ]
    decoded = [
        TabEvent(
            onset_s=0.3,
            duration_s=0.1,
            string_idx=1,
            fret=2,
            pitch_midi=64,
            confidence=0.9,
        )
    ]

    aligned = experiment._align_priors_to_decoded(
        raw_events,
        [first, second],
        decoded,
    )

    assert aligned == [second]


def test_align_priors_rejects_decoded_event_without_raw_origin() -> None:
    decoded = [
        TabEvent(
            onset_s=0.2,
            duration_s=0.1,
            string_idx=1,
            fret=2,
            pitch_midi=64,
            confidence=0.9,
        )
    ]

    with pytest.raises(RuntimeError, match="no exact raw-event prior alignment"):
        experiment._align_priors_to_decoded(
            [AudioEvent(0.1, 0.2, 64, 0.8, 0.9)],
            [None],
            decoded,
        )


def test_frozen_scoring_environment_rejects_tabvision_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABVISION_TEST_OVERRIDE", "1")

    with pytest.raises(RuntimeError, match="TABVISION_\\* overrides"):
        experiment.assert_frozen_scoring_environment()


def test_evaluate_rejects_transcription_even_before_clip_processing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="prebank-only"):
        experiment.evaluate(
            [],
            [],
            _paths(tmp_path),
            allow_transcribe_missing=True,
        )


def test_license_evidence_requires_exact_pinned_runtime_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "runtime_manifest",
        lambda: {
            "packages": {
                "torch": "2.11.0",
                "librosa": "0.11.0",
                "onnxruntime": "1.23.2",
            }
        },
    )

    checks = experiment.license_evidence(_model(tmp_path), _paths(tmp_path))["checks"]

    assert checks["torch_2_11_0_eval_only"] is True
    assert checks["librosa_0_11_0_eval_only"] is True
    assert checks["onnxruntime_1_23_2_eval_only"] is True


def test_development_result_requires_exact_cross_product_and_receipts() -> None:
    clip_ids = {f"guitarset/dev-{index:03d}" for index in range(300)}
    identity = {
        "code_revision": {"evaluation_sha256": "code"},
        "posterior_code_revision": {"evaluation_sha256": "posterior-code"},
        "runtime": {"python": "3.11"},
        "frozen_scoring": {"decoder": "baseline"},
        "protocol_identity": {"protocol_sha256": "frozen"},
        "guitarset_lopo": {"position_priors_sha256": "position"},
    }
    rows = [
        {
            "model": model,
            "clip_id": clip_id,
            "corpus": "guitarset-dev",
            "onset_pitch_invariant": True,
            "posterior_cache": {"determinism": {"verified": True}},
        }
        for model in experiment.MODELS
        for clip_id in sorted(clip_ids)
    ]
    payload: dict[str, Any] = {
        "models": {model: {"gate": {}} for model in experiment.MODELS},
        "per_clip": rows,
        "development_input_identity": identity,
        "code_revision": identity["code_revision"],
        "posterior_code_revision": identity["posterior_code_revision"],
        "runtime": identity["runtime"],
        "frozen_scoring": identity["frozen_scoring"],
        "protocol_identity": identity["protocol_identity"],
        "guitarset_lopo": identity["guitarset_lopo"],
        "q6_cache_reproduction": {"passed": True},
        "event_bank_ledgers": {"verified": True},
    }

    experiment._validate_completed_development_result(
        payload,
        expected_identity=identity,
        expected_clip_ids=clip_ids,
    )

    payload["protocol_identity"] = {"protocol_sha256": "tampered"}
    with pytest.raises(RuntimeError, match="protocol_identity"):
        experiment._validate_completed_development_result(
            payload,
            expected_identity=identity,
            expected_clip_ids=clip_ids,
        )
    payload["protocol_identity"] = identity["protocol_identity"]

    rows[-1] = {**rows[-1], "clip_id": rows[0]["clip_id"]}
    with pytest.raises(RuntimeError, match="exact 300-clip x two-model"):
        experiment._validate_completed_development_result(
            payload,
            expected_identity=identity,
            expected_clip_ids=clip_ids,
        )

    rows[-1] = {
        **rows[-2],
        "clip_id": sorted(clip_ids)[-1],
        "posterior_cache": {"determinism": {"verified": False}},
    }
    with pytest.raises(RuntimeError, match="determinism"):
        experiment._validate_completed_development_result(
            payload,
            expected_identity=identity,
            expected_clip_ids=clip_ids,
        )


def test_development_input_identity_is_json_round_trip_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_clip = replace(
        _clip(tmp_path),
        corpus="guitarset-dev",
        source="GuitarSet",
        split="dev",
        player="00",
        mode="solo",
    )
    clips = [
        replace(base_clip, clip_id=f"guitarset/dev-{index:03d}")
        for index in range(experiment.EXPECTED_CORPUS_COUNTS["guitarset-dev"])
    ]
    base_model = _model(tmp_path)
    specs = [replace(base_model, name=model) for model in experiment.MODELS]
    monkeypatch.setattr(
        experiment,
        "validate_q6_guitarset_cache",
        lambda _paths: {"passed": True, "guitar_config": {"tuning_midi": (40, 45)}},
    )
    monkeypatch.setattr(
        experiment,
        "validate_event_bank_ledgers",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        experiment,
        "license_evidence",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        experiment,
        "evaluation_code_revision",
        lambda: {"git_revision": "git", "evaluation_sha256": "evaluation"},
    )
    monkeypatch.setattr(experiment, "runtime_manifest", lambda: {"python": "test"})
    monkeypatch.setattr(experiment, "assert_frozen_scoring_environment", lambda: {})
    monkeypatch.setattr(experiment, "protocol_identity", lambda: {"verified": True})
    monkeypatch.setattr(
        experiment,
        "build_frozen_guitarset_lopo",
        lambda *_args, **_kwargs: ({}, {}, {}, {"players": ("00", "01")}),
    )

    identity = experiment._development_input_identity(clips, specs, _paths(tmp_path))

    assert json.loads(json.dumps(identity)) == identity
    assert identity["q6_cache_reproduction"]["guitar_config"]["tuning_midi"] == [40, 45]
    assert identity["guitarset_lopo"]["players"] == ["00", "01"]


def test_canonical_json_normalizes_nested_tuples_for_result_readback() -> None:
    result = {
        "guitar_config": {"tuning_midi": (40, 45, 50, 55, 59, 64)},
        "models": ("synthtab", "dafx"),
    }

    normalized = json.loads(experiment._canonical_json(result))

    assert normalized == json.loads(json.dumps(normalized))
    assert normalized["guitar_config"]["tuning_midi"] == [40, 45, 50, 55, 59, 64]
    assert normalized["models"] == ["synthtab", "dafx"]


def test_transfer_bank_refuses_unledgered_and_session_mismatched_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = _clip(tmp_path)
    paths = _paths(tmp_path)
    backend_identity = {"identity_sha256": "backend"}
    monkeypatch.setitem(experiment.EXPECTED_CORPUS_COUNTS, "egset12", 1)
    monkeypatch.setattr(
        experiment,
        "highres_bank_backend_identity",
        lambda: backend_identity,
    )

    with pytest.raises(RuntimeError, match="unledgered adoption"):
        experiment.bank_transfer_events(
            [clip],
            paths,
            allow_transcribe_missing=False,
        )

    ledger_path = paths.cache_root / "egset12-raw-event-bank-v2.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": clip.clip_id,
                        "audio_sha256": experiment.sha256_file(clip.audio_path),
                        "event_cache_sha256": experiment.sha256_file(clip.event_cache_path),
                        "backend_identity_sha256": "backend",
                        "session": {"instrument": "wrong"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="stale transfer bank ledger"):
        experiment.bank_transfer_events(
            [clip],
            paths,
            allow_transcribe_missing=False,
        )


def test_gaps_legacy_exception_requires_frozen_audio_and_event_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = replace(
        _clip(tmp_path),
        clip_id="gaps/fixed",
        corpus="gaps",
        source="GAPS",
        tier="classical",
        annotation_format="gaps",
    )
    expected = {
        clip.clip_id: {
            "audio_sha256": experiment.sha256_file(clip.audio_path),
            "event_cache_sha256": experiment.sha256_file(clip.event_cache_path),
        }
    }
    monkeypatch.setattr(experiment, "GAPS_LEGACY_CLEAN12", expected)
    monkeypatch.setitem(experiment.EXPECTED_CORPUS_COUNTS, "gaps", 1)

    evidence = experiment.validate_event_bank_ledgers(
        [clip],
        _paths(tmp_path),
    )
    assert evidence["corpora"]["gaps"]["verified"] is True
    assert evidence["corpora"]["gaps"]["origin"] == "hash_pinned_legacy_gaps_exception"

    original = clip.event_cache_path.read_bytes()
    clip.event_cache_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(RuntimeError, match="event-cache identity mismatch"):
        experiment.validate_event_bank_ledgers(
            [clip],
            _paths(tmp_path),
        )


@pytest.mark.parametrize(
    "corpus",
    ["guitarset-sealed", "gaps", "egset12", "guitar-techs"],
)
def test_limit_is_restricted_to_guitarset_development(
    corpus: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment, "assert_frozen_scoring_environment", lambda: {})
    monkeypatch.setattr(experiment, "protocol_identity", lambda: {})
    with pytest.raises(SystemExit, match="sealed and transfer peeks are forbidden"):
        experiment.main(["manifest", "--corpus", corpus, "--limit", "1"])


def test_sealed_bank_refuses_unledgered_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    audio = tmp_path / "sealed.wav"
    audio.write_bytes(b"wav")
    event_cache = tmp_path / "sealed.ensemble.json"
    event_cache.write_text("[]", encoding="utf-8")
    target = experiment.RawBankTarget(
        clip_id="guitarset/04_fixture_solo",
        track_id="04_fixture_solo",
        player="04",
        mode="solo",
        audio_path=audio,
        event_cache_path=event_cache,
    )
    monkeypatch.setattr(
        experiment, "_guitarset_audio_bank_targets", lambda *_args, **_kwargs: [target]
    )
    monkeypatch.setattr(experiment, "validate_development_unlock", lambda _paths: {"passed": True})
    monkeypatch.setattr(
        experiment,
        "highres_bank_backend_identity",
        lambda: {"identity_sha256": "backend"},
    )

    with pytest.raises(RuntimeError, match="unledgered adoption"):
        experiment.bank_guitarset_events(
            paths,
            split="sealed",
            allow_transcribe_missing=False,
        )


def test_lopo_identity_hashes_all_annotations_and_effective_priors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    annotation_root = paths.guitarset_root / "annotation"
    annotation_root.mkdir(parents=True)
    gold: dict[str, dict[str, list[Any]]] = {}
    for player in experiment.GUITARSET_PLAYERS:
        gold[player] = {}
        for index in range(60):
            track_id = f"{player}_fixture_{index:02d}"
            (annotation_root / f"{track_id}.jams").write_bytes(f"{player}:{index}".encode())
            gold[player][track_id] = []
    monkeypatch.setattr(experiment, "gold_by_player", lambda *_args: gold)
    monkeypatch.setattr(
        experiment,
        "build_loo_priors",
        lambda *_args: (
            {"04": np.asarray([[0.25, 0.75]], dtype=np.float64)},
            {"04": {"delta": (0.1, 0.9)}},
        ),
    )

    _gold, _positions, _sequences, first = experiment.build_frozen_guitarset_lopo(
        paths,
        GuitarConfig(),
    )
    assert first["sealed_annotation_use"] == "LOPO_training_only"
    assert len(first["sealed_annotations"]) == 60

    (annotation_root / "04_fixture_00.jams").write_bytes(b"changed")
    _gold, _positions, _sequences, changed = experiment.build_frozen_guitarset_lopo(
        paths,
        GuitarConfig(),
    )
    assert changed["annotations_sha256"] != first["annotations_sha256"]
    assert changed["position_priors_sha256"] == first["position_priors_sha256"]
