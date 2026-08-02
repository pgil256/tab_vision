"""Run the frozen, evaluation-only TabCNN complementarity experiment.

The high-resolution event stream is always loaded from a verified local cache.
TabCNN contributes only event-aligned ``AudioEvent.fret_prior`` evidence.  The
runner never downloads data, never mutates registered artifacts, and never
transcribes a missing clip unless ``--allow-transcribe-missing`` is explicit.

Stages are intentionally separable and resumable::

    python -m scripts.eval.tabcnn_complementarity_experiment manifest
    python -m scripts.eval.tabcnn_complementarity_experiment cache-posteriors
    python -m scripts.eval.tabcnn_complementarity_experiment evaluate
    python -m scripts.eval.tabcnn_complementarity_experiment all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.phase0_rotation_baseline import (
    BURNED_PLAYER,
    POSITION_ALPHA,
    POSITION_POWER,
    REPRODUCTION,
    REPRODUCTION_TOLERANCE,
    SEALED_PLAYER,
    SEQUENCE_ALPHA,
    SEQUENCE_BACKOFF_KAPPA,
    SEQUENCE_SCHEME,
    SEQUENCE_SINGLETON_ONLY,
    build_loo_priors,
    gold_by_player,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.egset12 import parse_egset12_jams, scan_egset12
from tabvision.eval.error_decomposition import decompose_errors
from tabvision.eval.guitar_techs import parse_guitar_techs_jams, scan_guitar_techs
from tabvision.eval.guitarset_audio import (
    AudioOnlyScore,
    EventF1Result,
    load_mono_audio,
    parse_guitarset_jams,
    score_audio_only,
)
from tabvision.eval.highres_event_bank import (
    events_to_json as _events_to_json,
)
from tabvision.eval.highres_event_bank import (
    highres_bank_backend_identity,
    read_banked_events,
)
from tabvision.eval.highres_event_bank import (
    load_mono_audio as load_bank_audio,
)
from tabvision.eval.highres_event_bank import (
    new_highres_bank_backend as _new_highres_bank_backend,
)
from tabvision.eval.manifest_builder import ClipEntry, scan_guitarset
from tabvision.eval.parsers.registry import get_parser
from tabvision.eval.tabcnn_artifacts import (
    DAFX_GUITARPROFX_ONNX,
    SHARED_CQT,
    SYNTHTAB_X4,
    artifact_manifest,
)
from tabvision.eval.tabcnn_complementarity import (
    CONSERVATIVE_BLEND_POLICY,
    POSTERIOR_ONLY_POLICY,
    AggregateEvaluation,
    ClipEvaluation,
    ComplementarityResult,
    aggregate_clip_evaluations,
    attach_tabcnn_priors,
    evaluate_complementarity,
    score_clip,
)
from tabvision.eval.tabcnn_posterior import (
    BINS_PER_OCTAVE,
    DEFAULT_CHUNK_SIZE,
    FMIN_HZ,
    HOP_LENGTH,
    N_BINS,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    CQTWindowBatch,
    DAFxTabCNNPosterior,
    FeatureNormalization,
    FramePosteriors,
    PosteriorBackend,
    SynthTabX4Posterior,
    cqt_windows,
    event_fret_prior,
    posterior_sha256,
    sha256_file,
    validate_checkpoint,
)
from tabvision.eval.tabcnn_runtime_evidence import (
    RECEIPT_FORMAT_VERSION,
    load_cache_performance_receipt,
    peak_rss_bytes,
    write_cache_performance_receipt,
)
from tabvision.fusion import chord, chord_shapes, playability
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.playability import set_transition_prior
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.string_physics import load_string_evidence, reference_stiffness_model
from tabvision.fusion.transition_prior import load_transition_prior
from tabvision.fusion.viterbi import assignment_decoder_context
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT, sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

CORPORA = ("guitarset-dev", "guitarset-sealed", "gaps", "egset12", "guitar-techs")
MODELS = ("synthtab", "dafx")
FRONTEND_VERSION = "tabcnn-librosa-cqt-22050-v2"
CACHE_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
EXPECTED_GUITARSET_CLIPS = 360
GUITARSET_PLAYERS = ("00", "01", "02", "03", "04", "05")
EXPECTED_CORPUS_COUNTS: Mapping[str, int] = {
    "guitarset-dev": 300,
    "guitarset-sealed": 60,
    "gaps": len(CLEAN_12),
    "egset12": 12,
    "guitar-techs": 82,
}
EXPECTED_GAPS_CLIPS = len(CLEAN_12)
EXPECTED_EGSET12_CLIPS = 12
EXPECTED_GUITAR_TECHS_CLIPS = 82
GAPS_LEGACY_CLEAN12: Mapping[str, Mapping[str, str]] = {
    "gaps/027_Zpswc": {
        "audio_sha256": "ded7c38360517b121caf95c872bfd06f5183963df0ac657098972dd2646ee0ef",
        "event_cache_sha256": "90de34f09c06614818711a49df6ffcf5a63263882463ec1b27e497a77d5ffe06",
    },
    "gaps/031_vpswc": {
        "audio_sha256": "83b48ebd04c9ec3d1402d79dbd153453570ecec63f03e9093ce8428459f1791d",
        "event_cache_sha256": "d97c0a31eb84d91097f181c31a8ec7bdee4d45cb07859a5a7abc43db13c5d1ea",
    },
    "gaps/043_bc1wc": {
        "audio_sha256": "e601c0cd15e863a153037a32199c7e3ca8760fc46d467ba153249d7c65cf8c8b",
        "event_cache_sha256": "85d94f855c66f0625da3aeed80a5d2df8c69fa41ce2b3fb6da06a0f0ce620701",
    },
    "gaps/063_bV1wc": {
        "audio_sha256": "cd131d37197a4a770f8b0f6c81fda47521b1398fa18e7f16d0c12c6e640d7942",
        "event_cache_sha256": "d75a3011e2f6069e9202694ea16bc93c0d14377e0149c4cc7a663e0b4f7efc7e",
    },
    "gaps/104_xf1wc": {
        "audio_sha256": "f3b703c8154739a68cce921f027141c5b9ad0650abc190a8a411d2e3609c02fe",
        "event_cache_sha256": "65299c7324a15f9b64503e906f8db2434872d003552ce6dfb1186e30af4f104c",
    },
    "gaps/118_VD1wc": {
        "audio_sha256": "adbbcb6914afdb2f52ecc42279e6fcc07709f7d8cb41abbba0ed8d8b41e782cb",
        "event_cache_sha256": "bc47e597f5502a1467fb4fb21d84517fff362f3e73c52417dc8fbb2611f17968",
    },
    "gaps/142_GD1wc": {
        "audio_sha256": "cec4deca269141974ffb3ffbaaa394d4185de622fb0543c371f1c94bf46ab27d",
        "event_cache_sha256": "1f8616a33cad1efbc923748db1ba4250ff586cf868147536f77ea3786cca369a",
    },
    "gaps/179_pM1wc": {
        "audio_sha256": "9404e058de84aa55f41f7b7ac502483d1a9383437ab4e77969b31dcef8d842c6",
        "event_cache_sha256": "c26d2d4a1b30d742cea490a9dd821ce48ef6fb620762f027d42d8db9eb4e3a4a",
    },
    "gaps/212_y41wc": {
        "audio_sha256": "41a17b3d1aa388c36def2ef196ce1c6a7b907abfadb4f6f91b163454f1680df3",
        "event_cache_sha256": "cf3588a8442dae13d4b8a75949a8178e86c2e1e17251b9ea30a76131667b7077",
    },
    "gaps/235_Ny1wc": {
        "audio_sha256": "5e79b650f3f44dcc2742968196e8d4e6e6876778a2fd0968abc2c3b3e461ccfd",
        "event_cache_sha256": "c1ad760d2a4b8b1e57f3be9050eec3e8af6343dcbdd23e2411065985fe15c4b1",
    },
    "gaps/294_BSswc": {
        "audio_sha256": "8cb6b7a761db18fd0fb8e1340a96d03dcfad93e23d541dd67d68ffa5bbeca070",
        "event_cache_sha256": "a269a88c717b42824ea7f063aecc6d9b5992720250d2b1cf47a54cd7d0152e63",
    },
    "gaps/341_1M1wc": {
        "audio_sha256": "5a5b961938fc99c3c0fa7dfc5168ca62cd7655223ae1fdae0d5042c027f72f41",
        "event_cache_sha256": "ff7888d5519f90ffdfa1b445338621bc2d22c689e9a6e2d2b8525a16425fea45",
    },
}
FROZEN_CURRENT_LATENCY_SECONDS = 262.495
PROTOCOL_RELATIVE_PATH = Path("docs/plans/2026-07-29-tabcnn-complementarity-experiment.md")
EXPECTED_PROTOCOL_SHA256 = "7d7aa1dd080f68e1672df4834bb8a874cd5ad3726e22bb45bea777bbf91a94d5"
FROZEN_POSTERIOR_GENERATION_REVISION: Mapping[str, str] = {
    "git_revision": "febd38c2d57c6409a1451e8b8ac5ffc958ea45a9",
    "evaluation_sha256": "f326875670b5cf4f56ace26b395e2f494e2d5b8a683ece277f20ad0e7ce6154c",
}
DEFAULT_CURRENT_LATENCY_SOURCE = (
    "docs/EVAL_REPORTS/string_assignment_phase7_2026-07-16.md "
    "cold current backend 258.045 s/60s + "
    "docs/EVAL_REPORTS/n1_partial_aware_isolation_2026-07-23.md "
    "worst dense partial-aware 4.45 s/60s"
)
PINNED_GUITAR_TECHS_REVISION = "4448053ced18e67a9f66bfab47ac2de3cc0b4521"
GUITAR_TECHS_METADATA_PATHS = (
    Path(".cache/huggingface/download/README.md.metadata"),
    Path(".cache/huggingface/download/clips/manifest.json.metadata"),
)
EXPECTED_LICENSES_SHA256 = "4512e606f39cc5fb9b12bf6825940573c88f719071560e1aea4fe37c835b18f8"
EVENT_BANK_LEDGER_VERSION = 2
FROZEN_LEGACY_Q6_DEV_ATTESTATION: Mapping[str, Any] = {
    "ledger_sha256": "761c4102c5100f82ad65e4bb8bb6a1ba763b7234aef57790ee9276bcbabb8261",
    "backend_identity_sha256": ("5fb01c8804c0e754a2362a8ab2cad20aaba95e8569117a61c12978852e2d27cf"),
    "evaluation_sha256": "4d7c884b532210cbac47184e11b3388933e0f5bcdff0e6982a90298a03c676c9",
    "git_revision": "febd38c2d57c6409a1451e8b8ac5ffc958ea45a9",
    "clips": 300,
    "published": {"baseline": 0.634, "shipped": 0.7346},
    "observed": {
        "baseline": 0.633962124525772,
        "shipped": 0.7345805362306649,
    },
    "tolerance": 0.0015,
}
FROZEN_LEGACY_Q6_SOURCE_SHA256: Mapping[str, str] = {
    "tabvision/audio/checkpoint_ensemble.py": (
        "e177a18578acf01df52a0e97a1e25cbfdc3c44ddb24e4eff9e8f0059c63f6307"
    ),
    "tabvision/audio/highres.py": (
        "b29fda503b508d3f1e7d6c64cdcf6442799be9f4b9aa4e0f0c41d5f9e7e62c46"
    ),
    "tabvision/audio/highres_ensemble.py": (
        "1e144f9aac75e95717203dc41e6608eb9bb17516d7b48cab4360d3d5dfe4096b"
    ),
    "tabvision/eval/guitarset_audio.py": (
        "cadef42787e29143de8911338525cc08fdb0f755bf75b3fd29cc17fd8703f168"
    ),
    "tabvision/types.py": ("2290081cadd929b79fbc4ea727521d289f8778a2956665548e571aa485046e34"),
}
FROZEN_LEGACY_Q6_RUNTIME_VERSIONS: Mapping[str, str] = {
    "torch": "2.11.0",
    "numpy": "2.4.4",
    "scipy": "1.17.1",
    "soundfile": "0.13.1",
    "hf-midi-transcription": "0.1.1",
    "piano-transcription-inference": "0.1.0",
    "pretty-midi": "0.2.11",
    "mido": "1.3.3",
}
ERROR_BUCKETS = (
    "correct",
    "wrong_position_same_pitch",
    "pitch_off",
    "timing_only",
    "missed_onset",
    "extra_detection",
)
SCORING_DATA_ROOT_ENV = "TABVISION_DATA_ROOT"

FROZEN_PLAYABILITY_CONSTANTS: Mapping[str, float | int | str] = {
    "low_fret_bias": 0.10,
    "open_string_bonus": 0.5,
    "fret_prior_weight": 1.0,
    "same_string_bonus": 0.5,
    "position_shift_cost": 2.5,
    "span_norm": 12.0,
    "max_hand_span": 5,
    "hand_span_barrier": 5.0,
    "transition_gap_tau": "inf",
    "string_confidence_temp": 1.0,
    "chord_shape_bonus": 0.1,
    "chord_shape_min_notes": 3,
    "chord_max_gap_s": 0.080,
}

SYNTHTAB_SHA256 = SYNTHTAB_X4.sha256
DAFX_SHA256 = DAFX_GUITARPROFX_ONNX.sha256
CQT_BIN_SHA256 = SHARED_CQT.sha256

ROUTING: Mapping[str, Mapping[str, Any]] = {
    "guitarset-dev": {
        "position": "leave-one-player-out",
        "sequence": "leave-one-player-out",
        "physics": "acoustic-physics-v1/partial_aware",
    },
    "guitarset-sealed": {
        "position": "leave-one-player-out",
        "sequence": "leave-one-player-out",
        "physics": "acoustic-physics-v1/partial_aware",
    },
    "gaps": {
        "position": "gaps-v1",
        "sequence": "gaps-seq-v1",
        "physics": None,
    },
    "egset12": {"position": None, "sequence": None, "physics": None},
    "guitar-techs": {"position": None, "sequence": None, "physics": None},
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    expected_sha256: str
    family: str
    guitarset_overlap: bool
    frontend_normalization: FeatureNormalization
    artifact_id: str
    source_revision: str
    license_id: str
    license_posture: str
    evaluation_allowed: bool
    shipping_redistribution_allowed: bool


@dataclass(frozen=True)
class ExperimentClip:
    clip_id: str
    corpus: str
    source: str
    split: str
    tier: str
    player: str | None
    mode: str | None
    audio_path: Path
    annotation_path: Path
    annotation_format: str
    event_cache_path: Path
    event_cache_strategy: str


@dataclass(frozen=True)
class PosteriorCacheIdentity:
    key: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CachedPosterior:
    frames: FramePosteriors
    metadata: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class PosteriorComputation:
    frames: FramePosteriors
    load_seconds: float
    resample_seconds: float
    cqt_seconds: float
    inference_seconds: float
    duration_s: float
    original_sample_rate: int


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    model_root: Path
    cache_root: Path
    guitarset_root: Path
    gaps_root: Path
    egset12_root: Path
    guitar_techs_root: Path
    q6_dev_cache: Path
    q6_player05_cache: Path
    q6_gaps_cache: Path
    legacy_guitarset_cache: Path


@dataclass(frozen=True)
class RawBankTarget:
    clip_id: str
    track_id: str
    player: str
    mode: str
    audio_path: Path
    event_cache_path: Path


class PosteriorComputer(Protocol):
    def __call__(
        self,
        audio_path: Path,
        backend: PosteriorBackend,
        *,
        chunk_size: int,
    ) -> PosteriorComputation: ...


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _scoring_source_paths() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[2]
    return (
        Path(__file__).resolve(),
        package_root / "scripts" / "eval" / "phase0_rotation_baseline.py",
        package_root / "tabvision" / "audio" / "checkpoint_ensemble.py",
        package_root / "tabvision" / "audio" / "filters.py",
        package_root / "tabvision" / "audio" / "highres.py",
        package_root / "tabvision" / "audio" / "highres_ensemble.py",
        package_root / "tabvision" / "audio" / "ensemble_v1.json",
        package_root / "tabvision" / "eval" / "error_decomposition.py",
        package_root / "tabvision" / "eval" / "guitarset_audio.py",
        package_root / "tabvision" / "eval" / "highres_event_bank.py",
        package_root / "tabvision" / "eval" / "metrics.py",
        package_root / "tabvision" / "eval" / "tabcnn_complementarity.py",
        package_root / "tabvision" / "eval" / "tabcnn_posterior.py",
        package_root / "tabvision" / "eval" / "tabcnn_runtime_evidence.py",
        package_root / "tabvision" / "fusion" / "candidates.py",
        package_root / "tabvision" / "fusion" / "chord.py",
        package_root / "tabvision" / "fusion" / "chord_shapes.py",
        package_root / "tabvision" / "fusion" / "evidence.py",
        package_root / "tabvision" / "fusion" / "inharmonicity.py",
        package_root / "tabvision" / "fusion" / "inference_policy.py",
        package_root / "tabvision" / "fusion" / "playability.py",
        package_root / "tabvision" / "fusion" / "position_prior.py",
        package_root / "tabvision" / "fusion" / "string_physics.py",
        package_root / "tabvision" / "fusion" / "transition_prior.py",
        package_root / "tabvision" / "fusion" / "viterbi.py",
        package_root / "tabvision" / "pipeline.py",
        package_root / "tabvision" / "fusion" / "priors" / "acoustic_physics_v1.json",
        package_root / "tabvision" / "fusion" / "priors" / "gaps_v1.json",
        package_root / "tabvision" / "fusion" / "priors" / "gaps_seq_v1.json",
    )


def evaluation_code_revision() -> dict[str, str]:
    """Identify both Git state and the exact evaluation source bytes."""

    source_digest = hashlib.sha256()
    for path in _scoring_source_paths():
        relative = path.relative_to(Path(__file__).resolve().parents[2])
        source_digest.update(relative.as_posix().encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(path.read_bytes())
    return {"git_revision": _git_revision(), "evaluation_sha256": source_digest.hexdigest()}


def posterior_generation_revision() -> dict[str, str]:
    """Identify the execution bytes that produced the verified posterior banks.

    The frozen revision was captured immediately before the two development
    cache runs.  Subsequent experiment-runner edits are restricted to
    downstream scoring diagnostics and do not alter audio loading, CQT,
    inference, cache serialization, or posterior mapping.
    """

    return dict(FROZEN_POSTERIOR_GENERATION_REVISION)


def protocol_identity() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[3] / PROTOCOL_RELATIVE_PATH
    observed = sha256_file(path) if path.is_file() else None
    verified = observed == EXPECTED_PROTOCOL_SHA256
    record = {
        "path": str(path),
        "expected_sha256": EXPECTED_PROTOCOL_SHA256,
        "observed_sha256": observed,
        "verified": verified,
    }
    if not verified:
        raise RuntimeError(
            f"the frozen complementarity protocol is missing or its bytes changed: {record}"
        )
    return record


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def runtime_manifest() -> dict[str, Any]:
    """Return runtime identity material that can affect output or CPU timing."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_model": _cpu_model(),
        "packages": {
            name: _package_version(name)
            for name in (
                "hf-midi-transcription",
                "huggingface-hub",
                "librosa",
                "numpy",
                "onnxruntime",
                "piano-transcription-inference",
                "scipy",
                "soundfile",
                "torch",
            )
        },
    }


def _effective_scoring_constants() -> dict[str, float | int | str]:
    return {
        "low_fret_bias": playability.LOW_FRET_BIAS,
        "open_string_bonus": playability.OPEN_STRING_BONUS,
        "fret_prior_weight": playability.FRET_PRIOR_WEIGHT,
        "same_string_bonus": playability.SAME_STRING_BONUS,
        "position_shift_cost": playability.POSITION_SHIFT_COST,
        "span_norm": playability.SPAN_NORM,
        "max_hand_span": playability.MAX_HAND_SPAN,
        "hand_span_barrier": playability.HAND_SPAN_BARRIER,
        "transition_gap_tau": (
            "inf" if math.isinf(playability.TRANSITION_GAP_TAU) else playability.TRANSITION_GAP_TAU
        ),
        "string_confidence_temp": playability.STRING_CONFIDENCE_TEMP,
        "chord_shape_bonus": chord_shapes.CHORD_SHAPE_BONUS,
        "chord_shape_min_notes": chord_shapes.CHORD_SHAPE_MIN_NOTES,
        "chord_max_gap_s": chord.CHORD_MAX_GAP_S,
    }


def assert_frozen_scoring_environment() -> dict[str, Any]:
    """Reject inherited experiment knobs and assert every effective constant."""

    overrides = sorted(
        name
        for name in os.environ
        if name.startswith("TABVISION_") and name != SCORING_DATA_ROOT_ENV
    )
    if overrides:
        raise RuntimeError(
            "scored TabCNN runs reject inherited TABVISION_* overrides: " + ", ".join(overrides)
        )
    effective = _effective_scoring_constants()
    if effective != dict(FROZEN_PLAYABILITY_CONSTANTS):
        raise RuntimeError(
            "effective decoder constants differ from the frozen protocol: "
            f"effective={effective}, expected={dict(FROZEN_PLAYABILITY_CONSTANTS)}"
        )
    if (
        POSTERIOR_ONLY_POLICY.tabcnn_weight != 1.0
        or POSTERIOR_ONLY_POLICY.include_existing
        or POSTERIOR_ONLY_POLICY.min_top_probability != 0.0
        or POSTERIOR_ONLY_POLICY.min_margin != 0.0
        or CONSERVATIVE_BLEND_POLICY.tabcnn_weight != 0.35
        or not CONSERVATIVE_BLEND_POLICY.include_existing
        or CONSERVATIVE_BLEND_POLICY.min_top_probability != 0.0
        or CONSERVATIVE_BLEND_POLICY.min_margin != 0.0
    ):
        raise RuntimeError("effective TabCNN fusion policies differ from the frozen protocol")
    if SEQUENCE_PRIOR_WEIGHT != 4.0:
        raise RuntimeError("effective sequence-prior weight differs from frozen value 4.0")
    return {
        "tabvision_environment_overrides": overrides,
        "decoder": "baseline",
        "playability": effective,
        "position_prior": {"alpha": POSITION_ALPHA, "power": POSITION_POWER},
        "sequence_prior": {
            "scheme": SEQUENCE_SCHEME,
            "alpha": SEQUENCE_ALPHA,
            "backoff_kappa": SEQUENCE_BACKOFF_KAPPA,
            "singleton_only": SEQUENCE_SINGLETON_ONLY,
            "decode_weight": SEQUENCE_PRIOR_WEIGHT,
        },
        "fusion": {
            "posterior_only": asdict(POSTERIOR_ONLY_POLICY),
            "current_plus_tabcnn": asdict(CONSERVATIVE_BLEND_POLICY),
        },
        "routing": ROUTING,
    }


def frontend_manifest(
    normalization: FeatureNormalization | None = None,
) -> dict[str, Any]:
    try:
        librosa_version = importlib.metadata.version("librosa")
    except importlib.metadata.PackageNotFoundError:
        librosa_version = "not-installed"
    return {
        "version": FRONTEND_VERSION,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "fmin_hz": FMIN_HZ,
        "n_bins": N_BINS,
        "bins_per_octave": BINS_PER_OCTAVE,
        "window_frames": WINDOW_FRAMES,
        "normalization": normalization or "model-specific",
        "implementation": "tabvision.eval.tabcnn_posterior.cqt_windows",
        "cqt_library": "librosa",
        "cqt_library_version": librosa_version,
        "reference_filterbank_loaded": False,
    }


def posterior_cache_identity(
    audio_path: str | Path,
    model_spec: ModelSpec,
    *,
    code_revision: Mapping[str, str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    runtime: Mapping[str, Any] | None = None,
) -> PosteriorCacheIdentity:
    """Return the content address for one audio/model/front-end tuple."""

    audio = Path(audio_path).resolve()
    checkpoint = Path(model_spec.checkpoint).resolve()
    validate_checkpoint(checkpoint, expected_sha256=model_spec.expected_sha256)
    metadata: dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "audio_path": str(audio),
        "audio_sha256": sha256_file(audio),
        "audio_size_bytes": audio.stat().st_size,
        "model": model_spec.name,
        "model_family": model_spec.family,
        "model_path": str(checkpoint),
        "model_sha256": model_spec.expected_sha256,
        "frontend": frontend_manifest(model_spec.frontend_normalization),
        "code_revision": dict(code_revision or posterior_generation_revision()),
        "runtime": dict(runtime or runtime_manifest()),
        "inference_chunk_size": chunk_size,
    }
    return PosteriorCacheIdentity(
        key=_sha256_bytes(_canonical_json(metadata).encode("utf-8")),
        metadata=metadata,
    )


def posterior_cache_path(
    cache_root: str | Path,
    clip: ExperimentClip,
    model_spec: ModelSpec,
    identity: PosteriorCacheIdentity,
) -> Path:
    return (
        Path(cache_root)
        / "posteriors"
        / model_spec.name
        / clip.corpus
        / f"{_safe_id(clip.clip_id)}.{identity.key}.npz"
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(raw_temp)
    try:
        np.savez_compressed(temp_path, **arrays)  # type: ignore[arg-type]
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _validate_posterior_frames(frames: FramePosteriors) -> None:
    probabilities = np.asarray(frames.probabilities)
    times_s = np.asarray(frames.times_s)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (6, 21):
        raise ValueError(f"invalid posterior shape {probabilities.shape}")
    if times_s.shape != (len(probabilities),):
        raise ValueError("posterior timestamps must have one value per frame")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("posterior probabilities must be finite and non-negative")
    if np.any(~np.isfinite(times_s)) or np.any(np.diff(times_s) < 0.0):
        raise ValueError("posterior timestamps must be finite and non-decreasing")
    totals = probabilities.sum(axis=-1)
    if not np.allclose(totals, 1.0, atol=1e-5):
        raise ValueError("posterior probabilities must sum to one per string")


def write_posterior_cache(
    path: str | Path,
    identity: PosteriorCacheIdentity,
    computation: PosteriorComputation,
) -> CachedPosterior:
    """Atomically commit a self-validating posterior cache as one NPZ."""

    destination = Path(path)
    _validate_posterior_frames(computation.frames)
    digest = posterior_sha256(
        computation.frames.probabilities,
        computation.frames.times_s,
    )
    metadata = {
        **identity.metadata,
        "cache_key": identity.key,
        "posterior_sha256": digest,
        "frames": len(computation.frames.times_s),
        "duration_s": computation.duration_s,
        "original_sample_rate": computation.original_sample_rate,
        "timing_seconds": {
            "audio_load": computation.load_seconds,
            "resample": computation.resample_seconds,
            "cqt": computation.cqt_seconds,
            "inference": computation.inference_seconds,
        },
    }
    metadata_json = _canonical_json(metadata)
    _atomic_npz(
        destination,
        probabilities=np.asarray(computation.frames.probabilities, dtype="<f4"),
        times_s=np.asarray(computation.frames.times_s, dtype="<f8"),
        metadata_json=np.asarray(metadata_json),
    )
    return load_posterior_cache(destination, identity)


def load_posterior_cache(
    path: str | Path,
    expected: PosteriorCacheIdentity,
) -> CachedPosterior:
    """Load a complete cache and reject stale metadata or partial contents."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            if set(payload.files) != {"probabilities", "times_s", "metadata_json"}:
                raise ValueError(f"incomplete posterior cache fields in {cache_path}")
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
            times_s = np.asarray(payload["times_s"], dtype=np.float64)
            metadata = json.loads(str(payload["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid posterior cache {cache_path}: {exc}") from exc
    if metadata.get("cache_key") != expected.key:
        raise ValueError(f"stale posterior cache key in {cache_path}")
    for name, value in expected.metadata.items():
        if metadata.get(name) != value:
            raise ValueError(f"stale posterior cache metadata field {name!r} in {cache_path}")
    frames = FramePosteriors(probabilities=probabilities, times_s=times_s)
    _validate_posterior_frames(frames)
    observed = posterior_sha256(probabilities, times_s)
    if observed != metadata.get("posterior_sha256"):
        raise ValueError(f"posterior digest mismatch in {cache_path}")
    return CachedPosterior(
        frames=frames,
        metadata=metadata,
        path=cache_path,
    )


def _resample_audio(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return np.asarray(waveform, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - optional environment readiness
        raise RuntimeError("scipy is required to resample TabCNN evaluation audio") from exc
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    return resample_poly(
        np.asarray(waveform, dtype=np.float32),
        up=SAMPLE_RATE // divisor,
        down=sample_rate // divisor,
    ).astype(np.float32, copy=False)


def compute_posteriors(
    audio_path: Path,
    backend: PosteriorBackend,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = load_mono_audio,
    feature_builder: Callable[..., CQTWindowBatch] = cqt_windows,
) -> PosteriorComputation:
    load_start = time.perf_counter()
    waveform, original_sr = audio_loader(audio_path)
    load_seconds = time.perf_counter() - load_start
    duration_s = len(waveform) / original_sr

    resample_start = time.perf_counter()
    resampled = _resample_audio(waveform, original_sr)
    resample_seconds = time.perf_counter() - resample_start

    cqt_start = time.perf_counter()
    windows = feature_builder(
        resampled,
        sample_rate=SAMPLE_RATE,
        normalization=backend.feature_normalization,
    )
    cqt_seconds = time.perf_counter() - cqt_start

    inference_start = time.perf_counter()
    probabilities = backend.predict_windows(windows.windows, chunk_size=chunk_size)
    inference_seconds = time.perf_counter() - inference_start
    frames = FramePosteriors(
        probabilities=np.asarray(probabilities, dtype=np.float32),
        times_s=np.asarray(windows.times_s, dtype=np.float64),
    )
    _validate_posterior_frames(frames)
    posterior_sha256(frames.probabilities, frames.times_s)
    return PosteriorComputation(
        frames=frames,
        load_seconds=load_seconds,
        resample_seconds=resample_seconds,
        cqt_seconds=cqt_seconds,
        inference_seconds=inference_seconds,
        duration_s=duration_s,
        original_sample_rate=original_sr,
    )


def ensure_posterior_cache(
    clip: ExperimentClip,
    model_spec: ModelSpec,
    cache_root: Path,
    backend: PosteriorBackend,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    code_revision: Mapping[str, str] | None = None,
    computer: PosteriorComputer = compute_posteriors,
) -> tuple[CachedPosterior, bool]:
    """Resume a valid cache or atomically compute it once."""

    identity = posterior_cache_identity(
        clip.audio_path,
        model_spec,
        code_revision=code_revision,
        chunk_size=chunk_size,
    )
    path = posterior_cache_path(cache_root, clip, model_spec, identity)
    if path.exists():
        return load_posterior_cache(path, identity), True
    computation = computer(clip.audio_path, backend, chunk_size=chunk_size)
    return write_posterior_cache(path, identity, computation), False


def determinism_marker_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.determinism.json")


def verify_posterior_determinism(
    clip: ExperimentClip,
    cached: CachedPosterior,
    backend: PosteriorBackend,
    *,
    chunk_size: int,
    computer: PosteriorComputer = compute_posteriors,
) -> dict[str, Any]:
    """Repeat the full front end + inference and record exact digest equality."""

    repeat = computer(clip.audio_path, backend, chunk_size=chunk_size)
    repeat_digest = posterior_sha256(
        repeat.frames.probabilities,
        repeat.frames.times_s,
    )
    expected_digest = str(cached.metadata["posterior_sha256"])
    marker = {
        "cache_key": cached.metadata["cache_key"],
        "expected_posterior_sha256": expected_digest,
        "repeat_posterior_sha256": repeat_digest,
        "verified": repeat_digest == expected_digest,
        "repeat_timing_seconds": {
            "audio_load": repeat.load_seconds,
            "resample": repeat.resample_seconds,
            "cqt": repeat.cqt_seconds,
            "inference": repeat.inference_seconds,
        },
    }
    _atomic_text(
        determinism_marker_path(cached.path),
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
    )
    if not marker["verified"]:
        raise RuntimeError(f"non-deterministic posterior output for {clip.clip_id}")
    return marker


def posterior_determinism_status(cached: CachedPosterior) -> dict[str, Any]:
    marker_path = determinism_marker_path(cached.path)
    if not marker_path.is_file():
        return {"verified": False, "reason": "repeat-run marker missing"}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verified": False, "reason": f"invalid marker: {exc}"}
    valid = (
        marker.get("verified") is True
        and marker.get("cache_key") == cached.metadata.get("cache_key")
        and marker.get("expected_posterior_sha256") == cached.metadata.get("posterior_sha256")
        and marker.get("repeat_posterior_sha256") == cached.metadata.get("posterior_sha256")
    )
    return {**marker, "verified": valid}


class RawEventProvider:
    """Read local caches and keep explicit transcription isolated."""

    def __init__(self, *, allow_transcribe_missing: bool) -> None:
        if allow_transcribe_missing:
            raise RuntimeError("raw-event transcription is isolated to the bank-events stage")
        self.allow_transcribe_missing = allow_transcribe_missing
        self._backend: None = None

    def load(self, clip: ExperimentClip) -> list[AudioEvent]:
        if clip.event_cache_path.is_file():
            return read_banked_events(clip.event_cache_path)
        if not self.allow_transcribe_missing:
            raise FileNotFoundError(
                f"missing banked events for {clip.clip_id}: {clip.event_cache_path}; "
                "the runner refuses implicit transcription"
            )
        raise AssertionError("unreachable: missing event banks are rejected above")


def _guitarset_audio_bank_targets(paths: RuntimePaths, *, split: str) -> list[RawBankTarget]:
    """Enumerate q6 bank targets from WAV names only; never construct labels."""

    if split not in {"dev", "sealed"}:
        raise ValueError("GuitarSet bank split must be dev or sealed")
    players = (
        tuple(player for player in ("00", "01", "02", "03", "04", "05") if player != SEALED_PLAYER)
        if split == "dev"
        else (SEALED_PLAYER,)
    )
    targets: list[RawBankTarget] = []
    for audio_path in sorted((paths.guitarset_root / "audio_mono-mic").glob("*_mic.wav")):
        track_id = audio_path.stem.removesuffix("_mic")
        player = track_id[:2]
        if player not in players:
            continue
        mode = "solo" if track_id.endswith("_solo") else "comp"
        targets.append(
            RawBankTarget(
                clip_id=f"guitarset/{track_id}",
                track_id=track_id,
                player=player,
                mode=mode,
                audio_path=audio_path.resolve(),
                event_cache_path=_q6_guitarset_path(track_id, paths),
            )
        )
    expected_total = (
        EXPECTED_CORPUS_COUNTS["guitarset-dev"]
        if split == "dev"
        else EXPECTED_CORPUS_COUNTS["guitarset-sealed"]
    )
    if len(targets) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} {split} GuitarSet WAV targets, found {len(targets)}"
        )
    counts = Counter((target.player, target.mode) for target in targets)
    expected_counts = {(player, mode): 30 for player in players for mode in ("solo", "comp")}
    if counts != Counter(expected_counts):
        raise RuntimeError(
            f"{split} GuitarSet WAV rotation is malformed: "
            f"observed={dict(counts)}, expected={expected_counts}"
        )
    if split == "dev":
        targets.sort(key=lambda target: (target.player != BURNED_PLAYER, target.track_id))
    return targets


def _event_bank_v2_path(paths: RuntimePaths, corpus: str) -> Path:
    if corpus == "guitarset-dev":
        name = "q6-dev-raw-event-bank-v2.json"
    elif corpus == "guitarset-sealed":
        name = "q6-sealed-raw-event-bank-v2.json"
    else:
        name = f"{_safe_id(corpus)}-raw-event-bank-v2.json"
    return paths.cache_root / name


def _legacy_q6_dev_paths(paths: RuntimePaths) -> tuple[Path, Path]:
    ledger = paths.cache_root / "q6-dev-raw-event-bank.json"
    backup = paths.cache_root / (
        f"q6-dev-raw-event-bank-{str(FROZEN_LEGACY_Q6_DEV_ATTESTATION['ledger_sha256'])[:16]}.json"
    )
    return ledger, backup


def _validate_frozen_legacy_q6_dev(
    paths: RuntimePaths,
    targets: Sequence[RawBankTarget],
    current_backend: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attest the one immutable pre-v2 q6 bank without relabelling its identity."""

    ledger_path, backup_path = _legacy_q6_dev_paths(paths)
    expected_ledger_sha = str(FROZEN_LEGACY_Q6_DEV_ATTESTATION["ledger_sha256"])
    for candidate in (ledger_path, backup_path):
        if not candidate.is_file() or sha256_file(candidate) != expected_ledger_sha:
            raise RuntimeError("frozen legacy q6 ledger or immutable backup is missing/tampered")
    if ledger_path.read_bytes() != backup_path.read_bytes():
        raise RuntimeError("frozen legacy q6 ledger differs from its immutable backup")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen legacy q6 ledger is invalid JSON") from exc
    if not isinstance(ledger, dict):
        raise RuntimeError("frozen legacy q6 ledger root must be an object")

    expected_clips = int(FROZEN_LEGACY_Q6_DEV_ATTESTATION["clips"])
    backend = ledger.get("backend")
    code_revision = backend.get("code_revision") if isinstance(backend, Mapping) else None
    if (
        ledger.get("format_version") != 1
        or ledger.get("split") != "dev"
        or ledger.get("complete") is not True
        or ledger.get("expected_clips") != expected_clips
        or not isinstance(backend, Mapping)
        or backend.get("identity_sha256")
        != FROZEN_LEGACY_Q6_DEV_ATTESTATION["backend_identity_sha256"]
        or code_revision
        != {
            "evaluation_sha256": FROZEN_LEGACY_Q6_DEV_ATTESTATION["evaluation_sha256"],
            "git_revision": FROZEN_LEGACY_Q6_DEV_ATTESTATION["git_revision"],
        }
        or not isinstance(ledger.get("clips"), list)
        or len(ledger["clips"]) != expected_clips
    ):
        raise RuntimeError("frozen legacy q6 ledger header/identity is mismatched")

    package_root = Path(__file__).resolve().parents[2]
    for relative, expected_sha in FROZEN_LEGACY_Q6_SOURCE_SHA256.items():
        if sha256_file(package_root / relative) != expected_sha:
            raise RuntimeError(f"legacy q6 bank dependency changed: {relative}")

    current_packages = current_backend.get("runtime", {}).get("packages", {})
    if not isinstance(current_packages, Mapping) or any(
        not isinstance(current_packages.get(name), Mapping)
        or current_packages[name].get("version") != version
        for name, version in FROZEN_LEGACY_Q6_RUNTIME_VERSIONS.items()
    ):
        raise RuntimeError("legacy q6 bank runtime dependencies are not equivalent")
    old_checkpoints = {
        (
            record.get("filename"),
            record.get("sha256"),
            record.get("size_bytes"),
            record.get("huggingface_revision"),
        )
        for record in backend.get("checkpoints", [])
        if isinstance(record, Mapping)
    }
    current_checkpoints = {
        (
            record.get("filename"),
            record.get("sha256"),
            record.get("size_bytes"),
            record.get("huggingface_revision"),
        )
        for record in current_backend.get("checkpoints", [])
        if isinstance(record, Mapping)
    }
    old_ensemble = backend.get("ensemble_artifact")
    current_ensemble = current_backend.get("ensemble_artifact")
    if (
        old_checkpoints != current_checkpoints
        or not isinstance(old_ensemble, Mapping)
        or not isinstance(current_ensemble, Mapping)
        or {
            "sha256": old_ensemble.get("sha256"),
            "size_bytes": old_ensemble.get("size_bytes"),
        }
        != dict(current_ensemble)
    ):
        raise RuntimeError("legacy q6 artifacts are not byte-equivalent to the v2 bank kernel")

    rows: dict[str, Mapping[str, Any]] = {}
    for raw in ledger["clips"]:
        if not isinstance(raw, Mapping):
            raise RuntimeError("frozen legacy q6 ledger contains a non-object row")
        clip_id = str(raw.get("clip_id"))
        if clip_id in rows:
            raise RuntimeError("frozen legacy q6 ledger contains duplicate clip IDs")
        rows[clip_id] = raw
    expected_ids = {target.clip_id for target in targets}
    if len(targets) != expected_clips or set(rows) != expected_ids:
        raise RuntimeError("frozen legacy q6 ledger does not match the exact 300 WAV targets")
    session = asdict(SessionConfig())
    for target in targets:
        row = rows[target.clip_id]
        if not target.audio_path.is_file() or not target.event_cache_path.is_file():
            raise RuntimeError(f"frozen legacy q6 files are incomplete for {target.clip_id}")
        read_banked_events(target.event_cache_path)
        expected = {
            "audio_sha256": sha256_file(target.audio_path),
            "event_cache_sha256": sha256_file(target.event_cache_path),
            "backend_identity_sha256": FROZEN_LEGACY_Q6_DEV_ATTESTATION["backend_identity_sha256"],
            "session": session,
            "origin": "generated",
        }
        mismatches = [field for field, value in expected.items() if row.get(field) != value]
        if mismatches:
            raise RuntimeError(
                f"frozen legacy q6 row mismatch for {target.clip_id}: " + ", ".join(mismatches)
            )

    for receipt_name in ("player05_control_reproduction", "reproduction"):
        reproduction = ledger.get(receipt_name)
        if (
            not isinstance(reproduction, Mapping)
            or reproduction.get("passed") is not True
            or reproduction.get("clips") != 60
            or reproduction.get("player") != "05"
            or reproduction.get("observed") != FROZEN_LEGACY_Q6_DEV_ATTESTATION["observed"]
            or reproduction.get("identity", {}).get("published")
            != FROZEN_LEGACY_Q6_DEV_ATTESTATION["published"]
            or reproduction.get("identity", {}).get("tolerance")
            != FROZEN_LEGACY_Q6_DEV_ATTESTATION["tolerance"]
        ):
            raise RuntimeError(f"frozen legacy q6 {receipt_name} receipt failed")

    attestation = {
        "verified": True,
        "provenance": "migrated_legacy_q6",
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": expected_ledger_sha,
        "immutable_backup_path": str(backup_path.resolve()),
        "legacy_backend_identity_sha256": backend["identity_sha256"],
        "current_bank_kernel_identity_sha256": current_backend["identity_sha256"],
        "clips": expected_clips,
    }
    return ledger, attestation


def bank_guitarset_events(
    paths: RuntimePaths,
    *,
    split: str,
    allow_transcribe_missing: bool,
) -> dict[str, Any]:
    """Build one q6 split from WAV only, atomically and without scoring gold."""

    assert_frozen_scoring_environment()
    unlock = validate_development_unlock(paths) if split == "sealed" else None
    targets = _guitarset_audio_bank_targets(paths, split=split)
    backend_identity = highres_bank_backend_identity()
    if split == "dev" and _legacy_q6_dev_paths(paths)[0].is_file():
        legacy, attestation = _validate_frozen_legacy_q6_dev(
            paths,
            targets,
            backend_identity,
        )
        return {**legacy, "frozen_legacy_attestation": attestation}
    ledger_path = _event_bank_v2_path(paths, f"guitarset-{split}")
    previous_rows: dict[str, Mapping[str, Any]] = {}
    if ledger_path.is_file():
        try:
            previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if isinstance(previous, Mapping) and isinstance(previous.get("clips"), list):
            previous_rows = {
                str(row.get("clip_id")): row
                for row in previous["clips"]
                if isinstance(row, Mapping)
            }

    backend: Any | None = None
    rows: list[dict[str, Any]] = []
    control_reproduction: dict[str, Any] | None = None
    session = asdict(SessionConfig())
    for target in targets:
        audio_sha = sha256_file(target.audio_path)
        prior = previous_rows.get(target.clip_id)
        if target.event_cache_path.is_file():
            events = read_banked_events(target.event_cache_path)
            event_sha = sha256_file(target.event_cache_path)
            if prior is not None:
                expected = {
                    "audio_sha256": audio_sha,
                    "event_cache_sha256": event_sha,
                    "backend_identity_sha256": backend_identity["identity_sha256"],
                    "session": session,
                }
                if any(prior.get(name) != value for name, value in expected.items()):
                    raise RuntimeError(
                        f"stale q6 bank ledger for {target.clip_id}; "
                        "refusing to adopt mismatched cached events"
                    )
                origin = "resumed"
            else:
                if split == "sealed":
                    raise RuntimeError(
                        f"sealed q6 cache for {target.clip_id} has no matching ledger; "
                        "refusing unledgered adoption"
                    )
                origin = "adopted_existing_pending_reproduction"
        else:
            if not allow_transcribe_missing:
                raise FileNotFoundError(
                    f"missing q6 events for {target.clip_id}; rerun bank-events with "
                    "--allow-transcribe-missing"
                )
            if backend is None:
                backend = _new_highres_bank_backend()
            waveform, sample_rate = load_bank_audio(target.audio_path)
            events = list(backend.transcribe(waveform, sample_rate, SessionConfig()))
            _atomic_text(
                target.event_cache_path,
                json.dumps(_events_to_json(events), indent=1, sort_keys=True) + "\n",
            )
            event_sha = sha256_file(target.event_cache_path)
            origin = "generated"
        rows.append(
            {
                "clip_id": target.clip_id,
                "player": target.player,
                "mode": target.mode,
                "audio_path": str(target.audio_path),
                "audio_sha256": audio_sha,
                "event_cache_path": str(target.event_cache_path),
                "event_cache_sha256": event_sha,
                "events": len(events),
                "backend_identity_sha256": backend_identity["identity_sha256"],
                "session": session,
                "origin": origin,
            }
        )
        ledger = {
            "format_version": EVENT_BANK_LEDGER_VERSION,
            "split": split,
            "complete": len(rows) == len(targets),
            "expected_clips": len(targets),
            "backend": backend_identity,
            "development_unlock": unlock,
            "player05_control_reproduction": control_reproduction,
            "clips": rows,
        }
        _atomic_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        if split == "dev" and len(rows) == EXPECTED_CORPUS_COUNTS["guitarset-sealed"]:
            control_reproduction = validate_q6_player05_control(paths)
            ledger["player05_control_reproduction"] = control_reproduction
            _atomic_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    reproduction = validate_q6_guitarset_cache(paths) if split == "dev" else None
    ledger["reproduction"] = reproduction
    _atomic_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def bank_transfer_events(
    clips: Sequence[ExperimentClip],
    paths: RuntimePaths,
    *,
    allow_transcribe_missing: bool,
) -> dict[str, Any]:
    """Content-address transfer event banks without reading their annotations."""

    if any(clip.corpus.startswith("guitarset") for clip in clips):
        raise ValueError("GuitarSet must use the WAV-only split banker")
    assert_frozen_scoring_environment()
    backend_identity = highres_bank_backend_identity()
    ledgers: dict[str, Any] = {}
    for corpus in sorted({clip.corpus for clip in clips}, key=CORPORA.index):
        expected_clips = EXPECTED_CORPUS_COUNTS[corpus]
        selected = sorted(
            (clip for clip in clips if clip.corpus == corpus),
            key=lambda clip: clip.clip_id,
        )
        if len(selected) != expected_clips:
            raise RuntimeError(
                f"{corpus} bank requires exactly {expected_clips} clips, found {len(selected)}"
            )
        ledger_path = _event_bank_v2_path(paths, corpus)
        previous_rows: dict[str, Mapping[str, Any]] = {}
        if ledger_path.is_file():
            try:
                previous = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if isinstance(previous, Mapping) and isinstance(previous.get("clips"), list):
                previous_rows = {
                    str(row.get("clip_id")): row
                    for row in previous["clips"]
                    if isinstance(row, Mapping)
                }

        backend: Any | None = None
        rows: list[dict[str, Any]] = []
        for clip in selected:
            session_config = session_for_clip(clip)
            session = asdict(session_config)
            audio_sha = sha256_file(clip.audio_path)
            prior = previous_rows.get(clip.clip_id)
            if corpus == "gaps":
                legacy_record = _verify_legacy_gaps_record(clip)
                events = read_banked_events(clip.event_cache_path)
                event_sha = legacy_record["event_cache_sha256"]
                origin = "verified_frozen_legacy_gaps"
            elif clip.event_cache_path.is_file():
                events = read_banked_events(clip.event_cache_path)
                event_sha = sha256_file(clip.event_cache_path)
                if prior is None:
                    raise RuntimeError(
                        f"{corpus} cache for {clip.clip_id} has no matching ledger; "
                        "refusing unledgered adoption"
                    )
                if any(
                    prior.get(name) != value
                    for name, value in {
                        "audio_sha256": audio_sha,
                        "event_cache_sha256": event_sha,
                        "backend_identity_sha256": backend_identity["identity_sha256"],
                        "session": session,
                    }.items()
                ):
                    raise RuntimeError(f"stale transfer bank ledger for {clip.clip_id}")
                origin = "resumed"
            else:
                if not allow_transcribe_missing:
                    raise FileNotFoundError(
                        f"missing banked events for {clip.clip_id}; use "
                        "bank-events --allow-transcribe-missing"
                    )
                if backend is None:
                    backend = _new_highres_bank_backend()
                waveform, sample_rate = load_bank_audio(clip.audio_path)
                events = list(
                    backend.transcribe(
                        waveform,
                        sample_rate,
                        session_config,
                    )
                )
                _atomic_text(
                    clip.event_cache_path,
                    json.dumps(_events_to_json(events), indent=1, sort_keys=True) + "\n",
                )
                event_sha = sha256_file(clip.event_cache_path)
                origin = "generated"
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "audio_path": str(clip.audio_path.resolve()),
                    "audio_sha256": audio_sha,
                    "event_cache_path": str(clip.event_cache_path.resolve()),
                    "event_cache_sha256": event_sha,
                    "backend_identity_sha256": (
                        None if corpus == "gaps" else backend_identity["identity_sha256"]
                    ),
                    "session": session,
                    "events": len(events),
                    "origin": origin,
                }
            )
            ledger = {
                "format_version": EVENT_BANK_LEDGER_VERSION,
                "corpus": corpus,
                "expected_clips": expected_clips,
                "complete": len(rows) == expected_clips,
                "backend": None if corpus == "gaps" else backend_identity,
                "clips": rows,
            }
            _atomic_text(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        ledgers[corpus] = ledger
    return {"format_version": EVENT_BANK_LEDGER_VERSION, "corpora": ledgers}


def session_for_clip(clip: ExperimentClip) -> SessionConfig:
    if clip.corpus.startswith("guitarset"):
        return SessionConfig()
    if clip.corpus == "gaps":
        return SessionConfig(instrument="classical", style="fingerstyle")
    if clip.corpus in {"egset12", "guitar-techs"}:
        if clip.mode in {None, "solo", "single_notes"}:
            return SessionConfig(instrument="electric", tone="clean", style="fingerstyle")
        return SessionConfig(instrument="electric", tone="clean", style="mixed")
    return SessionConfig(style="fingerstyle" if clip.mode == "solo" else "strumming")


def _verify_legacy_gaps_record(clip: ExperimentClip) -> dict[str, Any]:
    """Verify one frozen clean-12 audio/event pair without self-authentication."""

    expected = GAPS_LEGACY_CLEAN12.get(clip.clip_id)
    if expected is None:
        raise RuntimeError(f"unexpected clip outside frozen GAPS clean-12: {clip.clip_id}")
    if not clip.audio_path.is_file() or not clip.event_cache_path.is_file():
        raise RuntimeError(f"frozen legacy GAPS files are incomplete for {clip.clip_id}")
    observed_audio = sha256_file(clip.audio_path)
    observed_events = sha256_file(clip.event_cache_path)
    if observed_audio != expected["audio_sha256"]:
        raise RuntimeError(f"frozen legacy GAPS audio identity mismatch for {clip.clip_id}")
    if observed_events != expected["event_cache_sha256"]:
        raise RuntimeError(f"frozen legacy GAPS event-cache identity mismatch for {clip.clip_id}")
    return {
        "clip_id": clip.clip_id,
        "audio_sha256": observed_audio,
        "event_cache_sha256": observed_events,
        "session": asdict(session_for_clip(clip)),
    }


def validate_event_bank_ledgers(
    clips: Sequence[ExperimentClip],
    paths: RuntimePaths,
    *,
    require_complete_selection: bool = True,
) -> dict[str, Any]:
    """Require exact audio/event/backend/session provenance before inference."""

    grouped = {
        corpus: sorted(
            (clip for clip in clips if clip.corpus == corpus),
            key=lambda clip: clip.clip_id,
        )
        for corpus in sorted({clip.corpus for clip in clips}, key=CORPORA.index)
    }
    backend_identity: dict[str, Any] | None = None
    evidence: dict[str, Any] = {}
    for corpus, selected in grouped.items():
        expected_count = EXPECTED_CORPUS_COUNTS[corpus]
        if (require_complete_selection and len(selected) != expected_count) or not 0 < len(
            selected
        ) <= expected_count:
            raise RuntimeError(
                f"{corpus} event-bank validation requires {expected_count} clips, "
                f"found {len(selected)}"
            )
        if corpus == "gaps":
            observed_ids = {clip.clip_id for clip in selected}
            expected_ids = set(GAPS_LEGACY_CLEAN12)
            if observed_ids != expected_ids:
                raise RuntimeError(
                    "GAPS legacy exception requires the exact frozen clean-12 clip set"
                )
            legacy_records = [_verify_legacy_gaps_record(clip) for clip in selected]
            evidence[corpus] = {
                "verified": True,
                "origin": "hash_pinned_legacy_gaps_exception",
                "clips": len(legacy_records),
                "identity_sha256": _sha256_bytes(
                    _canonical_json(
                        {"expected": GAPS_LEGACY_CLEAN12, "clips": legacy_records}
                    ).encode("utf-8")
                ),
                "records": legacy_records,
            }
            continue

        if backend_identity is None:
            backend_identity = highres_bank_backend_identity()
        legacy_attestation = None
        if corpus == "guitarset-dev" and _legacy_q6_dev_paths(paths)[0].is_file():
            full_targets = _guitarset_audio_bank_targets(paths, split="dev")
            ledger, legacy_attestation = _validate_frozen_legacy_q6_dev(
                paths,
                full_targets,
                backend_identity,
            )
            ledger_path = _legacy_q6_dev_paths(paths)[0]
            ledger_backend_identity = str(
                FROZEN_LEGACY_Q6_DEV_ATTESTATION["backend_identity_sha256"]
            )
        else:
            ledger_path = _event_bank_v2_path(paths, corpus)
            try:
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"{corpus} requires a valid event-bank ledger") from exc
            ledger_backend_identity = str(backend_identity["identity_sha256"])
        if not isinstance(ledger, Mapping):
            raise RuntimeError(f"{corpus} event-bank ledger root must be an object")
        if (
            ledger.get("complete") is not True
            or ledger.get("expected_clips") != expected_count
            or not isinstance(ledger.get("backend"), Mapping)
            or ledger["backend"].get("identity_sha256") != ledger_backend_identity
            or not isinstance(ledger.get("clips"), list)
        ):
            raise RuntimeError(f"{corpus} event-bank ledger is incomplete or stale")
        rows = {str(row.get("clip_id")): row for row in ledger["clips"] if isinstance(row, Mapping)}
        expected_ids = {clip.clip_id for clip in selected}
        clip_set_matches = (
            set(rows) == expected_ids if require_complete_selection else expected_ids.issubset(rows)
        )
        if not clip_set_matches or len(rows) != expected_count:
            raise RuntimeError(f"{corpus} event-bank ledger clip set is mismatched")
        for clip in selected:
            row = rows[clip.clip_id]
            expected = {
                "audio_sha256": sha256_file(clip.audio_path),
                "event_cache_sha256": sha256_file(clip.event_cache_path),
                "backend_identity_sha256": ledger_backend_identity,
                "session": asdict(session_for_clip(clip)),
            }
            mismatches = [field for field, value in expected.items() if row.get(field) != value]
            if mismatches:
                raise RuntimeError(
                    f"{corpus} event-bank ledger mismatch for {clip.clip_id}: "
                    + ", ".join(mismatches)
                )
        if corpus == "guitarset-dev":
            reproduction = ledger.get("reproduction")
            control = ledger.get("player05_control_reproduction")
            if (
                not isinstance(reproduction, Mapping)
                or reproduction.get("passed") is not True
                or not isinstance(control, Mapping)
                or control.get("passed") is not True
            ):
                raise RuntimeError("development q6 ledger lacks passing reproduction receipts")
        evidence[corpus] = {
            "verified": True,
            "ledger_path": str(ledger_path.resolve()),
            "ledger_sha256": sha256_file(ledger_path),
            "backend_identity_sha256": ledger_backend_identity,
            "current_bank_kernel_identity_sha256": backend_identity["identity_sha256"],
            "frozen_legacy_attestation": legacy_attestation,
            "clips": expected_count,
        }
    return {"verified": True, "corpora": evidence}


def parse_gold(clip: ExperimentClip, cfg: GuitarConfig) -> list[TabEvent]:
    if clip.annotation_format == "guitarset_jams":
        return parse_guitarset_jams(clip.annotation_path, cfg)
    if clip.annotation_format == "gaps_musicxml_tab":
        return list(get_parser("gaps_musicxml_tab")(clip.annotation_path, cfg))
    if clip.annotation_format == "egset12_jams":
        return parse_egset12_jams(clip.annotation_path, cfg)
    if clip.annotation_format == "guitar_techs_jams":
        return parse_guitar_techs_jams(clip.annotation_path, cfg)
    raise ValueError(f"unsupported annotation format {clip.annotation_format!r}")


def _q6_guitarset_path(track_id: str, paths: RuntimePaths) -> Path:
    root = paths.q6_player05_cache if track_id[:2] == BURNED_PLAYER else paths.q6_dev_cache
    return root / f"{track_id}.ensemble.json"


def _complete_legacy_cache(
    guitarset_entries: Sequence[ClipEntry],
    legacy_root: Path,
) -> dict[str, Path] | None:
    if len(guitarset_entries) != EXPECTED_GUITARSET_CLIPS:
        return None
    result: dict[str, Path] = {}
    for entry in guitarset_entries:
        track_id = entry.id.split("/", 1)[1]
        matches = sorted(legacy_root.glob(f"{track_id}_mic.*.json"))
        if len(matches) != 1:
            return None
        read_banked_events(matches[0])
        result[track_id] = matches[0]
    return result if len(result) == EXPECTED_GUITARSET_CLIPS else None


def _guitarset_bank_identity(
    paths: RuntimePaths,
    entries: Sequence[ClipEntry],
    bank: Mapping[str, Path],
    *,
    bank_name: str,
    annotation_players: set[str] | None = None,
) -> dict[str, Any]:
    """Hash every banked/scoring input plus exact code, runtime, and constants."""

    frozen = assert_frozen_scoring_environment()
    annotation_records = [
        {
            "track_id": entry.id.split("/", 1)[1],
            "sha256": sha256_file(entry.annotation_path),
            "size_bytes": Path(entry.annotation_path).stat().st_size,
        }
        for entry in sorted(entries, key=lambda item: item.id)
        if annotation_players is None or entry.id.split("/", 1)[1][:2] in annotation_players
    ]
    bank_records: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: item.id):
        track_id = entry.id.split("/", 1)[1]
        cache_path = bank.get(track_id)
        if cache_path is None or not cache_path.is_file():
            raise RuntimeError(f"{bank_name} bank is missing {track_id}")
        read_banked_events(cache_path)
        player = track_id[:2]
        annotation_in_scope = annotation_players is None or player in annotation_players
        bank_records.append(
            {
                "track_id": track_id,
                "audio_sha256": sha256_file(entry.media_path),
                "audio_size_bytes": Path(entry.media_path).stat().st_size,
                "annotation_sha256": (
                    sha256_file(entry.annotation_path) if annotation_in_scope else None
                ),
                "annotation_size_bytes": (
                    Path(entry.annotation_path).stat().st_size if annotation_in_scope else None
                ),
                "event_cache_sha256": sha256_file(cache_path),
                "event_cache_size_bytes": cache_path.stat().st_size,
            }
        )
    content = {
        "annotations": annotation_records,
        "bank": bank_records,
    }
    return {
        "bank_name": bank_name,
        "bank_clips": len(bank_records),
        "content_sha256": _sha256_bytes(_canonical_json(content).encode("utf-8")),
        "code_revision": evaluation_code_revision(),
        "runtime": runtime_manifest(),
        "frozen_scoring": frozen,
        "guitar_config": asdict(GuitarConfig()),
        "published": REPRODUCTION,
        "tolerance": REPRODUCTION_TOLERANCE,
    }


def _validate_guitarset_bank(
    paths: RuntimePaths,
    entries: Sequence[ClipEntry],
    bank: Mapping[str, Path],
    *,
    bank_name: str,
    annotation_players: set[str] | None = None,
) -> dict[str, Any]:
    """Accept a bank only after the frozen player-05 reproduction."""

    player05_ids = sorted(
        entry.id.split("/", 1)[1]
        for entry in entries
        if entry.id.split("/", 1)[1][:2] == BURNED_PLAYER
    )
    if len(player05_ids) != EXPECTED_CORPUS_COUNTS["guitarset-sealed"]:
        raise RuntimeError(
            f"{bank_name} must include all 60 player-05 development clips for reproduction"
        )
    identity = _guitarset_bank_identity(
        paths,
        entries,
        bank,
        bank_name=bank_name,
        annotation_players=annotation_players,
    )
    marker_path = paths.cache_root / f"{_safe_id(bank_name)}-guitarset-reproduction.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = {}
        if marker.get("identity") == identity and marker.get("passed") is True:
            return marker

    cfg = GuitarConfig()
    player05_gold = {
        entry.id.split("/", 1)[1]: parse_guitarset_jams(entry.annotation_path, cfg)
        for entry in entries
        if entry.id.split("/", 1)[1][:2] == BURNED_PLAYER
    }
    position = load_pitch_position_prior("guitarset-v1", cfg=cfg)
    sequence = load_transition_prior("guitarset-seq-v1")
    baseline: list[float] = []
    shipped: list[float] = []
    for track_id in player05_ids:
        events = read_banked_events(bank[track_id])
        clip_gold = player05_gold[track_id]
        session = SessionConfig()
        prepared = apply_pitch_position_prior(events, position, cfg)
        base = _decode_with_transition(
            prepared,
            clip_gold,
            cfg=cfg,
            session=session,
            transition=sequence,
        )
        waveform, sample_rate = load_mono_audio(
            paths.guitarset_root / "audio_mono-mic" / f"{track_id}_mic.wav"
        )
        current_events, _tally = _prepare_guitarset_current(
            events,
            waveform,
            sample_rate,
            cfg,
        )
        prepared_current = apply_pitch_position_prior(
            current_events,
            position,
            cfg,
        )
        current = _decode_with_transition(
            prepared_current,
            clip_gold,
            cfg=cfg,
            session=session,
            transition=sequence,
        )
        baseline.append(base.tab.f1)
        shipped.append(current.tab.f1)

    observed = {
        "baseline": float(np.mean(baseline)),
        "shipped": float(np.mean(shipped)),
    }
    drift = {name: observed[name] - REPRODUCTION[name] for name in REPRODUCTION}
    failed = [name for name, value in drift.items() if abs(value) > REPRODUCTION_TOLERANCE]
    marker = {
        "identity": identity,
        "player": BURNED_PLAYER,
        "clips": len(baseline),
        "observed": observed,
        "drift": drift,
        "passed": not failed,
    }
    _atomic_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    if failed:
        raise RuntimeError(
            f"{bank_name} GuitarSet reproduction failed for {', '.join(failed)}; "
            f"observed={observed}, published={REPRODUCTION}"
        )
    return marker


def validate_legacy_guitarset_cache(
    paths: RuntimePaths,
    corpora: Sequence[str] = ("guitarset-dev", "guitarset-sealed"),
) -> dict[str, Any]:
    all_entries = scan_guitarset(paths.guitarset_root, validation_player=SEALED_PLAYER)
    legacy = _complete_legacy_cache(all_entries, paths.legacy_guitarset_cache)
    if legacy is None:
        raise RuntimeError("legacy GuitarSet cache is not a complete 360-track bank")
    selected = set(corpora)
    entries = [
        entry
        for entry in all_entries
        if (
            "guitarset-sealed"
            if entry.id.split("/", 1)[1][:2] == SEALED_PLAYER
            else "guitarset-dev"
        )
        in selected
    ]
    bank_name = "legacy-a3-" + "-".join(
        sorted(corpus.removeprefix("guitarset-") for corpus in selected)
    )
    return _validate_guitarset_bank(paths, entries, legacy, bank_name=bank_name)


def validate_q6_player05_control(paths: RuntimePaths) -> dict[str, Any]:
    """Fail fast on the frozen 60-clip player-05 control before the dev bank."""

    targets = [
        target
        for target in _guitarset_audio_bank_targets(paths, split="dev")
        if target.player == BURNED_PLAYER
    ]
    if len(targets) != EXPECTED_CORPUS_COUNTS["guitarset-sealed"]:
        raise RuntimeError("q6 control requires exactly 60 player-05 clips")
    bank = {target.track_id: target.event_cache_path for target in targets}
    entries = [
        ClipEntry(
            id=target.clip_id,
            tier=(
                "clean_acoustic_single_line" if target.mode == "solo" else "clean_acoustic_strummed"
            ),
            source="GuitarSet",
            split="dev-control",
            media_path=str(target.audio_path),
            annotation_path=str(paths.guitarset_root / "annotation" / f"{target.track_id}.jams"),
            annotation_format="guitarset_jams",
        )
        for target in targets
    ]
    return _validate_guitarset_bank(
        paths,
        entries,
        bank,
        bank_name="q6-player05-control",
        annotation_players={BURNED_PLAYER},
    )


def validate_q6_guitarset_cache(paths: RuntimePaths) -> dict[str, Any]:
    validate_q6_player05_control(paths)
    targets = _guitarset_audio_bank_targets(paths, split="dev")
    q6 = {target.track_id: target.event_cache_path for target in targets}
    if not all(path.is_file() for path in q6.values()):
        missing = [track_id for track_id, path in q6.items() if not path.is_file()]
        raise RuntimeError(
            f"q6 development bank requires exactly 300 clips; missing {len(missing)}"
        )
    entries = [
        ClipEntry(
            id=target.clip_id,
            tier="clean_acoustic_single_line"
            if target.mode == "solo"
            else "clean_acoustic_strummed",
            source="GuitarSet",
            split="dev",
            media_path=str(target.audio_path),
            annotation_path=(
                str(paths.guitarset_root / "annotation" / f"{target.track_id}.jams")
                if target.player == BURNED_PLAYER
                else ""
            ),
            annotation_format="guitarset_jams",
        )
        for target in targets
    ]
    return _validate_guitarset_bank(
        paths,
        entries,
        q6,
        bank_name="q6-dev",
        annotation_players={BURNED_PLAYER},
    )


def _guitarset_entries(
    paths: RuntimePaths,
    corpora: Sequence[str] = ("guitarset-dev", "guitarset-sealed"),
    *,
    force_q6: bool = False,
) -> tuple[list[ClipEntry], dict[str, Path], str]:
    selected = set(corpora)
    targets = [
        target
        for split in ("dev", "sealed")
        if f"guitarset-{split}" in selected
        for target in _guitarset_audio_bank_targets(paths, split=split)
    ]
    entries: list[ClipEntry] = []
    q6: dict[str, Path] = {}
    for target in targets:
        annotation_path = paths.guitarset_root / "annotation" / f"{target.track_id}.jams"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"missing GuitarSet annotation: {annotation_path}")
        entries.append(
            ClipEntry(
                id=target.clip_id,
                tier=(
                    "clean_acoustic_single_line"
                    if target.mode == "solo"
                    else "clean_acoustic_strummed"
                ),
                source="GuitarSet",
                split="sealed" if target.player == SEALED_PLAYER else "dev",
                media_path=str(target.audio_path),
                annotation_path=str(annotation_path),
                annotation_format="guitarset_jams",
            )
        )
        q6[target.track_id] = target.event_cache_path
    expected = sum(EXPECTED_CORPUS_COUNTS[corpus] for corpus in selected)
    if len(entries) != expected:
        raise RuntimeError(f"expected {expected} selected GuitarSet clips, found {len(entries)}")
    complete = all(path.is_file() for path in q6.values())
    if not force_q6 and not complete:
        missing = [track_id for track_id, path in q6.items() if not path.is_file()]
        raise FileNotFoundError(
            f"{len(missing)} selected q6 GuitarSet caches are missing; run bank-events "
            f"--allow-transcribe-missing first (first: {missing[0]})"
        )
    return entries, q6, "q6" if complete else "q6-pending-bank"


def discover_clips(
    paths: RuntimePaths,
    corpora: Sequence[str],
    *,
    limit: int = 0,
    force_q6: bool = False,
    corpus_status: dict[str, dict[str, Any]] | None = None,
) -> list[ExperimentClip]:
    """Build a deterministic, local-only corpus manifest."""

    selected = set(corpora)
    unknown = selected - set(CORPORA)
    if unknown:
        raise ValueError(f"unknown corpora: {sorted(unknown)}")
    clips: list[ExperimentClip] = []
    if selected & {"guitarset-dev", "guitarset-sealed"}:
        entries, cache_paths, strategy = _guitarset_entries(
            paths, sorted(selected & {"guitarset-dev", "guitarset-sealed"}), force_q6=force_q6
        )
        for entry in entries:
            track_id = entry.id.split("/", 1)[1]
            player = track_id[:2]
            corpus = "guitarset-sealed" if player == SEALED_PLAYER else "guitarset-dev"
            if corpus not in selected:
                continue
            clips.append(
                ExperimentClip(
                    clip_id=entry.id,
                    corpus=corpus,
                    source=entry.source,
                    split="sealed" if corpus == "guitarset-sealed" else "dev",
                    tier="solo" if track_id.endswith("_solo") else "comp",
                    player=player,
                    mode="solo" if track_id.endswith("_solo") else "comp",
                    audio_path=Path(entry.media_path),
                    annotation_path=Path(entry.annotation_path),
                    annotation_format=entry.annotation_format,
                    event_cache_path=cache_paths[track_id],
                    event_cache_strategy=strategy,
                )
            )
    if "gaps" in selected:
        for track_id in CLEAN_12:
            audio = paths.gaps_root / "audio" / f"{track_id}.wav"
            annotation = paths.gaps_root / "musicxml" / f"{track_id}.xml"
            if not audio.is_file() or not annotation.is_file():
                raise FileNotFoundError(f"incomplete GAPS clean-12 clip {track_id}")
            clips.append(
                ExperimentClip(
                    clip_id=f"gaps/{track_id}",
                    corpus="gaps",
                    source="GAPS",
                    split="test",
                    tier="solo",
                    player=None,
                    mode="solo",
                    audio_path=audio.resolve(),
                    annotation_path=annotation.resolve(),
                    annotation_format="gaps_musicxml_tab",
                    event_cache_path=paths.q6_gaps_cache / f"{track_id}.ensemble.json",
                    event_cache_strategy="q6-gaps",
                )
            )
    if "egset12" in selected:
        egset_entries = scan_egset12(paths.egset12_root)
        if len(egset_entries) != EXPECTED_EGSET12_CLIPS:
            if corpus_status is not None:
                corpus_status["egset12"] = {
                    "status": "blocked_unscored",
                    "expected_clips": EXPECTED_EGSET12_CLIPS,
                    "discovered_clips": len(egset_entries),
                    "reason": "official digest-verified EGSet12 WAV/JAMS pairs are incomplete",
                }
            egset_entries = []
        for egset_entry in egset_entries:
            clips.append(
                ExperimentClip(
                    clip_id=egset_entry.id,
                    corpus="egset12",
                    source=egset_entry.source,
                    split="test",
                    tier="clean_electric",
                    player=None,
                    mode=None,
                    audio_path=Path(egset_entry.media_path),
                    annotation_path=Path(egset_entry.annotation_path),
                    annotation_format=egset_entry.annotation_format,
                    event_cache_path=paths.cache_root
                    / "raw-events"
                    / "egset12"
                    / f"{_safe_id(egset_entry.id)}.ensemble.json",
                    event_cache_strategy="experiment-bank",
                )
            )
    if "guitar-techs" in selected:
        root = paths.guitar_techs_root
        if (root / "clips" / "guitar-techs").is_dir():
            root = root / "clips" / "guitar-techs"
        guitar_techs_entries = scan_guitar_techs(root)
        if len(guitar_techs_entries) != EXPECTED_GUITAR_TECHS_CLIPS:
            raise RuntimeError(
                f"expected {EXPECTED_GUITAR_TECHS_CLIPS} Guitar-TECHS clips, "
                f"found {len(guitar_techs_entries)}"
            )
        for guitar_techs_entry in guitar_techs_entries:
            clips.append(
                ExperimentClip(
                    clip_id=guitar_techs_entry.clip_id,
                    corpus="guitar-techs",
                    source=guitar_techs_entry.source,
                    split="test",
                    tier="clean_electric",
                    player=guitar_techs_entry.player_id,
                    mode=guitar_techs_entry.content_type,
                    audio_path=guitar_techs_entry.wav_path,
                    annotation_path=guitar_techs_entry.jams_path,
                    annotation_format="guitar_techs_jams",
                    event_cache_path=paths.cache_root
                    / "raw-events"
                    / "guitar-techs"
                    / f"{_safe_id(guitar_techs_entry.clip_id)}.ensemble.json",
                    event_cache_strategy="experiment-bank",
                )
            )
    clips.sort(key=lambda clip: (CORPORA.index(clip.corpus), clip.clip_id))
    if limit > 0:
        limited: list[ExperimentClip] = []
        for corpus in CORPORA:
            limited.extend([clip for clip in clips if clip.corpus == corpus][:limit])
        clips = limited
    if corpus_status is not None:
        counts = Counter(clip.corpus for clip in clips)
        for corpus in sorted(selected, key=CORPORA.index):
            record = corpus_status.setdefault(corpus, {})
            expected_count = EXPECTED_CORPUS_COUNTS[corpus]
            discovered_count = counts[corpus]
            record.update({"expected_clips": expected_count, "discovered_clips": discovered_count})
            if discovered_count == expected_count:
                record["status"] = "ready"
                record.pop("reason", None)
            elif record.get("status") != "blocked_unscored":
                record["status"] = "debug_limited" if limit > 0 else "incomplete"
                record["reason"] = (
                    "debug --limit selection cannot satisfy evidence gates"
                    if limit > 0
                    else "canonical corpus count is incomplete"
                )
    return clips


def _development_clips_for_unlock(paths: RuntimePaths) -> list[ExperimentClip]:
    """Rebuild dev identity without discovering or constructing sealed labels."""

    targets = _guitarset_audio_bank_targets(paths, split="dev")
    q6 = {target.track_id: target.event_cache_path for target in targets}
    missing = [track_id for track_id, path in q6.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"development unlock requires the complete q6 event bank; missing {len(missing)} clips"
        )
    cache_paths = q6
    strategy = "q6"

    clips: list[ExperimentClip] = []
    for target in targets:
        annotation = paths.guitarset_root / "annotation" / f"{target.track_id}.jams"
        if not annotation.is_file():
            raise RuntimeError(f"missing development annotation {annotation}")
        clips.append(
            ExperimentClip(
                clip_id=target.clip_id,
                corpus="guitarset-dev",
                source="GuitarSet",
                split="dev",
                tier=target.mode,
                player=target.player,
                mode=target.mode,
                audio_path=target.audio_path,
                annotation_path=annotation,
                annotation_format="guitarset_jams",
                event_cache_path=cache_paths[target.track_id],
                event_cache_strategy=strategy,
            )
        )
    return sorted(clips, key=lambda clip: clip.clip_id)


def model_specs(
    model_root: Path,
    selected: Sequence[str],
    *,
    synthtab_checkpoint: Path | None = None,
    dafx_checkpoint: Path | None = None,
) -> list[ModelSpec]:
    specs = {
        "synthtab": ModelSpec(
            name="synthtab",
            checkpoint=synthtab_checkpoint or SYNTHTAB_X4.path_below(model_root),
            expected_sha256=SYNTHTAB_SHA256,
            family="SynthTab TabCNN x4",
            guitarset_overlap=False,
            frontend_normalization="synthtab",
            artifact_id=SYNTHTAB_X4.artifact_id,
            source_revision=SYNTHTAB_X4.source_revision,
            license_id=SYNTHTAB_X4.license_id,
            license_posture=SYNTHTAB_X4.license_posture,
            evaluation_allowed=True,
            shipping_redistribution_allowed=False,
        ),
        "dafx": ModelSpec(
            name="dafx",
            checkpoint=dafx_checkpoint or DAFX_GUITARPROFX_ONNX.path_below(model_root),
            expected_sha256=DAFX_SHA256,
            family="DAFx GuitarProFX TabCNN",
            guitarset_overlap=True,
            frontend_normalization="guitarprofx",
            artifact_id=DAFX_GUITARPROFX_ONNX.artifact_id,
            source_revision=DAFX_GUITARPROFX_ONNX.source_revision,
            license_id=DAFX_GUITARPROFX_ONNX.license_id,
            license_posture=DAFX_GUITARPROFX_ONNX.license_posture,
            evaluation_allowed=True,
            shipping_redistribution_allowed=False,
        ),
    }
    unknown = set(selected) - set(specs)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    result = [specs[name] for name in selected]
    for spec in result:
        validate_checkpoint(spec.checkpoint, expected_sha256=spec.expected_sha256)
    return result


def load_model_backend(spec: ModelSpec) -> PosteriorBackend:
    if spec.name == "synthtab":
        return SynthTabX4Posterior.from_checkpoint(
            spec.checkpoint,
            expected_sha256=spec.expected_sha256,
        )
    if spec.name == "dafx":
        return DAFxTabCNNPosterior.from_checkpoint(
            spec.checkpoint,
            expected_sha256=spec.expected_sha256,
        )
    raise ValueError(f"unsupported model {spec.name!r}")


def guitar_techs_provenance(
    paths: RuntimePaths,
    clips: Sequence[ExperimentClip],
) -> dict[str, Any]:
    selected = [clip for clip in clips if clip.corpus == "guitar-techs"]
    if not selected:
        return {"status": "not_selected", "verified": False}
    metadata_records: list[dict[str, Any]] = []
    revisions: list[str] = []
    for relative in GUITAR_TECHS_METADATA_PATHS:
        path = paths.guitar_techs_root / relative
        if not path.is_file():
            metadata_records.append({"path": str(path), "present": False})
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        revision = lines[0].strip() if lines else ""
        revisions.append(revision)
        metadata_records.append(
            {
                "path": str(path),
                "present": True,
                "sha256": sha256_file(path),
                "revision": revision,
            }
        )
    stem_manifest = paths.guitar_techs_root / "clips" / "manifest.json"
    stem_manifest_record = {
        "path": str(stem_manifest),
        "present": stem_manifest.is_file(),
        "sha256": sha256_file(stem_manifest) if stem_manifest.is_file() else None,
    }
    stem_records = [
        {
            "clip_id": clip.clip_id,
            "audio_sha256": sha256_file(clip.audio_path),
            "annotation_sha256": sha256_file(clip.annotation_path),
        }
        for clip in sorted(selected, key=lambda item: item.clip_id)
    ]
    verified = (
        len(selected) == EXPECTED_GUITAR_TECHS_CLIPS
        and len(revisions) == len(GUITAR_TECHS_METADATA_PATHS)
        and all(revision == PINNED_GUITAR_TECHS_REVISION for revision in revisions)
        and stem_manifest.is_file()
    )
    return {
        "status": "verified" if verified else "blocked_unverified",
        "verified": verified,
        "expected_revision": PINNED_GUITAR_TECHS_REVISION,
        "metadata": metadata_records,
        "stem_manifest": stem_manifest_record,
        "scored_stems_sha256": _sha256_bytes(
            _canonical_json({"clips": stem_records}).encode("utf-8")
        ),
        "clips": len(selected),
    }


def license_evidence(
    spec: ModelSpec,
    paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    """Verify the exact executable, CQT, runtime, and license-audit contract."""

    registered = {
        "synthtab": SYNTHTAB_X4,
        "dafx": DAFX_GUITARPROFX_ONNX,
    }.get(spec.name)
    licenses_path = Path(__file__).resolve().parents[3] / "LICENSES.md"
    licenses_sha = sha256_file(licenses_path) if licenses_path.is_file() else None
    checkpoint_sha = sha256_file(spec.checkpoint) if spec.checkpoint.is_file() else None
    checkpoint_size = spec.checkpoint.stat().st_size if spec.checkpoint.is_file() else None
    cqt_path = SHARED_CQT.path_below(paths.model_root) if paths is not None else None
    cqt_sha = sha256_file(cqt_path) if cqt_path is not None and cqt_path.is_file() else None
    cqt_size = cqt_path.stat().st_size if cqt_path is not None and cqt_path.is_file() else None
    runtime = runtime_manifest()
    packages = runtime["packages"]
    checks = {
        "registered_family": registered is not None,
        "artifact_id": registered is not None and spec.artifact_id == registered.artifact_id,
        "expected_sha256": registered is not None and spec.expected_sha256 == registered.sha256,
        "observed_sha256": registered is not None and checkpoint_sha == registered.sha256,
        "observed_size_bytes": registered is not None and checkpoint_size == registered.size_bytes,
        "source_revision": (
            registered is not None and spec.source_revision == registered.source_revision
        ),
        "license_id": registered is not None and spec.license_id == registered.license_id,
        "license_posture": (
            registered is not None and spec.license_posture == registered.license_posture
        ),
        "evaluation_allowed": spec.evaluation_allowed is True,
        "shipping_redistribution_forbidden": (spec.shipping_redistribution_allowed is False),
        "reference_cqt_sha256": cqt_sha == SHARED_CQT.sha256,
        "reference_cqt_size_bytes": cqt_size == SHARED_CQT.size_bytes,
        "licenses_sha256": licenses_sha == EXPECTED_LICENSES_SHA256,
        "torch_2_11_0_eval_only": packages["torch"].split("+", 1)[0] == "2.11.0",
        "librosa_0_11_0_eval_only": packages["librosa"] == "0.11.0",
        "onnxruntime_1_23_2_eval_only": packages["onnxruntime"] == "1.23.2",
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "artifact": {
            "artifact_id": spec.artifact_id,
            "expected_sha256": spec.expected_sha256,
            "observed_sha256": checkpoint_sha,
            "observed_size_bytes": checkpoint_size,
            "source_revision": spec.source_revision,
            "license_id": spec.license_id,
            "license_posture": spec.license_posture,
            "evaluation_allowed": spec.evaluation_allowed,
            "shipping_redistribution_allowed": spec.shipping_redistribution_allowed,
        },
        "reference_cqt": {
            "path": str(cqt_path) if cqt_path is not None else None,
            "expected_sha256": SHARED_CQT.sha256,
            "observed_sha256": cqt_sha,
            "expected_size_bytes": SHARED_CQT.size_bytes,
            "observed_size_bytes": cqt_size,
            "used_as_runtime_input": False,
        },
        "official_equivalence_verified": (
            registered.official_equivalence_verified if registered is not None else None
        ),
        "licenses_path": str(licenses_path),
        "licenses_expected_sha256": EXPECTED_LICENSES_SHA256,
        "licenses_sha256": licenses_sha,
        "runtime_contract": {
            "observed": runtime,
            "required": {
                "torch_base": "2.11.0",
                "librosa": "0.11.0",
                "onnxruntime": "1.23.2",
            },
        },
    }


def _identity_value(value: Any) -> Any:
    """Convert priors into a stable, compact JSON identity payload."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes()),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": type(value).__qualname__,
            "fields": {
                field.name: _identity_value(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": [
                [_identity_value(key), _identity_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported identity value {type(value).__qualname__}")


def build_frozen_guitarset_lopo(
    paths: RuntimePaths,
    cfg: GuitarConfig,
) -> tuple[dict[str, dict[str, list[TabEvent]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Sole authorized reader of sealed annotations: deterministic LOPO training."""

    annotation_paths = sorted((paths.guitarset_root / "annotation").glob("*.jams"))
    counts = Counter(path.stem[:2] for path in annotation_paths)
    if len(annotation_paths) != EXPECTED_GUITARSET_CLIPS or counts != Counter(
        {player: 60 for player in GUITARSET_PLAYERS}
    ):
        raise RuntimeError(
            "frozen GuitarSet LOPO requires exactly 60 annotations for each of six players"
        )
    annotations = [
        {
            "track_id": path.stem,
            "player": path.stem[:2],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in annotation_paths
    ]
    gold = gold_by_player(paths.guitarset_root, cfg)
    parsed_counts = {player: len(gold.get(player, {})) for player in GUITARSET_PLAYERS}
    if parsed_counts != {player: 60 for player in GUITARSET_PLAYERS}:
        raise RuntimeError(f"incomplete GuitarSet LOPO parses: {parsed_counts}")
    positions, sequences = build_loo_priors(gold, cfg)
    position_payload = _identity_value(positions)
    sequence_payload = _identity_value(sequences)
    identity = {
        "sealed_annotation_use": "LOPO_training_only",
        "players": list(GUITARSET_PLAYERS),
        "annotations": EXPECTED_GUITARSET_CLIPS,
        "annotations_sha256": _sha256_bytes(
            _canonical_json({"annotations": annotations}).encode("utf-8")
        ),
        "sealed_annotations": [
            record for record in annotations if record["player"] == SEALED_PLAYER
        ],
        "position_priors_sha256": _sha256_bytes(
            _canonical_json({"positions": position_payload}).encode("utf-8")
        ),
        "sequence_priors_sha256": _sha256_bytes(
            _canonical_json({"sequences": sequence_payload}).encode("utf-8")
        ),
        "position_prior_parameters": {"alpha": POSITION_ALPHA, "power": POSITION_POWER},
        "sequence_prior_parameters": {
            "scheme": SEQUENCE_SCHEME,
            "alpha": SEQUENCE_ALPHA,
            "backoff_kappa": SEQUENCE_BACKOFF_KAPPA,
            "singleton_only": SEQUENCE_SINGLETON_ONLY,
            "decode_weight": SEQUENCE_PRIOR_WEIGHT,
        },
    }
    return gold, positions, sequences, identity


def _clip_manifest_record(clip: ExperimentClip) -> dict[str, Any]:
    event_present = clip.event_cache_path.is_file()
    return {
        **asdict(clip),
        "audio_path": str(clip.audio_path.resolve()),
        "annotation_path": str(clip.annotation_path.resolve()),
        "event_cache_path": str(clip.event_cache_path.resolve()),
        "audio_size_bytes": clip.audio_path.stat().st_size,
        "audio_sha256": sha256_file(clip.audio_path),
        "annotation_size_bytes": clip.annotation_path.stat().st_size,
        "annotation_sha256": sha256_file(clip.annotation_path),
        "event_cache_present": event_present,
        "event_cache_size_bytes": clip.event_cache_path.stat().st_size if event_present else None,
        "event_cache_sha256": sha256_file(clip.event_cache_path) if event_present else None,
    }


def build_manifest(
    clips: Sequence[ExperimentClip],
    specs: Sequence[ModelSpec],
    paths: RuntimePaths,
    *,
    corpus_status: Mapping[str, Mapping[str, Any]] | None = None,
    debug_limit: int = 0,
) -> dict[str, Any]:
    lopo_identity = None
    if any(clip.corpus.startswith("guitarset") for clip in clips):
        _, _, _, lopo_identity = build_frozen_guitarset_lopo(
            paths,
            GuitarConfig(),
        )
    revision = evaluation_code_revision()
    cqt_bin = SHARED_CQT.path_below(paths.model_root)
    cqt_record: dict[str, Any] = {
        "path": str(cqt_bin),
        "expected_sha256": CQT_BIN_SHA256,
        "used_as_runtime_input": False,
        "note": (
            "pinned reference/validation artifact only; not loaded by this runner, "
            "which executes librosa.cqt"
        ),
    }
    if cqt_bin.is_file():
        cqt_record["sha256"] = sha256_file(cqt_bin)
        cqt_record["size_bytes"] = cqt_bin.stat().st_size
        cqt_record["verified"] = (
            cqt_record["sha256"] == CQT_BIN_SHA256
            and cqt_record["size_bytes"] == SHARED_CQT.size_bytes
        )
    else:
        cqt_record["verified"] = False
    return {
        "format_version": RESULT_FORMAT_VERSION,
        "protocol": str(PROTOCOL_RELATIVE_PATH),
        "protocol_identity": protocol_identity(),
        "evaluation_only": True,
        "debug_limit": debug_limit,
        "corpus_status": dict(corpus_status or {}),
        "runtime": runtime_manifest(),
        "frozen_scoring": assert_frozen_scoring_environment(),
        "frontend": frontend_manifest(),
        "code_revision": revision,
        "posterior_code_revision": posterior_generation_revision(),
        "fusion": {
            "posterior_only": asdict(POSTERIOR_ONLY_POLICY),
            "current_plus_tabcnn": asdict(CONSERVATIVE_BLEND_POLICY),
        },
        "frozen_artifacts": artifact_manifest(models_root=paths.model_root),
        "models": [
            {
                **asdict(spec),
                "checkpoint": str(spec.checkpoint.resolve()),
                "observed_sha256": sha256_file(spec.checkpoint),
                "size_bytes": spec.checkpoint.stat().st_size,
                "license_evidence": license_evidence(spec, paths),
            }
            for spec in specs
        ],
        "reference_cqt_filterbank": cqt_record,
        "routing": ROUTING,
        "guitar_techs_provenance": guitar_techs_provenance(paths, clips),
        "guitarset_lopo": lopo_identity,
        "clips": [_clip_manifest_record(clip) for clip in clips],
    }


DEVELOPMENT_UNLOCK_FILENAME = "guitarset-development-complete.json"


def _development_input_identity(
    clips: Sequence[ExperimentClip],
    specs: Sequence[ModelSpec],
    paths: RuntimePaths,
) -> dict[str, Any]:
    counts = Counter(clip.corpus for clip in clips)
    if counts != Counter({"guitarset-dev": EXPECTED_CORPUS_COUNTS["guitarset-dev"]}):
        raise RuntimeError("development unlock requires exactly 300 GuitarSet development clips")
    if {spec.name for spec in specs} != set(MODELS) or len(specs) != len(MODELS):
        raise RuntimeError("development unlock requires both frozen TabCNN models")
    clip_records = [
        {
            "clip_id": clip.clip_id,
            "audio_sha256": sha256_file(clip.audio_path),
            "annotation_sha256": sha256_file(clip.annotation_path),
            "event_cache_sha256": (
                sha256_file(clip.event_cache_path) if clip.event_cache_path.is_file() else None
            ),
            "event_cache_strategy": clip.event_cache_strategy,
        }
        for clip in sorted(clips, key=lambda item: item.clip_id)
    ]
    if any(record["event_cache_sha256"] is None for record in clip_records):
        raise RuntimeError("development unlock requires a complete 300-clip event bank")
    q6_reproduction = validate_q6_guitarset_cache(paths)
    if q6_reproduction.get("passed") is not True:
        raise RuntimeError("development unlock requires passing q6 reproduction")
    event_bank_ledgers = validate_event_bank_ledgers(clips, paths)
    if event_bank_ledgers.get("verified") is not True:
        raise RuntimeError("development unlock requires verified event-bank ledgers")
    model_records: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item.name):
        licenses = license_evidence(spec, paths)
        if not licenses["verified"]:
            raise RuntimeError(f"development unlock license contract failed for {spec.name}")
        model_records.append(
            {
                "name": spec.name,
                "artifact_id": spec.artifact_id,
                "expected_sha256": spec.expected_sha256,
                "observed_sha256": sha256_file(spec.checkpoint),
                "license_evidence": licenses,
            }
        )
    identity = {
        "corpus": "guitarset-dev",
        "clips": EXPECTED_CORPUS_COUNTS["guitarset-dev"],
        "models": list(MODELS),
        "inputs_sha256": _sha256_bytes(
            _canonical_json({"clips": clip_records, "models": model_records}).encode("utf-8")
        ),
        "code_revision": evaluation_code_revision(),
        "posterior_code_revision": posterior_generation_revision(),
        "runtime": runtime_manifest(),
        "frozen_scoring": assert_frozen_scoring_environment(),
        "current_60s_latency_seconds": FROZEN_CURRENT_LATENCY_SECONDS,
        "current_60s_latency_source": DEFAULT_CURRENT_LATENCY_SOURCE,
        "protocol_identity": protocol_identity(),
        "q6_cache_reproduction": q6_reproduction,
        "event_bank_ledgers": event_bank_ledgers,
    }
    _, _, _, lopo_identity = build_frozen_guitarset_lopo(
        paths,
        GuitarConfig(),
    )
    identity["guitarset_lopo"] = lopo_identity
    # Unlock evidence is persisted as JSON before it is validated again.  Keep
    # the in-memory identity in that same value domain so tuples nested in
    # dataclass-derived receipts (for example, guitar tuning) cannot compare
    # unequal to their JSON-array representation after the round trip.
    return json.loads(_canonical_json(identity))


def _validate_completed_development_result(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_clip_ids: set[str],
) -> None:
    models = payload.get("models")
    rows = payload.get("per_clip")
    if not isinstance(models, Mapping) or set(models) != set(MODELS):
        raise RuntimeError("development result does not contain both frozen models")
    if any(
        not isinstance(model, Mapping)
        or model.get("status") == "blocked_unscored"
        or not isinstance(model.get("gate"), Mapping)
        for model in models.values()
    ):
        raise RuntimeError("development result contains an incomplete model evaluation")
    if not isinstance(rows, list):
        raise RuntimeError("development result has no per-clip evidence rows")
    expected_pairs = {(model, clip_id) for model in MODELS for clip_id in expected_clip_ids}
    observed_pairs = [
        (str(row.get("model")), str(row.get("clip_id"))) for row in rows if isinstance(row, Mapping)
    ]
    if (
        len(rows) != len(expected_pairs)
        or len(set(observed_pairs)) != len(observed_pairs)
        or set(observed_pairs) != expected_pairs
    ):
        raise RuntimeError(
            "development result does not contain the exact 300-clip x two-model cross-product"
        )
    if any(not isinstance(row, Mapping) or row.get("corpus") != "guitarset-dev" for row in rows):
        raise RuntimeError("development result must not contain sealed or transfer rows")
    if any(row.get("onset_pitch_invariant") is not True for row in rows):
        raise RuntimeError("development result contains a failed onset/pitch invariant")
    if any(
        not isinstance(row.get("posterior_cache"), Mapping)
        or not isinstance(row["posterior_cache"].get("determinism"), Mapping)
        or row["posterior_cache"]["determinism"].get("verified") is not True
        for row in rows
    ):
        raise RuntimeError("development result lacks verified determinism receipts")
    if payload.get("development_input_identity") != expected_identity:
        raise RuntimeError("development result input identity is stale or missing")
    for field in (
        "code_revision",
        "posterior_code_revision",
        "runtime",
        "frozen_scoring",
        "protocol_identity",
    ):
        if payload.get(field) != expected_identity.get(field):
            raise RuntimeError(f"development result {field} does not match unlock identity")
    if payload.get("guitarset_lopo") != expected_identity.get("guitarset_lopo"):
        raise RuntimeError("development result LOPO identity does not match unlock identity")
    q6 = payload.get("q6_cache_reproduction")
    if not isinstance(q6, Mapping) or q6.get("passed") is not True:
        raise RuntimeError("development result lacks a passing q6 reproduction receipt")
    ledgers = payload.get("event_bank_ledgers")
    if not isinstance(ledgers, Mapping) or ledgers.get("verified") is not True:
        raise RuntimeError("development result lacks verified event-bank ledger receipts")


def write_development_unlock(
    paths: RuntimePaths,
    clips: Sequence[ExperimentClip],
    specs: Sequence[ModelSpec],
    *,
    manifest_path: Path,
    results_path: Path,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically unlock sealed access after a complete two-model dev result."""

    identity = _development_input_identity(clips, specs, paths)
    expected_clip_ids = {clip.clip_id for clip in clips}
    _validate_completed_development_result(
        results,
        expected_identity=identity,
        expected_clip_ids=expected_clip_ids,
    )
    resolved_manifest = manifest_path.resolve()
    resolved_results = results_path.resolve()
    if not resolved_manifest.is_file() or not resolved_results.is_file():
        raise RuntimeError(
            "development manifest and result must be atomically written before unlock"
        )
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    on_disk_results = json.loads(resolved_results.read_text(encoding="utf-8"))
    if manifest.get("development_input_identity") != identity:
        raise RuntimeError("development manifest input identity is stale or missing")
    if on_disk_results != results:
        raise RuntimeError("development result bytes do not match the validated payload")
    _validate_completed_development_result(
        on_disk_results,
        expected_identity=identity,
        expected_clip_ids=expected_clip_ids,
    )
    marker = {
        "identity": identity,
        "development_manifest_path": str(resolved_manifest),
        "development_manifest_sha256": sha256_file(resolved_manifest),
        "development_results_path": str(resolved_results),
        "development_results_sha256": sha256_file(resolved_results),
        "completed_models": list(MODELS),
        "completed_clips": EXPECTED_CORPUS_COUNTS["guitarset-dev"],
        "passed": True,
    }
    marker_path = paths.cache_root / DEVELOPMENT_UNLOCK_FILENAME
    _atomic_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return marker


def validate_development_unlock(paths: RuntimePaths) -> dict[str, Any]:
    """Require an untampered, content-matched dev result before sealed access."""

    marker_path = paths.cache_root / DEVELOPMENT_UNLOCK_FILENAME
    if not marker_path.is_file():
        raise RuntimeError(
            "sealed GuitarSet access is locked until the exact 300-clip, two-model "
            "development evaluation completes"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("development unlock marker is invalid") from exc
    clips = _development_clips_for_unlock(paths)
    specs = model_specs(paths.model_root, MODELS)
    expected_identity = _development_input_identity(clips, specs, paths)
    if marker.get("passed") is not True or marker.get("identity") != expected_identity:
        raise RuntimeError("development unlock identity is stale or mismatched")
    manifest_path = Path(str(marker.get("development_manifest_path", "")))
    if not manifest_path.is_file():
        raise RuntimeError("development manifest referenced by unlock marker is missing")
    if sha256_file(manifest_path) != marker.get("development_manifest_sha256"):
        raise RuntimeError("development manifest referenced by unlock marker was modified")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("development_input_identity") != expected_identity:
        raise RuntimeError("development manifest identity is stale or mismatched")
    results_path = Path(str(marker.get("development_results_path", "")))
    if not results_path.is_file():
        raise RuntimeError("development result referenced by unlock marker is missing")
    if sha256_file(results_path) != marker.get("development_results_sha256"):
        raise RuntimeError("development result referenced by unlock marker was modified")
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("development result referenced by unlock marker is invalid") from exc
    if not isinstance(results, Mapping):
        raise RuntimeError("development result root must be an object")
    _validate_completed_development_result(
        results,
        expected_identity=expected_identity,
        expected_clip_ids={clip.clip_id for clip in clips},
    )
    return marker


def require_sealed_unlock(
    paths: RuntimePaths,
    clips: Sequence[ExperimentClip],
    *,
    stage: str,
) -> dict[str, Any] | None:
    if stage == "manifest" or not any(clip.corpus == "guitarset-sealed" for clip in clips):
        return None
    return validate_development_unlock(paths)


def _event_signature(events: Sequence[AudioEvent]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            event.onset_s,
            event.offset_s,
            event.pitch_midi,
            event.velocity,
            event.confidence,
            id(event.pitch_logits),
            event.tags,
        )
        for event in events
    )


def _align_priors_to_decoded(
    raw_events: Sequence[AudioEvent],
    priors: Sequence[np.ndarray | None],
    decoded: Sequence[TabEvent],
) -> list[np.ndarray | None]:
    """Project raw-event-aligned priors onto the decoder's retained events."""

    if len(raw_events) != len(priors):
        raise ValueError("raw events and TabCNN priors must have equal length")
    buckets: dict[tuple[float, int], list[np.ndarray | None]] = {}
    for raw_event, prior in zip(raw_events, priors, strict=True):
        buckets.setdefault((raw_event.onset_s, raw_event.pitch_midi), []).append(prior)
    consumed: dict[tuple[float, int], int] = {}
    aligned: list[np.ndarray | None] = []
    for decoded_event in decoded:
        key = (decoded_event.onset_s, decoded_event.pitch_midi)
        index = consumed.get(key, 0)
        available = buckets.get(key, [])
        if index >= len(available):
            raise RuntimeError(
                "decoded event has no exact raw-event prior alignment: "
                f"onset={decoded_event.onset_s}, pitch={decoded_event.pitch_midi}"
            )
        aligned.append(available[index])
        consumed[key] = index + 1
    return aligned


def map_posteriors_to_events(
    events: Sequence[AudioEvent],
    cached: CachedPosterior,
    *,
    cfg: GuitarConfig,
) -> tuple[list[np.ndarray | None], dict[str, int], float]:
    start = time.perf_counter()
    priors: list[np.ndarray | None] = []
    reasons = {
        "covered": 0,
        "unplayable_pitch": 0,
        "structural_abstention": 0,
        "unsupported_candidates": 0,
        "unsupported_non_neutral": 0,
    }
    for event in events:
        candidates = [
            (string_idx, event.pitch_midi - open_pitch)
            for string_idx, open_pitch in enumerate(cfg.tuning_midi)
            if cfg.capo <= event.pitch_midi - open_pitch <= cfg.max_fret
        ]
        supported = [(string_idx, fret) for string_idx, fret in candidates if fret <= 19]
        if not candidates:
            priors.append(None)
            reasons["unplayable_pitch"] += 1
            continue
        if len(supported) < 2:
            priors.append(None)
            reasons["structural_abstention"] += 1
            continue
        try:
            prior = event_fret_prior(
                event.pitch_midi,
                cached.frames.probabilities,
                cached.frames.times_s,
                event.onset_s,
                max_fret=cfg.max_fret,
            )
        except ValueError:
            prior = None
        priors.append(prior)
        if prior is None:
            reasons["structural_abstention"] += 1
            continue
        reasons["covered"] += 1
        for string_idx, fret in candidates:
            if fret <= 19:
                continue
            reasons["unsupported_candidates"] += 1
            if not np.isclose(prior[string_idx, fret], 1.0, atol=1e-12):
                reasons["unsupported_non_neutral"] += 1
    return priors, reasons, time.perf_counter() - start


def _score_audio_only_frozen(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig,
    session: SessionConfig,
) -> AudioOnlyScore:
    """Score through the explicitly frozen baseline assignment decoder."""

    with assignment_decoder_context("baseline"):
        return score_audio_only(events, gold, cfg=cfg, session=session)


def _decode_with_transition(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig,
    session: SessionConfig,
    transition: Any,
) -> AudioOnlyScore:
    set_transition_prior(transition, weight=SEQUENCE_PRIOR_WEIGHT)
    try:
        return _score_audio_only_frozen(events, gold, cfg=cfg, session=session)
    finally:
        set_transition_prior(None)


def _prepare_guitarset_current(
    events: Sequence[AudioEvent],
    waveform: np.ndarray,
    sample_rate: int,
    cfg: GuitarConfig,
) -> tuple[list[AudioEvent], dict[str, int]]:
    evidence = load_string_evidence()
    return attach_inharmonicity_evidence(
        events,
        waveform,
        sample_rate,
        reference_stiffness_model(),
        cfg,
        weight=evidence.weight,
        min_r2=evidence.min_r2,
        sigma=evidence.sigma,
        isolation=evidence.isolation,
    )


def _attach_candidate_evidence(
    events: Sequence[AudioEvent],
    priors: Sequence[np.ndarray | None],
    *,
    cfg: GuitarConfig,
) -> tuple[list[AudioEvent], float]:
    start = time.perf_counter()
    attached = list(
        attach_tabcnn_priors(
            events,
            priors,
            policy=CONSERVATIVE_BLEND_POLICY,
            cfg=cfg,
        )
    )
    return attached, time.perf_counter() - start


def _apply_guitarset_position_pair(
    current_events: Sequence[AudioEvent],
    candidate_events: Sequence[AudioEvent],
    position: Any,
    *,
    cfg: GuitarConfig,
    candidate_evidence_seconds: float,
) -> tuple[list[AudioEvent], list[AudioEvent], dict[str, float]]:
    start = time.perf_counter()
    positioned_current = apply_pitch_position_prior(current_events, position, cfg)
    current_seconds = time.perf_counter() - start
    start = time.perf_counter()
    positioned_candidate = apply_pitch_position_prior(candidate_events, position, cfg)
    candidate_seconds = time.perf_counter() - start
    incremental = max(
        0.0,
        candidate_evidence_seconds + candidate_seconds - current_seconds,
    )
    return (
        positioned_current,
        positioned_candidate,
        {
            "candidate_evidence": candidate_evidence_seconds,
            "current_position": current_seconds,
            "candidate_position": candidate_seconds,
            "candidate_incremental_prep": incremental,
        },
    )


def _score_three_arms(
    clip: ExperimentClip,
    raw_events: Sequence[AudioEvent],
    priors: Sequence[np.ndarray | None],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig,
    guitarset_positions: Mapping[str, Any] | None,
    guitarset_sequences: Mapping[str, Any] | None,
) -> tuple[
    dict[str, AudioOnlyScore],
    dict[str, float],
    dict[str, int],
    dict[str, float],
]:
    signature = _event_signature(raw_events)
    posterior_events = attach_tabcnn_priors(
        raw_events,
        priors,
        policy=POSTERIOR_ONLY_POLICY,
        cfg=cfg,
    )
    if _event_signature(posterior_events) != signature:
        raise AssertionError("posterior-only attachment mutated onset/pitch event fields")

    session = session_for_clip(clip)
    mapping_timing = {
        "candidate_evidence": 0.0,
        "current_position": 0.0,
        "candidate_position": 0.0,
        "candidate_incremental_prep": 0.0,
    }
    physics_tally = {"events": 0, "isolated": 0, "fitted": 0, "applied": 0}
    if clip.corpus.startswith("guitarset"):
        if clip.player is None or guitarset_positions is None or guitarset_sequences is None:
            raise RuntimeError("GuitarSet scoring requires leave-one-player-out priors")
        waveform, sample_rate = load_mono_audio(clip.audio_path)
        current_events, physics_tally = _prepare_guitarset_current(
            raw_events,
            waveform,
            sample_rate,
            cfg,
        )
        candidate_events, candidate_evidence_seconds = _attach_candidate_evidence(
            current_events,
            priors,
            cfg=cfg,
        )
        position = guitarset_positions[clip.player]
        current_events, candidate_events, mapping_timing = _apply_guitarset_position_pair(
            current_events,
            candidate_events,
            position,
            cfg=cfg,
            candidate_evidence_seconds=candidate_evidence_seconds,
        )

        decode_times: dict[str, float] = {}
        start = time.perf_counter()
        current = _decode_with_transition(
            current_events,
            gold,
            cfg=cfg,
            session=session,
            transition=guitarset_sequences[clip.player],
        )
        decode_times["current"] = time.perf_counter() - start
        start = time.perf_counter()
        posterior_only = _score_audio_only_frozen(posterior_events, gold, cfg=cfg, session=session)
        decode_times["posterior_only"] = time.perf_counter() - start
        start = time.perf_counter()
        candidate = _decode_with_transition(
            candidate_events,
            gold,
            cfg=cfg,
            session=session,
            transition=guitarset_sequences[clip.player],
        )
        decode_times["current_plus_tabcnn"] = time.perf_counter() - start
    elif clip.corpus == "gaps":
        position = load_pitch_position_prior("gaps-v1", cfg=cfg)
        current_events = apply_pitch_position_prior(raw_events, position, cfg)
        candidate_events, candidate_evidence_seconds = _attach_candidate_evidence(
            current_events,
            priors,
            cfg=cfg,
        )
        mapping_timing["candidate_evidence"] = candidate_evidence_seconds
        mapping_timing["candidate_incremental_prep"] = candidate_evidence_seconds
        decode_times = {}
        start = time.perf_counter()
        with sequence_decode_context("gaps-seq-v1"):
            current = _score_audio_only_frozen(current_events, gold, cfg=cfg, session=session)
        decode_times["current"] = time.perf_counter() - start
        start = time.perf_counter()
        posterior_only = _score_audio_only_frozen(posterior_events, gold, cfg=cfg, session=session)
        decode_times["posterior_only"] = time.perf_counter() - start
        start = time.perf_counter()
        with sequence_decode_context("gaps-seq-v1"):
            candidate = _score_audio_only_frozen(candidate_events, gold, cfg=cfg, session=session)
        decode_times["current_plus_tabcnn"] = time.perf_counter() - start
    else:
        candidate_events, candidate_evidence_seconds = _attach_candidate_evidence(
            raw_events,
            priors,
            cfg=cfg,
        )
        mapping_timing["candidate_evidence"] = candidate_evidence_seconds
        mapping_timing["candidate_incremental_prep"] = candidate_evidence_seconds
        decode_times = {}
        start = time.perf_counter()
        current = _score_audio_only_frozen(raw_events, gold, cfg=cfg, session=session)
        decode_times["current"] = time.perf_counter() - start
        start = time.perf_counter()
        posterior_only = _score_audio_only_frozen(posterior_events, gold, cfg=cfg, session=session)
        decode_times["posterior_only"] = time.perf_counter() - start
        start = time.perf_counter()
        candidate = _score_audio_only_frozen(candidate_events, gold, cfg=cfg, session=session)
        decode_times["current_plus_tabcnn"] = time.perf_counter() - start

    current_signature = tuple((event.onset_s, event.pitch_midi) for event in current.decoded)
    candidate_signature = tuple((event.onset_s, event.pitch_midi) for event in candidate.decoded)
    if current_signature != candidate_signature:
        raise AssertionError(f"{clip.clip_id}: onset/pitch predictions changed under TabCNN")
    if current.onset != candidate.onset or current.pitch != candidate.pitch:
        raise AssertionError(f"{clip.clip_id}: onset/pitch scores changed under TabCNN")
    return (
        {
            "current": current,
            "posterior_only": posterior_only,
            "current_plus_tabcnn": candidate,
        },
        decode_times,
        physics_tally,
        mapping_timing,
    )


def _event_metric_dict(result: EventF1Result) -> dict[str, Any]:
    return asdict(result)


def _score_dict(score: AudioOnlyScore) -> dict[str, Any]:
    return {
        "onset": _event_metric_dict(score.onset),
        "pitch": _event_metric_dict(score.pitch),
        "tab": asdict(score.tab),
        "decoded_events": len(score.decoded),
    }


def _complementarity_dict(result: ComplementarityResult) -> dict[str, Any]:
    payload = asdict(result)
    payload.update(
        {
            "covered": result.covered,
            "abstained": result.abstained,
            "coverage": result.coverage,
            "abstention_rate": result.abstention_rate,
            "p_tabcnn_correct": result.p_tabcnn_correct,
            "p_tabcnn_correct_given_current_wrong": (result.p_tabcnn_correct_given_current_wrong),
            "oracle_ceiling": result.oracle_ceiling,
            "oracle_gain": result.oracle_gain,
        }
    )
    return payload


def _aggregate_dict(result: AggregateEvaluation) -> dict[str, Any]:
    return {
        "clips": result.clips,
        "current_macro": asdict(result.current_macro),
        "candidate_macro": asdict(result.candidate_macro),
        "paired_delta": asdict(result.paired_delta),
        "current_micro": asdict(result.current_micro),
        "candidate_micro": asdict(result.candidate_micro),
        "current_errors": result.current_errors.to_dict(),
        "candidate_errors": result.candidate_errors.to_dict(),
        "wrong_position_reduction": result.wrong_position_reduction,
        "wrong_position_relative_reduction": result.wrong_position_relative_reduction,
        "improved_clips": list(result.improved_clips),
        "regressed_clips": list(result.regressed_clips),
        "unchanged_clips": list(result.unchanged_clips),
    }


def _sum_complementarity(items: Sequence[ComplementarityResult]) -> ComplementarityResult:
    names = [field.name for field in fields(ComplementarityResult)]
    totals = {name: sum(int(getattr(item, name)) for item in items) for name in names}
    return ComplementarityResult(**totals)


def _group_evaluations(
    evaluations: Sequence[tuple[ExperimentClip, ClipEvaluation]],
    key: Callable[[ExperimentClip], str | None],
) -> dict[str, Any]:
    groups: dict[str, list[ClipEvaluation]] = {}
    for clip, evaluation in evaluations:
        name = key(clip)
        if name is not None:
            groups.setdefault(name, []).append(evaluation)
    return {
        name: _aggregate_dict(
            aggregate_clip_evaluations(
                group,
                n_bootstrap=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            )
        )
        for name, group in sorted(groups.items())
    }


def _group_complementarity(
    items: Sequence[tuple[ExperimentClip, ComplementarityResult]],
    key: Callable[[ExperimentClip], str | None],
) -> dict[str, Any]:
    groups: dict[str, list[ComplementarityResult]] = {}
    for clip, result in items:
        name = key(clip)
        if name is not None:
            groups.setdefault(name, []).append(result)
    return {
        name: _complementarity_dict(_sum_complementarity(group))
        for name, group in sorted(groups.items())
    }


def canonical_corpus_count_status(
    evaluations: Sequence[tuple[ExperimentClip, ClipEvaluation]],
) -> dict[str, Any]:
    observed = Counter(clip.corpus for clip, _evaluation in evaluations)
    checks = {
        corpus: observed[corpus] == expected for corpus, expected in EXPECTED_CORPUS_COUNTS.items()
    }
    return {
        "passed": all(checks.values()),
        "expected": dict(EXPECTED_CORPUS_COUNTS),
        "observed": dict(observed),
        "checks": checks,
    }


def _micro_event_metrics(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> dict[str, Any]:
    values = [row["scores"][arm][metric] for row in rows]
    true_positives = sum(int(value["true_positives"]) for value in values)
    false_positives = sum(int(value["false_positives"]) for value in values)
    false_negatives = sum(int(value["false_negatives"]) for value in values)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _aggregate_onset_pitch(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty onset/pitch rows")
    result: dict[str, Any] = {"clips": len(rows)}
    for metric in ("onset", "pitch"):
        current = [float(row["scores"]["current"][metric]["f1"]) for row in rows]
        candidate = [float(row["scores"]["current_plus_tabcnn"][metric]["f1"]) for row in rows]
        deltas = [after - before for before, after in zip(current, candidate, strict=True)]
        result[metric] = {
            "current_macro": asdict(
                bootstrap_ci(current, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            ),
            "candidate_macro": asdict(
                bootstrap_ci(candidate, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            ),
            "paired_delta": asdict(
                bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
            ),
            "current_micro": _micro_event_metrics(rows, "current", metric),
            "candidate_micro": _micro_event_metrics(
                rows,
                "current_plus_tabcnn",
                metric,
            ),
        }
    return result


def _group_onset_pitch(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups.setdefault(str(value), []).append(row)
    return {name: _aggregate_onset_pitch(group) for name, group in sorted(groups.items())}


def _large_tier_player_regression_groups(
    evaluations: Sequence[tuple[ExperimentClip, ClipEvaluation]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    results: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for dimension, getter in (
        ("tier", lambda clip: clip.tier),
        ("player", lambda clip: clip.player),
    ):
        names = sorted(
            {str(value) for clip, _evaluation in evaluations if (value := getter(clip)) is not None}
        )
        for name in names:
            population = [
                evaluation for clip, evaluation in evaluations if str(getter(clip)) == name
            ]
            if len(population) < 10:
                continue
            summary = aggregate_clip_evaluations(
                population,
                n_bootstrap=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED,
            )
            label = f"{dimension}:{name}"
            results[label] = _aggregate_dict(summary)
            checks[label] = summary.paired_delta.statistic >= -0.020
    return results, checks


def _cold_latency_summary(
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    model_load_seconds: float,
) -> dict[str, float]:
    duration_seconds = sum(float(row["duration_s"]) for row in timing_rows)
    warm_added_seconds = sum(
        float(row["timing_seconds"]["resample"])
        + float(row["timing_seconds"]["cqt"])
        + float(row["timing_seconds"]["inference"])
        + float(row["timing_seconds"]["mapping"])
        + max(
            0.0,
            float(row["timing_seconds"]["decode"]["current_plus_tabcnn"])
            - float(row["timing_seconds"]["decode"]["current"]),
        )
        for row in timing_rows
    )
    current_decode_seconds = sum(
        float(row["timing_seconds"]["decode"]["current"]) for row in timing_rows
    )
    warm_added_60s = 60.0 * warm_added_seconds / duration_seconds if duration_seconds else math.inf
    current_decode_60s = (
        60.0 * current_decode_seconds / duration_seconds if duration_seconds else math.inf
    )
    return {
        "evaluated_duration_seconds": duration_seconds,
        "model_load_seconds": model_load_seconds,
        "warm_added_60s_seconds": warm_added_60s,
        "cold_added_60s_seconds": model_load_seconds + warm_added_60s,
        "current_decode_only_60s_seconds": current_decode_60s,
    }


def _frozen_gate(
    model: ModelSpec,
    evaluations: Sequence[tuple[ExperimentClip, ClipEvaluation]],
    model_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    current_60s_latency_seconds: float | None,
    current_60s_latency_source: str | None,
    performance_receipt: Mapping[str, Any] | None = None,
    guitar_techs_provenance_verified: bool = False,
    paths: RuntimePaths | None = None,
) -> dict[str, Any]:
    count_status = canonical_corpus_count_status(evaluations)
    if not count_status["passed"]:
        return {
            "status": "blocked_incomplete_corpora",
            "corpus_counts": count_status,
            "evidence_positive": False,
            "decision": "do_not_integrate",
        }
    by_corpus = {
        corpus: [evaluation for clip, evaluation in evaluations if clip.corpus == corpus]
        for corpus in CORPORA
    }
    eligible = list(by_corpus["gaps"])
    if model.name == "synthtab":
        eligible.extend(by_corpus["guitarset-sealed"])
    comp = [
        evaluation
        for clip, evaluation in evaluations
        if clip.corpus == "guitarset-sealed" and clip.tier == "comp"
    ]
    solo = [evaluation for evaluation in eligible if evaluation.tier == "solo"]
    eligible_summary = aggregate_clip_evaluations(
        eligible, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED
    )
    solo_summary = aggregate_clip_evaluations(solo, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    comp_summary = aggregate_clip_evaluations(comp, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    electric_summaries = {
        corpus: aggregate_clip_evaluations(
            by_corpus[corpus],
            n_bootstrap=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
        )
        for corpus in ("egset12", "guitar-techs")
    }

    group_results, group_checks = _large_tier_player_regression_groups(evaluations)
    performance_verified = bool(
        isinstance(performance_receipt, Mapping)
        and performance_receipt.get("verified") is True
        and performance_receipt.get("model") == model.name
        and performance_receipt.get("fresh_process_per_model") is True
    )
    model_load_seconds = (
        float(performance_receipt["model_load_seconds"])
        if performance_verified and performance_receipt is not None
        else math.inf
    )
    latency = _cold_latency_summary(
        timing_rows,
        model_load_seconds=model_load_seconds,
    )
    added_60s = latency["cold_added_60s_seconds"]
    latency_known = current_60s_latency_seconds is not None and current_60s_latency_seconds > 0.0
    total_60s = (
        current_60s_latency_seconds + added_60s if current_60s_latency_seconds is not None else None
    )
    added_ratio = (
        added_60s / current_60s_latency_seconds
        if latency_known and current_60s_latency_seconds is not None
        else None
    )

    electric_checks = {
        corpus: summary.paired_delta.statistic >= -0.005
        for corpus, summary in electric_summaries.items()
    }
    invariant = all(bool(row["onset_pitch_invariant"]) for row in model_rows)
    deterministic = all(
        bool(row["posterior_cache"]["determinism"].get("verified")) for row in model_rows
    )
    unsupported_neutral = all(
        int(row["abstention"].get("unsupported_non_neutral", 0)) == 0 for row in model_rows
    )
    artifact_contracts = {
        "synthtab": {
            "artifact_id": SYNTHTAB_X4.artifact_id,
            "sha256": SYNTHTAB_SHA256,
            "source_revision": SYNTHTAB_X4.source_revision,
        },
        "dafx": {
            "artifact_id": DAFX_GUITARPROFX_ONNX.artifact_id,
            "sha256": DAFX_SHA256,
            "source_revision": DAFX_GUITARPROFX_ONNX.source_revision,
        },
    }
    expected_artifact = artifact_contracts.get(model.name, {})
    transport_equivalence_ok = (
        model.name != "dafx" or DAFX_GUITARPROFX_ONNX.official_equivalence_verified is True
    )
    provenance_verified = bool(
        model.checkpoint.is_file()
        and sha256_file(model.checkpoint) == model.expected_sha256
        and model.artifact_id == expected_artifact.get("artifact_id")
        and model.expected_sha256 == expected_artifact.get("sha256")
        and model.source_revision == expected_artifact.get("source_revision")
        and transport_equivalence_ok
    )
    licenses = license_evidence(model, paths)
    evaluation_license_ok = bool(licenses["verified"])
    checks = {
        "aggregate_delta_at_least_0_020": eligible_summary.paired_delta.statistic >= 0.020,
        "aggregate_lower95_positive": eligible_summary.paired_delta.lower > 0.0,
        "solo_delta_at_least_0_030": solo_summary.paired_delta.statistic >= 0.030,
        "wrong_position_reduction_at_least_10pct": (
            solo_summary.wrong_position_relative_reduction >= 0.10
        ),
        "comp_delta_at_least_minus_0_005": comp_summary.paired_delta.statistic >= -0.005,
        "egset12_delta_at_least_minus_0_005": electric_checks["egset12"],
        "guitar_techs_delta_at_least_minus_0_005": electric_checks["guitar-techs"],
        "all_large_tier_player_groups_at_least_minus_0_020": all(group_checks.values()),
        "onset_pitch_exactly_invariant": invariant,
        "repeat_run_deterministic": deterministic,
        "provenance_verified": provenance_verified,
        "evaluation_license_ok": evaluation_license_ok,
        "guitar_techs_revision_verified": guitar_techs_provenance_verified,
        "unsupported_positions_neutral": unsupported_neutral,
        "cache_performance_receipt_verified": performance_verified,
        "total_60s_cpu_below_5min": bool(total_60s is not None and total_60s < 300.0),
        "added_cpu_at_most_20pct": bool(added_ratio is not None and added_ratio <= 0.20),
    }
    blockers: list[str] = []
    if current_60s_latency_seconds != FROZEN_CURRENT_LATENCY_SECONDS:
        blockers.append("current_60s_latency_seconds_differs_from_frozen_262.495")
    if current_60s_latency_source != DEFAULT_CURRENT_LATENCY_SOURCE:
        blockers.append("current_60s_latency_source_differs_from_frozen_protocol")
    if not guitar_techs_provenance_verified:
        blockers.append("guitar_techs_pinned_revision_not_verified")
    if model.name == "dafx" and not transport_equivalence_ok:
        blockers.append("dafx_official_checkpoint_to_onnx_equivalence_not_verified")
    passed = all(checks.values()) and not blockers
    return {
        "status": "evaluated" if not blockers else "blocked_protocol_evidence",
        "blockers": blockers,
        "corpus_counts": count_status,
        "eligible_pool": (["gaps", "guitarset-sealed"] if model.name == "synthtab" else ["gaps"]),
        "checks": checks,
        "evidence_positive": passed,
        "decision": "evidence_positive" if passed else "do_not_integrate",
        "eligible": _aggregate_dict(eligible_summary),
        "solo": _aggregate_dict(solo_summary),
        "comp": _aggregate_dict(comp_summary),
        "electric": {
            name: _aggregate_dict(summary) for name, summary in electric_summaries.items()
        },
        "large_tier_player_groups": group_results,
        "performance": {
            "current_60s_latency_seconds": current_60s_latency_seconds,
            "current_60s_latency_source": current_60s_latency_source,
            "model_load_seconds": latency["model_load_seconds"],
            "warm_added_60s_cpu_seconds": latency["warm_added_60s_seconds"],
            "current_decode_only_60s_seconds": latency["current_decode_only_60s_seconds"],
            "added_60s_cpu_seconds": added_60s,
            "total_60s_cpu_seconds": total_60s,
            "added_to_current_ratio": added_ratio,
            "note": (
                "current full-pipeline latency is an explicit external measurement "
                "because this evaluation consumes banked highres events"
            ),
        },
        "provenance_license": {
            "checkpoint_sha256_verified": provenance_verified,
            "transport_equivalence_verified": transport_equivalence_ok,
            "evaluation_license_ok": evaluation_license_ok,
            "license_evidence": licenses,
            "license_id": model.license_id,
            "license_posture": model.license_posture,
            "shipping_or_redistribution_authorized": (model.shipping_redistribution_allowed),
        },
    }


def evaluate(
    clips: Sequence[ExperimentClip],
    specs: Sequence[ModelSpec],
    paths: RuntimePaths,
    *,
    allow_transcribe_missing: bool,
    current_60s_latency_seconds: float = FROZEN_CURRENT_LATENCY_SECONDS,
    current_60s_latency_source: str = DEFAULT_CURRENT_LATENCY_SOURCE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    corpus_status: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if allow_transcribe_missing:
        raise RuntimeError(
            "evaluate is prebank-only; transcription is permitted only in bank-events"
        )
    assert_frozen_scoring_environment()
    if current_60s_latency_seconds != FROZEN_CURRENT_LATENCY_SECONDS:
        raise RuntimeError(
            "the scored current-latency baseline is frozen at "
            f"{FROZEN_CURRENT_LATENCY_SECONDS} seconds"
        )
    if current_60s_latency_source != DEFAULT_CURRENT_LATENCY_SOURCE:
        raise RuntimeError("the scored current-latency provenance string is frozen")
    debug_limited = any(
        isinstance(record, Mapping) and record.get("status") == "debug_limited"
        for record in (corpus_status or {}).values()
    )
    if chunk_size != DEFAULT_CHUNK_SIZE and not debug_limited:
        raise RuntimeError(
            "promotion-eligible evaluation requires the frozen default inference chunk size"
        )

    legacy_validation = None
    guitarset_corpora = sorted(
        {clip.corpus for clip in clips if clip.corpus.startswith("guitarset")}
    )
    if any(clip.event_cache_strategy.startswith("legacy-a3") for clip in clips):
        legacy_validation = validate_legacy_guitarset_cache(
            paths,
            corpora=guitarset_corpora,
        )
    q6_validation = None
    if any(clip.event_cache_strategy == "q6" for clip in clips):
        q6_validation = validate_q6_guitarset_cache(paths)
    missing_banks = [clip.clip_id for clip in clips if not clip.event_cache_path.is_file()]
    if missing_banks:
        raise RuntimeError(
            f"evaluation requires prebanked events; missing {len(missing_banks)} clips"
        )
    event_bank_ledgers = validate_event_bank_ledgers(
        clips,
        paths,
        require_complete_selection=not debug_limited,
    )

    cfg = GuitarConfig()
    provider = RawEventProvider(allow_transcribe_missing=False)
    code_revision = evaluation_code_revision()
    posterior_revision = posterior_generation_revision()
    runtime = runtime_manifest()
    performance_receipts: dict[str, dict[str, Any]] = {}
    performance_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    if clips:
        clip_ids = [clip.clip_id for clip in clips]
        for spec in specs:
            receipt, receipt_rows = load_cache_performance_receipt(
                paths.cache_root,
                model_name=spec.name,
                expected_clip_ids=clip_ids,
                expected_code_revision=posterior_revision,
                expected_runtime=runtime,
                expected_chunk_size=chunk_size,
            )
            performance_receipts[spec.name] = receipt
            performance_rows[spec.name] = receipt_rows
    guitar_techs_identity = guitar_techs_provenance(paths, clips)
    guitarset_lopo = None
    guitarset_gold = None
    guitarset_positions = None
    guitarset_sequences = None
    if any(clip.corpus.startswith("guitarset") for clip in clips):
        guitarset_gold, guitarset_positions, guitarset_sequences, guitarset_lopo = (
            build_frozen_guitarset_lopo(paths, cfg)
        )

    all_rows: list[dict[str, Any]] = []
    model_results: dict[str, Any] = {}
    for spec in specs:
        model_rows: list[dict[str, Any]] = []
        evaluations: list[tuple[ExperimentClip, ClipEvaluation]] = []
        complementarity: list[tuple[ExperimentClip, ComplementarityResult]] = []
        timing_rows: list[dict[str, Any]] = []
        for clip in clips:
            identity = posterior_cache_identity(
                clip.audio_path,
                spec,
                code_revision=posterior_revision,
                chunk_size=chunk_size,
            )
            cache_path = posterior_cache_path(paths.cache_root, clip, spec, identity)
            cached = load_posterior_cache(cache_path, identity)
            receipt_row = performance_rows[spec.name][clip.clip_id]
            determinism = posterior_determinism_status(cached)
            expected_performance_row = {
                "cache_key": cached.metadata["cache_key"],
                "posterior_sha256": cached.metadata["posterior_sha256"],
                "size_bytes": cached.path.stat().st_size,
                "duration_s": cached.metadata["duration_s"],
                "timing_seconds": cached.metadata["timing_seconds"],
                "determinism": determinism,
            }
            mismatches = [
                field
                for field, value in expected_performance_row.items()
                if receipt_row.get(field) != value
            ]
            if mismatches:
                raise RuntimeError(
                    f"{spec.name}/{clip.clip_id} cache performance receipt mismatch: "
                    + ", ".join(mismatches)
                )
            raw_events = provider.load(clip)
            raw_signature = _event_signature(raw_events)
            priors, abstention, mapping_seconds = map_posteriors_to_events(
                raw_events,
                cached,
                cfg=cfg,
            )
            if _event_signature(raw_events) != raw_signature:
                raise AssertionError(f"{clip.clip_id}: posterior mapping mutated raw events")
            if guitarset_gold is not None and clip.corpus.startswith("guitarset"):
                assert clip.player is not None
                gold = guitarset_gold[clip.player][clip.clip_id.split("/", 1)[1]]
            else:
                gold = parse_gold(clip, cfg)
            scores, decode_times, physics_tally, candidate_mapping = _score_three_arms(
                clip,
                raw_events,
                priors,
                gold,
                cfg=cfg,
                guitarset_positions=guitarset_positions,
                guitarset_sequences=guitarset_sequences,
            )
            current = scores["current"]
            candidate = scores["current_plus_tabcnn"]
            clip_evaluation = score_clip(
                clip.clip_id,
                clip.tier,
                current.decoded,
                candidate.decoded,
                gold,
            )
            comp = evaluate_complementarity(
                current.decoded,
                gold,
                _align_priors_to_decoded(raw_events, priors, current.decoded),
                cfg=cfg,
            )
            evaluations.append((clip, clip_evaluation))
            complementarity.append((clip, comp))
            timing = dict(cached.metadata["timing_seconds"])
            timing["mapping_components"] = {
                "posterior_to_prior": mapping_seconds,
                **candidate_mapping,
            }
            timing["mapping"] = mapping_seconds + candidate_mapping["candidate_incremental_prep"]
            timing["decode"] = decode_times
            timing_row = {
                "clip_id": clip.clip_id,
                "duration_s": cached.metadata["duration_s"],
                "timing_seconds": timing,
            }
            timing_rows.append(timing_row)
            row = {
                "model": spec.name,
                "clip_id": clip.clip_id,
                "corpus": clip.corpus,
                "source": clip.source,
                "split": clip.split,
                "tier": clip.tier,
                "player": clip.player,
                "mode": clip.mode,
                "gold_events": len(gold),
                "raw_events": len(raw_events),
                "scores": {name: _score_dict(score) for name, score in scores.items()},
                "errors": {
                    name: decompose_errors(score.decoded, gold).to_dict()
                    for name, score in scores.items()
                },
                "delta_tab_f1": candidate.tab.f1 - current.tab.f1,
                "complementarity": _complementarity_dict(comp),
                "abstention": abstention,
                "physics": physics_tally,
                "onset_pitch_invariant": True,
                "posterior_cache": {
                    "path": str(cached.path),
                    "key": cached.metadata["cache_key"],
                    "sha256": cached.metadata["posterior_sha256"],
                    "size_bytes": cached.path.stat().st_size,
                    "determinism": determinism,
                },
                "duration_s": cached.metadata["duration_s"],
                "timing_seconds": timing,
            }
            model_rows.append(row)
            all_rows.append(row)

        if not evaluations:
            model_results[spec.name] = {
                "status": "blocked_unscored",
                "model": {**asdict(spec), "checkpoint": str(spec.checkpoint)},
                "gate": _frozen_gate(
                    spec,
                    evaluations,
                    model_rows,
                    timing_rows,
                    current_60s_latency_seconds=current_60s_latency_seconds,
                    current_60s_latency_source=current_60s_latency_source,
                    performance_receipt=performance_receipts.get(spec.name),
                    guitar_techs_provenance_verified=bool(guitar_techs_identity.get("verified")),
                    paths=paths,
                ),
            }
            continue

        aggregate = aggregate_clip_evaluations(
            [evaluation for _, evaluation in evaluations],
            n_bootstrap=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
        )
        aggregate_comp = _sum_complementarity([result for _clip, result in complementarity])
        model_results[spec.name] = {
            "model": {
                **asdict(spec),
                "checkpoint": str(spec.checkpoint),
            },
            "aggregate": _aggregate_dict(aggregate),
            "complementarity": {
                **_complementarity_dict(aggregate_comp),
                "by_corpus": _group_complementarity(complementarity, lambda clip: clip.corpus),
                "by_source": _group_complementarity(complementarity, lambda clip: clip.source),
                "by_tier": _group_complementarity(complementarity, lambda clip: clip.tier),
                "by_player": _group_complementarity(complementarity, lambda clip: clip.player),
            },
            "by_corpus": _group_evaluations(evaluations, lambda clip: clip.corpus),
            "by_source": _group_evaluations(evaluations, lambda clip: clip.source),
            "by_tier": _group_evaluations(evaluations, lambda clip: clip.tier),
            "by_player": _group_evaluations(evaluations, lambda clip: clip.player),
            "onset_pitch": {
                "aggregate": _aggregate_onset_pitch(model_rows),
                "by_corpus": _group_onset_pitch(model_rows, "corpus"),
                "by_source": _group_onset_pitch(model_rows, "source"),
                "by_tier": _group_onset_pitch(model_rows, "tier"),
                "by_player": _group_onset_pitch(model_rows, "player"),
            },
            "gate": _frozen_gate(
                spec,
                evaluations,
                model_rows,
                timing_rows,
                current_60s_latency_seconds=current_60s_latency_seconds,
                current_60s_latency_source=current_60s_latency_source,
                performance_receipt=performance_receipts.get(spec.name),
                guitar_techs_provenance_verified=bool(guitar_techs_identity.get("verified")),
                paths=paths,
            ),
        }
    return (
        {
            "format_version": RESULT_FORMAT_VERSION,
            "protocol_frozen": True,
            "protocol_identity": protocol_identity(),
            "evaluation_only": True,
            "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
            "code_revision": code_revision,
            "posterior_code_revision": posterior_revision,
            "runtime": runtime,
            "frozen_scoring": assert_frozen_scoring_environment(),
            "corpus_status": dict(corpus_status or {}),
            "guitarset_lopo": guitarset_lopo,
            "guitar_techs_provenance": guitar_techs_identity,
            "peak_rss_bytes": (
                max(int(receipt["peak_rss_bytes"]) for receipt in performance_receipts.values())
                if performance_receipts
                else None
            ),
            "evaluation_peak_rss_bytes": peak_rss_bytes(),
            "cache_performance_receipts": performance_receipts,
            "legacy_cache_reproduction": legacy_validation,
            "q6_cache_reproduction": q6_validation,
            "event_bank_ledgers": event_bank_ledgers,
            "current_60s_latency_baseline": {
                "seconds": current_60s_latency_seconds,
                "source": current_60s_latency_source,
            },
            "models": model_results,
            "per_clip": all_rows,
        },
        all_rows,
    )


def cache_all_posteriors(
    clips: Sequence[ExperimentClip],
    specs: Sequence[ModelSpec],
    paths: RuntimePaths,
    *,
    chunk_size: int,
    verify_determinism: bool = True,
) -> dict[str, Any]:
    if len(specs) != 1:
        raise RuntimeError("cache performance evidence requires one fresh process per model")
    spec = specs[0]
    revision = posterior_generation_revision()
    runtime = runtime_manifest()
    rows: list[dict[str, Any]] = []
    load_start = time.perf_counter()
    backend = load_model_backend(spec)
    model_load_seconds = time.perf_counter() - load_start
    for clip in clips:
        cached, resumed = ensure_posterior_cache(
            clip,
            spec,
            paths.cache_root,
            backend,
            chunk_size=chunk_size,
            code_revision=revision,
        )
        determinism = posterior_determinism_status(cached)
        determinism_reused = bool(
            verify_determinism and resumed and determinism.get("verified") is True
        )
        if verify_determinism and not determinism_reused:
            determinism = verify_posterior_determinism(
                clip,
                cached,
                backend,
                chunk_size=chunk_size,
            )
        rows.append(
            {
                "model": spec.name,
                "clip_id": clip.clip_id,
                "path": str(cached.path),
                "resumed": resumed,
                "cache_key": cached.metadata["cache_key"],
                "posterior_sha256": cached.metadata["posterior_sha256"],
                "size_bytes": cached.path.stat().st_size,
                "duration_s": cached.metadata["duration_s"],
                "timing_seconds": cached.metadata["timing_seconds"],
                "determinism": determinism,
                "determinism_reused": determinism_reused,
            }
        )
    observed_peak = peak_rss_bytes()
    if observed_peak is None:
        raise RuntimeError("cache-stage peak RSS is unavailable on this platform")
    return {
        "format_version": RECEIPT_FORMAT_VERSION,
        "model": spec.name,
        "fresh_process_per_model": True,
        "code_revision": revision,
        "runtime": runtime,
        "inference_chunk_size": chunk_size,
        "model_load_seconds": model_load_seconds,
        "peak_rss_bytes": observed_peak,
        "duration_s": sum(float(row["duration_s"]) for row in rows),
        "posteriors": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat = [
        {
            "model": row["model"],
            "clip_id": row["clip_id"],
            "corpus": row["corpus"],
            "source": row["source"],
            "split": row["split"],
            "tier": row["tier"],
            "player": row["player"],
            "mode": row["mode"],
            "gold_events": row["gold_events"],
            "raw_events": row["raw_events"],
            "current_onset_f1": row["scores"]["current"]["onset"]["f1"],
            "candidate_onset_f1": row["scores"]["current_plus_tabcnn"]["onset"]["f1"],
            "current_pitch_f1": row["scores"]["current"]["pitch"]["f1"],
            "candidate_pitch_f1": row["scores"]["current_plus_tabcnn"]["pitch"]["f1"],
            "current_tab_f1": row["scores"]["current"]["tab"]["f1"],
            "posterior_only_tab_f1": row["scores"]["posterior_only"]["tab"]["f1"],
            "candidate_tab_f1": row["scores"]["current_plus_tabcnn"]["tab"]["f1"],
            "delta_tab_f1": row["delta_tab_f1"],
            "coverage": row["complementarity"]["coverage"],
            "p_tabcnn_correct": row["complementarity"]["p_tabcnn_correct"],
            "p_tabcnn_correct_given_current_wrong": row["complementarity"][
                "p_tabcnn_correct_given_current_wrong"
            ],
            "posterior_sha256": row["posterior_cache"]["sha256"],
            "posterior_size_bytes": row["posterior_cache"]["size_bytes"],
            "cqt_seconds": row["timing_seconds"]["cqt"],
            "inference_seconds": row["timing_seconds"]["inference"],
            "mapping_seconds": row["timing_seconds"]["mapping"],
            "current_decode_seconds": row["timing_seconds"]["decode"]["current"],
            "candidate_decode_seconds": row["timing_seconds"]["decode"]["current_plus_tabcnn"],
        }
        for row in rows
    ]
    if not flat:
        raise ValueError("cannot write an empty per-clip CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        os.replace(raw_temp, path)
    finally:
        if os.path.exists(raw_temp):
            os.unlink(raw_temp)


def _default_data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data"))


def _runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    data_root = args.data_root.expanduser()
    model_root = (args.model_root or data_root / "models").expanduser()
    cache_root = (args.cache_root or data_root / "tabcnn-complementarity").expanduser()
    guitarset_default = Path.home() / "mir_datasets" / "guitarset"
    return RuntimePaths(
        data_root=data_root,
        model_root=model_root,
        cache_root=cache_root,
        guitarset_root=(args.guitarset_root or guitarset_default).expanduser(),
        gaps_root=(args.gaps_root or data_root / "gaps").expanduser(),
        egset12_root=(args.egset12_root or data_root / "egset12").expanduser(),
        guitar_techs_root=(args.guitar_techs_root or data_root / "guitar-techs-hf").expanduser(),
        q6_dev_cache=(args.q6_dev_cache or model_root / "q6_full_dev_cache").expanduser(),
        q6_player05_cache=(args.q6_player05_cache or model_root / "q6_player05_cache").expanduser(),
        q6_gaps_cache=(args.q6_gaps_cache or model_root / "q6_gaps_cache").expanduser(),
        legacy_guitarset_cache=(
            args.legacy_guitarset_cache or Path.home() / ".tabvision" / "cache" / "a3_fusion_sweep"
        ).expanduser(),
    )


def _selection(values: Sequence[str] | None, choices: Sequence[str]) -> list[str]:
    if not values or "all" in values:
        return list(choices)
    result: list[str] = []
    for value in values:
        for item in value.split(","):
            if item not in choices:
                raise ValueError(f"unknown selection {item!r}; choices: {', '.join(choices)}")
            if item not in result:
                result.append(item)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("bank-events", "manifest", "cache-posteriors", "evaluate", "all"),
    )
    parser.add_argument("--corpus", action="append", help="repeat, comma-separate, or use all")
    parser.add_argument("--model", action="append", help="repeat, comma-separate, or use all")
    parser.add_argument("--limit", type=int, default=0, help="debug cap per selected corpus")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--data-root", type=Path, default=_default_data_root())
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--guitarset-root", type=Path)
    parser.add_argument("--gaps-root", type=Path)
    parser.add_argument("--egset12-root", type=Path)
    parser.add_argument("--guitar-techs-root", type=Path)
    parser.add_argument("--q6-dev-cache", type=Path)
    parser.add_argument("--q6-player05-cache", type=Path)
    parser.add_argument("--q6-gaps-cache", type=Path)
    parser.add_argument("--legacy-guitarset-cache", type=Path)
    parser.add_argument("--synthtab-checkpoint", type=Path)
    parser.add_argument("--dafx-checkpoint", type=Path)
    parser.add_argument(
        "--current-60s-latency-seconds",
        type=float,
        default=FROZEN_CURRENT_LATENCY_SECONDS,
        help="measured current full-pipeline CPU latency for a 60-second clip",
    )
    parser.add_argument(
        "--current-60s-latency-source",
        default=DEFAULT_CURRENT_LATENCY_SOURCE,
        help="reader-facing provenance for --current-60s-latency-seconds",
    )
    parser.add_argument(
        "--skip-determinism-check",
        action="store_true",
        help="debug only; leaves the frozen determinism gate blocked",
    )
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--csv", dest="csv_path", type=Path)
    parser.add_argument(
        "--allow-transcribe-missing",
        action="store_true",
        help="explicitly permit highres transcription into the external event bank",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.current_60s_latency_seconds != FROZEN_CURRENT_LATENCY_SECONDS:
        raise SystemExit(
            f"--current-60s-latency-seconds is frozen at {FROZEN_CURRENT_LATENCY_SECONDS}"
        )
    if args.current_60s_latency_source != DEFAULT_CURRENT_LATENCY_SOURCE:
        raise SystemExit("--current-60s-latency-source is frozen by the protocol")
    if args.limit == 0 and args.chunk_size != DEFAULT_CHUNK_SIZE:
        raise SystemExit("promotion-eligible runs require the default --chunk-size")
    if args.skip_determinism_check and args.limit == 0:
        raise SystemExit("--skip-determinism-check is debug-only and requires --limit")
    assert_frozen_scoring_environment()
    protocol = protocol_identity()
    corpora = _selection(args.corpus, CORPORA)
    selected_models = _selection(args.model, MODELS)
    if args.stage in {"cache-posteriors", "all"} and len(selected_models) != 1:
        raise SystemExit(
            "cache performance evidence requires one model per fresh process; "
            "run synthtab and dafx separately"
        )
    if args.limit and set(corpora) != {"guitarset-dev"}:
        raise SystemExit(
            "--limit is restricted to GuitarSet development; "
            "sealed and transfer peeks are forbidden"
        )
    paths = _runtime_paths(args)

    if "guitarset-sealed" in corpora:
        validate_development_unlock(paths)

    if args.stage == "bank-events":
        banks: dict[str, Any] = {}
        if "guitarset-dev" in corpora:
            banks["guitarset-dev"] = bank_guitarset_events(
                paths,
                split="dev",
                allow_transcribe_missing=args.allow_transcribe_missing,
            )
        if "guitarset-sealed" in corpora:
            banks["guitarset-sealed"] = bank_guitarset_events(
                paths,
                split="sealed",
                allow_transcribe_missing=args.allow_transcribe_missing,
            )
        transfer_corpora = [
            corpus for corpus in corpora if corpus in {"gaps", "egset12", "guitar-techs"}
        ]
        corpus_status: dict[str, dict[str, Any]] = {}
        if transfer_corpora:
            transfer_clips = discover_clips(
                paths,
                transfer_corpora,
                limit=args.limit,
                corpus_status=corpus_status,
            )
            bankable = [
                clip
                for clip in transfer_clips
                if corpus_status.get(clip.corpus, {}).get("status") in {"ready", "debug_limited"}
            ]
            if bankable:
                banks["transfer"] = bank_transfer_events(
                    bankable,
                    paths,
                    allow_transcribe_missing=args.allow_transcribe_missing,
                )
        summary = {
            "format_version": RESULT_FORMAT_VERSION,
            "protocol_identity": protocol,
            "corpus_status": corpus_status,
            "banks": banks,
        }
        destination = args.json_path or paths.cache_root / "event-bank-summary.json"
        _atomic_text(destination, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {destination}")
        return 0

    if args.allow_transcribe_missing:
        raise SystemExit("--allow-transcribe-missing is valid only for bank-events")
    specs = model_specs(
        paths.model_root,
        selected_models,
        synthtab_checkpoint=args.synthtab_checkpoint,
        dafx_checkpoint=args.dafx_checkpoint,
    )
    corpus_status = {}
    clips = discover_clips(paths, corpora, limit=args.limit, corpus_status=corpus_status)
    if not clips and not any(
        record.get("status") == "blocked_unscored" for record in corpus_status.values()
    ):
        raise SystemExit("no clips selected")
    manifest = build_manifest(
        clips, specs, paths, corpus_status=corpus_status, debug_limit=args.limit
    )
    manifest["current_60s_latency_baseline"] = {
        "seconds": args.current_60s_latency_seconds,
        "source": args.current_60s_latency_source,
    }
    development_run = (
        args.limit == 0
        and set(corpora) == {"guitarset-dev"}
        and set(selected_models) == set(MODELS)
    )
    if development_run:
        manifest["development_input_identity"] = _development_input_identity(
            clips,
            specs,
            paths,
        )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_digest = _sha256_bytes(manifest_text.encode("utf-8"))
    manifest_path = args.manifest_json or (
        paths.cache_root / f"development-manifest-{manifest_digest[:16]}.json"
        if development_run
        else paths.cache_root / "manifest.json"
    )
    _atomic_text(manifest_path, manifest_text)
    if args.stage == "manifest":
        print(f"wrote {manifest_path} ({len(clips)} clips, {len(specs)} models)")
        return 0

    if args.stage in {"cache-posteriors", "all"}:
        validate_event_bank_ledgers(
            clips,
            paths,
            require_complete_selection=args.limit == 0,
        )
        cache_summary = cache_all_posteriors(
            clips,
            specs,
            paths,
            chunk_size=args.chunk_size,
            verify_determinism=not args.skip_determinism_check,
        )
        cache_receipt = write_cache_performance_receipt(
            paths.cache_root,
            cache_summary,
            model_name=specs[0].name,
            destination=args.json_path,
        )
        if args.stage == "cache-posteriors":
            print(f"wrote {cache_receipt['path']}")
            return 0

    results, rows = evaluate(
        clips,
        specs,
        paths,
        allow_transcribe_missing=args.allow_transcribe_missing,
        current_60s_latency_seconds=args.current_60s_latency_seconds,
        current_60s_latency_source=args.current_60s_latency_source,
        chunk_size=args.chunk_size,
        corpus_status=corpus_status,
    )
    if development_run:
        results["development_input_identity"] = manifest["development_input_identity"]
        results["development_manifest"] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        }
    # Hash, persist, and validate the exact same JSON-domain value.  Evaluation
    # receipts contain dataclass-derived tuples, which JSON represents as
    # arrays; normalizing here prevents an otherwise valid atomic write from
    # failing its immediate read-back comparison.
    results = json.loads(_canonical_json(results))
    results_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    results_digest = _sha256_bytes(results_text.encode("utf-8"))
    json_path = args.json_path or (
        paths.cache_root / f"development-results-{results_digest[:16]}.json"
        if development_run
        else paths.cache_root / "results.json"
    )
    csv_path = args.csv_path or (
        paths.cache_root / f"development-per-clip-{results_digest[:16]}.csv"
        if development_run
        else paths.cache_root / "per-clip.csv"
    )
    _atomic_text(json_path, results_text)
    if rows:
        _write_csv(csv_path, rows)
    if development_run:
        write_development_unlock(
            paths,
            clips,
            specs,
            manifest_path=manifest_path,
            results_path=json_path,
            results=results,
        )
    suffix = f" and {csv_path}" if rows else " (blocked/unscored; no CSV)"
    print(f"wrote {json_path}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
