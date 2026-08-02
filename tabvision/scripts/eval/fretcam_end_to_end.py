"""Paired real-audio/current-FretCam Tab F1 evaluation on aligned GAPS video.

This runner answers the question the gold-pitch bridge probe cannot: does the
*current stabilized FretCam solver* improve end-to-end Tab F1 when fused with
real high-resolution audio predictions?

By default, the two arms share the current production prediction cache. The
cached ``TabEvent`` records are stripped only of string/fret, converted back
to detected pitch/onset events, and passed through the current clean-classical
policy (``gaps-v1`` + ``gaps-seq-v1`` at weight 4, baseline assignment
decoder). The runner requires that this reconstruction reproduce every cached
baseline assignment exactly before it scores FretCam. A raw-``AudioEvent``
cache mode is also available for the clean-12 development subset.

No gold pitch, gold string, cached CV anchor, policy tuning, or model training
enters either prediction arm.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from fretcam.tabvision_adapter import FretCamPositionAnalyzer

from scripts.acquire.gaps_video import CLEAN_12
from tabvision.eval.bootstrap import BootstrapResult, bootstrap_ci
from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.eval.metrics import TabF1Result, tab_f1
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.contact_prior import apply_contact_priors
from tabvision.fusion.inference_policy import ResolvedInferencePolicy, resolve_inference_policy
from tabvision.fusion.position_window_prior import apply_position_window_priors
from tabvision.fusion.viterbi import fuse
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT, run_pipeline_with_artifacts
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent
from tabvision.video.position import PositionWindowObservation, VideoObservations

SCHEMA_VERSION = 2
POSITION_PRIOR_NAME = "gaps-v1"
SEQUENCE_PRIOR_NAME = "gaps-seq-v1"
ONSET_TOLERANCE_S = 0.05
MIN_ALIGNMENT_PEAK_RATIO = 2.0
MAX_DURATION_MISMATCH_S = 0.5
ALIGNMENT_COVERAGE_SLACK_S = 0.15
DIRECT_ALIGNMENT_SR = 8_000
DIRECT_ALIGNMENT_SEARCH_S = 0.5
DIRECT_ALIGNMENT_GUARD_S = 0.02
DIRECT_ALIGNMENT_OFFSET_TOLERANCE_S = 0.01

# The official GAPS test-22 population is the clean-12 development bank plus
# these ten source-disjoint videos. Eight other nominal test scores have less
# than the accepted 80% gold coverage and remain outside the test-22 metric.
SOURCE_DISJOINT_10: tuple[str, ...] = (
    "019_Vpswc",
    "111_hf1wc",
    "112_mf1wc",
    "126_XD1wc",
    "201_gk1wc",
    "222_W41wc",
    "247_sy1wc",
    "270_Jw1wc",
    "291_3Sswc",
    "358_441wc",
)
GAPS_TEST_22: tuple[str, ...] = CLEAN_12 + SOURCE_DISJOINT_10

FUSION_OVERRIDE_ENV_VARS: tuple[str, ...] = (
    "TABVISION_ASSIGNMENT_DECODER",
    "TABVISION_CHORD_MAX_GAP_S",
    "TABVISION_CHORD_SHAPE_BONUS",
    "TABVISION_CHORD_SHAPE_MIN_NOTES",
    "TABVISION_FRET_PRIOR_WEIGHT",
    "TABVISION_HAND_SPAN_BARRIER",
    "TABVISION_LOW_FRET_BIAS",
    "TABVISION_MAX_HAND_SPAN",
    "TABVISION_OPEN_STRING_BONUS",
    "TABVISION_POSITION_SHIFT_COST",
    "TABVISION_PRIOR_ALPHA",
    "TABVISION_PRIOR_POWER",
    "TABVISION_SAME_STRING_BONUS",
    "TABVISION_SPAN_NORM",
    "TABVISION_TRANSITION_GAP_TAU",
    "TABVISION_TRANSITION_PRIOR",
    "TABVISION_TRANSITION_PRIOR_WEIGHT",
)

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"
DEFAULT_VIDEO_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video"
DEFAULT_OFFSET_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain"
DEFAULT_AUDIO_CACHE = Path.home() / ".tabvision" / "cache" / "a3_fusion_sweep"
DEFAULT_PRODUCTION_TAB_CACHE = Path.home() / ".tabvision" / "cache" / "v1_1_second_corpus"
DEFAULT_RESULT_CACHE = Path.home() / ".tabvision" / "cache" / "fretcam_end_to_end"
DEFAULT_OUTPUT_JSON = (
    Path.home() / ".tabvision" / "reports" / "fretcam_end_to_end_current_solver.json"
)


@dataclass(frozen=True)
class ClipResult:
    """One paired baseline-versus-FretCam clip result."""

    stem: str
    media_duration_s: float
    offset_s: float
    offset_peak_ratio: float
    direct_alignment_offset_s: float | None
    direct_alignment_peak_ratio: float | None
    gold_notes: int
    audio_events: int
    accepted_observations: int
    affected_audio_events: int
    evaluation_runtime_s: float
    prediction_cache_sha256: str
    baseline_tab: TabF1Result
    fretcam_tab: TabF1Result
    baseline_errors: ErrorDecomposition
    fretcam_errors: ErrorDecomposition

    @property
    def delta_tab_f1(self) -> float:
        return self.fretcam_tab.f1 - self.baseline_tab.f1

    @property
    def wrong_position_delta(self) -> int:
        return (
            self.fretcam_errors.wrong_position_same_pitch
            - self.baseline_errors.wrong_position_same_pitch
        )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["delta_tab_f1"] = self.delta_tab_f1
        payload["wrong_position_delta"] = self.wrong_position_delta
        return payload


class CachedPredictionBackend:
    """Return fresh copies of one clip's real cached audio predictions."""

    def __init__(self, name: str, events: Sequence[AudioEvent]) -> None:
        self.name = name
        self._events = tuple(events)

    def transcribe(
        self,
        _wav: np.ndarray,
        _sr: int,
        _session: SessionConfig,
    ) -> list[AudioEvent]:
        return [
            replace(
                event,
                pitch_logits=(None if event.pitch_logits is None else event.pitch_logits.copy()),
                fret_prior=(None if event.fret_prior is None else event.fret_prior.copy()),
            )
            for event in self._events
        ]


class GoldClockPositionAnalyzer:
    """Shift live FretCam observations from source-video time to GAPS gold time."""

    def __init__(
        self,
        inner: FretCamPositionAnalyzer,
        *,
        video_minus_gold_offset_s: float,
    ) -> None:
        self._inner = inner
        self._offset_s = float(video_minus_gold_offset_s)

    def analyze(
        self,
        frames: Any,
        *,
        stride: int = 1,
    ) -> list[PositionWindowObservation]:
        observations = self._inner.analyze(frames, stride=stride)
        return align_observations_to_gold_clock(
            observations,
            video_minus_gold_offset_s=self._offset_s,
        )

    def analyze_all(self, frames: Any, *, stride: int = 1) -> VideoObservations:
        """Shift both evidence types onto the gold clock from one traversal."""
        bundle = self._inner.analyze_all(frames, stride=stride)
        return VideoObservations(
            windows=tuple(
                align_observations_to_gold_clock(
                    list(bundle.windows),
                    video_minus_gold_offset_s=self._offset_s,
                )
            ),
            contacts=tuple(
                replace(contact, timestamp_s=float(contact.timestamp_s) - self._offset_s)
                for contact in bundle.contacts
            ),
        )


def audio_cache_path(
    wav_path: Path,
    *,
    backend_name: str,
    cache_dir: Path,
) -> Path:
    """Return the canonical A3 raw-AudioEvent cache path."""
    key = json.dumps(
        {
            "media": str(wav_path.resolve()),
            "backend": backend_name,
            "mtime": wav_path.stat().st_mtime_ns,
        }
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return cache_dir / f"{wav_path.stem}.{digest}.json"


def load_audio_events(path: Path) -> list[AudioEvent]:
    """Load trusted, deterministic highres AudioEvents from the raw cache."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"audio cache must contain a list: {path}")
    events: list[AudioEvent] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"audio cache contains a non-object row: {path}")
        raw_tags = item.get("tags", ())
        tags = tuple(str(tag) for tag in raw_tags) if isinstance(raw_tags, list) else ()
        events.append(
            AudioEvent(
                onset_s=float(item["onset_s"]),
                offset_s=float(item["offset_s"]),
                pitch_midi=int(item["pitch_midi"]),
                velocity=float(item["velocity"]),
                confidence=float(item["confidence"]),
                tags=tags,
            )
        )
    return events


def production_tab_cache_path(
    stem: str,
    *,
    backend_name: str,
    cache_dir: Path,
) -> Path:
    """Resolve the current production-policy TabEvent cache without guessing a hash."""
    expected = {
        "backend": backend_name,
        "position_prior": POSITION_PRIOR_NAME,
        "melodic_prior": False,
        "video": False,
    }
    matches: list[Path] = []
    for path in sorted(cache_dir.glob(f"{stem}.*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("key_fields") == expected and isinstance(payload.get("events"), list):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one current production cache for {stem}, found {len(matches)}"
        )
    return matches[0]


def load_tab_events(path: Path) -> list[TabEvent]:
    """Load current production TabEvents from the resumable composite cache."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"production cache has no event list: {path}")
    return [
        TabEvent(
            onset_s=float(item["onset_s"]),
            duration_s=float(item["duration_s"]),
            string_idx=int(item["string_idx"]),
            fret=int(item["fret"]),
            pitch_midi=int(item["pitch_midi"]),
            confidence=float(item["confidence"]),
            techniques=tuple(str(tag) for tag in item.get("techniques", ())),
        )
        for item in raw_events
        if isinstance(item, Mapping)
    ]


def tab_events_to_audio_surrogate(events: Sequence[TabEvent]) -> list[AudioEvent]:
    """Remove only decoded position while preserving real detected pitch/timing."""
    return [
        AudioEvent(
            onset_s=event.onset_s,
            offset_s=event.onset_s + event.duration_s,
            pitch_midi=event.pitch_midi,
            velocity=event.confidence,
            confidence=event.confidence,
            tags=event.techniques,
        )
        for event in events
    ]


def align_observations_to_gold_clock(
    observations: Sequence[PositionWindowObservation],
    *,
    video_minus_gold_offset_s: float,
) -> list[PositionWindowObservation]:
    """Map production video/audio timestamps onto the GAPS gold/WAV clock."""
    offset = float(video_minus_gold_offset_s)
    if not math.isfinite(offset):
        raise ValueError("video-minus-gold offset must be finite")
    return [
        replace(observation, timestamp_s=float(observation.timestamp_s) - offset)
        for observation in observations
    ]


def validate_alignment(
    *,
    offset_s: float,
    peak_ratio: float,
    audio_duration_s: float,
    video_duration_s: float,
    latest_gold_onset_s: float,
    direct_offset_s: float | None = None,
    direct_peak_ratio: float | None = None,
) -> None:
    """Reject ambiguous or incomplete video/WAV alignment before scoring."""
    values = {
        "offset_s": offset_s,
        "peak_ratio": peak_ratio,
        "audio_duration_s": audio_duration_s,
        "video_duration_s": video_duration_s,
        "latest_gold_onset_s": latest_gold_onset_s,
    }
    non_finite = [name for name, value in values.items() if not math.isfinite(float(value))]
    if non_finite:
        raise ValueError("alignment values must be finite: " + ", ".join(non_finite))
    if peak_ratio < MIN_ALIGNMENT_PEAK_RATIO:
        if direct_offset_s is None or direct_peak_ratio is None:
            raise ValueError(
                f"alignment peak ratio {peak_ratio:.3f} is below "
                f"{MIN_ALIGNMENT_PEAK_RATIO:.1f} without direct-waveform corroboration"
            )
        if not math.isfinite(direct_offset_s) or not math.isfinite(direct_peak_ratio):
            raise ValueError("direct-waveform alignment values must be finite")
        if direct_peak_ratio < MIN_ALIGNMENT_PEAK_RATIO:
            raise ValueError(
                f"direct-waveform peak ratio {direct_peak_ratio:.3f} is below "
                f"{MIN_ALIGNMENT_PEAK_RATIO:.1f}"
            )
        offset_delta = abs(offset_s - direct_offset_s)
        if offset_delta > DIRECT_ALIGNMENT_OFFSET_TOLERANCE_S:
            raise ValueError(
                f"alignment methods differ by {offset_delta:.3f}s, above "
                f"{DIRECT_ALIGNMENT_OFFSET_TOLERANCE_S:.2f}s"
            )
    if audio_duration_s <= 0.0 or video_duration_s <= 0.0:
        raise ValueError("alignment media durations must be positive")
    duration_delta = abs(audio_duration_s - video_duration_s)
    if duration_delta > MAX_DURATION_MISMATCH_S:
        raise ValueError(
            f"aligned media durations differ by {duration_delta:.3f}s, above "
            f"{MAX_DURATION_MISMATCH_S:.1f}s"
        )
    latest_video_onset_s = latest_gold_onset_s + offset_s
    if latest_video_onset_s < -ALIGNMENT_COVERAGE_SLACK_S or (
        latest_video_onset_s > video_duration_s + ALIGNMENT_COVERAGE_SLACK_S
    ):
        raise ValueError(
            f"latest aligned gold onset {latest_video_onset_s:.3f}s is outside "
            f"the {video_duration_s:.3f}s video"
        )


def direct_waveform_alignment(audio_path: Path, video_path: Path) -> tuple[float, float]:
    """Corroborate a weak onset-envelope offset using raw waveform correlation."""
    from scipy import signal

    audio = _decode_mono(audio_path, DIRECT_ALIGNMENT_SR)
    video = _decode_mono(video_path, DIRECT_ALIGNMENT_SR)
    audio = (audio - audio.mean()) / (audio.std() + 1e-12)
    video = (video - video.mean()) / (video.std() + 1e-12)
    correlation = signal.correlate(video, audio, mode="full", method="fft")
    lags = signal.correlation_lags(video.size, audio.size, mode="full")
    search_samples = round(DIRECT_ALIGNMENT_SEARCH_S * DIRECT_ALIGNMENT_SR)
    search_mask = np.abs(lags) <= search_samples
    local_correlation = correlation[search_mask]
    local_lags = lags[search_mask]
    if local_correlation.size == 0:
        raise ValueError("direct-waveform alignment search window is empty")
    best_index = int(np.argmax(local_correlation))
    best_lag = int(local_lags[best_index])
    best_value = float(local_correlation[best_index])
    guard_samples = round(DIRECT_ALIGNMENT_GUARD_S * DIRECT_ALIGNMENT_SR)
    competing = np.abs(local_lags - best_lag) >= guard_samples
    if not competing.any():
        raise ValueError("direct-waveform alignment has no competing lags")
    next_best = float(np.max(local_correlation[competing]))
    peak_ratio = best_value / (next_best + 1e-12)
    return best_lag / DIRECT_ALIGNMENT_SR, peak_ratio


def _decode_mono(path: Path, sample_rate: int) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        message = process.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg failed to decode {path}: {message}")
    return np.frombuffer(process.stdout, dtype=np.float32).astype(np.float64)


def micro_tab_f1(results: Sequence[TabF1Result]) -> TabF1Result:
    """Sum per-clip confusion counts without cross-clip onset collisions."""
    tp = sum(result.true_positives for result in results)
    fp = sum(result.false_positives for result in results)
    fn = sum(result.false_negatives for result in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return TabF1Result(precision, recall, f1, tp, fp, fn)


def aggregate_results(
    clips: Sequence[ClipResult],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Aggregate paired clip metrics with a clip-stratified bootstrap."""
    if not clips:
        raise ValueError("at least one clip result is required")
    baseline_values = [clip.baseline_tab.f1 for clip in clips]
    fretcam_values = [clip.fretcam_tab.f1 for clip in clips]
    deltas = [clip.delta_tab_f1 for clip in clips]
    baseline_errors = aggregate_decompositions(clip.baseline_errors for clip in clips)
    fretcam_errors = aggregate_decompositions(clip.fretcam_errors for clip in clips)
    baseline_wrong = baseline_errors.wrong_position_same_pitch
    wrong_reduction = baseline_wrong - fretcam_errors.wrong_position_same_pitch
    return {
        "clips": len(clips),
        "media_duration_s": sum(clip.media_duration_s for clip in clips),
        "evaluation_runtime_s": sum(clip.evaluation_runtime_s for clip in clips),
        "gold_notes": sum(clip.gold_notes for clip in clips),
        "audio_events": sum(clip.audio_events for clip in clips),
        "accepted_observations": sum(clip.accepted_observations for clip in clips),
        "affected_audio_events": sum(clip.affected_audio_events for clip in clips),
        "direct_alignment_checks": sum(
            clip.direct_alignment_offset_s is not None for clip in clips
        ),
        "baseline_macro": asdict(
            bootstrap_ci(
                baseline_values,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        ),
        "fretcam_macro": asdict(
            bootstrap_ci(
                fretcam_values,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        ),
        "paired_delta": asdict(
            bootstrap_ci(
                deltas,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
        ),
        "baseline_micro": asdict(micro_tab_f1([clip.baseline_tab for clip in clips])),
        "fretcam_micro": asdict(micro_tab_f1([clip.fretcam_tab for clip in clips])),
        "baseline_errors": asdict(baseline_errors),
        "fretcam_errors": asdict(fretcam_errors),
        "wrong_position_reduction": wrong_reduction,
        "wrong_position_relative_reduction": (
            wrong_reduction / baseline_wrong if baseline_wrong else 0.0
        ),
        "regressed_clips": [clip.stem for clip in clips if clip.delta_tab_f1 < 0.0],
        "improved_clips": [clip.stem for clip in clips if clip.delta_tab_f1 > 0.0],
        "unchanged_clips": [clip.stem for clip in clips if clip.delta_tab_f1 == 0.0],
    }


def format_report(payload: Mapping[str, object]) -> str:
    """Render a concise, auditable Markdown result."""
    aggregate = _require_mapping(payload["aggregate"])
    baseline_macro = _bootstrap_from_mapping(_require_mapping(aggregate["baseline_macro"]))
    fretcam_macro = _bootstrap_from_mapping(_require_mapping(aggregate["fretcam_macro"]))
    paired_delta = _bootstrap_from_mapping(_require_mapping(aggregate["paired_delta"]))
    baseline_micro = _tab_f1_from_mapping(_require_mapping(aggregate["baseline_micro"]))
    fretcam_micro = _tab_f1_from_mapping(_require_mapping(aggregate["fretcam_micro"]))
    baseline_errors = _errors_from_mapping(_require_mapping(aggregate["baseline_errors"]))
    fretcam_errors = _errors_from_mapping(_require_mapping(aggregate["fretcam_errors"]))
    wrong_reduction = int(aggregate["wrong_position_reduction"])
    if wrong_reduction > 0:
        wrong_change = f"{wrong_reduction:,} fewer"
    elif wrong_reduction < 0:
        wrong_change = f"{abs(wrong_reduction):,} more"
    else:
        wrong_change = "unchanged"

    lines = [
        "# FretCam current-solver paired end-to-end Tab F1",
        "",
        f"**Population:** {payload['population_label']}; **as of:** {payload['generated_at']}.",
        "",
        "## Result",
        "",
        "| Metric | Audio baseline | + current FretCam | Delta |",
        "|---|---:|---:|---:|",
        f"| Macro per-clip Tab F1 | {baseline_macro.statistic:.6f} | "
        f"{fretcam_macro.statistic:.6f} | {paired_delta.statistic:+.6f} |",
        f"| Macro lower-95 | {baseline_macro.lower:.6f} | {fretcam_macro.lower:.6f} | — |",
        f"| Micro Tab F1 | {baseline_micro.f1:.6f} | "
        f"{fretcam_micro.f1:.6f} | {fretcam_micro.f1 - baseline_micro.f1:+.6f} |",
        f"| Wrong-position/same-pitch | "
        f"{baseline_errors.wrong_position_same_pitch:,} | "
        f"{fretcam_errors.wrong_position_same_pitch:,} | "
        f"{wrong_change} |",
        "",
        f"Paired 95% bootstrap interval for the macro delta: "
        f"`[{paired_delta.lower:+.6f}, {paired_delta.upper:+.6f}]`.",
        "",
        "## Per clip",
        "",
        "| Clip | Baseline | + FretCam | Delta | Wrong-pos Δ | Obs | "
        "Events affected | Paired runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    raw_clips = payload["clips"]
    if not isinstance(raw_clips, list):
        raise TypeError("report payload clips must be a list")
    for raw_clip in raw_clips:
        clip = _clip_from_mapping(_require_mapping(raw_clip))
        lines.append(
            f"| `{clip.stem}` | {clip.baseline_tab.f1:.6f} | "
            f"{clip.fretcam_tab.f1:.6f} | {clip.delta_tab_f1:+.6f} | "
            f"{clip.wrong_position_delta:+d} | {clip.accepted_observations:,} | "
            f"{clip.affected_audio_events:,} | {clip.evaluation_runtime_s:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"- Audio prediction input: {payload['audio_input']}.",
            "- FretCam `DetectionChain` + `PositionEstimator` runs live over each "
            "source MP4 using production demux timestamps and stride "
            f"`{payload['video_stride']}`.",
            "- Cached cross-correlation offsets map FretCam observations from video "
            "time to the GAPS WAV/gold clock.",
            f"- Alignment requires onset-envelope peak ratio >= "
            f"`{MIN_ALIGNMENT_PEAK_RATIO:.1f}`; weaker peaks require an agreeing "
            "raw-waveform offset with the same peak-ratio floor.",
            "- Both arms use the current clean-classical automatic policy: "
            f"`{payload['position_prior']}` + `{payload['sequence_prior']}` at "
            f"weight `{SEQUENCE_PRIOR_WEIGHT:.1f}`, assignment decoder "
            f"`{payload['assignment_decoder']}`.",
            "- Canonical Tab F1: exact string + fret and onset within 50 ms; "
            "macro mean and clip-stratified bootstrap use 10,000 resamples, seed 42.",
            "- No gold pitch/string, cached CV anchor, policy tuning, download, or "
            "training enters prediction.",
            "",
            "## Runtime and coverage",
            "",
            f"- Clips: `{aggregate['clips']}`; media: "
            f"`{float(aggregate['media_duration_s']) / 60.0:.2f} min`; "
            f"paired pipeline runtime: "
            f"`{float(aggregate['evaluation_runtime_s']) / 60.0:.2f} min`.",
            f"- Accepted observations: `{aggregate['accepted_observations']}`; "
            f"audio events affected: `{aggregate['affected_audio_events']}`.",
            f"- Direct waveform alignment checks required: "
            f"`{aggregate['direct_alignment_checks']}`.",
            f"- Improved / unchanged / regressed clips: "
            f"`{len(aggregate['improved_clips'])}` / "
            f"`{len(aggregate['unchanged_clips'])}` / "
            f"`{len(aggregate['regressed_clips'])}`.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_clip(
    stem: str,
    *,
    data_root: Path,
    video_cache: Path,
    offset_cache: Path,
    audio_cache: Path,
    production_tab_cache: Path,
    result_cache: Path,
    backend_name: str,
    audio_source: str,
    video_stride: int,
    vision_weight: float,
    contact_evidence: bool,
    cfg: GuitarConfig,
    session: SessionConfig,
    policy: ResolvedInferencePolicy,
    analyzer: FretCamPositionAnalyzer,
    refresh_video: bool,
) -> ClipResult:
    """Run or resume one paired clip."""
    gaps_root = data_root / "gaps"
    xml_path = gaps_root / "musicxml" / f"{stem}.xml"
    wav_path = gaps_root / "audio" / f"{stem}.wav"
    video_path = video_cache / f"{stem}.mp4"
    offset_path = offset_cache / f"{stem}.offset.pkl"
    source_tabs: list[TabEvent] | None = None
    if audio_source == "production-tab-cache":
        prediction_cache_path = production_tab_cache_path(
            stem,
            backend_name=backend_name,
            cache_dir=production_tab_cache,
        )
    elif audio_source == "raw-audio-cache":
        prediction_cache_path = audio_cache_path(
            wav_path,
            backend_name=backend_name,
            cache_dir=audio_cache,
        )
    else:  # pragma: no cover - argparse and run_evaluation guard this
        raise ValueError(f"unsupported audio source: {audio_source}")
    for path in (xml_path, wav_path, video_path, offset_path, prediction_cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    offset_record = _load_trusted_pickle(offset_path)
    offset_s = float(offset_record.offset_s)
    peak_ratio = float(offset_record.peak_ratio)
    gold = parse_gaps(xml_path, cfg)
    direct_offset_s: float | None = None
    direct_peak_ratio: float | None = None
    if peak_ratio < MIN_ALIGNMENT_PEAK_RATIO:
        direct_offset_s, direct_peak_ratio = direct_waveform_alignment(wav_path, video_path)
    validate_alignment(
        offset_s=offset_s,
        peak_ratio=peak_ratio,
        audio_duration_s=float(offset_record.audio_duration_s),
        video_duration_s=float(offset_record.video_duration_s),
        latest_gold_onset_s=max((event.onset_s for event in gold), default=0.0),
        direct_offset_s=direct_offset_s,
        direct_peak_ratio=direct_peak_ratio,
    )
    signature = _clip_signature(
        stem=stem,
        files=(xml_path, wav_path, video_path, offset_path, prediction_cache_path),
        policy=policy,
        audio_source=audio_source,
        video_stride=video_stride,
        vision_weight=vision_weight,
        contact_evidence=contact_evidence,
        offset_s=offset_s,
    )
    cache_path = result_cache / f"{stem}.{signature[:16]}.json"
    if not refresh_video and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature:
            result = _clip_from_mapping(_require_mapping(cached["result"]))
            print(
                f"[cache] {stem}: {result.baseline_tab.f1:.6f} -> {result.fretcam_tab.f1:.6f}",
                flush=True,
            )
            return result

    started = time.monotonic()
    if audio_source == "production-tab-cache":
        source_tabs = load_tab_events(prediction_cache_path)
        raw_audio_events = tab_events_to_audio_surrogate(source_tabs)
    else:
        raw_audio_events = load_audio_events(prediction_cache_path)
    baseline_artifacts = run_pipeline_with_artifacts(
        video_path,
        audio_backend=CachedPredictionBackend(backend_name, raw_audio_events),
        audio_backend_name=backend_name,
        lambda_vision=vision_weight,
        video_stride=video_stride,
        video_enabled=False,
        video_backend="legacy",
        position_prior="auto",
        sequence_prior="auto",
        string_evidence="auto",
        assignment_decoder="baseline",
        melodic_prior_enabled=False,
        cfg=cfg,
        session=session,
    )
    fretcam_artifacts = run_pipeline_with_artifacts(
        video_path,
        audio_backend=CachedPredictionBackend(backend_name, raw_audio_events),
        audio_backend_name=backend_name,
        position_analyzer=GoldClockPositionAnalyzer(
            analyzer,
            video_minus_gold_offset_s=offset_s,
        ),
        lambda_vision=vision_weight,
        video_stride=video_stride,
        video_enabled=True,
        video_backend="fretcam",
        contact_evidence=contact_evidence,
        position_prior="auto",
        sequence_prior="auto",
        string_evidence="auto",
        assignment_decoder="baseline",
        melodic_prior_enabled=False,
        cfg=cfg,
        session=session,
    )
    if fretcam_artifacts.resolved_video_backend != "fretcam":
        raise RuntimeError(
            "candidate arm did not resolve the FretCam video backend: "
            f"{fretcam_artifacts.resolved_video_backend}"
        )
    for arm_name, arm_policy in (
        ("baseline", baseline_artifacts.policy),
        ("fretcam", fretcam_artifacts.policy),
    ):
        if (
            arm_policy.resolved_position_prior != policy.resolved_position_prior
            or arm_policy.resolved_sequence_prior != policy.resolved_sequence_prior
            or arm_policy.resolved_assignment_decoder != policy.resolved_assignment_decoder
        ):
            raise RuntimeError(f"{arm_name} arm resolved a different inference policy")

    baseline_tabs = list(baseline_artifacts.tab_events)
    fretcam_tabs = list(fretcam_artifacts.tab_events)
    if len(baseline_tabs) != len(fretcam_tabs):
        raise RuntimeError("position-only bridge changed the decoded event count")
    if _tab_event_stream_signature(baseline_tabs) != _tab_event_stream_signature(fretcam_tabs):
        raise RuntimeError("position-only bridge changed pitch, timing, or duration")
    if source_tabs is not None and _tab_assignment_signature(baseline_tabs) != (
        _tab_assignment_signature(source_tabs)
    ):
        raise RuntimeError(
            "position-stripped production cache did not reproduce its baseline assignments"
        )

    baseline_score = tab_f1(
        baseline_tabs,
        gold,
        onset_tolerance_s=ONSET_TOLERANCE_S,
    )
    fretcam_score = tab_f1(
        fretcam_tabs,
        gold,
        onset_tolerance_s=ONSET_TOLERANCE_S,
    )
    if baseline_score.total_gold != fretcam_score.total_gold:
        raise RuntimeError("paired arms have different gold denominators")
    result = ClipResult(
        stem=stem,
        media_duration_s=float(offset_record.video_duration_s),
        offset_s=offset_s,
        offset_peak_ratio=peak_ratio,
        direct_alignment_offset_s=direct_offset_s,
        direct_alignment_peak_ratio=direct_peak_ratio,
        gold_notes=len(gold),
        audio_events=len(raw_audio_events),
        accepted_observations=fretcam_artifacts.position_observation_count,
        affected_audio_events=fretcam_artifacts.notes_affected_by_video,
        evaluation_runtime_s=time.monotonic() - started,
        prediction_cache_sha256=_sha256(prediction_cache_path),
        baseline_tab=baseline_score,
        fretcam_tab=fretcam_score,
        baseline_errors=decompose_errors(
            baseline_tabs,
            gold,
            onset_tolerance_s=ONSET_TOLERANCE_S,
        ),
        fretcam_errors=decompose_errors(
            fretcam_tabs,
            gold,
            onset_tolerance_s=ONSET_TOLERANCE_S,
        ),
    )
    result_cache.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "signature": signature,
                "result": result.as_dict(),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[run]   {stem}: {result.baseline_tab.f1:.6f} -> "
        f"{result.fretcam_tab.f1:.6f} ({result.delta_tab_f1:+.6f}); "
        f"{result.accepted_observations} obs, "
        f"{result.affected_audio_events} affected, "
        f"{result.evaluation_runtime_s:.1f}s",
        flush=True,
    )
    return result


def run_evaluation(
    stems: Sequence[str],
    *,
    population_label: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    video_cache: Path = DEFAULT_VIDEO_CACHE,
    offset_cache: Path = DEFAULT_OFFSET_CACHE,
    audio_cache: Path = DEFAULT_AUDIO_CACHE,
    production_tab_cache: Path = DEFAULT_PRODUCTION_TAB_CACHE,
    result_cache: Path = DEFAULT_RESULT_CACHE,
    backend_name: str = "highres",
    audio_source: str = "production-tab-cache",
    video_stride: int = 3,
    vision_weight: float = 1.0,
    contact_evidence: bool = False,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 42,
    refresh_video: bool = False,
) -> dict[str, object]:
    """Run a frozen paired evaluation and return its serializable payload."""
    active_overrides = [name for name in FUSION_OVERRIDE_ENV_VARS if name in os.environ]
    if active_overrides:
        raise RuntimeError(
            "paired evaluation requires default fusion settings; unset: "
            + ", ".join(active_overrides)
        )
    if isinstance(video_stride, bool) or not isinstance(video_stride, int) or video_stride < 1:
        raise ValueError("video_stride must be a positive integer")
    if not math.isfinite(float(vision_weight)) or float(vision_weight) < 0.0:
        raise ValueError("vision_weight must be finite and non-negative")
    if not stems:
        raise ValueError("at least one clip is required")
    if audio_source not in {"production-tab-cache", "raw-audio-cache"}:
        raise ValueError("audio_source must be production-tab-cache or raw-audio-cache")

    os.environ.setdefault("YOLO_OFFLINE", "1")
    cfg = GuitarConfig()
    session = SessionConfig(instrument="classical", tone="clean", style="mixed")
    policy = resolve_inference_policy(
        requested_position_prior="auto",
        requested_sequence_prior="auto",
        requested_string_evidence="auto",
        requested_assignment_decoder=None,
        cfg=cfg,
        session=session,
        audio_backend_name=backend_name,
    )
    if (
        policy.resolved_position_prior != POSITION_PRIOR_NAME
        or policy.resolved_sequence_prior != SEQUENCE_PRIOR_NAME
        or policy.resolved_assignment_decoder != "baseline"
    ):
        raise RuntimeError(
            "clean-classical production policy drifted: "
            f"{policy.resolved_position_prior}/"
            f"{policy.resolved_sequence_prior}/"
            f"{policy.resolved_assignment_decoder}"
        )

    analyzer = FretCamPositionAnalyzer(cfg)
    clips = [
        evaluate_clip(
            stem,
            data_root=data_root,
            video_cache=video_cache,
            offset_cache=offset_cache,
            audio_cache=audio_cache,
            production_tab_cache=production_tab_cache,
            result_cache=result_cache,
            backend_name=backend_name,
            audio_source=audio_source,
            video_stride=video_stride,
            vision_weight=float(vision_weight),
            contact_evidence=contact_evidence,
            cfg=cfg,
            session=session,
            policy=policy,
            analyzer=analyzer,
            refresh_video=refresh_video,
        )
        for stem in stems
    ]
    aggregate = aggregate_results(
        clips,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population_label": population_label,
        "clips_requested": list(stems),
        "backend": backend_name,
        "audio_input": (
            "current production TabEvent cache stripped only of position; "
            "baseline assignment reproduction required"
            if audio_source == "production-tab-cache"
            else "cached real highres AudioEvents from canonical GAPS WAV"
        ),
        "video_input": "live current FretCam inference from aligned source MP4",
        "position_prior": policy.resolved_position_prior,
        "sequence_prior": policy.resolved_sequence_prior,
        "sequence_prior_weight": SEQUENCE_PRIOR_WEIGHT,
        "assignment_decoder": policy.resolved_assignment_decoder,
        "video_stride": video_stride,
        "vision_weight": float(vision_weight),
        "contact_evidence": bool(contact_evidence),
        "onset_tolerance_s": ONSET_TOLERANCE_S,
        "bootstrap_n": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "clips": [clip.as_dict() for clip in clips],
        "aggregate": aggregate,
    }


def _clip_signature(
    *,
    stem: str,
    files: Sequence[Path],
    policy: ResolvedInferencePolicy,
    audio_source: str,
    video_stride: int,
    vision_weight: float,
    contact_evidence: bool,
    offset_s: float,
) -> str:
    code_paths = (
        Path(__file__),
        Path(inspect.getsourcefile(FretCamPositionAnalyzer) or ""),
        Path(inspect.getsourcefile(apply_position_window_priors) or ""),
        Path(inspect.getsourcefile(apply_contact_priors) or ""),
        Path(inspect.getsourcefile(fuse) or ""),
        Path(inspect.getsourcefile(run_pipeline_with_artifacts) or ""),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stem": stem,
        "inputs": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": _sha256(path) if path.suffix in {".json", ".xml", ".pkl"} else None,
            }
            for path in files
        ],
        "code": {str(path.resolve()): _sha256(path) for path in code_paths},
        "policy": {
            "position": policy.resolved_position_prior,
            "sequence": policy.resolved_sequence_prior,
            "decoder": policy.resolved_assignment_decoder,
            "artifacts": [asdict(identity) for identity in policy.artifacts],
        },
        "audio_source": audio_source,
        "video_stride": video_stride,
        "vision_weight": vision_weight,
        "contact_evidence": bool(contact_evidence),
        "offset_s": offset_s,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _clip_from_mapping(payload: Mapping[str, Any]) -> ClipResult:
    return ClipResult(
        stem=str(payload["stem"]),
        media_duration_s=float(payload["media_duration_s"]),
        offset_s=float(payload["offset_s"]),
        offset_peak_ratio=float(payload["offset_peak_ratio"]),
        direct_alignment_offset_s=(
            None
            if payload.get("direct_alignment_offset_s") is None
            else float(payload["direct_alignment_offset_s"])
        ),
        direct_alignment_peak_ratio=(
            None
            if payload.get("direct_alignment_peak_ratio") is None
            else float(payload["direct_alignment_peak_ratio"])
        ),
        gold_notes=int(payload["gold_notes"]),
        audio_events=int(payload["audio_events"]),
        accepted_observations=int(payload["accepted_observations"]),
        affected_audio_events=int(payload["affected_audio_events"]),
        evaluation_runtime_s=float(payload["evaluation_runtime_s"]),
        prediction_cache_sha256=str(payload["prediction_cache_sha256"]),
        baseline_tab=_tab_f1_from_mapping(_require_mapping(payload["baseline_tab"])),
        fretcam_tab=_tab_f1_from_mapping(_require_mapping(payload["fretcam_tab"])),
        baseline_errors=_errors_from_mapping(_require_mapping(payload["baseline_errors"])),
        fretcam_errors=_errors_from_mapping(_require_mapping(payload["fretcam_errors"])),
    )


def _tab_f1_from_mapping(payload: Mapping[str, Any]) -> TabF1Result:
    return TabF1Result(
        precision=float(payload["precision"]),
        recall=float(payload["recall"]),
        f1=float(payload["f1"]),
        true_positives=int(payload["true_positives"]),
        false_positives=int(payload["false_positives"]),
        false_negatives=int(payload["false_negatives"]),
    )


def _errors_from_mapping(payload: Mapping[str, Any]) -> ErrorDecomposition:
    return ErrorDecomposition(
        correct=int(payload["correct"]),
        wrong_position_same_pitch=int(payload["wrong_position_same_pitch"]),
        pitch_off=int(payload["pitch_off"]),
        timing_only=int(payload["timing_only"]),
        missed_onset=int(payload["missed_onset"]),
        extra_detection=int(payload["extra_detection"]),
        pitch_off_deltas=tuple(int(value) for value in payload.get("pitch_off_deltas", ())),
    )


def _bootstrap_from_mapping(payload: Mapping[str, Any]) -> BootstrapResult:
    return BootstrapResult(
        statistic=float(payload["statistic"]),
        lower=float(payload["lower"]),
        upper=float(payload["upper"]),
        n_observations=int(payload["n_observations"]),
        n_bootstrap=int(payload["n_bootstrap"]),
        confidence=float(payload["confidence"]),
    )


def _require_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value


def _load_trusted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted locally generated alignment cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tab_assignment_signature(events: Sequence[TabEvent]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            event.onset_s,
            event.duration_s,
            event.string_idx,
            event.fret,
            event.pitch_midi,
        )
        for event in events
    )


def _tab_event_stream_signature(
    events: Sequence[TabEvent],
) -> tuple[tuple[object, ...], ...]:
    """Pitch/timing identity that a position-only bridge must preserve."""
    return tuple(
        (
            event.onset_s,
            event.duration_s,
            event.pitch_midi,
        )
        for event in events
    )


def _resolve_stems(spec: str) -> tuple[str, ...]:
    normalized = spec.strip().lower()
    if normalized == "clean12":
        return CLEAN_12
    if normalized in {"source-disjoint10", "heldout10"}:
        return SOURCE_DISJOINT_10
    if normalized == "test22":
        return GAPS_TEST_22
    stems = tuple(item.strip() for item in spec.split(",") if item.strip())
    if not stems:
        raise ValueError("--clips must select at least one stem")
    return stems


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips",
        default="clean12",
        help="'clean12', 'source-disjoint10', 'test22', or comma-separated stems",
    )
    parser.add_argument("--population-label", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    parser.add_argument("--offset-cache", type=Path, default=DEFAULT_OFFSET_CACHE)
    parser.add_argument("--audio-cache", type=Path, default=DEFAULT_AUDIO_CACHE)
    parser.add_argument(
        "--production-tab-cache",
        type=Path,
        default=DEFAULT_PRODUCTION_TAB_CACHE,
    )
    parser.add_argument("--result-cache", type=Path, default=DEFAULT_RESULT_CACHE)
    parser.add_argument("--backend", default="highres")
    parser.add_argument(
        "--audio-source",
        choices=["production-tab-cache", "raw-audio-cache"],
        default="production-tab-cache",
    )
    parser.add_argument("--video-stride", type=int, default=3)
    parser.add_argument("--vision-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--contact-evidence",
        action="store_true",
        help="also apply FretCam per-finger (string, fret) contacts as a capped prior",
    )
    parser.add_argument("--refresh-video", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stems = _resolve_stems(args.clips)
    label = args.population_label or args.clips
    payload = run_evaluation(
        stems,
        population_label=label,
        data_root=args.data_root,
        video_cache=args.video_cache,
        offset_cache=args.offset_cache,
        audio_cache=args.audio_cache,
        production_tab_cache=args.production_tab_cache,
        result_cache=args.result_cache,
        backend_name=args.backend,
        audio_source=args.audio_source,
        video_stride=args.video_stride,
        vision_weight=args.vision_weight,
        contact_evidence=args.contact_evidence,
        n_bootstrap=args.bootstrap_n,
        bootstrap_seed=args.bootstrap_seed,
        refresh_video=args.refresh_video,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"JSON: {args.output_json}", flush=True)
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(format_report(payload), encoding="utf-8")
        print(f"Markdown: {args.output_report}", flush=True)

    aggregate = _require_mapping(payload["aggregate"])
    baseline = _bootstrap_from_mapping(_require_mapping(aggregate["baseline_macro"]))
    fretcam = _bootstrap_from_mapping(_require_mapping(aggregate["fretcam_macro"]))
    delta = _bootstrap_from_mapping(_require_mapping(aggregate["paired_delta"]))
    print(
        f"macro Tab F1 {baseline.statistic:.6f} -> {fretcam.statistic:.6f}; "
        f"delta {delta.statistic:+.6f} "
        f"[{delta.lower:+.6f}, {delta.upper:+.6f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
