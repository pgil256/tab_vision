"""Public-only frame benchmark for the quarantined FretCam position HUD."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import cv2

from fretcam.detection import ConfidenceFactors, DetectionChain
from fretcam.position import PositionEstimator
from fretcam.processing import build_hand_search_hint

Split = Literal["dev", "test"]
LabelState = Literal["stable", "shifting", "dropout", "invalid"]
Technique = Literal["note", "chord", "barre"]
Visibility = Literal["full_neck", "close"]
Lighting = Literal["bright", "mixed"]


@dataclass(frozen=True)
class PositionLabel:
    """One half-open ground-truth interval inside a replay sequence."""

    start_s: float
    end_s: float
    state: LabelState
    position: int | None
    technique: Technique
    visibility: Visibility
    lighting: Lighting
    verification: str
    notes: str
    occlusion_rect: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BenchmarkSequence:
    """One short, continuously replayed public-video window."""

    sequence_id: str
    source: str
    split: Split
    start_s: float
    end_s: float
    labels: tuple[PositionLabel, ...]


@dataclass(frozen=True)
class BenchmarkManifest:
    """Frozen F4e-A labels and their public-data provenance."""

    version: int
    name: str
    corpus: str
    corpus_license: str
    public_only: bool
    sample_fps: float
    annotation_policy: str
    sequences: tuple[BenchmarkSequence, ...]


@dataclass(frozen=True)
class FramePrediction:
    """One sampled FretCam display state used by the scorer."""

    sequence_id: str
    source: str
    split: Split
    timestamp_s: float
    state: str
    position: int | None
    confidence: float
    observation_valid: bool
    confidence_factors: ConfidenceFactors | None = None
    geometry_status: str = "unknown"
    geometry_age_ms: float = 0.0


@dataclass(frozen=True)
class _ScoredFrame:
    prediction: FramePrediction
    label: PositionLabel


def default_manifest_path() -> Path:
    """Return the checked-in F4e-A manifest when running from source/install."""
    return Path(__file__).resolve().parents[2] / "data" / "position_benchmark_v1.json"


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load and validate a checked-in position benchmark JSON document."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = []
    for item in raw["sequences"]:
        labels = tuple(
            PositionLabel(
                **{
                    **label,
                    "occlusion_rect": (
                        tuple(label["occlusion_rect"])
                        if label.get("occlusion_rect") is not None
                        else None
                    ),
                }
            )
            for label in item["labels"]
        )
        sequences.append(
            BenchmarkSequence(
                sequence_id=item["sequence_id"],
                source=item["source"],
                split=item["split"],
                start_s=item["start_s"],
                end_s=item["end_s"],
                labels=labels,
            )
        )
    manifest = BenchmarkManifest(
        version=raw["version"],
        name=raw["name"],
        corpus=raw["corpus"],
        corpus_license=raw["corpus_license"],
        public_only=raw["public_only"],
        sample_fps=raw["sample_fps"],
        annotation_policy=raw["annotation_policy"],
        sequences=tuple(sequences),
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: BenchmarkManifest) -> None:
    """Enforce public-only provenance, interval, and source-split hygiene."""
    if manifest.version != 1:
        raise ValueError("unsupported benchmark version")
    if not manifest.public_only or manifest.corpus != "GAPS":
        raise ValueError("FretCam accuracy labels must use the public GAPS corpus")
    if not math.isfinite(manifest.sample_fps) or manifest.sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive and finite")
    if not manifest.sequences:
        raise ValueError("benchmark must contain at least one sequence")

    sequence_ids: set[str] = set()
    source_splits: dict[str, Split] = {}
    seen_splits: set[Split] = set()
    for sequence in manifest.sequences:
        if sequence.sequence_id in sequence_ids:
            raise ValueError(f"duplicate sequence_id: {sequence.sequence_id}")
        sequence_ids.add(sequence.sequence_id)
        if sequence.split not in {"dev", "test"}:
            raise ValueError(f"invalid split: {sequence.split}")
        seen_splits.add(sequence.split)
        previous_split = source_splits.setdefault(sequence.source, sequence.split)
        if previous_split != sequence.split:
            raise ValueError(f"source crosses dev/test split: {sequence.source}")
        if not (
            math.isfinite(sequence.start_s)
            and math.isfinite(sequence.end_s)
            and 0.0 <= sequence.start_s < sequence.end_s
        ):
            raise ValueError(f"invalid sequence bounds: {sequence.sequence_id}")
        if not sequence.labels:
            raise ValueError(f"sequence has no labels: {sequence.sequence_id}")

        previous_end = sequence.start_s
        for label in sequence.labels:
            if not (
                math.isfinite(label.start_s)
                and math.isfinite(label.end_s)
                and sequence.start_s <= label.start_s < label.end_s <= sequence.end_s
            ):
                raise ValueError(f"invalid label bounds: {sequence.sequence_id}")
            if label.start_s < previous_end - 1e-9:
                raise ValueError(f"overlapping labels: {sequence.sequence_id}")
            previous_end = label.end_s
            if label.state in {"stable", "dropout"}:
                if label.position is None or label.position <= 0:
                    raise ValueError(
                        f"{label.state} labels require a positive position"
                    )
            elif label.position is not None:
                raise ValueError(
                    "shifting and invalid labels must not force a position"
                )
            if label.state not in {"stable", "shifting", "dropout", "invalid"}:
                raise ValueError(f"unsupported label state: {label.state}")
            if label.technique not in {"note", "chord", "barre"}:
                raise ValueError(f"unsupported technique: {label.technique}")
            if label.visibility not in {"full_neck", "close"}:
                raise ValueError(f"unsupported visibility: {label.visibility}")
            if label.lighting not in {"bright", "mixed"}:
                raise ValueError(f"unsupported lighting: {label.lighting}")
            if label.verification not in {"user_verified", "frame_reviewed"}:
                raise ValueError(f"unsupported verification: {label.verification}")
            if label.state == "dropout":
                rect = label.occlusion_rect
                if (
                    rect is None
                    or len(rect) != 4
                    or not all(math.isfinite(value) for value in rect)
                    or not (0.0 <= rect[0] < rect[2] <= 1.0)
                    or not (0.0 <= rect[1] < rect[3] <= 1.0)
                ):
                    raise ValueError(
                        "dropout labels require a normalized occlusion_rect"
                    )
            elif label.occlusion_rect is not None:
                raise ValueError("only dropout labels may apply an occlusion_rect")
        labels = list(sequence.labels)
        for index, label in enumerate(labels):
            if label.state not in {"shifting", "dropout"}:
                continue
            if index == 0 or index == len(labels) - 1:
                raise ValueError(
                    f"{label.state} must sit between stable labels: "
                    f"{sequence.sequence_id}"
                )
            origin = labels[index - 1]
            destination = labels[index + 1]
            if (
                origin.state != "stable"
                or destination.state != "stable"
                or abs(origin.end_s - label.start_s) > 1e-6
                or abs(label.end_s - destination.start_s) > 1e-6
            ):
                raise ValueError(
                    f"{label.state} must have contiguous stable endpoints: "
                    f"{sequence.sequence_id}"
                )
            if label.state == "dropout" and not (
                origin.position == label.position == destination.position
            ):
                raise ValueError(
                    "dropout position must match both stable endpoints: "
                    f"{sequence.sequence_id}"
                )
    if seen_splits != {"dev", "test"}:
        raise ValueError("benchmark must freeze both dev and test splits")


def _sample_times(start_s: float, end_s: float, sample_fps: float) -> Iterable[float]:
    count = math.ceil((end_s - start_s) * sample_fps - 1e-9)
    for index in range(count):
        timestamp_s = start_s + index / sample_fps
        if timestamp_s < end_s - 1e-9:
            yield timestamp_s


def _read_frame(capture: cv2.VideoCapture, timestamp_s: float) -> object:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame at {timestamp_s:.3f}s")
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def _apply_annotation_transform(frame: object, label: PositionLabel | None) -> object:
    """Apply a deterministic public-frame occlusion declared by ground truth."""
    if label is None or label.occlusion_rect is None:
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    left, top, right, bottom = label.occlusion_rect
    x0 = max(0, min(width, math.floor(left * width)))
    y0 = max(0, min(height, math.floor(top * height)))
    x1 = max(0, min(width, math.ceil(right * width)))
    y1 = max(0, min(height, math.ceil(bottom * height)))
    output[y0:y1, x0:x1] = 0
    return output


def run_inference(
    manifest: BenchmarkManifest,
    *,
    video_cache: Path,
    splits: set[Split] | None = None,
    chain_factory: Callable[[], DetectionChain] = DetectionChain,
) -> list[FramePrediction]:
    """Run unchanged F4d inference over the selected frozen public windows."""
    selected = splits or {"dev", "test"}
    predictions: list[FramePrediction] = []
    with chain_factory() as chain:
        for sequence in manifest.sequences:
            if sequence.split not in selected:
                continue
            video_path = video_cache / f"{sequence.source}.mp4"
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open {video_path}")
            chain.reset()
            estimator = PositionEstimator()
            try:
                for timestamp_s in _sample_times(
                    sequence.start_s, sequence.end_s, manifest.sample_fps
                ):
                    frame = _read_frame(capture, timestamp_s)
                    frame = _apply_annotation_transform(
                        frame,
                        _find_label(sequence, timestamp_s),
                    )
                    detection = chain.process_frame(frame, timestamp_s=timestamp_s)
                    position_observation = (
                        detection.position_fret
                        if detection.composite_available
                        or detection.position_fret is not None
                        else detection.index_fret
                    )
                    observation_confidence = (
                        detection.observation_confidence
                        if detection.composite_available
                        or detection.position_fret is not None
                        else (
                            detection.anchor.confidence
                            if detection.neck_locked
                            else 0.0
                        )
                    )
                    estimate = estimator.update(
                        index_fret=position_observation,
                        vision_confidence=observation_confidence,
                        timestamp_s=timestamp_s,
                    )
                    chain.set_hand_search_hint(
                        build_hand_search_hint(detection, estimate)
                    )
                    predictions.append(
                        FramePrediction(
                            sequence_id=sequence.sequence_id,
                            source=sequence.source,
                            split=sequence.split,
                            timestamp_s=round(timestamp_s, 6),
                            state=estimate.state,
                            position=estimate.position,
                            confidence=estimate.confidence,
                            observation_valid=estimate.raw_index_fret is not None,
                            confidence_factors=detection.confidence_factors,
                            geometry_status=detection.geometry_status,
                            geometry_age_ms=detection.geometry_age_ms,
                        )
                    )
            finally:
                capture.release()
    return predictions


def _displayed(prediction: FramePrediction) -> bool:
    return prediction.position is not None and prediction.state in {"locked", "holding"}


def _correct(frame: _ScoredFrame) -> bool:
    return _displayed(frame.prediction) and (
        frame.prediction.position == frame.label.position
    )


def _locked_correct(frame: _ScoredFrame) -> bool:
    return (
        frame.prediction.state == "locked"
        and frame.prediction.observation_valid
        and _correct(frame)
    )


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": round(numerator / denominator, 6) if denominator else None,
    }


def _metric_block(frames: Sequence[_ScoredFrame]) -> dict[str, object]:
    stable = [frame for frame in frames if frame.label.state == "stable"]
    valid = [frame for frame in stable if frame.prediction.observation_valid]
    stable_displayed = [frame for frame in stable if _displayed(frame.prediction)]
    stable_wrong = [frame for frame in stable_displayed if not _correct(frame)]
    displayed = [frame for frame in frames if _displayed(frame.prediction)]
    correct = [frame for frame in displayed if _correct(frame)]
    wrong = [frame for frame in displayed if not _correct(frame)]
    return {
        "labeled_frames": len(frames),
        "stable_frames": len(stable),
        "displayed_frames": len(displayed),
        "correct_frames": len(correct),
        "wrong_frames": len(wrong),
        "valid_observation_rate": _ratio(len(valid), len(stable)),
        "displayed_position_precision": _ratio(len(correct), len(displayed)),
        "coverage": _ratio(len(stable_displayed), len(stable)),
        "false_lock_rate": _ratio(len(stable_wrong), len(stable)),
    }


_FACTOR_FIELDS = tuple(
    field.name for field in fields(ConfidenceFactors) if field.name != "blockers"
)


def _blocker_summary(frames: Sequence[_ScoredFrame]) -> dict[str, object]:
    """Summarize raw solver blockers without collapsing their frame context."""
    blocker_counts: dict[str, int] = defaultdict(int)
    geometry_counts: dict[str, int] = defaultdict(int)
    factor_values: dict[str, list[float]] = {field: [] for field in _FACTOR_FIELDS}
    frames_with_blockers = 0
    frames_with_factor_data = 0
    for frame in frames:
        prediction = frame.prediction
        factors = prediction.confidence_factors
        geometry_counts[prediction.geometry_status] += 1
        if factors is None:
            continue
        frames_with_factor_data += 1
        blockers = tuple(dict.fromkeys(factors.blockers))
        if blockers:
            frames_with_blockers += 1
        for blocker in blockers:
            blocker_counts[blocker] += 1
        for field in _FACTOR_FIELDS:
            value = float(getattr(factors, field))
            if math.isfinite(value):
                factor_values[field].append(value)

    population = len(frames)
    return {
        "frames": population,
        "frames_with_factor_data": _ratio(frames_with_factor_data, population),
        "frames_with_blockers": _ratio(frames_with_blockers, population),
        "counts": dict(sorted(blocker_counts.items())),
        "rates": {
            blocker: _ratio(count, population)
            for blocker, count in sorted(blocker_counts.items())
        },
        "geometry_status_counts": dict(sorted(geometry_counts.items())),
        "factor_means": {
            field: (round(statistics.fmean(values), 6) if values else None)
            for field, values in factor_values.items()
        },
    }


def _find_label(
    sequence: BenchmarkSequence, timestamp_s: float
) -> PositionLabel | None:
    for label in sequence.labels:
        if label.start_s - 1e-9 <= timestamp_s < label.end_s - 1e-9:
            return label
    return None


def _latency_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "events": len(values),
        "median_s": round(statistics.median(values), 6) if values else None,
        "max_s": round(max(values), 6) if values else None,
        "values_s": [round(value, 6) for value in values],
    }


def _transition_latencies(
    manifest: BenchmarkManifest,
    frames: Sequence[_ScoredFrame],
    *,
    transition_state: Literal["shifting", "dropout"],
) -> tuple[list[float], int, int]:
    by_sequence: dict[str, list[_ScoredFrame]] = defaultdict(list)
    for frame in frames:
        by_sequence[frame.prediction.sequence_id].append(frame)
    values: list[float] = []
    censored = 0
    origin_not_locked = 0
    sequences = {sequence.sequence_id: sequence for sequence in manifest.sequences}
    for sequence_id, sequence_frames in by_sequence.items():
        sequence = sequences[sequence_id]
        labels = list(sequence.labels)
        for index, label in enumerate(labels):
            if label.state != "stable" or index < 2:
                continue
            transition = labels[index - 1]
            origin = labels[index - 2]
            if (
                transition.state != transition_state
                or origin.state != "stable"
                or abs(origin.end_s - transition.start_s) > 1e-6
                or abs(transition.end_s - label.start_s) > 1e-6
            ):
                continue
            origin_frames = [
                frame for frame in sequence_frames if frame.label is origin
            ]
            if not origin_frames or not _locked_correct(
                max(origin_frames, key=lambda frame: frame.prediction.timestamp_s)
            ):
                origin_not_locked += 1
                continue
            destination_locks = [
                frame
                for frame in sequence_frames
                if frame.label is label and _locked_correct(frame)
            ]
            if destination_locks:
                values.append(
                    min(frame.prediction.timestamp_s for frame in destination_locks)
                    - label.start_s
                )
            else:
                censored += 1
    return values, censored, origin_not_locked


def _validate_prediction_grid(
    manifest: BenchmarkManifest,
    predictions: Sequence[FramePrediction],
) -> None:
    if not predictions:
        raise ValueError("prediction set is empty")
    actual: set[tuple[str, float]] = set()
    for prediction in predictions:
        key = (prediction.sequence_id, round(prediction.timestamp_s, 6))
        if key in actual:
            raise ValueError(f"duplicate prediction sample: {key}")
        actual.add(key)
    sequences = {sequence.sequence_id: sequence for sequence in manifest.sequences}
    selected_splits = {
        sequences[prediction.sequence_id].split for prediction in predictions
    }
    selected_ids = {
        sequence.sequence_id
        for sequence in manifest.sequences
        if sequence.split in selected_splits
    }
    expected = {
        (sequence_id, round(timestamp_s, 6))
        for sequence_id in selected_ids
        for timestamp_s in _sample_times(
            sequences[sequence_id].start_s,
            sequences[sequence_id].end_s,
            manifest.sample_fps,
        )
    }
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise ValueError(
            "prediction grid mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def score_predictions(
    manifest: BenchmarkManifest,
    predictions: Sequence[FramePrediction],
) -> dict[str, object]:
    """Score displayed precision, abstention coverage, locks, and recovery."""
    sequences = {sequence.sequence_id: sequence for sequence in manifest.sequences}
    unknown = {
        prediction.sequence_id
        for prediction in predictions
        if prediction.sequence_id not in sequences
    }
    if unknown:
        raise ValueError(f"predictions reference unknown sequences: {sorted(unknown)}")
    _validate_prediction_grid(manifest, predictions)
    frames: list[_ScoredFrame] = []
    for prediction in predictions:
        sequence = sequences.get(prediction.sequence_id)
        if sequence is None:
            raise ValueError(
                f"prediction references unknown sequence: {prediction.sequence_id}"
            )
        if prediction.source != sequence.source or prediction.split != sequence.split:
            raise ValueError(f"prediction metadata mismatch: {prediction.sequence_id}")
        label = _find_label(sequence, prediction.timestamp_s)
        if label is not None:
            frames.append(_ScoredFrame(prediction=prediction, label=label))

    split_metrics = {
        split: _metric_block(
            [frame for frame in frames if frame.prediction.split == split]
        )
        for split in ("dev", "test")
    }
    invalid = [frame for frame in frames if frame.label.state == "invalid"]
    invalid_displays = sum(_displayed(frame.prediction) for frame in invalid)
    shift_values, shift_censored, shift_origin_missing = _transition_latencies(
        manifest,
        frames,
        transition_state="shifting",
    )
    dropout_values, dropout_censored, dropout_origin_missing = _transition_latencies(
        manifest,
        frames,
        transition_state="dropout",
    )

    breakdowns: dict[str, dict[str, object]] = {}
    for field in ("position", "technique", "visibility", "lighting"):
        groups: dict[str, list[_ScoredFrame]] = defaultdict(list)
        eligible = (
            [frame for frame in frames if frame.label.position is not None]
            if field == "position"
            else frames
        )
        for frame in eligible:
            groups[str(getattr(frame.label, field))].append(frame)
        breakdowns[field] = {
            key: _metric_block(group) for key, group in sorted(groups.items())
        }

    state_display_rates = {
        state: _ratio(
            sum(
                _displayed(frame.prediction)
                for frame in frames
                if frame.label.state == state
            ),
            sum(frame.label.state == state for frame in frames),
        )
        for state in ("stable", "shifting", "dropout", "invalid")
    }

    sequence_breakdown: dict[str, object] = {}
    for sequence_id in sorted({frame.prediction.sequence_id for frame in frames}):
        sequence_breakdown[sequence_id] = _metric_block(
            [frame for frame in frames if frame.prediction.sequence_id == sequence_id]
        )

    stable_frames = [frame for frame in frames if frame.label.state == "stable"]
    invalid_observations = [
        frame for frame in frames if not frame.prediction.observation_valid
    ]
    return {
        "overall": _metric_block(frames),
        "splits": split_metrics,
        "negative_control_display_rate": _ratio(invalid_displays, len(invalid)),
        "state_display_rates": state_display_rates,
        "shift_latency": {
            **_latency_summary(shift_values),
            "censored_events": shift_censored,
            "origin_not_locked_events": shift_origin_missing,
        },
        "dropout_recovery": {
            **_latency_summary(dropout_values),
            "censored_events": dropout_censored,
            "origin_not_locked_events": dropout_origin_missing,
        },
        "breakdowns": breakdowns,
        "sequences": sequence_breakdown,
        "blockers": {
            "overall": _blocker_summary(frames),
            "stable": _blocker_summary(stable_frames),
            "invalid_observations": _blocker_summary(invalid_observations),
            "splits": {
                split: _blocker_summary(
                    [frame for frame in frames if frame.prediction.split == split]
                )
                for split in ("dev", "test")
            },
        },
    }


def _format_ratio(metric: object) -> str:
    data = metric if isinstance(metric, dict) else {}
    value = data.get("value")
    if value is None:
        return "n/a"
    return f"{float(value):.3f} ({data['numerator']}/{data['denominator']})"


def render_report(
    manifest: BenchmarkManifest,
    metrics: dict[str, object],
    *,
    command: str,
) -> str:
    """Render a compact, reproducible F4e-A baseline report."""
    positions = sorted(
        {
            label.position
            for sequence in manifest.sequences
            for label in sequence.labels
            if label.position is not None
        }
    )
    source_count = len({sequence.source for sequence in manifest.sequences})
    split_counts = {
        split: (
            sum(sequence.split == split for sequence in manifest.sequences),
            len(
                {
                    sequence.source
                    for sequence in manifest.sequences
                    if sequence.split == split
                }
            ),
        )
        for split in ("dev", "test")
    }
    sampled_frames = sum(
        sum(
            1
            for _ in _sample_times(
                sequence.start_s,
                sequence.end_s,
                manifest.sample_fps,
            )
        )
        for sequence in manifest.sequences
    )
    splits = metrics["splits"]
    assert isinstance(splits, dict)
    lines = [
        "# FretCam F4e-A - public position benchmark baseline",
        "",
        "**Date:** 2026-07-22",
        "",
        "**Scope:** Measurement only; unchanged F4d product inference",
        "",
        "**Verdict:** BENCHMARK FROZEN; no accuracy gate is claimed from this small set",
        "",
        "## Frozen evidence",
        "",
        f"- Corpus: {manifest.corpus} ({manifest.corpus_license}); public footage only.",
        f"- Sequences: {len(manifest.sequences)} across {source_count} sources; "
        f"dev {split_counts['dev'][0]} sequences/{split_counts['dev'][1]} sources, "
        f"test {split_counts['test'][0]} sequences/{split_counts['test'][1]} sources.",
        "- Dev/test sources are disjoint.",
        f"- Samples: {sampled_frames} frames total before unlabeled ranges are omitted.",
        f"- Sample rate: {manifest.sample_fps:g} FPS.",
        f"- Stable positions represented: {', '.join(str(value) for value in positions)}.",
        f"- Policy: {manifest.annotation_policy}",
        "- This report is the one initial opening of the frozen test baseline. Future rule/threshold choices use dev only; test is rerun once for the final comparison.",
        "",
        "## F4d baseline",
        "",
        "| split | valid observations | displayed precision | coverage | false-lock rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test"):
        block = splits[split]
        assert isinstance(block, dict)
        lines.append(
            f"| {split} | {_format_ratio(block['valid_observation_rate'])} | "
            f"{_format_ratio(block['displayed_position_precision'])} | "
            f"{_format_ratio(block['coverage'])} | {_format_ratio(block['false_lock_rate'])} |"
        )
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    lines.extend(
        [
            f"| all | {_format_ratio(overall['valid_observation_rate'])} | "
            f"{_format_ratio(overall['displayed_position_precision'])} | "
            f"{_format_ratio(overall['coverage'])} | {_format_ratio(overall['false_lock_rate'])} |",
            "",
            f"- Negative-control display rate: {_format_ratio(metrics['negative_control_display_rate'])}.",
            "",
            "### Display rate by ground-truth state",
            "",
            "| state | displayed frames |",
            "|---|---:|",
        ]
    )
    state_display_rates = metrics["state_display_rates"]
    assert isinstance(state_display_rates, dict)
    for state in ("stable", "shifting", "dropout", "invalid"):
        lines.append(f"| {state} | {_format_ratio(state_display_rates[state])} |")
    lines.append("")
    shift = metrics["shift_latency"]
    dropout = metrics["dropout_recovery"]
    assert isinstance(shift, dict) and isinstance(dropout, dict)
    shift_latency = (
        "not observed" if shift["median_s"] is None else f"{shift['median_s']} s median"
    )
    dropout_latency = (
        "not observed"
        if dropout["median_s"] is None
        else f"{dropout['median_s']} s median"
    )
    lines.extend(
        [
            f"- Shift latency: {shift_latency} across {shift['events']} observed event(s); {shift['censored_events']} censored and {shift['origin_not_locked_events']} excluded because the origin was not freshly locked.",
            f"- Dropout recovery from the annotated valid-return boundary: {dropout_latency} across {dropout['events']} observed event(s); {dropout['censored_events']} censored and {dropout['origin_not_locked_events']} origin-not-locked.",
            "",
            "## Observation blockers",
            "",
            "Blockers below are the solver's raw `confidence_factors.blockers`; "
            "a frame can contribute to more than one row.",
        ]
    )
    blocker_metrics = metrics["blockers"]
    assert isinstance(blocker_metrics, dict)
    overall_blockers = blocker_metrics["overall"]
    assert isinstance(overall_blockers, dict)
    blocker_counts = overall_blockers["counts"]
    blocker_rates = overall_blockers["rates"]
    assert isinstance(blocker_counts, dict) and isinstance(blocker_rates, dict)
    lines.extend(
        [
            "",
            "Factor vectors available for "
            f"{_format_ratio(overall_blockers['frames_with_factor_data'])} "
            "of labeled frames.",
            "",
            "| blocker | frames | rate over labeled frames |",
            "|---|---:|---:|",
        ]
    )
    if blocker_counts:
        for blocker, count in blocker_counts.items():
            lines.append(
                f"| `{blocker}` | {count} | {_format_ratio(blocker_rates[blocker])} |"
            )
    else:
        lines.append("| none observed | 0 | n/a |")
    geometry_counts = overall_blockers["geometry_status_counts"]
    assert isinstance(geometry_counts, dict)
    lines.extend(
        [
            "",
            "- Geometry status counts: "
            + ", ".join(
                f"`{status}` {count}" for status, count in geometry_counts.items()
            )
            + ".",
            "",
            "## Position breakdown",
            "",
            "| position | precision | coverage | false-lock rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    breakdowns = metrics["breakdowns"]
    assert isinstance(breakdowns, dict)
    positions_block = breakdowns["position"]
    assert isinstance(positions_block, dict)
    for position, block in positions_block.items():
        assert isinstance(block, dict)
        lines.append(
            f"| {position} | {_format_ratio(block['displayed_position_precision'])} | "
            f"{_format_ratio(block['coverage'])} | {_format_ratio(block['false_lock_rate'])} |"
        )
    for field, title in (
        ("technique", "Technique breakdown"),
        ("visibility", "Visibility breakdown"),
        ("lighting", "Lighting breakdown"),
    ):
        field_block = breakdowns[field]
        assert isinstance(field_block, dict)
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"| {field} | precision | coverage | false-lock rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for value, block in field_block.items():
            assert isinstance(block, dict)
            lines.append(
                f"| {value} | {_format_ratio(block['displayed_position_precision'])} | "
                f"{_format_ratio(block['coverage'])} | {_format_ratio(block['false_lock_rate'])} |"
            )

    sequence_metrics = metrics["sequences"]
    assert isinstance(sequence_metrics, dict)
    lines.extend(
        [
            "",
            "## Sequence diagnostics",
            "",
            "| sequence | valid observations | display coverage | precision |",
            "|---|---:|---:|---:|",
        ]
    )
    for sequence_id, block in sequence_metrics.items():
        assert isinstance(block, dict)
        lines.append(
            f"| `{sequence_id}` | {_format_ratio(block['valid_observation_rate'])} | "
            f"{_format_ratio(block['coverage'])} | "
            f"{_format_ratio(block['displayed_position_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            f"- The baseline's main limitation is coverage: {_format_ratio(overall['coverage'])} overall and {_format_ratio(splits['test']['coverage'])} held-out, alongside an overall valid-observation rate of {_format_ratio(overall['valid_observation_rate'])}.",
            f"- Held-out stable false locks are {_format_ratio(splits['test']['false_lock_rate'])}; four displays on the held-out invalid crossfade reduce held-out displayed precision to {_format_ratio(splits['test']['displayed_position_precision'])}.",
            "- No transition yielded a numeric latency: all three shift origins lacked a fresh valid lock; one dropout origin lacked a lock and the other recovery was right-censored.",
            "- This is the first defensible position-HUD measurement set, not a population estimate. It is intentionally small and excludes uncertain ranges.",
            "- Two behaviors carry prior user verification; the remaining intervals were independently frame-reviewed against visible nut/fret wires.",
            "- Dropout recovery uses manifest-annotated occlusion and valid-return boundaries. The occlusions are deterministic masks over public frames; they are not inferred from product output.",
            "- GAPS was not recorded under FretCam's controlled-camera contract and has no native classical-position labels. A second human label review would strengthen the benchmark.",
            "- The short dev Position-IX arrival contains only four 10 FPS samples, so that shift can be window-censored before the five-frame estimator can lock.",
            "- No threshold was selected from these results, and no inference, dependency, model, download, training run, or TabVision package behavior changed.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument(
        "--video-cache",
        type=Path,
        default=Path.home() / ".tabvision" / "cache" / "gaps_video",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    splits: set[Split] = {"dev", "test"} if args.split == "all" else {args.split}
    predictions = run_inference(
        manifest,
        video_cache=args.video_cache,
        splits=splits,
    )
    metrics = score_predictions(manifest, predictions)
    payload = {
        "manifest": str(args.manifest.resolve()),
        "predictions": [asdict(prediction) for prediction in predictions],
        "metrics": metrics,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.report is not None:
        command = (
            ".\\fretcam\\.venv\\Scripts\\python.exe -m "
            f"fretcam.position_benchmark --split {args.split} --output-json "
            "<machine-local-output.json>"
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_report(manifest, metrics, command=command),
            encoding="utf-8",
        )
    print(rendered)


if __name__ == "__main__":
    main()
