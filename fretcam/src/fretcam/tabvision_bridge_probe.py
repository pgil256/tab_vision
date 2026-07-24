"""Reproduce the cache-only FretCam-to-audio fixed-policy bridge check.

The probe deliberately isolates string/fret assignment by turning the GAPS
gold pitches into :class:`~tabvision.types.AudioEvent` objects.  Corrected
FretCam cache anchors are mapped from video time back to the gold/audio clock,
then passed through the same causal, bounded position-window prior used by the
production bridge.  Both arms first receive the paired GAPS position and
sequence priors selected by clean-classical automatic routing.  No model
inference, download, or training occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fretcam.gaps_anchor_probe import (
    _video_fps,
    corrected_cached_anchor,
    position_from_centroid,
)
from fretcam.position import position_window
from scripts.acquire.gaps_video import CLEAN_12
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion import playability
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.position_window_prior import (
    MIN_POSITION_OBSERVATION_CONFIDENCE,
    POSITION_OBSERVATION_LEAD_S,
    POSITION_OBSERVATION_LOOKBACK_S,
    apply_position_window_priors,
)
from tabvision.fusion.transition_prior import TransitionPrior, load_transition_prior
from tabvision.fusion.viterbi import assignment_decoder_context, fuse
from tabvision.pipeline import SEQUENCE_PRIOR_WEIGHT, sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, TabEvent
from tabvision.video.position import PositionWindowObservation

ONSET_KEY_DIGITS = 6
"""Decimal places used to align decoded events with gold onset/pitch queues."""

POSITION_PRIOR_NAME = "gaps-v1"
SEQUENCE_PRIOR_NAME = "gaps-seq-v1"
FUSION_OVERRIDE_ENV_VARS = (
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
DEFAULT_CV_CACHE = Path.home() / ".tabvision" / "cache" / "gaps_video_chain"
DEFAULT_OUTPUT_JSON = (
    Path.home() / ".tabvision" / "reports" / "fretcam_audio_bridge_fixed_policy.json"
)
DEFAULT_OUTPUT_REPORT = (
    Path.home() / ".tabvision" / "reports" / "fretcam_audio_bridge_fixed_policy.md"
)


class AnchorLike(Protocol):
    """Fields used from a corrected cache anchor."""

    @property
    def center_fret(self) -> float: ...

    @property
    def confidence(self) -> float: ...


AnchorBuilder = Callable[[Any, GuitarConfig], AnchorLike]


def _default_anchor_builder(record: Any, cfg: GuitarConfig) -> AnchorLike:
    """Narrow the cache corrector to the two-argument probe callback shape."""
    return corrected_cached_anchor(record, cfg)


@dataclass(frozen=True)
class AlignmentScore:
    """Exact string/fret score after robust onset/pitch alignment."""

    playable_notes: int
    matched_notes: int
    correct_notes: int


@dataclass(frozen=True)
class ClipBridgeResult:
    """Fixed-policy bridge result for one GAPS clip."""

    clip: str
    playable_notes: int
    baseline_matched: int
    bridge_matched: int
    baseline_correct: int
    bridge_correct: int
    events_enriched: int
    accepted_observations: int

    @property
    def net(self) -> int:
        return self.bridge_correct - self.baseline_correct

    def as_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["net"] = self.net
        return payload


@dataclass(frozen=True)
class AggregateBridgeResult:
    """Aggregate fixed-policy metrics over a sequence of clips."""

    playable_notes: int
    baseline_matched: int
    bridge_matched: int
    baseline_correct: int
    bridge_correct: int
    events_enriched: int
    accepted_observations: int

    @property
    def net(self) -> int:
        return self.bridge_correct - self.baseline_correct

    @property
    def assignment_scored_notes(self) -> int:
        """Notes present in both gold-pitch decoder arms and eligible to score."""
        return min(self.baseline_matched, self.bridge_matched)

    @property
    def excluded_playable_notes(self) -> int:
        return max(0, self.playable_notes - self.assignment_scored_notes)

    @property
    def baseline_accuracy(self) -> float:
        return _ratio(self.baseline_correct, self.assignment_scored_notes)

    @property
    def bridge_accuracy(self) -> float:
        return _ratio(self.bridge_correct, self.assignment_scored_notes)

    @property
    def absolute_accuracy_change(self) -> float:
        return self.bridge_accuracy - self.baseline_accuracy

    @property
    def relative_error_reduction(self) -> float:
        return _ratio(
            self.net,
            self.assignment_scored_notes - self.baseline_correct,
        )

    def as_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = asdict(self)
        payload.update(
            {
                "net": self.net,
                "assignment_scored_notes": self.assignment_scored_notes,
                "excluded_playable_notes": self.excluded_playable_notes,
                "baseline_accuracy": self.baseline_accuracy,
                "bridge_accuracy": self.bridge_accuracy,
                "absolute_accuracy_change": self.absolute_accuracy_change,
                "relative_error_reduction": self.relative_error_reduction,
            }
        )
        return payload


@dataclass(frozen=True)
class BridgeProbeResult:
    """Complete clean-12 result with per-clip and aggregate metrics."""

    clips: tuple[ClipBridgeResult, ...]
    aggregate: AggregateBridgeResult

    def as_dict(self) -> dict[str, object]:
        return {
            "methodology": {
                "clips": list(CLEAN_12),
                "audio_input": "gold_pitch_with_classical_auto_string_policy",
                "cache_only": True,
                "model_inference": False,
                "policy_route": "clean_classical_auto",
                "position_prior": POSITION_PRIOR_NAME,
                "sequence_prior": SEQUENCE_PRIOR_NAME,
                "sequence_prior_weight": SEQUENCE_PRIOR_WEIGHT,
                "fusion_environment_overrides": "rejected",
                "minimum_observation_confidence": MIN_POSITION_OBSERVATION_CONFIDENCE,
                "position_observation_lead_s": POSITION_OBSERVATION_LEAD_S,
                "position_observation_lookback_s": POSITION_OBSERVATION_LOOKBACK_S,
                "position_window": "{0} union [N-1,N+4]",
                "assignment_decoder": "baseline (current auto rollback)",
                "onset_key_digits": ONSET_KEY_DIGITS,
            },
            "clips": [result.as_dict() for result in self.clips],
            "aggregate": self.aggregate.as_dict(),
        }


def gold_pitch_audio_events(gold: Sequence[TabEvent]) -> list[AudioEvent]:
    """Convert gold tab to audio events while intentionally hiding positions."""
    return [
        AudioEvent(
            onset_s=event.onset_s,
            offset_s=event.onset_s + event.duration_s,
            pitch_midi=event.pitch_midi,
            velocity=1.0,
            confidence=1.0,
        )
        for event in gold
    ]


def classical_gold_pitch_audio_events(
    gold: Sequence[TabEvent],
    prior: PitchPositionPrior,
    cfg: GuitarConfig,
) -> list[AudioEvent]:
    """Apply the clean-classical automatic string policy to gold pitches."""
    return apply_pitch_position_prior(gold_pitch_audio_events(gold), prior, cfg)


def anchor_to_observation(
    anchor: AnchorLike,
    *,
    frame_index: int,
    fps: float,
    offset_s: float,
    cfg: GuitarConfig,
) -> PositionWindowObservation | None:
    """Map one valid corrected cache anchor onto the gold/audio media clock.

    GAPS offsets satisfy ``video_time = gold_time + offset``.  Therefore a
    cached video frame at ``frame_index / fps`` becomes an audio-clock
    observation at ``frame_index / fps - offset``.
    """
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
    ):
        raise ValueError("frame_index must be a non-negative integer")
    if not math.isfinite(float(fps)) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(float(offset_s)):
        raise ValueError("offset_s must be finite")

    confidence = float(anchor.confidence)
    center_fret = float(anchor.center_fret)
    timestamp_s = frame_index / float(fps) - float(offset_s)
    if (
        not math.isfinite(confidence)
        or not MIN_POSITION_OBSERVATION_CONFIDENCE <= confidence <= 1.0
        or not math.isfinite(center_fret)
    ):
        return None

    position = position_from_centroid(center_fret)
    if position > cfg.max_fret:
        return None
    return PositionWindowObservation(
        timestamp_s=timestamp_s,
        position=position,
        window_frets=position_window(position, max_fret=cfg.max_fret),
        confidence=confidence,
        state="locked",
    )


def cached_position_observations(
    records: Mapping[int, Any | None],
    *,
    fps: float,
    offset_s: float,
    cfg: GuitarConfig,
    anchor_builder: AnchorBuilder = _default_anchor_builder,
) -> list[PositionWindowObservation]:
    """Convert every usable corrected rich-cache record deterministically."""
    observations: list[PositionWindowObservation] = []
    for frame_index in sorted(records):
        record = records[frame_index]
        if record is None:
            continue
        anchor = anchor_builder(record, cfg)
        observation = anchor_to_observation(
            anchor,
            frame_index=frame_index,
            fps=fps,
            offset_s=offset_s,
            cfg=cfg,
        )
        if observation is not None:
            observations.append(observation)
    return observations


def score_decoded_against_gold(
    gold: Sequence[TabEvent],
    decoded: Sequence[TabEvent],
    cfg: GuitarConfig,
) -> AlignmentScore:
    """Score exact positions using rounded onset/pitch queues.

    A queue rather than positional ``zip`` keeps one dropped event from
    shifting the remainder of a clip.  It also preserves deterministic order
    for same-pitch unisons at a shared onset.
    """
    gold_queues: dict[tuple[float, int], deque[TabEvent]] = defaultdict(deque)
    playable_notes = 0
    for event in gold:
        if not candidate_positions(event.pitch_midi, cfg):
            continue
        playable_notes += 1
        gold_queues[_event_key(event.onset_s, event.pitch_midi)].append(event)

    matched = 0
    correct = 0
    for event in decoded:
        queue = gold_queues.get(_event_key(event.onset_s, event.pitch_midi))
        if not queue:
            continue
        expected = queue.popleft()
        matched += 1
        if (event.string_idx, event.fret) == (expected.string_idx, expected.fret):
            correct += 1
    return AlignmentScore(playable_notes, matched, correct)


def aggregate_results(results: Sequence[ClipBridgeResult]) -> AggregateBridgeResult:
    """Sum per-clip counts and derive aggregate accuracy metrics."""
    fields = (
        "playable_notes",
        "baseline_matched",
        "bridge_matched",
        "baseline_correct",
        "bridge_correct",
        "events_enriched",
        "accepted_observations",
    )
    return AggregateBridgeResult(
        **{
            field: sum(int(getattr(result, field)) for result in results)
            for field in fields
        }
    )


def probe_clip(
    stem: str,
    *,
    data_root: Path,
    video_cache: Path,
    cv_cache: Path,
    cfg: GuitarConfig,
    position_prior: PitchPositionPrior,
) -> ClipBridgeResult:
    """Run the cache-only fixed bridge policy for one clip."""
    xml_path = data_root / "gaps" / "musicxml" / f"{stem}.xml"
    video_path = video_cache / f"{stem}.mp4"
    raw_path = cv_cache / f"{stem}.rawcv.c0.25.pkl"
    offset_path = cv_cache / f"{stem}.offset.pkl"
    for path in (xml_path, video_path, raw_path, offset_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    gold = parse_gaps(xml_path, cfg)
    audio_events = classical_gold_pitch_audio_events(gold, position_prior, cfg)
    records: Mapping[int, Any | None] = _load_trusted_pickle(raw_path)
    offset_record = _load_trusted_pickle(offset_path)
    offset_s = float(offset_record.offset_s)
    observations = cached_position_observations(
        records,
        fps=_video_fps(video_path),
        offset_s=offset_s,
        cfg=cfg,
    )
    enriched_events = apply_position_window_priors(
        audio_events,
        observations,
        cfg,
    )

    baseline = fuse(audio_events, (), cfg, lambda_vision=0.0)
    bridge = fuse(enriched_events, (), cfg, lambda_vision=0.0)

    baseline_score = score_decoded_against_gold(gold, baseline, cfg)
    bridge_score = score_decoded_against_gold(gold, bridge, cfg)
    if baseline_score.playable_notes != bridge_score.playable_notes:
        raise RuntimeError("baseline and bridge playable-note totals differ")
    if baseline_score.matched_notes != bridge_score.matched_notes:
        raise RuntimeError("baseline and bridge assignment-scored totals differ")

    return ClipBridgeResult(
        clip=stem,
        playable_notes=baseline_score.playable_notes,
        baseline_matched=baseline_score.matched_notes,
        bridge_matched=bridge_score.matched_notes,
        baseline_correct=baseline_score.correct_notes,
        bridge_correct=bridge_score.correct_notes,
        events_enriched=sum(
            before is not after
            for before, after in zip(audio_events, enriched_events, strict=True)
        ),
        accepted_observations=len(observations),
    )


def run_probe(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    video_cache: Path = DEFAULT_VIDEO_CACHE,
    cv_cache: Path = DEFAULT_CV_CACHE,
) -> BridgeProbeResult:
    """Run the frozen clean-12 cache-only bridge reproduction."""
    active_overrides = [name for name in FUSION_OVERRIDE_ENV_VARS if name in os.environ]
    if active_overrides:
        joined = ", ".join(active_overrides)
        raise RuntimeError(
            f"fixed-policy probe requires default fusion settings; unset: {joined}"
        )
    cfg = GuitarConfig()
    # Clean-classical ``auto`` routing resolves to this paired GAPS policy.
    # Load each artifact once, outside the per-clip loop.
    position_prior = load_pitch_position_prior(POSITION_PRIOR_NAME, cfg=cfg)
    sequence_prior = load_transition_prior(SEQUENCE_PRIOR_NAME)
    with _fixed_classical_decode_context(sequence_prior):
        clips = tuple(
            probe_clip(
                stem,
                data_root=data_root,
                video_cache=video_cache,
                cv_cache=cv_cache,
                cfg=cfg,
                position_prior=position_prior,
            )
            for stem in CLEAN_12
        )
    return BridgeProbeResult(clips=clips, aggregate=aggregate_results(clips))


def format_report(result: BridgeProbeResult) -> str:
    """Render a compact Markdown account of the frozen policy and results."""
    total = result.aggregate
    lines = [
        "# FretCam to audio fusion bridge: cache-only fixed-policy reproduction",
        "",
        "This report uses GAPS gold pitches and cached corrected CV anchors only. "
        "It performs no model inference, download, training, or policy tuning.",
        "",
        "## Frozen policy",
        "",
        f"- Clean-classical automatic string policy: `{POSITION_PRIOR_NAME}` "
        "position prior paired with "
        f"`{SEQUENCE_PRIOR_NAME}` at weight `{SEQUENCE_PRIOR_WEIGHT:.1f}`.",
        "- The position artifact is loaded once and applied before FretCam enrichment; "
        "both arms use the same sequence prior and baseline assignment decoder.",
        f"- Accepted confidence: at least `{MIN_POSITION_OBSERVATION_CONFIDENCE:.2f}`.",
        "- Causal join: latest observation in the "
        f"{POSITION_OBSERVATION_LOOKBACK_S:.2f} s lookback ending at "
        f"`onset - {POSITION_OBSERVATION_LEAD_S:.2f} s`.",
        "- Position support: `{0} union [N-1,N+4]`; production bounded prior.",
        "- Assignment decoder: pinned baseline, matching the current automatic "
        "rollback.",
        "",
        "## Results",
        "",
        "| clip | assignment-scored notes | baseline correct | bridge correct "
        "| net | events enriched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for clip in result.clips:
        lines.append(
            f"| {clip.clip} | {clip.baseline_matched:,} | {clip.baseline_correct:,} | "
            f"{clip.bridge_correct:,} | {clip.net:+d} | {clip.events_enriched:,} |"
        )
    lines.extend(
        [
            f"| **Total** | **{total.assignment_scored_notes:,}** | "
            f"**{total.baseline_correct:,}** | **{total.bridge_correct:,}** | "
            f"**{total.net:+d}** | **{total.events_enriched:,}** |",
            "",
            f"Aggregate exact string/fret accuracy: `{total.baseline_accuracy:.6f}` to "
            f"`{total.bridge_accuracy:.6f}` "
            f"(`{total.absolute_accuracy_change:+.6f}` absolute).",
            "",
            f"Relative error reduction: `{total.net} / "
            f"{total.assignment_scored_notes - total.baseline_correct} = "
            f"{total.relative_error_reduction:.4%}`.",
            "",
            f"Coverage: `{total.assignment_scored_notes:,}` of "
            f"`{total.playable_notes:,}` individually playable gold notes; "
            f"`{total.excluded_playable_notes:,}` duplicated/unison chord notes "
            "cannot survive the decoder's per-string monophony constraint and "
            "are excluded symmetrically from both assignment arms.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(result: BridgeProbeResult, path: Path) -> None:
    """Write deterministic machine-readable results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report(result: BridgeProbeResult, path: Path) -> None:
    """Write the optional Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(result), encoding="utf-8")


def _event_key(onset_s: float, pitch_midi: int) -> tuple[float, int]:
    return round(float(onset_s), ONSET_KEY_DIGITS), int(pitch_midi)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


@contextmanager
def _fixed_classical_decode_context(
    sequence_prior: TransitionPrior,
) -> Iterator[None]:
    """Install the production classical sequence policy without env drift."""
    # The public pipeline context supplies the same decode lock used by real
    # jobs. Installing the already-loaded object ourselves fixes the accepted
    # weight even when sweep environment variables happen to be present.
    with sequence_decode_context("none"):
        playability.set_transition_prior(
            sequence_prior,
            weight=SEQUENCE_PRIOR_WEIGHT,
        )
        try:
            with assignment_decoder_context("baseline"):
                yield
        finally:
            playability.set_transition_prior(None)


def _load_trusted_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted, locally generated cache


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-cache", type=Path, default=DEFAULT_VIDEO_CACHE)
    parser.add_argument("--cv-cache", type=Path, default=DEFAULT_CV_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-report",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT_REPORT,
        default=None,
        help="optionally write Markdown (default path when flag has no value)",
    )
    args = parser.parse_args(argv)

    result = run_probe(
        data_root=args.data_root,
        video_cache=args.video_cache,
        cv_cache=args.cv_cache,
    )
    write_json(result, args.output_json)
    if args.output_report is not None:
        write_report(result, args.output_report)

    aggregate = result.aggregate
    print(
        f"{aggregate.assignment_scored_notes} assignment-scored "
        f"of {aggregate.playable_notes} playable; "
        f"{aggregate.baseline_correct} -> {aggregate.bridge_correct} correct; "
        f"net {aggregate.net:+d}; {aggregate.events_enriched} enriched"
    )
    print(f"JSON: {args.output_json}")
    if args.output_report is not None:
        print(f"Markdown: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
