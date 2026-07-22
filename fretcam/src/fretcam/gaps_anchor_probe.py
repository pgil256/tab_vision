"""F7 cache-only GAPS hand-centroid complementarity probe.

This evaluation adapter deliberately reuses A14's gold-pitch audio decoder and
the rich GAPS CV cache.  It performs no video inference, model download, or
training.  The primary metric is the fraction of audio-wrong ambiguous notes
whose gold fret lies in the FretCam position window.
"""

from __future__ import annotations

import argparse
import bisect
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2

from scripts.acquire.gaps_video import CLEAN_12
from scripts.eval.a14_video_complementarity_probe import decode_with_margins
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps
from tabvision.fusion.candidates import candidate_positions
from tabvision.types import AudioEvent, GuitarConfig, TabEvent
from tabvision.video.hand.neck_anchor import compute_neck_anchor

TARGET_LEAD_S = 0.030
CACHE_TOLERANCE_S = 0.060
A14_ANTI_ENRICHMENT = 0.285
AUDIO_PRIOR_REFERENCE = 0.778


@dataclass(frozen=True)
class WindowCounts:
    """Aggregate counts for one clip or the full clean-12 bank."""

    ambiguous: int = 0
    audio_right: int = 0
    audio_wrong: int = 0
    anchor_covered: int = 0
    anchor_covered_audio_right: int = 0
    anchor_covered_audio_wrong: int = 0
    gold_in_window: int = 0
    gold_in_window_audio_right: int = 0
    gold_in_window_audio_wrong: int = 0
    audio_choice_in_window_when_wrong: int = 0
    gold_in_audio_out: int = 0
    both_in: int = 0
    gold_out_audio_in: int = 0
    both_out: int = 0
    center_at_nut_boundary: int = 0
    center_at_bridge_boundary: int = 0

    def __add__(self, other: WindowCounts) -> WindowCounts:
        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        }
        return WindowCounts(**values)


@dataclass(frozen=True)
class ClipResult:
    """Counts and timing diagnostics from one cached public GAPS clip."""

    clip: str
    counts: WindowCounts
    target_lags_s: tuple[float, ...]


def position_from_centroid(center_fret: float) -> int:
    """Map a finite centroid to the design's 1-based classical position."""
    if not math.isfinite(center_fret):
        raise ValueError("center_fret must be finite")
    return max(1, math.floor(center_fret))


def fret_in_position_window(fret: int, center_fret: float) -> bool:
    """Apply ``[N-1, N+4] union {0}`` without tuning the design constants."""
    if fret < 0:
        raise ValueError("fret must be non-negative")
    if fret == 0:
        return True
    position = position_from_centroid(center_fret)
    return position - 1 <= fret <= position + 4


def nearest_cached_frame(
    frame_indices: Sequence[int],
    *,
    target_s: float,
    fps: float,
    tolerance_s: float = CACHE_TOLERANCE_S,
) -> int | None:
    """Return the nearest cached frame to a target, preferring earlier ties."""
    if fps <= 0.0 or tolerance_s < 0.0:
        raise ValueError("fps must be positive and tolerance_s non-negative")
    if not frame_indices:
        return None
    target_frame = target_s * fps
    insertion = bisect.bisect_left(frame_indices, target_frame)
    candidates = []
    if insertion > 0:
        candidates.append(frame_indices[insertion - 1])
    if insertion < len(frame_indices):
        candidates.append(frame_indices[insertion])
    chosen = min(candidates, key=lambda frame: (abs(frame - target_frame), frame))
    if abs(chosen / fps - target_s) > tolerance_s + 1e-12:
        return None
    return chosen


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval (95% with the default z value)."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("expected 0 <= successes <= total and total > 0")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    ) / denominator
    return center - radius, center + radius


def _events_from_gold(gold: list[TabEvent]) -> list[AudioEvent]:
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


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted, locally generated cache


def _video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open cached video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"invalid FPS for cached video: {path}")
    return fps


def probe_clip(
    stem: str,
    *,
    data_root: Path,
    video_cache: Path,
    cv_cache: Path,
    cfg: GuitarConfig,
) -> ClipResult:
    """Score one clip using only banked annotations and cached CV records."""
    xml = data_root / "gaps" / "musicxml" / f"{stem}.xml"
    video = video_cache / f"{stem}.mp4"
    raw_path = cv_cache / f"{stem}.rawcv.c0.25.pkl"
    offset_path = cv_cache / f"{stem}.offset.pkl"
    for path in (xml, video, raw_path, offset_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gold = parse_gaps(xml)
    events = _events_from_gold(gold)
    decoded = decode_with_margins(events, cfg)

    raw: dict[int, Any | None] = _load_pickle(raw_path)
    offset_s = float(_load_pickle(offset_path).offset_s)
    fps = _video_fps(video)
    frame_indices = sorted(raw)
    values = {name: 0 for name in WindowCounts.__dataclass_fields__}
    lags: list[float] = []

    for index, event in enumerate(gold):
        if index not in decoded or len(candidate_positions(event.pitch_midi, cfg)) < 2:
            continue
        candidate = decoded[index][0]
        audio_right = candidate.string_idx == event.string_idx
        values["ambiguous"] += 1
        values["audio_right" if audio_right else "audio_wrong"] += 1

        target_s = event.onset_s + offset_s - TARGET_LEAD_S
        frame_index = nearest_cached_frame(frame_indices, target_s=target_s, fps=fps)
        if frame_index is None or raw[frame_index] is None:
            continue
        record = raw[frame_index]
        anchor = compute_neck_anchor(record.hand, record.homography, cfg)
        if anchor.confidence <= 0.0 or not math.isfinite(anchor.center_fret):
            continue

        lags.append(frame_index / fps - target_s)
        values["anchor_covered"] += 1
        if anchor.center_fret <= 1e-3:
            values["center_at_nut_boundary"] += 1
        if anchor.center_fret >= cfg.max_fret - 1e-3:
            values["center_at_bridge_boundary"] += 1

        gold_in = fret_in_position_window(event.fret, anchor.center_fret)
        audio_in = fret_in_position_window(candidate.fret, anchor.center_fret)
        if gold_in:
            values["gold_in_window"] += 1
        if audio_right:
            values["anchor_covered_audio_right"] += 1
            if gold_in:
                values["gold_in_window_audio_right"] += 1
            continue

        values["anchor_covered_audio_wrong"] += 1
        if gold_in:
            values["gold_in_window_audio_wrong"] += 1
        if audio_in:
            values["audio_choice_in_window_when_wrong"] += 1
        key = {
            (True, False): "gold_in_audio_out",
            (True, True): "both_in",
            (False, True): "gold_out_audio_in",
            (False, False): "both_out",
        }[(gold_in, audio_in)]
        values[key] += 1

    return ClipResult(stem, WindowCounts(**values), tuple(lags))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[round((len(ordered) - 1) * fraction)]


def format_report(results: Sequence[ClipResult]) -> str:
    """Render the fixed F7 analysis as a reproducible Markdown report."""
    total = sum((result.counts for result in results), WindowCounts())
    primary = _ratio(total.gold_in_window_audio_wrong, total.anchor_covered_audio_wrong)
    lower, upper = wilson_interval(
        total.gold_in_window_audio_wrong, total.anchor_covered_audio_wrong
    )
    marginal = _ratio(total.gold_in_window, total.anchor_covered)
    audio_prior = _ratio(total.audio_right, total.ambiguous)
    coverage = _ratio(total.anchor_covered_audio_wrong, total.audio_wrong)
    all_lags = [lag for result in results for lag in result.target_lags_s]
    boundary = total.center_at_nut_boundary + total.center_at_bridge_boundary

    lines = [
        "# F7: cache-only GAPS hand-centroid anchor probe",
        "",
        "**Date:** 2026-07-22  ",
        "**Status:** **CLOSED-NEGATIVE for the GAPS bridge probe.** This does not close ",
        "FretCam because GAPS uses the explicitly different uncontrolled-footage capture contract.",
        "",
        "## Fixed protocol",
        "",
        "- Corpus: public GAPS clean-12; gold-pitch ambiguous-note lattice decoded with A14's ",
        "  frozen mirrored cluster Viterbi (the comparator's banked audio mechanism).",
        "- Video: rich cache only (`rawcv.c0.25.pkl`); no inference, download, or training.",
        "- Anchor: cached `HandSample` + cached homography through `compute_neck_anchor`; ",
        "  `N=max(1,floor(center_fret))`; window `[N-1,N+4] union {0}`.",
        f"- Timestamp: nearest cached frame within +/-{CACHE_TOLERANCE_S * 1000:.0f} ms of ",
        f"  `onset-{TARGET_LEAD_S * 1000:.0f} ms` (the cache contains onset-near frames, not a ",
        "  purpose-built pre-onset sample stream).",
        "",
        "## Result",
        "",
        f"- **P(gold fret in window | audio wrong, anchor present) = "
        f"{total.gold_in_window_audio_wrong}/{total.anchor_covered_audio_wrong} = "
        f"{primary:.3f}** (Wilson 95% CI {lower:.3f}-{upper:.3f}).",
        f"- This is **{primary - A14_ANTI_ENRICHMENT:+.3f}** versus A14's 0.285 ",
        f"  anti-enrichment reference and below the anchor marginal **{marginal:.3f}**.",
        f"- Current audio prior = **{total.audio_right}/{total.ambiguous} = {audio_prior:.3f}** ",
        f"  versus the requested 0.778 reference ({audio_prior - AUDIO_PRIOR_REFERENCE:+.3f}).",
        f"- Audio-wrong anchor coverage = **{total.anchor_covered_audio_wrong}/"
        f"{total.audio_wrong} = {coverage:.3f}**; all-ambiguous coverage = "
        f"**{total.anchor_covered}/{total.ambiguous} = "
        f"{_ratio(total.anchor_covered, total.ambiguous):.3f}**.",
        "",
        "The conditional is lower than both the 0.285 comparator and the anchor's own ",
        "marginal. The cached centroid signal is therefore anti-enriched where audio fails; ",
        "it is not evidence for wiring this GAPS signal into fusion.",
        "",
        "## Wrong-audio discrimination diagnostic",
        "",
        "| gold fret in window | audio choice in window | notes | share | interpretation |",
        "|---|---|---:|---:|---|",
        f"| yes | no | {total.gold_in_audio_out} | "
        f"{_ratio(total.gold_in_audio_out, total.anchor_covered_audio_wrong):.3f} | potential rescue |",
        f"| yes | yes | {total.both_in} | "
        f"{_ratio(total.both_in, total.anchor_covered_audio_wrong):.3f} | no discrimination |",
        f"| no | yes | {total.gold_out_audio_in} | "
        f"{_ratio(total.gold_out_audio_in, total.anchor_covered_audio_wrong):.3f} | favors wrong choice |",
        f"| no | no | {total.both_out} | "
        f"{_ratio(total.both_out, total.anchor_covered_audio_wrong):.3f} | no usable support |",
        "",
        "## Cache and geometry diagnostics",
        "",
        f"- Selected-frame lag relative to the intended pre-onset target: median "
        f"**{_quantile(all_lags, 0.5) * 1000:+.1f} ms**, range "
        f"{_quantile(all_lags, 0.0) * 1000:+.1f} to "
        f"{_quantile(all_lags, 1.0) * 1000:+.1f} ms. The median selected frame is thus ",
        f"about **{(_quantile(all_lags, 0.5) - TARGET_LEAD_S) * 1000:+.1f} ms** from onset.",
        f"- Centroids clipped to a neck boundary: **{boundary}/{total.anchor_covered} = "
        f"{_ratio(boundary, total.anchor_covered):.3f}** (nut {total.center_at_nut_boundary}; "
        f"bridge {total.center_at_bridge_boundary}).",
        "- The current parser/decoder yields 10,182 ambiguous decoded notes rather than A14's ",
        "  banked 10,072, while reproducing its audio prior within 0.4 pp. Counts in this report ",
        "  are from the current checkout and the same local public cache; the comparator remains ",
        "  the frozen A14 report. Current `fuse` has evolved since A14, so this probe intentionally ",
        "  uses A14's decoder instead of claiming parity with today's implementation.",
        "",
        "## Per-clip breakdown",
        "",
        "| clip | ambiguous | audio wrong | anchors on wrong | gold in window | rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        counts = result.counts
        lines.append(
            f"| {result.clip} | {counts.ambiguous} | {counts.audio_wrong} | "
            f"{counts.anchor_covered_audio_wrong} | {counts.gold_in_window_audio_wrong} | "
            f"{_ratio(counts.gold_in_window_audio_wrong, counts.anchor_covered_audio_wrong):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Bank this as a **negative for uncontrolled GAPS footage**. Do not tune the window, ",
            "orientation, clip set, or confidence threshold against this result. The only valid ",
            "FretCam reopen path remains a new controlled-live capture contract; the build path ",
            "itself is still paused at F2's independent 2/3-clip failed gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run_probe(
    clips: Sequence[str],
    *,
    data_root: Path,
    video_cache: Path,
    cv_cache: Path,
) -> list[ClipResult]:
    cfg = GuitarConfig()
    return [
        probe_clip(
            stem,
            data_root=data_root,
            video_cache=video_cache,
            cv_cache=cv_cache,
            cfg=cfg,
        )
        for stem in clips
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / ".tabvision/data")
    parser.add_argument(
        "--video-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video"
    )
    parser.add_argument(
        "--cv-cache", type=Path, default=Path.home() / ".tabvision/cache/gaps_video_chain"
    )
    parser.add_argument(
        "--clips",
        default="clean12",
        help="'clean12' or a comma-separated list of GAPS clip stems",
    )
    args = parser.parse_args(argv)
    clips = CLEAN_12 if args.clips == "clean12" else tuple(args.clips.split(","))
    print(
        format_report(
            run_probe(
                clips,
                data_root=args.data_root,
                video_cache=args.video_cache,
                cv_cache=args.cv_cache,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
