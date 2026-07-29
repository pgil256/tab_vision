"""Stage 1 ceiling probe — segment-level position-window reranking on GAPS clean-12.

Bounds what position-window evidence could *ever* contribute through the
segment reranker, by feeding it gold-derived windows degraded to FretCam-like
statistics (precision 1.0, coverage 0.416, 4 Hz cadence). If this does not
fire, no detector improvement can make the mechanism pay, and gate G1 closes
the line.

Every constant is frozen in ``docs/plans/2026-07-28-segment-position-window-design.md``
§5a, committed before this script was first run. Run from ``tabvision/``::

    TABVISION_DATA_ROOT=~/.tabvision/data python -m scripts.eval.segment_window_stage1 \
        --json ../docs/EVAL_REPORTS/segment_window_stage1_2026-07-29.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.acquire.gaps_video import CLEAN_12
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.metrics import tab_f1
from tabvision.eval.parsers.registry import get_parser
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.position_window_prior import (
    MIN_POSITION_OBSERVATION_CONFIDENCE,
    POSITION_OBSERVATION_LEAD_S,
    POSITION_OBSERVATION_LOOKBACK_S,
    _is_valid_observation,
)
from tabvision.fusion.segment_decoder import SegmentBoundary
from tabvision.fusion.viterbi import decode_segment_v1_with_analysis
from tabvision.pipeline import sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent
from tabvision.video.position import PositionWindowObservation

# ---------------------------------------------------------------- frozen §5a

OBSERVATION_HZ = 4.0
"""Candidate gold-window cadence before coverage degradation."""

COVERAGE = 0.416
"""F5c frozen stable coverage; applied by deterministic Bresenham retention."""

OBSERVATION_CONFIDENCE = 0.26
"""Documented FretCam median (Appendix A). Gates validity only, never scoring."""

LOOKBEHIND_S = 0.25
"""Gold notes this far before a timestamp still describe the observed hand."""

LOOKAHEAD_S = 0.35
"""...and this far after: FretCam sees the hand pre-positioned."""

SEGMENT_ATTRIBUTION_LEAD_S = POSITION_OBSERVATION_LEAD_S + POSITION_OBSERVATION_LOOKBACK_S
"""0.18 s — the bridge's causal window, reused at segment granularity."""

CAP = 1.0
"""Maximum segment-level penalty in nats (inherits MAX_POSITION_LOG_BONUS)."""

WEIGHT = 1.0

K_PATHS = 3

G1_MIN_DELTA = 0.010
G1_MAX_PER_CLIP_REGRESSION = -0.002

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def _event_from_json(payload: dict[str, Any]) -> AudioEvent:
    return AudioEvent(
        onset_s=float(payload["onset_s"]),
        offset_s=float(payload["offset_s"]),
        pitch_midi=int(payload["pitch_midi"]),
        velocity=float(payload["velocity"]),
        confidence=float(payload["confidence"]),
    )


# ------------------------------------------------------- gold-window oracle


def synthesize_gold_windows(
    gold: Sequence[TabEvent],
    cfg: GuitarConfig,
) -> tuple[list[PositionWindowObservation], dict[str, int]]:
    """Gold-derived position windows, degraded to FretCam-like statistics.

    Precision is 1.0 by construction: the window is built from the frets the
    player actually holds. Realism comes from dropping moments a single window
    cannot cover (where real FretCam destabilises) and from thinning the grid
    to the frozen 0.416 coverage.
    """
    if not gold:
        return [], {"grid": 0, "no_fretted": 0, "span_too_wide": 0, "dropped_coverage": 0}

    end_s = max(float(event.onset_s) + float(event.duration_s) for event in gold)
    step = 1.0 / OBSERVATION_HZ
    n_ticks = int(math.floor(end_s / step)) + 1

    stats = {"grid": 0, "no_fretted": 0, "span_too_wide": 0, "dropped_coverage": 0}
    eligible: list[PositionWindowObservation] = []

    for tick in range(n_ticks):
        t = tick * step
        stats["grid"] += 1
        frets = [
            int(event.fret)
            for event in gold
            if int(event.fret) > 0 and -LOOKBEHIND_S <= float(event.onset_s) - t <= LOOKAHEAD_S
        ]
        if not frets:
            stats["no_fretted"] += 1
            continue
        position = min(frets)
        if max(frets) > position + 4:
            stats["span_too_wide"] += 1
            continue
        position = max(1, min(position, cfg.max_fret))
        window = (
            0,
            *range(max(1, position - 1), min(cfg.max_fret, position + 4) + 1),
        )
        eligible.append(
            PositionWindowObservation(
                timestamp_s=t,
                position=position,
                window_frets=window,
                confidence=OBSERVATION_CONFIDENCE,
                state="locked",
            )
        )

    # Deterministic Bresenham retention to the frozen coverage — no RNG.
    retained: list[PositionWindowObservation] = []
    for index, observation in enumerate(eligible):
        if math.floor((index + 1) * COVERAGE) > math.floor(index * COVERAGE):
            retained.append(observation)
        else:
            stats["dropped_coverage"] += 1

    for observation in retained:
        assert _is_valid_observation(observation, cfg), observation
    assert OBSERVATION_CONFIDENCE >= MIN_POSITION_OBSERVATION_CONFIDENCE
    return retained, stats


# ----------------------------------------------------------- segment rerank


def _segment_slices(
    segments: Sequence[SegmentBoundary],
    n_events: int,
) -> list[tuple[int, int]]:
    """Map segments onto flat-event index ranges via their note counts."""
    slices: list[tuple[int, int]] = []
    start = 0
    for segment in segments:
        end = start + int(segment.note_count)
        slices.append((start, end))
        start = end
    if start != n_events:
        raise ValueError(f"segment note counts sum to {start}, expected {n_events}")
    return slices


def path_raw_score(
    events: Sequence[TabEvent],
    segments: Sequence[SegmentBoundary],
    slices: Sequence[tuple[int, int]],
    observations: Sequence[PositionWindowObservation],
    cfg: GuitarConfig,
) -> tuple[float, int]:
    """median agreement over (segment, observation) pairs x log(1 + n_obs)."""
    agreements: list[float] = []
    for segment, (start, end) in zip(segments, slices, strict=True):
        fretted = [
            event
            for event in events[start:end]
            if int(event.fret) > 0 and int(event.fret) != int(cfg.capo)
        ]
        if not fretted:
            continue
        lo = float(segment.start_onset_s) - SEGMENT_ATTRIBUTION_LEAD_S
        hi = float(segment.end_onset_s)
        for observation in observations:
            timestamp = float(observation.timestamp_s)
            if not lo <= timestamp <= hi:
                continue
            supported = frozenset(observation.window_frets)
            hits = sum(1 for event in fretted if int(event.fret) in supported)
            agreements.append(hits / len(fretted))
    if not agreements:
        return 0.0, 0
    return statistics.median(agreements) * math.log(1.0 + len(agreements)), len(agreements)


def rerank(
    paths: Sequence[Any],
    segments: Sequence[SegmentBoundary],
    observations: Sequence[PositionWindowObservation],
    cfg: GuitarConfig,
    n_events: int,
) -> tuple[int, list[float], int]:
    """Return (winning path index, raw scores, contributing observation pairs)."""
    if not observations or len(paths) < 2:
        return 0, [], 0
    slices = _segment_slices(segments, n_events)
    raws: list[float] = []
    pairs = 0
    for path in paths:
        raw, count = path_raw_score(path.events, segments, slices, observations, cfg)
        raws.append(raw)
        pairs = max(pairs, count)
    best_raw = max(raws)
    if all(abs(raw - best_raw) < 1e-12 for raw in raws):
        return 0, raws, pairs
    adjusted = [
        float(path.cost) + min(CAP, WEIGHT * (best_raw - raw))
        for path, raw in zip(paths, raws, strict=True)
    ]
    winner = min(range(len(adjusted)), key=lambda i: (adjusted[i], i))
    return winner, raws, pairs


# ------------------------------------------------------------------- runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", os.path.expanduser("~/.tabvision/data")))
    gaps = args.gaps_root or (data_root / "gaps")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_gaps_cache")

    cfg = GuitarConfig()
    session = SessionConfig(style="fingerstyle")
    prior = load_pitch_position_prior("gaps-v1")
    parse = get_parser("gaps_musicxml_tab")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for clip in CLEAN_12:
        xml_path = gaps / "musicxml" / f"{clip}.xml"
        cache = cache_dir / f"{clip}.ensemble.json"
        if not xml_path.is_file() or not cache.is_file():
            print(f"  {clip}: missing annotation or cached events, skipped", flush=True)
            continue
        gold = list(parse(xml_path, cfg))
        events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
        prepared = apply_pitch_position_prior(events, prior)

        with sequence_decode_context("gaps-seq-v1"):
            decoded = decode_segment_v1_with_analysis(
                prepared, cfg=cfg, session=session, k_paths=K_PATHS
            )
        if not decoded.paths:
            print(f"  {clip}: decoder returned no paths, skipped", flush=True)
            continue

        observations, obs_stats = synthesize_gold_windows(gold, cfg)
        winner, raws, pairs = rerank(
            decoded.paths, decoded.segments, observations, cfg, len(decoded.audio_events)
        )

        base_events = decoded.paths[0].events
        arm_events = decoded.paths[winner].events
        base = tab_f1(base_events, gold)
        arm = tab_f1(arm_events, gold)
        changed = sum(
            1
            for b, a in zip(base_events, arm_events, strict=True)
            if (b.string_idx, b.fret) != (a.string_idx, a.fret)
        )

        rows.append(
            {
                "clip": clip,
                "gold_notes": len(gold),
                "decoded_notes": len(decoded.audio_events),
                "segments": len(decoded.segments),
                "paths": len(decoded.paths),
                "path_margins": [float(p.score_delta_from_best) for p in decoded.paths],
                "observations": len(observations),
                "observation_pairs": pairs,
                "observation_stats": obs_stats,
                "raw_scores": [float(r) for r in raws],
                "winner_path": winner,
                "abstained": winner == 0,
                "notes_changed": changed,
                "base_tab_f1": float(base.f1),
                "arm_tab_f1": float(arm.f1),
                "delta": float(arm.f1 - base.f1),
            }
        )
        print(
            f"  {clip}: base={base.f1:.4f} arm={arm.f1:.4f} "
            f"delta={arm.f1 - base.f1:+.4f} path={winner} "
            f"obs={len(observations)} changed={changed}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no clips scored — build the audio-event cache first")

    deltas = [row["delta"] for row in rows]
    mean_delta = sum(deltas) / len(deltas)
    ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    regressions = [row["clip"] for row in rows if row["delta"] < G1_MAX_PER_CLIP_REGRESSION]
    fired = [row["clip"] for row in rows if not row["abstained"]]
    passed = mean_delta >= G1_MIN_DELTA and not regressions

    summary = {
        "stage": 1,
        "corpus": "GAPS clean-12",
        "clips": len(rows),
        "frozen_constants": {
            "observation_hz": OBSERVATION_HZ,
            "coverage": COVERAGE,
            "observation_confidence": OBSERVATION_CONFIDENCE,
            "cap_nats": CAP,
            "weight": WEIGHT,
            "k_paths": K_PATHS,
            "lookbehind_s": LOOKBEHIND_S,
            "lookahead_s": LOOKAHEAD_S,
            "segment_attribution_lead_s": SEGMENT_ATTRIBUTION_LEAD_S,
        },
        "base_aggregate_tab_f1": sum(r["base_tab_f1"] for r in rows) / len(rows),
        "arm_aggregate_tab_f1": sum(r["arm_tab_f1"] for r in rows) / len(rows),
        "mean_delta": mean_delta,
        "ci95": [ci.lower, ci.upper],
        "clips_reranked": fired,
        "per_clip_regressions": regressions,
        "g1_min_delta": G1_MIN_DELTA,
        "g1_max_per_clip_regression": G1_MAX_PER_CLIP_REGRESSION,
        "g1_pass": passed,
        "wall_seconds": time.perf_counter() - started,
        "per_clip": rows,
    }
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\naggregate base {summary['base_aggregate_tab_f1']:.4f} -> "
        f"arm {summary['arm_aggregate_tab_f1']:.4f}"
    )
    print(f"mean delta {mean_delta:+.4f} [{ci.lower:+.4f}, {ci.upper:+.4f}] over {len(rows)} clips")
    print(f"clips reranked: {len(fired)}/{len(rows)} {fired}")
    print(f"per-clip regressions worse than {G1_MAX_PER_CLIP_REGRESSION}: {regressions or 'none'}")
    print(f"\nG1 (>= {G1_MIN_DELTA:+.3f}, no regression): {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
