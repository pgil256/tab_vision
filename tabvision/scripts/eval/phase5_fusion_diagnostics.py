"""Phase 5 fusion-consumption diagnostic.

Usage:
    python -m scripts.eval.phase5_fusion_diagnostics --clip-id training-01
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import numpy as np

from tabvision.fusion.candidates import Candidate, candidate_positions

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_INDEX = (
    REPO_ROOT / "tabvision-server" / "tests" / "fixtures" / "benchmarks" / "index.json"
)
DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0)
EPS = 1e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect whether Phase 5 fusion consumes video evidence."
    )
    parser.add_argument("--clip-id", default="training-01")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
        help="lambda_vision values to decode; 0.0 is audio-only.",
    )
    parser.add_argument("--sample-events", type=int, default=5)
    args = parser.parse_args(argv)

    bench = None if args.video is not None else _benchmark_for_clip(args.clip_id)
    video = args.video if args.video is not None else REPO_ROOT / bench["video_path"]
    report = diagnose_fusion_consumption(
        video,
        benchmark=bench,
        lambdas=args.lambdas,
        sample_events=args.sample_events,
    )
    print(_format_report(args.clip_id, video, report))
    return 0


def diagnose_fusion_consumption(
    video: Path,
    *,
    benchmark: dict | None = None,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    sample_events: int = 5,
) -> dict:
    from tabvision.audio.backend import make as make_audio_backend
    from tabvision.demux import demux
    from tabvision.fusion import apply_neck_anchor_priors, fuse, playability
    from tabvision.pipeline import (
        _make_fretboard_backend,
        _make_guitar_backend,
        _make_hand_backend,
        _run_video_stack,
    )
    from tabvision.types import GuitarConfig, SessionConfig

    cfg = GuitarConfig()
    session = SessionConfig()

    audio_demuxed = demux(video)
    audio_backend = make_audio_backend("highres")
    audio_events = list(
        audio_backend.transcribe(audio_demuxed.wav, audio_demuxed.sample_rate, session)
    )

    video_demuxed = demux(video)
    hand_backend = _make_hand_backend()
    try:
        video_result = _run_video_stack(
            video_demuxed.frame_iterator,
            stride=3,
            cfg=cfg,
            guitar_backend=_make_guitar_backend(),
            fretboard_backend=_make_fretboard_backend(),
            hand_backend=hand_backend,
        )
    finally:
        close = getattr(hand_backend, "close", None)
        if close is not None:
            close()

    fingerings = video_result.fingerings
    anchors = video_result.neck_anchors
    enriched_events = apply_neck_anchor_priors(audio_events, anchors, cfg)
    gold_events, aligned_gold, gold_offset, gold_matches = _aligned_gold_events(
        benchmark=benchmark,
        audio_events=audio_events,
        video_duration_s=audio_demuxed.duration_s,
    )

    prior_events = [ev for ev in enriched_events if ev.fret_prior is not None]
    nearby_fingerings = [
        playability.find_fingering_at(ev.onset_s, fingerings) for ev in audio_events
    ]
    nearby_count = sum(f is not None for f in nearby_fingerings)
    both_count = sum(
        ev.fret_prior is not None and f is not None
        for ev, f in zip(enriched_events, nearby_fingerings, strict=True)
    )

    audio_only = list(fuse(audio_events, [], cfg, session, lambda_vision=0.0))
    decoded = {0.0: audio_only}
    for lambda_vision in lambdas:
        if lambda_vision == 0.0:
            continue
        decoded[lambda_vision] = list(
            fuse(
                enriched_events,
                fingerings,
                cfg,
                session,
                lambda_vision=lambda_vision,
            )
        )

    decode_rows = []
    for lambda_vision in sorted(decoded):
        tabs = decoded[lambda_vision]
        decode_rows.append(
            {
                "lambda": lambda_vision,
                "tab_events": len(tabs),
                "different_from_audio_only": _position_differences(audio_only, tabs),
                "unique_positions": len({(t.string_idx, t.fret) for t in tabs}),
            }
        )

    return {
        "audio_events": len(audio_events),
        "fingerings": len(fingerings),
        "anchors": len(anchors),
        "prior_events": len(prior_events),
        "nearby_fingering_events": nearby_count,
        "both_prior_and_fingering": both_count,
        "prior_stats": _prior_stats(enriched_events, cfg),
        "gold_events": len(gold_events),
        "aligned_gold_events": len(aligned_gold),
        "gold_offset_s": gold_offset,
        "gold_alignment_matches": gold_matches,
        "posterior_gold_alignment": _posterior_gold_alignment_stats(
            enriched_events,
            fingerings,
            aligned_gold,
            cfg,
        ),
        "decode_rows": decode_rows,
        "samples": _sample_emission_terms(
            enriched_events,
            fingerings,
            cfg,
            aligned_gold,
            sample_events=sample_events,
        ),
    }


def _benchmark_for_clip(clip_id: str) -> dict:
    if not BENCHMARK_INDEX.exists():
        raise FileNotFoundError(f"benchmark index not found: {BENCHMARK_INDEX}")
    benchmarks = json.loads(BENCHMARK_INDEX.read_text()).get("benchmarks", [])
    for bench in benchmarks:
        if bench.get("id") == clip_id:
            video = REPO_ROOT / bench["video_path"]
            if not video.exists():
                raise FileNotFoundError(f"benchmark video not found: {video}")
            return bench
    raise KeyError(f"benchmark clip id not found: {clip_id}")


def _position_differences(base: Sequence, other: Sequence) -> int:
    n = min(len(base), len(other))
    diffs = sum(
        (base[i].string_idx, base[i].fret) != (other[i].string_idx, other[i].fret) for i in range(n)
    )
    return diffs + abs(len(base) - len(other))


def _prior_stats(events: Sequence, cfg) -> dict:
    values = []
    for ev in events:
        if ev.fret_prior is None:
            continue
        arr = np.asarray(ev.fret_prior, dtype=np.float64)
        for c in candidate_positions(ev.pitch_midi, cfg):
            values.append(float(arr[c.string_idx, c.fret]))
    return _stats(values)


def _sample_emission_terms(
    events: Sequence,
    fingerings: Sequence,
    cfg,
    aligned_gold: Sequence,
    *,
    sample_events: int,
) -> list[dict]:
    from tabvision.fusion import playability

    samples = []
    for ev in events:
        candidates = candidate_positions(ev.pitch_midi, cfg)
        fingering = playability.find_fingering_at(ev.onset_s, fingerings)
        if ev.fret_prior is None or fingering is None or not candidates:
            continue

        marginal = fingering.marginal_string_fret()
        rows = [_candidate_terms(ev, c, marginal) for c in candidates]
        gold = _nearest_gold_event(ev, aligned_gold)
        samples.append(
            {
                "onset_s": float(ev.onset_s),
                "pitch_midi": int(ev.pitch_midi),
                "candidate_count": len(candidates),
                "nearest_gold": _gold_summary(ev, gold),
                "top_posterior_cells": _top_posterior_cells(marginal),
                "same_pitch_candidates": [
                    {
                        "string_idx": c.string_idx,
                        "fret": c.fret,
                        "vision_prob": float(marginal[c.string_idx, c.fret]),
                        "prior": float(np.asarray(ev.fret_prior)[c.string_idx, c.fret]),
                    }
                    for c in candidates
                ],
                "best_prior": _best(rows, "prior_cost"),
                "best_vision": _best(rows, "vision_cost"),
                "best_total_lambda1": _best(rows, "total_lambda1"),
                "prior_cost": _stats([r["prior_cost"] for r in rows]),
                "vision_cost": _stats([r["vision_cost"] for r in rows]),
                "low_fret_open_cost": _stats([r["low_fret_open_cost"] for r in rows]),
            }
        )
        if len(samples) >= sample_events:
            break
    return samples


def _posterior_gold_alignment_stats(
    events: Sequence,
    fingerings: Sequence,
    aligned_gold: Sequence,
    cfg,
    *,
    max_dt_s: float = 0.15,
) -> dict:
    from tabvision.fusion import playability

    matched = []
    for ev in events:
        gold = _nearest_gold_event(ev, aligned_gold)
        if gold is None or gold.pitch_midi != ev.pitch_midi:
            continue
        dt = abs(ev.onset_s - gold.onset_s)
        if dt > max_dt_s:
            continue
        fingering = playability.find_fingering_at(ev.onset_s, fingerings)
        if fingering is None:
            continue
        marginal = fingering.marginal_string_fret()
        candidates = candidate_positions(ev.pitch_midi, cfg)
        if not candidates:
            continue

        gold_prob = float(marginal[gold.string_idx, gold.fret])
        global_rank = _rank_cell(marginal, gold.string_idx, gold.fret)
        candidate_probs = [(c, float(marginal[c.string_idx, c.fret])) for c in candidates]
        candidate_probs.sort(key=lambda item: item[1], reverse=True)
        same_pitch_rank = next(
            i + 1
            for i, (c, _prob) in enumerate(candidate_probs)
            if c.string_idx == gold.string_idx and c.fret == gold.fret
        )
        top_global = _top_posterior_cells(marginal, n=1)[0]
        top_same_pitch, top_same_pitch_prob = candidate_probs[0]
        matched.append(
            {
                "dt_s": float(ev.onset_s - gold.onset_s),
                "gold_string_idx": int(gold.string_idx),
                "gold_fret": int(gold.fret),
                "gold_prob": gold_prob,
                "global_rank": global_rank,
                "same_pitch_rank": same_pitch_rank,
                "top_global": top_global,
                "top_same_pitch": {
                    "string_idx": int(top_same_pitch.string_idx),
                    "fret": int(top_same_pitch.fret),
                    "prob": float(top_same_pitch_prob),
                },
                "is_open": gold.fret == 0,
            }
        )

    return {
        "matched_events": len(matched),
        "all": _posterior_rank_summary(matched),
        "open": _posterior_rank_summary([m for m in matched if m["is_open"]]),
        "fretted": _posterior_rank_summary([m for m in matched if not m["is_open"]]),
        "examples": matched[:8],
    }


def _rank_cell(marginal: np.ndarray, string_idx: int, fret: int) -> int:
    target = float(marginal[string_idx, fret])
    return int(np.sum(marginal > target) + 1)


def _posterior_rank_summary(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "gold_prob": _stats([]),
            "global_rank": _stats([]),
            "same_pitch_rank": _stats([]),
            "global_top1": 0,
            "global_top5": 0,
            "same_pitch_top1": 0,
        }
    return {
        "count": len(rows),
        "gold_prob": _stats([r["gold_prob"] for r in rows]),
        "global_rank": _stats([float(r["global_rank"]) for r in rows]),
        "same_pitch_rank": _stats([float(r["same_pitch_rank"]) for r in rows]),
        "global_top1": sum(r["global_rank"] == 1 for r in rows),
        "global_top5": sum(r["global_rank"] <= 5 for r in rows),
        "same_pitch_top1": sum(r["same_pitch_rank"] == 1 for r in rows),
    }


def _aligned_gold_events(
    *,
    benchmark: dict | None,
    audio_events: Sequence,
    video_duration_s: float,
) -> tuple[list, list, float, int]:
    if benchmark is None:
        return [], [], 0.0, 0

    from tests.eval.test_phase5_eval import (
        _align_gold_to_audio_only,
        _load_gold_tab_events,
    )

    gold_path = REPO_ROOT / benchmark["ground_truth_path"]
    gold = _load_gold_tab_events(
        gold_path,
        bpm=benchmark.get("bpm"),
        video_duration_s=video_duration_s,
    )
    aligned, offset_s, matches = _align_gold_to_audio_only(
        audio_only=[],
        gold=gold,
        video_duration_s=video_duration_s,
    )
    if audio_events:
        audio_like = [
            _PitchTimeEvent(onset_s=ev.onset_s, pitch_midi=ev.pitch_midi) for ev in audio_events
        ]
        aligned, offset_s, matches = _align_gold_to_audio_only(
            audio_only=audio_like,
            gold=gold,
            video_duration_s=video_duration_s,
        )
    return gold, aligned, offset_s, matches


class _PitchTimeEvent:
    def __init__(self, *, onset_s: float, pitch_midi: int) -> None:
        self.onset_s = onset_s
        self.pitch_midi = pitch_midi


def _nearest_gold_event(event, aligned_gold: Sequence):
    if not aligned_gold:
        return None
    pitch_matches = [g for g in aligned_gold if g.pitch_midi == event.pitch_midi]
    pool = pitch_matches or aligned_gold
    return min(pool, key=lambda g: abs(g.onset_s - event.onset_s))


def _gold_summary(event, gold) -> dict | None:
    if gold is None:
        return None
    return {
        "dt_s": float(event.onset_s - gold.onset_s),
        "string_idx": int(gold.string_idx),
        "fret": int(gold.fret),
        "pitch_midi": int(gold.pitch_midi),
    }


def _top_posterior_cells(marginal: np.ndarray, *, n: int = 5) -> list[dict]:
    flat_order = np.argsort(marginal.reshape(-1))[::-1][:n]
    out = []
    for flat_idx in flat_order:
        string_idx, fret = np.unravel_index(int(flat_idx), marginal.shape)
        out.append(
            {
                "string_idx": int(string_idx),
                "fret": int(fret),
                "prob": float(marginal[string_idx, fret]),
            }
        )
    return out


def _candidate_terms(event, candidate: Candidate, marginal: np.ndarray) -> dict:
    from tabvision.fusion import playability

    prior = float(np.asarray(event.fret_prior)[candidate.string_idx, candidate.fret])
    vision_prob = float(marginal[candidate.string_idx, candidate.fret])
    low_fret_open = playability.LOW_FRET_BIAS * candidate.fret
    if candidate.fret == 0:
        low_fret_open -= playability.OPEN_STRING_BONUS
    prior_cost = -math.log(max(prior, EPS))
    vision_cost = -math.log(max(vision_prob, playability.VISION_FLOOR))
    return {
        "string_idx": candidate.string_idx,
        "fret": candidate.fret,
        "prior": prior,
        "vision_prob": vision_prob,
        "prior_cost": prior_cost,
        "vision_cost": vision_cost,
        "low_fret_open_cost": low_fret_open,
        "total_lambda1": prior_cost + vision_cost + low_fret_open,
    }


def _best(rows: Sequence[dict], key: str) -> dict:
    row = min(rows, key=lambda r: r[key])
    return {
        "string_idx": row["string_idx"],
        "fret": row["fret"],
        "cost": row[key],
        "prior": row["prior"],
        "vision_prob": row["vision_prob"],
    }


def _stats(values: Sequence[float]) -> dict:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": fmean(values), "max": max(values)}


def _format_report(clip_id: str, video: Path, report: dict) -> str:
    lines = [
        f"clip={clip_id}",
        f"video={video}",
        f"audio_events={report['audio_events']}",
        f"fingerings={report['fingerings']} anchors={report['anchors']}",
        (
            f"fret_prior_events={report['prior_events']}/{report['audio_events']} "
            f"nearby_fingering_events={report['nearby_fingering_events']}/"
            f"{report['audio_events']} both={report['both_prior_and_fingering']}/"
            f"{report['audio_events']}"
        ),
        (
            f"gold_events={report['gold_events']} aligned_gold_events="
            f"{report['aligned_gold_events']} gold_offset_s="
            f"{report['gold_offset_s']:.2f} gold_alignment_matches="
            f"{report['gold_alignment_matches']}"
        ),
        _stat_line("candidate_prior_probability", report["prior_stats"]),
        "posterior_vs_aligned_gold:",
        _posterior_summary_line("all", report["posterior_gold_alignment"]["all"]),
        _posterior_summary_line("open", report["posterior_gold_alignment"]["open"]),
        _posterior_summary_line(
            "fretted",
            report["posterior_gold_alignment"]["fretted"],
        ),
        "decode_by_lambda:",
    ]
    for row in report["decode_rows"]:
        lines.append(
            "  "
            f"lambda={row['lambda']:.2f} tab_events={row['tab_events']} "
            f"diffs_vs_audio_only={row['different_from_audio_only']} "
            f"unique_positions={row['unique_positions']}"
        )
    lines.append("sample_emission_terms:")
    for sample in report["samples"]:
        lines.append(
            "  "
            f"t={sample['onset_s']:.3f} pitch={sample['pitch_midi']} "
            f"candidates={sample['candidate_count']}"
        )
        if sample["nearest_gold"] is not None:
            gold = sample["nearest_gold"]
            lines.append(
                "    "
                f"nearest_gold: dt={gold['dt_s']:+.3f}s "
                f"s={gold['string_idx']} f={gold['fret']} "
                f"pitch={gold['pitch_midi']}"
            )
        lines.append("    top_posterior:")
        for cell in sample["top_posterior_cells"]:
            lines.append(f"      s={cell['string_idx']} f={cell['fret']} p={cell['prob']:.6f}")
        lines.append("    same_pitch_candidates:")
        for c in sample["same_pitch_candidates"]:
            lines.append(
                "      "
                f"s={c['string_idx']} f={c['fret']} "
                f"vision={c['vision_prob']:.6f} prior={c['prior']:.6f}"
            )
        for label in ("best_prior", "best_vision", "best_total_lambda1"):
            best = sample[label]
            lines.append(
                "    "
                f"{label}: s={best['string_idx']} f={best['fret']} "
                f"cost={best['cost']:.3f} prior={best['prior']:.6f} "
                f"vision={best['vision_prob']:.6f}"
            )
        lines.append("    " + _stat_line("prior_cost", sample["prior_cost"]))
        lines.append("    " + _stat_line("vision_cost", sample["vision_cost"]))
        lines.append("    " + _stat_line("low_fret_open_cost", sample["low_fret_open_cost"]))
    return "\n".join(lines)


def _posterior_summary_line(label: str, stats: dict) -> str:
    if stats["count"] == 0:
        return f"  {label}: count=0"
    return (
        f"  {label}: count={stats['count']} "
        f"gold_prob_mean={stats['gold_prob']['mean']:.6f} "
        f"global_rank_mean={stats['global_rank']['mean']:.2f} "
        f"same_pitch_rank_mean={stats['same_pitch_rank']['mean']:.2f} "
        f"global_top1={stats['global_top1']}/{stats['count']} "
        f"global_top5={stats['global_top5']}/{stats['count']} "
        f"same_pitch_top1={stats['same_pitch_top1']}/{stats['count']}"
    )


def _stat_line(label: str, stats: dict) -> str:
    if stats["min"] is None:
        return f"{label}: none"
    return f"{label}: min={stats['min']:.6f} mean={stats['mean']:.6f} max={stats['max']:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
