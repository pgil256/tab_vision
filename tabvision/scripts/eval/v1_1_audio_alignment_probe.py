"""v1.1 chunk-4 audio transcription/alignment probe for UT-Austin.

Runs one or more audio backends on the Kaggle/UT-Austin WAV clips, caches the
raw note events locally, then scores raw, per-clip-calibrated, and
global-calibrated audio against the UT-Austin tab labels. This is deliberately
audio-first: real video is not touched here. The oracle-video score uses gold
``FrameFingering`` evidence only to measure the ceiling once the audio events
are corrected for pitch/time alignment.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.v1_1_kaggle_oracle_probe import (
    _load_timestamps,
    _oracle_fingerings,
    parse_clip,
)
from tabvision.eval.metrics import EventF1Result, TabF1Result, event_f1, tab_f1
from tabvision.fusion import fuse
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

_DEFAULT_ROOT = (
    Path.home()
    / ".tabvision/data/datasets/guitar-transcription-utaustin"
    / "tablature_dataset/tablature_dataset"
)
_DEFAULT_CACHE_DIR = Path.home() / ".tabvision/cache/v1_1_audio_alignment"
_DEFAULT_REPORT_DIR = Path.home() / ".tabvision/reports/v1_1_audio_alignment"
_DEFAULT_BACKENDS = ("highres", "basicpitch", "highres-fl")


@dataclass(frozen=True)
class AlignmentChoice:
    """Best whole-semitone + time-origin correction for one event set."""

    pitch_shift: int
    time_shift_s: float
    matches: int


@dataclass(frozen=True)
class ScoreBundle:
    """Onset/pitch/tab/oracle scores for one decoded condition."""

    onset_f1: float
    pitch_f1: float
    tab_f1: float
    oracle_tab_f1: float
    decoded_event_count: int


@dataclass(frozen=True)
class ClipScore:
    """All scores emitted for one backend/clip pair."""

    backend: str
    clip_id: str
    status: str
    n_gold: int
    raw_event_count: int
    pitch_shift: int
    time_shift_s: float
    alignment_matches: int
    raw: ScoreBundle
    per_clip: ScoreBundle
    global_calibrated: ScoreBundle | None = None
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = GuitarConfig()
    clip_ids = _resolve_clip_ids(args.root, args.clips)
    backends = _parse_csv(args.backends)

    report = run_probe(
        root=args.root,
        clip_ids=clip_ids,
        backend_names=backends,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        skip_unavailable=not args.no_skip_unavailable,
        cfg=cfg,
        onset_tolerance_s=args.onset_tolerance_s,
        alignment_tolerance_s=args.alignment_tolerance_s,
        max_abs_pitch_shift=args.max_abs_pitch_shift,
        max_abs_time_shift_s=args.max_abs_time_shift_s,
        time_step_s=args.time_step_s,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.report_name}.json"
    md_path = out_dir / f"{args.report_name}.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def run_probe(
    *,
    root: Path,
    clip_ids: Sequence[str],
    backend_names: Sequence[str],
    cache_dir: Path,
    refresh_cache: bool,
    skip_unavailable: bool,
    cfg: GuitarConfig,
    onset_tolerance_s: float,
    alignment_tolerance_s: float,
    max_abs_pitch_shift: int,
    max_abs_time_shift_s: float,
    time_step_s: float,
) -> dict[str, Any]:
    """Run the full audio-alignment probe and return a JSON-ready report."""

    ts = _load_timestamps(root)
    gold_by_clip = {cid: parse_clip(cid, root, ts, cfg) for cid in clip_ids}
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    all_rows: list[ClipScore] = []
    backend_summaries: dict[str, dict[str, Any]] = {}
    backend_errors: dict[str, str] = {}
    for backend_name in backend_names:
        try:
            backend = _make_backend(backend_name)
        except Exception as exc:
            if not skip_unavailable:
                raise
            backend_errors[backend_name] = str(exc)
            backend_summaries[backend_name] = {
                "status": "unavailable",
                "error": str(exc),
                "clips": 0,
            }
            continue

        raw_events_by_clip: dict[str, list[AudioEvent]] = {}
        alignments_by_clip: dict[str, AlignmentChoice] = {}
        score_maps_by_clip: dict[str, dict[tuple[int, float], int]] = {}
        rows_for_backend: list[ClipScore] = []

        for clip_id in clip_ids:
            gold = gold_by_clip[clip_id]
            if not gold:
                rows_for_backend.append(
                    ClipScore(
                        backend=backend_name,
                        clip_id=clip_id,
                        status="skipped_no_gold",
                        n_gold=0,
                        raw_event_count=0,
                        pitch_shift=0,
                        time_shift_s=0.0,
                        alignment_matches=0,
                        raw=_zero_score(),
                        per_clip=_zero_score(),
                        error="no parsed gold events",
                    )
                )
                continue
            try:
                events = cached_backend_events(
                    backend_name=backend_name,
                    backend=backend,
                    clip_id=clip_id,
                    root=root,
                    cache_dir=cache_dir,
                    refresh_cache=refresh_cache,
                )
                alignment, score_map = estimate_audio_alignment(
                    events,
                    gold,
                    tolerance_s=alignment_tolerance_s,
                    max_abs_pitch_shift=max_abs_pitch_shift,
                    max_abs_time_shift_s=max_abs_time_shift_s,
                    time_step_s=time_step_s,
                )
                raw = score_events(
                    events,
                    gold,
                    cfg=cfg,
                    onset_tolerance_s=onset_tolerance_s,
                )
                calibrated = score_events(
                    shift_audio_events(
                        events,
                        alignment.pitch_shift,
                        alignment.time_shift_s,
                    ),
                    gold,
                    cfg=cfg,
                    onset_tolerance_s=onset_tolerance_s,
                )
            except Exception as exc:
                row = ClipScore(
                    backend=backend_name,
                    clip_id=clip_id,
                    status="error",
                    n_gold=len(gold),
                    raw_event_count=0,
                    pitch_shift=0,
                    time_shift_s=0.0,
                    alignment_matches=0,
                    raw=_zero_score(),
                    per_clip=_zero_score(),
                    error=str(exc),
                )
                rows_for_backend.append(row)
                continue

            raw_events_by_clip[clip_id] = events
            alignments_by_clip[clip_id] = alignment
            score_maps_by_clip[clip_id] = score_map
            rows_for_backend.append(
                ClipScore(
                    backend=backend_name,
                    clip_id=clip_id,
                    status="ok",
                    n_gold=len(gold),
                    raw_event_count=len(events),
                    pitch_shift=alignment.pitch_shift,
                    time_shift_s=alignment.time_shift_s,
                    alignment_matches=alignment.matches,
                    raw=raw,
                    per_clip=calibrated,
                )
            )

        global_alignment = estimate_global_alignment(score_maps_by_clip.values())
        finalized_rows: list[ClipScore] = []
        for row in rows_for_backend:
            if row.status != "ok":
                finalized_rows.append(row)
                continue
            events = raw_events_by_clip[row.clip_id]
            global_score = score_events(
                shift_audio_events(
                    events,
                    global_alignment.pitch_shift,
                    global_alignment.time_shift_s,
                ),
                gold_by_clip[row.clip_id],
                cfg=cfg,
                onset_tolerance_s=onset_tolerance_s,
            )
            finalized_rows.append(replace(row, global_calibrated=global_score))

        all_rows.extend(finalized_rows)
        backend_summaries[backend_name] = summarize_backend(
            backend_name,
            finalized_rows,
            global_alignment,
        )

    return {
        "generated_at": generated_at,
        "dataset": "KaggleUTAustin",
        "root": str(root),
        "cache_dir": str(cache_dir),
        "clip_ids": list(clip_ids),
        "backends": list(backend_names),
        "scoring": {
            "onset_tolerance_s": onset_tolerance_s,
            "alignment_tolerance_s": alignment_tolerance_s,
            "max_abs_pitch_shift": max_abs_pitch_shift,
            "max_abs_time_shift_s": max_abs_time_shift_s,
            "time_step_s": time_step_s,
        },
        "summaries": backend_summaries,
        "rows": [clip_score_to_dict(row) for row in all_rows],
        "backend_errors": backend_errors,
        "diagnosis": diagnose_report(backend_summaries),
    }


def cached_backend_events(
    *,
    backend_name: str,
    backend: Any,
    clip_id: str,
    root: Path,
    cache_dir: Path,
    refresh_cache: bool,
) -> list[AudioEvent]:
    """Load backend events for one clip from cache, or run and cache them."""

    wav_path = root / "tablature_audio" / f"{clip_id}.wav"
    cache_path = cache_dir / backend_name / f"{clip_id}.json"
    source_mtime_ns = wav_path.stat().st_mtime_ns
    if not refresh_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("backend") == backend_name
            and cached.get("clip_id") == clip_id
            and cached.get("source_mtime_ns") == source_mtime_ns
        ):
            return events_from_json(cached.get("events", []))

    wav, sr = _load_wav(wav_path)
    events = list(backend.transcribe(wav, sr, SessionConfig()))
    events.sort(key=lambda event: (event.onset_s, event.pitch_midi))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": backend_name,
        "clip_id": clip_id,
        "source_wav": str(wav_path),
        "source_mtime_ns": source_mtime_ns,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "events": events_to_json(events),
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return events


def estimate_audio_alignment(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    tolerance_s: float = 0.12,
    max_abs_pitch_shift: int = 3,
    max_abs_time_shift_s: float = 3.0,
    time_step_s: float = 0.02,
) -> tuple[AlignmentChoice, dict[tuple[int, float], int]]:
    """Estimate whole-semitone and time-origin offsets from pitch matches."""

    if not events or not gold:
        return AlignmentChoice(0, 0.0, 0), {(0, 0.0): 0}

    scores: dict[tuple[int, float], int] = {}
    n_time_steps = int(round(max_abs_time_shift_s / time_step_s))
    shifts = range(-max_abs_pitch_shift, max_abs_pitch_shift + 1)
    for pitch_shift in shifts:
        for offset_step in range(-n_time_steps, n_time_steps + 1):
            time_shift_s = round(offset_step * time_step_s, 10)
            matches = count_pitch_time_matches(
                events,
                gold,
                pitch_shift=pitch_shift,
                time_shift_s=time_shift_s,
                tolerance_s=tolerance_s,
            )
            scores[(pitch_shift, time_shift_s)] = matches

    best_pitch_shift, best_time_shift_s = max(
        scores,
        key=lambda candidate: (
            scores[candidate],
            -abs(candidate[0]),
            -abs(candidate[1]),
        ),
    )
    return (
        AlignmentChoice(
            best_pitch_shift,
            best_time_shift_s,
            scores[(best_pitch_shift, best_time_shift_s)],
        ),
        scores,
    )


def estimate_global_alignment(
    score_maps: Iterable[Mapping[tuple[int, float], int]],
) -> AlignmentChoice:
    """Pick one alignment correction by summing per-clip match-count grids."""

    totals: Counter[tuple[int, float]] = Counter()
    for score_map in score_maps:
        totals.update(score_map)
    if not totals:
        return AlignmentChoice(0, 0.0, 0)
    best_pitch_shift, best_time_shift_s = max(
        totals,
        key=lambda candidate: (
            totals[candidate],
            -abs(candidate[0]),
            -abs(candidate[1]),
        ),
    )
    return AlignmentChoice(
        best_pitch_shift,
        best_time_shift_s,
        int(totals[(best_pitch_shift, best_time_shift_s)]),
    )


def count_pitch_time_matches(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    pitch_shift: int,
    time_shift_s: float,
    tolerance_s: float,
) -> int:
    """Count greedy pitch+time matches after an alignment correction."""

    gold_used = [False] * len(gold)
    matches = 0
    for event in sorted(events, key=lambda item: item.onset_s):
        aligned_onset = event.onset_s + time_shift_s
        aligned_pitch = event.pitch_midi + pitch_shift
        best_j = -1
        best_dt = tolerance_s + 1e-9
        for j, gold_event in enumerate(gold):
            if gold_used[j] or gold_event.pitch_midi != aligned_pitch:
                continue
            dt = abs(gold_event.onset_s - aligned_onset)
            if dt <= tolerance_s and dt < best_dt:
                best_j = j
                best_dt = dt
        if best_j >= 0:
            gold_used[best_j] = True
            matches += 1
    return matches


def shift_audio_events(
    events: Sequence[AudioEvent],
    pitch_shift: int,
    time_shift_s: float = 0.0,
) -> list[AudioEvent]:
    """Return calibrated audio events without mutating backend output."""

    if pitch_shift == 0 and time_shift_s == 0.0:
        return list(events)
    return [
        replace(
            event,
            onset_s=event.onset_s + time_shift_s,
            offset_s=event.offset_s + time_shift_s,
            pitch_midi=event.pitch_midi + pitch_shift,
        )
        for event in events
    ]


def score_events(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig,
    onset_tolerance_s: float,
) -> ScoreBundle:
    """Score audio-only and oracle-video decoded tabs for one clip."""

    decoded = fuse(events, [], cfg)
    oracle_decoded = fuse(events, _oracle_fingerings(list(gold), cfg), cfg)
    onset: EventF1Result = event_f1(
        decoded,
        gold,
        match_pitch=False,
        onset_tolerance_s=onset_tolerance_s,
    )
    pitch: EventF1Result = event_f1(
        decoded,
        gold,
        match_pitch=True,
        onset_tolerance_s=onset_tolerance_s,
    )
    tab: TabF1Result = tab_f1(decoded, gold, onset_tolerance_s=onset_tolerance_s)
    oracle_tab: TabF1Result = tab_f1(
        oracle_decoded,
        gold,
        onset_tolerance_s=onset_tolerance_s,
    )
    return ScoreBundle(
        onset_f1=onset.f1,
        pitch_f1=pitch.f1,
        tab_f1=tab.f1,
        oracle_tab_f1=oracle_tab.f1,
        decoded_event_count=len(decoded),
    )


def summarize_backend(
    backend_name: str,
    rows: Sequence[ClipScore],
    global_alignment: AlignmentChoice,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.status == "ok"]
    skipped_rows = [row for row in rows if row.status == "skipped_no_gold"]
    error_rows = [row for row in rows if row.status == "error"]
    unique_errors = sorted({row.error for row in error_rows if row.error})
    if not ok_rows:
        return {
            "status": "error" if error_rows else "empty",
            "clips": 0,
            "skipped_clips": [row.clip_id for row in skipped_rows],
            "errors": unique_errors,
            "diagnosis": (
                [f"Backend did not score any clips: {unique_errors[0]}"]
                if unique_errors
                else ["No clips were scored."]
            ),
        }

    pitch_counts = Counter(row.pitch_shift for row in ok_rows)
    large_time_rows = [row.clip_id for row in ok_rows if abs(row.time_shift_s) >= 0.5]
    zero_oracle_rows = [row.clip_id for row in ok_rows if row.per_clip.oracle_tab_f1 == 0.0]
    diagnosis = diagnose_backend(ok_rows)
    if skipped_rows:
        skipped = ", ".join(row.clip_id for row in skipped_rows)
        diagnosis.append(f"Skipped clips with no parsed gold events: {skipped}.")
    return {
        "status": "ok" if not error_rows else "partial",
        "backend": backend_name,
        "clips": len(ok_rows),
        "skipped_clips": [row.clip_id for row in skipped_rows],
        "errors": unique_errors,
        "raw_event_count_mean": _mean(row.raw_event_count for row in ok_rows),
        "raw_event_count_total": sum(row.raw_event_count for row in ok_rows),
        "global_alignment": asdict(global_alignment),
        "pitch_shift_counts": dict(sorted(pitch_counts.items())),
        "mode_pitch_shift": pitch_counts.most_common(1)[0][0],
        "large_time_shift_clips": large_time_rows,
        "zero_oracle_clips": zero_oracle_rows,
        "raw": aggregate_scores(row.raw for row in ok_rows),
        "per_clip": aggregate_scores(row.per_clip for row in ok_rows),
        "global_calibrated": aggregate_scores(
            row.global_calibrated for row in ok_rows if row.global_calibrated is not None
        ),
        "diagnosis": diagnosis,
    }


def aggregate_scores(scores: Iterable[ScoreBundle | None]) -> dict[str, Any]:
    present = [score for score in scores if score is not None]
    if not present:
        return {}
    return {
        "onset_f1_mean": _mean(score.onset_f1 for score in present),
        "pitch_f1_mean": _mean(score.pitch_f1 for score in present),
        "tab_f1_mean": _mean(score.tab_f1 for score in present),
        "oracle_tab_f1_mean": _mean(score.oracle_tab_f1 for score in present),
        "decoded_event_count_mean": _mean(score.decoded_event_count for score in present),
    }


def diagnose_backend(rows: Sequence[ClipScore]) -> list[str]:
    """Return short human-readable diagnostic bullets for one backend."""

    if not rows:
        return ["No successful clips were scored."]

    out: list[str] = []
    pitch_counts = Counter(row.pitch_shift for row in rows)
    pitch_mode, pitch_mode_count = pitch_counts.most_common(1)[0]
    if pitch_mode == -1 and pitch_mode_count >= max(2, int(len(rows) * 0.6)):
        out.append(
            "Most clips prefer a -1 semitone correction; treat corpus tuning/reference "
            "pitch as a first-class suspect."
        )
    elif pitch_mode != 0 and pitch_mode_count >= max(2, int(len(rows) * 0.6)):
        out.append(
            f"Most clips prefer a {pitch_mode:+d} semitone correction; inspect "
            "corpus tuning/reference pitch before backend changes."
        )

    large_time = [row for row in rows if abs(row.time_shift_s) >= 0.5]
    large_time_ids = {row.clip_id for row in large_time}
    if {"0", "1"}.issubset(large_time_ids):
        out.append(
            "Clips 0 and 1 both need large time-origin shifts; inspect timestamp "
            "or label origin handling separately from the rest of the corpus."
        )
    elif large_time:
        ids = ", ".join(row.clip_id for row in large_time[:8])
        out.append(f"Large time-origin shifts appear on clips {ids}; inspect alignment.")

    zero_oracle = [row.clip_id for row in rows if row.per_clip.oracle_tab_f1 == 0.0]
    if zero_oracle:
        shown = ", ".join(zero_oracle[:10])
        out.append(
            f"{len(zero_oracle)} clips still have zero oracle-video Tab F1 "
            f"after per-clip calibration ({shown}); suspect thresholds, grouping, "
            "or backend mismatch."
        )

    if not out:
        out.append("No dominant pitch/time calibration failure pattern was detected.")
    return out


def diagnose_report(summaries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Compare backends and return top-level diagnostic bullets."""

    ok = {
        name: summary
        for name, summary in summaries.items()
        if summary.get("status") in {"ok", "partial"}
    }
    if not ok:
        return ["No backend completed successfully."]

    oracle_scores = {
        name: float(summary.get("per_clip", {}).get("oracle_tab_f1_mean", 0.0))
        for name, summary in ok.items()
    }
    best_backend, best_score = max(oracle_scores.items(), key=lambda item: item[1])
    out = [f"Best per-clip oracle-video ceiling is {best_backend} at {best_score:.4f}."]
    global_oracle_scores = {
        name: float(summary.get("global_calibrated", {}).get("oracle_tab_f1_mean", 0.0))
        for name, summary in ok.items()
    }
    best_global_backend, best_global_score = max(
        global_oracle_scores.items(),
        key=lambda item: item[1],
    )
    out.append(
        f"Best global-calibration oracle-video ceiling is {best_global_backend} "
        f"at {best_global_score:.4f}."
    )
    if "highres" in oracle_scores and "basicpitch" in oracle_scores:
        delta = oracle_scores["basicpitch"] - oracle_scores["highres"]
        if delta > 0.01:
            out.append(
                f"Basic Pitch beats highres on oracle ceiling by {delta:.4f}; "
                "do not assume highres is the better UT-Austin backend."
            )
        elif delta < -0.01:
            out.append(f"Highres beats Basic Pitch on oracle ceiling by {-delta:.4f}.")
        else:
            out.append("Basic Pitch and highres are effectively tied on oracle ceiling.")
    return out


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the JSON report as a compact Markdown eval report."""

    lines = [
        "# v1.1 audio alignment probe",
        "",
        f"**Date:** {report['generated_at']}",
        f"**Dataset:** {report['dataset']}",
        f"**Root:** `{report['root']}`",
        f"**Cache:** `{report['cache_dir']}`",
        "",
        "## Summary",
        "",
        "| Backend | Status | Clips | Raw events | Global shift | Global time | "
        "Per-clip onset | Per-clip pitch | Per-clip Tab | Per-clip oracle | "
        "Global oracle | Pitch mode | Large-time clips | Zero-oracle clips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, summary in report["summaries"].items():
        if summary.get("status") == "unavailable":
            lines.append(
                f"| {backend} | unavailable | 0 | - | - | - | - | - | - | - | - | - | - | - |"
            )
            continue
        per_clip = summary.get("per_clip", {})
        global_scores = summary.get("global_calibrated", {})
        alignment = summary.get("global_alignment", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    backend,
                    str(summary.get("status", "")),
                    str(summary.get("clips", 0)),
                    _fmt(summary.get("raw_event_count_total")),
                    _fmt(alignment.get("pitch_shift"), signed=True, precision=0),
                    _fmt(alignment.get("time_shift_s"), signed=True, precision=2),
                    _fmt(per_clip.get("onset_f1_mean")),
                    _fmt(per_clip.get("pitch_f1_mean")),
                    _fmt(per_clip.get("tab_f1_mean")),
                    _fmt(per_clip.get("oracle_tab_f1_mean")),
                    _fmt(global_scores.get("oracle_tab_f1_mean")),
                    _fmt(summary.get("mode_pitch_shift"), signed=True, precision=0),
                    str(len(summary.get("large_time_shift_clips", []))),
                    str(len(summary.get("zero_oracle_clips", []))),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Diagnosis", ""])
    for item in report.get("diagnosis", []):
        lines.append(f"- {item}")
    for backend, summary in report["summaries"].items():
        for item in summary.get("diagnosis", []):
            lines.append(f"- {backend}: {item}")

    lines.extend(["", "## Per-Clip Results", ""])
    by_backend: dict[str, list[Mapping[str, Any]]] = {}
    for row in report["rows"]:
        by_backend.setdefault(row["backend"], []).append(row)
    for backend, rows in by_backend.items():
        lines.extend(
            [
                f"### {backend}",
                "",
                "| Clip | Gold | Raw events | Pitch shift | Time shift | Align matches | "
                "Onset | Pitch | Tab | Oracle | Raw pitch | Raw oracle |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda item: int(item["clip_id"])):
            if row["status"] != "ok":
                label = "skipped" if row["status"] == "skipped_no_gold" else "error"
                lines.append(
                    f"| {row['clip_id']} | {row['n_gold']} | {label} | - | - | - | "
                    f"- | - | - | - | - | - |"
                )
                continue
            per_clip = row["per_clip"]
            raw = row["raw"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["clip_id"],
                        str(row["n_gold"]),
                        str(row["raw_event_count"]),
                        _fmt(row["pitch_shift"], signed=True, precision=0),
                        _fmt(row["time_shift_s"], signed=True, precision=2),
                        str(row["alignment_matches"]),
                        _fmt(per_clip["onset_f1"]),
                        _fmt(per_clip["pitch_f1"]),
                        _fmt(per_clip["tab_f1"]),
                        _fmt(per_clip["oracle_tab_f1"]),
                        _fmt(raw["pitch_f1"]),
                        _fmt(raw["oracle_tab_f1"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    lines.append(
        "- Per-clip metrics apply each clip's best whole-semitone and time-origin correction."
    )
    lines.append("- Global metrics apply one correction shared across all clips for that backend.")
    lines.append(
        "- Oracle-video Tab F1 uses gold string/fret evidence, so low oracle scores point "
        "back to audio event quality/alignment."
    )
    return "\n".join(lines) + "\n"


def clip_score_to_dict(row: ClipScore) -> dict[str, Any]:
    payload = asdict(row)
    return payload


def events_to_json(events: Sequence[AudioEvent]) -> list[dict[str, Any]]:
    return [
        {
            "onset_s": event.onset_s,
            "offset_s": event.offset_s,
            "pitch_midi": event.pitch_midi,
            "velocity": event.velocity,
            "confidence": event.confidence,
            "tags": list(event.tags),
        }
        for event in events
    ]


def events_from_json(payload: Sequence[Mapping[str, Any]]) -> list[AudioEvent]:
    return [
        AudioEvent(
            onset_s=float(item["onset_s"]),
            offset_s=float(item["offset_s"]),
            pitch_midi=int(item["pitch_midi"]),
            velocity=float(item["velocity"]),
            confidence=float(item["confidence"]),
            tags=tuple(str(tag) for tag in item.get("tags", ())),
        )
        for item in payload
    ]


def _make_backend(name: str) -> Any:
    from tabvision.audio.backend import make

    return make(name)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), always_2d=False)
    wav_arr = np.asarray(wav, dtype=np.float32)
    if wav_arr.ndim == 2:
        wav_arr = wav_arr.mean(axis=1)
    return wav_arr, int(sr)


def _resolve_clip_ids(root: Path, clips: str | None) -> list[str]:
    if clips:
        return _parse_csv(clips)
    label_dir = root / "tablature_labels"
    return sorted((path.stem for path in label_dir.glob("*.npy")), key=int)


def _parse_csv(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = value
    return [item.strip() for item in raw_items if item.strip()]


def _zero_score() -> ScoreBundle:
    return ScoreBundle(
        onset_f1=0.0,
        pitch_f1=0.0,
        tab_f1=0.0,
        oracle_tab_f1=0.0,
        decoded_event_count=0,
    )


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(
    value: Any,
    *,
    signed: bool = False,
    precision: int = 4,
) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    if isinstance(value, float):
        if precision == 0:
            return f"{value:+.0f}" if signed else f"{value:.0f}"
        sign = "+" if signed else ""
        return f"{value:{sign}.{precision}f}"
    return str(value)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--clips", default=None, help="comma-separated clip ids; default all")
    ap.add_argument(
        "--backends",
        default=",".join(_DEFAULT_BACKENDS),
        help="comma-separated backend names",
    )
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument(
        "--no-skip-unavailable",
        action="store_true",
        help="raise if a backend cannot be imported or run",
    )
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    ap.add_argument(
        "--report-name",
        default=f"v1_1_audio_alignment_probe_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    ap.add_argument("--onset-tolerance-s", type=float, default=0.05)
    ap.add_argument("--alignment-tolerance-s", type=float, default=0.12)
    ap.add_argument("--max-abs-pitch-shift", type=int, default=3)
    ap.add_argument("--max-abs-time-shift-s", type=float, default=3.0)
    ap.add_argument("--time-step-s", type=float, default=0.02)
    return ap.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
