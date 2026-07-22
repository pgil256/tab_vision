"""Accuracy-loop Q4 (ROI deep-dive §3.3) — standing second-opinion bench.

N1/N2 built the methodology; Q1 found the hole in it. Complementarity —
P(candidate right | ensemble wrong) — counts rescues and charges **nothing**
for the false notes a merge would admit alongside them. MuScriptor passed
that gate by 3.8x (0.3818) and still produced no admissible merge, because
its admitted notes were only 0.181 precise against a 0.6855-precision stream
(`n2_muscriptor_merge_pilot_2026-07-21.md`).

So this bench gates on **both legs**, measured in the same offline replay:

1. ``P(candidate right | ensemble wrong) >= 0.10`` — is there anything to gain;
2. ``added-note precision >= 0.5`` under the candidate's best admission rule
   — can the gain be separated from the noise it arrives with.

Leg 2 is free once events are banked, and it is the leg that kills merges.

Candidates are in-repo audio backends (Basic Pitch is Apache-2.0 and already
registered), so unlike N2 there is no isolated venv or CLI shelling. The
ensemble side reuses the ``AudioEvent`` cache banked by
``n2_muscriptor_merge.py``, so a new candidate costs only its own inference.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.eval.n2_muscriptor_merge import (
    COMPLEMENTARITY_GATE,
    VARIANTS,
    _complementarity_summary,
    _event_from_json,
    _event_to_json,
    _score,
    _variant_summary,
    added_note_yield,
    build_oof_priors,
    gold_hits,
    merge_events,
    select_clips,
)
from tabvision.eval.error_decomposition import ErrorDecomposition
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig

ADDED_PRECISION_GATE = 0.50


def candidate_cache(workdir: Path, track_id: str, candidate: str) -> Path:
    return workdir / f"{track_id}.{candidate}.json"


def ensemble_cache(workdir: Path, track_id: str) -> Path:
    return workdir / f"{track_id}.ensemble.json"


def run_cache_stage(
    clips: list[str],
    *,
    candidate: str,
    data_home: Path,
    workdir: Path,
) -> None:
    """Bank the candidate backend's events for every clip (resumable)."""
    import soundfile as sf

    from tabvision.audio.backend import make

    pending = [
        track_id
        for track_id in clips
        if not candidate_cache(workdir, track_id, candidate).is_file()
    ]
    if not pending:
        print(f"{candidate}: all {len(clips)} clips already cached", flush=True)
        return

    session = SessionConfig()
    backend = make(candidate)
    try:
        for track_id in pending:
            wav_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
            wav, sr = sf.read(wav_path, dtype="float32")
            started = time.perf_counter()
            events = list(backend.transcribe(wav, int(sr), session))
            seconds = time.perf_counter() - started
            candidate_cache(workdir, track_id, candidate).write_text(
                json.dumps([_event_to_json(event) for event in events], indent=1) + "\n",
                encoding="utf-8",
            )
            print(f"{track_id}: {candidate} {len(events)} events ({seconds:.0f}s)", flush=True)
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            closer()


def run_sweep_stage(
    clips: list[str],
    *,
    candidate: str,
    data_home: Path,
    workdir: Path,
) -> dict[str, Any]:
    """Offline replay: both gate legs plus the merge-variant sweep."""
    cfg = GuitarConfig()
    session = SessionConfig()
    oof_priors = build_oof_priors(data_home, cfg)

    per_clip: list[dict[str, Any]] = []
    per_variant_scores: dict[str, list[dict[str, float]]] = {v.name: [] for v in VARIANTS}
    per_variant_decomp: dict[str, list[ErrorDecomposition]] = {v.name: [] for v in VARIANTS}

    for track_id in clips:
        ensemble_path = ensemble_cache(workdir, track_id)
        candidate_path = candidate_cache(workdir, track_id, candidate)
        if not ensemble_path.is_file():
            raise SystemExit(f"missing ensemble cache for {track_id} (run the N2 cache stage)")
        if not candidate_path.is_file():
            raise SystemExit(f"missing {candidate} cache for {track_id}; run --stage cache first")

        ensemble: list[AudioEvent] = [
            _event_from_json(item) for item in json.loads(ensemble_path.read_text("utf-8"))
        ]
        opinion: list[AudioEvent] = [
            _event_from_json(item) for item in json.loads(candidate_path.read_text("utf-8"))
        ]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        prior = oof_priors[track_id[:2]]

        ens_hits = gold_hits(gold, ensemble)
        cand_hits = gold_hits(gold, opinion)
        ens_wrong = sum(1 for hit in ens_hits if not hit)
        rescued = sum(1 for ens, cand in zip(ens_hits, cand_hits, strict=True) if not ens and cand)

        row: dict[str, Any] = {
            "track_id": track_id,
            "mode": "solo" if track_id.endswith("_solo") else "comp",
            "gold": len(gold),
            "ens_events": len(ensemble),
            "candidate_events": len(opinion),
            "ens_wrong": ens_wrong,
            "ms_rescued": rescued,
            "ens_recall": sum(ens_hits) / len(gold) if gold else 0.0,
            "candidate_recall": sum(cand_hits) / len(gold) if gold else 0.0,
        }
        for variant in VARIANTS:
            merged, added = merge_events(ensemble, opinion, variant)
            metrics, decomposition = _score(merged, gold, cfg=cfg, session=session, prior=prior)
            metrics["added"] = float(len(added))
            metrics["added_true"] = float(added_note_yield(added, gold, ens_hits))
            per_variant_scores[variant.name].append(metrics)
            per_variant_decomp[variant.name].append(decomposition)
            row[variant.name] = metrics
        per_clip.append(row)
        print(
            f"{track_id}: rescued={rescued}/{ens_wrong} "
            f"tab_f1 base={row['ensemble']['tab_f1']:.4f} "
            f"cluster={row['cluster']['tab_f1']:.4f}",
            flush=True,
        )

    complementarity = _complementarity_summary(per_clip)
    variants = _variant_summary(per_variant_scores, per_variant_decomp)

    # Leg 2 is judged on the admission rule that actually adds notes and does
    # best on Tab F1 — a rule that admits nothing has undefined precision and
    # cannot carry a merge.
    admitting = [
        (name, payload)
        for name, payload in variants.items()
        if name != "ensemble" and payload["added_notes"] > 0
    ]
    best_rule = max(admitting, key=lambda item: item[1]["tab_f1_delta"])[0] if admitting else None
    leg1 = complementarity["pooled"]["complementarity"]
    leg2 = variants[best_rule]["added_precision"] if best_rule else float("nan")
    return {
        "candidate": candidate,
        "clips": clips,
        "complementarity": complementarity,
        "variants": variants,
        "per_clip": per_clip,
        "gate": {
            "leg1_complementarity": leg1,
            "leg1_threshold": COMPLEMENTARITY_GATE,
            "leg1_pass": bool(leg1 >= COMPLEMENTARITY_GATE),
            "leg2_rule": best_rule,
            "leg2_added_precision": leg2,
            "leg2_threshold": ADDED_PRECISION_GATE,
            "leg2_pass": bool(leg2 >= ADDED_PRECISION_GATE),
            "pass": bool(leg1 >= COMPLEMENTARITY_GATE and leg2 >= ADDED_PRECISION_GATE),
        },
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    candidate = summary["candidate"]
    gate = summary["gate"]
    complementarity = summary["complementarity"]
    variants = summary["variants"]
    lines = [
        f"# Q4 second-opinion bench — {candidate} vs `highres-ensemble`",
        "",
        f"{len(summary['clips'])} GuitarSet dev clips (10 comp + 10 solo), offline "
        "replay of banked events; shipped clean-acoustic decode with the "
        "leave-one-player-out position prior + `guitarset-seq-v1` @ w=4.0.",
        "",
        "## Two-leg gate",
        "",
        "| leg | quantity | value | threshold | verdict |",
        "|---|---|---:|---:|---|",
        f"| 1 | P({candidate} right \\| ensemble wrong) | {gate['leg1_complementarity']:.4f} "
        f"| ≥ {gate['leg1_threshold']:.2f} | {'PASS' if gate['leg1_pass'] else 'FAIL'} |",
        f"| 2 | added-note precision (`{gate['leg2_rule']}`) | "
        f"{gate['leg2_added_precision']:.4f} | ≥ {gate['leg2_threshold']:.2f} "
        f"| {'PASS' if gate['leg2_pass'] else 'FAIL'} |",
        "",
        f"**Verdict: {'PASS — merge work justified' if gate['pass'] else 'FAIL — close'}**",
        "",
        "## Complementarity by mode",
        "",
        "| mode | clips | gold notes | ensemble wrong | rescued | complementarity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("solo", "comp", "pooled"):
        row = complementarity[mode]
        lines.append(
            f"| {mode} | {row['clips']} | {row['gold_notes']} | {row['ensemble_wrong']} "
            f"| {row['rescued']} | {row['complementarity']:.4f} |"
        )
    lines += [
        "",
        "## Merge variants (paired bootstrap vs ensemble alone)",
        "",
        "| variant | added | of which real | added precision | Tab F1 | ΔTab F1 [lo-95, hi-95] "
        "| onset F1 | Δonset F1 [lo-95, hi-95] |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for variant in VARIANTS:
        row = variants[variant.name]
        precision = row["added_precision"]
        cell = "—" if precision != precision else f"{precision:.3f}"
        lines.append(
            f"| `{variant.name}` | {row['added_notes']} | {row['added_true_notes']} | {cell} "
            f"| {row['tab_f1_mean']:.4f} "
            f"| {row['tab_f1_delta']:+.4f} [{row['tab_f1_delta_lo95']:+.4f}, "
            f"{row['tab_f1_delta_hi95']:+.4f}] | {row['onset_f1_mean']:.4f} "
            f"| {row['onset_f1_delta']:+.4f} [{row['onset_f1_delta_lo95']:+.4f}, "
            f"{row['onset_f1_delta_hi95']:+.4f}] |"
        )
    lines += [
        "",
        "## Six-bucket decomposition",
        "",
        "| variant | correct | wrong_position | pitch_off | timing_only | missed_onset "
        "| extra_detection |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        buckets = variants[variant.name]["decomposition"]
        lines.append(
            f"| `{variant.name}` | {buckets['correct']} "
            f"| {buckets['wrong_position_same_pitch']} | {buckets['pitch_off']} "
            f"| {buckets['timing_only']} | {buckets['missed_onset']} "
            f"| {buckets['extra_detection']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="basicpitch")
    parser.add_argument("--stage", choices=("cache", "sweep", "all"), default="all")
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--comp-clips", type=int, default=10)
    parser.add_argument("--solo-clips", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    # Default to the N2 workdir so the banked ensemble events are reused.
    workdir = args.workdir or (data_root / "models" / "muscriptor_probe")
    workdir.mkdir(parents=True, exist_ok=True)

    clips = select_clips(data_home, "comp", args.comp_clips)
    clips += select_clips(data_home, "solo", args.solo_clips)

    if args.stage in ("cache", "all"):
        run_cache_stage(clips, candidate=args.candidate, data_home=data_home, workdir=workdir)
    if args.stage == "cache":
        return 0

    summary = run_sweep_stage(clips, candidate=args.candidate, data_home=data_home, workdir=workdir)
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output is not None:
        write_report(summary, args.output)

    gate = summary["gate"]
    print(
        f"leg1 complementarity={gate['leg1_complementarity']:.4f} "
        f"({'PASS' if gate['leg1_pass'] else 'FAIL'})"
    )
    print(
        f"leg2 added precision={gate['leg2_added_precision']:.4f} "
        f"via `{gate['leg2_rule']}` ({'PASS' if gate['leg2_pass'] else 'FAIL'})"
    )
    print(f"verdict: {'PASS' if gate['pass'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
