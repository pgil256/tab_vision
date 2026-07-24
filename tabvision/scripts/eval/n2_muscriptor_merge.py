"""Program N Phase N2 — MuScriptor merge-variant pilot (bank + offline replay).

The N2 entry probe (`n2_muscriptor_probe_2026-07-21.md`) measured
P(MuScriptor right | ensemble wrong) = 0.3818 on a slice that turned out to
be **comp-only**, and cached nothing but MuScriptor MIDI. This script closes
both gaps:

``--stage cache`` banks, per clip, the registered ``highres-ensemble``
``AudioEvent`` stream (JSON) alongside the MuScriptor MIDI, for a
mode-balanced deterministic slice of the GuitarSet development players
(00-04). The comp half reuses the entry probe's stride, so its MIDI cache is
free; the solo half is new coverage.

``--stage sweep`` replays those banked artifacts offline — no model
inference — and reports:

- per-mode complementarity (solo vs comp vs pooled), the number the entry
  probe could only estimate on comp material;
- predeclared merge variants scored end-to-end through the shipped
  clean-acoustic config (``guitarset-v1`` position prior + ``guitarset-seq-v1``
  sequence prior at weight 4.0), with onset/pitch/Tab F1, the six-bucket
  error decomposition, and paired bootstrap CIs against ensemble-alone.

Merges are constructed in ``AudioEvent`` space and handed to the existing
``fuse()`` via ``score_audio_only`` — no pipeline code changes, no registry
or routing change. MuScriptor weights are CC-BY-NC-4.0 (SPEC §1.5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tabvision.audio.highres_ensemble import HighResEnsembleBackend
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import (
    ErrorDecomposition,
    aggregate_decompositions,
    decompose_errors,
)
from tabvision.eval.guitarset_audio import parse_guitarset_jams, score_audio_only
from tabvision.fusion.position_prior import (
    PitchPositionPrior,
    apply_pitch_position_prior,
    learn_pitch_position_prior,
    load_pitch_position_prior,
)
from tabvision.fusion.transition_prior import CLUSTER_GAP_S
from tabvision.pipeline import sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
MATCH_TOLERANCE_S = 0.05
COMPLEMENTARITY_GATE = 0.10
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42

# MuScriptor MIDI carries a constant velocity 100 on every note (verified on
# the entry probe's cache), so a per-note confidence floor is not available;
# the added-note gates below are structural instead. The value assigned to a
# merged event is decode-neutral: playability turns confidence into a
# per-event constant that cannot change the argmax over strings
# (`fusion/playability.py` §"does not affect").
MERGED_CONFIDENCE = 0.5


@dataclass(frozen=True)
class ClipContext:
    """Ensemble-side structure a merge rule may consult for one clip."""

    onsets: tuple[float, ...]
    clusters: tuple[tuple[float, float, int], ...]
    """(start_s, end_s, member_count) for each ensemble onset cluster."""

    def near_onset(self, onset_s: float, window_s: float) -> bool:
        return any(abs(onset_s - value) <= window_s for value in self.onsets)

    def in_dense_cluster(self, onset_s: float, *, min_size: int = 2) -> bool:
        return any(
            size >= min_size and start - CLUSTER_GAP_S <= onset_s <= end + CLUSTER_GAP_S
            for start, end, size in self.clusters
        )


@dataclass(frozen=True)
class MergeVariant:
    """A predeclared rule for admitting MuScriptor notes into the stream."""

    name: str
    description: str
    admit: Callable[[AudioEvent, ClipContext], bool]


def _admit_all(_event: AudioEvent, _ctx: ClipContext) -> bool:
    return True


def _admit_near(event: AudioEvent, ctx: ClipContext) -> bool:
    return ctx.near_onset(event.onset_s, CLUSTER_GAP_S)


def _admit_cluster(event: AudioEvent, ctx: ClipContext) -> bool:
    return ctx.in_dense_cluster(event.onset_s)


def _admit_cluster_dur(event: AudioEvent, ctx: ClipContext) -> bool:
    return ctx.in_dense_cluster(event.onset_s) and (event.offset_s - event.onset_s) >= 0.06


def _admit_union_dur(event: AudioEvent, _ctx: ClipContext) -> bool:
    return (event.offset_s - event.onset_s) >= 0.06


VARIANTS: tuple[MergeVariant, ...] = (
    MergeVariant("ensemble", "baseline — registered ensemble alone", lambda _e, _c: False),
    MergeVariant("union", "every non-duplicate MuScriptor note", _admit_all),
    MergeVariant("union-dur60", "union, notes shorter than 60 ms dropped", _admit_union_dur),
    MergeVariant("near80", "onset within 80 ms of any ensemble onset", _admit_near),
    MergeVariant("cluster", "inside an ensemble onset cluster of >= 2 notes", _admit_cluster),
    MergeVariant(
        "cluster-dur60",
        "cluster-scoped plus the 60 ms duration floor",
        _admit_cluster_dur,
    ),
)


def select_clips(data_home: Path, mode: str, count: int) -> list[str]:
    """Deterministic stride over the dev players, restricted to one mode.

    The ``comp`` stride reproduces the entry probe's clip list exactly (that
    probe strided the mixed solo+comp id list and landed on comp only), so
    its MuScriptor MIDI cache is reused rather than recomputed.
    """
    if count <= 0:
        return []
    annotation_dir = data_home / "annotation"
    ids = sorted(
        path.stem
        for path in annotation_dir.glob("*.jams")
        if path.stem[:2] in DEV_PLAYERS and path.stem.endswith(f"_{mode}")
    )
    if not ids:
        raise SystemExit(f"no {mode} dev clips under {annotation_dir}")
    step = max(1, len(ids) // count)
    return [ids[index * step] for index in range(min(count, len(ids)))]


def _event_to_json(event: AudioEvent) -> dict[str, float | int | list[str]]:
    return {
        "onset_s": float(event.onset_s),
        "offset_s": float(event.offset_s),
        "pitch_midi": int(event.pitch_midi),
        "velocity": float(event.velocity),
        "confidence": float(event.confidence),
        "tags": list(event.tags),
    }


def _event_from_json(payload: dict[str, Any]) -> AudioEvent:
    return AudioEvent(
        onset_s=float(payload["onset_s"]),
        offset_s=float(payload["offset_s"]),
        pitch_midi=int(payload["pitch_midi"]),
        velocity=float(payload["velocity"]),
        confidence=float(payload["confidence"]),
        tags=tuple(payload.get("tags", ()) or ()),
    )


def _muscriptor_events(midi_path: Path) -> tuple[list[AudioEvent], dict[str, int]]:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    events: list[AudioEvent] = []
    programs: dict[str, int] = {}
    for instrument in midi.instruments:
        key = f"{instrument.program}{'d' if instrument.is_drum else ''}"
        programs[key] = programs.get(key, 0) + len(instrument.notes)
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            events.append(
                AudioEvent(
                    onset_s=float(note.start),
                    offset_s=float(note.end),
                    pitch_midi=int(note.pitch),
                    velocity=float(note.velocity) / 127.0,
                    confidence=MERGED_CONFIDENCE,
                    tags=("muscriptor",),
                )
            )
    events.sort(key=lambda event: (event.onset_s, event.pitch_midi))
    return events, programs


def _clip_context(ensemble: Sequence[AudioEvent]) -> ClipContext:
    onsets = sorted(event.onset_s for event in ensemble)
    clusters: list[tuple[float, float, int]] = []
    for onset in onsets:
        if clusters and onset - clusters[-1][1] <= CLUSTER_GAP_S:
            start, end, size = clusters[-1]
            clusters[-1] = (start, onset, size + 1)
        else:
            clusters.append((onset, onset, 1))
    return ClipContext(onsets=tuple(onsets), clusters=tuple(clusters))


def _is_duplicate(candidate: AudioEvent, ensemble: Sequence[AudioEvent]) -> bool:
    """Pitch-exact match within the onset tolerance — already transcribed."""
    return any(
        event.pitch_midi == candidate.pitch_midi
        and abs(event.onset_s - candidate.onset_s) <= MATCH_TOLERANCE_S
        for event in ensemble
    )


def merge_events(
    ensemble: Sequence[AudioEvent],
    muscriptor: Sequence[AudioEvent],
    variant: MergeVariant,
) -> tuple[list[AudioEvent], list[AudioEvent]]:
    """Ensemble stream plus the MuScriptor notes ``variant`` admits.

    Returns the merged stream and the admitted notes themselves, so the
    caller can charge the merge for its false additions — complementarity
    only counts rescues and is blind to that half of the trade.
    """
    context = _clip_context(ensemble)
    merged = list(ensemble)
    added: list[AudioEvent] = []
    for candidate in muscriptor:
        if _is_duplicate(candidate, ensemble):
            continue
        if not variant.admit(candidate, context):
            continue
        merged.append(candidate)
        added.append(candidate)
    merged.sort(key=lambda event: (event.onset_s, event.pitch_midi))
    return merged, added


def added_note_yield(
    added: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    ensemble_hits: Sequence[bool],
) -> int:
    """How many admitted notes are real notes the ensemble was missing.

    Greedy pitch-exact one-to-one matching of the admitted notes against
    the gold notes the ensemble failed to hit. ``len(added) - result`` is
    the number of new false detections the merge pays for.
    """
    missed = [event for event, hit in zip(gold, ensemble_hits, strict=True) if not hit]
    used = [False] * len(missed)
    matched = 0
    for candidate in added:
        best = -1
        best_dt = MATCH_TOLERANCE_S + 1e-9
        for index, gold_event in enumerate(missed):
            if used[index] or gold_event.pitch_midi != candidate.pitch_midi:
                continue
            dt = abs(gold_event.onset_s - candidate.onset_s)
            if dt <= MATCH_TOLERANCE_S and dt < best_dt:
                best = index
                best_dt = dt
        if best >= 0:
            used[best] = True
            matched += 1
    return matched


def gold_hits(gold: Sequence[TabEvent], predicted: Sequence[AudioEvent]) -> list[bool]:
    """Greedy one-to-one pitch-exact matching within the onset tolerance."""
    used = [False] * len(predicted)
    hits: list[bool] = []
    for gold_event in gold:
        best = -1
        best_dt = MATCH_TOLERANCE_S + 1e-9
        for index, event in enumerate(predicted):
            if used[index] or event.pitch_midi != gold_event.pitch_midi:
                continue
            dt = abs(event.onset_s - gold_event.onset_s)
            if dt <= MATCH_TOLERANCE_S and dt < best_dt:
                best = index
                best_dt = dt
        if best >= 0:
            used[best] = True
            hits.append(True)
        else:
            hits.append(False)
    return hits


def _cache_paths(workdir: Path, track_id: str, model: str) -> tuple[Path, Path]:
    return workdir / f"{track_id}.ensemble.json", workdir / f"{track_id}.{model}.mid"


def run_cache_stage(
    clips: Sequence[str],
    *,
    data_home: Path,
    workdir: Path,
    exe: Path,
    model: str,
    device: str,
) -> None:
    """Bank ensemble events + MuScriptor MIDI for every clip (resumable)."""
    import soundfile as sf

    session = SessionConfig()
    pending_ensemble = [
        track_id for track_id in clips if not _cache_paths(workdir, track_id, model)[0].is_file()
    ]
    ensemble = HighResEnsembleBackend() if pending_ensemble else None
    try:
        for track_id in clips:
            events_path, midi_path = _cache_paths(workdir, track_id, model)
            wav_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
            if not events_path.is_file():
                assert ensemble is not None  # constructed iff work remains
                wav, sr = sf.read(wav_path, dtype="float32")
                started = time.perf_counter()
                events = list(ensemble.transcribe(wav, int(sr), session))
                seconds = time.perf_counter() - started
                events_path.write_text(
                    json.dumps([_event_to_json(event) for event in events], indent=1) + "\n",
                    encoding="utf-8",
                )
                print(f"{track_id}: ensemble {len(events)} events ({seconds:.0f}s)", flush=True)
            if not midi_path.is_file():
                started = time.perf_counter()
                command = [str(exe), "transcribe", "--model", model, "--device", device]
                command += [str(wav_path), "-o", str(midi_path)]
                result = subprocess.run(command, capture_output=True, text=True)
                seconds = time.perf_counter() - started
                if result.returncode != 0 or not midi_path.is_file():
                    raise SystemExit(
                        f"muscriptor failed on {track_id} "
                        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
                    )
                print(f"{track_id}: muscriptor MIDI cached ({seconds:.0f}s)", flush=True)
    finally:
        closer = getattr(ensemble, "close", None)
        if callable(closer):
            closer()


def _score(
    events: Sequence[AudioEvent],
    gold: Sequence[TabEvent],
    *,
    cfg: GuitarConfig,
    session: SessionConfig,
    prior: PitchPositionPrior,
) -> tuple[dict[str, float], ErrorDecomposition]:
    """Shipped clean-acoustic decode of one event stream."""
    prepared = apply_pitch_position_prior(list(events), prior)
    with sequence_decode_context("guitarset-seq-v1"):
        scored = score_audio_only(prepared, gold, cfg=cfg, session=session)
    decomposition = decompose_errors(scored.decoded, gold)
    return (
        {
            "onset_f1": scored.onset.f1,
            "pitch_f1": scored.pitch.f1,
            "tab_f1": scored.tab.f1,
            "tab_precision": scored.tab.precision,
            "tab_recall": scored.tab.recall,
            "decoded": float(len(scored.decoded)),
        },
        decomposition,
    )


def build_oof_priors(data_home: Path, cfg: GuitarConfig) -> dict[str, PitchPositionPrior]:
    """Leave-one-player-out position priors over the development players.

    ``guitarset-v1`` was trained on players 00-04 (manifest
    ``training_players``) — the very clips this pilot scores — so the
    registered artifact memorizes their fingerings. The house "development
    OOF" protocol (`string_assignment_phase4.py::_oof_position_prior`)
    rebuilds the prior per fold from the other four players, at the
    manifest's own hyper-parameters.
    """
    gold_by_player: dict[str, list[TabEvent]] = {player: [] for player in DEV_PLAYERS}
    for path in sorted((data_home / "annotation").glob("*.jams")):
        player = path.stem[:2]
        if player in gold_by_player:
            gold_by_player[player].extend(parse_guitarset_jams(path, cfg))
    priors: dict[str, PitchPositionPrior] = {}
    for player in DEV_PLAYERS:
        examples = [
            event for other, events in gold_by_player.items() if other != player for event in events
        ]
        priors[player] = learn_pitch_position_prior(examples, cfg=cfg, alpha=1.0, power=2.0)
    return priors


def run_sweep_stage(
    clips: Sequence[str],
    *,
    data_home: Path,
    workdir: Path,
    model: str,
    prior_mode: str = "oof",
) -> dict[str, Any]:
    """Offline replay: complementarity by mode + merge-variant scoring."""
    cfg = GuitarConfig()
    session = SessionConfig()
    oof_priors = build_oof_priors(data_home, cfg) if prior_mode == "oof" else {}
    registered_prior = (
        load_pitch_position_prior("guitarset-v1") if prior_mode == "registered" else None
    )

    per_clip: list[dict[str, Any]] = []
    per_variant_scores: dict[str, list[dict[str, float]]] = {v.name: [] for v in VARIANTS}
    per_variant_decomp: dict[str, list[ErrorDecomposition]] = {v.name: [] for v in VARIANTS}
    program_totals: dict[str, int] = {}

    for track_id in clips:
        events_path, midi_path = _cache_paths(workdir, track_id, model)
        if not events_path.is_file() or not midi_path.is_file():
            raise SystemExit(f"missing cache for {track_id}; run --stage cache first")
        ensemble = [_event_from_json(item) for item in json.loads(events_path.read_text("utf-8"))]
        muscriptor, programs = _muscriptor_events(midi_path)
        for key, count in programs.items():
            program_totals[key] = program_totals.get(key, 0) + count
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        prior = registered_prior if registered_prior is not None else oof_priors[track_id[:2]]

        ens_hits = gold_hits(gold, ensemble)
        ms_hits = gold_hits(gold, muscriptor)
        ens_wrong = sum(1 for hit in ens_hits if not hit)
        rescued = sum(1 for ens, ms in zip(ens_hits, ms_hits, strict=True) if not ens and ms)

        row: dict[str, Any] = {
            "track_id": track_id,
            "mode": "solo" if track_id.endswith("_solo") else "comp",
            "gold": len(gold),
            "ens_events": len(ensemble),
            "ms_events": len(muscriptor),
            "ens_wrong": ens_wrong,
            "ms_rescued": rescued,
            "ens_recall": sum(ens_hits) / len(gold) if gold else 0.0,
            "ms_recall": sum(ms_hits) / len(gold) if gold else 0.0,
        }
        for variant in VARIANTS:
            merged, added = merge_events(ensemble, muscriptor, variant)
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
            f"union={row['union']['tab_f1']:.4f} "
            f"cluster={row['cluster']['tab_f1']:.4f}",
            flush=True,
        )

    return {
        "clips": list(clips),
        "model": model,
        "prior_mode": prior_mode,
        "complementarity": _complementarity_summary(per_clip),
        "variants": _variant_summary(per_variant_scores, per_variant_decomp),
        "muscriptor_program_note_counts": program_totals,
        "per_clip": per_clip,
    }


def _complementarity_summary(per_clip: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("solo", "comp", "pooled"):
        rows = [row for row in per_clip if mode == "pooled" or row["mode"] == mode]
        wrong = sum(int(row["ens_wrong"]) for row in rows)
        rescued = sum(int(row["ms_rescued"]) for row in rows)
        gold = sum(int(row["gold"]) for row in rows)
        value = rescued / wrong if wrong else float("nan")
        summary[mode] = {
            "clips": len(rows),
            "gold_notes": gold,
            "ensemble_wrong": wrong,
            "rescued": rescued,
            "complementarity": value,
            "gate_pass": bool(wrong) and value >= COMPLEMENTARITY_GATE,
        }
    return summary


def _variant_summary(
    scores: dict[str, list[dict[str, float]]],
    decompositions: dict[str, list[ErrorDecomposition]],
) -> dict[str, Any]:
    def _column(name: str, metric: str) -> np.ndarray:
        return np.asarray([row[metric] for row in scores[name]], dtype=np.float64)

    baseline = _column("ensemble", "tab_f1")
    baseline_onset = _column("ensemble", "onset_f1")
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = scores[variant.name]
        tab = _column(variant.name, "tab_f1")
        delta = tab - baseline
        ci = bootstrap_ci(delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        # The merge adds onsets, so the onset gate moves too and has to be
        # reported alongside Tab F1 rather than assumed unchanged.
        onset_delta = _column(variant.name, "onset_f1") - baseline_onset
        onset_ci = bootstrap_ci(onset_delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        aggregate = aggregate_decompositions(decompositions[variant.name])
        added_total = int(sum(row["added"] for row in rows))
        added_true = int(sum(row["added_true"] for row in rows))
        summary[variant.name] = {
            "description": variant.description,
            "added_notes": added_total,
            "added_true_notes": added_true,
            "added_precision": (added_true / added_total) if added_total else float("nan"),
            "tab_f1_mean": float(tab.mean()),
            "onset_f1_mean": float(np.mean([row["onset_f1"] for row in rows])),
            "pitch_f1_mean": float(np.mean([row["pitch_f1"] for row in rows])),
            "tab_precision_mean": float(np.mean([row["tab_precision"] for row in rows])),
            "tab_recall_mean": float(np.mean([row["tab_recall"] for row in rows])),
            "tab_f1_delta": float(delta.mean()),
            "tab_f1_delta_lo95": ci.lower,
            "tab_f1_delta_hi95": ci.upper,
            "onset_f1_delta": float(onset_delta.mean()),
            "onset_f1_delta_lo95": onset_ci.lower,
            "onset_f1_delta_hi95": onset_ci.upper,
            "decomposition": aggregate.to_dict(),
        }
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    complementarity = summary["complementarity"]
    variants = summary["variants"]
    clips = summary["clips"]
    lines = [
        "# N2 MuScriptor merge-variant pilot — GuitarSet dev (solo + comp)",
        "",
        f"Model: muscriptor-{summary['model']} (isolated venv) vs registered "
        f"`highres-ensemble` | {len(clips)} clips | offline replay of banked "
        "events; clean-acoustic decode with the "
        + (
            "leave-one-player-out position prior"
            if summary["prior_mode"] == "oof"
            else "registered `guitarset-v1` position prior"
        )
        + " + `guitarset-seq-v1` @ w=4.0",
        "",
        "## Complementarity by mode — P(MuScriptor right | ensemble wrong)",
        "",
        "| mode | clips | gold notes | ensemble wrong | rescued | complementarity | gate ≥ 0.10 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode in ("solo", "comp", "pooled"):
        row = complementarity[mode]
        lines.append(
            f"| {mode} | {row['clips']} | {row['gold_notes']} | {row['ensemble_wrong']} "
            f"| {row['rescued']} | {row['complementarity']:.4f} "
            f"| {'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Merge variants — shipped decode, paired bootstrap vs ensemble alone",
        "",
        "| variant | added notes | of which real | added precision | Tab F1 | Tab P | Tab R "
        "| ΔTab F1 [lo-95, hi-95] | onset F1 | Δonset F1 [lo-95, hi-95] | pitch F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for variant in VARIANTS:
        row = variants[variant.name]
        added_precision = row["added_precision"]
        precision_cell = "—" if added_precision != added_precision else f"{added_precision:.3f}"
        lines.append(
            f"| `{variant.name}` | {row['added_notes']} | {row['added_true_notes']} "
            f"| {precision_cell} | {row['tab_f1_mean']:.4f} "
            f"| {row['tab_precision_mean']:.4f} | {row['tab_recall_mean']:.4f} "
            f"| {row['tab_f1_delta']:+.4f} [{row['tab_f1_delta_lo95']:+.4f}, "
            f"{row['tab_f1_delta_hi95']:+.4f}] "
            f"| {row['onset_f1_mean']:.4f} "
            f"| {row['onset_f1_delta']:+.4f} [{row['onset_f1_delta_lo95']:+.4f}, "
            f"{row['onset_f1_delta_hi95']:+.4f}] "
            f"| {row['pitch_f1_mean']:.4f} |"
        )
    lines += [
        "",
        "## Six-bucket decomposition (counts over the same clips)",
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
    lines += [
        "",
        f"Bootstrap: paired per-clip ΔTab F1, N={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}. "
        "Acceptance for a ship decision is lo-95 > 0 on the full dev set plus the "
        "GAPS clean-12 strict no-regression check; this pilot is a variant filter, "
        "not the ship gate.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--stage", choices=("cache", "sweep", "all"), default="all")
    parser.add_argument("--solo-clips", type=int, default=10)
    parser.add_argument("--comp-clips", type=int, default=10)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--prior", choices=("oof", "registered"), default="oof")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--muscriptor-exe", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    data_home = args.data_home or (Path(data_root) / "guitarset")
    workdir = args.workdir or (Path(data_root) / "models" / "muscriptor_probe")
    workdir.mkdir(parents=True, exist_ok=True)

    clips = select_clips(data_home, "comp", args.comp_clips)
    clips += select_clips(data_home, "solo", args.solo_clips)

    if args.stage in ("cache", "all"):
        exe = args.muscriptor_exe or (
            Path.home() / ".tabvision" / "probe-envs" / "muscriptor" / "Scripts" / "muscriptor.exe"
        )
        if not exe.is_file():
            raise SystemExit(f"muscriptor CLI not found: {exe}")
        run_cache_stage(
            clips,
            data_home=data_home,
            workdir=workdir,
            exe=exe,
            model=args.model,
            device=args.device,
        )

    if args.stage == "cache":
        return 0

    summary = run_sweep_stage(
        clips,
        data_home=data_home,
        workdir=workdir,
        model=args.model,
        prior_mode=args.prior,
    )
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output is not None:
        _write_report(summary, args.output)

    pooled = summary["complementarity"]["pooled"]
    print(f"pooled complementarity={pooled['complementarity']:.4f}")
    for variant in VARIANTS:
        row = summary["variants"][variant.name]
        print(
            f"{variant.name}: tab_f1={row['tab_f1_mean']:.4f} "
            f"delta={row['tab_f1_delta']:+.4f} "
            f"[{row['tab_f1_delta_lo95']:+.4f}, {row['tab_f1_delta_hi95']:+.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
