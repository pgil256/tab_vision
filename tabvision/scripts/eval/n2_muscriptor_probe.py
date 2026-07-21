"""Program N Phase N2 — MuScriptor complementarity probe.

Runs the registered ``highres-ensemble`` backend and MuScriptor-large
(1.3B, CC-BY-NC-4.0 weights, process-isolated in its own probe venv) over
a deterministic slice of GuitarSet development players (00-04 only) and
measures, per gold note (pitch-exact match within 50 ms, greedy one-to-one):

- ensemble hit rate and MuScriptor hit rate (pitch recall);
- **complementarity**: P(MuScriptor right | ensemble wrong) — the plan's
  continue/close gate at 0.10;
- onset/pitch event F1 for both systems.

MuScriptor is invoked via its CLI from the isolated venv (no shipping
dependency): ``muscriptor transcribe --model <m> <wav> -o <mid>``. Its MIDI
outputs are cached in the workdir so the probe is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import soundfile as sf

from tabvision.audio.highres_ensemble import HighResEnsembleBackend
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.eval.metrics import event_f1
from tabvision.types import SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
MATCH_TOLERANCE_S = 0.05
COMPLEMENTARITY_GATE = 0.10


def _select_clips(data_home: Path, count: int) -> list[str]:
    annotation_dir = data_home / "annotation"
    ids = sorted(
        path.stem for path in annotation_dir.glob("*.jams") if path.stem[:2] in DEV_PLAYERS
    )
    step = len(ids) // count
    return [ids[index * step] for index in range(count)]


def _as_tab(notes: list[tuple[float, float, int]]) -> list[TabEvent]:
    return [
        TabEvent(
            onset_s=onset,
            duration_s=max(0.0, duration),
            string_idx=0,
            fret=0,
            pitch_midi=pitch,
            confidence=1.0,
        )
        for onset, duration, pitch in notes
    ]


def _gold_hits(gold: list[TabEvent], predicted: list[tuple[float, float, int]]) -> list[bool]:
    """Greedy one-to-one pitch-exact matching within the onset tolerance."""
    used = [False] * len(predicted)
    hits: list[bool] = []
    for gold_event in gold:
        best = -1
        best_dt = MATCH_TOLERANCE_S + 1e-9
        for index, (onset, _duration, pitch) in enumerate(predicted):
            if used[index] or pitch != gold_event.pitch_midi:
                continue
            dt = abs(onset - gold_event.onset_s)
            if dt <= MATCH_TOLERANCE_S and dt < best_dt:
                best = index
                best_dt = dt
        if best >= 0:
            used[best] = True
            hits.append(True)
        else:
            hits.append(False)
    return hits


def _muscriptor_notes(midi_path: Path) -> tuple[list[tuple[float, float, int]], dict]:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[tuple[float, float, int]] = []
    programs: dict[str, int] = {}
    for instrument in midi.instruments:
        key = f"{instrument.program}{'d' if instrument.is_drum else ''}"
        programs[key] = programs.get(key, 0) + len(instrument.notes)
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((float(note.start), float(note.end - note.start), int(note.pitch)))
    notes.sort()
    return notes, programs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--model", default="large")
    parser.add_argument("--muscriptor-exe", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    data_home = args.data_home or (Path(data_root) / "guitarset")
    exe = args.muscriptor_exe or (
        Path.home() / ".tabvision" / "probe-envs" / "muscriptor" / "Scripts" / "muscriptor.exe"
    )
    workdir = args.workdir or (Path(data_root) / "models" / "muscriptor_probe")
    workdir.mkdir(parents=True, exist_ok=True)
    if not exe.is_file():
        raise SystemExit(f"muscriptor CLI not found: {exe}")

    clips = _select_clips(data_home, args.clips)
    session = SessionConfig()
    ensemble = HighResEnsembleBackend()

    rows: list[dict[str, object]] = []
    total_gold = ens_hit_count = ms_hit_count = 0
    ens_wrong = ms_right_given_ens_wrong = 0
    program_totals: dict[str, int] = {}
    try:
        for track_id in clips:
            wav_path = data_home / "audio_mono-mic" / f"{track_id}_mic.wav"
            gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams")

            wav, sr = sf.read(wav_path, dtype="float32")
            started = time.perf_counter()
            ens_events = ensemble.transcribe(wav, int(sr), session)
            ens_seconds = time.perf_counter() - started
            ens_notes = [
                (event.onset_s, event.offset_s - event.onset_s, event.pitch_midi)
                for event in ens_events
            ]

            midi_path = workdir / f"{track_id}.{args.model}.mid"
            ms_seconds = 0.0
            if not midi_path.is_file():
                started = time.perf_counter()
                command = [str(exe), "transcribe", "--model", args.model]
                command += [str(wav_path), "-o", str(midi_path)]
                result = subprocess.run(command, capture_output=True, text=True)
                ms_seconds = time.perf_counter() - started
                if result.returncode != 0 or not midi_path.is_file():
                    raise SystemExit(
                        f"muscriptor failed on {track_id} "
                        f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
                    )
            ms_notes, programs = _muscriptor_notes(midi_path)
            for key, count in programs.items():
                program_totals[key] = program_totals.get(key, 0) + count

            ens_hits = _gold_hits(gold, ens_notes)
            ms_hits = _gold_hits(gold, ms_notes)
            clip_ens_wrong = sum(1 for hit in ens_hits if not hit)
            clip_rescued = sum(
                1 for ens, ms in zip(ens_hits, ms_hits, strict=True) if not ens and ms
            )
            total_gold += len(gold)
            ens_hit_count += sum(ens_hits)
            ms_hit_count += sum(ms_hits)
            ens_wrong += clip_ens_wrong
            ms_right_given_ens_wrong += clip_rescued

            ens_onset = event_f1(_as_tab(ens_notes), gold, match_pitch=False).f1
            ens_pitch = event_f1(_as_tab(ens_notes), gold, match_pitch=True).f1
            ms_onset = event_f1(_as_tab(ms_notes), gold, match_pitch=False).f1
            ms_pitch = event_f1(_as_tab(ms_notes), gold, match_pitch=True).f1
            rows.append(
                {
                    "track_id": track_id,
                    "gold": len(gold),
                    "ens_recall": sum(ens_hits) / len(gold) if gold else 0.0,
                    "ms_recall": sum(ms_hits) / len(gold) if gold else 0.0,
                    "ens_wrong": clip_ens_wrong,
                    "ms_rescued": clip_rescued,
                    "ens_onset_f1": ens_onset,
                    "ens_pitch_f1": ens_pitch,
                    "ms_onset_f1": ms_onset,
                    "ms_pitch_f1": ms_pitch,
                    "ens_seconds": round(ens_seconds, 1),
                    "ms_seconds": round(ms_seconds, 1),
                }
            )
            print(
                f"{track_id}: ens_recall={rows[-1]['ens_recall']:.3f} "
                f"ms_recall={rows[-1]['ms_recall']:.3f} "
                f"rescued={clip_rescued}/{clip_ens_wrong} "
                f"(ens {ens_seconds:.0f}s, ms {ms_seconds:.0f}s)",
                flush=True,
            )
    finally:
        closer = getattr(ensemble, "close", None)
        if callable(closer):
            closer()

    complementarity = ms_right_given_ens_wrong / ens_wrong if ens_wrong else float("nan")
    gate_pass = complementarity >= COMPLEMENTARITY_GATE
    summary = {
        "clips": clips,
        "model": args.model,
        "total_gold_notes": total_gold,
        "ensemble_pitch_recall": ens_hit_count / total_gold if total_gold else 0.0,
        "muscriptor_pitch_recall": ms_hit_count / total_gold if total_gold else 0.0,
        "ensemble_wrong_notes": ens_wrong,
        "muscriptor_rescued_notes": ms_right_given_ens_wrong,
        "complementarity": complementarity,
        "gate_threshold": COMPLEMENTARITY_GATE,
        "gate_pass": gate_pass,
        "muscriptor_program_note_counts": program_totals,
        "per_clip": rows,
    }
    args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# N2 MuScriptor complementarity probe — GuitarSet dev clips",
        "",
        f"Model: muscriptor-{args.model} (isolated venv) vs registered "
        f"`highres-ensemble` | {len(clips)} clips | pitch-exact 50 ms greedy "
        "matching",
        "",
        "| clip | gold | ens recall | ms recall | rescued/ens-wrong "
        "| ens onset/pitch F1 | ms onset/pitch F1 | ms s |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['track_id']} | {row['gold']} | {row['ens_recall']:.3f} "
            f"| {row['ms_recall']:.3f} | {row['ms_rescued']}/{row['ens_wrong']} "
            f"| {row['ens_onset_f1']:.3f}/{row['ens_pitch_f1']:.3f} "
            f"| {row['ms_onset_f1']:.3f}/{row['ms_pitch_f1']:.3f} "
            f"| {row['ms_seconds']} |"
        )
    lines += [
        "",
        f"**Complementarity P(MuScriptor right | ensemble wrong) = "
        f"{complementarity:.4f}** ({ms_right_given_ens_wrong}/{ens_wrong}; "
        f"gate ≥ {COMPLEMENTARITY_GATE} → "
        f"**{'PASS — full dev eval justified' if gate_pass else 'FAIL — close Program N'}**)",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"complementarity={complementarity:.4f} gate={'PASS' if gate_pass else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
