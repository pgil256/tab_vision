"""Program N Phase N1 — ``guitar_kroma`` smoke vs the registered members.

Runs ``guitar_gaps``, ``guitar_fl``, and the converted ``guitar-kroma.pth``
over a small deterministic slice of GuitarSet development players (00-04
only; the frozen player-05 confirmation set is never touched) and reports
per-clip and mean onset/pitch F1 on the raw event streams.

Gate (plan N1): kroma loads cleanly and its mean onset F1 and pitch F1 are
each within 0.05 of the better registered member on the same clips. This is
a sanity gate, not a promotion gate.

Probe mechanism: the kroma checkpoint is loaded through the existing
``guitar_electric`` variant's env-var path
(``TABVISION_HIGHRES_ELECTRIC_CKPT``) so the probe needs no runtime code
change. Constructor settings are identical across all three conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf

from tabvision.audio.highres import HIGHRES_ELECTRIC_CKPT_ENV, HighResBackend
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.eval.metrics import event_f1
from tabvision.types import AudioEvent, SessionConfig, TabEvent

DEV_PLAYERS = ("00", "01", "02", "03", "04")
GATE_TOLERANCE = 0.05


def _audio_as_tab(events: Sequence[AudioEvent]) -> tuple[TabEvent, ...]:
    return tuple(
        TabEvent(
            onset_s=event.onset_s,
            duration_s=max(0.0, event.offset_s - event.onset_s),
            string_idx=0,
            fret=0,
            pitch_midi=event.pitch_midi,
            confidence=event.confidence,
        )
        for event in events
    )


def _select_clips(data_home: Path, count: int) -> list[str]:
    annotation_dir = data_home / "annotation"
    ids = sorted(
        path.stem for path in annotation_dir.glob("*.jams") if path.stem[:2] in DEV_PLAYERS
    )
    if len(ids) < count:
        raise SystemExit(f"only {len(ids)} dev annotations under {annotation_dir}")
    step = len(ids) // count
    return [ids[index * step] for index in range(count)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-home",
        default=None,
        help="GuitarSet root (default: $TABVISION_DATA_ROOT/guitarset).",
    )
    parser.add_argument("--kroma-pth", default=None)
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--json", dest="json_path", required=True)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    data_home = Path(args.data_home or (Path(data_root) / "guitarset"))
    if not (data_home / "annotation").is_dir():
        raise SystemExit(f"no GuitarSet annotations under {data_home}")
    kroma_pth = Path(args.kroma_pth or (Path(data_root) / "models" / "guitar-kroma.pth"))
    if not kroma_pth.is_file():
        raise SystemExit(f"missing {kroma_pth}; run convert_kroma_checkpoint.py first")

    clips = _select_clips(data_home, args.clips)
    session = SessionConfig()
    conditions: dict[str, dict[str, str]] = {
        "guitar_gaps": {"checkpoint": "guitar_gaps"},
        "guitar_fl": {"checkpoint": "guitar_fl"},
        "guitar_kroma": {"checkpoint": "guitar_electric"},
    }

    rows: list[dict[str, object]] = []
    means: dict[str, dict[str, float]] = {}
    for name, spec in conditions.items():
        if name == "guitar_kroma":
            os.environ[HIGHRES_ELECTRIC_CKPT_ENV] = str(kroma_pth)
        backend = HighResBackend(checkpoint=spec["checkpoint"])
        onset_scores: list[float] = []
        pitch_scores: list[float] = []
        try:
            for track_id in clips:
                wav, sr = sf.read(
                    data_home / "audio_mono-mic" / f"{track_id}_mic.wav",
                    dtype="float32",
                )
                gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams")
                started = time.perf_counter()
                events = backend.transcribe(wav, int(sr), session)
                elapsed = time.perf_counter() - started
                predicted = _audio_as_tab(events)
                onset = event_f1(predicted, gold, match_pitch=False).f1
                pitch = event_f1(predicted, gold, match_pitch=True).f1
                onset_scores.append(onset)
                pitch_scores.append(pitch)
                rows.append(
                    {
                        "condition": name,
                        "track_id": track_id,
                        "onset_f1": onset,
                        "pitch_f1": pitch,
                        "events": len(events),
                        "gold_notes": len(gold),
                        "seconds": round(elapsed, 3),
                    }
                )
                print(
                    f"{name} {track_id}: onset={onset:.4f} pitch={pitch:.4f} "
                    f"events={len(events)} ({elapsed:.1f}s)",
                    flush=True,
                )
        finally:
            backend.close()
            os.environ.pop(HIGHRES_ELECTRIC_CKPT_ENV, None)
        means[name] = {
            "onset_f1": _mean(onset_scores),
            "pitch_f1": _mean(pitch_scores),
        }

    best_onset = max(means["guitar_gaps"]["onset_f1"], means["guitar_fl"]["onset_f1"])
    best_pitch = max(means["guitar_gaps"]["pitch_f1"], means["guitar_fl"]["pitch_f1"])
    onset_delta = means["guitar_kroma"]["onset_f1"] - best_onset
    pitch_delta = means["guitar_kroma"]["pitch_f1"] - best_pitch
    gate_pass = onset_delta >= -GATE_TOLERANCE and pitch_delta >= -GATE_TOLERANCE

    payload = {
        "clips": clips,
        "selection_rule": "sorted dev annotations, every len//count-th stem",
        "per_clip": rows,
        "means": means,
        "gate": {
            "tolerance": GATE_TOLERANCE,
            "onset_delta_vs_best_registered": onset_delta,
            "pitch_delta_vs_best_registered": pitch_delta,
            "pass": gate_pass,
        },
        "kroma_pth": str(kroma_pth),
        "probe_mechanism": f"{HIGHRES_ELECTRIC_CKPT_ENV} + checkpoint=guitar_electric",
    }
    Path(args.json_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# N1 kroma smoke — onset/pitch F1 on 5 GuitarSet dev clips",
        "",
        "| condition | clip | onset F1 | pitch F1 | events | gold | s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['track_id']} | {row['onset_f1']:.4f} "
            f"| {row['pitch_f1']:.4f} | {row['events']} | {row['gold_notes']} "
            f"| {row['seconds']} |"
        )
    lines += [
        "",
        "| condition | mean onset F1 | mean pitch F1 |",
        "|---|---:|---:|",
    ]
    for name, stats in means.items():
        lines.append(f"| {name} | {stats['onset_f1']:.4f} | {stats['pitch_f1']:.4f} |")
    lines += [
        "",
        f"Gate (±{GATE_TOLERANCE} vs best registered member): "
        f"onset Δ {onset_delta:+.4f}, pitch Δ {pitch_delta:+.4f} → "
        f"**{'PASS' if gate_pass else 'FAIL'}**",
        "",
    ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"gate: {'PASS' if gate_pass else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
