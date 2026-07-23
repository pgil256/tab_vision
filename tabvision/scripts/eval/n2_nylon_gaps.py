"""Accuracy-loop N2 — does a nylon stiffness table help classical Tab F1?

Q6's channel abstains on classical because no nylon table existed, so the GAPS
cross-domain gate was satisfied by *abstention*. N2 builds the nylon table
(`classical_stiffness_model`, specification-derived like the steel one) and
tests whether, applied to classical audio through the shipped classical
routing, it converts that gate into a real *measurement* — a positive result
on the tier where the physics is theoretically strongest.

Honest caveat carried from the table's construction: the three nylon trebles
are first-principles (plain monofilament, mass from density and gauge) but the
three wound basses rest on approximate core diameters, and ``B ~ d_core^4``, so
the bass rows are rough. This eval is the test of whether they are good enough.

GAPS clean-12, classical session (so the ensemble uses the gaps checkpoint and
the nylon table is selected), gaps-v1 position prior + gaps-seq-v1 sequence
prior — the shipped classical decode. Three arms scored on one transcription:
baseline (no evidence), strict, partial_aware. Ensemble events cache per clip,
so the ~47-minute run is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.acquire.gaps_video import CLEAN_12
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import aggregate_decompositions, decompose_errors
from tabvision.eval.guitarset_audio import load_mono_audio, score_audio_only
from tabvision.eval.parsers.registry import get_parser
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.position_prior import apply_pitch_position_prior, load_pitch_position_prior
from tabvision.fusion.string_physics import classical_stiffness_model
from tabvision.pipeline import sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig

WEIGHT = 1.0
MIN_R2 = 0.50
ARMS = ("baseline", "strict", "partial_aware")


def _to_json(e: AudioEvent) -> dict[str, Any]:
    return {
        "onset_s": float(e.onset_s),
        "offset_s": float(e.offset_s),
        "pitch_midi": int(e.pitch_midi),
        "velocity": float(e.velocity),
        "confidence": float(e.confidence),
    }


def _from_json(p: dict[str, Any]) -> AudioEvent:
    return AudioEvent(
        onset_s=float(p["onset_s"]),
        offset_s=float(p["offset_s"]),
        pitch_midi=int(p["pitch_midi"]),
        velocity=float(p["velocity"]),
        confidence=float(p["confidence"]),
    )


def _decode(events, gold, cfg, session, prior):
    prepared = apply_pitch_position_prior(list(events), prior)
    with sequence_decode_context("gaps-seq-v1"):
        scored = score_audio_only(prepared, gold, cfg=cfg, session=session)
    return (
        {"tab_f1": scored.tab.f1, "onset_f1": scored.onset.f1, "pitch_f1": scored.pitch.f1},
        decompose_errors(scored.decoded, gold),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    gaps = args.gaps_root or (data_root / "gaps")
    cache_dir = args.cache_dir or (data_root / "models" / "n2_nylon_gaps_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = GuitarConfig()
    session = SessionConfig(instrument="classical", style="fingerstyle")
    prior = load_pitch_position_prior("gaps-v1")
    model = classical_stiffness_model()
    parse = get_parser("gaps_musicxml_tab")

    backend = None
    scores: dict[str, list[dict[str, float]]] = {a: [] for a in ARMS}
    decomps: dict[str, list] = {a: [] for a in ARMS}
    cover: dict[str, dict[str, int]] = {
        a: {"events": 0, "isolated": 0, "fitted": 0, "applied": 0}
        for a in ("strict", "partial_aware")
    }
    rows: list[dict[str, Any]] = []

    for clip in CLEAN_12:
        wav_path = gaps / "audio" / f"{clip}.wav"
        xml_path = gaps / "musicxml" / f"{clip}.xml"
        if not wav_path.is_file() or not xml_path.is_file():
            print(f"  {clip}: missing, skipped", flush=True)
            continue
        gold = list(parse(xml_path, cfg))
        wav, sr = load_mono_audio(wav_path)
        cache = cache_dir / f"{clip}.ensemble.json"
        if cache.is_file():
            events = [_from_json(x) for x in json.loads(cache.read_text("utf-8"))]
        else:
            if backend is None:
                from tabvision.audio.highres_ensemble import HighResEnsembleBackend

                backend = HighResEnsembleBackend()
            events = list(backend.transcribe(np.asarray(wav), int(sr), session))
            cache.write_text(
                json.dumps([_to_json(e) for e in events], indent=1) + "\n", encoding="utf-8"
            )
            print(f"  {clip}: transcribed {len(events)} events", flush=True)

        row: dict[str, Any] = {"clip": clip, "gold": len(gold)}
        for arm in ARMS:
            if arm == "baseline":
                prepared = events
            else:
                prepared, tally = attach_inharmonicity_evidence(
                    events,
                    np.asarray(wav),
                    int(sr),
                    model,
                    cfg,
                    weight=WEIGHT,
                    min_r2=MIN_R2,
                    isolation=arm,
                )
                for k in cover[arm]:
                    cover[arm][k] += tally[k]
            metrics, decomp = _decode(prepared, gold, cfg, session, prior)
            scores[arm].append(metrics)
            decomps[arm].append(decomp)
            row[arm] = metrics["tab_f1"]
        rows.append(row)
        print(
            f"  {clip}: base={row['baseline']:.4f} strict={row['strict']:.4f} "
            f"pa={row['partial_aware']:.4f}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no GAPS clips scored")

    base = np.asarray([r["tab_f1"] for r in scores["baseline"]])
    out: dict[str, Any] = {"clips": len(rows), "coverage": cover, "arms": {}, "per_clip": rows}
    for arm in ARMS:
        tab = np.asarray([r["tab_f1"] for r in scores[arm]])
        delta = tab - base
        ci = bootstrap_ci(delta, n_bootstrap=10_000, seed=42)
        out["arms"][arm] = {
            "tab_f1": float(tab.mean()),
            "delta": float(delta.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
            "onset_f1": float(np.mean([r["onset_f1"] for r in scores[arm]])),
            "pitch_f1": float(np.mean([r["pitch_f1"] for r in scores[arm]])),
            "decomposition": aggregate_decompositions(decomps[arm]).to_dict(),
            "regressions": [r["clip"] for r in rows if r[arm] < r["baseline"] - 1e-9],
        }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print()
    for arm in ARMS:
        a = out["arms"][arm]
        print(
            f"  {arm:>14}: tab={a['tab_f1']:.4f} delta={a['delta']:+.4f} "
            f"[{a['lo95']:+.4f}, {a['hi95']:+.4f}] onset={a['onset_f1']:.4f}"
        )
    for arm in ("strict", "partial_aware"):
        c = cover[arm]
        print(
            f"  {arm:>14} coverage: {c['applied']}/{c['events']} "
            f"({c['applied'] / max(c['events'], 1):.2%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
