"""Accuracy-loop Q6 — GAPS clean-12 cross-domain no-regression gate.

Anything touching ``fuse()`` must clear the GAPS clean-12 strict per-clip
no-regression check before acceptance (house rule; see the ROI deep-dive §6).
For this channel that check carries unusual weight, because it is the first
place the physics can be *wrong* rather than merely absent:

**GAPS is classical guitar — nylon trebles over a wound-nylon/metal bass.**
The specification table in :mod:`tabvision.fusion.string_physics` is derived
for a steel-string set. Nylon's Young modulus is roughly three orders of
magnitude below steel's, so its inharmonicity is far lower and differently
ordered across strings. Either the fits fail and the channel abstains
(harmless), or they succeed and are scored against a table that does not
describe the instrument (a regression). This gate decides which.

Config is **predeclared before the run** — weight 1.0, ``min_r2`` 0.50, raw
physics table, no offset — deliberately, because the GuitarSet pilot chose
its weight on the set it reported and that optimism should not be repeated
here.

Classical routing is the shipped one: ``gaps-v1`` position prior with
``gaps-seq-v1``.
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
from tabvision.fusion.string_physics import reference_stiffness_model
from tabvision.pipeline import sequence_decode_context
from tabvision.types import AudioEvent, GuitarConfig, SessionConfig

WEIGHT = 1.0
MIN_R2 = 0.50
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def _event_to_json(event: AudioEvent) -> dict[str, Any]:
    return {
        "onset_s": float(event.onset_s),
        "offset_s": float(event.offset_s),
        "pitch_midi": int(event.pitch_midi),
        "velocity": float(event.velocity),
        "confidence": float(event.confidence),
    }


def _event_from_json(payload: dict[str, Any]) -> AudioEvent:
    return AudioEvent(
        onset_s=float(payload["onset_s"]),
        offset_s=float(payload["offset_s"]),
        pitch_midi=int(payload["pitch_midi"]),
        velocity=float(payload["velocity"]),
        confidence=float(payload["confidence"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    gaps = args.gaps_root or (data_root / "gaps")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_gaps_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = GuitarConfig()
    session = SessionConfig(style="fingerstyle")
    prior = load_pitch_position_prior("gaps-v1")
    physics = reference_stiffness_model()
    parse = get_parser("gaps_musicxml_tab")

    backend = None
    rows: list[dict[str, Any]] = []
    base_decomps = []
    arm_decomps = []

    for clip in CLEAN_12:
        wav_path = gaps / "audio" / f"{clip}.wav"
        xml_path = gaps / "musicxml" / f"{clip}.xml"
        if not wav_path.is_file() or not xml_path.is_file():
            print(f"  {clip}: missing media/annotation, skipped", flush=True)
            continue
        gold = list(parse(xml_path, cfg))
        cache = cache_dir / f"{clip}.ensemble.json"
        if cache.is_file():
            events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
            wav, sr = load_mono_audio(wav_path)  # GAPS ships stereo
        else:
            if backend is None:
                from tabvision.audio.highres_ensemble import HighResEnsembleBackend

                backend = HighResEnsembleBackend()
            wav, sr = load_mono_audio(wav_path)
            events = list(backend.transcribe(np.asarray(wav), int(sr), session))
            cache.write_text(
                json.dumps([_event_to_json(e) for e in events], indent=1) + "\n", encoding="utf-8"
            )
        prepared = apply_pitch_position_prior(list(events), prior)
        with sequence_decode_context("gaps-seq-v1"):
            base = score_audio_only(prepared, gold, cfg=cfg, session=session)

        moved, tally = attach_inharmonicity_evidence(
            events, np.asarray(wav), int(sr), physics, cfg, weight=WEIGHT, min_r2=MIN_R2
        )
        prepared_arm = apply_pitch_position_prior(moved, prior)
        with sequence_decode_context("gaps-seq-v1"):
            arm = score_audio_only(prepared_arm, gold, cfg=cfg, session=session)

        base_decomps.append(decompose_errors(base.decoded, gold))
        arm_decomps.append(decompose_errors(arm.decoded, gold))
        rows.append(
            {
                "clip": clip,
                "gold": len(gold),
                "applied": tally["applied"],
                "isolated": tally["isolated"],
                "base_tab_f1": base.tab.f1,
                "arm_tab_f1": arm.tab.f1,
                "delta": arm.tab.f1 - base.tab.f1,
                "base_onset_f1": base.onset.f1,
                "arm_onset_f1": arm.onset.f1,
            }
        )
        print(
            f"  {clip}: base={base.tab.f1:.4f} physics={arm.tab.f1:.4f} "
            f"delta={arm.tab.f1 - base.tab.f1:+.4f} (applied {tally['applied']})",
            flush=True,
        )

    if not rows:
        raise SystemExit("no GAPS clips scored — check the audio/musicxml paths")

    deltas = np.asarray([r["delta"] for r in rows], dtype=np.float64)
    ci = bootstrap_ci(deltas, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    regressions = [r["clip"] for r in rows if r["delta"] < -1e-9]
    summary = {
        "weight": WEIGHT,
        "min_r2": MIN_R2,
        "table": "physics (raw)",
        "clips": len(rows),
        "notes_applied": int(sum(r["applied"] for r in rows)),
        "mean_delta": float(deltas.mean()),
        "lo95": ci.lower,
        "hi95": ci.upper,
        "per_clip_regressions": regressions,
        "strict_no_regression_pass": not regressions,
        "base_decomposition": aggregate_decompositions(base_decomps).to_dict(),
        "arm_decomposition": aggregate_decompositions(arm_decomps).to_dict(),
        "per_clip": rows,
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"\nmean delta {deltas.mean():+.4f} [{ci.lower:+.4f}, {ci.upper:+.4f}] "
        f"over {len(rows)} clips; notes applied {summary['notes_applied']}"
    )
    print(
        "strict per-clip no-regression: "
        + ("PASS" if not regressions else f"FAIL on {regressions}")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
