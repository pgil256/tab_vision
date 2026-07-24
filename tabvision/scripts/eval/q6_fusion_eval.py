"""Accuracy-loop Q6 — does inharmonicity evidence actually move Tab F1?

Gates A/B established the physics (0.92 string accuracy vs a 0.65 control)
and the detected-notes probe showed it survives real onsets and pitches. Both
measured *string classification*. This measures the only thing that decides
promotion: **Tab F1 through the real `fuse()`**, with the evidence folded in
as a bounded product-of-experts term next to the corpus prior.

The A14 precedent is exactly this gap — per-note evidence that scored well in
isolation and still failed to lift the decoder, because the decoder was
already right on most of the notes the evidence covered.

Offline replay of the banked 20-clip ensemble cache with the leave-one-
player-out position prior and `guitarset-seq-v1` @ w=4.0. The stiffness model
is calibrated per fold from *other* players' gold notes. `auto` is untouched:
nothing here changes pipeline defaults.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from scripts.eval.n2_muscriptor_merge import (
    _event_from_json,
    _score,
    build_oof_priors,
    select_clips,
)
from scripts.eval.q6_gate_a import LOG2, collect_measurements
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import ErrorDecomposition, aggregate_decompositions
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.fusion.inharmonicity import (
    StringStiffnessModel,
    attach_inharmonicity_evidence,
)
from tabvision.types import GuitarConfig, SessionConfig

DEV_PLAYERS = ("00", "01", "02", "03", "04")
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
CALIBRATION_MIN_R2 = 0.70


def calibrate(rows: list[dict[str, Any]]) -> dict[str, StringStiffnessModel]:
    """Leave-one-player-out stiffness models from gold measurements."""
    usable = [row for row in rows if row["r2"] >= CALIBRATION_MIN_R2]
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_player[row["player"]].append(row)
    models: dict[str, StringStiffnessModel] = {}
    for held_out in DEV_PLAYERS:
        train = [r for p, items in by_player.items() if p != held_out for r in items]
        table: dict[int, float] = {}
        for string in range(6):
            values = [r["log_b"] - (r["fret"] / 6.0) * LOG2 for r in train if r["string"] == string]
            if values:
                table[string] = float(np.median(values))
        models[held_out] = StringStiffnessModel(log_b0=table)
    return models


def run(
    clips: list[str],
    *,
    data_home: Path,
    workdir: Path,
    models: dict[str, StringStiffnessModel],
    arms: list[tuple[str, float, float]],
) -> dict[str, Any]:
    cfg = GuitarConfig()
    session = SessionConfig()
    priors = build_oof_priors(data_home, cfg)

    scores: dict[str, list[dict[str, float]]] = {name: [] for name, _, _ in arms}
    decomps: dict[str, list[ErrorDecomposition]] = {name: [] for name, _, _ in arms}
    tallies: Counter[str] = Counter()
    per_clip: list[dict[str, Any]] = []

    for track_id in clips:
        cache = workdir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            raise SystemExit(f"missing ensemble cache for {track_id}")
        events = [_event_from_json(item) for item in json.loads(cache.read_text("utf-8"))]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        wav, sr = sf.read(
            data_home / "audio_mono-mic" / f"{track_id}_mic.wav", dtype="float32", always_2d=False
        )
        player = track_id[:2]
        prior = priors[player]
        model = models[player]
        mode = "solo" if track_id.endswith("_solo") else "comp"

        row: dict[str, Any] = {"track_id": track_id, "mode": mode}
        for name, weight, min_r2 in arms:
            prepared = events
            if weight > 0.0:
                prepared, tally = attach_inharmonicity_evidence(
                    events, wav, int(sr), model, cfg, weight=weight, min_r2=min_r2
                )
                if name == arms[1][0]:
                    for key, value in tally.items():
                        tallies[key] += value
            metrics, decomposition = _score(prepared, gold, cfg=cfg, session=session, prior=prior)
            scores[name].append(metrics)
            decomps[name].append(decomposition)
            row[name] = metrics
        per_clip.append(row)
        print(
            f"{track_id} ({mode}): "
            + " ".join(f"{name}={row[name]['tab_f1']:.4f}" for name, _, _ in arms),
            flush=True,
        )

    summary: dict[str, Any] = {"clips": clips, "coverage": dict(tallies), "per_clip": per_clip}
    baseline = np.asarray([r["tab_f1"] for r in scores[arms[0][0]]], dtype=np.float64)
    solo_mask = np.asarray([r["mode"] == "solo" for r in per_clip])
    arms_out: dict[str, Any] = {}
    for name, weight, min_r2 in arms:
        tab = np.asarray([r["tab_f1"] for r in scores[name]], dtype=np.float64)
        delta = tab - baseline
        ci = bootstrap_ci(delta, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        solo_ci = bootstrap_ci(delta[solo_mask], n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
        arms_out[name] = {
            "weight": weight,
            "min_r2": min_r2,
            "tab_f1": float(tab.mean()),
            "tab_f1_delta": float(delta.mean()),
            "tab_f1_lo95": ci.lower,
            "tab_f1_hi95": ci.upper,
            "solo_tab_f1": float(tab[solo_mask].mean()),
            "solo_delta": float(delta[solo_mask].mean()),
            "solo_lo95": solo_ci.lower,
            "solo_hi95": solo_ci.upper,
            "onset_f1": float(np.mean([r["onset_f1"] for r in scores[name]])),
            "pitch_f1": float(np.mean([r["pitch_f1"] for r in scores[name]])),
            "decomposition": aggregate_decompositions(decomps[name]).to_dict(),
        }
    summary["arms"] = arms_out
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    workdir = args.workdir or (data_root / "models" / "muscriptor_probe")

    print("calibrating stiffness models (LOPO, gold mono-mic)...", flush=True)
    models = calibrate(collect_measurements(data_home, DEV_PLAYERS, 0, "mono"))

    arms = [
        ("baseline", 0.0, 0.0),
        ("inharm-w0.5", 0.5, 0.50),
        ("inharm-w1.0", 1.0, 0.50),
        ("inharm-w0.5-r70", 0.5, 0.70),
    ]
    clips = select_clips(data_home, "comp", 10) + select_clips(data_home, "solo", 10)
    summary = run(clips, data_home=data_home, workdir=workdir, models=models, arms=arms)

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\ncoverage: {summary['coverage']}")
    for name, _, _ in arms:
        row = summary["arms"][name]
        print(
            f"  {name:>16}: tab={row['tab_f1']:.4f} "
            f"delta={row['tab_f1_delta']:+.4f} "
            f"[{row['tab_f1_lo95']:+.4f}, {row['tab_f1_hi95']:+.4f}] | "
            f"solo {row['solo_delta']:+.4f} "
            f"[{row['solo_lo95']:+.4f}, {row['solo_hi95']:+.4f}] | "
            f"onset={row['onset_f1']:.4f} pitch={row['pitch_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
