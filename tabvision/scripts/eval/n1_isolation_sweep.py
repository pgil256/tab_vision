"""Accuracy-loop N1 — does partial-aware isolation convert coverage to Tab F1?

The coverage diagnostic found strict isolation costs **88% of all reachable
notes**, far more than fit quality (2.2%). ``partial_aware`` drops only the
partials a simultaneous note collides with and fits the rest. The open
question is whether the notes it recovers carry usable measurements or
contaminated ones — more coverage at lower accuracy could easily be worse.

Paired against the shipped v1 (`strict`) on the banked 20-clip set, LOPO
priors, everything else identical, so the only variable is the isolation mode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import (
    _event_from_json,
    _score,
    build_oof_priors,
    select_clips,
)
from tabvision.eval.bootstrap import bootstrap_ci
from tabvision.eval.error_decomposition import aggregate_decompositions
from tabvision.eval.guitarset_audio import load_mono_audio, parse_guitarset_jams
from tabvision.fusion.inharmonicity import attach_inharmonicity_evidence
from tabvision.fusion.string_physics import load_string_evidence
from tabvision.types import GuitarConfig, SessionConfig

ARMS = ("baseline", "strict", "pa4", "pa6", "pa8")
PA = {"pa4": 4, "pa6": 6, "pa8": 8}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=10)
    parser.add_argument("--all-dev", action="store_true")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    cache_dir = args.cache_dir or (data_root / "models" / "q6_full_dev_cache")

    cfg, session = GuitarConfig(), SessionConfig()
    evidence = load_string_evidence()
    priors = build_oof_priors(data_home, cfg)
    if args.all_dev:
        from scripts.eval.n2_muscriptor_merge import DEV_PLAYERS

        clips = sorted(
            path.stem
            for path in (data_home / "annotation").glob("*.jams")
            if path.stem[:2] in DEV_PLAYERS
        )
    else:
        clips = select_clips(data_home, "comp", args.clips) + select_clips(
            data_home, "solo", args.clips
        )

    scores: dict[str, list[dict[str, float]]] = {a: [] for a in ARMS}
    decomps: dict[str, list] = {a: [] for a in ARMS}
    cover: dict[str, dict[str, int]] = {
        a: {"events": 0, "isolated": 0, "fitted": 0, "applied": 0} for a in ARMS
    }
    modes: list[str] = []

    for track_id in clips:
        cache = cache_dir / f"{track_id}.ensemble.json"
        if not cache.is_file():
            continue
        events = [_event_from_json(x) for x in json.loads(cache.read_text("utf-8"))]
        gold = parse_guitarset_jams(data_home / "annotation" / f"{track_id}.jams", cfg)
        wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
        prior = priors[track_id[:2]]
        modes.append("solo" if track_id.endswith("_solo") else "comp")

        for arm in ARMS:
            if arm == "baseline":
                prepared = events
            else:
                prepared, tally = attach_inharmonicity_evidence(
                    events,
                    wav,
                    int(sr),
                    evidence.model,
                    cfg,
                    weight=evidence.weight,
                    min_r2=evidence.min_r2,
                    sigma=evidence.sigma,
                    isolation="strict" if arm == "strict" else "partial_aware",
                    min_clean_partials=PA.get(arm, 4),
                )
                for key in cover[arm]:
                    cover[arm][key] += tally[key]
            metrics, decomp = _score(prepared, gold, cfg=cfg, session=session, prior=prior)
            scores[arm].append(metrics)
            decomps[arm].append(decomp)
        if len(scores["baseline"]) % 25 == 0:
            print(f"  [{len(scores['baseline'])}/{len(clips)}]", flush=True)

    base = np.asarray([r["tab_f1"] for r in scores["baseline"]])
    solo = np.asarray([m == "solo" for m in modes])
    out: dict[str, Any] = {"clips": len(base), "coverage": cover, "arms": {}}
    for arm in ARMS:
        tab = np.asarray([r["tab_f1"] for r in scores[arm]])
        delta = tab - base
        ci = bootstrap_ci(delta, n_bootstrap=10_000, seed=42)
        sci = bootstrap_ci(delta[solo], n_bootstrap=10_000, seed=42)
        cci = bootstrap_ci(delta[~solo], n_bootstrap=10_000, seed=42)
        out["arms"][arm] = {
            "tab_f1": float(tab.mean()),
            "delta": float(delta.mean()),
            "lo95": ci.lower,
            "hi95": ci.upper,
            "solo_delta": float(delta[solo].mean()),
            "solo_lo95": sci.lower,
            "comp_delta": float(delta[~solo].mean()),
            "comp_lo95": cci.lower,
            "onset_f1": float(np.mean([r["onset_f1"] for r in scores[arm]])),
            "decomposition": aggregate_decompositions(decomps[arm]).to_dict(),
        }
    # Paired strict -> partial_aware, the comparison that decides the change.
    pa = np.asarray([r["tab_f1"] for r in scores["pa6"]])
    st = np.asarray([r["tab_f1"] for r in scores["strict"]])
    head = bootstrap_ci(pa - st, n_bootstrap=10_000, seed=42)
    out["pa6_vs_strict"] = {
        "delta": float((pa - st).mean()),
        "lo95": head.lower,
        "hi95": head.upper,
    }

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print()
    for arm in ARMS:
        a = out["arms"][arm]
        print(
            f"  {arm:>14}: tab={a['tab_f1']:.4f} delta={a['delta']:+.4f} "
            f"[{a['lo95']:+.4f}, {a['hi95']:+.4f}] solo={a['solo_delta']:+.4f} "
            f"comp={a['comp_delta']:+.4f} onset={a['onset_f1']:.4f}"
        )
    for arm in ("strict", "pa4", "pa6", "pa8"):
        c = cover[arm]
        print(
            f"  {arm:>14} coverage: applied={c['applied']}/{c['events']} "
            f"({c['applied'] / max(c['events'], 1):.2%})"
        )
    h = out["pa6_vs_strict"]
    print(f"\n  pa6 - strict = {h['delta']:+.4f} [{h['lo95']:+.4f}, {h['hi95']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
