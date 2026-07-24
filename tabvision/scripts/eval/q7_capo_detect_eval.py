"""Accuracy-loop Q7 — how well can the capo be detected from a recording?

The capo-covariant prior is worth ~+0.37 Tab F1 to a capo user, but only if
the capo is known; today it must be supplied by hand. This measures whether
it can be inferred, on ground truth we constructed and therefore know exactly.

60 cases: 20 GuitarSet clips at capo 0 (original audio) and at capo 2 and 4
(the pitch-shifted audio banked by ``q7_capo_audio_eval.py``). Two estimators,
both from :mod:`tabvision.preflight.capo`:

- ``pitches`` — physical floor plus open-string occupancy. A repertoire
  heuristic; pitch content alone cannot separate a capo from a transposition.
- ``inharmonicity`` — measured stiffness against the stiffness each capo
  hypothesis implies. ``B`` depends on the absolute fret, so this is causally
  tied to the capo rather than to what was played.

Reported as exact accuracy, within-one accuracy, and mean signed error, so a
detector that is merely biased is distinguishable from one that is noisy.
Nothing here changes routing.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.n2_muscriptor_merge import _event_from_json, select_clips
from tabvision.eval.guitarset_audio import load_mono_audio
from tabvision.fusion.string_physics import load_string_evidence
from tabvision.preflight.capo import (
    detect_capo_from_inharmonicity,
    detect_capo_from_pitches,
)
from tabvision.types import GuitarConfig

CAPOS = (0, 2, 4)


def _load_events(path: Path) -> list:
    return [_event_from_json(item) for item in json.loads(path.read_text("utf-8"))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--capo-cache", type=Path, default=None)
    parser.add_argument("--capo0-cache", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(os.environ.get("TABVISION_DATA_ROOT", ""))
    data_home = args.data_home or (data_root / "guitarset")
    capo_cache = args.capo_cache or (data_root / "models" / "q7_capo_cache")
    capo0_cache = args.capo0_cache or (data_root / "models" / "q6_full_dev_cache")

    cfg = GuitarConfig()
    evidence = load_string_evidence()
    clips = select_clips(data_home, "comp", 10) + select_clips(data_home, "solo", 10)

    rows: list[dict[str, Any]] = []
    for track_id in clips:
        for capo in CAPOS:
            if capo == 0:
                events_path = capo0_cache / f"{track_id}.ensemble.json"
                if not events_path.is_file():
                    continue
                wav, sr = load_mono_audio(data_home / "audio_mono-mic" / f"{track_id}_mic.wav")
            else:
                events_path = capo_cache / f"{track_id}.capo{capo}.json"
                shifted = capo_cache / f"{track_id}.shift{capo}.npy"
                if not events_path.is_file() or not shifted.is_file():
                    continue
                wav = np.load(shifted)
                sr = 44100
            events = _load_events(events_path)
            if not events:
                continue

            by_pitch = detect_capo_from_pitches(events, cfg)
            by_physics = detect_capo_from_inharmonicity(
                events, wav, int(sr), evidence.model, cfg, min_r2=evidence.min_r2
            )
            rows.append(
                {
                    "track_id": track_id,
                    "true_capo": capo,
                    "pitches": by_pitch.capo,
                    "pitches_conf": by_pitch.confidence,
                    "physics": by_physics.capo,
                    "physics_conf": by_physics.confidence,
                    "physics_method": by_physics.method,
                    "upper_bound": by_pitch.upper_bound,
                }
            )
            print(
                f"  {track_id} capo{capo}: pitches={by_pitch.capo} "
                f"physics={by_physics.capo} (bound<={by_pitch.upper_bound})",
                flush=True,
            )

    if not rows:
        raise SystemExit("no cases scored — check the caches")

    summary: dict[str, Any] = {"cases": len(rows), "methods": {}}
    for method in ("pitches", "physics"):
        errors = [row[method] - row["true_capo"] for row in rows]
        exact = sum(1 for e in errors if e == 0) / len(errors)
        within1 = sum(1 for e in errors if abs(e) <= 1) / len(errors)
        summary["methods"][method] = {
            "exact": exact,
            "within_1": within1,
            "mean_signed_error": float(np.mean(errors)),
            "mean_abs_error": float(np.mean(np.abs(errors))),
            "per_true_capo": {
                str(c): {
                    "exact": sum(1 for row in rows if row["true_capo"] == c and row[method] == c)
                    / max(1, sum(1 for row in rows if row["true_capo"] == c)),
                    "predictions": dict(
                        Counter(row[method] for row in rows if row["true_capo"] == c)
                    ),
                }
                for c in CAPOS
            },
        }
    summary["bound_valid"] = sum(1 for row in rows if row["upper_bound"] >= row["true_capo"]) / len(
        rows
    )
    summary["per_case"] = rows

    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} cases")
    for method in ("pitches", "physics"):
        m = summary["methods"][method]
        print(
            f"  {method:>12}: exact={m['exact']:.3f} within1={m['within_1']:.3f} "
            f"mean_signed={m['mean_signed_error']:+.2f} mae={m['mean_abs_error']:.2f}"
        )
        for c in CAPOS:
            row = m["per_true_capo"][str(c)]
            print(f"      true capo {c}: exact={row['exact']:.2f} preds={row['predictions']}")
    print(f"  physical bound valid (bound >= true capo): {summary['bound_valid']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
