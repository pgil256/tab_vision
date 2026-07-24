"""Accuracy-loop Q6 — is a specification-derived stiffness table usable?

The pilot's table was fitted from GuitarSet labels, which made the channel an
artefact of one dataset. :mod:`tabvision.fusion.string_physics` computes the
same quantity from published string specifications instead, so it applies to
any instrument whose set and scale length are known. This asks what that
costs in Tab F1, and separates two failure modes:

- ``physics`` — the table exactly as computed.
- ``physics+offset`` — the same table shifted by one scalar. Scale length,
  core-wire tolerance and Young's modulus all enter as a near-common factor,
  and a single offset is the cheapest thing a session could ever learn, so
  this isolates *shape* error from *level* error.

Shape is what matters: the decision only needs the strings to be correctly
*ordered and spaced*, since a shared level shift moves every candidate alike.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.q6_gate_a import LOG2, collect_measurements
from scripts.eval.q6_self_calibration import base_decomp_events  # noqa: F401
from tabvision.fusion.inharmonicity import StringStiffnessModel
from tabvision.fusion.string_physics import reference_stiffness_model

DEV_PLAYERS = ("00", "01", "02", "03", "04")


def fitted_table(rows: list[dict[str, Any]]) -> dict[int, float]:
    table: dict[int, float] = {}
    for string in range(6):
        values = [r["log_b"] - (r["fret"] / 6.0) * LOG2 for r in rows if r["string"] == string]
        if values:
            table[string] = float(np.median(values))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_home = Path(os.environ.get("TABVISION_DATA_ROOT", "")) / "guitarset"
    rows = [r for r in collect_measurements(data_home, DEV_PLAYERS, 0, "mono") if r["r2"] >= 0.70]
    fitted = fitted_table(rows)
    physics = reference_stiffness_model()
    offset = float(np.median([physics.log_b0[s] - fitted[s] for s in fitted]))
    shifted = StringStiffnessModel(log_b0={s: physics.log_b0[s] - offset for s in physics.log_b0})
    residual = [shifted.log_b0[s] - fitted[s] for s in fitted]
    summary = {
        "median_offset_log": offset,
        "offset_as_b_ratio": float(np.exp(offset)),
        "residual_after_offset_sd_log": float(np.std(residual)),
        "residual_as_b_ratio": float(np.exp(np.std(residual))),
        "per_string": {
            str(s): {
                "physics": physics.log_b0[s],
                "physics_shifted": shifted.log_b0[s],
                "fitted": fitted[s],
                "residual": shifted.log_b0[s] - fitted[s],
            }
            for s in sorted(fitted)
        },
        "separation_to_resolve_b_ratio": [1.59, 1.78],
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
