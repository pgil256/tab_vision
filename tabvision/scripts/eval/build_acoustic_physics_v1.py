"""Build the ``acoustic-physics-v1`` string-evidence artifact.

Emits the inharmonicity channel's stiffness table plus its gate-passed
configuration as a hash-verified registry artifact, so the exact numbers that
cleared full-dev OOF and the player-05 confirmation are reproducible rather
than recomputed at import time.

Unlike every other artifact in this registry the payload is **not fitted to a
dataset** — it is computed from published string specifications
(:mod:`tabvision.fusion.string_physics`), so GuitarSet is a *test* of it, not
its source. The provenance block records the specification inputs so the table
can be rebuilt or re-derived for a different string set or scale length.

The decode configuration (weight, ``min_r2``, ``sigma``) lives in the artifact
deliberately: it was frozen before the full-dev run and must not drift, so it
is covered by the artifact hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from tabvision.fusion.inharmonicity import DEFAULT_MIN_R2, DEFAULT_SIGMA
from tabvision.fusion.string_physics import (
    ACOUSTIC_LIGHT_SET,
    DEFAULT_SCALE_LENGTH_IN,
    STEEL_YOUNGS_MODULUS_PA,
    inharmonicity_coefficient,
    reference_stiffness_model,
)

NAME = "acoustic-physics-v1"
GATE_WEIGHT = 1.0
"""Weight the full-dev and player-05 gates ran at (not the module default)."""


def build_artifact() -> dict[str, object]:
    model = reference_stiffness_model()
    return {
        "schema_version": 1,
        "name": NAME,
        "artifact_kind": "string_evidence",
        "method": "inharmonicity",
        "log_b0": {str(index): value for index, value in sorted(model.log_b0.items())},
        "fret_exponent": model.fret_exponent,
        "open_b": {
            str(index): inharmonicity_coefficient(spec)
            for index, spec in enumerate(ACOUSTIC_LIGHT_SET)
        },
        "decode": {
            "weight": GATE_WEIGHT,
            "min_r2": DEFAULT_MIN_R2,
            "sigma": DEFAULT_SIGMA,
        },
        "physics": {
            "relation": "B = pi^3 * E * d_core^4 / (256 * mu * L^4 * f^2)",
            "fret_law": "B(s, n) = B0_s * 2^(n/6), derived from L_n = L*2^(-n/12)",
            "scale_length_in": DEFAULT_SCALE_LENGTH_IN,
            "youngs_modulus_pa": STEEL_YOUNGS_MODULUS_PA,
            "string_set": [asdict(spec) for spec in ACOUSTIC_LIGHT_SET],
        },
        "domain": {
            "instrument": "acoustic",
            "tone": "clean",
            "tuning": "standard",
            "capo": 0,
            "note": (
                "Steel-string only. Nylon is ~65x less inharmonic (E ~3 GPa vs "
                "~200 GPa) and needs its own table; the channel abstains "
                "outside this domain via stiffness_model_for_session."
            ),
        },
    }


def build_manifest(artifact_file: str, sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "string_evidence",
        "artifact_version": NAME,
        "name": NAME,
        "artifact_file": artifact_file,
        "artifact_sha256": sha256,
        "registered": True,
        "compatible_position_prior": None,
        "compatible_sequence_prior": None,
        "mode": "all",
        "dataset": "none — specification-derived, not fitted",
        "dataset_reference": (
            "Stiff-string theory; typical light-gauge phosphor-bronze acoustic set"
        ),
        "license": "n/a (computed from published string specifications)",
        "construction_command": "python -m scripts.eval.build_acoustic_physics_v1",
        "parameters": {"weight": GATE_WEIGHT, "min_r2": DEFAULT_MIN_R2, "sigma": DEFAULT_SIGMA},
        "gate": {
            "report": "docs/EVAL_REPORTS/q6_player05_confirm_2026-07-22.md",
            "comparison": "inharmonicity evidence vs baseline, frozen config",
            "development": {
                "report": "docs/EVAL_REPORTS/q6_full_dev_2026-07-22.md",
                "clips": 300,
                "mean_delta": 0.0443,
                "lower_95": 0.0339,
                "upper_95": 0.0555,
            },
            "held_out": {
                "player": "05",
                "clips": 60,
                "mean_delta": 0.0780,
                "lower_95": 0.0502,
                "upper_95": 0.1078,
                "solo_mean_delta": 0.1396,
                "comp_mean_delta": 0.0164,
            },
            "decision": "passed development OOF and player-05 confirmation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors-dir", type=Path, default=None)
    args = parser.parse_args()

    priors_dir = args.priors_dir or (
        Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors"
    )
    artifact_file = "acoustic_physics_v1.json"
    artifact_path = priors_dir / artifact_file
    manifest_path = priors_dir / "acoustic_physics_v1.manifest.json"

    artifact_path.write_text(json.dumps(build_artifact(), indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(build_manifest(artifact_file, sha), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {artifact_path}\nwrote {manifest_path}\nsha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
