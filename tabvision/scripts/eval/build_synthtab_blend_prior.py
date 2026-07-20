"""Program S Phase S1a — blend SynthTab counts into the GuitarSet priors.

Second predeclared S1a arm (after the swap conditions): interpolated count
blends ``guitarset + lambda * mass-normalized(synthtab)``. SynthTab counts
are scaled so their total mass equals the GuitarSet artifact's mass, then
weighted by ``--blend-lambda`` and added; the result keeps the registered
artifact class hyperparameters. Operates on already-built artifact JSONs —
no corpus re-scan. Outputs are evaluation-only (default
``$TABVISION_DATA_ROOT/models/synthtab_priors``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

_PRIORS_DIR = Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors"
GS_POSITION = _PRIORS_DIR / "guitarset_v1.json"
GS_SEQUENCE = _PRIORS_DIR / "guitarset_seq_v1.json"


def _blend(base_path: Path, add_path: Path, lam: float, name: str) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    add = json.loads(add_path.read_text(encoding="utf-8"))
    base_total = sum(row[3] for row in base["counts"])
    add_total = sum(row[3] for row in add["counts"])
    scale = lam * base_total / add_total if add_total else 0.0
    merged: dict[tuple[int, int, int], float] = {
        (row[0], row[1], row[2]): float(row[3]) for row in base["counts"]
    }
    for row in add["counts"]:
        key = (row[0], row[1], row[2])
        merged[key] = merged.get(key, 0.0) + row[3] * scale
    payload = dict(base)
    payload["name"] = name
    payload["source"] = (
        f"Count blend: {base['name']} + {lam} x mass-normalized {add['name']} "
        f"(scale {scale:.6f}). Inherits CC-BY-NC-4.0 from the SynthTab side; "
        "evaluation-only until registered."
    )
    payload["counts"] = [
        [key[0], key[1], key[2], int(round(value))]
        for key, value in sorted(merged.items())
        if int(round(value)) > 0
    ]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priors-dir", type=Path, default=None)
    parser.add_argument("--variant", choices=("acoustic", "all"), default="all")
    parser.add_argument("--blend-lambda", type=float, action="append", dest="lambdas", default=None)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    priors_dir = args.priors_dir or (Path(data_root) / "models" / "synthtab_priors")
    lambdas = args.lambdas or [0.25, 1.0]

    st_pos = priors_dir / f"synthtab_v1_{args.variant}.json"
    st_seq = priors_dir / f"synthtab_seq_v1_{args.variant}.json"
    for lam in lambdas:
        tag = f"{lam:g}".replace(".", "p")
        outputs = (
            (
                priors_dir / f"synthtab_blend_pos_{args.variant}_l{tag}.json",
                _blend(GS_POSITION, st_pos, lam, f"synthtab-blend-pos-{tag}"),
            ),
            (
                priors_dir / f"synthtab_blend_seq_{args.variant}_l{tag}.json",
                _blend(GS_SEQUENCE, st_seq, lam, f"synthtab-blend-seq-{tag}"),
            ),
        )
        for path, payload in outputs:
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, indent=2) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"output={path}")
            print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
