"""Build the ``pdmx-seq-v1`` and ``guitarset-pdmx-seq-v1`` prior artifacts (A15).

Reads the locally cached PDMX transition counts (produced by
``scripts.acquire.pdmx_extract_transitions`` — the dataset itself is never
committed; these artifacts hold only derived ``(Δpitch, Δstring,
prev-anchor fret)`` count statistics) and writes:

- ``pdmx_seq_v1.json`` — PDMX-only counts, same schema/hyperparameters as
  ``guitarset-seq-v1``;
- ``guitarset_pdmx_seq_v1.json`` — pooled variant. Raw summing would let
  PDMX (~10× the samples) drown GuitarSet, making the pool a near-copy of
  the PDMX prior; instead the GuitarSet counts are integer-scaled up by
  ``round(pdmx_total / guitarset_total)`` so the two corpora contribute
  roughly equal probability mass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PRIORS_DIR = Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors"

HYPERPARAMS = {"scheme": "delta_fret", "alpha": 0.5, "backoff_kappa": 8.0}
"""Same hyperparameters as ``guitarset-seq-v1`` (gate-accepted config)."""


def _counts(payload: dict) -> Counter[tuple[int, int, int]]:
    return Counter({(r[0], r[1], r[2]): r[3] for r in payload["counts"]})


def _rows(counts: Counter[tuple[int, int, int]]) -> list[list[int]]:
    return [[dp, ds, prev_fret, n] for (dp, ds, prev_fret), n in sorted(counts.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transitions",
        type=Path,
        required=True,
        help="local PDMX transitions cache (pdmx_extract_transitions output)",
    )
    parser.add_argument(
        "--guitarset-artifact",
        type=Path,
        default=PRIORS_DIR / "guitarset_seq_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PRIORS_DIR)
    args = parser.parse_args()

    cache = json.loads(args.transitions.read_text(encoding="utf-8"))
    if cache.get("schema_version") != 1:
        raise SystemExit(f"unsupported transitions cache schema: {args.transitions}")
    pdmx = _counts(cache)
    pdmx_total = sum(pdmx.values())

    gs_payload = json.loads(args.guitarset_artifact.read_text(encoding="utf-8"))
    gs = _counts(gs_payload)
    gs_total = sum(gs.values())

    pdmx_payload = {
        "schema_version": 1,
        "name": "pdmx-seq-v1",
        "source": cache["source"],
        "training_scores": cache["stats"]["used"],
        **HYPERPARAMS,
        "singleton_only": True,
        "counts": _rows(pdmx),
    }

    scale = max(1, round(pdmx_total / gs_total))
    pooled = Counter({key: n * scale for key, n in gs.items()})
    pooled.update(pdmx)
    pooled_payload = {
        "schema_version": 1,
        "name": "guitarset-pdmx-seq-v1",
        "source": (
            f"Pooled guitarset-seq-v1 (counts x{scale} to match the PDMX mass; "
            f"{gs_total} raw samples) + pdmx-seq-v1 ({pdmx_total} samples). "
            "See the component artifacts for extraction provenance."
        ),
        "guitarset_scale": scale,
        **HYPERPARAMS,
        "singleton_only": True,
        "counts": _rows(pooled),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fname, payload in (
        ("pdmx_seq_v1.json", pdmx_payload),
        ("guitarset_pdmx_seq_v1.json", pooled_payload),
    ):
        out = args.output_dir / fname
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        total = sum(_counts(payload).values())
        print(f"{payload['name']}: rows={len(payload['counts'])} total={total} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
