"""Accuracy-loop Q6 (ROI deep-dive §4.1) — inharmonicity separability precursor.

Gate A needs GuitarSet's hexaphonic partition, which is a multi-GB download
this repo's acquirer deliberately skips. Before spending it, this asks a
cheaper question on data already banked: **is string identity separable by
inharmonicity at all, for the notes the decoder actually gets wrong?**

Physics. For a stiff string the partials run
``f_k = k*f0*sqrt(1 + B*k^2)``, and B scales as ``1/L^2``. Fretting at fret
``n`` shortens the speaking length by ``2^(-n/12)``, so

    B(s, n) = B0_s * 2^(n/6)

for open-string coefficient ``B0_s``. Two candidate positions for the same
pitch therefore differ in log B by

    dlogB = (log B0_s1 - log B0_s2) + (n1 - n2)/6 * log 2

The second term is **assumption-free** — it depends only on the fret
difference the lattice already records. The first depends on string
construction (plain vs wound), is genuinely large in practice, and is
*ignored* here. So every number this prints is a **lower bound** on
discriminability: real strings are easier to tell apart than this says.

Given a per-note relative error ``sigma`` on the B estimate, a two-candidate
decision is a Gaussian mean-separation problem, so the achievable accuracy is
``Phi(dlogB / (2*sigma))``. Sweeping sigma converts "how good must the
B-estimator be" into "what string accuracy would follow" — which is exactly
the quantity Gate A would measure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

LOG2 = math.log(2.0)
SIGMAS = (0.02, 0.05, 0.10, 0.20, 0.30)
GATE_A = 0.85
GATE_B = 0.70


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    fret_gaps: Counter[int] = Counter()
    by_mode: dict[str, Counter[int]] = {"solo": Counter(), "comp": Counter()}
    with args.lattice.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "production_equivalent":
                continue
            if row["evaluation_split"] != "development_oof":
                continue
            if row["ambiguous_pitch_match"] != "1":
                continue
            candidates = [item for item in row["candidate_path"].split(";") if item]
            if len(candidates) < 2:
                continue
            first = candidates[0].split(":")
            second = candidates[1].split(":")
            gap = abs(int(first[1]) - int(second[1]))
            fret_gaps[gap] += 1
            by_mode[row["mode"]][gap] += 1

    total = sum(fret_gaps.values())
    if not total:
        raise SystemExit("no ambiguous notes with >= 2 candidates found")

    # Expected accuracy of the rank-1 vs rank-2 decision at each sigma,
    # weighting each note by how often its fret gap occurs.
    accuracy: dict[float, float] = {}
    for sigma in SIGMAS:
        hits = 0.0
        for gap, count in fret_gaps.items():
            separation = (gap / 6.0) * LOG2
            hits += count * normal_cdf(separation / (2.0 * sigma))
        accuracy[sigma] = hits / total

    summary = {
        "ambiguous_notes": total,
        "fret_gap_histogram": dict(sorted(fret_gaps.items())),
        "median_fret_gap": sorted(fret_gaps.elements())[total // 2],
        "b_ratio_at_median": 2 ** (sorted(fret_gaps.elements())[total // 2] / 6.0),
        "accuracy_lower_bound_by_sigma": accuracy,
        "gate_a": GATE_A,
        "gate_b": GATE_B,
        "sigma_needed_for_gate_a": next((s for s in SIGMAS if accuracy[s] >= GATE_A), None),
        "per_mode_fret_gap": {
            mode: dict(sorted(counter.items())) for mode, counter in by_mode.items()
        },
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"ambiguous notes (dev-OOF, >=2 candidates): {total}")
    print("\nfret gap between rank-1 and rank-2 candidates:")
    for gap, count in sorted(fret_gaps.items()):
        share = count / total
        print(
            f"  {gap:2d} frets: {count:6d} ({share:6.1%})  "
            f"B ratio from length alone = {2 ** (gap / 6.0):.2f}x"
        )
    print("\nlower-bound string accuracy vs B-estimator relative error:")
    for sigma in SIGMAS:
        verdict = "clears gate A" if accuracy[sigma] >= GATE_A else ""
        print(f"  sigma={sigma:4.0%}: {accuracy[sigma]:.4f} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
