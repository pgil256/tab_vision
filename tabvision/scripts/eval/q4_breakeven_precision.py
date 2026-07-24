"""Accuracy-loop Q4 — calibrate the second-opinion bench's leg-2 threshold.

Q1 introduced "added-note precision >= 0.5" as the second gate leg after
complementarity alone waved MuScriptor through to a merge that lost 0.0167
Tab F1. The 0.5 was a judgement call. This derives it instead, from the
metric's own algebra, and checks the result against the banked N2 sweep.

Setup. A merge admits ``a`` notes, of which a fraction ``p`` (the added-note
precision) are real notes the ensemble missed. Each real one converts a
false negative into a true positive *only if* the decoder also assigns it the
right string — call that probability ``alpha``. The rest become false
positives. Writing ``D = 2TP + FP + FN`` and ``F1 = 2TP/D``:

    TP' = TP + alpha*p*a
    FN' = FN - p*a
    FP' = FP + a - alpha*p*a
    D'  = D + a*(1 + alpha*p - p)

Requiring ``F1' > F1`` and substituting ``TP/D = F1/2`` gives a threshold
independent of ``a`` — how *many* notes you admit does not change the sign,
only the magnitude:

    p > (F1/2) / (alpha*(1 - F1/2) + F1/2)

So the bar rises with the stream's existing F1 (the better the transcription,
the purer an addition must be to help) and falls as the decoder gets better
at placing rescued notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def breakeven_precision(f1: float, alpha: float) -> float:
    """Minimum added-note precision for a merge to be F1-neutral."""
    half = f1 / 2.0
    return half / (alpha * (1.0 - half) + half)


def measured_alpha(variants: dict[str, Any]) -> dict[str, float]:
    """Per-variant P(added real note ends up tab-correct), from the bank.

    ``added_true_notes`` counts real notes admitted; the rise in the
    decomposition's ``correct`` bucket counts how many of them survived
    string assignment. Their ratio is alpha, measured rather than assumed.
    """
    baseline_correct = variants["ensemble"]["decomposition"]["correct"]
    out: dict[str, float] = {}
    for name, payload in variants.items():
        if name == "ensemble" or not payload["added_true_notes"]:
            continue
        gained = payload["decomposition"]["correct"] - baseline_correct
        out[name] = gained / payload["added_true_notes"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_json.read_text(encoding="utf-8"))
    variants = pilot["variants"]
    baseline_f1 = variants["ensemble"]["tab_f1_mean"]

    alphas = measured_alpha(variants)
    pooled_alpha = sum(alphas.values()) / len(alphas) if alphas else float("nan")

    rows = []
    for name, payload in variants.items():
        if name == "ensemble":
            continue
        alpha = alphas.get(name)
        threshold = breakeven_precision(baseline_f1, alpha) if alpha else None
        precision = payload["added_precision"]
        rows.append(
            {
                "variant": name,
                "added_notes": payload["added_notes"],
                "added_precision": precision,
                "measured_alpha": alpha,
                "breakeven_precision": threshold,
                "predicted_sign": (
                    None
                    if threshold is None
                    else ("positive" if precision > threshold else "negative")
                ),
                "observed_tab_f1_delta": payload["tab_f1_delta"],
                "observed_sign": "positive" if payload["tab_f1_delta"] > 0 else "negative",
            }
        )

    agree = sum(
        1 for row in rows if row["predicted_sign"] and row["predicted_sign"] == row["observed_sign"]
    )
    checked = sum(1 for row in rows if row["predicted_sign"])
    summary = {
        "source": str(args.pilot_json),
        "baseline_tab_f1": baseline_f1,
        "measured_alpha_per_variant": alphas,
        "pooled_alpha": pooled_alpha,
        "breakeven_at_pooled_alpha": breakeven_precision(baseline_f1, pooled_alpha),
        "breakeven_at_alpha_0_65": breakeven_precision(baseline_f1, 0.65),
        "variants": rows,
        "sign_agreement": f"{agree}/{checked}",
    }
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"baseline Tab F1 = {baseline_f1:.4f}")
    print(f"measured alpha (added real note -> tab-correct): {pooled_alpha:.4f}")
    print(
        f"break-even added-note precision = "
        f"{summary['breakeven_at_pooled_alpha']:.4f} "
        f"(at alpha=0.65 it would be {summary['breakeven_at_alpha_0_65']:.4f})"
    )
    print()
    for row in rows:
        threshold = row["breakeven_precision"]
        print(
            f"  {row['variant']:>14}: precision={row['added_precision']:.3f} "
            f"vs breakeven={threshold:.3f} -> predicted {row['predicted_sign']}, "
            f"observed {row['observed_sign']} ({row['observed_tab_f1_delta']:+.4f})"
            if threshold
            else f"  {row['variant']:>14}: no real additions"
        )
    print(f"\nsign agreement: {summary['sign_agreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
