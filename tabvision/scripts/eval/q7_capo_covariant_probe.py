"""Accuracy-loop Q7 (ROI deep-dive §4.3) — capo-covariant prior entry probe.

Today ``resolve_inference_policy`` routes *any* capo>0 session to
``priors=none``, so the +22 pp position-prior lift silently vanishes on
exactly the recordings a personal user makes with a capo. §4.3 proposes
making the prior **capo-covariant** — shift the fret axis by the capo before
applying it — so the validated domain widens to capo 0-7.

This is the probe-before-build entry gate: does the shift actually recover the
lift, mechanism-only, before any preflight/pipeline code is written?

It works at the label level on GuitarSet dev gold (players 00-04), so it needs
no audio and no re-transcription — the same move Q2/Q6 used to decide whether
a lever existed. A note ``(s0, f0, pitch0)`` played under a capo at ``C`` is
the same shape ``C`` frets up: gold becomes ``(s0, f0+C, pitch0+C)``. The
capo-covariant score for a candidate ``(s, fret)`` at capo ``C`` is the
capo-0 prior read ``C`` frets and semitones lower:

    covariant(s, fret | P, C) = prior_capo0(s, fret - C | P - C)

Arms, scored by top-1 assignment accuracy on ambiguous notes (the slice where
string identity is actually in question):

- ``capo0-ref``   — no capo shift at all; anchors the code against Q2's known
  0.6548 dev-OOF number.
- ``covariant``   — the proposed shift. Should track ``capo0-ref``.
- ``naive``       — capo-0 prior applied at absolute shifted coordinates, i.e.
  the prior used *without* capo awareness. Shows the damage of ignoring it.
- ``none-uniform``/``none-lowfret`` — no prior, what capo sessions get today:
  uniform expectation, and the concrete lowest-fret decoder default.

Leave-one-player-out priors throughout (``guitarset-v1`` was trained on these
players). Nothing here touches the pipeline; ``auto`` is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.eval.n2_muscriptor_merge import DEV_PLAYERS, build_oof_priors
from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.fusion.candidates import candidate_positions
from tabvision.fusion.position_prior import PitchPositionPrior
from tabvision.types import GuitarConfig

CAPOS = (0, 2, 4, 7)


def _covariant_score(prior: PitchPositionPrior, pitch: int, string: int, fret: int, capo: int):
    matrix = prior.matrix_for_pitch(pitch - capo)
    if matrix is None:
        return None
    row, col = string, fret - capo
    if not (0 <= row < matrix.shape[0] and 0 <= col < matrix.shape[1]):
        return None
    return float(matrix[row, col])


def _naive_score(prior: PitchPositionPrior, pitch: int, string: int, fret: int):
    matrix = prior.matrix_for_pitch(pitch)
    if matrix is None:
        return None
    if not (0 <= string < matrix.shape[0] and 0 <= fret < matrix.shape[1]):
        return None
    return float(matrix[string, fret])


def _argmax_hit(scores: list[tuple[float, int, int]], gold: tuple[int, int]) -> int:
    """Top-1 hit under a scorer; ties keep the lowest-fret candidate.

    ``scores`` is (score, fret, ...) so that ``max`` with a fret tiebreak
    reproduces the candidate ordering ``candidate_positions`` already uses.
    """
    best = max(scores, key=lambda item: (item[0], -item[1], -item[2]))
    return int((best[1], best[2]) == (gold[1], gold[0]))


def run(data_home: Path) -> dict[str, Any]:
    cfg0 = GuitarConfig()
    priors = build_oof_priors(data_home, cfg0)
    tuning = cfg0.tuning_midi

    per_capo: dict[int, dict[str, list[int]]] = {capo: defaultdict(list) for capo in CAPOS}

    for jams_path in sorted((data_home / "annotation").glob("*.jams")):
        player = jams_path.stem[:2]
        if player not in DEV_PLAYERS:
            continue
        prior = priors[player]
        for event in parse_guitarset_jams(jams_path, cfg0):
            base_fret = event.pitch_midi - tuning[event.string_idx]
            if base_fret < 0:
                continue
            for capo in CAPOS:
                fret = base_fret + capo
                pitch = event.pitch_midi + capo
                if fret > cfg0.max_fret:
                    continue
                cfg = GuitarConfig(capo=capo)
                candidates = candidate_positions(pitch, cfg)
                if len(candidates) < 2:
                    continue  # unambiguous — string identity is not in question
                gold = (event.string_idx, fret)

                cov = [
                    (s, c.string_idx, c.fret)
                    for c in candidates
                    if (s := _covariant_score(prior, pitch, c.string_idx, c.fret, capo)) is not None
                ]
                naive = [
                    (s, c.string_idx, c.fret)
                    for c in candidates
                    if (s := _naive_score(prior, pitch, c.string_idx, c.fret)) is not None
                ]
                bucket = per_capo[capo]
                bucket["n"].append(1)
                if len(cov) == len(candidates):
                    bucket["covariant"].append(_argmax_hit([(s, f, st) for s, st, f in cov], gold))
                if len(naive) == len(candidates):
                    bucket["naive"].append(_argmax_hit([(s, f, st) for s, st, f in naive], gold))
                # No prior: uniform expectation, and the concrete lowest-fret
                # default the decoder falls back to when priors are off.
                bucket["none_uniform"].append(1.0 / len(candidates))
                low = min(candidates, key=lambda c: (c.fret, c.string_idx))
                bucket["none_lowfret"].append(int((low.string_idx, low.fret) == gold))

    summary: dict[str, Any] = {"capos": {}}
    for capo in CAPOS:
        bucket = per_capo[capo]
        n = len(bucket["n"])
        summary["capos"][str(capo)] = {
            "ambiguous_notes": n,
            "covariant": sum(bucket["covariant"]) / len(bucket["covariant"])
            if bucket["covariant"]
            else float("nan"),
            "naive": sum(bucket["naive"]) / len(bucket["naive"])
            if bucket["naive"]
            else float("nan"),
            "none_uniform": sum(bucket["none_uniform"]) / n if n else float("nan"),
            "none_lowfret": sum(bucket["none_lowfret"]) / n if n else float("nan"),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    args = parser.parse_args()

    data_home = args.data_home or (Path(os.environ.get("TABVISION_DATA_ROOT", "")) / "guitarset")
    summary = run(data_home)
    if args.json_path is not None:
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{'capo':>5} {'n':>7} {'covariant':>10} {'naive':>8} {'none-unif':>10} {'none-low':>9}")
    for capo in CAPOS:
        row = summary["capos"][str(capo)]
        print(
            f"{capo:>5} {row['ambiguous_notes']:>7} {row['covariant']:>10.4f} "
            f"{row['naive']:>8.4f} {row['none_uniform']:>10.4f} {row['none_lowfret']:>9.4f}"
        )
    ref = summary["capos"]["0"]["covariant"]
    recovered = {
        c: summary["capos"][str(c)]["covariant"] - summary["capos"][str(c)]["none_lowfret"]
        for c in CAPOS
        if c
    }
    print(f"\ncapo0 reference top-1 = {ref:.4f} (expect ~0.6548, the Q2 dev-OOF number)")
    print(
        "covariant minus none-lowfret (lift recovered): "
        + ", ".join(f"capo{c}={v:+.4f}" for c, v in recovered.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
