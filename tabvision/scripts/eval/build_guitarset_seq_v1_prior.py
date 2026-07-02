"""Build the checked-in ``guitarset-seq-v1`` transition-prior artifact (A15).

Counts are ``(Δpitch, Δstring, prev-anchor fret)`` transition samples from
the GuitarSet train split (validation player excluded — same hygiene as the
``guitarset-v1`` unigram prior). Samples are restricted to
singleton→singleton cluster moves because the decode gates the learned
term to exactly those transitions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tabvision.eval.guitarset_audio import (
    DEFAULT_VALIDATION_PLAYER,
    list_guitarset_track_ids,
    parse_guitarset_jams,
)
from tabvision.fusion.transition_prior import extract_transitions

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "tabvision"
    / "fusion"
    / "priors"
    / "guitarset_seq_v1.json"
)


def build_payload(
    *,
    data_home: Path,
    validation_player: str,
    scheme: str,
    alpha: float,
    backoff_kappa: float,
    singleton_only: bool,
) -> dict:
    track_ids = list_guitarset_track_ids(
        data_home,
        split="train",
        validation_player=validation_player,
    )
    if not track_ids:
        raise RuntimeError(f"no GuitarSet train tracks found under {data_home}")

    counts: Counter[tuple[int, int, int]] = Counter()
    for track_id in track_ids:
        jams_path = data_home / "annotation" / f"{track_id}.jams"
        events = parse_guitarset_jams(jams_path)
        for sample in extract_transitions(events, singleton_only=singleton_only):
            counts[sample] += 1

    rows = [[dp, ds, prev_fret, count] for (dp, ds, prev_fret), count in sorted(counts.items())]
    return {
        "schema_version": 1,
        "name": "guitarset-seq-v1",
        "source": (
            "Anchor-to-anchor transition counts from the GuitarSet train split "
            "(validation player excluded), singleton-to-singleton cluster moves only "
            "to match the decode-side gating."
        ),
        "validation_player": validation_player,
        "training_tracks": len(track_ids),
        "scheme": scheme,
        "alpha": alpha,
        "backoff_kappa": backoff_kappa,
        "singleton_only": singleton_only,
        "counts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, required=True)
    parser.add_argument("--validation-player", default=DEFAULT_VALIDATION_PLAYER)
    parser.add_argument("--scheme", choices=("delta", "delta_fret"), default="delta_fret")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--backoff-kappa", type=float, default=8.0)
    parser.add_argument("--all-moves", action="store_true", help="include chord-cluster moves")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(
        data_home=args.data_home,
        validation_player=args.validation_player,
        scheme=args.scheme,
        alpha=args.alpha,
        backoff_kappa=args.backoff_kappa,
        singleton_only=not args.all_moves,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"tracks={payload['training_tracks']}")
    print(f"rows={len(payload['counts'])}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
