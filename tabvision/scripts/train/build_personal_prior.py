"""Build a personal position-prior artifact from harvested labels.

Usage (from the ``tabvision/`` package directory):

    python -m scripts.train.build_personal_prior \
        ~/.tabvision/personal/labels.jsonl \
        -o ~/.tabvision/personal/personal-prior.json

Labels come from ``tabvision transcribe ... --video-backend fretcam
--harvest-personal-labels STORE.jsonl`` and accumulate across sessions.
The output artifact is consumed with ``--position-prior <path>.json``.

Posture: SPEC §1.5 carve-out (2026-08-02). The artifact is local to this
user — never shipped, never registered as a default, never used in eval
corpora or published figures. Track C priced its in-sample ceiling at
+0.0305 aggregate Tab F1 (`c_prior_adaptation_2026-07-25.md`); what a
finite label store actually reaches is unmeasured.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tabvision.fusion.personal_prior import (
    build_personal_prior_payload,
    read_personal_labels,
    write_personal_prior_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", type=Path, help="JSONL label store written by the harvest flag")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output artifact path (pass to --position-prior; must end in .json)",
    )
    parser.add_argument(
        "--merge-population",
        default="guitarset-v1",
        help=(
            "registered position prior supplying counts for pitches without "
            "enough personal labels (default guitarset-v1; 'none' for a pure "
            "personal artifact)"
        ),
    )
    parser.add_argument(
        "--min-labels-per-pitch",
        type=int,
        default=5,
        help=(
            "personalize a pitch only at this many harvested labels "
            "(default 5 — a conservatism knob, never swept)"
        ),
    )
    args = parser.parse_args(argv)

    if not str(args.output).lower().endswith(".json"):
        parser.error("--output must end in .json (the CLI detects personal priors by suffix)")

    labels = read_personal_labels(args.store)
    payload = build_personal_prior_payload(
        labels,
        merge_population=args.merge_population,
        min_labels_per_pitch=args.min_labels_per_pitch,
    )
    write_personal_prior_artifact(args.output, payload)

    personalized = payload["personalized_pitches"]
    assert isinstance(personalized, list)
    print(f"labels read:          {len(labels)}")
    print(f"personalized pitches: {len(personalized)}")
    print(f"population base:      {payload['population_base'] or 'none (pure personal)'}")
    print(f"artifact:             {args.output}")
    print(f"use it with:          tabvision transcribe ... --position-prior {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
