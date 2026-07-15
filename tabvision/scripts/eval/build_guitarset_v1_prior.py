"""Build the checked-in ``guitarset-v1`` pitch-position prior artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tabvision.eval.guitarset_audio import (
    DEFAULT_DATA_HOME,
    DEFAULT_POSITION_PRIOR_ALPHA,
    DEFAULT_POSITION_PRIOR_POWER,
    DEFAULT_VALIDATION_PLAYER,
    list_guitarset_track_ids,
    parse_guitarset_jams,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors" / "guitarset_v1.json"
)


def build_payload(
    *,
    data_home: Path,
    validation_player: str,
    mode: str = "all",
    name: str = "guitarset-v1",
) -> dict:
    counts: Counter[tuple[int, int, int]] = Counter()
    track_ids = list_guitarset_track_ids(
        data_home,
        split="train",
        validation_player=validation_player,
    )
    if not track_ids:
        raise RuntimeError(f"no GuitarSet train tracks found under {data_home}")
    if mode != "all":
        track_ids = [track_id for track_id in track_ids if track_id.endswith(f"_{mode}")]
        if not track_ids:
            raise RuntimeError(f"no GuitarSet {mode} train tracks found under {data_home}")

    for track_id in track_ids:
        jams_path = data_home / "annotation" / f"{track_id}.jams"
        for event in parse_guitarset_jams(jams_path):
            counts[(event.pitch_midi, event.string_idx, event.fret)] += 1

    rows = [
        [pitch_midi, string_idx, fret, count]
        for (pitch_midi, string_idx, fret), count in sorted(
            counts.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2]),
        )
    ]
    return {
        "schema_version": 1,
        "name": name,
        "source": (
            "Pitch-position counts built from the GuitarSet train split. "
            "Validation player is excluded so runtime loading does not require raw GuitarSet files."
        ),
        "validation_player": validation_player,
        "training_tracks": len(track_ids),
        "alpha": DEFAULT_POSITION_PRIOR_ALPHA,
        "power": DEFAULT_POSITION_PRIOR_POWER,
        "counts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--validation-player", default=DEFAULT_VALIDATION_PLAYER)
    parser.add_argument("--mode", choices=("all", "solo", "comp"), default="all")
    parser.add_argument("--name", default="guitarset-v1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(
        data_home=args.data_home,
        validation_player=args.validation_player,
        mode=args.mode,
        name=args.name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(f"tracks={payload['training_tracks']}")
    print(f"rows={len(payload['counts'])}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
