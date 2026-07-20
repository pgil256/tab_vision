"""Build the ``gaps-v1`` position + ``gaps-seq-v1`` transition prior artifacts.

Classical-route analog of ``build_guitarset_v1_prior`` /
``build_guitarset_seq_v1_prior`` (2026-07-20 personal-posture program,
DECISIONS.md). Counts come from the **GAPS train split only** (official
``gaps_metadata_with_splits.csv`` splits; the eval test split never enters),
restricted to standard-tuning scores — the same filter the eval manifest
applies — so the artifact matches the domain the classical route serves.

GAPS is CC-BY-NC-SA-4.0; the derived count tables inherit that license and
are labeled NC-SA in LICENSES.md. Media is never committed — only counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tabvision.eval.manifest_builder import _gaps_is_standard_tuning, _gaps_splits_from_csv
from tabvision.eval.parsers.gaps_musicxml_tab import parse as parse_gaps_gold
from tabvision.fusion.transition_prior import extract_transitions

_PRIORS_DIR = Path(__file__).resolve().parents[2] / "tabvision" / "fusion" / "priors"
DEFAULT_POSITION_OUTPUT = _PRIORS_DIR / "gaps_v1.json"
DEFAULT_SEQUENCE_OUTPUT = _PRIORS_DIR / "gaps_seq_v1.json"

# Mirror the accepted guitarset-v1 / guitarset-seq-v1 hyperparameters — this
# artifact class is deliberately identical; only the corpus differs.
POSITION_ALPHA = 1.0
POSITION_POWER = 2.0
SEQUENCE_SCHEME = "delta_fret"
SEQUENCE_ALPHA = 0.5
SEQUENCE_BACKOFF_KAPPA = 8.0


def _train_stems(root: Path) -> list[str]:
    csv_path = root / "gaps_metadata_with_splits.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"GAPS split CSV not found: {csv_path}")
    splits = _gaps_splits_from_csv(csv_path)
    stems = []
    for xml_path in sorted((root / "musicxml").glob("*.xml")):
        stem = xml_path.stem
        if splits.get(stem) != "train":
            continue
        if not _gaps_is_standard_tuning(xml_path):
            continue
        midi_path = root / "midi" / f"{stem}.mid"
        sync_path = root / "syncpoints" / f"{stem}.json"
        if not (midi_path.is_file() and sync_path.is_file()):
            continue
        stems.append(stem)
    if not stems:
        raise RuntimeError(f"no standard-tuning GAPS train stems found under {root}")
    return stems


def build_payloads(*, root: Path) -> tuple[dict, dict, dict]:
    stems = _train_stems(root)

    position_counts: Counter[tuple[int, int, int]] = Counter()
    transition_counts: Counter[tuple[int, int, int]] = Counter()
    parsed = 0
    skipped: list[str] = []
    for stem in stems:
        try:
            events = parse_gaps_gold(root / "musicxml" / f"{stem}.xml")
        except (OSError, ValueError, RuntimeError, KeyError, SyntaxError) as exc:
            # Corrupt/malformed train scores (e.g. a blank <step>) are skipped
            # and reported; the artifact must never depend on repairing them.
            skipped.append(f"{stem}: {type(exc).__name__}: {exc}")
            continue
        if not events:
            skipped.append(f"{stem}: no aligned gold events")
            continue
        parsed += 1
        for event in events:
            position_counts[(event.pitch_midi, event.string_idx, event.fret)] += 1
        for sample in extract_transitions(events, singleton_only=True):
            transition_counts[sample] += 1

    if not parsed:
        raise RuntimeError("no GAPS train stems produced gold events")

    stems_sha = hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()
    provenance = {
        "train_stems": len(stems),
        "parsed_stems": parsed,
        "skipped_stems": len(skipped),
        "skipped_detail": skipped,
        "stems_sha256": stems_sha,
        "position_rows": len(position_counts),
        "position_events": sum(position_counts.values()),
        "transition_rows": len(transition_counts),
        "transition_samples": sum(transition_counts.values()),
    }

    position_payload = {
        "schema_version": 1,
        "name": "gaps-v1",
        "source": (
            "Pitch-position counts built from the GAPS train split "
            "(standard-tuning scores only; official metadata splits — the "
            "eval test split is excluded). CC-BY-NC-SA-4.0 derived counts."
        ),
        "validation_player": "gaps-test-split",
        "training_tracks": parsed,
        "alpha": POSITION_ALPHA,
        "power": POSITION_POWER,
        "counts": [
            [pitch_midi, string_idx, fret, count]
            for (pitch_midi, string_idx, fret), count in sorted(position_counts.items())
        ],
    }
    sequence_payload = {
        "schema_version": 1,
        "name": "gaps-seq-v1",
        "source": (
            "Anchor-to-anchor transition counts from the GAPS train split "
            "(standard-tuning scores only), singleton-to-singleton cluster "
            "moves to match the decode-side gating. CC-BY-NC-SA-4.0 derived "
            "counts."
        ),
        "validation_player": "gaps-test-split",
        "training_tracks": parsed,
        "scheme": SEQUENCE_SCHEME,
        "alpha": SEQUENCE_ALPHA,
        "backoff_kappa": SEQUENCE_BACKOFF_KAPPA,
        "singleton_only": True,
        "counts": [
            [dp, ds, prev_fret, count]
            for (dp, ds, prev_fret), count in sorted(transition_counts.items())
        ],
    }
    return position_payload, sequence_payload, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaps-root", type=Path, required=True)
    parser.add_argument("--position-output", type=Path, default=DEFAULT_POSITION_OUTPUT)
    parser.add_argument("--sequence-output", type=Path, default=DEFAULT_SEQUENCE_OUTPUT)
    args = parser.parse_args()

    position_payload, sequence_payload, provenance = build_payloads(root=args.gaps_root)
    for path, payload in (
        (args.position_output, position_payload),
        (args.sequence_output, sequence_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"output={path}")
        print(f"sha256={digest}")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
