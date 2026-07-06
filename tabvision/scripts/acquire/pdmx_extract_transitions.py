"""PDMX transition extraction (roadmap A15, PDMX acquisition step 3).

Streams ``mxl.tar.gz`` once (same CSV filter as the yield scan,
``scripts.acquire.pdmx_tab_scan``), walks each TAB-bearing score with the
GAPS MusicXML tab walk, and extracts anchor-to-anchor fingering
transitions — the ``(Δpitch, Δstring, prev-anchor fret)`` samples that
feed a ``transition_prior`` count artifact. The aggregated counts are
cached in the local data root; nothing from the dataset itself is
committed (CC-BY dataset, PD/CC0 scores — same conditions as the scan).

Clustering semantics differ from GuitarSet *by construction*: PDMX scores
have only score time (divisions), no absolute seconds, so the 80 ms
``CLUSTER_GAP_S`` rule cannot apply literally. Instead notes are clustered
at ``cluster_gap_s=0.0``: same-onset notes (``<chord>`` marks share their
head note's onset) cluster exactly, and **any** positive score-time gap is
a transition. That is cleaner than audio-time clustering — no risk of
merging fast runs — but it means PDMX "singleton moves" are defined
slightly differently than GuitarSet's: two written notes 40 ms apart in a
performance would be one GuitarSet cluster and a PDMX transition.

Per-score filters (counts reported, decisions banked in DECISIONS.md):

- scores whose declared ``<staff-tuning>`` is not standard EADGBE are
  skipped entirely — the decode assumes standard tuning and scordatura
  changes the Δpitch↔Δstring geometry the prior learns;
- notes failing pitch consistency (``pitch != open_string + fret``) or
  outside the standard-config bounds (string 1..6, fret 0..24) are
  dropped before clustering (the scan sampled one score at 0.942
  consistency — bad exporter frets must not poison the counts);
- when a score has several TAB parts, the first one is used (same as the
  scan's validation path).

CLI (from the ``tabvision/`` package dir)::

    python -m scripts.acquire.pdmx_extract_transitions \\
        --data-root ~/.tabvision/data [--limit 20] [--out transitions.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.acquire.pdmx_tab_scan import (
    STANDARD_TUNING,
    is_guitar_row,
    normalize_member,
    score_xml_bytes,
)
from tabvision.eval.parsers.gaps_musicxml_tab import _staff_tuning, _tab_part, _walk_tab_notes
from tabvision.fusion.transition_prior import extract_transitions
from tabvision.types import GuitarConfig, TabEvent

SCHEMA_VERSION = 1


def score_transitions(xml_bytes: bytes, cfg: GuitarConfig) -> dict:
    """Extract singleton-move transition samples from one MXL score.

    Returns per-score stats plus the ``(Δpitch, Δstring, prev-fret)``
    samples; ``skipped_nonstandard`` short-circuits with no samples.
    """
    part = _tab_part(ET.fromstring(xml_bytes))
    declared = _staff_tuning(part)
    if declared and declared != STANDARD_TUNING:
        return {"skipped_nonstandard": True, "samples": [], "n_notes": 0, "n_dropped": 0}
    tuning = declared or STANDARD_TUNING

    notes, _measure_starts, _total = _walk_tab_notes(part)
    events: list[TabEvent] = []
    dropped = 0
    for n in notes:
        string_idx = 6 - n.mxml_string
        if (
            n.mxml_string not in tuning
            or n.pitch != tuning[n.mxml_string] + n.fret
            or not (0 <= string_idx < cfg.n_strings)
            or not (cfg.capo <= n.fret <= cfg.max_fret)
        ):
            dropped += 1
            continue
        events.append(
            TabEvent(
                onset_s=float(n.score_onset),  # score divisions, not seconds (see docstring)
                duration_s=0.0,
                string_idx=string_idx,
                fret=n.fret,
                pitch_midi=n.pitch,
                confidence=1.0,
            )
        )
    samples = extract_transitions(events, cluster_gap_s=0.0, singleton_only=True)
    return {
        "skipped_nonstandard": False,
        "samples": samples,
        "n_notes": len(notes),
        "n_dropped": dropped,
    }


def extract(data_root: Path, limit: int | None) -> dict:
    pdmx = data_root / "datasets" / "pdmx"
    csv_path = pdmx / "PDMX.csv"
    archive = pdmx / "mxl.tar.gz"

    wanted: set[str] = set()
    csv.field_size_limit(1 << 24)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        wanted = {normalize_member(row["mxl"]) for row in csv.DictReader(fh) if is_guitar_row(row)}
    print(f"CSV filter: {len(wanted)} guitar x clean x MXL members wanted", flush=True)

    cfg = GuitarConfig()
    counts: Counter[tuple[int, int, int]] = Counter()
    stats = Counter(
        scanned=0,
        tab_bearing=0,
        used=0,
        skipped_nonstandard=0,
        parse_errors=0,
        unreadable=0,
        notes_total=0,
        notes_dropped=0,
    )
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = normalize_member(member.name)
            if name not in wanted or not member.isfile():
                continue
            stats["scanned"] += 1
            fh = tar.extractfile(member)
            xml_bytes = score_xml_bytes(fh.read()) if fh is not None else None
            if xml_bytes is None:
                stats["unreadable"] += 1
                continue
            if b"<fret>" not in xml_bytes or b"<string>" not in xml_bytes:
                continue
            stats["tab_bearing"] += 1
            try:
                result = score_transitions(xml_bytes, cfg)
            except (ET.ParseError, ValueError):
                stats["parse_errors"] += 1
                continue
            if result["skipped_nonstandard"]:
                stats["skipped_nonstandard"] += 1
                continue
            stats["used"] += 1
            stats["notes_total"] += result["n_notes"]
            stats["notes_dropped"] += result["n_dropped"]
            for sample in result["samples"]:
                counts[sample] += 1
            if stats["scanned"] % 500 == 0:
                print(
                    f"  ...{stats['scanned']}/{len(wanted)} scanned, "
                    f"{stats['used']} used, {sum(counts.values())} transitions",
                    flush=True,
                )
            if limit is not None and stats["tab_bearing"] >= limit:
                break

    return {
        "schema_version": SCHEMA_VERSION,
        "source": (
            "PDMX (Long et al., ICASSP 2025; DOI 10.5281/zenodo.15571083) — "
            "anchor-to-anchor transition counts from TAB-staff scores passing the "
            "guitar x no_license_conflict x MXL CSV filter; standard-tuning scores "
            "only; pitch-inconsistent notes dropped; score-time clustering "
            "(exact-onset chords, any positive gap is a transition), "
            "singleton-to-singleton moves only."
        ),
        "generated_by": "scripts/acquire/pdmx_extract_transitions.py",
        "stats": dict(stats),
        "n_transitions": sum(counts.values()),
        "counts": [[dp, ds, prev_fret, n] for (dp, ds, prev_fret), n in sorted(counts.items())],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("TABVISION_DATA_ROOT"),
        help="dataset root (default: $TABVISION_DATA_ROOT)",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N TAB-bearing scores")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="cache path (default: <data-root>/datasets/pdmx/transitions_v1.json)",
    )
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("--data-root or TABVISION_DATA_ROOT required")

    data_root = Path(args.data_root).expanduser()
    out = args.out or data_root / "datasets" / "pdmx" / "transitions_v1.json"
    payload = extract(data_root, args.limit)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "counts"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
