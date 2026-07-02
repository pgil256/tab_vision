"""PDMX MXL TAB-staff yield scan (roadmap A15, PDMX acquisition step 2).

The PDMX metadata CSV resolves the *guitar* count (3,435 songs with a MIDI
guitar program in ``tracks`` x ``no_license_conflict`` x has-MXL — see
``docs/2026-07-02-pdmx-license-yield-review.md``), but TAB-staff presence is
not in the metadata: only the MXL score itself shows whether MuseScore's
3.6.2 export carries ``<technical><string>/<fret>``. This script streams the
``mxl.tar.gz`` archive once, opens only the CSV-filtered guitar members, and
counts the TAB-bearing subset — the number that decides whether a PDMX
n-gram corpus is real.

License conditions honoured (CC-BY dataset, PD/CC0 scores): the archive and
scores stay in the local data root; nothing from the dataset is committed —
this script's outputs are derived count statistics only. Validation reuses
the GAPS MusicXML tab walk (``eval.parsers.gaps_musicxml_tab``) — the same
code path a later extraction would use — and checks per-note pitch
consistency: ``pitch == open_string_midi + fret``.

CLI (from the ``tabvision/`` package dir)::

    python -m scripts.acquire.pdmx_tab_scan \\
        --data-root ~/.tabvision/data [--sample 10] [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from tabvision.eval.parsers.gaps_musicxml_tab import _staff_tuning, _tab_part, _walk_tab_notes

GUITAR_PROGRAMS = frozenset(range(24, 32))

STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
"""MusicXML string number (1 = high E) -> open-string MIDI, standard tuning."""


def is_guitar_row(row: dict[str, str]) -> bool:
    """CSV row filter: guitar program x no_license_conflict x has-MXL.

    Mirrors the metadata-only yield scan banked in the 2026-07-02 review doc
    (3,435 rows pass on the v3.01 CSV).
    """
    if row.get("subset:no_license_conflict") != "True":
        return False
    if row.get("mxl") in (None, "", "NA"):
        return False
    tracks = row.get("tracks", "")
    if tracks in ("", "NA"):
        return False
    for tok in tracks.split("-"):
        try:
            if int(tok) in GUITAR_PROGRAMS:
                return True
        except ValueError:
            continue
    return False


def normalize_member(path: str) -> str:
    """Normalize a CSV/tar path (``./mxl/...`` or ``mxl/...``) for matching."""
    return path.removeprefix("./").replace("\\", "/")


def score_xml_bytes(mxl_bytes: bytes) -> bytes | None:
    """The score XML inside an MXL (compressed MusicXML) container.

    Follows ``META-INF/container.xml``'s rootfile when present; falls back
    to the first non-META-INF ``.xml``/``.musicxml`` entry.
    """
    with zipfile.ZipFile(io.BytesIO(mxl_bytes)) as zf:
        names = zf.namelist()
        rootfile: str | None = None
        if "META-INF/container.xml" in names:
            try:
                container = ET.fromstring(zf.read("META-INF/container.xml"))
                el = container.find(".//rootfile")
                if el is not None:
                    rootfile = el.get("full-path")
            except ET.ParseError:
                rootfile = None
        if rootfile is None or rootfile not in names:
            candidates = [
                n
                for n in names
                if not n.startswith("META-INF/") and n.lower().endswith((".xml", ".musicxml"))
            ]
            if not candidates:
                return None
            rootfile = candidates[0]
        return zf.read(rootfile)


def validate_tab_walk(xml_bytes: bytes) -> dict:
    """Walk the TAB part with the GAPS parser core; report note-level sanity.

    Returns counts plus the fraction of notes whose pitch equals
    ``open_string + fret`` under the score's staff-tuning (standard when the
    score omits it) — the MuseScore-exporter behaviour a later n-gram
    extraction depends on.
    """
    part = _tab_part(ET.fromstring(xml_bytes))
    notes, _measure_starts, _total = _walk_tab_notes(part)
    tuning = _staff_tuning(part) or STANDARD_TUNING
    consistent = sum(
        1 for n in notes if n.mxml_string in tuning and n.pitch == tuning[n.mxml_string] + n.fret
    )
    return {
        "n_tab_notes": len(notes),
        "n_pitch_consistent": consistent,
        "consistency": (consistent / len(notes)) if notes else 0.0,
        "nonstandard_tuning": bool(_staff_tuning(part)),
    }


def scan(data_root: Path, sample_n: int) -> dict:
    pdmx = data_root / "datasets" / "pdmx"
    csv_path = pdmx / "PDMX.csv"
    archive = pdmx / "mxl.tar.gz"

    wanted: dict[str, str] = {}  # member name -> genres
    csv.field_size_limit(1 << 24)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if is_guitar_row(row):
                wanted[normalize_member(row["mxl"])] = row.get("genres", "NA")
    print(f"CSV filter: {len(wanted)} guitar x clean x MXL members wanted", flush=True)

    seen = 0
    tab_bearing = 0
    unreadable = 0
    tab_genres: Counter[str] = Counter()
    samples: list[dict] = []
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = normalize_member(member.name)
            genres = wanted.get(name)
            if genres is None or not member.isfile():
                continue
            seen += 1
            fh = tar.extractfile(member)
            if fh is None:
                unreadable += 1
                continue
            xml_bytes = score_xml_bytes(fh.read())
            if xml_bytes is None:
                unreadable += 1
                continue
            if b"<fret>" not in xml_bytes or b"<string>" not in xml_bytes:
                continue
            tab_bearing += 1
            tab_genres[genres] += 1
            if len(samples) < sample_n:
                try:
                    result = validate_tab_walk(xml_bytes)
                except (ET.ParseError, ValueError) as exc:
                    result = {"error": str(exc)}
                samples.append({"member": name, **result})
            if seen % 500 == 0:
                print(f"  ...{seen}/{len(wanted)} scanned, {tab_bearing} TAB-bearing", flush=True)

    return {
        "wanted": len(wanted),
        "found_in_archive": seen,
        "unreadable": unreadable,
        "tab_bearing": tab_bearing,
        "tab_fraction_of_guitar": (tab_bearing / seen) if seen else 0.0,
        "tab_genres_top": tab_genres.most_common(12),
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("TABVISION_DATA_ROOT"),
        help="dataset root (default: $TABVISION_DATA_ROOT)",
    )
    parser.add_argument("--sample", type=int, default=10, help="validation sample size")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON summary here")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("--data-root or TABVISION_DATA_ROOT required")

    summary = scan(Path(args.data_root).expanduser(), args.sample)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
