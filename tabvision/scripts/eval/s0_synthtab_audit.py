"""Program S Phase S0 — SynthTab acquisition audit.

Verifies the downloaded SynthTab archives without extracting them: zip
integrity, SHA-256, member counts by extension, and a JAMS spot-parse that
reports whether per-note ``(string, fret, onset)`` sequences are derivable
(the S1 substrate requirement). Writes a markdown report plus a manifest
JSON. Media/annotations stay under ``TABVISION_DATA_ROOT`` and are never
committed.

Gate (plan S0): >= 50k tracks parse with usable string+fret+onset in
standard-tuning six-string form. This script reports counts and a sampled
parse; the gate decision is recorded in DECISIONS.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

JAMS_ZIP = "all_jams_midi_V2_60000_tracks.zip"
DEV_ZIP = "SynthTab_Dev.zip"
SAMPLE_JAMS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_zip(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        by_ext = Counter(Path(name).suffix.lower() or "<dir>" for name in names)
        crc_ok = archive.testzip() is None
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "members": len(names),
        "by_extension": dict(by_ext.most_common(12)),
        "crc_ok": crc_ok,
    }


def _spot_parse_jams(path: Path) -> list[dict[str, object]]:
    """Load a few JAMS members raw (they are JSON) and profile annotations."""
    samples: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        jams_names = [n for n in archive.namelist() if n.lower().endswith(".jams")]
        stride = max(1, len(jams_names) // SAMPLE_JAMS)
        for name in jams_names[::stride][:SAMPLE_JAMS]:
            raw = json.loads(archive.read(name))
            annotations = raw.get("annotations", [])
            namespaces = Counter(a.get("namespace", "?") for a in annotations)
            # SynthTab stores one `note_tab` annotation per string with
            # sandbox {string_index, open_tuning} and values {fret, velocity}.
            note_tab = [a for a in annotations if a.get("namespace") == "note_tab"]
            strings = [
                {
                    "string_index": a.get("sandbox", {}).get("string_index"),
                    "open_tuning": a.get("sandbox", {}).get("open_tuning"),
                    "notes": len(a.get("data", [])),
                }
                for a in note_tab
            ]
            first_note = None
            for a in note_tab:
                if a.get("data"):
                    observation = a["data"][0]
                    first_note = {
                        "time": observation.get("time"),
                        "duration": observation.get("duration"),
                        "value": observation.get("value"),
                    }
                    break
            samples.append(
                {
                    "member": name,
                    "namespaces": dict(namespaces),
                    "note_tab_annotations": len(note_tab),
                    "strings": strings,
                    "first_note": first_note,
                    "track_sandbox": raw.get("sandbox", {}),
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        dest="directory",
        default=None,
        help="Download dir (default: $TABVISION_DATA_ROOT/datasets/synthtab).",
    )
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--json", dest="json_path", required=True)
    args = parser.parse_args()

    data_root = os.environ.get("TABVISION_DATA_ROOT", "")
    directory = Path(args.directory or (Path(data_root) / "datasets" / "synthtab"))

    payload: dict[str, object] = {"directory": str(directory), "archives": []}
    lines = ["# S0 SynthTab acquisition audit", ""]
    archives: list[dict[str, object]] = []
    for zip_name in (JAMS_ZIP, DEV_ZIP):
        path = directory / zip_name
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        summary = _summarize_zip(path)
        archives.append(summary)
        lines += [
            f"## {zip_name}",
            "",
            f"- bytes: {summary['bytes']:,}",
            f"- SHA-256: `{summary['sha256']}`",
            f"- members: {summary['members']:,} (CRC ok: {summary['crc_ok']})",
            f"- by extension: `{summary['by_extension']}`",
            "",
        ]
    payload["archives"] = archives

    samples = _spot_parse_jams(directory / JAMS_ZIP)
    payload["jams_samples"] = samples
    lines += ["## JAMS spot-parse", ""]
    for sample in samples:
        lines += [
            f"- `{sample['member']}`: namespaces `{sample['namespaces']}`, "
            f"note_tab × {sample['note_tab_annotations']}, strings "
            f"`{sample['strings']}`, first note `{sample['first_note']}`, "
            f"track sandbox `{sample['track_sandbox']}`",
        ]
    lines.append("")

    Path(args.json_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"archives": archives}, indent=2))


if __name__ == "__main__":
    main()
