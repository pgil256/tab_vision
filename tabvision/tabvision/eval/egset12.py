"""EGSet12 annotation parsing and deterministic dataset discovery.

EGSet12 is a twelve-track clean-electric evaluation corpus published as
Zenodo record 11406378.  The record is intentionally consumed from an
external data root; audio and annotations are never vendored in this
repository.

The paper authors' reference loader reads one ``note_midi`` annotation per
guitar string and uses ``annotation_metadata.data_source`` as a zero-based
low-E-to-high-E string index.  ``parse_egset12_jams`` implements that observed
form without depending on the optional :mod:`jams` package.  It also accepts
the ``note_tab`` form used by closely related generated corpora when a file
provides explicit string/tuning metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabvision.eval.manifest_builder import ClipEntry
from tabvision.types import DEFAULT_TUNING_MIDI, GuitarConfig, TabEvent

ZENODO_RECORD_ID = "11406378"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ANNOTATION_FORMAT = "egset12_jams"
TRACK_IDS: tuple[str, ...] = tuple(f"{index:02d}" for index in range(1, 13))


@dataclass(frozen=True)
class PublishedFile:
    """One immutable file entry from the published Zenodo record."""

    name: str
    size_bytes: int
    md5: str


# Exact byte sizes and MD5 digests reported by the Zenodo record API.
# Scope is deliberately limited to the 12 WAV + 12 JAMS evaluation files:
# the Guitar Pro sources and published model checkpoint are separate artifacts.
PUBLISHED_FILES: tuple[PublishedFile, ...] = (
    PublishedFile("01.jams", 52_172, "083c7dae8e6556c20b9a2d762e2c977f"),
    PublishedFile("01.wav", 8_640_044, "2eb739c5fb73e6327bb47267afe3eddf"),
    PublishedFile("02.jams", 120_087, "848f984b17b261a65585e25fba977a33"),
    PublishedFile("02.wav", 8_886_896, "69b8701ea9a81428a6346e0d3d4b9b85"),
    PublishedFile("03.jams", 45_984, "721ec50f570892f9cfa88fb1e22a6113"),
    PublishedFile("03.wav", 10_368_044, "28141f17e46399553c52f5ed27bc10e2"),
    PublishedFile("04.jams", 86_938, "87426719ac4353d73e1af09970c31eb1"),
    PublishedFile("04.wav", 8_797_130, "6fe2f6f915953e8ae28b8a84a7677d0f"),
    PublishedFile("05.jams", 56_724, "c5c2fd376031177e87a3eb4ad12d220c"),
    PublishedFile("05.wav", 9_600_044, "3435348c2b6702524dade471be70e4eb"),
    PublishedFile("06.jams", 186_045, "5ababdcf7741400dc93768334f6c899d"),
    PublishedFile("06.wav", 8_723_924, "9f7ead382f373259b466ccd1884ed173"),
    PublishedFile("07.jams", 38_330, "e693844f4b46fd3831c7c4ee0a2c3aa8"),
    PublishedFile("07.wav", 8_755_244, "77f752ab3e7a5c606a21ac7b0df4fa1c"),
    PublishedFile("08.jams", 48_855, "513e00c522d53adac0ed9966a5b4c8cd"),
    PublishedFile("08.wav", 9_169_016, "a59f373c00b8a327b37ce28f6601404c"),
    PublishedFile("09.jams", 49_728, "9f08cae003c6c3d9dc745c2e319496d4"),
    PublishedFile("09.wav", 8_730_980, "593aec1394a905a0c8b255a847f54139"),
    PublishedFile("10.jams", 42_653, "8cbf70e1b086f4a8fe5cac79572635ae"),
    PublishedFile("10.wav", 8_730_980, "123818ef1020102252192d9e7a231e07"),
    PublishedFile("11.jams", 57_003, "f392b5bba5f3b99866bba91cb4d35a9a"),
    PublishedFile("11.wav", 9_032_774, "4bbaecaaa3e58bef6bb15a6cd0979fe2"),
    PublishedFile("12.jams", 47_846, "21217bda094eb8f29edfd1ed2f23ba45"),
    PublishedFile("12.wav", 6_702_584, "e1ee73508f37d5c28c69877a588665d2"),
)

PUBLISHED_FILE_BY_NAME: Mapping[str, PublishedFile] = {
    published.name: published for published in PUBLISHED_FILES
}


def default_egset12_root() -> Path:
    """Return ``$TABVISION_DATA_ROOT/egset12`` with the project default fallback."""

    data_root = Path(
        os.environ.get("TABVISION_DATA_ROOT", Path.home() / ".tabvision" / "data")
    ).expanduser()
    return data_root / "egset12"


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)  # accepts JSON numbers and numeric strings
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    integer = int(number)
    return integer if number == integer else None


def _declared_open_pitch(annotation: Mapping[str, Any]) -> int | None:
    metadata = _as_mapping(annotation.get("annotation_metadata"))
    sandbox = _as_mapping(annotation.get("sandbox"))
    for container in (sandbox, metadata):
        for field in ("open_tuning", "open_pitch", "tuning"):
            value = _as_int(container.get(field))
            if value is not None:
                return value
    return None


def _string_index(
    annotation: Mapping[str, Any],
    *,
    namespace: str,
    cfg: GuitarConfig,
) -> int:
    """Resolve a source string label to TabVision's low-E-first index."""

    declared_open = _declared_open_pitch(annotation)
    if declared_open is not None:
        try:
            return cfg.tuning_midi.index(declared_open)
        except ValueError as exc:
            raise ValueError(
                f"EGSet12 annotation declares non-standard open pitch {declared_open}"
            ) from exc

    metadata = _as_mapping(annotation.get("annotation_metadata"))
    sandbox = _as_mapping(annotation.get("sandbox"))
    raw = metadata.get("data_source")
    if raw is None:
        raw = sandbox.get("string_index")
    index = _as_int(raw)
    if index is None:
        raise ValueError("EGSet12 string annotation has no usable string index")

    # Observed EGSet12/GuitarSet note_midi form: 0=low E .. 5=high E.
    if namespace == "note_midi" and 0 <= index < cfg.n_strings:
        return index

    # Related note_tab corpora use 1=high E .. 6=low E.  A zero-based
    # note_tab label remains accepted as low-E-first for interoperability.
    if namespace == "note_tab":
        if index == 0:
            return 0
        if 1 <= index <= cfg.n_strings:
            return cfg.n_strings - index

    raise ValueError(f"EGSet12 annotation has unsupported string index {index!r}")


def _row_confidence(row: Mapping[str, Any]) -> float:
    confidence = _as_float(row.get("confidence"))
    if confidence is None:
        return 1.0
    return min(1.0, max(0.0, confidence))


def _parse_note_midi_row(
    row: Mapping[str, Any],
    *,
    string_idx: int,
    cfg: GuitarConfig,
) -> TabEvent | None:
    value = row.get("value")
    if isinstance(value, Mapping):
        value = value.get("midi", value.get("pitch"))
    pitch = _as_float(value)
    if pitch is None:
        return None
    try:
        onset_s = float(row["time"])
        duration_s = float(row.get("duration", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    pitch_midi = int(round(pitch))

    fret = pitch_midi - cfg.tuning_midi[string_idx]
    if fret < cfg.capo or fret > cfg.max_fret:
        return None
    return TabEvent(
        onset_s=onset_s,
        duration_s=max(0.0, duration_s),
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch_midi,
        confidence=_row_confidence(row),
    )


def _parse_note_tab_row(
    row: Mapping[str, Any],
    *,
    string_idx: int,
    cfg: GuitarConfig,
) -> TabEvent | None:
    value = row.get("value")
    value_mapping = _as_mapping(value)
    fret_value = value_mapping.get("fret") if value_mapping else value
    fret = _as_int(fret_value)
    try:
        onset_s = float(row["time"])
        duration_s = float(row.get("duration", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if fret is None or fret < cfg.capo or fret > cfg.max_fret:
        return None

    pitch_midi = cfg.tuning_midi[string_idx] + fret
    declared_pitch = value_mapping.get("midi", value_mapping.get("pitch"))
    if declared_pitch is not None:
        parsed_pitch = _as_int(declared_pitch)
        if parsed_pitch is None or parsed_pitch != pitch_midi:
            raise ValueError(
                "EGSet12 note_tab pitch is inconsistent with its standard-tuning "
                f"string/fret position ({string_idx=}, {fret=})"
            )
    return TabEvent(
        onset_s=onset_s,
        duration_s=max(0.0, duration_s),
        string_idx=string_idx,
        fret=fret,
        pitch_midi=pitch_midi,
        confidence=_row_confidence(row),
    )


def parse_egset12_jams(
    jams_path: str | Path,
    cfg: GuitarConfig | None = None,
) -> list[TabEvent]:
    """Parse an EGSet12 JAMS file into canonical :class:`TabEvent` objects.

    ``note_midi`` is preferred when both supported namespaces are present so
    duplicated note annotations cannot inflate the gold set.  Files without a
    supported string-index annotation fail loudly instead of yielding a
    deceptively empty reference.
    """

    if cfg is None:
        cfg = GuitarConfig()
    if cfg.n_strings != 6 or tuple(cfg.tuning_midi) != DEFAULT_TUNING_MIDI:
        raise ValueError("EGSet12 parser supports six-string standard tuning only")

    payload = json.loads(Path(jams_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("EGSet12 JAMS root must be a JSON object")
    raw_annotations = payload.get("annotations")
    if not isinstance(raw_annotations, list):
        raise ValueError("EGSet12 JAMS annotations must be a list")

    annotations = [annotation for annotation in raw_annotations if isinstance(annotation, Mapping)]
    note_midi = [
        annotation for annotation in annotations if annotation.get("namespace") == "note_midi"
    ]
    note_tab = [
        annotation for annotation in annotations if annotation.get("namespace") == "note_tab"
    ]
    if note_midi:
        namespace = "note_midi"
        selected = note_midi
    elif note_tab:
        namespace = "note_tab"
        selected = note_tab
    else:
        raise ValueError("EGSet12 JAMS contains no note_midi or note_tab string annotations")

    events: list[TabEvent] = []
    for annotation in selected:
        rows = annotation.get("data")
        if not isinstance(rows, list):
            continue
        string_idx = _string_index(annotation, namespace=namespace, cfg=cfg)
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            if namespace == "note_midi":
                event = _parse_note_midi_row(raw_row, string_idx=string_idx, cfg=cfg)
            else:
                event = _parse_note_tab_row(raw_row, string_idx=string_idx, cfg=cfg)
            if event is not None:
                events.append(event)

    events.sort(key=lambda event: (event.onset_s, event.string_idx, event.fret))
    return events


# Uniform parser-interface alias used by experiment code.
parse = parse_egset12_jams


def _candidate_path(root: Path, track_id: str, suffix: str) -> Path | None:
    """Find a published file in flat or authors' ``audios``/``jams`` layout."""

    name = f"{track_id}{suffix}"
    candidates = [root / name]
    if suffix == ".wav":
        candidates.append(root / "audios" / name)
    else:
        candidates.extend((root / "jams" / name, root / "jams_corrected" / name))
    return next((path for path in candidates if path.is_file()), None)


def _md5_file(path: Path) -> str:
    """Return the publisher-declared MD5 identity for one local file."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_egset12(root: str | Path | None = None) -> list[ClipEntry]:
    """Return all twelve EGSet12 manifest rows, or ``[]`` for an incomplete root.

    Discovery is intentionally all-or-nothing.  Every WAV/JAMS pair must exist
    and match both its published byte size and MD5 identity.  Discovery repeats
    the acquisition check so same-size replacement files cannot enter scoring.
    """

    dataset_root = Path(root) if root is not None else default_egset12_root()
    if not dataset_root.is_dir():
        return []

    pairs: list[tuple[str, Path, Path]] = []
    for track_id in TRACK_IDS:
        wav_path = _candidate_path(dataset_root, track_id, ".wav")
        jams_path = _candidate_path(dataset_root, track_id, ".jams")
        if wav_path is None or jams_path is None:
            return []
        wav_spec = PUBLISHED_FILE_BY_NAME[f"{track_id}.wav"]
        jams_spec = PUBLISHED_FILE_BY_NAME[f"{track_id}.jams"]
        if wav_path.stat().st_size != wav_spec.size_bytes:
            return []
        if jams_path.stat().st_size != jams_spec.size_bytes:
            return []
        if _md5_file(wav_path) != wav_spec.md5:
            return []
        if _md5_file(jams_path) != jams_spec.md5:
            return []
        pairs.append((track_id, wav_path, jams_path))

    return [
        ClipEntry(
            id=f"egset12/{track_id}",
            tier="clean_electric",
            source="EGSet12",
            split="test",
            media_path=str(wav_path.resolve()),
            annotation_path=str(jams_path.resolve()),
            annotation_format=ANNOTATION_FORMAT,
        )
        for track_id, wav_path, jams_path in pairs
    ]


__all__ = [
    "ANNOTATION_FORMAT",
    "PUBLISHED_FILES",
    "PUBLISHED_FILE_BY_NAME",
    "PublishedFile",
    "TRACK_IDS",
    "ZENODO_RECORD_ID",
    "ZENODO_RECORD_URL",
    "default_egset12_root",
    "parse",
    "parse_egset12_jams",
    "scan_egset12",
]
