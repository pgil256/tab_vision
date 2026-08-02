"""Read-only helpers for the fixed Guitar-TECHS WAV/JAMS evaluation slice.

The pinned ``ryangowe/guitar-chord-mix`` mirror stores Guitar-TECHS as flat,
same-stem ``.wav``/``.jams`` pairs.  Its dataset card and all 82 cached files
agree on the annotation convention: one ``note_midi`` annotation per string,
with ``annotation_metadata.data_source`` numbering strings from low E (``0``)
to high E (``5``).

The note representation is therefore semantically identical to GuitarSet's
JAMS representation.  This module validates the stricter fixed-slice schema,
then delegates event construction to the existing GuitarSet parser.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tabvision.eval.guitarset_audio import parse_guitarset_jams
from tabvision.types import GuitarConfig, TabEvent

DATASET_SOURCE = "GuitarTECHS"
SPEC_TIER = "clean_electric"
ANNOTATION_FORMAT = "guitar_techs_jams"
GUITAR_TECHS_REVISION = "4448053ced18e67a9f66bfab47ac2de3cc0b4521"
_MANIFEST_PREFIX = "guitar-techs/"

_FILENAME_RE = re.compile(
    r"^(?P<player>P[1-9][0-9]*)_"
    r"(?P<content_type>chords|scales|singlenotes)_"
    r"(?P<content_id>.+)$"
)
_CONTENT_TYPE_LABELS = {
    "chords": "chords",
    "scales": "scales",
    "singlenotes": "single_notes",
}


@dataclass(frozen=True)
class GuitarTechsClip:
    """One immutable, locally paired Guitar-TECHS evaluation clip."""

    clip_id: str
    wav_path: Path
    jams_path: Path
    source: str
    tier: str
    player_id: str
    content_type: str
    content_id: str


def _paired_files(root: Path) -> list[tuple[str, Path, Path]]:
    """Return complete same-stem WAV/JAMS pairs, failing on partial data."""

    if not root.is_dir():
        raise FileNotFoundError(f"Guitar-TECHS root is not a directory: {root}")

    wav_by_stem: dict[str, Path] = {}
    jams_by_stem: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".wav", ".jams"}:
            continue
        destination = wav_by_stem if suffix == ".wav" else jams_by_stem
        if path.stem in destination:
            raise ValueError(f"duplicate Guitar-TECHS {suffix} stem {path.stem!r} under {root}")
        destination[path.stem] = path

    missing_wav = sorted(set(jams_by_stem) - set(wav_by_stem))
    missing_jams = sorted(set(wav_by_stem) - set(jams_by_stem))
    if missing_wav or missing_jams:
        details: list[str] = []
        if missing_wav:
            details.append(f"missing WAV for {', '.join(missing_wav)}")
        if missing_jams:
            details.append(f"missing JAMS for {', '.join(missing_jams)}")
        raise ValueError(f"incomplete Guitar-TECHS pairs under {root}: {'; '.join(details)}")

    return [(stem, wav_by_stem[stem], jams_by_stem[stem]) for stem in sorted(wav_by_stem)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_provenance_paths(dataset_root: Path) -> tuple[Path, Path]:
    manifest_path = dataset_root.parent / "manifest.json"
    mirror_root = dataset_root.parents[1]
    metadata_root = mirror_root / ".cache" / "huggingface" / "download" / "clips" / "guitar-techs"
    return manifest_path, metadata_root


def _manifest_stems(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError(f"Guitar-TECHS manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Guitar-TECHS manifest is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Guitar-TECHS manifest must be a JSON object: {path}")
    stems = {
        key.removeprefix(_MANIFEST_PREFIX)
        for key in payload
        if isinstance(key, str) and key.startswith(_MANIFEST_PREFIX)
    }
    if not stems or any(not stem or "/" in stem for stem in stems):
        raise ValueError(f"Guitar-TECHS manifest has no valid fixed-slice keys: {path}")
    return stems


def _verify_hf_metadata(path: Path, metadata_path: Path) -> None:
    if not metadata_path.is_file():
        raise ValueError(f"Guitar-TECHS metadata is missing: {metadata_path}")
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Guitar-TECHS metadata is unreadable: {metadata_path}") from exc
    if len(lines) < 2:
        raise ValueError(f"Guitar-TECHS metadata is malformed: {metadata_path}")
    revision = lines[0].strip()
    etag = lines[1].strip().lower()
    if revision != GUITAR_TECHS_REVISION:
        raise ValueError(
            f"Guitar-TECHS metadata revision mismatch for {path.name}: "
            f"expected {GUITAR_TECHS_REVISION}, found {revision}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", etag):
        observed = _sha256_file(path)
        algorithm = "SHA-256"
    elif re.fullmatch(r"[0-9a-f]{40}", etag):
        observed = _git_blob_sha1(path)
        algorithm = "Git blob SHA-1"
    else:
        raise ValueError(f"Guitar-TECHS metadata ETag is malformed: {metadata_path}")
    if observed != etag:
        raise ValueError(
            f"Guitar-TECHS {algorithm} mismatch for {path.name}: expected {etag}, found {observed}"
        )


def _verify_fixed_slice(
    pairs: list[tuple[str, Path, Path]],
    *,
    manifest_path: Path,
    metadata_root: Path,
) -> None:
    actual_stems = {stem for stem, _wav_path, _jams_path in pairs}
    expected_stems = _manifest_stems(manifest_path)
    if actual_stems != expected_stems:
        missing = sorted(expected_stems - actual_stems)
        unexpected = sorted(actual_stems - expected_stems)
        raise ValueError(
            f"Guitar-TECHS fixed slice mismatch: missing={missing!r}; unexpected={unexpected!r}"
        )
    for _stem, wav_path, jams_path in pairs:
        for path in (wav_path, jams_path):
            _verify_hf_metadata(path, metadata_root / f"{path.name}.metadata")


def scan_guitar_techs(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    metadata_root: str | Path | None = None,
) -> list[GuitarTechsClip]:
    """Discover deterministic clip records in a flat Guitar-TECHS mirror.

    Filenames must follow the observed
    ``P<player>_<chords|scales|singlenotes>_<content>`` convention.  Unknown
    names and incomplete WAV/JAMS pairs fail loudly so an evaluation cannot
    silently run on a partial or mislabeled corpus.

    The discovered stem set must exactly match the mirror manifest, and every
    file must match its pinned-revision Hugging Face metadata ETag.
    """

    dataset_root = Path(root).expanduser()
    default_manifest, default_metadata = _default_provenance_paths(dataset_root)
    resolved_manifest = Path(manifest_path) if manifest_path is not None else default_manifest
    resolved_metadata = Path(metadata_root) if metadata_root is not None else default_metadata
    pairs = _paired_files(dataset_root)
    clips: list[GuitarTechsClip] = []
    for stem, wav_path, jams_path in pairs:
        match = _FILENAME_RE.fullmatch(stem)
        if match is None:
            raise ValueError(f"malformed Guitar-TECHS clip filename: {stem!r}")
        raw_content_type = match.group("content_type")
        clips.append(
            GuitarTechsClip(
                clip_id=f"guitar-techs/{stem}",
                wav_path=wav_path.resolve(),
                jams_path=jams_path.resolve(),
                source=DATASET_SOURCE,
                tier=SPEC_TIER,
                player_id=match.group("player"),
                content_type=_CONTENT_TYPE_LABELS[raw_content_type],
                content_id=match.group("content_id"),
            )
        )
    _verify_fixed_slice(
        pairs,
        manifest_path=resolved_manifest,
        metadata_root=resolved_metadata,
    )
    return clips


def _number(value: object, *, field: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: {field} must be numeric")
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"{path}: {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path}: {field} must be finite")
    return number


def _validate_note_rows(rows: object, *, path: Path, source: int) -> None:
    if not isinstance(rows, list):
        raise ValueError(f"{path}: note_midi source {source} data must be a list")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: note_midi source {source} row {row_index} must be an object")
        onset_s = _number(
            row.get("time"),
            field=f"source {source} row {row_index} time",
            path=path,
        )
        duration_s = _number(
            row.get("duration"),
            field=f"source {source} row {row_index} duration",
            path=path,
        )
        pitch = _number(
            row.get("value"),
            field=f"source {source} row {row_index} value",
            path=path,
        )
        if onset_s < 0.0:
            raise ValueError(f"{path}: note onset must be non-negative")
        if duration_s < 0.0:
            raise ValueError(f"{path}: note duration must be non-negative")
        if not pitch.is_integer():
            raise ValueError(f"{path}: note_midi value must be integer-valued")


def _validate_jams_schema(path: Path, *, n_strings: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable or invalid JAMS JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: JAMS root must be an object")

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"{path}: JAMS annotations must be a list")
    note_annotations = [
        annotation
        for annotation in annotations
        if isinstance(annotation, Mapping) and annotation.get("namespace") == "note_midi"
    ]
    if len(note_annotations) != n_strings:
        raise ValueError(
            f"{path}: expected {n_strings} note_midi string annotations, "
            f"found {len(note_annotations)}"
        )

    sources: set[int] = set()
    for annotation in note_annotations:
        metadata = annotation.get("annotation_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}: note_midi annotation_metadata must be an object")
        raw_source = metadata.get("data_source")
        if not isinstance(raw_source, str) or not raw_source.isdigit():
            raise ValueError(f"{path}: note_midi data_source must be a string index")
        source = int(raw_source)
        if str(source) != str(raw_source):
            raise ValueError(f"{path}: note_midi data_source must be a canonical string index")
        if source in sources:
            raise ValueError(f"{path}: duplicate note_midi data_source {source}")
        sources.add(source)
        _validate_note_rows(annotation.get("data"), path=path, source=source)

    expected_sources = set(range(n_strings))
    if sources != expected_sources:
        raise ValueError(
            f"{path}: note_midi data_source set must be "
            f"{sorted(expected_sources)}, found {sorted(sources)}"
        )


def parse_guitar_techs_jams(
    jams_path: str | Path,
    cfg: GuitarConfig | None = None,
) -> list[TabEvent]:
    """Parse one validated mirror JAMS file into canonical ``TabEvent`` gold.

    Rows whose pitch is physically outside the supplied guitar configuration
    are excluded by the shared parser.  This matches existing evaluation
    behavior and filters the fixed mirror's handful of sub-frame MIDI
    conversion artifacts without changing valid labels.
    """

    config = cfg or GuitarConfig()
    if config.n_strings != 6 or len(config.tuning_midi) != config.n_strings:
        raise ValueError("Guitar-TECHS JAMS requires a six-string guitar configuration")
    path = Path(jams_path)
    _validate_jams_schema(path, n_strings=config.n_strings)
    return parse_guitarset_jams(path, config)


parse = parse_guitar_techs_jams
scan = scan_guitar_techs


__all__ = [
    "ANNOTATION_FORMAT",
    "DATASET_SOURCE",
    "GUITAR_TECHS_REVISION",
    "SPEC_TIER",
    "GuitarTechsClip",
    "parse",
    "parse_guitar_techs_jams",
    "scan",
    "scan_guitar_techs",
]
