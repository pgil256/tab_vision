"""Tests for the read-only Guitar-TECHS WAV/JAMS evaluation helper."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from tabvision.eval.guitar_techs import (
    DATASET_SOURCE,
    GUITAR_TECHS_REVISION,
    SPEC_TIER,
    GuitarTechsClip,
    parse_guitar_techs_jams,
    scan_guitar_techs,
)


def _annotations(
    notes_by_string: dict[int, list[tuple[float, float, float]]] | None = None,
) -> list[dict[str, object]]:
    notes_by_string = notes_by_string or {}
    return [
        {
            "namespace": "note_midi",
            "annotation_metadata": {"data_source": str(string_idx)},
            "data": [
                {
                    "time": onset_s,
                    "duration": duration_s,
                    "value": pitch,
                    "confidence": None,
                }
                for onset_s, duration_s, pitch in notes_by_string.get(string_idx, [])
            ],
        }
        for string_idx in range(6)
    ]


def _write_jams(
    path: Path,
    *,
    annotations: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "annotations": annotations if annotations is not None else _annotations(),
        "file_metadata": {"duration": 1.0},
        "sandbox": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pair(root: Path, stem: str) -> None:
    (root / f"{stem}.wav").write_bytes(b"RIFF")
    _write_jams(root / f"{stem}.jams")


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload,
        usedforsecurity=False,
    ).hexdigest()


def _write_provenance(root: Path) -> tuple[Path, Path]:
    stems = sorted(path.stem for path in root.glob("*.wav"))
    manifest_path = root / "slice-manifest.json"
    manifest_path.write_text(
        json.dumps({f"guitar-techs/{stem}": "test" for stem in stems}),
        encoding="utf-8",
    )
    metadata_root = root / "metadata"
    metadata_root.mkdir()
    for path in sorted((*root.glob("*.wav"), *root.glob("*.jams"))):
        etag = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix == ".wav"
            else _git_blob_sha1(path)
        )
        (metadata_root / f"{path.name}.metadata").write_text(
            f"{GUITAR_TECHS_REVISION}\n{etag}\n0\n",
            encoding="utf-8",
        )
    return manifest_path, metadata_root


def _scan_with_provenance(root: Path) -> list[GuitarTechsClip]:
    manifest_path, metadata_root = _write_provenance(root)
    return scan_guitar_techs(
        root,
        manifest_path=manifest_path,
        metadata_root=metadata_root,
    )


def test_scan_is_deterministic_and_infers_filename_labels(tmp_path: Path) -> None:
    _write_pair(tmp_path, "P2_scales_Bb")
    _write_pair(tmp_path, "P1_singlenotes_allsinglenotes")
    _write_pair(tmp_path, "P1_chords_Set1_maj")

    clips = _scan_with_provenance(tmp_path)

    assert [clip.clip_id for clip in clips] == [
        "guitar-techs/P1_chords_Set1_maj",
        "guitar-techs/P1_singlenotes_allsinglenotes",
        "guitar-techs/P2_scales_Bb",
    ]
    assert [(clip.player_id, clip.content_type) for clip in clips] == [
        ("P1", "chords"),
        ("P1", "single_notes"),
        ("P2", "scales"),
    ]
    assert all(clip.source == DATASET_SOURCE for clip in clips)
    assert all(clip.tier == SPEC_TIER for clip in clips)
    assert all(clip.wav_path.is_absolute() and clip.jams_path.is_absolute() for clip in clips)


def test_parse_maps_low_e_first_string_and_pitch_to_fret(tmp_path: Path) -> None:
    jams_path = tmp_path / "P1_scales_C.jams"
    _write_jams(
        jams_path,
        annotations=_annotations(
            {
                0: [(0.50, 0.25, 43.0)],
                2: [(0.25, 0.10, 55.0)],
                5: [(0.25, 0.50, 76.0)],
            }
        ),
    )

    events = parse_guitar_techs_jams(jams_path)

    assert [
        (
            event.onset_s,
            event.duration_s,
            event.string_idx,
            event.fret,
            event.pitch_midi,
            event.confidence,
        )
        for event in events
    ] == [
        (0.25, 0.1, 2, 5, 55, 1.0),
        (0.25, 0.5, 5, 12, 76, 1.0),
        (0.5, 0.25, 0, 3, 43, 1.0),
    ]


def test_scan_rejects_missing_pair(tmp_path: Path) -> None:
    (tmp_path / "P1_chords_Set1_maj.wav").write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="missing JAMS"):
        scan_guitar_techs(tmp_path)


def test_scan_rejects_malformed_filename(tmp_path: Path) -> None:
    _write_pair(tmp_path, "unknown_clip")

    with pytest.raises(ValueError, match="malformed Guitar-TECHS clip filename"):
        scan_guitar_techs(tmp_path)


def test_scan_rejects_manifest_stem_mismatch(tmp_path: Path) -> None:
    _write_pair(tmp_path, "P1_scales_C")
    manifest_path, metadata_root = _write_provenance(tmp_path)
    manifest_path.write_text(
        json.dumps({"guitar-techs/P1_scales_D": "test"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed slice mismatch"):
        scan_guitar_techs(
            tmp_path,
            manifest_path=manifest_path,
            metadata_root=metadata_root,
        )


def test_scan_rejects_same_size_file_replacement(tmp_path: Path) -> None:
    _write_pair(tmp_path, "P1_scales_C")
    manifest_path, metadata_root = _write_provenance(tmp_path)
    (tmp_path / "P1_scales_C.wav").write_bytes(b"RUFF")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        scan_guitar_techs(
            tmp_path,
            manifest_path=manifest_path,
            metadata_root=metadata_root,
        )


def test_scan_rejects_wrong_metadata_revision(tmp_path: Path) -> None:
    _write_pair(tmp_path, "P1_scales_C")
    manifest_path, metadata_root = _write_provenance(tmp_path)
    metadata_path = metadata_root / "P1_scales_C.jams.metadata"
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    metadata_path.write_text(
        f"wrong-revision\n{lines[1]}\n0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        scan_guitar_techs(
            tmp_path,
            manifest_path=manifest_path,
            metadata_root=metadata_root,
        )


def test_scan_rejects_git_blob_sha1_mismatch(tmp_path: Path) -> None:
    _write_pair(tmp_path, "P1_scales_C")
    manifest_path, metadata_root = _write_provenance(tmp_path)
    jams_path = tmp_path / "P1_scales_C.jams"
    payload = jams_path.read_bytes()
    jams_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(ValueError, match="Git blob SHA-1 mismatch"):
        scan_guitar_techs(
            tmp_path,
            manifest_path=manifest_path,
            metadata_root=metadata_root,
        )


def test_parse_rejects_duplicate_or_missing_string_annotations(tmp_path: Path) -> None:
    annotations = _annotations()
    annotations[-1]["annotation_metadata"] = {"data_source": "4"}
    jams_path = tmp_path / "P1_scales_C.jams"
    _write_jams(jams_path, annotations=annotations)

    with pytest.raises(ValueError, match="duplicate note_midi data_source 4"):
        parse_guitar_techs_jams(jams_path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"time": -0.1, "duration": 0.1, "value": 40.0}, "onset must be non-negative"),
        ({"time": 0.0, "duration": "bad", "value": 40.0}, "duration must be numeric"),
        ({"time": 0.0, "duration": 0.1, "value": 40.5}, "integer-valued"),
    ],
)
def test_parse_rejects_malformed_note_rows(
    tmp_path: Path,
    row: dict[str, object],
    message: str,
) -> None:
    annotations = _annotations()
    annotations[0]["data"] = [row]
    jams_path = tmp_path / "P1_scales_C.jams"
    _write_jams(jams_path, annotations=annotations)

    with pytest.raises(ValueError, match=message):
        parse_guitar_techs_jams(jams_path)


def test_local_helpers_do_not_open_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", _reject_network)
    _write_pair(tmp_path, "P1_scales_C")

    clips = _scan_with_provenance(tmp_path)
    assert len(clips) == 1
    assert parse_guitar_techs_jams(clips[0].jams_path) == []
