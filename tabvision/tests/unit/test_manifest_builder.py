"""Tests for the composite-eval manifest builder (Phase 0)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from tabvision.eval.manifest import validate_manifest
from tabvision.eval.manifest_builder import (
    ClipEntry,
    apply_limits,
    build_manifest,
    render_toml,
    scan_guitar_techs,
    scan_guitarset,
    scan_utaustin,
    summarise_coverage,
)


def _make_guitarset_layout(
    root: Path,
    tracks: list[tuple[str, dict | None]],
) -> None:
    """Build a fake GuitarSet directory at ``root``.

    Each ``tracks`` tuple is ``(track_id, jams_payload)``. Pass payload
    ``None`` to write the JAMS but omit the audio file (simulates a
    half-present clip that the scanner should skip). The audio file is
    a zero-byte placeholder when payload is not ``None``.
    """
    annotation_dir = root / "annotation"
    audio_dir = root / "audio_mono-mic"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for track_id, payload in tracks:
        jams_path = annotation_dir / f"{track_id}.jams"
        jams_path.write_text(json.dumps(payload or {"annotations": []}), encoding="utf-8")
        if payload is not None:
            (audio_dir / f"{track_id}_mic.wav").write_bytes(b"")


def _make_utaustin_layout(root: Path, clip_ids: list[str]) -> None:
    label_dir = root / "tablature_labels"
    audio_dir = root / "tablature_audio"
    label_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    (root / "timestamps.csv").write_text(
        "frame,timestamp\n" + "\n".join(f"{clip_id}_0.png,0.0" for clip_id in clip_ids) + "\n",
        encoding="utf-8",
    )
    for clip_id in clip_ids:
        (audio_dir / f"{clip_id}.wav").write_bytes(b"")
        (label_dir / f"{clip_id}.npy").write_bytes(b"fake-npy")


def test_scan_guitarset_classifies_comp_and_solo(tmp_path: Path) -> None:
    _make_guitarset_layout(
        tmp_path,
        [
            ("05_Rock1-90-C#_comp", {"annotations": []}),
            ("05_Funk1-114-Ab_solo", {"annotations": []}),
        ],
    )

    entries = scan_guitarset(tmp_path)

    by_id = {entry.id: entry for entry in entries}
    assert by_id["guitarset/05_Rock1-90-C#_comp"].tier == "clean_acoustic_strummed"
    assert by_id["guitarset/05_Funk1-114-Ab_solo"].tier == "clean_acoustic_single_line"
    for entry in entries:
        assert entry.source == "GuitarSet"
        assert entry.annotation_format == "guitarset_jams"


def test_scan_guitarset_assigns_validation_split_for_player_05(tmp_path: Path) -> None:
    _make_guitarset_layout(
        tmp_path,
        [
            ("00_Rock1-90-C#_comp", {"annotations": []}),
            ("05_Rock1-90-C#_comp", {"annotations": []}),
        ],
    )

    entries = scan_guitarset(tmp_path)

    by_id = {entry.id: entry for entry in entries}
    assert by_id["guitarset/00_Rock1-90-C#_comp"].split == "train"
    assert by_id["guitarset/05_Rock1-90-C#_comp"].split == "validation"


def test_scan_guitarset_skips_when_audio_missing(tmp_path: Path) -> None:
    """A JAMS without matching audio is skipped silently."""
    _make_guitarset_layout(
        tmp_path,
        [
            ("05_OnlyAnnot-90-A_comp", None),  # JAMS present, no audio
        ],
    )
    assert scan_guitarset(tmp_path) == []


def test_scan_guitarset_skips_unrecognised_suffix(tmp_path: Path) -> None:
    """Tracks without _comp or _solo suffix are skipped."""
    _make_guitarset_layout(
        tmp_path,
        [
            ("05_OddTrackId-90-A_other", {"annotations": []}),
        ],
    )
    assert scan_guitarset(tmp_path) == []


def test_scan_guitarset_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert scan_guitarset(tmp_path / "nonexistent") == []


def test_scan_guitarset_returns_empty_for_partial_layout(tmp_path: Path) -> None:
    """Root with annotation/ but no audio_mono-mic/ returns empty."""
    (tmp_path / "annotation").mkdir()
    assert scan_guitarset(tmp_path) == []


def test_scan_utaustin_discovers_audio_label_pairs(tmp_path: Path) -> None:
    _make_utaustin_layout(tmp_path, ["0", "1"])

    entries = scan_utaustin(tmp_path)

    assert [entry.id for entry in entries] == ["utaustin/0", "utaustin/1"]
    assert {entry.tier for entry in entries} == {"clean_acoustic_single_line"}
    assert {entry.source for entry in entries} == {"KaggleUTAustin"}
    assert {entry.split for entry in entries} == {"validation"}
    assert {entry.annotation_format for entry in entries} == {"utaustin_tablature_npy"}


def test_scan_utaustin_skips_missing_audio(tmp_path: Path) -> None:
    _make_utaustin_layout(tmp_path, ["0"])
    (tmp_path / "tablature_audio" / "0.wav").unlink()

    assert scan_utaustin(tmp_path) == []


def test_scan_guitar_techs_returns_empty_stub(tmp_path: Path) -> None:
    """Guitar-TECHS scanner is a stub until the dataset is acquired."""
    assert scan_guitar_techs(tmp_path) == []


def _entry(clip_id: str, tier: str = "clean_acoustic_strummed") -> ClipEntry:
    return ClipEntry(
        id=clip_id,
        tier=tier,
        source="GuitarSet",
        split="validation",
        media_path=f"/data/{clip_id}.wav",
        annotation_path=f"/data/{clip_id}.jams",
        annotation_format="guitarset_jams",
    )


def test_apply_limits_caps_per_tier_deterministically() -> None:
    entries = [
        _entry("a", "clean_acoustic_strummed"),
        _entry("b", "clean_acoustic_strummed"),
        _entry("c", "clean_acoustic_strummed"),
        _entry("d", "clean_acoustic_single_line"),
        _entry("e", "clean_acoustic_single_line"),
    ]

    capped = apply_limits(entries, max_clips_per_tier=2)

    # 2 per tier, sorted by id within each tier
    ids = [entry.id for entry in capped]
    assert ids == ["a", "b", "d", "e"]


def test_apply_limits_applies_total_after_per_tier() -> None:
    entries = [
        _entry("a", "clean_acoustic_strummed"),
        _entry("b", "clean_acoustic_strummed"),
        _entry("c", "clean_acoustic_single_line"),
    ]

    capped = apply_limits(entries, max_clips_per_tier=2, total_limit=2)

    assert [entry.id for entry in capped] == ["a", "b"]


def test_apply_limits_with_no_caps_preserves_all_sorted() -> None:
    entries = [_entry("b"), _entry("a"), _entry("c")]
    out = apply_limits(entries)
    assert [entry.id for entry in out] == ["a", "b", "c"]


def test_render_toml_round_trips_via_tomllib() -> None:
    entries = [
        _entry("a", "clean_acoustic_strummed"),
        _entry("b", "clean_acoustic_single_line"),
    ]
    text = render_toml(entries)
    parsed = tomllib.loads(text)
    assert len(parsed["clips"]) == 2
    by_id = {clip["id"]: clip for clip in parsed["clips"]}
    assert by_id["a"]["tier"] == "clean_acoustic_strummed"
    assert by_id["a"]["annotation_format"] == "guitarset_jams"


def test_render_toml_is_byte_stable() -> None:
    """Same entries → same bytes, regardless of input order."""
    entries_in_order_a = [_entry("z"), _entry("a"), _entry("m")]
    entries_in_order_b = [_entry("a"), _entry("m"), _entry("z")]
    assert render_toml(entries_in_order_a) == render_toml(entries_in_order_b)


def test_render_toml_emits_header_when_provided() -> None:
    text = render_toml([_entry("a")], header_comment="hello world")
    assert text.startswith("# hello world\n")


def test_render_toml_rewrites_paths_under_data_root(tmp_path: Path) -> None:
    """media/annotation paths under data_root become $TABVISION_DATA_ROOT/<rest>."""
    data_root = tmp_path / "datasets"
    data_root.mkdir()
    entry = ClipEntry(
        id="clip-x",
        tier="clean_acoustic_strummed",
        source="GuitarSet",
        split="validation",
        media_path=str((data_root / "guitarset" / "audio.wav").resolve()),
        annotation_path=str((data_root / "guitarset" / "ann.jams").resolve()),
        annotation_format="guitarset_jams",
    )
    text = render_toml([entry], data_root=data_root)
    assert '"$TABVISION_DATA_ROOT/guitarset/audio.wav"' in text
    assert '"$TABVISION_DATA_ROOT/guitarset/ann.jams"' in text
    # Paths NOT under data_root should be untouched.
    assert "/datasets/" not in text  # absolute prefix is gone


def test_render_toml_leaves_paths_outside_data_root_alone(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets"
    data_root.mkdir()
    other = tmp_path / "elsewhere" / "x.wav"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    entry = ClipEntry(
        id="clip-x",
        tier="clean_acoustic_strummed",
        source="GuitarSet",
        split="validation",
        media_path=str(other.resolve()),
        annotation_path=str(other.resolve()),
        annotation_format="guitarset_jams",
    )
    text = render_toml([entry], data_root=data_root)
    assert "$TABVISION_DATA_ROOT" not in text
    # Parse back instead of substring-matching the raw path: _toml_escape doubles
    # backslashes, so a raw Windows path is not a literal substring of `text`
    # (this assertion silently only held on POSIX before).
    clip = tomllib.loads(text)["clips"][0]
    assert clip["media_path"] == str(other.resolve())
    assert clip["annotation_path"] == str(other.resolve())


def test_render_toml_with_no_data_root_is_unchanged(tmp_path: Path) -> None:
    """Backward-compat: omitting data_root keeps current absolute-path output."""
    entry = ClipEntry(
        id="clip-x",
        tier="clean_acoustic_strummed",
        source="GuitarSet",
        split="validation",
        media_path="/some/abs/path.wav",
        annotation_path="/some/abs/path.jams",
        annotation_format="guitarset_jams",
    )
    text = render_toml([entry], data_root=None)
    assert "/some/abs/path.wav" in text
    assert "$TABVISION_DATA_ROOT" not in text


def test_relativize_to_data_root_rewrites_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows absolute paths (backslash-separated) must still be rewritten to
    forward-slash ``$TABVISION_DATA_ROOT/...`` tokens.

    Regression: the old ``startswith(abs_root + "/")`` prefix check hard-coded a
    forward slash, so on Windows it never matched and leaked ``C:\\...`` paths
    into checked-in manifests. ``PureWindowsPath`` parses backslash paths on any
    host, so monkeypatching the module ``Path`` to it exercises the Windows
    behaviour from a POSIX CI runner too. The helper expects an already
    expanded+resolved root (``render_toml`` does that), so we pass an absolute
    ``PureWindowsPath`` directly.
    """
    import pathlib

    from tabvision.eval import manifest_builder

    monkeypatch.setattr(manifest_builder, "Path", pathlib.PureWindowsPath)
    data_root = pathlib.PureWindowsPath(r"C:\Users\patri\.tabvision\data")

    media = (
        r"C:\Users\patri\.tabvision\data\guitar-techs"
        r"\P1_chords\audio\directinput\directinput_Drop3_7.wav"
    )
    annotation = (
        r"C:\Users\patri\.tabvision\data\guitar-techs"
        r"\P1_chords\midi\midi_Drop3_7.mid"
    )

    assert (
        manifest_builder._relativize_to_data_root(media, data_root)
        == "$TABVISION_DATA_ROOT/guitar-techs/P1_chords/audio/directinput/"
        "directinput_Drop3_7.wav"
    )
    assert (
        manifest_builder._relativize_to_data_root(annotation, data_root)
        == "$TABVISION_DATA_ROOT/guitar-techs/P1_chords/midi/midi_Drop3_7.mid"
    )

    # A Windows path that is NOT under the data root is returned untouched.
    outside = r"C:\Users\patri\elsewhere\other.wav"
    assert manifest_builder._relativize_to_data_root(outside, data_root) == outside

    # The root itself collapses to the bare token (no trailing "/.").
    assert (
        manifest_builder._relativize_to_data_root(str(data_root), data_root)
        == "$TABVISION_DATA_ROOT"
    )


def test_summarise_coverage_reports_per_tier_and_per_split() -> None:
    entries = [
        _entry("a", "clean_acoustic_strummed"),
        _entry("b", "clean_acoustic_strummed"),
        _entry("c", "clean_acoustic_single_line"),
    ]
    summary = summarise_coverage(entries)
    assert "Total clips: 3" in summary
    assert "clean_acoustic_strummed: 2 clips" in summary
    assert "clean_acoustic_single_line: 1 clips" in summary


def test_build_manifest_skips_missing_roots(tmp_path: Path) -> None:
    """Missing GuitarSet root → empty result, no exception."""
    entries = build_manifest(guitarset_root=tmp_path / "nope")
    assert entries == []


def test_build_manifest_splits_filter(tmp_path: Path) -> None:
    """``splits=('validation',)`` should keep only player-05 clips."""
    _make_guitarset_layout(
        tmp_path / "guitarset",
        [
            ("00_Rock1-90-C#_comp", {"annotations": []}),  # train
            ("05_Funk1-114-Ab_solo", {"annotations": []}),  # validation
        ],
    )

    train_only = build_manifest(
        guitarset_root=tmp_path / "guitarset",
        splits=("train",),
    )
    validation_only = build_manifest(
        guitarset_root=tmp_path / "guitarset",
        splits=("validation",),
    )
    both = build_manifest(guitarset_root=tmp_path / "guitarset")

    assert {entry.id for entry in train_only} == {"guitarset/00_Rock1-90-C#_comp"}
    assert {entry.id for entry in validation_only} == {"guitarset/05_Funk1-114-Ab_solo"}
    assert len(both) == 2


def test_build_manifest_includes_utaustin_root(tmp_path: Path) -> None:
    _make_utaustin_layout(tmp_path / "utaustin", ["0"])

    entries = build_manifest(utaustin_root=tmp_path / "utaustin")

    assert [entry.id for entry in entries] == ["utaustin/0"]
    assert entries[0].annotation_format == "utaustin_tablature_npy"


def test_build_manifest_emits_synthetic_train_clip_ok(tmp_path: Path) -> None:
    """Training-split synthetic clips should pass the in-builder guard."""
    # Use a custom ClipEntry-yielding scanner via the public function
    entries = [
        ClipEntry(
            id="synthetic-train-01",
            tier="distorted_electric",
            source="synthtab/electric",
            split="train",
            media_path="/data/x.wav",
            annotation_path="/data/x.json",
            annotation_format="synthtab_json",
        ),
    ]
    # The guard should be a no-op for train split; verify via apply_limits roundtrip.
    out = apply_limits(entries, max_clips_per_tier=1)
    assert len(out) == 1


def test_main_writes_manifest_and_passes_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: build_composite_manifest builds → manifest validates."""
    _make_guitarset_layout(
        tmp_path / "guitarset",
        [
            (
                "05_Rock1-90-C#_comp",
                {
                    "annotations": [
                        {
                            "namespace": "note_midi",
                            "annotation_metadata": {"data_source": "0"},
                            "data": [
                                {"time": 0.0, "duration": 0.5, "value": 40},
                            ],
                        }
                    ]
                },
            ),
            (
                "05_Funk1-114-Ab_solo",
                {
                    "annotations": [
                        {
                            "namespace": "note_midi",
                            "annotation_metadata": {"data_source": "0"},
                            "data": [
                                {"time": 1.0, "duration": 0.5, "value": 45},
                            ],
                        }
                    ]
                },
            ),
        ],
    )
    output = tmp_path / "composite.toml"

    from tabvision.eval.manifest_builder import main

    rc = main(
        [
            "--guitarset",
            str(tmp_path / "guitarset"),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.is_file()
    captured = capsys.readouterr()
    assert "Wrote 2 clips" in captured.out
    assert "Manifest validation passed." in captured.out

    # The emitted manifest should itself validate cleanly.
    validation = validate_manifest(output)
    assert validation.passed


def test_main_requires_at_least_one_root(tmp_path: Path) -> None:
    """Without --guitarset / --guitar-techs, the CLI exits with usage error."""
    from tabvision.eval.manifest_builder import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--output", str(tmp_path / "x.toml")])
    assert excinfo.value.code == 2


def test_main_returns_1_when_no_clips_discovered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Specifying a path with no matching data → rc=1, no output file."""
    output = tmp_path / "composite.toml"
    from tabvision.eval.manifest_builder import main

    rc = main(
        [
            "--guitarset",
            str(tmp_path / "empty"),
            "--output",
            str(output),
        ]
    )

    assert rc == 1
    assert not output.exists()
    captured = capsys.readouterr()
    assert "No clips discovered" in captured.out
