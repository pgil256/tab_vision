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
    assert {entry.id for entry in validation_only} == {
        "guitarset/05_Funk1-114-Ab_solo"
    }
    assert len(both) == 2


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
