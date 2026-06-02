"""Unit tests for ``manifest_builder.scan_guitar_techs``.

The synthetic tree mirrors the *real* Guitar-TECHS layout (verified 2026-06-02
against Zenodo record 14963133): ``<Pn_category>/midi/midi_<content>.mid`` paired
with ``<Pn_category>/audio/<capture>/<capture>_<content>.<ext>``. MIDI and audio
share the ``<content>`` token, NOT a common prefix.

Runnable two ways:
  - ``pytest tabvision/tests/unit/test_scan_guitar_techs.py``
  - ``python tabvision/tests/unit/test_scan_guitar_techs.py``  (no pytest dep)
"""

from __future__ import annotations

from pathlib import Path

from tabvision.eval.manifest_builder import scan_guitar_techs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _build_tree(root: Path) -> None:
    # P1 chords -> train; MIDI 'midi_Drop3_7' pairs with audio 'directinput_Drop3_7'
    # (shared content 'Drop3_7', different prefixes). DI preferred over mic'd amp.
    _touch(root / "P1_chords" / "midi" / "midi_Drop3_7.mid")
    _touch(root / "P1_chords" / "audio" / "directinput" / "directinput_Drop3_7.wav")
    _touch(root / "P1_chords" / "audio" / "micamp" / "micamp_Drop3_7.wav")
    # P3 -> validation
    _touch(root / "P3_scales" / "midi" / "midi_Cmaj.mid")
    _touch(root / "P3_scales" / "audio" / "directinput" / "directinput_Cmaj.wav")
    # stretch technique (path contains 'bend') -> skipped
    _touch(root / "P1_bends" / "midi" / "midi_slow.mid")
    _touch(root / "P1_bends" / "audio" / "directinput" / "directinput_slow.wav")
    # MIDI with no matching audio in its group -> dropped
    _touch(root / "P2_singlenotes" / "midi" / "midi_E5.mid")
    # macOS zip cruft -> ignored
    _touch(root / "__MACOSX" / "P1_chords" / "midi" / "._midi_Drop3_7.mid")


def _by_id(entries: list) -> dict:
    return {e.id: e for e in entries}


def test_scan_guitar_techs_real_layout(tmp_path: Path | None = None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(tmp_path) if tmp_path is not None else Path(td)
        _build_tree(root)
        entries = scan_guitar_techs(root)
        by_id = _by_id(entries)

        # Kept: P1_chords/Drop3_7 + P3_scales/Cmaj. bend skipped; E5 dropped; cruft ignored.
        assert len(entries) == 2, [e.id for e in entries]
        assert "guitar-techs/P1_chords/midi/midi_Drop3_7" in by_id
        assert "guitar-techs/P3_scales/midi/midi_Cmaj" in by_id
        assert not any("bend" in cid or "slow" in cid for cid in by_id)
        assert not any("E5" in cid for cid in by_id)
        assert not any("MACOSX" in cid for cid in by_id)

        for entry in entries:
            assert entry.tier == "clean_electric"
            assert entry.source == "GuitarTECHS"
            assert entry.annotation_format == "guitar_techs_midi"

        # cross-prefix content pairing + DI preference
        p1 = by_id["guitar-techs/P1_chords/midi/midi_Drop3_7"]
        assert p1.media_path.endswith("directinput_Drop3_7.wav"), p1.media_path
        assert p1.split == "train"

        # performer split: P3 -> validation
        assert by_id["guitar-techs/P3_scales/midi/midi_Cmaj"].split == "validation"


def test_scan_guitar_techs_missing_root() -> None:
    assert scan_guitar_techs(Path("/no/such/guitar-techs/root")) == []


if __name__ == "__main__":
    test_scan_guitar_techs_real_layout()
    test_scan_guitar_techs_missing_root()
    print("PASS: scan_guitar_techs real-layout + missing-root")
