"""Unit tests for ``manifest_builder.scan_guitar_techs``.

The Guitar-TECHS on-disk layout is *inferred* (arXiv:2501.03720 + project
page) until the real download is verified, so these tests pin the scanner's
heuristics against a synthetic tree: tier assignment, performer→split,
exact/prefix audio pairing, DI/clean preference, split audio+midi trees, and
stretch-technique skipping.

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
    # exact-stem pairing, player 01 → train
    _touch(root / "player01" / "scales" / "Cmaj.mid")
    _touch(root / "player01" / "scales" / "Cmaj.wav")
    # prefix-stem pairing + DI/clean preference, player 02 → train
    _touch(root / "player02" / "excerpts" / "song.mid")
    _touch(root / "player02" / "excerpts" / "song_amp.wav")
    _touch(root / "player02" / "excerpts" / "song_DI.wav")
    # player 03 → validation
    _touch(root / "player03" / "scales" / "Amin.mid")
    _touch(root / "player03" / "scales" / "Amin.wav")
    # stretch technique → skipped
    _touch(root / "player01" / "techniques" / "bend_fast.mid")
    _touch(root / "player01" / "techniques" / "bend_fast.wav")
    # split midi/ + audio/ trees, exact stem found via whole-root index
    _touch(root / "player02" / "split" / "midi" / "riff.mid")
    _touch(root / "player02" / "split" / "audio" / "riff.flac")
    # MIDI with no audio anywhere → dropped
    _touch(root / "player01" / "orphans" / "noaudio.mid")


def _by_id(entries: list) -> dict[str, object]:
    return {e.id: e for e in entries}


def test_scan_guitar_techs_synthetic(tmp_path: Path | None = None) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(tmp_path) if tmp_path is not None else Path(td)
        _build_tree(root)
        entries = scan_guitar_techs(root)
        by_id = _by_id(entries)

        # 4 kept: Cmaj, song, Amin, riff. bend_* skipped; noaudio dropped.
        assert len(entries) == 4, [e.id for e in entries]
        assert "guitar-techs/player01/scales/Cmaj" in by_id
        assert "guitar-techs/player02/excerpts/song" in by_id
        assert "guitar-techs/player03/scales/Amin" in by_id
        assert "guitar-techs/player02/split/midi/riff" in by_id
        assert not any("bend" in cid for cid in by_id)
        assert not any("noaudio" in cid for cid in by_id)

        # every kept clip is the clean_electric tier from GuitarTECHS via MIDI
        for entry in entries:
            assert entry.tier == "clean_electric"
            assert entry.source == "GuitarTECHS"
            assert entry.annotation_format == "guitar_techs_midi"

        # performer split: player 03 → validation, others → train
        assert by_id["guitar-techs/player03/scales/Amin"].split == "validation"
        assert by_id["guitar-techs/player01/scales/Cmaj"].split == "train"

        # DI/clean render preferred when several share a stem prefix
        assert by_id["guitar-techs/player02/excerpts/song"].media_path.endswith(
            "song_DI.wav"
        )
        # split audio/ tree resolved
        assert by_id["guitar-techs/player02/split/midi/riff"].media_path.endswith(
            "riff.flac"
        )


def test_scan_guitar_techs_missing_root() -> None:
    assert scan_guitar_techs(Path("/no/such/guitar-techs/root")) == []


if __name__ == "__main__":
    test_scan_guitar_techs_synthetic()
    test_scan_guitar_techs_missing_root()
    print("PASS: scan_guitar_techs synthetic + missing-root")
