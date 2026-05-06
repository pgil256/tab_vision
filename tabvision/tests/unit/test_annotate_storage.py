"""Unit tests for ``scripts.annotate.storage`` — JSON IO + schema."""

from __future__ import annotations

import json

import pytest

from scripts.annotate import storage

# ----- clip_id slugging -----


def test_clip_id_strips_extensions_and_normalises_special_chars():
    assert storage.clip_id("/path/to/training-01.MOV") == "training-01"
    assert storage.clip_id("clip with spaces.mp4") == "clip-with-spaces"
    assert storage.clip_id("foo!!bar.mov") == "foo-bar"


def test_clip_id_falls_back_to_unnamed_for_empty_input():
    assert storage.clip_id("") == "unnamed"


def test_clip_id_is_idempotent_under_repeated_normalisation():
    once = storage.clip_id("clip!!.mov")
    twice = storage.clip_id(once + ".mov")
    assert once == twice


# ----- default_eval_root override -----


def test_default_eval_root_uses_env_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv(storage.DEFAULT_EVAL_ROOT_ENV, str(tmp_path / "labels"))
    assert storage.default_eval_root() == tmp_path / "labels"


def test_default_eval_root_falls_back_to_repo_data_dir(monkeypatch):
    monkeypatch.delenv(storage.DEFAULT_EVAL_ROOT_ENV, raising=False)
    p = storage.default_eval_root()
    # tabvision/data/eval is the convention per CLAUDE.md.
    assert p.parts[-3:] == ("tabvision", "data", "eval")


# ----- framing -----


def test_framing_round_trip(tmp_path):
    label = storage.FramingLabel(
        clip_path="/clips/a.mov",
        label="bad",
        tags=["off-center", "dim"],
        notes="dark and tilted",
    )
    storage.save_framing(label, eval_root=tmp_path)
    loaded = storage.load_framing("/clips/a.mov", eval_root=tmp_path)
    assert loaded == label


def test_framing_save_creates_directory(tmp_path):
    label = storage.FramingLabel(clip_path="/c/x.mov", label="good")
    storage.save_framing(label, eval_root=tmp_path)
    assert (tmp_path / "framing" / "x.json").exists()


def test_framing_load_returns_none_when_missing(tmp_path):
    assert storage.load_framing("/clips/nope.mov", eval_root=tmp_path) is None


# ----- fretboard -----


def test_fretboard_round_trip(tmp_path):
    label = storage.FretboardLabel(
        clip_path="/clips/a.mov",
        frame_idx=42,
        points=[
            storage.FretIntersection(fret=5, edge="top", x=100.0, y=20.0),
            storage.FretIntersection(fret=5, edge="bottom", x=110.0, y=80.0),
            storage.FretIntersection(fret=12, edge="top", x=200.0, y=22.0),
            storage.FretIntersection(fret=12, edge="bottom", x=210.0, y=82.0),
        ],
    )
    storage.save_fretboard(label, eval_root=tmp_path)
    loaded = storage.load_fretboard("/clips/a.mov", eval_root=tmp_path)
    assert loaded == label
    assert loaded.is_complete()


def test_fretboard_is_complete_requires_all_four_corners():
    label = storage.FretboardLabel(
        clip_path="x", frame_idx=0,
        points=[
            storage.FretIntersection(fret=5, edge="top", x=0, y=0),
            storage.FretIntersection(fret=5, edge="bottom", x=0, y=0),
            storage.FretIntersection(fret=12, edge="top", x=0, y=0),
        ],
    )
    assert not label.is_complete()


def test_fretboard_handles_extra_or_duplicate_points(tmp_path):
    """Loader should not silently drop unexpected points; round-tripping
    preserves them. is_complete() still accepts duplicates as long as the
    required (fret, edge) set is covered."""
    label = storage.FretboardLabel(
        clip_path="/clips/a.mov",
        frame_idx=0,
        points=[
            storage.FretIntersection(fret=5, edge="top", x=10.0, y=20.0),
            storage.FretIntersection(fret=5, edge="top", x=11.0, y=21.0),  # dup
            storage.FretIntersection(fret=5, edge="bottom", x=12.0, y=80.0),
            storage.FretIntersection(fret=12, edge="top", x=200.0, y=22.0),
            storage.FretIntersection(fret=12, edge="bottom", x=210.0, y=82.0),
        ],
    )
    storage.save_fretboard(label, eval_root=tmp_path)
    loaded = storage.load_fretboard("/clips/a.mov", eval_root=tmp_path)
    assert len(loaded.points) == 5
    assert loaded.is_complete()


# ----- fingering -----


def test_fingering_round_trip(tmp_path):
    label = storage.FingeringLabel(
        clip_path="/clips/a.mov",
        frames=[
            storage.FrameLabel(
                frame_idx=10,
                fingers=[
                    storage.FingerLabel("index", string=2, fret=5),
                    storage.FingerLabel("middle", string=3, fret=7),
                    storage.FingerLabel("ring", string=None, fret=None),
                    storage.FingerLabel("pinky", string=1, fret=8),
                ],
            ),
            storage.FrameLabel(
                frame_idx=30,
                fingers=[
                    storage.FingerLabel("index", string=2, fret=0),  # open
                ],
            ),
        ],
    )
    storage.save_fingering(label, eval_root=tmp_path)
    loaded = storage.load_fingering("/clips/a.mov", eval_root=tmp_path)
    assert loaded == label


def test_finger_label_is_fretting_predicate():
    pressed = storage.FingerLabel("index", string=1, fret=3)
    open_string = storage.FingerLabel("index", string=1, fret=0)
    not_fretting = storage.FingerLabel("ring", string=None, fret=None)
    assert pressed.is_fretting
    assert open_string.is_fretting  # fret=0 means open string, still labeled
    assert not not_fretting.is_fretting


# ----- list_labeled_clips -----


def test_list_labeled_clips_returns_sorted_ids(tmp_path):
    storage.save_framing(
        storage.FramingLabel(clip_path="/c/zz.mov", label="good"),
        eval_root=tmp_path,
    )
    storage.save_framing(
        storage.FramingLabel(clip_path="/c/aa.mov", label="bad"),
        eval_root=tmp_path,
    )
    assert storage.list_labeled_clips("framing", eval_root=tmp_path) == ["aa", "zz"]


def test_list_labeled_clips_returns_empty_when_dir_missing(tmp_path):
    assert storage.list_labeled_clips("fretboard", eval_root=tmp_path) == []


def test_list_labeled_clips_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="unknown label kind"):
        storage.list_labeled_clips("not-a-kind", eval_root=tmp_path)


# ----- atomic write -----


def test_save_atomically_replaces_existing_file(tmp_path):
    """The save helper writes via a tmp file + replace; a partial JSON
    must never replace a complete one even if the write is interrupted
    mid-flight."""
    label = storage.FramingLabel(clip_path="/c/x.mov", label="good")
    p = storage.save_framing(label, eval_root=tmp_path)
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["label"] == "good"
    # No leftover .tmp file.
    assert not p.with_suffix(p.suffix + ".tmp").exists()
