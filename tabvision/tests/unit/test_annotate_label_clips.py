"""Unit tests for ``scripts.annotate.label_clips`` Flask app.

Exercises the route handlers end-to-end against a synthetic 1 s clip.
flask is required (it's the labeling tool's only extra dep beyond
opencv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
flask = pytest.importorskip("flask")

# ruff: noqa: E402, I001
from scripts.annotate import storage
from scripts.annotate.label_clips import discover_clips, make_app


def _write_clip(path: Path, n_frames: int = 30, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()


@pytest.fixture
def app(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _write_clip(clips_dir / "a.mp4")
    _write_clip(clips_dir / "b.mov")
    eval_root = tmp_path / "labels"

    clips = discover_clips(clips_dir)
    app = make_app(clips, eval_root, fingering_frames=4)
    app.config["TESTING"] = True
    app.config["EVAL_ROOT"] = eval_root  # smuggle for assertions
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ----- discovery -----


def test_discover_clips_filters_by_suffix(tmp_path):
    (tmp_path / "ok.mp4").touch()
    (tmp_path / "ok.MOV").touch()
    (tmp_path / "skip.txt").touch()
    found = discover_clips(tmp_path)
    assert {p.name for p in found} == {"ok.mp4", "ok.MOV"}


def test_discover_clips_raises_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit, match="no video files"):
        discover_clips(tmp_path)


def test_discover_clips_raises_on_missing_dir(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        discover_clips(tmp_path / "nope")


# ----- index page -----


def test_index_lists_all_clips(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "a" in body and "b" in body  # clip ids
    assert "framing" in body and "fretboard" in body and "fingering" in body


# ----- /clip/<id>/frame/<n>.jpg -----


def test_frame_endpoint_returns_jpeg(client):
    r = client.get("/clip/a/frame/0.jpg")
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "image/jpeg"
    assert r.data.startswith(b"\xff\xd8")  # JPEG SOI marker


def test_frame_endpoint_404s_on_missing_clip(client):
    r = client.get("/clip/nope/frame/0.jpg")
    assert r.status_code == 404


def test_frame_endpoint_404s_on_out_of_range_index(client):
    r = client.get("/clip/a/frame/99999.jpg")
    assert r.status_code == 404


# ----- framing -----


def test_framing_post_saves_label(client, app):
    r = client.post("/framing/a", json={"label": "good", "tags": [], "notes": "ok"})
    assert r.status_code == 200
    eval_root = app.config["EVAL_ROOT"]
    saved = storage.load_framing("a.mp4", eval_root=eval_root)
    assert saved is not None
    assert saved.label == "good"
    assert saved.notes == "ok"


def test_framing_post_records_tags(client, app):
    client.post("/framing/a", json={"label": "bad", "tags": ["dim", "off-center"]})
    saved = storage.load_framing("a.mp4", eval_root=app.config["EVAL_ROOT"])
    assert sorted(saved.tags) == ["dim", "off-center"]


def test_framing_get_renders_form(client):
    r = client.get("/framing/a")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "framing" in body.lower()


# ----- fretboard -----


def test_fretboard_post_rejects_incomplete_set(client):
    r = client.post(
        "/fretboard/a",
        json={"frame_idx": 5, "points": [{"fret": 5, "edge": "top", "x": 1, "y": 2}]},
    )
    assert r.status_code == 400


def test_fretboard_post_accepts_complete_set(client, app):
    points = [
        {"fret": 5, "edge": "top", "x": 10, "y": 20},
        {"fret": 5, "edge": "bottom", "x": 10, "y": 100},
        {"fret": 12, "edge": "top", "x": 80, "y": 22},
        {"fret": 12, "edge": "bottom", "x": 80, "y": 102},
    ]
    r = client.post("/fretboard/a", json={"frame_idx": 5, "points": points})
    assert r.status_code == 200
    saved = storage.load_fretboard("a.mp4", eval_root=app.config["EVAL_ROOT"])
    assert saved is not None
    assert saved.is_complete()
    assert saved.frame_idx == 5


# ----- fingering -----


def test_fingering_post_saves_per_frame_labels(client, app):
    payload = {
        "frames": [
            {
                "frame_idx": 5,
                "fingers": [
                    {"finger": "index", "string": "2", "fret": "5"},
                    {"finger": "middle", "string": "", "fret": ""},
                ],
            },
            {
                "frame_idx": 10,
                "fingers": [
                    {"finger": "ring", "string": 1, "fret": 8},
                ],
            },
        ],
    }
    r = client.post("/fingering/a", json=payload)
    assert r.status_code == 200
    saved = storage.load_fingering("a.mp4", eval_root=app.config["EVAL_ROOT"])
    assert saved is not None
    assert len(saved.frames) == 2
    f0 = saved.frames[0]
    assert f0.fingers[0].is_fretting
    assert not f0.fingers[1].is_fretting   # blank string/fret -> not fretting


def test_fingering_get_pre_populates_from_existing(client, app):
    """After a save, GET should embed the existing labels in the page so
    refreshing doesn't lose state."""
    client.post(
        "/fingering/b",
        json={"frames": [{"frame_idx": 0,
                          "fingers": [{"finger": "index", "string": 3, "fret": 7}]}]},
    )
    r = client.get("/fingering/b")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    # The labels are passed via Jinja's |tojson filter; the inner
    # FingerLabel fields show up verbatim. (The dict key "0" — frame_idx
    # — gets stringified as a JSON object key and isn't searchable as
    # `"frame_idx": 0`.)
    assert "\"finger\": \"index\"" in body
    assert "\"string\": 3" in body and "\"fret\": 7" in body
