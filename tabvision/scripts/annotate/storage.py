"""JSON storage for the labeling harness.

Three label types, one per Phase-3/4 acceptance gate:

- :class:`FramingLabel` — preflight gate (good vs bad framing + tags).
- :class:`FretboardLabel` — keypoint-fretboard gate (4 fret-intersection
  clicks at frets 5 + 12, top + bottom edges).
- :class:`FingeringLabel` — Phase-4 gate (per-frame finger → (string, fret)
  for the four fretting fingers).

Each clip's labels live under ``<eval_root>/{framing,fretboard,fingering}/<clip_id>.json``
where ``clip_id`` is a filesystem-safe slug derived from the clip path.

The :class:`Label` dataclasses are the canonical schema; the eval harnesses
in ``tests/eval/test_phase3_eval.py`` and ``tests/eval/test_phase4_eval.py``
load JSON via the loaders here so the schema has exactly one source of
truth.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# The eval root defaults to ``tabvision/data/eval`` per CLAUDE.md layout;
# overridable via env so a CI run can point at a checked-in fixtures dir.
DEFAULT_EVAL_ROOT_ENV = "TABVISION_EVAL_ROOT"


def default_eval_root() -> Path:
    if env := os.environ.get(DEFAULT_EVAL_ROOT_ENV):
        return Path(env)
    # tabvision/scripts/annotate/storage.py → tabvision/data/eval
    return Path(__file__).resolve().parents[2] / "data" / "eval"


def clip_id(clip_path: str | Path) -> str:
    """Filesystem-safe identifier derived from the clip filename.

    Two clips with the same filename in different directories collide on
    purpose — the labeling tool refuses to start with two different
    sources resolving to the same id.
    """
    stem = Path(clip_path).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "unnamed"


# ----- framing -----

# Tag vocabulary for "bad" framing classifications. Free-form notes are
# allowed too; tags are a controlled vocabulary so the eval harness can
# bucket failures.
FRAMING_TAGS = (
    "off-center",
    "partial-occlusion",
    "dim",
    "oblique-angle",
    "drift",
    "motion-blur",
    "over-exposed",
)


@dataclass
class FramingLabel:
    clip_path: str
    label: Literal["good", "bad"]
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> FramingLabel:
        return cls(
            clip_path=data["clip_path"],
            label=data["label"],
            tags=list(data.get("tags", [])),
            notes=str(data.get("notes", "")),
        )


# ----- fretboard -----


@dataclass
class FretIntersection:
    """One hand-clicked fret-line endpoint, in image-pixel coords."""

    fret: int               # the fret number (5 or 12 per the spec)
    edge: Literal["top", "bottom"]   # high-E (top) or low-E (bottom) side
    x: float
    y: float


@dataclass
class FretboardLabel:
    clip_path: str
    frame_idx: int           # the representative frame the clicks were made on
    points: list[FretIntersection]
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "clip_path": self.clip_path,
            "frame_idx": self.frame_idx,
            "points": [asdict(p) for p in self.points],
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict) -> FretboardLabel:
        return cls(
            clip_path=data["clip_path"],
            frame_idx=int(data["frame_idx"]),
            points=[FretIntersection(**p) for p in data["points"]],
            notes=str(data.get("notes", "")),
        )

    def is_complete(self) -> bool:
        """Spec: 4 intersections — frets 5 and 12 × top and bottom edges."""
        seen = {(p.fret, p.edge) for p in self.points}
        required = {(5, "top"), (5, "bottom"), (12, "top"), (12, "bottom")}
        return required.issubset(seen)


# ----- fingering -----


FINGER_NAMES = ("index", "middle", "ring", "pinky")


@dataclass
class FingerLabel:
    """Per-finger label for one labeled frame.

    ``string`` and ``fret`` are 1-indexed-from-low-E and 0-indexed
    respectively to match the spec convention (``string=6`` = high E).
    Set both to ``None`` for a finger that isn't fretting in this frame.
    """

    finger: Literal["index", "middle", "ring", "pinky"]
    string: int | None
    fret: int | None

    @property
    def is_fretting(self) -> bool:
        return self.string is not None and self.fret is not None


@dataclass
class FrameLabel:
    frame_idx: int
    fingers: list[FingerLabel] = field(default_factory=list)


@dataclass
class FingeringLabel:
    clip_path: str
    frames: list[FrameLabel] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "clip_path": self.clip_path,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "fingers": [
                        {
                            "finger": fl.finger,
                            "string": fl.string,
                            "fret": fl.fret,
                        }
                        for fl in f.fingers
                    ],
                }
                for f in self.frames
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> FingeringLabel:
        return cls(
            clip_path=data["clip_path"],
            frames=[
                FrameLabel(
                    frame_idx=int(f["frame_idx"]),
                    fingers=[FingerLabel(**fl) for fl in f["fingers"]],
                )
                for f in data["frames"]
            ],
        )


# ----- save / load -----


_KIND_DIRS: dict[str, str] = {
    "framing": "framing",
    "fretboard": "fretboard",
    "fingering": "fingering",
}


def _path_for(kind: str, clip_path: str | Path, *, eval_root: Path | None = None) -> Path:
    root = eval_root or default_eval_root()
    if kind not in _KIND_DIRS:
        raise ValueError(f"unknown label kind {kind!r}; pick one of {list(_KIND_DIRS)}")
    return root / _KIND_DIRS[kind] / f"{clip_id(clip_path)}.json"


def save_framing(label: FramingLabel, *, eval_root: Path | None = None) -> Path:
    return _save("framing", label.clip_path, label.to_json(), eval_root=eval_root)


def load_framing(clip_path: str | Path, *, eval_root: Path | None = None) -> FramingLabel | None:
    return _load("framing", clip_path, FramingLabel.from_json, eval_root=eval_root)


def save_fretboard(label: FretboardLabel, *, eval_root: Path | None = None) -> Path:
    return _save("fretboard", label.clip_path, label.to_json(), eval_root=eval_root)


def load_fretboard(
    clip_path: str | Path, *, eval_root: Path | None = None
) -> FretboardLabel | None:
    return _load("fretboard", clip_path, FretboardLabel.from_json, eval_root=eval_root)


def save_fingering(label: FingeringLabel, *, eval_root: Path | None = None) -> Path:
    return _save("fingering", label.clip_path, label.to_json(), eval_root=eval_root)


def load_fingering(
    clip_path: str | Path, *, eval_root: Path | None = None
) -> FingeringLabel | None:
    return _load("fingering", clip_path, FingeringLabel.from_json, eval_root=eval_root)


def _save(kind: str, clip_path: str | Path, payload: dict, *, eval_root: Path | None) -> Path:
    p = _path_for(kind, clip_path, eval_root=eval_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)  # atomic on POSIX
    return p


def _load(kind: str, clip_path: str | Path, factory, *, eval_root: Path | None):
    p = _path_for(kind, clip_path, eval_root=eval_root)
    if not p.exists():
        return None
    return factory(json.loads(p.read_text()))


def list_labeled_clips(kind: str, *, eval_root: Path | None = None) -> list[str]:
    """Return the list of clip ids that already have a saved label of ``kind``."""
    if kind not in _KIND_DIRS:
        raise ValueError(f"unknown label kind {kind!r}")
    root = (eval_root or default_eval_root()) / _KIND_DIRS[kind]
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


__all__ = [
    "FRAMING_TAGS",
    "FINGER_NAMES",
    "DEFAULT_EVAL_ROOT_ENV",
    "FingerLabel",
    "FrameLabel",
    "FingeringLabel",
    "FramingLabel",
    "FretIntersection",
    "FretboardLabel",
    "clip_id",
    "default_eval_root",
    "list_labeled_clips",
    "load_fingering",
    "load_framing",
    "load_fretboard",
    "save_fingering",
    "save_framing",
    "save_fretboard",
]
