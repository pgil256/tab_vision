"""Model/dependency readiness helper for the v1 CLI path.

This module is intentionally noninteractive. Public/package-backed models are
reported through import checks. The trained YOLO-OBB detector and MediaPipe
HandLandmarker task are reported through the exact paths used by the video
backends. The YOLO weights are produced by the training script rather than
downloaded from a public release artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"
YOLO_CHECKPOINT_ENV = "TABVISION_GUITAR_YOLO_CHECKPOINT"
YOLO_CHECKPOINT_NAME = "guitar-yolo-obb-finetuned.pt"
HAND_MODEL_ENV = "TABVISION_MEDIAPIPE_HAND_MODEL"
HAND_MODEL_NAME = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class ReadinessItem:
    name: str
    status: str
    detail: str
    action: str


def data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", DEFAULT_DATA_ROOT))


def yolo_checkpoint_path(root: Path | None = None) -> Path:
    env_path = os.environ.get(YOLO_CHECKPOINT_ENV)
    if env_path:
        return Path(env_path)
    return (root or data_root()) / "models" / YOLO_CHECKPOINT_NAME


def hand_model_path() -> Path:
    env_path = os.environ.get(HAND_MODEL_ENV)
    if env_path:
        return Path(env_path)
    return Path.home() / ".mediapipe" / "models" / HAND_MODEL_NAME


def collect_status(root: Path | None = None) -> list[ReadinessItem]:
    root = root or data_root()
    yolo_path = yolo_checkpoint_path(root)
    hand_path = hand_model_path()
    return [
        _module_item(
            name="basic-pitch audio baseline",
            modules=("basic_pitch", "soundfile", "scipy"),
            action="Install with: python -m pip install -e '.[audio-baseline]'",
        ),
        _module_item(
            name="highres audio backend",
            modules=("hf_midi_transcription", "torch", "pretty_midi", "soundfile"),
            action="Install with: python -m pip install -e '.[audio-highres]'",
        ),
        _module_item(
            name="render extras",
            modules=("mido", "music21", "guitarpro"),
            action="Install with: python -m pip install -e '.[render]'",
        ),
        _module_item(
            name="vision dependencies",
            modules=("cv2", "mediapipe", "ultralytics"),
            action="Install with: python -m pip install -e '.[vision]'",
        ),
        ReadinessItem(
            name="yolo-obb checkpoint",
            status="ready" if yolo_path.exists() else "missing",
            detail=str(yolo_path),
            action=(
                "Train/acquire with: modal run tabvision/scripts/train/"
                "yolo_guitar_obb_modal.py, or set "
                f"{YOLO_CHECKPOINT_ENV}=<checkpoint.pt>"
            ),
        ),
        ReadinessItem(
            name="mediapipe hand-landmarker model",
            status="ready" if hand_path.exists() else "missing",
            detail=str(hand_path),
            action=(
                f"Download {HAND_MODEL_URL} to this path, or set "
                f"{HAND_MODEL_ENV}=<hand_landmarker.task>"
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acquire-models")
    parser.add_argument(
        "command",
        choices=["list", "status", "prepare-yolo-dir", "prepare-hand-dir"],
        help="model/dependency readiness command",
    )
    args = parser.parse_args(argv)

    if args.command == "list":
        print("Supported model/dependency groups:")
        print("  basic-pitch audio baseline — pip extra: .[audio-baseline]")
        print("  highres audio backend — pip extra: .[audio-highres]")
        print("  render extras — pip extra: .[render]")
        print("  vision dependencies — pip extra: .[vision]")
        print(f"  YOLO-OBB checkpoint — {yolo_checkpoint_path()}")
        print(f"  MediaPipe hand-landmarker model — {hand_model_path()}")
        return 0

    if args.command == "prepare-yolo-dir":
        target = yolo_checkpoint_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"YOLO checkpoint path: {target}")
        print(f"Place the trained checkpoint there, or set {YOLO_CHECKPOINT_ENV}=<checkpoint.pt>.")
        return 0

    if args.command == "prepare-hand-dir":
        target = hand_model_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"MediaPipe hand-landmarker model path: {target}")
        print(f"Download: {HAND_MODEL_URL}")
        print(f"Or set {HAND_MODEL_ENV}=<hand_landmarker.task>.")
        return 0

    if args.command == "status":
        failed = False
        for item in collect_status():
            print(f"{item.status:7} {item.name}: {item.detail}")
            if item.status != "ready":
                failed = True
                print(f"        {item.action}")
        return 1 if failed else 0

    return 2


def _module_item(
    *,
    name: str,
    modules: tuple[str, ...],
    action: str,
) -> ReadinessItem:
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if not missing:
        return ReadinessItem(
            name=name,
            status="ready",
            detail=", ".join(modules),
            action="No action needed.",
        )
    return ReadinessItem(
        name=name,
        status="missing",
        detail="missing " + ", ".join(missing),
        action=action,
    )


if __name__ == "__main__":
    raise SystemExit(main())
