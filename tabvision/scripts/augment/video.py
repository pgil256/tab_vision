"""Phase 7 video augmentation scaffold for automated/optional labeled data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "script": "video",
        "phase": 7,
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "status": "ready" if args.dry_run else "optional_future",
        "inputs": {
            "frame_manifest": str(args.frame_manifest),
            "label_manifest": str(args.label_manifest),
        },
        "outputs": {
            "output_dir": str(args.output_dir),
            "report": str(args.output),
        },
        "steps": [
            "load automated or optional hand/fretboard frame labels",
            "apply deterministic crop, perspective, blur, and exposure variants",
            "transform available labels through the same image-space operations",
            "write augmented frame manifest for hand fine-tuning",
        ],
        "blockers": []
        if args.dry_run
        else ["full video augmentation requires non-interactive frame manifests"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", default="data/eval/frame_manifest.json")
    parser.add_argument("--label-manifest", default="data/eval/hand_labels.json")
    parser.add_argument("--output-dir", default="data/augmented/video")
    parser.add_argument("--output", type=Path, default=Path("video_augment_plan.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    payload = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
