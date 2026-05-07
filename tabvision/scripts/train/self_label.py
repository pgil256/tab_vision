"""Phase 7 self-labeling scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "script": "self_label",
        "phase": 7,
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "status": "ready" if args.dry_run else "blocked",
        "inputs": {
            "unlabeled_manifest": str(args.unlabeled_manifest),
            "audio_checkpoint": str(args.audio_checkpoint),
            "hand_checkpoint": str(args.hand_checkpoint),
            "min_confidence": args.min_confidence,
        },
        "outputs": {
            "output_manifest": str(args.output_manifest),
            "report": str(args.output),
        },
        "steps": [
            "run current audio and video models on unlabeled home clips",
            "keep only agreement labels above the confidence threshold",
            "write pseudo-label manifest with provenance and seed metadata",
            "compare next-round eval deltas against the stop condition",
        ],
        "blockers": []
        if args.dry_run
        else ["self-labeling requires unlabeled home clips and trained checkpoints"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlabeled-manifest", default="data/eval/unlabeled.toml")
    parser.add_argument("--audio-checkpoint", default="data/augmented/checkpoints/audio/latest")
    parser.add_argument("--hand-checkpoint", default="data/augmented/checkpoints/hand/latest")
    parser.add_argument("--output-manifest", default="data/augmented/self_label/manifest.json")
    parser.add_argument("--output", type=Path, default=Path("self_label_plan.json"))
    parser.add_argument("--min-confidence", type=float, default=0.85)
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
