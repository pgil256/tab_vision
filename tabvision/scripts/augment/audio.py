"""Phase 7 audio augmentation scaffold.

Dry-run mode is deterministic and writes the exact plan that a GPU/data runner
can execute later. Full augmentation is optional until automated manifests and
IR/noise assets are supplied.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "script": "audio",
        "phase": 7,
        "dry_run": bool(args.dry_run),
        "seed": int(args.seed),
        "status": "ready" if args.dry_run else "optional_future",
        "inputs": {
            "manifest": str(args.manifest),
            "ir_dir": str(args.ir_dir),
            "noise_dir": str(args.noise_dir),
        },
        "outputs": {
            "output_dir": str(args.output_dir),
            "report": str(args.output),
        },
        "steps": [
            "load automated/public audio manifest",
            "apply deterministic gain, EQ, room IR, and distortion variants",
            "write augmented clips with onset-aligned labels",
            "emit augmentation manifest for fine-tuning",
        ],
        "blockers": []
        if args.dry_run
        else ["full audio augmentation requires automated manifests and augmentation assets"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/eval/manifest.toml")
    parser.add_argument("--ir-dir", default="data/augmentation/irs")
    parser.add_argument("--noise-dir", default="data/augmentation/noise")
    parser.add_argument("--output-dir", default="data/augmented/audio")
    parser.add_argument("--output", type=Path, default=Path("audio_augment_plan.json"))
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
