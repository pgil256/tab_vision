"""Dataset acquisition — see SPEC.md §6.2.

Each subcommand fetches one dataset, verifies a checksum where possible,
and places it under ``$TABVISION_DATA_ROOT`` (defaults to
``~/.tabvision/data``). Idempotent — skips if already present.

Usage::

    # Download the YOLO-OBB guitar detector training set (Phase 3).
    # Requires ROBOFLOW_API_KEY env var.
    python -m scripts.acquire.datasets roboflow-guitar

    # List supported datasets.
    python -m scripts.acquire.datasets list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".tabvision" / "data"


def _data_root() -> Path:
    return Path(os.environ.get("TABVISION_DATA_ROOT", DEFAULT_DATA_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acquire-datasets")
    sub = parser.add_subparsers(dest="dataset", required=True)

    sub.add_parser("list", help="list supported datasets")

    rb = sub.add_parser(
        "roboflow-guitar",
        help="Roboflow b101/guitar-3 (YOLO-OBB training, Phase 3)",
    )
    rb.add_argument("--workspace", default="b101")
    rb.add_argument("--project", default="guitar-3")
    rb.add_argument("--version", type=int, default=3)
    rb.add_argument(
        "--format",
        default="yolov8-obb",
        help="export format; yolov8-obb is what we train on (oriented bboxes)",
    )

    args = parser.parse_args(argv)

    if args.dataset == "list":
        print("Supported datasets:")
        print("  roboflow-guitar — Roboflow b101/guitar-3 (Phase 3, YOLO-OBB)")
        return 0

    if args.dataset == "roboflow-guitar":
        return _acquire_roboflow_guitar(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            export_format=args.format,
        )

    parser.error(f"unknown dataset: {args.dataset}")
    return 2


def _acquire_roboflow_guitar(
    *,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
) -> int:
    target = _data_root() / "datasets" / f"roboflow-{workspace}-{project}-v{version}"
    if target.exists() and any(target.iterdir()):
        print(f"already present: {target}")
        print("(delete the directory to force re-download)")
        return 0

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "error: ROBOFLOW_API_KEY env var is required.\n\n"
            "How to get one:\n"
            "  1. Sign up free at https://roboflow.com\n"
            "  2. Settings → API → 'Private API Key'\n"
            "  3. export ROBOFLOW_API_KEY=...\n",
            file=sys.stderr,
        )
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "error: roboflow package not installed. "
            "Install with: pip install '.[vision]'",
            file=sys.stderr,
        )
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading roboflow {workspace}/{project} v{version} → {target}")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ver = proj.version(version)
    dataset = ver.download(export_format, location=str(target))

    license_info = getattr(ver, "license", None) or "unknown"
    citation = (
        f"Roboflow Universe project {workspace}/{project} v{version}, "
        f"accessed {dataset.location}"
    )
    print(f"\nattribution required:\n  {citation}\n  license: {license_info}")
    print(
        "Add the above to docs/HISTORY.md and to the repo README "
        "before merging Phase 3."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
