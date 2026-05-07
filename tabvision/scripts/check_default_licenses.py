#!/usr/bin/env python3
"""Check that default runtime dependencies stay license-clean.

This scaffold intentionally checks only ``[project].dependencies``. Optional
extras such as ``vision`` and ``render`` have separate documented trade-offs in
LICENSES.md and should not silently move into the shipping default.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

BLOCKED_DEFAULT_PACKAGES = {
    "ultralytics": "AGPL-3.0 detector is optional/accepted, not a default dependency",
    "pyguitarpro": "LGPL render dependency must remain in the render extra",
    "basic-pitch": "audio baseline extra pulls model/runtime dependencies",
    "hf-midi-transcription": (
        "high-res backend must remain opt-in until Phase 2 license gate closes"
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="path to pyproject.toml",
    )
    args = parser.parse_args(argv)

    deps = _default_dependencies(args.pyproject)
    failures = []
    for dep in deps:
        name = _dependency_name(dep)
        if name in BLOCKED_DEFAULT_PACKAGES:
            failures.append(f"{name}: {BLOCKED_DEFAULT_PACKAGES[name]}")

    if failures:
        print("default dependency policy: FAIL")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("default dependency policy: PASS")
    print(f"checked {len(deps)} default dependencies")
    return 0


def _default_dependencies(pyproject: Path) -> list[str]:
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return list(data.get("project", {}).get("dependencies", []))


def _dependency_name(requirement: str) -> str:
    head = requirement.split(";", 1)[0].split("@", 1)[0]
    for separator in ("<", ">", "=", "!", "~", "["):
        head = head.split(separator, 1)[0]
    return head.strip().lower().replace("_", "-")


if __name__ == "__main__":
    sys.exit(main())
