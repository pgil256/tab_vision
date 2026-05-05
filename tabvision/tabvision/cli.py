"""TabVision CLI entry point — see SPEC.md §3.3, §7 Phase 1.

Phase 1 deliverable: ``tabvision transcribe input.mov -o output.tab``.
Phase 3 adds ``tabvision check input.mov`` for preflight only.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Phase 0 stub — does nothing useful yet."""
    parser = argparse.ArgumentParser(prog="tabvision")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_subparsers(dest="command", required=False)
    args = parser.parse_args(argv)

    if args.version:
        from tabvision import __version__

        print(f"tabvision {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
