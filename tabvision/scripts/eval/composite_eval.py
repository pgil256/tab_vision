"""CLI wrapper for the v1 composite per-tier eval.

See ``docs/plans/2026-05-13-tab-f1-phase-0-implementation.md`` §3.4 for
the canonical invocation.
"""

from tabvision.eval.composite import main

if __name__ == "__main__":
    raise SystemExit(main())
