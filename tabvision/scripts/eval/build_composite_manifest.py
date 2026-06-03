"""CLI wrapper for the composite-eval manifest builder.

See ``docs/plans/2026-05-13-tab-f1-phase-0-implementation.md`` §3.3 for
the canonical invocation.
"""

from tabvision.eval.manifest_builder import main

if __name__ == "__main__":
    raise SystemExit(main())
