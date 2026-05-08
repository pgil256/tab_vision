"""One-command eval runner.

Examples:
    python -m scripts.eval.run --scope full
    python -m scripts.eval.run --scope smoke --twice-and-diff
"""

from __future__ import annotations

from tabvision.eval.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
