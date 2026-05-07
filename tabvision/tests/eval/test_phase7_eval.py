"""Phase 7 automated accuracy-work placeholder.

Full Phase 7 eval requires trained/fine-tuned checkpoints and non-interactive
automated/public eval data. The local CPU test path verifies that the command
collects cleanly without requiring manual labels.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.eval
def test_phase7_full_accuracy_eval_requires_data_and_gpu():
    if os.environ.get("TABVISION_RUN_PHASE7_EVAL") != "1":
        pytest.skip(
            "Phase 7 full eval requires automated/public eval data plus GPU-trained "
            "audio/hand checkpoints; scaffold dry-run tests cover local readiness."
        )

    pytest.fail(
        "TABVISION_RUN_PHASE7_EVAL=1 was set, but the full Phase 7 training/eval "
        "runner is not implemented in this worktree."
    )
