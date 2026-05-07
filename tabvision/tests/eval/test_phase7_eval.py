"""Phase 7 accuracy-work acceptance placeholder.

Full Phase 7 eval requires trained/fine-tuned checkpoints and the held-out
home-video eval set. The local CPU test path verifies that the acceptance
command collects cleanly and reports the blocker explicitly.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.eval
def test_phase7_full_accuracy_eval_requires_data_and_gpu():
    if os.environ.get("TABVISION_RUN_PHASE7_EVAL") != "1":
        pytest.skip(
            "Phase 7 full eval requires held-out home-video data plus GPU-trained "
            "audio/hand checkpoints; scaffold dry-run tests cover local readiness."
        )

    pytest.fail(
        "TABVISION_RUN_PHASE7_EVAL=1 was set, but the full Phase 7 training/eval "
        "runner is not implemented in this worktree."
    )
