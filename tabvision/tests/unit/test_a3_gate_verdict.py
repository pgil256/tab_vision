"""A3 gate probe — the two-bar verdict logic.

The lower-95 CI bar always gates. Per-clip no-regression is a HARD bar only for
the GAPS clean-12 cross-domain leg (``strict_per_clip=True``); on the in-domain
GuitarSet 60-clip confirm it is informational, so a few clips regressing while
the aggregate lower-95 holds is a PASS (this was the mislabel the earlier gate
run hit — GuitarSet single-line 0.4570→0.4627, strummed 0.6058→0.6175 lower-95,
both up, with 5/60 per-clip regressions).
"""

from __future__ import annotations

import pytest

from scripts.eval.a3_gate_probe import gate_passed


@pytest.mark.parametrize(
    ("lower95_held", "n_regressions", "strict", "expected"),
    [
        # Lower-95 regressed -> always FAIL, regardless of the per-clip count / mode.
        (False, 0, False, False),
        (False, 0, True, False),
        (False, 5, False, False),
        # In-domain confirm (not strict): lower-95 held -> PASS even with regressions.
        (True, 0, False, True),
        (True, 5, False, True),  # the GuitarSet 60-clip case: 5 regressions, still PASS
        (True, 30, False, True),
        # GAPS cross-domain (strict): any per-clip regression -> FAIL.
        (True, 0, True, True),
        (True, 1, True, False),
        (True, 5, True, False),
    ],
)
def test_gate_passed(lower95_held: bool, n_regressions: int, strict: bool, expected: bool) -> None:
    assert gate_passed(lower95_held, n_regressions, strict) is expected


def test_guitarset_60clip_case_passes() -> None:
    """The exact situation the earlier run mislabeled FAIL: both tiers' lower-95
    improved, 5/60 per-clip regressions, in-domain (not strict) -> PASS."""
    assert gate_passed(lower95_held=True, n_regressions=5, strict_per_clip=False) is True
