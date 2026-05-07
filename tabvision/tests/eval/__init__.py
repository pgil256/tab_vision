"""Optional Phase 3 + Phase 4 manual-validation harnesses.

These are gated (``-m guitar_eval`` / ``-m preflight_eval`` /
``-m fretboard_eval`` / ``-m hand_eval``) so they don't run in the
default unit-test pass and are only invoked when labeled fixtures are
available under ``$TABVISION_EVAL_ROOT`` or ``tabvision/data/eval/``.

Each harness skips itself with a helpful message if the corresponding
labels haven't been collected yet. Those skips are expected for v1; the
manual labels are future validation data, not release blockers.
"""
