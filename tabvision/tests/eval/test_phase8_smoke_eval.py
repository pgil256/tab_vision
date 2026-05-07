"""Phase 8 CI smoke eval.

This test uses the deterministic smoke scope so it is safe for default CI:
it does not require external datasets or heavy model dependencies, but it
does exercise the same report writer used by the full eval command.
"""

from __future__ import annotations

import time

from tabvision.eval.runner import run_eval


def test_phase8_smoke_eval_is_deterministic_and_under_budget(tmp_path) -> None:
    manifest = tmp_path / "missing-manifest.toml"
    started = time.perf_counter()

    first = run_eval(
        manifest_path=manifest,
        output_dir=tmp_path / "run-a",
        scope="smoke",
        seed=0,
        timestamp="2026-05-07T00:00:00Z",
    )
    second = run_eval(
        manifest_path=manifest,
        output_dir=tmp_path / "run-b",
        scope="smoke",
        seed=0,
        timestamp="2026-05-07T00:00:00Z",
    )

    elapsed_s = time.perf_counter() - started
    assert first.json_bytes == second.json_bytes
    assert first.markdown == second.markdown
    assert elapsed_s < 180.0
