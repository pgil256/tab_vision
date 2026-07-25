# TabVision Release Evidence

This page collects existing/generated artifacts that support the portfolio
narrative without manual annotation, new recordings, or private media.

## Headline accuracy

- `../EVAL_REPORTS/player05_batched_confirm_2026-07-24.md` — **the current
  default's numbers.** Held-out player-05: single-line 0.7257, strummed
  0.7435, aggregate 0.7346 (+0.1006 [+0.0615, +0.1416]). Also the run that
  refuted the +0.60 level correction and confirmed partial-aware isolation.
- `../EVAL_REPORTS/v1_acceptance_2026-06-03.md` — the **v1.0.0 acceptance
  record** (aggregate 0.600). Historical: the configuration the tag was cut
  against, not the current default.

## Where the negatives are

The refutations are as much of the evidence as the wins; these are the ones
the narrative cites directly.

- `../EVAL_REPORTS/fretcam_e2e_source_disjoint10_2026-07-24.md` — video
  end-to-end on source-disjoint clips: +0.000836, CI lower bound 0. Why
  FretCam ships opt-in.
- `../EVAL_REPORTS/fretcam_gaps_anchor_probe_calibrated_2026-07-22.md` — the
  corrected anchor probe (0.763) that superseded the earlier 0.285
  "anti-enriched" result, which was a fret-mapping bug.
- `../EVAL_REPORTS/n5_table_mismatch_2026-07-24.md` — the physics table
  survives real-guitar mismatch across 17 pre-declared perturbations.
- `../EVAL_REPORTS/a14_video_complementarity_2026-07-06.md` and
  `../EVAL_REPORTS/v1_1_gaps_chunk6_ws1_2026-06-25.md` — the original video
  work, kept as measured evidence.

## Runbooks and fixtures

- `fresh-user-path.md` — reproducible fresh-clone CLI runbook using the
  checked-in A440 fixture and optional Basic Pitch extra.
- `sample-a440-ascii.tab` — generated ASCII render for the checked-in A440
  fixture expectation; also the desktop shell's bootstrap smoke golden.
- `../EVAL_REPORTS/eval_full_20260507T000000Z.md` — eval harness report.
  Manual Phase 1.5/3/4 gates are listed as `optional_future`, not v1 blockers.
- `../EVAL_REPORTS/phase5_position_prior_2026-05-07.md` — GuitarSet
  high-resolution audio evidence for the `guitarset-v1` position prior.

## Release Checks

Run from `tabvision/`:

```bash
.venv/bin/python -m scripts.eval.run --scope smoke --twice-and-diff --output-dir /tmp/tabvision-eval-smoke
.venv/bin/python scripts/check_default_licenses.py --pyproject pyproject.toml
.venv/bin/python -m scripts.acquire.models list
bash scripts/test_fresh_install.sh
```

The license gate now checks **two** policies — dependencies and loaded default
artifacts. The artifact half matters more than it used to, because the default
path loads model artifacts with mixed licenses (see `../../LICENSES.md`).

Suite counts and gates, verified on `main` 2026-07-25:

- `pytest tests/unit` in `tabvision/`: **1103 passed, 3 skipped**.
- `pytest tests` in `fretcam/`: **240 passed, 1 skipped, 5 subtests**.
- `dotnet test` in `desktop-client/`: **54 passed**.
- `ruff check .`, `ruff format --check .` (314 files), `mypy tabvision`
  (84 sources): all clean.
- License gate: `default dependency policy: PASS`,
  `default artifact policy: PASS`.

Historical, 2026-05-07 (not re-run since):

- Smoke eval: `deterministic=true`, `smoke_budget_s=180`.
- Fresh install: package installs in a clean clone, `tabvision --version`
  works, render smoke reports `2 passed, 10 skipped`.
- Fixture transcription: fresh Python 3.11 venv with `.[audio-baseline]`
  transcribes `data/fixtures/test_a440.mp4` to `sample-a440-ascii.tab` shape
  (A440 as high-E string, fret 5).

The desktop shell has its own clean-install gate, recorded in
`../DECISIONS.md` (2026-07-22, host-isolated clean-install acceptance).

## Demo Asset Policy

Use checked-in fixtures, generated reports, screenshots, and small derived
media. Do not add large raw clips to this directory, and do not make
hand-labeled user-video examples part of the v1 release gate.
