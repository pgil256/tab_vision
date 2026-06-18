# TabVision Release Evidence

This page collects existing/generated artifacts that can support a Phase 9
portfolio demo without manual annotation, new recordings, or private media.

## Automated Reports

- `fresh-user-path.md` — reproducible fresh-clone CLI runbook using the
  checked-in A440 fixture and optional Basic Pitch extra.
- `sample-a440-ascii.tab` — generated ASCII render for the checked-in A440
  fixture expectation.
- `../EVAL_REPORTS/eval_full_20260507T000000Z.md` — current eval harness
  report. Manual Phase 1.5/3/4 gates are listed as `optional_future`, not v1
  blockers.
- `../EVAL_REPORTS/phase5_position_prior_2026-05-07.md` — GuitarSet
  high-resolution audio evidence for the optional `guitarset-v1` position
  prior.
- `../EVAL_REPORTS/v1_acceptance_2026-06-03.md` - accepted v1 public/fixture
  evidence.
- `../EVAL_REPORTS/v1_1_chunk3_real_video_robustness_2026-06-11.md` - current
  v1.1 real-video robustness evidence.
## Release Checks

Run from `tabvision/`:

```bash
.venv/bin/python -m scripts.eval.run --scope smoke --twice-and-diff --output-dir /tmp/tabvision-eval-smoke
.venv/bin/python scripts/check_default_licenses.py --pyproject pyproject.toml
.venv/bin/python -m scripts.acquire.models list
bash scripts/test_fresh_install.sh
```

Latest local results, 2026-05-07:

- Smoke eval: `deterministic=true`, `smoke_budget_s=180`.
- License gate: `default dependency policy: PASS`, `checked 2 default dependencies`.
- Fresh install: package installs in a clean clone, `tabvision --version`
  works, render smoke reports `2 passed, 10 skipped`.
- Fixture transcription: fresh Python 3.11 venv with `.[audio-baseline]`
  transcribes `data/fixtures/test_a440.mp4` to `sample-a440-ascii.tab` shape
  (A440 as high-E string, fret 5).

## Demo Asset Policy

Use checked-in fixtures, generated reports, screenshots, and small derived
media. Do not add large raw clips to this directory, and do not make
hand-labeled user-video examples part of the v1 release gate.
