# 2026-05-12 Session Handoff

> **SUPERSEDED (2026-07-09):** the production backend URL below is stale. The
> `pgil256` Modal workspace is orphaned (no local credentials) and its app runs
> a pre-fix May image. Production now points at
> `https://pgilhooley95--tabvision-api-flask-app.modal.run` (Modal workspace
> `pgilhooley95`), deployed from `main`. See `docs/DECISIONS.md`
> ("2026-07-09 — production backend repointed").

## Current Focus

TabVision production now uses Modal for the Flask API/runtime and Vercel for
the Vite frontend. The production transcription path has been moved to the
newer `tabvision` v1 pipeline with the high-resolution audio backend and the
checked-in `guitarset-v1` position prior.

Important correction from the user: do not use the 20 personal training videos
as the primary accuracy gate. Their labels are not reliable enough for tuning.
Use GuitarSet held-out validation as the accuracy gate.

## Live Production State

- Frontend: `https://tabvision.patbuilds.dev`
- Modal API: `https://pgil256--tabvision-api-flask-app.modal.run`
- Modal app: `tabvision-api`
- Vercel production deployment: `dpl_4GvPYTm1uJBQFqLCjrrvpguXrv1X`
- Live custom-domain JS asset checked after deploy:
  - `/assets/index-lvg42Y9C.js`
  - Does not contain `http://localhost:5000`
  - Does contain `pgil256--tabvision-api-flask-app.modal.run`

Production Modal defaults:

- `TABVISION_PIPELINE=v1`
- `TABVISION_AUDIO_BACKEND=highres`
- `TABVISION_POSITION_PRIOR=guitarset-v1`
- `TABVISION_VIDEO_ENABLED=false`
- `TABVISION_MELODIC_PRIOR_ENABLED=false`
- `TABVISION_FALLBACK_AUDIO_BACKEND=none`

Basic Pitch is not part of the current Modal production worker path. It remains
in the repo for the legacy `legacy_v0` path and local/tests/history.

## What Changed

- Added Modal production entrypoint:
  - `tabvision-server/modal_app.py`
- Added Modal durable storage helpers:
  - `tabvision-server/app/modal_storage.py`
  - `tabvision-server/app/result_io.py`
- Added the Flask-to-v1 adapter:
  - `tabvision-server/app/v1_adapter.py`
- Refactored backend processing enough for Modal:
  - `create_app(config_overrides=...)`
  - configured job storage/dispatcher/result hooks
  - durable `Job.to_record()` / `Job.from_record()`
  - per-stage `storage.save(job)` updates
  - lazy legacy imports so the highres worker does not import TensorFlow,
    Basic Pitch, OpenCV, or MediaPipe unless legacy v0 is selected
- Promoted v1 highres + `guitarset-v1` as the production candidate.
- Added guided upload fields in the frontend:
  - instrument, tone, style, accuracy mode, ROI coordinates
- Added frontend production API guards:
  - no production `localhost:5000`
  - Vercel build fails if `VITE_API_URL` is missing
- Removed stale Railway-only deployment files.

## Accuracy Gate

Use this report as the current source of truth:

- `docs/EVAL_REPORTS/guitarset_accuracy_boost_2026-05-08.md`

Fresh GuitarSet validation evidence:

| Condition | Onset F1 | Pitch F1 | Tab F1 |
| --- | ---: | ---: | ---: |
| Highres, no prior | `0.9218` | `0.9022` | `0.3878` |
| Highres, `guitarset-v1` | `0.9218` | `0.9022` | `0.6104` |

Delta: `+22.26 pp` Tab F1 with no onset/pitch regression.

Melodic prior note: the experimental melodic-segment prior is implemented but
disabled by default. It helped a personal fast-scale clip but regressed the
GuitarSet aggregate Tab F1 from `0.6104` to `0.5989`.

## Production Smoke

Do not describe this as an accuracy benchmark; it is only a live API smoke.

- Source audio: GuitarSet `05_Rock1-90-C#_comp_mic.wav`
- Fixture: `/tmp/tabvision-guitarset-05_Rock1-90-Csharp_comp_12s.mp4`
- Endpoint: `https://pgil256--tabvision-api-flask-app.modal.run`
- Job: `5e0b8da3-fc3b-48e7-a0ab-5505524d7ac5`
- Result: completed
- Result metadata:
  - `pipeline=v1`
  - `backend=highres`
  - `prior=guitarset-v1`
  - `video=false`
  - `fallbackUsed=false`

## Verification At Handoff

Last verification commands completed successfully in this workspace:

```bash
cd tabvision-server
./venv/bin/python -m pytest
```

Result: `269 passed, 1 skipped, 4 warnings`.

```bash
cd tabvision
../tabvision-server/venv/bin/python -m pytest \
  tests/unit/test_audio_filters.py \
  tests/unit/test_guitarset_audio_eval.py \
  tests/unit/test_melodic_prior.py \
  tests/unit/test_pipeline.py \
  tests/unit/test_position_prior.py
```

Result: `51 passed`.

```bash
cd web-client
export VERCEL=1 VERCEL_ENV=production VITE_API_URL=https://api.example.test
node scripts/require-vercel-api-url.mjs
node node_modules/typescript/bin/tsc --noEmit
node scripts/require-vercel-api-url.test.mjs
node node_modules/vite/bin/vite.js build
node scripts/assert-prod-api-config.mjs
```

Result: production frontend build and guards passed.

Live checks:

```bash
curl -fsS https://pgil256--tabvision-api-flask-app.modal.run/health
```

Result: `{"status":"ok"}`.

CORS preflight for `https://tabvision.patbuilds.dev` returned
`access-control-allow-origin: https://tabvision.patbuilds.dev`.

## Basic Pitch Status

The user asked why Basic Pitch is still around now that highres is better.
Current answer:

- Production Modal path is not using Basic Pitch.
- Basic Pitch is still present for legacy rollback and old tests.
- `V1PipelineConfig` now defaults `fallback_audio_backend` to `None`.
  Basic Pitch fallback is opt-in via `TABVISION_FALLBACK_AUDIO_BACKEND=basicpitch`
  or an explicit `V1PipelineConfig(fallback_audio_backend="basicpitch")`.

## Next Best Steps

1. Decide whether to keep a separate Basic Pitch Modal fallback image. Do not
   install Basic Pitch and highres in the same worker image; their `resampy`
   requirements conflict.
2. Commit the current production/accuracy work or open a PR before more
   accuracy experimentation. The worktree is dirty with a large scoped change.
3. Run a full browser E2E on `https://tabvision.patbuilds.dev` after the next
   frontend tweak if browser tooling is available.
4. Continue accuracy work only through GuitarSet-backed reports. Personal clips
   can be used as qualitative smoke tests, not as acceptance gates.

## Worktree Note

The worktree is intentionally dirty at this handoff. Do not assume every dirty
file belongs to one atomic change; inspect before staging. The main active
change set is the Modal production backend, v1 adapter, GuitarSet accuracy
gate, frontend API hardening, and guided upload UI.
