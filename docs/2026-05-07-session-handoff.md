# 2026-05-07 Session Handoff

## Current Focus

Phase 5 is blocked on trustworthy audio-to-tab mapping before any more
video-calibration or `lambda_vision` tuning.

The key finding from today is that highres audio is strong at GuitarSet
onset/pitch, but default pitch-to-string/fret decoding is weak. A learned
GuitarSet train-split position prior improves Tab F1 substantially, but
it should not become an unconditional production default yet.

## What Changed Today

- Added a raw GuitarSet audio-only evaluator:
  - `tabvision/tabvision/eval/guitarset_audio.py`
  - `tabvision/scripts/eval/guitarset_audio_eval.py`
  - `tabvision/tests/unit/test_guitarset_audio_eval.py`
- Added learned pitch-position prior support:
  - `tabvision/tabvision/fusion/position_prior.py`
  - `tabvision/tests/unit/test_position_prior.py`
- Added a Modal L4 GPU runner for full highres eval:
  - `tabvision/scripts/eval/guitarset_highres_modal.py`
- Recorded the Phase 5 position-prior decision in `docs/DECISIONS.md`.

The existing GuitarSet TFRecords were inspected and are insufficient for
Tab F1 because they do not retain string/fret labels. Raw JAMS/WAV is the
correct source for Tab F1.

## Metric Snapshot

Full GuitarSet validation split is player `05` (60 tracks).

Modal L4 highres eval, full validation:

| Run | Onset F1 | Pitch F1 | Tab F1 |
| --- | ---: | ---: | ---: |
| no position prior | `0.9218` | `0.9022` | `0.3878` |
| GuitarSet train prior | `0.9218` | `0.9022` | `0.6104` |

Delta from prior: `+22.26 pp` Tab F1, with onset/pitch unchanged.

Per-track effect: 51/60 improved, 8/60 regressed, 1/60 unchanged. Mean
track Tab F1 moved from `0.347` to `0.589`.

Reports:

- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-none-2026-05-07.md`
- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-guitarset-train-2026-05-07.md`

## Promotion Decision

Do not make the GuitarSet train prior an unconditional production default
yet.

Recommended path:

1. Promote the position prior as a versioned/configured production option.
2. Create a checked-in prior artifact or generator so raw GuitarSet is not
   required at runtime.
3. Classify the 8 validation regressions, especially the SS/comp cases
   where no-prior was already strong.
4. Run the home-video Phase 5 benchmark with and without the prior.
5. Make it the default only if the home-video benchmark has no regression
   and the GuitarSet regressions are understood or accepted.

## Modal State

The Modal GPU path is now working.

Useful commands:

```bash
tabvision-server/venv/bin/modal run tabvision/scripts/eval/guitarset_highres_modal.py --limit 1
tabvision-server/venv/bin/modal run tabvision/scripts/eval/guitarset_highres_modal.py
tabvision-server/venv/bin/modal run tabvision/scripts/eval/guitarset_highres_modal.py --position-prior none
```

The Modal volume `tabvision-guitarset` has been hydrated with 360 raw
GuitarSet JAMS/WAV tracks from `taohu/guitarset`.

## Verification At Handoff

Last full fast suite:

```bash
cd tabvision
.venv/bin/python -m pytest -q
```

Result: `249 passed, 9 skipped`.

Focused checks after the eval/Modal changes:

```bash
cd tabvision
.venv/bin/python -m pytest tests/unit/test_guitarset_audio_eval.py tests/unit/test_position_prior.py -q
.venv/bin/python -m ruff check tabvision/eval/guitarset_audio.py tabvision/fusion/position_prior.py scripts/eval/guitarset_audio_eval.py scripts/eval/guitarset_highres_modal.py tests/unit/test_guitarset_audio_eval.py tests/unit/test_position_prior.py
```

Result: 11 focused tests passed; ruff passed.

## Worktree Note

The worktree is still dirty. Some dirty files predate this handoff. The
most relevant new files from this session are:

- `tabvision/tabvision/eval/guitarset_audio.py`
- `tabvision/tabvision/fusion/position_prior.py`
- `tabvision/scripts/eval/guitarset_audio_eval.py`
- `tabvision/scripts/eval/guitarset_highres_modal.py`
- `tabvision/tests/unit/test_guitarset_audio_eval.py`
- `tabvision/tests/unit/test_position_prior.py`
- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-none-2026-05-07.{md,csv}`
- `tabvision-server/tools/outputs/guitarset_audio_eval-highres-validation-guitarset-train-2026-05-07.{md,csv}`

There are also Phase 5 eval/fusion/video diagnostics and prior Phase 5
implementation changes in the worktree. Do not assume every dirty file
belongs to the GuitarSet work.

## Next Best Step

Build the production integration for the position prior as an explicit
config option/artifact, then run home-video Phase 5 with prior on/off.

Do not proceed to Phase 6 until Phase 5 is recorded and the user explicitly
says `proceed`.
