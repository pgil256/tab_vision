# Phase 5 Pitch-Position Prior Decision

Date: 2026-05-07

## Summary

The pitch-position prior is productionized as an explicit option:

```bash
tabvision transcribe input.mov --position-prior guitarset-v1
```

Default behavior remains:

```bash
tabvision transcribe input.mov --position-prior none
```

The checked-in artifact is
`tabvision/tabvision/fusion/priors/guitarset_v1.json`; raw GuitarSet files are
not required at runtime.

## Existing Evidence

Full GuitarSet validation highres run from 2026-05-07:

| Condition | Onset F1 | Pitch F1 | Tab F1 |
| --- | ---: | ---: | ---: |
| No prior | 0.9218 | 0.9022 | 0.3878 |
| GuitarSet train-split prior | 0.9218 | 0.9022 | 0.6104 |

Delta: `+22.26 pp` Tab F1. Per-track result: 51/60 improved, 8/60 regressed,
1/60 unchanged.

## Optional Future Home-Video Prior On/Off Benchmark

Prepared command shape:

```bash
pytest -m eval -k phase5 --ablation
tabvision transcribe <home_clip.mov> --position-prior none
tabvision transcribe <home_clip.mov> --position-prior guitarset-v1
```

v1 policy: this home-video ablation is `optional_future`, not a release
blocker. The worktree does not need held-out home-video media, manual labels,
YOLO checkpoint, MediaPipe model, or highres audio dependencies to pass the
remaining v1 gates.

Local command result in this worktree:

```text
../venv/bin/python -m pytest -m eval -k phase5 --ablation -q
sss [100%]
10 skipped, 228 deselected
```

Phase 7 command result in this worktree:

```text
../venv/bin/python -m pytest -m eval -k phase7 -q
s [100%]
8 skipped, 230 deselected
```

## Decision

Keep `guitarset-v1` optional. Promote only after the home-video ablation shows
no regression, or after equivalent automated/public evidence shows no
regression and the remaining GuitarSet regressions are accepted or reduced.
