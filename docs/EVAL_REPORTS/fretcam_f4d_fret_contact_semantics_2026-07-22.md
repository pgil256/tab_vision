# FretCam F4d — technique-aware fret contact semantics

**Date:** 2026-07-22

**Scope:** Explicitly approved, quarantined FretCam-only position fix

**Verdict:** PASS headlessly; L1/L2 remain pending live gates

## Failure reproduced

FretCam previously sent a continuous index-tip coordinate directly to the
nearest-fret position estimator. That assumes the fingertip represents a
contact near the centre of a fret cell. Guitarists instead fret immediately
behind the body-side wire, and a barre's angled fingertip can visually enter
the next cell even though the index shaft is correctly behind the prior wire.

Public clip `031_vpswc` demonstrates the issue: its first-position extended
index spans about 75–79% across the canonical neck while its continuous tip
coordinate reads 1.656–1.977. The old path therefore locked Position II even
though the across-neck contact line supports Position I.

## Bounded correction

- The FretCam MediaPipe adapter now retains the index MCP/PIP/DIP/tip axis in a
  FretCam-only `HandObservation`; no shared `tabvision/` type or §8 contract
  changed.
- Locking classifies a physical fret cell from calibrated fret-wire intervals,
  rather than rounding the continuous cell-centre coordinate.
- A barre correction is enabled only when the index is extended
  (`curl_ratio >= 0.85`), its distal axis spans at least 70% across the
  canonical neck, and its cross-neck span is at least 3× its along-neck span.
- For that confirmed barre only, the median PIP/DIP/tip coordinate is used and
  a 35%-of-local-cell deadband favors the fret immediately behind a wire. The
  fraction scales with real fret width instead of subtracting a fixed fret
  offset.
- Curled or partial-span index poses use exact wire-cell containment with no
  deadband. This preserves the user-verified Position II→VI move in
  `104_xf1wc`, whose high-position index spans only about 48–62% across the
  neck.
- `FrameDetection.index_fret` is now the discrete contact cell used for lock;
  `index_fret_raw` preserves the previous continuous fingertip coordinate for
  diagnostics. Replay overlays display both (`contact` and `tip-x`).

Rejected variants were retained as diagnostic reasoning, not shipped: a
global 35% deadband prevented the verified VI lock, while taking the nut-most
barre joint changed the same move to Position V.

No model, dependency, training, file under `tabvision/`, or main-pipeline
behavior changed.

## Verification

- Full FretCam suite: **44 passed** (one pre-existing Starlette deprecation
  warning).
- Ruff check and format check: passed for changed source and tests.
- Seven distinct cached public sources replayed at 10 FPS:

| clip | source window | accepted / frames | state counts | emitted positions |
|---|---:|---:|---|---|
| `023_Ypswc` | 2–8 s | 57 / 60 | 23 locked, 30 shifting, 3 holding, 4 acquiring | VI/VII/VIII |
| `031_vpswc` | 2–8 s | 42 / 60 | 27 locked, 10 holding, 8 lost, 7 shifting, 8 acquiring | I (34 frames), II (3) |
| `104_xf1wc` | 12–19 s | 49 / 70 | 32 locked, 12 holding, 9 lost, 7 shifting, 10 acquiring | II (27), VI (17) |
| `178_SM1wc` | 16–22 s | 39 / 60 | 16 locked, 17 shifting, 6 holding, 15 lost, 6 acquiring | II (18), V (4) |
| `238_My1wc` | 16–22 s | 60 / 60 | 43 locked, 17 acquiring | II only |
| `077_vV1wc` | 18–24 s | 5 / 60 | 55 lost, 5 acquiring | none |
| `105_Qf1wc` | 10–16 s | 2 / 60 | 58 lost, 2 acquiring | none |

The verified `104` II→VI counts exactly match F4c, and both F4c wrong-hand
clips still emit no position. The GAPS sources do not provide visual classical
position truth, so reclassifications in the other clips are technique-aware
sanity evidence rather than an A2 accuracy claim.

Generated visual diagnostics (local, not committed):

- `f4d_031_barre.mp4` / `.png`: Position I, contact 1, tip-x 1.83.
- `f4d_104_ii_vi.mp4` / `.png`: preserved Position II→VI replay.

## Gate interpretation

F4d removes the centre-rounding assumption and represents a confirmed barre by
its across-neck contact line without applying a universal one-fret correction.
Live F-barre holds across multiple positions and camera angles remain necessary
in A2 before this can be called generally accurate.
