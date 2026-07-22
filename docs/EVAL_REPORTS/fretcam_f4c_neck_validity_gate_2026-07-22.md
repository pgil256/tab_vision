# FretCam F4c — neck-validity gate before position locking

**Date:** 2026-07-22

**Scope:** Explicitly approved, quarantined FretCam-only geometry fix

**Verdict:** PASS headlessly; L1/L2 remain pending live gates

## Failure reproduced

Wide public shots could select the picking hand beside the soundhole. FretCam
then projected that hand through the board homography and allowed
`numpy.interp` or the rule-of-18 fallback to clamp off-board coordinates into
a plausible fret. Repeated observations could therefore lock Position I,
XII, or XXIV even though the fretting hand was elsewhere.

The `105_Qf1wc` diagnostic isolated the second wrong-hand path: the picking
index briefly entered the canonical neck at approximately `(0.058, 0.795)`,
but its wrist and three other fingertips were outside. An index-only inside
test was therefore insufficient even after boundary clipping was removed.

## Bounded correction

- Require at least three of the four fretting fingertips to project inside the
  canonical neck. The wrist is excluded because a genuine fretting wrist
  normally lies beyond the cross-string edge.
- Require the index landmark itself to be finite and inside the canonical
  neck before producing an index-fret observation.
- Bound calibrated interpolation to the physical outer edges of fret 1 and
  the configured last fret. Coordinates outside that support become invalid
  instead of being clamped to map endpoints.
- Reject rule-of-18 coordinates outside canonical `[0, 1]` instead of clipping
  them.
- Convert every failed check into a dropout: suppress hand markers, zero the
  anchor confidence, and pass `index_fret=None` to the temporal estimator.

No model, dependency, training, file under `tabvision/`, §8 contract, or main
pipeline behavior changed.

## Verification

- Full FretCam suite: **38 passed** (one pre-existing Starlette deprecation
  warning).
- Ruff check and format check: passed for changed source and tests.
- Five distinct public cached sources replayed at 10 FPS with the calibrated
  fret map locked on every frame:

| clip | source window | accepted / frames | state counts | emitted positions |
|---|---:|---:|---|---|
| `023_Ypswc` | 2–8 s | 57 / 60 | 22 locked, 31 shifting, 3 holding, 4 acquiring | VI/VII/VIII/IX only |
| `031_vpswc` | 2–8 s | 42 / 60 | 34 locked, 10 holding, 8 lost, 8 acquiring | II only |
| `104_xf1wc` | 12–19 s | 49 / 70 | 32 locked, 12 holding, 9 lost, 7 shifting, 10 acquiring | II and VI only |
| `077_vV1wc` | 18–24 s | 5 / 60 | 55 lost, 5 acquiring | none |
| `105_Qf1wc` | 10–16 s | 2 / 60 | 58 lost, 2 acquiring | none |

`023`, `031`, and the visually verified `104` Position II→VI move retain
position output. The two previously false-locking wide shots now emit no
position at all. This is a rejection/sanity gate, not position-accuracy ground
truth for GAPS; live A2 remains the accuracy acceptance test.

## Gate interpretation

F4c closes the reproduced off-neck and endpoint-saturation paths before the
temporal lock. It does not claim that every full-neck video will produce a
position: conservative dropout is the intended result when MediaPipe selects
the wrong hand or the calibrated geometry cannot support the observation.
