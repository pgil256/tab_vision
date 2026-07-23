# FretCam F5 quality overhaul — 2026-07-23

## Outcome

F5 is complete inside the quarantined `fretcam/` prototype. The build now
combines all fretting-finger contact evidence, tracks fretboard geometry
between detector passes, locks by elapsed evidence time, abstains through a
composite confidence gate, bounds/adapts per-frame work, and exposes the
requested browser controls and diagnostics.

No dependency, model, download, training run, private recording, §8 contract,
or `tabvision/` package behavior changed.

## Frozen development benchmark

The final rules were checked on the frozen public-only development split:

| Metric | F4d baseline | F5 final |
|---|---:|---:|
| Displayed-position precision | 0.825 | 76/80 = 0.950 |
| Stable coverage | 0.323 | 71/161 = 0.441 |
| Stable false-lock rate | 0.062 | 0/161 = 0.000 |
| Valid-observation rate | — | 85/161 = 0.528 |
| Negative-control display rate | ≤0.031 | 0/120 = 0.000 |

Machine-local result:
`~/.tabvision/cache/fretcam_artifacts/f5-quality-dev-final-current.json`.

The source-disjoint test split was opened once before the final lifecycle and
state-machine hardening: precision 12/13, coverage 13/115, false locks 1/115,
and negative displays 0/120. It was not reopened or used for threshold/rule
selection, so that result is retained only as the single-opening audit
snapshot rather than claimed as a final-code measurement.

## Verification

- 106 FretCam tests passed. The only warning is Starlette's existing
  `TestClient`/`httpx` deprecation warning.
- Ruff checks passed, changed Python files are formatted, JavaScript syntax
  passed, and the diff has no whitespace errors.
- A real localhost Chrome session with a synthetic camera verified:
  pre-permission camera discovery, start and three consecutive restarts,
  handedness, mirroring, calibration start/timeout/reset, diagnostics export,
  desktop/mobile layout, and the WebSocket HUD.
- The browser run produced no console or page errors. Observed smoke values
  were 18.8–29.5 HUD FPS and 18.0–34.6 ms end-to-end. These synthetic-camera
  numbers verify the product path; they do not replace real-camera acceptance.
- The diagnostics export stayed within its 300-sample bound and contained
  scalar/status/timing data only—no frames, camera identifiers, landmarks, or
  fretboard coordinates.

## Implemented safeguards

- Background detector work never runs on the live response path; reconnects
  retire obsolete jobs without waiting.
- Delayed first detections can acquire on texture-poor but unchanged frames.
- LK points are reseeded after failed flow, and both detector and flow updates
  reject canonical-axis flips and implausible geometry jumps.
- Weak visible-hand evidence reports `Acquiring` rather than `No hand`.
- Unconfirmed large jumps and low temporal agreement cannot refresh or publish
  an old confident lock.
- Position-I calibration uses a continuous contact residual and completes at
  the supported 2–8 FPS range with stable evidence.
