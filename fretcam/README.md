# FretCam

FretCam is a quarantined local prototype for TabVision's live fretboard and
playing-position HUD. The browser captures one in-memory JPEG at a time over a
localhost WebSocket. The local server returns neck/fret geometry, hand points,
a stable Roman-numeral position, confidence, and grounded framing guidance for
the browser to draw over the camera preview.

Nothing is recorded or persisted by the server.

## Run

```powershell
cd fretcam
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install --no-deps -e ..\tabvision
.venv\Scripts\fretcam
```

The second editable install exposes TabVision's existing vision modules as a
library without installing its unrelated audio/render extras. FretCam's own
package declares the pre-approved vision dependencies it uses.

Open <http://127.0.0.1:8765>, select **Start camera**, and grant camera
permission. A rear/environment camera is preferred when the browser exposes
one.

The page keeps the full camera preview separate from a bounded adaptive
inference canvas. It draws the tracked fretboard border and fret map over the
preview, then shows the elapsed-time-stabilized hand position beside it.
Camera selection, right/left-handed player mode, preview mirroring, optional
Position-I or two-point Position-I-plus-V/IX calibration, lock diagnostics,
and a session diagnostics export are available below the preview. The export
is capped at 300 timing/confidence samples and contains no frames, camera
identifiers, landmarks, or fretboard coordinates.

After the fretboard locks, hand acquisition immediately searches an enlarged
neck crop as well as the full frame. A narrower last-known-hand crop takes over
once a hand is acquired, with timestamped full-frame refreshes and optical-flow
landmark tracking between detector updates. Position evidence distinguishes
visible, hovering, and pressing fingers; uses distal finger-pad and barre
geometry; and abstains when the finger, board-freshness, or temporal evidence
does not agree.

## Verify

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python -m fretcam.benchmark --rounds 100
.venv\Scripts\python -m fretcam.replay_gaps
.venv\Scripts\python -m fretcam.replay_position
.venv\Scripts\python -m fretcam.benchmark_hud
.venv\Scripts\python -m fretcam.position_benchmark --split dev
.venv\Scripts\python -m fretcam.live_position_benchmark --list-conditions
```

The original benchmark retains F1's echo-only transport harness and round-trips
an in-memory synthetic JPEG. It does not access a camera or write image data to
disk. The HUD benchmark sends public cached GAPS frames through the complete
localhost WebSocket path and reports warm end-to-end throughput and latency.

The GAPS replay samples three public cached MP4s at 640 px, runs the F2 chain,
and prints a JSON gate report with neck/anchor outcomes and per-stage latency.
It reads from `~/.tabvision/cache/gaps_video/` and writes nothing.

The position replay samples six seconds of public `031_vpswc`, runs F2b + F3,
and writes a machine-local diagnostic MP4 and still under
`~/.tabvision/cache/fretcam_artifacts/`. These reproducible artifacts are not
committed and are not position-accuracy evidence.

The F4e-A position benchmark reads the frozen public-only annotations in
`data/position_benchmark_v1.json`. It reports displayed-position precision,
coverage, false locks, shift latency, dropout recovery, negative-control
display rate, category breakdowns, raw confidence factors, blocker incidence,
and geometry freshness. Use only the `dev` split while choosing rules or
thresholds; the source-disjoint `test` split is reserved for the final
comparison. The benchmark reads cached GAPS media but never commits or
redistributes it.

The live-position regression harness reuses those labels but sends every JPEG
through the real localhost uvicorn/FastAPI WebSocket route. Its bounded
15-condition `coverage` matrix exercises 2/5/10/20 FPS, JPEG quality
50/72/90, 320/480/640 px inference inputs, native/bright/dim/warm/cool/uneven
lighting, temporary occlusion, and camera motion. Replays are paced so the
time-based estimator sees the requested cadence. Full predictions are written
only when `--output-json` is supplied; put that optional artifact under
`~/.tabvision/cache/fretcam_artifacts/`.

```powershell
.venv\Scripts\python -m fretcam.live_position_benchmark `
  --split dev `
  --output-json "$HOME\.tabvision\cache\fretcam_artifacts\live-position-dev.json"
```

See [docs/live-position-benchmark.md](docs/live-position-benchmark.md) for
matrix selection, deterministic perturbation semantics, and focused-run
examples.

The implementation and final development-only accuracy evidence for
neck-guided acquisition, temporal landmarks, physical contacts, upper-neck
geometry, and two-point calibration are summarized in
[docs/accuracy-phase-2.md](docs/accuracy-phase-2.md).

## Optional local finger-evaluation set

`fretcam-local-eval` manages a locally cached development set for positions
I-XII and per-finger contact labels. Its default root is
`~/.tabvision/cache/fretcam_local_eval/`; roots inside any Git repository are
rejected. The command never opens a camera, records video, or extracts frames.
It only registers public-licensed or reproducibly synthetic media manually
placed under the local dataset root. Private and user recordings are rejected
in every evaluation, annotation, tuning, and training role.

The manifest validates source URLs, licenses, media hashes, five-finger
visibility/pressing/string/fret labels, and coverage across techniques,
framing, handedness, lighting, and capture diversity. See
[the local evaluation-set guide](docs/local-eval-dataset.md) for the schema
and CLI workflow.
