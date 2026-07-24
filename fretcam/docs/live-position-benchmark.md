# Live-position accuracy regression harness

`fretcam.live_position_benchmark` measures position accuracy through the same
localhost WebSocket route used by the browser. It reads only source names
declared by the checked-in public GAPS manifest, loads their existing MP4s from
the machine-local cache, encodes in-memory JPEGs, and sends one frame at a time
to a real uvicorn/FastAPI server. It does not download media, save frames, or
record camera data.

## Default coverage suite

The default `coverage` matrix changes one axis at a time around the browser
baseline of 10 FPS, JPEG quality 72, 640 px, native lighting, and no synthetic
perturbation. This produces 15 conditions:

- cadence: 2, 5, 10, and 20 FPS;
- JPEG quality: 50, 72, and 90;
- longest inference dimension: 320, 480, and 640 px;
- lighting: native, bright, dim, warm, cool, and spatially uneven;
- perturbation: none, temporary occlusion, and bounded camera motion.

The transforms are deterministic. Temporary occlusion masks a fixed normalized
screen region during the middle fifth of each sequence. Camera motion uses a
bounded periodic affine rotation/translation with reflected borders. Manifest
dropout masks remain active independently, so the original annotated recovery
events retain their exact boundaries.

Frames are wall-clock paced. A 6-second source window therefore takes about
6 seconds at any requested FPS; the FPS changes the number and spacing of
observations, not playback speed. Transport output includes effective FPS,
end-to-end/server latency, and scheduling lag so an overloaded condition is
visible rather than silently treated as an accuracy result.

The detector decisions in each prediction reflect the same adaptive feedback
used by the browser. Acquiring, shifting, and recovery states may request a
15 Hz hand refresh; a healthy lock backs off to 5 Hz while optical flow carries
intervening landmarks. Reports preserve the actual detector-call count,
ordered search attempts, accepted/consumed asynchronous results, pose quality,
and selected search source.

List the exact conditions without loading models or media:

```powershell
cd fretcam
.venv\Scripts\python -m fretcam.live_position_benchmark --list-conditions
```

## Run the development split

Use the development split for implementation and threshold work:

```powershell
.venv\Scripts\python -m fretcam.live_position_benchmark `
  --split dev `
  --output-json "$HOME\.tabvision\cache\fretcam_artifacts\live-position-dev.json"
```

The console receives a compact condition summary. `--output-json` additionally
writes every prediction, the complete factor vector, raw blocker list, geometry
status/age, accuracy metrics, blocker summaries, and transport measurements.
The command creates the requested parent directory, but no output is written
when that option is omitted.

Run one sequence or condition while developing:

```powershell
.venv\Scripts\python -m fretcam.live_position_benchmark `
  --split dev `
  --sequence dev_104_ii_to_vi `
  --only fps-20_q-72_px-640_native_none
```

Custom dimensions use space-separated values:

```powershell
.venv\Scripts\python -m fretcam.live_position_benchmark `
  --fps 2 10 20 `
  --jpeg-quality 60 72 `
  --inference-size 320 640 `
  --lighting native dim uneven `
  --perturbation none occlusion
```

`--matrix cartesian` evaluates every combination instead of the bounded
one-factor suite. With all defaults that is 648 paced conditions, so list or
narrow the dimensions first.

## Reproduce one exact browser trace

For a difference that occurs only in the live browser, open **Local accuracy
tools** and select **Start exact comparison trace**. The clean, non-rolling
prefix stops accepting new packets after 10 seconds, 120 exact JPEGs, or
24 MB. Select **Save exact comparison trace** to make the second explicit
local-save decision.

Replay and compare that package without opening a camera:

```powershell
fretcam-trace-compare `
  "$HOME\.tabvision\cache\fretcam_diagnostics\traces\<trace-id>" `
  --output-json "$HOME\.tabvision\cache\fretcam_artifacts\trace-comparison.json"
```

The command validates every packet hash, byte count, sequence, JPEG dimension,
and browser context field before inference. It compares nonvolatile HUD,
position, confidence, blocker, contact, geometry, hand-search, adaptive
schedule, and pose fields. Latency is reported by the normal live benchmark
and is intentionally excluded from exact equality.

The two-second **failure buffer** is a different private troubleshooting
package. It can include the user's expected position, pressed fingers, and an
optional note, but it cannot be loaded by the trace comparator or either
evaluation workflow.

## Evaluation hygiene

- Keep rule and threshold choices on `--split dev`.
- Do not repeatedly open the source-disjoint `test` split.
- Cached MP4s stay under `~/.tabvision/cache/gaps_video/`.
- JSON results belong under
  `~/.tabvision/cache/fretcam_artifacts/`; do not commit them.
- The runner rejects manifests that are not marked public GAPS evidence.
- Browser diagnostic traces and marked-failure packages are private
  troubleshooting material, never benchmark, tuning, training, or release
  evidence.
- A blocker count is per frame and blockers can co-occur, so blocker rows do
  not sum to the frame population.
- Geometry freshness is reported separately from solver blockers. In
  particular, `geometry_status_counts.stale` identifies stale-board frames
  even when the solver also reports `low_confidence` or another blocker.
