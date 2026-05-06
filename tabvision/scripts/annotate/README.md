# Labeling harness

Closes the data-bound acceptance gates for Phase 3 and Phase 4 of the
TabVision spec.

## What it labels

| Mode | What you click / type | Closes |
|---|---|---|
| **framing** | "good" or "bad" + reason tags (off-center, dim, oblique, …) | Phase 3 preflight gate (≥ 9/10) |
| **fretboard** | 4 fret-intersections per clip (frets 5 + 12, top + bottom edges) | Phase 3 keypoint-fretboard gate (≤ 5 px median error) |
| **fingering** | Per-finger (string, fret) on N evenly-spaced frames per clip | Phase 4 gate (top-1 ≥ 0.75 over 100 frames) |

## Run it

From anywhere on the box:

```bash
bash tabvision/scripts/annotate/launch.sh
# → opens http://127.0.0.1:5005/
```

That's the whole setup.  The launcher:

1. Finds an existing venv that has flask + cv2 (checks
   `$TABVISION_VENV`, then `./venv`, then `./tabvision-server/venv`).
2. `cd`s into the v1 package directory so `python -m scripts.annotate.label_clips`
   resolves correctly.
3. Defaults `--clips` to
   `test-data/training-tabs/training-tabs-video` (the existing 20
   training clips) — pass an explicit dir to override.

If no suitable venv exists, the script tells you the exact command to
prepare one:

```bash
source venv/bin/activate           # or wherever your venv is
pip install flask opencv-python
```

(Don't `pip install` against system Python on Ubuntu 24.04 — PEP 668
will refuse.  Always go through a venv.)

### CLI flags

```bash
bash tabvision/scripts/annotate/launch.sh /path/to/clips 5005
```

Position 1 = clip directory (default = the training-tabs videos).
Position 2 = port (default 5005).  For more control invoke the module
directly:

```bash
source venv/bin/activate
cd tabvision
python -m scripts.annotate.label_clips \
    --clips ../test-data/eval-clips \
    --fingering-frames 20 \
    --host 0.0.0.0 \
    --port 5005
```

Flags:
- `--clips <dir>` — directory of `.mp4`/`.mov`/`.m4v`/`.avi`/`.mkv` files
- `--eval-root <dir>` — output JSON root. Default `tabvision/data/eval/`,
  override via `TABVISION_EVAL_ROOT`
- `--fingering-frames N` — frames per clip for fingering (default 20;
  spec calls for 100 total, so 5 clips × 20 ≈ that)
- `--host 127.0.0.1` / `--port 5005` — bind address. Use `0.0.0.0` to
  expose on the LAN (e.g. for labeling from a tablet).

### Picking your eval set

Collect 5–10 clips for the eval set under a single directory.  A mix
of clean takes you already plan to test on, plus a few intentionally
badly framed ones for the preflight set:

```
test-data/eval-clips/
├── good_open_chords.mov
├── good_pentatonic_run.mov
├── bad_offcenter.mov
├── bad_dim.mov
└── ...
```

## Workflow

The index page lists every clip with three columns — one per label
type — plus progress (`done` shows green, `pending` shows orange).
Pick a clip, label it, save, go back, repeat. State is per-clip so you
can interrupt at any time and resume later.

### Framing
1. Look at the representative frame.
2. Press **g** for good, **b** for bad. If "bad", check any tags that
   apply.
3. Save (or Ctrl/Cmd-S).

Aim for ~5 good and ~5 bad clips, with the bad set spanning multiple
failure modes (off-center, dim, oblique, partial occlusion, drift).

### Fretboard
1. Adjust the **frame** input if the default 1.5 s frame is occluded.
2. Click **fret 5 top → fret 5 bottom → fret 12 top → fret 12 bottom**
   in order. The labels appear as you click.
3. **Undo last point** removes the most recent click; **Clear all** resets.
4. Save (button enables when 4 points are placed).

A "top" point is on the high-E side of the neck (top of frame in the
standard iPhone-on-lap framing).

### Fingering
1. The page shows N evenly-spaced frames as a grid.
2. For each frame, set the (string, fret) of each fretting finger:
   - `string` is **1-indexed from high E** (string 1 = high E, string 6 = low E).
   - `fret` is `0` for an open string up to `12` for the spec range.
   - Leave both blank to mark a finger as **not fretting** in this frame.
3. **Save all** writes a single JSON for the whole clip.

Eight fretting fingers per frame × 100 frames = ~400 dropdown picks
total; budget ~30 minutes if you're moving briskly.

## Storage layout

```
tabvision/data/eval/
├── framing/<clip_id>.json
├── fretboard/<clip_id>.json
└── fingering/<clip_id>.json
```

`<clip_id>` is the filename stem with non-`[A-Za-z0-9._-]` chars
replaced by hyphens. The JSON schema is the
:class:`storage.{FramingLabel,FretboardLabel,FingeringLabel}` shape.

## Run the gated acceptance tests

Once you have labels:

```bash
pytest -m preflight_eval     # Phase 3 preflight
pytest -m fretboard_eval     # Phase 3 keypoint fretboard
pytest -m hand_eval          # Phase 4 fingertip top-1
```

Each test skips with a helpful message if you haven't collected
enough labels yet (10 framing / 5 fretboard / 100 finger labels).
