# Fresh User Path

This is the reproducible v1 CLI path from a clean clone. It uses only
checked-in fixtures and optional pip extras; it does not require new recordings
or hand-authored annotations.

## Minimal Install Smoke

```bash
cd tabvision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
tabvision --version
python scripts/check_default_licenses.py --pyproject pyproject.toml
```

## Fixture Transcription Smoke

The checked-in `data/fixtures/test_a440.mp4` fixture is a small A440 clip. Run
the audio-only path with the Basic Pitch extra. On Linux this path uses Python
3.11 because Basic Pitch depends on TensorFlow wheels that do not resolve under
Python 3.12.

```bash
cd tabvision
python3.11 -m venv .venv-audio
source .venv-audio/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,audio-baseline]'
tabvision transcribe data/fixtures/test_a440.mp4 \
  --audio-backend basicpitch \
  --no-video \
  --no-preflight \
  --format ascii \
  -o /tmp/tabvision-a440.tab
```

Expected musical shape: MIDI 69 / A440 renders as high-E string, fret 5. See
`sample-a440-ascii.tab` for the compact expected render shape.

## Optional Full Pipeline Readiness

```bash
cd tabvision
python -m scripts.acquire.models list
python -m scripts.acquire.models status
python -m scripts.acquire.models prepare-yolo-dir
```

Install groups as needed:

```bash
python -m pip install -e '.[audio-highres]'
python -m pip install -e '.[vision]'
python -m pip install -e '.[render]'
```

The video stack requires a trained YOLO-OBB checkpoint. Place it at the path
printed by `prepare-yolo-dir`, or set:

```bash
export TABVISION_GUITAR_YOLO_CHECKPOINT=/path/to/guitar-yolo-obb-finetuned.pt
```

The full user-video command is then:

```bash
tabvision transcribe input.mov \
  --audio-backend highres \
  --position-prior none \
  --format ascii \
  -o output.tab
```

`--position-prior guitarset-v1` remains explicit experimental behavior.

## Verified Local Fixture Result

On May 7, 2026, the fixture command above was verified in a fresh Python 3.11
venv. It installed `.[audio-baseline]`, ran `tabvision transcribe` on
`data/fixtures/test_a440.mp4`, and wrote `/tmp/tabvision-fresh311/a440.tab`
matching `sample-a440-ascii.tab`.
