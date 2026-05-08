# tabvision

Python CLI for guitar tablature transcription from iPhone video. The v1
pipeline follows the canonical spec in `../SPEC.md`: demux video, transcribe
audio, add vision evidence for string/fret placement, fuse the evidence, and
render tab.

Status: active v1 build, currently through Phase 6 renderer polish plus Phase
9 release scaffolding. The frozen v0 app remains in the repository as prior
art; new work lives in this package.

## Install

```bash
cd tabvision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[audio-baseline]'  # Basic Pitch baseline
python -m pip install -e '.[audio-highres]'   # high-resolution guitar backend
python -m pip install -e '.[render]'          # GP5, MusicXML, MIDI writers
python -m pip install -e '.[vision]'          # video stack; see license note
```

On Linux, the Basic Pitch extra currently needs Python 3.11 because its
TensorFlow dependency does not publish compatible Python 3.12 wheels.

Check optional model/dependency readiness:

```bash
python -m scripts.acquire.models list
python -m scripts.acquire.models status
python -m scripts.acquire.models prepare-yolo-dir
```

## Quickstart

Render ASCII tab:

```bash
tabvision transcribe input.mov --format ascii -o output.tab
```

Fresh-clone fixture smoke with a checked-in file:

```bash
python3.11 -m venv .venv-audio
source .venv-audio/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,audio-baseline]'
tabvision transcribe data/fixtures/test_a440.mp4 --audio-backend basicpitch --no-video --no-preflight --format ascii -o /tmp/tabvision-a440.tab
```

Write another supported format:

```bash
tabvision transcribe input.mov --format midi -o output.mid
tabvision transcribe input.mov --format musicxml -o output.musicxml
tabvision transcribe input.mov --format gp5 -o output.gp5
```

Useful context flags:

```bash
tabvision transcribe input.mov --instrument electric --tone clean --style mixed --capo 0
tabvision transcribe input.mov --no-video --format ascii
tabvision transcribe input.mov --position-prior guitarset-v1
```

`--position-prior none` is the default. `guitarset-v1` is an explicit
experimental prior backed by a checked-in artifact; it is not promoted to the
silent default until automated public/home-domain evidence shows no regression.

## Diagnose

`diagnose` writes a self-contained HTML report for one clip. The report always
contains overlay, audio, tab, and confidence sections. If optional video/audio
backends are unavailable, the report records the failure and keeps placeholders
instead of silently dropping sections.

```bash
tabvision diagnose input.mov -o input.diagnose.html
tabvision diagnose input.mov --no-video --no-preflight
```

## Output Formats

- `ascii`: dependency-free, confidence-aware tab. Notes below confidence `0.5`
  are marked with `?`.
- `gp5`: Guitar Pro 5 via PyGuitarPro. Tests skip gracefully when the optional
  dependency is not installed.
- `musicxml`: MusicXML via music21.
- `midi`: Standard MIDI via mido. Channel assignment is deterministic:
  channel `0` is low E, through channel `5` high E.

## Verification

```bash
pytest -q
python -m scripts.eval.run --scope smoke --twice-and-diff --output-dir /tmp/tabvision-eval-smoke
pytest -m render
python scripts/check_default_licenses.py --pyproject pyproject.toml
bash scripts/test_fresh_install.sh
```

`scripts/test_fresh_install.sh` is the Phase 9 fresh-clone scaffold. It clones
the repository, creates a venv, installs `.[dev]`, checks `tabvision --version`,
runs the default license policy check, and runs render tests.

Full hand-labeled, user-recorded eval remains useful future validation, but it
is not a v1 release prerequisite. v1 release evidence must be automated:
deterministic smoke fixtures, public/programmatic dataset reports such as
GuitarSet validation, license checks, fresh-install checks, and renderer tests.

## License Posture

The shipping default dependency set is intentionally small and portfolio-safe.
Optional extras carry their own documented trade-offs in `../LICENSES.md`.
Notably, the vision extra includes ultralytics under AGPL-3.0 by explicit
project decision, so it must remain opt-in until integration confirms the
shipping policy.

## Portfolio Docs

Portfolio/demo scaffolding lives at `../docs/DEMO/` and `../docs/NARRATIVE.md`.
Those files use existing/generated assets for v1. Hand-labeled user-video
side-by-side examples are optional future additions, not required release work.
Start with `../docs/DEMO/fresh-user-path.md` for the reproducible CLI demo.
