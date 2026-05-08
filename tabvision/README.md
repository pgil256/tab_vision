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
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[audio-baseline]'  # Basic Pitch baseline
python -m pip install -e '.[render]'          # GP5, MusicXML, MIDI writers
python -m pip install -e '.[vision]'          # video stack; see license note
```

## Quickstart

Render ASCII tab:

```bash
tabvision transcribe input.mov --format ascii -o output.tab
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
```

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
pytest -m render
python scripts/check_default_licenses.py --pyproject pyproject.toml
bash scripts/test_fresh_install.sh
```

`scripts/test_fresh_install.sh` is the Phase 9 fresh-clone scaffold. It clones
the repository, creates a venv, installs `.[dev]`, checks `tabvision --version`,
runs the default license policy check, and runs render tests.

## License Posture

The shipping default dependency set is intentionally small and portfolio-safe.
Optional extras carry their own documented trade-offs in `../LICENSES.md`.
Notably, the vision extra includes ultralytics under AGPL-3.0 by explicit
project decision, so it must remain opt-in until integration confirms the
shipping policy.

## Portfolio Docs

Portfolio/demo scaffolding lives at `../docs/DEMO/` and `../docs/NARRATIVE.md`.
Those files are placeholders for final screen recordings, side-by-side eval
examples, and the architecture story after integration.
