# tabvision

Python CLI for guitar tablature transcription from iPhone video. The v1
pipeline follows the canonical spec in `../SPEC.md`: demux video, transcribe
audio, add vision evidence for string/fret placement, fuse the evidence, and
render tab.

Status: active v1 build, currently through Phase 6 renderer polish plus Phase
9 release scaffolding. The frozen v0 app remains in the repository as prior
art; new work lives in this package.

## Accuracy

v1 is **acoustic, audio-only** by scope — an evidence-based decision, not a
relaxation (electric was measured and found blocked on a model that does not yet
exist; the full story is in [`../docs/NARRATIVE.md`](../docs/NARRATIVE.md)).
Measured on the GuitarSet held-out player-05 validation set (60 clips), formal
acceptance run 2026-06-03
([report](../docs/EVAL_REPORTS/v1_acceptance_2026-06-03.md)):

| Metric | v1 gate | Measured (mean / lower-95) |
|---|---:|---:|
| Single-line Tab F1 | ≥ 0.45 | **0.523** / 0.457 |
| Strummed Tab F1 | ≥ 0.60 | **0.676** / 0.606 |
| Aggregate Tab F1 | ≥ 0.55 | **0.600** |
| Onset F1 (50 ms) | ≥ 0.92 | 0.94 / 0.92 |
| Pitch F1 (50 ms) | ≥ 0.90 | 0.93 / 0.90 |
| Latency (60 s clip, laptop CPU) | ≤ 5 min | ~45 s (0.74× realtime) |

Acceptance is `lower_95_CI ≥ target` over clips (bootstrap CIs). Full targets and
scope rationale: [`../SPEC.md`](../SPEC.md) §1.4 / §1.4.1.

**Honest limits (measured, not hedged):**

- **Single-line is information-limited.** Audio cannot tell which string a pitch
  was played on — the dominant error is `wrong_position_same_pitch` (pitch
  correct, string wrong). Video looked like the fix but was **refuted** on real
  in-the-wild footage: the audio playability prior resolves contested strings at
  0.778 vs the best real video chain's 0.574
  ([A14](../docs/EVAL_REPORTS/a14_video_complementarity_2026-07-06.md),
  [GAPS video-chain](../docs/EVAL_REPORTS/v1_1_gaps_video_chain_2026-06-22.md)).
  The video stack stays in the repo as measured evidence, not a shipping default.
- **A compact timbral string ranker did not generalize.** Five-player OOF
  evaluation scored 0.633 candidate top-1 versus the 0.655 prior-only baseline,
  so no timbral model or paid training ships
  ([Phase 2 report](../docs/EVAL_REPORTS/string_assignment_phase2_free_2026-07-14.md)).
- **Electric → v2.** The acoustic backbone drops to 0.73 pitch / 0.12 Tab F1 on
  electric ([cross-dataset](../docs/EVAL_REPORTS/cross_dataset_prior_2026-06-02.md));
  closing it needs a spend-gated fine-tune. The `--instrument electric` toggle is
  wired for when that checkpoint exists.
- **Expressive markings (bends / slides / hammer-ons / pull-offs) are not
  detected** — technique-detection baseline is 0.00 (no detector yet;
  `../docs/EVAL_REPORTS/d1b_technique_baseline_2026-07-09.md`).

Every number is reproducible from the harness — `python -m scripts.eval.run` and
the per-report scripts under `scripts/eval/`.

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

`transcribe` defaults to the accepted v1 config: the `highres` audio backend
(via `--audio-backend auto`, which routes `--instrument electric` to the
electric checkpoint) plus domain-aware `--position-prior auto` and
`--sequence-prior auto`. Clean acoustic, standard-tuning, capo-zero sessions
use the hash-verified GuitarSet pair; classical, electric, nonstandard-tuning,
and capo sessions use neutral learned position evidence. The first run
downloads the highres checkpoint once (~37 s); later runs load it from the
local cache. Requires the `audio-highres` extra (torch).

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
tabvision transcribe input.mov --position-prior none --sequence-prior none --audio-backend basicpitch
tabvision transcribe input.mov --string-evidence none
```

`auto` is the production default. Explicit `guitarset-v1` /
`guitarset-seq-v1` force the compatible registered pair for reproducible
evaluation or rollback; `none` preserves the bare decode. The rejected
`guitarset-timbre-v1` artifact is not registered, so automatic string evidence
currently resolves to `none`. `basicpitch` remains available as the lightweight
CPU baseline (needs the `audio-baseline` extra; Python 3.11 on some platforms —
see Install).

Server rollback controls:

```text
TABVISION_POSITION_PRIOR=auto|none|guitarset-v1
TABVISION_SEQUENCE_PRIOR=auto|none|guitarset-seq-v1
TABVISION_STRING_EVIDENCE=auto|none
TABVISION_PHRASE_REFINEMENT=false
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
pytest -q
python -m scripts.eval.run --scope smoke --twice-and-diff --output-dir /tmp/tabvision-eval-smoke
pytest -m render
python scripts/check_default_licenses.py --pyproject pyproject.toml
bash scripts/test_fresh_install.sh
```

`scripts/test_fresh_install.sh` is the Phase 9 fresh-clone scaffold. It clones
the repository, creates a venv, installs `.[dev]`, checks `tabvision --version`,
runs the default license policy check, and runs render tests.

Full hand-labeled private-video eval is no longer part of the roadmap. v1
release evidence must be automated:
deterministic smoke fixtures, public/programmatic dataset reports such as
GuitarSet validation, license checks, fresh-install checks, and renderer tests.

## License Posture

The shipping **default** pipeline is intentionally small and **permissive** —
`highres` audio (MIT), ffmpeg, numpy — and carries **no copyleft**. Because v1
is audio-only, the default path pulls in **no AGPL code**. Copyleft dependencies
live only in **opt-in extras**, enforced by `scripts/check_default_licenses.py`
(CI fails if one reaches `[project].dependencies`):

- **`vision` extra → AGPL-3.0.** The YOLO guitar detector (ultralytics) is
  AGPL-3.0, accepted by explicit project decision because no permissive
  pretrained guitar detector exists. Installing `vision` makes your working copy
  a "work based on" ultralytics under AGPL; the **default audio-only install does
  not**.
- **`render` extra → LGPL-3.0 (opt-in, not contagious).** Guitar Pro export uses
  PyGuitarPro (LGPL-3.0-only); MusicXML uses music21 (BSD-3-Clause); MIDI uses
  mido (MIT). LGPL permits use-by-import from a pip-installed CLI without making
  TabVision copyleft — kept in the opt-in `render` extra, used unmodified, with
  attribution (export-dependency license review 2026-07-09).

Full dependency / asset license map + phase gates: [`../LICENSES.md`](../LICENSES.md).

### Dataset attribution

The shipped fingering-prior artifacts under `tabvision/fusion/priors/` are
derived count statistics (never redistributed score/audio content) from:

- **GuitarSet** (Xi et al., ISMIR 2018) — `guitarset-v1`,
  `guitarset-seq-v1`, plus unregistered solo/comp diagnostic artifacts. All
  manifests record players 00–04 as construction data and player 05 as excluded.
- **PDMX** (Long et al., "PDMX: A Large-Scale Public Domain MusicXML
  Dataset for Symbolic Music Processing", ICASSP 2025;
  DOI [10.5281/zenodo.15571083](https://doi.org/10.5281/zenodo.15571083)) —
  `pdmx-seq-v1`, `guitarset-pdmx-seq-v1`; built from the
  `no_license_conflict` subset only, per the license review in
  `../LICENSES.md`.

## Portfolio Docs

Portfolio/demo scaffolding lives at `../docs/DEMO/` and `../docs/NARRATIVE.md`.
Those files use existing/generated assets for v1. Hand-labeled user-video
side-by-side examples are optional future additions, not required release work.
Start with `../docs/DEMO/fresh-user-path.md` for the reproducible CLI demo.
