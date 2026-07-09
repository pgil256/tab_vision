# TabVision

**Turn a recording of solo guitar into tablature — the string-and-fret notation
a guitarist actually reads.** TabVision v1 is an audio-first Python CLI, scoped
to acoustic guitar, built around the one problem a pitch model doesn't solve:
*which string did you play it on?*

![TabVision transcribing a clip to a confidence-graded ASCII tab](docs/DEMO/demo.gif)

*`tabvision transcribe` printing a confidence-graded tab (a real decoded
GuitarSet excerpt). Frets are coloured by model confidence — green ≥ 0.8, amber
≥ 0.5, red < 0.5 — and low-confidence notes are also marked `?`.*

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Scope](https://img.shields.io/badge/v1%20scope-acoustic%20·%20audio--only-0a7)
![Default license](https://img.shields.io/badge/default%20deps-permissive%20·%20no%20copyleft-2b7)
![Status](https://img.shields.io/badge/release-v1.0.0-blue)

## The idea

> **Pitch does not determine position.** The same note — say E4 — can be played
> in five or six places on the neck. A correct *transcription* gets the pitch; a
> correct *tab* gets the specific string and fret the player used.

So the whole project lives or dies on **string assignment**, and that turns out
to be exactly where audio alone runs out of information. TabVision v1 leans into
that honestly: it ships what the evidence supports (acoustic, audio-first),
names what it can't do, and keeps every number measured. The full story —
including a video approach that looked obvious and was **refuted** on real
footage — is in [`docs/NARRATIVE.md`](docs/NARRATIVE.md).

## Accuracy

Measured on the GuitarSet held-out player-05 validation set (60 clips), formal
acceptance run 2026-06-03
([report](docs/EVAL_REPORTS/v1_acceptance_2026-06-03.md)):

| Metric | v1 gate | Measured (mean / lower-95) |
|---|---:|---:|
| Single-line Tab F1 | ≥ 0.45 | **0.523** / 0.457 |
| Strummed Tab F1 | ≥ 0.60 | **0.676** / 0.606 |
| Aggregate Tab F1 | ≥ 0.55 | **0.600** |
| Onset F1 (50 ms) | ≥ 0.92 | 0.94 / 0.92 |
| Pitch F1 (50 ms) | ≥ 0.90 | 0.93 / 0.90 |
| Latency (60 s clip, laptop CPU) | ≤ 5 min | ~45 s (0.74× realtime) |

Acceptance is `lower_95_CI ≥ target` over clips (bootstrap CIs). Scope and full
targets: [`SPEC.md`](SPEC.md) §1.4 / §1.4.1. A concrete per-tier example — the
same piece played single-line vs. strummed — is in
[`docs/DEMO/per-tier-examples.md`](docs/DEMO/per-tier-examples.md), and it shows
the thesis in one line: on the single-line clip TabVision hears **93%** of notes
correctly but only **33%** land on the right string (a **0.600** gap that is
pure string-assignment); on the strummed clip that gap is **0.027**, because
chord shapes pin the positions down.

**Honest limits (measured, not hedged):**

- **Single-line is information-limited.** Audio cannot tell which string a pitch
  was played on. Video looked like the fix but was **refuted** on real
  in-the-wild footage (the audio prior resolves contested strings at 0.778 vs.
  the best real video chain's 0.574). The video stack stays in the repo as
  measured evidence, not a shipping default.
- **Electric guitar is v2.** The acoustic backbone drops to 0.73 pitch / 0.12
  Tab F1 on electric; closing it needs a spend-gated fine-tune. The
  `--instrument electric` tone toggle is already wired for that checkpoint.
- **Expressive markings (bends / slides / hammer-ons) are not detected** —
  the technique-detection baseline is a measured 0.00 (no detector yet).

## Install & quickstart

The v1 package lives in [`tabvision/`](tabvision/) — see its
[README](tabvision/README.md) for the full cookbook.

```bash
cd tabvision
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev,audio-highres]'         # highres backend (torch)

# Transcribe to a confidence-graded ASCII tab (low-confidence notes marked '?',
# colour-graded green/amber/red in a terminal):
tabvision transcribe input.mov --format ascii -o out.tab

# Other formats: MIDI / MusicXML / Guitar Pro 5
tabvision transcribe input.mov --format midi -o out.mid

# One-clip HTML debug report (waveform + decoded tab + confidence map):
tabvision diagnose input.mov -o report.html
```

The default config is the accepted v1 setup: `highres` audio backend +
`guitarset-v1` position prior, audio-only. The first run downloads the highres
checkpoint once (~37 s), then caches it.

## How it works

```
input.mov ──► demux (ffmpeg) ──► audio transcription ──► note events
                                   (highres backend)          │
                                                              ▼
                                              fusion: Viterbi playability
                                              + learned position priors
                                                              │
   [video evidence — built, measured, refuted as a lever] ····┘
                                                              ▼
                                        render ──► ASCII · MIDI · MusicXML · GP5
```

Strict dataclass contracts between stages ([`SPEC.md`](SPEC.md) §8) let each
source of evidence improve without entangling the rest. It ships three ways: the
local CLI, a Modal production deploy, and a one-command "studio" loop
(`studio.ps1`) that records from the browser and prints tab end-to-end.

## Repository layout

| Path | What |
|---|---|
| [`tabvision/`](tabvision/) | **v1 (active)** — the Python package + CLI. Start here. |
| [`docs/NARRATIVE.md`](docs/NARRATIVE.md) | The project story: what was hard, what worked, what's next. |
| [`docs/DEMO/`](docs/DEMO/) | Portfolio assets — architecture brief, per-tier examples, demo path. |
| [`docs/EVAL_REPORTS/`](docs/EVAL_REPORTS/) | Every accuracy claim's reproducible run. |
| [`SPEC.md`](SPEC.md) · [`LICENSES.md`](LICENSES.md) · [`docs/DECISIONS.md`](docs/DECISIONS.md) | Spec, license map, decision log. |
| `tabvision-server/` · `tabvision-client/` · `web-client/` | **v0 (frozen)** — the original Electron + Flask desktop demo. Kept as prior art and porting source; **not** the v1 shipping artifact. |

## License posture

The shipping **default** pipeline is intentionally small and **permissive** —
`highres` audio (MIT), ffmpeg, numpy — and carries **no copyleft**. Because v1
is audio-only, the default path pulls in **no AGPL code**. Copyleft lives only
in opt-in extras (the `vision` extra's YOLO detector is AGPL-3.0; the `render`
extra's Guitar Pro writer is LGPL-3.0), enforced in CI by
`scripts/check_default_licenses.py`. Full map: [`LICENSES.md`](LICENSES.md).

## Status

v1.0.0 — acoustic, audio-only, released against the acceptance gate above. Active
work and next levers (audio-side single-line, electric v2) are tracked in
[`docs/NARRATIVE.md`](docs/NARRATIVE.md) and [`SPEC.md`](SPEC.md).
