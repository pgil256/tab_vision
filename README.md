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
![Scope](https://img.shields.io/badge/scope-acoustic%20·%20audio--first-0a7)
![Default license](https://img.shields.io/badge/default%20deps-permissive%20·%20no%20copyleft-2b7)
![Posture](https://img.shields.io/badge/posture-personal%20·%20non--commercial-orange)
![Status](https://img.shields.io/badge/release-v1.0.0-blue)

## The idea

> **Pitch does not determine position.** The same note — say E4 — can be played
> in five or six places on the neck. A correct *transcription* gets the pitch; a
> correct *tab* gets the specific string and fret the player used.

So the whole project lives or dies on **string assignment**, and that is exactly
where audio runs out of the *obvious* information. TabVision leans into that
honestly: it ships what the evidence supports (acoustic, audio-first), names
what it can't do, and keeps every number measured. The full story — including a
video approach that looked obvious and didn't pay, and the string physics that
did — is in [`docs/NARRATIVE.md`](docs/NARRATIVE.md).

## Accuracy

All numbers on the GuitarSet held-out player-05 validation set (60 clips).
**Current default pipeline** — `highres-ensemble` audio, `guitarset-v1`
position prior, and the `acoustic-physics-v1` string-evidence channel with
partial-aware isolation, measured 2026-07-24
([report](docs/EVAL_REPORTS/player05_batched_confirm_2026-07-24.md)):

| Tier | Before the physics channel | **Current default** | Δ |
|---|---:|---:|---:|
| Single-line Tab F1 | 0.5503 | **0.7257** | +0.1754 |
| Strummed Tab F1 | 0.7176 | **0.7435** | +0.0258 |
| Aggregate Tab F1 | 0.6340 | **0.7346** | **+0.1006** |

Aggregate delta `+0.1006`, 95% bootstrap CI `[+0.0615, +0.1416]`, on a sealed
held-out player opened once for confirmation.

**The v1.0.0 acceptance record** (2026-06-03, `highres` + `guitarset-v1`, the
configuration the release was gated on —
[report](docs/EVAL_REPORTS/v1_acceptance_2026-06-03.md)):

| Metric | v1 gate | Measured (mean / lower-95) |
|---|---:|---:|
| Single-line Tab F1 | ≥ 0.45 | 0.523 / 0.457 |
| Strummed Tab F1 | ≥ 0.60 | 0.676 / 0.606 |
| Aggregate Tab F1 | ≥ 0.55 | 0.600 |
| Onset F1 (50 ms) | ≥ 0.92 | 0.94 / 0.92 |
| Pitch F1 (50 ms) | ≥ 0.90 | 0.93 / 0.90 |
| Latency (60 s clip, laptop CPU) | ≤ 5 min | ~45 s (0.74× realtime) |

Acceptance is `lower_95_CI ≥ target` over clips (bootstrap CIs). Scope and full
targets: [`SPEC.md`](SPEC.md) §1.4 / §1.4.1.

Onset and pitch F1 are **0.9491 / 0.9403** on the current default — improved by
the ensemble backend, and untouched by the string-evidence channel, which only
re-assigns positions and cannot add, remove, or retime a note. Every Tab F1
gain above is therefore string assignment, not better note detection.

A concrete worked example — the same piece by the same player, single-line vs.
strummed — is in
[`docs/DEMO/per-tier-examples.md`](docs/DEMO/per-tier-examples.md). On that
single-line clip the model hears 93% of notes correctly; under v1.0.0 only 33%
landed on the right string, and under the current default 67% do.

**What moved, and why it matters.** Single-line was the tier this project
called information-limited: the same pitch is acoustically near-identical
across strings, so the model heard the right note and put it on the wrong
string. Video was supposed to be the fix and wasn't. What actually broke it
open was **audio physics** — string stiffness makes a note's overtones stretch
sharp by an amount that depends on the string it was played on, so the
inharmonicity of a single note carries string identity. That channel is
specification-derived rather than fitted, and it abstains per note whenever the
partial structure is unreadable, so its failure mode is "no evidence" rather
than "wrong evidence."

**Honest limits (measured, not hedged):**

- **Single-line is still the weak tier**, and the physics channel only applies
  where it can read partials — 2,227 of 8,709 notes on the held-out set. The
  rest still fall back to the playability prior.
- **The channel is scoped to clean steel-string acoustic** in standard tuning.
  Classical/nylon, electric, alternate tunings, and non-`highres` backends
  abstain by construction — a nylon table was tried and banked as a negative.
- **Video does not currently earn its place in the default path.** An earlier
  "video is anti-informative" result turned out to be a fret-mapping bug, and
  the corrected probe is positive (gold fret inside the predicted window 76.3%
  of the time). But end-to-end against real audio predictions the effect is
  indistinguishable from zero: +0.000836 on ten source-disjoint clips with a CI
  touching zero, −0.000155 on clean-12. It ships as explicit opt-in
  (`--video-backend fretcam`), not as a default.
- **Electric guitar is v2.** The acoustic backbone drops to 0.73 pitch / 0.12
  Tab F1 on electric; closing it needs a spend-gated fine-tune. The
  `--instrument electric` tone toggle is already wired for that checkpoint.
- **Expressive markings (bends / slides / hammer-ons) are not detected** —
  the technique-detection baseline is a measured 0.00 (no detector yet).
- **Capo position is user-supplied.** With `--capo N` set, the position prior
  transforms covariantly (worth +0.387 at capo 2 versus the old routing).
  Detecting a capo from audio was refuted — it is pitch-identical to a
  transposition, and the probe recovered 1 of 60.

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

The default config resolves automatically per session: `highres-ensemble` audio,
`guitarset-v1` position prior, the `acoustic-physics-v1` string-evidence channel
inside its clean steel-string acoustic domain, audio-only. Classical/nylon
sessions route to the GAPS-trained priors instead; `--capo N` switches the
position prior to its capo-covariant form. The first run downloads the
checkpoints once, then caches them.

## How it works

```
input.mov ──► demux (ffmpeg) ──► audio transcription ──► note events
                              (highres-ensemble backend)      │
                                                              ▼
                                              fusion: Viterbi playability
                                              + learned position priors
                                              + inharmonicity string evidence
                                                              │
   [video position — opt-in, measured at ~0 end-to-end] ······┘
                                                              ▼
                                        render ──► ASCII · MIDI · MusicXML · GP5
```

Strict dataclass contracts between stages ([`SPEC.md`](SPEC.md) §8) let each
source of evidence improve without entangling the rest — the physics channel was
added without touching a single contract. It ships four ways: the local CLI, a
Modal production deploy, a one-command "studio" loop (`studio.ps1`) that records
from the browser and prints tab end-to-end, and a Windows desktop shell
([`desktop-client/`](desktop-client/)) with an installer, offline-after-bootstrap
transcription, and in-app camera + microphone recording.

## Repository layout

| Path | What |
|---|---|
| [`tabvision/`](tabvision/) | **v1 (active)** — the Python package + CLI. Start here. |
| [`desktop-client/`](desktop-client/) | Windows (WPF/.NET 8) desktop shell over the CLI. Thin and **disposable by design**. |
| [`fretcam/`](fretcam/) | Live fretboard/hand-position HUD prototype; also the opt-in `--video-backend fretcam` source. |
| [`docs/NARRATIVE.md`](docs/NARRATIVE.md) | The project story: what was hard, what worked, what's next. |
| [`docs/DEMO/`](docs/DEMO/) | Portfolio assets — architecture brief, per-tier examples, demo path. |
| [`docs/EVAL_REPORTS/`](docs/EVAL_REPORTS/) | Every accuracy claim's reproducible run. |
| [`SPEC.md`](SPEC.md) · [`LICENSES.md`](LICENSES.md) · [`docs/DECISIONS.md`](docs/DECISIONS.md) | Spec, license map, decision log. |
| `tabvision-server/` · `tabvision-client/` · `web-client/` | **v0 (frozen)** — the original Electron + Flask desktop demo. Kept as prior art and porting source; **not** the v1 shipping artifact. |

## License posture

**TabVision is a personal, non-commercial application** (SPEC §1.5, amended
2026-07-20). That posture is what makes the current default pipeline possible,
and it has a concrete consequence worth stating plainly:

- **Code dependencies** in the default path stay permissive with **no
  copyleft** — `highres-ensemble` audio (MIT), ffmpeg, numpy. Copyleft lives
  only in opt-in extras (the `vision` extra's YOLO detector is AGPL-3.0; the
  `render` extra's Guitar Pro writer is LGPL-3.0). Enforced in CI by
  `scripts/check_default_licenses.py`.
- **Model artifacts** are not uniformly permissive. The classical/nylon route
  loads `gaps-v1` / `gaps-seq-v1`, which are derived from the GAPS train split
  and inherit **CC-BY-NC-SA-4.0**. They are reachable from the default `auto`
  path when a session is classical, so **the default pipeline is not
  commercially redistributable as-is** — those two priors would have to be
  rebuilt from permissive data first.

Every NC-derived artifact is labeled in [`LICENSES.md`](LICENSES.md) precisely
so that a future commercialization knows what to replace. Private and
user-supplied recordings remain banned from all training, eval, and labeling
roles.

## Status

**v1.0.0** is the tagged release — acoustic, audio-only, cut against the
acceptance gate above. `main` has moved on since: an accuracy program raised the
default aggregate to **0.7346** on the held-out player (single-line 0.5503 →
0.7257), capo sessions now route to a covariant prior, and a Windows desktop
shell and live FretCam HUD landed alongside the CLI.

Next levers, in rough order of measured promise: extending string evidence past
the 26% of notes whose partials are currently readable; a technique detector
(greenfield from a measured 0.00); and the spend-gated electric fine-tune for
v2. Tracked in [`docs/NARRATIVE.md`](docs/NARRATIVE.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md).
