# TabVision

TabVision turns a recording of solo guitar into tablature: the string-and-fret
notation guitarists actually read. v1 is an audio-first Python CLI scoped to
acoustic guitar, and most of the work in it goes into the question a pitch model
leaves open — *which string did you play it on?*

![TabVision transcribing a clip to a confidence-graded ASCII tab](docs/DEMO/demo.gif)

*`tabvision transcribe` printing a confidence-graded tab from a real decoded
GuitarSet excerpt. Frets are coloured by model confidence (green ≥ 0.8, amber
≥ 0.5, red < 0.5), and low-confidence notes are also marked `?`.*

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Scope](https://img.shields.io/badge/scope-acoustic%20·%20audio--first-0a7)
![Default license](https://img.shields.io/badge/default%20deps-permissive%20·%20no%20copyleft-2b7)
![Posture](https://img.shields.io/badge/posture-personal%20·%20non--commercial-orange)
![Status](https://img.shields.io/badge/release-v1.0.0-blue)

## The idea

> Pitch does not determine position. The same note, say E4, can be played in
> five or six places on the neck. A correct *transcription* gets the pitch. A
> correct *tab* gets the specific string and fret the player used.

So the project succeeds or fails on string assignment, which is also where audio
stops handing you the answer. v1 ships what the evidence supports (acoustic,
audio-first), says where it falls short, and keeps a measurement behind both.
The longer version, including a video approach that looked obvious and didn't
pay off and the string physics that did, is in
[`docs/NARRATIVE.md`](docs/NARRATIVE.md).

📝 **Blog post:** [The Same Note, Six Places: Seven Months of Trying to Read a
Guitarist's Hands](https://pgil256.github.io/blog/posts/tabvision/) — the
start-to-finish account: every hypothesis, the eleven negative results, and why
the biggest accuracy win came from 19th-century string physics instead of a
bigger model.

## Accuracy

Current default pipeline: `highres-ensemble` audio, `guitarset-v1` position
prior, and the `acoustic-physics-v1` string-evidence channel with partial-aware
isolation. Measured on GuitarSet under leave-one-player-out priors, 2026-07-25
([report](docs/EVAL_REPORTS/phase0_rotation_baseline_2026-07-25.md)):

| Tier | Before the physics channel | **Current default** | Δ |
|---|---:|---:|---:|
| **Held-out player 04** (60 clips) | | | |
| Single-line Tab F1 | 0.5854 | **0.6686** | +0.0832 |
| Strummed Tab F1 | 0.6320 | **0.6533** | +0.0213 |
| Aggregate Tab F1 | 0.6087 | **0.6609** | **+0.0522** `[+0.0259, +0.0809]` |
| **Development, 5 players** (300 clips) | | | |
| Aggregate Tab F1 | 0.6083 | **0.6801** | **+0.0718** `[+0.0558, +0.0885]` |

An earlier revision of this table reported 0.7346 aggregate. That number is
correct for the player it was measured on, and the current harness reproduces it
exactly. But measuring all six players put the channel's gain between `+0.047`
and `+0.101`, and that player sat at the top of the range, so `+0.05` to `+0.07`
is the better estimate. A held-out player is a single draw from a population
that isn't uniform.

The v1.0.0 acceptance record (2026-06-03, `highres` + `guitarset-v1`, the
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
targets are in [`SPEC.md`](SPEC.md) §1.4 / §1.4.1.

Onset and pitch F1 vary by player like everything else here: 0.9270 / 0.9094
across the five development players, 0.9032 / 0.8673 on held-out player 04. Both
are identical in the two arms, since the string-evidence channel only re-assigns
positions and can't add, remove, or retime a note. Between arms the
`missed_onset`, `extra_detection`, `pitch_off` and `timing_only` buckets match
event for event, and only wrong-position errors move, so the Tab F1 gains above
come from string assignment rather than from better note detection.

A worked example, the same piece by the same player in single-line and strummed
form, is in [`docs/DEMO/per-tier-examples.md`](docs/DEMO/per-tier-examples.md).
On that single-line clip the model hears 93% of notes correctly. Under v1.0.0
only 33% landed on the right string; under the current default 67% do.

### Where the gain came from

Single-line was the tier this project called information-limited: the same pitch
sounds near-identical across strings, so the model heard the right note and put
it on the wrong string. Video was supposed to fix that and didn't. Audio physics
did. String stiffness stretches a note's overtones sharp by an amount that
depends on the string it was played on, so the inharmonicity of one note carries
string identity. The channel is derived from specification rather than fitted,
and it abstains per note whenever the partial structure is unreadable, so a
failure produces no evidence instead of bad evidence.

### Known limits

- Single-line is still the weak tier, and most of the remaining loss lives
  there: wrong-position errors are 63.6% of single-line loss against 42.4% of
  strummed. The physics channel only applies where it can read partials, which
  is 22.4% of notes. The rest fall back to the playability prior.
- Accuracy varies a lot by player. Aggregate Tab F1 across the six GuitarSet
  players spans 0.59 to 0.71 under the current default, so any single held-out
  number, including the ones above, is one draw.
- The channel is scoped to clean steel-string acoustic in standard tuning.
  Classical/nylon, electric, alternate tunings and non-`highres` backends
  abstain by construction. A nylon table was tried and banked as a negative.
- Video doesn't currently earn its place in the default path. An earlier "video
  is anti-informative" result turned out to be a fret-mapping bug, and the
  corrected probe is positive (gold fret inside the predicted window 76.3% of
  the time). End-to-end against real audio predictions, though, the effect is
  indistinguishable from zero: +0.000836 on ten source-disjoint clips with a CI
  touching zero, −0.000155 on clean-12. It ships as explicit opt-in
  (`--video-backend fretcam`).
- Electric guitar is v2. The acoustic backbone drops to 0.73 pitch / 0.12 Tab F1
  on electric, and closing that needs a spend-gated fine-tune. The
  `--instrument electric` tone toggle is already wired for that checkpoint.
- Expressive markings (bends, slides, hammer-ons) aren't detected. The
  technique-detection baseline is a measured 0.00, since there's no detector yet.
- Capo position is user-supplied. With `--capo N` set the position prior
  transforms covariantly, worth +0.387 at capo 2 against the old routing.
  Detecting a capo from audio was refuted: it is pitch-identical to a
  transposition, and the probe recovered 1 of 60.

## Improving it on your own playing

The decoder can learn a **personal position prior** from your corrected
transcriptions. Giving a player their own prior measured **+0.0305
[+0.0183, +0.0430]** aggregate Tab F1 — the largest single lever left after the
accuracy program — and it pays most where a player's habits deviate from the
population average: +0.076 for the most-deviating GuitarSet player, roughly
zero for the most average one
([report](docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.md)). The loop is
local by design (SPEC §1.5): the artifact never ships, never becomes a default,
and never enters an eval corpus.

The loop: record in the studio (`studio.ps1`), fix the transcription with the
review keys, press **Bank gold** (labels accumulate in
`~/.tabvision/personal/labels.jsonl`; the button only exists when the local
server advertises it — the deployed site answers 404 by construction), then
build and use the artifact:

```bash
cd tabvision
python -m scripts.train.build_personal_prior ~/.tabvision/personal/labels.jsonl \
    -o ~/.tabvision/personal/personal-prior.json
tabvision transcribe input.mov --position-prior ~/.tabvision/personal/personal-prior.json
```

The `auto` route never resolves to a personal artifact on its own — you pass it
explicitly (for the studio server, set `TABVISION_POSITION_PRIOR` to the
artifact path instead of `auto`).

### What to record — the material that pays

Not every take teaches it equally. The loss decomposition says where the
remaining errors are, and therefore what a corrected take is worth:

- **Single-line melodies over strummed chords.** Wrong-position errors — right
  pitch, wrong string — are 47.6% of all remaining loss, and 63.6% of
  single-line loss against 42.4% strummed. Chord voicings largely constrain
  their own members (contextual evidence measured ~6× more useful on chords
  than on lines — [report](docs/EVAL_REPORTS/s1b_context_probe_2026-07-22.md)),
  so banked chords are cheap confirmations but low-yield. Melody notes are
  where the prior earns.
- **Mid-neck, in the positions you actually use.** Most mid-register pitches
  have 3–4 playable candidates, and the competing candidates sit 4–5 frets
  apart ([report](docs/EVAL_REPORTS/q6_separability_2026-07-22.md)) — exactly
  the decision the decoder gets wrong. Its defaults carry a low-fret bias and
  an open-string bonus, and the population prior is trained on GuitarSet
  players, so if you take melodies up the neck instead of at the nut, the
  system is currently systematically wrong about you — those are the most
  valuable labels. Conversely, don't record exercises in positions you never
  use: the prior learns per-pitch habits, and habits you don't have pollute it.
- **Plain-string melodies (B and high E) are the least-defended notes.** The
  physics channel's gain lives almost entirely on the wound strings
  ([report](docs/EVAL_REPORTS/n5_table_mismatch_2026-07-24.md)), so an
  ambiguous note on the trebles is resolved by the prior alone.
- **Sustained, separated notes.** Notes under 150 ms are missed at 1.61× base
  rate and notes inside 3-plus-note simultaneity at 1.63×, while an isolated
  note is missed at less than half base rate
  ([report](docs/EVAL_REPORTS/d_detection_probe_2026-07-25.md)). A missed note
  never reaches the review queue and yields no label, so material the detector
  already transcribes well returns far more labels per minute of correction.
- **Repetition beats variety.** A pitch personalizes only at ≥5 confirmed
  labels — below that it keeps the population prior — so a handful of corrected
  takes of the same piece activates the prior across its whole range, where
  twenty pieces played once each may activate nothing.

Hard constraints: standard tuning and capo 0 (banking refuses capo takes),
clean steel-string acoustic declared as such, tuned to pitch first. Mic and
room quality measured as irrelevant
([report](docs/EVAL_REPORTS/a8_studio_degradation_val24_2026-07-07.md)) — don't
buy gear for this. Bends and slides only add pitch-track noise, since there is
no technique modeling yet.

One honesty note: **+0.0305 is a ceiling**, measured with a perfectly estimated
prior. What a finite label store actually reaches is unmeasured *by design* —
evaluating on your own recordings is the other half of the §1.5 ban. The
working signal is your own correction rate drifting down on the material you
have covered, starting with exactly the pitches and positions you banked most.

## Install & quickstart

The v1 package lives in [`tabvision/`](tabvision/); its
[README](tabvision/README.md) has the full cookbook.

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

The default config resolves per session: `highres-ensemble` audio,
`guitarset-v1` position prior, the `acoustic-physics-v1` string-evidence channel
inside its clean steel-string acoustic domain, audio only. Classical/nylon
sessions route to the GAPS-trained priors instead, and `--capo N` switches the
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

Stages talk through strict dataclass contracts ([`SPEC.md`](SPEC.md) §8), so one
source of evidence can improve without entangling the rest; the physics channel
went in without touching a contract. There are four ways to run it: the local
CLI, a Modal production deploy, a one-command "studio" loop (`studio.ps1`) that
records from the browser and prints tab end to end, and a Windows desktop shell
([`desktop-client/`](desktop-client/)) with an installer,
offline-after-bootstrap transcription, and in-app camera and microphone
recording.

## Repository layout

| Path | What |
|---|---|
| [`tabvision/`](tabvision/) | **v1 (active)** — the Python package + CLI. Start here. |
| [`desktop-client/`](desktop-client/) | Windows (WPF/.NET 8) desktop shell over the CLI. Thin and disposable by design. |
| [`fretcam/`](fretcam/) | Live fretboard/hand-position HUD prototype, and the source behind the opt-in `--video-backend fretcam`. |
| [`docs/NARRATIVE.md`](docs/NARRATIVE.md) | The project story: what was hard, what worked, what's next. |
| [`docs/DEMO/blog-post.md`](docs/DEMO/blog-post.md) | Source of the [published blog post](https://pgil256.github.io/blog/posts/tabvision/). |
| [`docs/DEMO/`](docs/DEMO/) | Portfolio assets: architecture brief, per-tier examples, demo path. |
| [`docs/EVAL_REPORTS/`](docs/EVAL_REPORTS/) | Every accuracy claim's reproducible run. |
| [`SPEC.md`](SPEC.md) · [`LICENSES.md`](LICENSES.md) · [`docs/DECISIONS.md`](docs/DECISIONS.md) | Spec, license map, decision log. |
| `tabvision-server/` · `tabvision-client/` · `web-client/` | **v0 (frozen)** — the original Electron + Flask desktop demo. Kept as prior art and porting source, not as the v1 shipping artifact. |

## License posture

TabVision is a personal, non-commercial application (SPEC §1.5, amended
2026-07-20). That posture is what makes the current default pipeline possible,
and it has a consequence worth stating plainly:

- Code dependencies in the default path stay permissive with no copyleft:
  `highres-ensemble` audio (MIT), ffmpeg, numpy. Copyleft lives only in opt-in
  extras (the `vision` extra's YOLO detector is AGPL-3.0, the `render` extra's
  Guitar Pro writer is LGPL-3.0). CI enforces this through
  `scripts/check_default_licenses.py`.
- Model artifacts are not uniformly permissive. The classical/nylon route loads
  `gaps-v1` / `gaps-seq-v1`, which derive from the GAPS train split and inherit
  CC-BY-NC-SA-4.0. They are reachable from the default `auto` path when a
  session is classical, so the default pipeline is not commercially
  redistributable as it stands; those two priors would have to be rebuilt from
  permissive data first.

[`LICENSES.md`](LICENSES.md) labels every NC-derived artifact so that a future
commercialization knows what to replace. Private and user-supplied recordings
stay banned from all training, eval and labeling roles.

## Status

v1.0.0 is the tagged release: acoustic, audio-only, cut against the acceptance
gate above. `main` has moved on since. An accuracy program added the physics
string-evidence channel (+0.05 to +0.07 aggregate, +0.13 on single-line
development clips), capo sessions now route to a covariant prior, and a Windows
desktop shell and live FretCam HUD landed alongside the CLI.

Next levers, roughly in order of measured promise: the personal position prior
above, whose value accrues with use rather than with new research; extending
string evidence past the 22% of notes whose partials are currently readable;
the note-detection buckets, where `missed_onset` and `extra_detection` are a
third of remaining loss and have had the least investment; a technique
detector, greenfield from a measured 0.00; and the spend-gated electric
fine-tune for v2. Tracked in
[`docs/NARRATIVE.md`](docs/NARRATIVE.md) and
[`docs/parallel-program-state.md`](docs/parallel-program-state.md).
