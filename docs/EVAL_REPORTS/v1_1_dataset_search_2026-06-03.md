# v1.1 eval-data search + decision — 2026-06-03

**Context.** v1.1 (video string-resolution) needs an eval corpus with (a)
fretting-hand video and (b) per-note **string + fret** labels, to drive the
already-validated resolver (see `v1_1_oracle_string_probe.py`). GuitarSet and
Guitar-TECHS are audio-only, so this is the gating decision (design §6, §9). A
deep-research pass (98 agents, 16 sources, 19 adversarially-verified claims)
mapped the public-dataset landscape.

## Finding: no portfolio-clean public dataset has BOTH video AND per-string labels

The corpus space splits into two disjoint buckets — the intersection is empty.

**Per-string labels + clean license, but NO video** (synthetic-base candidates):

| Dataset | License | Why it fails |
|---|---|---|
| GuitarSet | MIT | audio-only (hex-pickup per-string labels; no video) |
| Guitar-TECHS (Zenodo 14963133) | CC-BY-4.0 | audio-only — 4 audio capture positions incl. a head-mounted *mic* (not a camera); per-string MIDI; **no video** (verified arXiv:2501.03720) |
| GOAT (ISMIR 2025) | research-only / request-gated | audio-only (Guitar Pro tabs; DI audio) |
| EGDB | author grant (eval-only) | rendered audio only; no human performance is filmed |
| IDMT-SMT-Guitar | CC-BY-NC-ND | audio-only |

**Video + per-string labels, but NOT a clean license** (real-video candidates):

| Dataset | License | Notes |
|---|---|---|
| **Kaggle "guitar-transcription-dataset" (UT-Austin)** | CC-BY-NC-SA-4.0 | **video frames + genuine string(1–6)+fret(1–20) labels**; 4.4 GB; the single closest match — fails *only* the license gate |
| GAPS (QMUL) | CC-BY-NC-SA + custom | performance video is YouTube-linked (not redistributed) + MusicXML tablature (unverified vs the performer's actual choices) |
| TapToTab | request-gated | video request-gated; the public IEEE-Dataport version is audio + pitch-only (no string) |

Primary sources: zenodo 3371780 + github marl/GuitarSet (GuitarSet); arXiv:2501.03720
(Guitar-TECHS); arXiv:2509.22655 (GOAT); arXiv:2202.09907 (EGDB); Fraunhofer IDMT
page; kaggle.com/datasets/jacksonlightfoot/guitar-transcription-dataset; arXiv:2408.08653
+ aim-qmul.github.io/GAPS (GAPS); arXiv:2409.08618 (TapToTab). Full verified report:
deep-research run `wf_d6833878-6c5`.

## Decision: use the Kaggle UT-Austin dataset as the v1.1 eval set

**License reasoning (corrects an over-strict earlier reading).** SPEC §1.5's
portfolio-clean rule governs the **shipping default pipeline**: *"every dataset
used in the shipping default pipeline must permit demonstration … Non-commercial-only
… must not be required by the default end-to-end pipeline."* TabVision's product
runs on the **user's own video** and bundles **no dataset**; datasets are used
offline for **training** (the prior) and **eval** (the acceptance number). An eval
set is downloaded to produce a metric — never shipped or redistributed — exactly
how GuitarSet and EGDB are already used (gitignored under `~/.tabvision/data`, never
committed). So **CC-BY-NC-SA is acceptable for the eval/acceptance set**: download +
measure + cite-with-attribution + don't redistribute. The deep-research brief
treated NC as disqualifying "the shipping acceptance gate," conflating *acceptance
gate* with *shipping pipeline*; that conflation is corrected here and in design §10.

**Residual caveats** (none are the license):
- Labels are per-finger *static fingerings* keyed to frames, not note-onset events
  → a derivation step is required (done in chunk-1, below).
- Single-source provenance (a UT-Austin ECE-382V term project; 25 clips / ~2k
  frames) — strong to *prove* v1.1, weaker as a headline number than a peer-reviewed
  corpus.
- Do not commit the data; note the NC provenance in the eval report; if TabVision
  is ever commercialised, revisit.

**Synthetic-from-GuitarSet remains the portfolio-clean fallback** (design §6.1) if a
fully-clean headline number is ever required.

## Chunk-1 validation (the data pipeline is locked)

`scripts/eval/v1_1_kaggle_oracle_probe.py`. The labels
(`[frame][finger] = [active, fret, their_string]`, shape `(n, 4, 3)`) are parsed
into per-note gold `TabEvent`s: a **new `(fret, string)` placement** vs the previous
frame = a note onset; **only the highest fret on a string sounds** (collapse
simultaneous same-string finger rests); `our_idx = 6 − their_string`
(audio-verified against the sounded pitch); onsets via `timestamps.csv`.
Reproducing the oracle probe on these REAL clips:

| | audio-only | + oracle (perfect hand) |
|---|---:|---:|
| 25 clips / 527 notes | **0.42** | **1.00** (every clip 1.0) |

So the dataset is eval-usable, the gold derivation is correct, and the resolver
lifts real-video clips **0.42 → 1.00** given a perfect hand signal — mirroring
GuitarSet (0.52 → 0.99). Everything up to the camera is validated.

## What remains — the MediaPipe CV chain (chunks 2–3)

The only open unknown is whether the real video → `FrameFingering` chain (MediaPipe
hand → fretboard homography → `fingertip_to_fret`) produces good-enough fingerings
on this footage:

- **Chunk 2:** install MediaPipe; PNG frame → `HandSample` → per-frame homography →
  `FrameFingering`; sanity-check detection quality on these frames (a different rig
  than the iPhone footage our detector was built for).
- **Chunk 3:** real highres audio → `AudioEvent`s (calibrate the ~+1 semitone tuning
  offset between labels and audio); `fuse(audio, real_fingerings)` vs audio-only →
  the real-video Tab F1, vs the §8 acceptance targets.

If chunk 2 lifts single-line on real video, v1.1 is proven end-to-end. If it does
not, the failure is localised to hand/fretboard **detection** on this footage (a
CV-quality problem, not the resolver) → chunk-2 robustness work.
