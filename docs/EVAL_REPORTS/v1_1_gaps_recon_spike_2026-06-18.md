# v1.1 GAPS recon spike — is the acceptance gate reachable?

**Date:** 2026-06-18
**Branch:** `v1.1/oracle-string-resolution`
**Verdict:** **GO (qualified)** — GAPS is viable as the v1.1 video-tier acceptance
corpus. Full integration is the next chunk ("chunk-5").
**Depends on:** `docs/plans/2026-06-03-v1.1-video-string-resolution-design.md`
(§6 eval-data gate, §9 decision tree).

## Why this spike

The chunk-4 conclusion (`docs/plans/2026-06-11-v1.1-audio-transcription-alignment-design.md`)
left v1.1 blocked on a corpus problem: **no available dataset had clean acoustic
audio AND real fretting-hand video AND per-string labels at once.** GuitarSet has
clean audio + string/fret but no video; UT-Austin has video + string/fret but
electric/out-of-domain audio that `highres` transcribes near-zero (raw pitch ≈ 0).

GAPS (Guitar-Aligned Performance Scores, ISMIR 2024; arXiv:2408.08653;
Zenodo 13962272) claims all three: classical-guitar audio, performance video, and
MusicXML **tablature** (string/fret). And `highres` is the GAPS-lineage model
(`guitar-gaps.pth`), so GAPS audio should be **in-domain** — the UT-Austin audio
floor likely does not apply. This spike checks the two unknowns cheaply before
committing to full integration: (1) are the labels real/parseable, and (2) is the
fretting hand actually visible to MediaPipe in GAPS-style performance video?

## Data acquired (offline eval use only; not committed/redistributed)

Zenodo is unreachable from this environment (`curl` → `000`, while GitHub/YouTube/
PyPI return 200), so the full dataset + piece→YouTube metadata could not be pulled
here. The GAPS project page hosts a complete worked sample on github.io, which was
sufficient for the spike:

- `tw1wc.xml` (MusicXML score w/ tab), `tw1wc-fine-aligned.mid` (note-level
  performance alignment), `tw1wc-syncpoints.json`, `tw1wc.wav` (49 MB) —
  from `https://aim-qmul.github.io/GAPS/static/`.
- A representative continuous classical-guitar performance video (Guitar Salon
  International — the same source family GAPS draws from), via `yt-dlp`, for the
  hand-visibility check. (The page's embedded `xifkG2tTEwU` is the GAPS **talk**
  video — slides — and must not be used for hand stats; see below.)

## Result 1 — labels are clean and derivable

Parsed `tw1wc.xml` with stdlib `xml.etree` (no `music21`/`partitura` needed):

- **1923 notes carry `(string, fret)`** (`<notations><technical><string>/<fret>`),
  strings **1–6**, frets **0–13**.
- `(string, fret) → pitch` is internally consistent: e.g. `(str1, fr12) → MIDI 76`
  (E5) ✓, `(str6, fr0) → MIDI 40` (E2) ✓, `(str2, fr5) → MIDI 64` (E4) ✓.
- The fine-aligned performance MIDI (1736 notes, 242.5 s) has **100% pitch overlap**
  with the score → onset-timed gold `(string, fret, onset)` is derivable by matching
  MIDI onsets to the score tab within onset clusters. (Score has 1923 tab notes vs
  1736 performed — repeats/ties/ornaments — so the match is per-cluster, not a blind
  zip; the 100% pitch overlap makes this tractable.)

This is the GAPS gold-derivation chunk-5 needs, and it parses with no new dependency.

## Result 2 — the fretting hand is MediaPipe-visible

Ran the repo's MediaPipe Tasks-API `HandLandmarker` (model
`~/.mediapipe/models/hand_landmarker.task`, same loader as
`tabvision/video/hand/mediapipe_backend.py`) over evenly-sampled frames.

| Video | Frames w/ ≥1 hand | Both hands | Note |
|---|---:|---:|---|
| GAPS ISMIR **talk** (`xifkG2tTEwU`) | 12/40 (30%) | 9/40 | **misleading** — 70% no-hand frames are slides |
| Representative GSI **performance** (90 s) | **38/39 (97%)** | 29/39 (74%) | continuous playing, fixed medium shot |

On the real performance, visual inspection confirms the **fretting hand is detected
on the neck/fretboard** (alongside the picking hand at the soundhole) — exactly the
signal the string-resolution chain consumes.

## Verdict and caveats

**GO (qualified).** GAPS provides clean string/fret labels + MediaPipe-detectable
fretting-hand video + (expected) in-domain audio, so the v1.1 acceptance gate is
reachable in principle. The synthetic-from-GuitarSet fallback and the §9
"data-blocked / escalate" branch are **not** needed.

Honest bounds:

1. **97% is the professional medium-shot best case.** GAPS spans 200+ performers
   with heterogeneous framing/angle/occlusion, so real per-clip video usability will
   vary. Chunk-5 must rely on the **per-clip video-confidence gate chunk-3 already
   built** (`homography_confidence`-scaled fusion weight + collapse-to-audio when
   video is sparse/weak) and likely a clip-quality filter.
2. **Labels and video were validated on different pairings.** Labels on `tw1wc`;
   video on a same-family GSI performance — not the identical clip. The exact
   piece→YouTube map lives in the **Zenodo metadata, unreachable from this
   environment.** Chunk-5 needs Zenodo access (or the dataset fetched out-of-band) to
   get matched video+label pairs at scale.
3. **Audio in-domain is expected, not yet measured.** `highres` = `guitar-gaps.pth`,
   so GAPS audio should transcribe far better than UT-Austin. Confirm first thing in
   chunk-5 by transcribing `tw1wc.wav` and scoring against the derived gold.
4. **License:** GAPS is NC / offline-eval-only (download + measure + cite; never
   commit or redistribute media) — same posture already accepted for UT-Austin.

## Implied chunk-5 (full GAPS integration)

1. Resolve dataset access (Zenodo from a connected environment, or fetch the record
   + metadata out-of-band) → matched video + MusicXML + aligned-MIDI per piece.
2. Add a `gaps_musicxml_tab` annotation parser (new format alongside
   `utaustin_tablature_npy` / `guitarset_jams`) → onset-timed gold `TabEvent`s from
   MusicXML tab + fine-aligned MIDI.
3. Video acquisition pipeline (yt-dlp, offline) + frame↔score time alignment via the
   syncpoints.
4. Confirm audio in-domain (transcribe `tw1wc.wav`); then real-chain eval (audio +
   real video) with the chunk-3 confidence gate, bootstrap CI per design §8.
5. Apply a per-clip video-quality filter; report dropped clips (no silent caps).

## Reproduce

Spike scripts (throwaway, in `~/.tabvision/cache/gaps_recon/`, not committed):
`parse_gaps.py` (MusicXML tab + MIDI cross-check) and `hand_check.py` (MediaPipe
frame sampling, Tasks API). Tooling: `yt-dlp` (installed into the venv),
`mediapipe 0.10.35`, OpenCV, ffmpeg. Sample files pulled from
`https://aim-qmul.github.io/GAPS/static/`.
