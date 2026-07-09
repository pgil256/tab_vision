# Per-tier examples — ground truth vs. TabVision output

*SPEC §7 Phase 9 deliverable 6b: one worked example per acoustic difficulty
tier, ground truth vs. model output.* The two v1 acoustic tiers are
**single-line** (one note at a time — melody/solo) and **strummed** (chords).
Electric tiers are **v2** (out of v1 scope, SPEC §1.4.1) so there is no v1
example for them.

Both examples are the same piece by the same player, so the only variable is
texture: `05_BN1-129-Eb` from **GuitarSet** (held-out player 05 — the formal
acceptance validation player), `_solo` (single-line) and `_comp` (strummed).
Decoded with the shipped default config: `--audio-backend highres`
(`xavriley/midi-transcription-models`, `guitar-gaps.pth`), `--position-prior
guitarset-v1`, audio-only. Scored with `tabvision.eval.metrics` at the
acceptance onset tolerance (50 ms).

## The numbers

| Tier | Clip | Tab F1 | Precision | Recall | Pitch+onset F1 | **String-assignment gap** |
|---|---|---:|---:|---:|---:|---:|
| Single-line | `05_BN1-129-Eb_solo` | **0.333** | 0.326 | 0.341 | 0.933 | **+0.600** |
| Strummed | `05_BN1-129-Eb_comp` | **0.946** | 0.940 | 0.953 | 0.973 | +0.027 |

*Pitch+onset F1 scores the same notes ignoring string/fret (`event_f1`); the
**gap** is how much accuracy is lost purely to putting a correctly-heard note on
the wrong string.*

**This one table is the whole project thesis, measured.** On the single-line
clip TabVision hears **93%** of the notes correctly (pitch + timing) but only
**33%** land on the right string/fret — a **0.600** gap that is *entirely*
string assignment, not missed notes. On the strummed clip the gap collapses to
**0.027**: once notes are stacked into a chord, the shape pins down which
strings they must be on, and the audio-only decode gets it right. Single-line is
information-limited exactly where strummed is not. This is why v1 is scoped
audio-only and why the string-resolution story (`../NARRATIVE.md`) is the point.

## Single-line (`05_BN1-129-Eb_solo`) — TabVision output, first 8 notes

```
e|----------------|
B|--4---6---6-----|
G|5---5---7---7-8-|
D|----------------|
A|----------------|
E|----------------|
```

44 gold notes, 46 decoded. The pitches track the melody well (0.933 pitch+onset
F1); the errors are overwhelmingly the same pitch placed one string over — e.g.
a note that is really 5th-string/fret-7 decoded as 4th-string/fret-2. Audio
cannot distinguish those; only the fretting hand can, and v1 does not watch it
(the video lever was built and **refuted** — `../NARRATIVE.md`).

## Strummed (`05_BN1-129-Eb_comp`) — TabVision output, first 8 notes

```
e|----------------|
B|6-----6-------6-|
G|----5---5-------|
D|--5-------5-----|
A|----------------|
E|------------6---|
```

148 gold notes, 150 decoded, Tab F1 **0.946**. Chord shapes constrain the
positions, so the same audio-only decode that struggled on the solo is nearly
exact here.

## Honest caveats

- **These are single clips**, chosen to be the *same piece* across tiers for a
  clean contrast. Per-clip Tab F1 varies widely — single-line especially,
  because it is information-limited. The **acceptance numbers are the 60-clip
  aggregate** over held-out player 05: single-line **0.523**, strummed
  **0.676**, aggregate **0.600** (`../EVAL_REPORTS/v1_acceptance_2026-06-03.md`).
  This solo (0.333) is a below-median single-line clip; this comp (0.946) is an
  above-median strummed one. Both are shown as-is.
- **Ground truth is not committed.** GuitarSet is CC-BY-4.0 and, by project
  convention (`../../LICENSES.md`), its content lives only in the local data
  root, not the repo. Only TabVision's own output and the derived metrics appear
  here.

## Reproduce

With GuitarSet under `$TABVISION_DATA_ROOT` (see `../../tabvision/README.md`):

```bash
# TabVision output for either clip
tabvision transcribe \
  "$TABVISION_DATA_ROOT/guitarset/audio_mono-mic/05_BN1-129-Eb_solo_mic.wav" \
  --audio-backend highres --no-video --no-preflight --format ascii

# Ground truth + Tab F1 are produced by the eval harness
python -m scripts.eval.run --scope smoke
```
