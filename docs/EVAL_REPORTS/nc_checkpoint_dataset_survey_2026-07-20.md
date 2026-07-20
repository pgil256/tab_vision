# NC checkpoint + dataset survey — Program N0/S0 evidence — 2026-07-20

**Context.** User directive (chat, 2026-07-20): execute (1) the pretrained-NC-
checkpoint second-opinion program and (2) the DadaGP/SynthTab-scale training
program. Both were unlocked by the SPEC §1.5 personal non-commercial amendment
(DECISIONS.md 2026-07-20): CC-BY-NC[-SA] datasets/weights are now admissible in
the shipping default and as training substrate, labeled NC in LICENSES.md.
This report is the shared Phase 0 (survey/identification) evidence packet for
the program plan at
`docs/plans/2026-07-20-nc-second-opinion-and-synthtab-program.md`.

**Method.** Targeted web survey (search + primary-source fetches of HF model
cards/API, GitHub READMEs, arXiv) plus direct artifact inspection: HF API file
listing of the backend's checkpoint repo and a ranged-download parse of the
`guitar_kroma.safetensors` header (tensor names/shapes/params). No repo code,
production config, or dependency changed. All claims below cite their source;
nothing is taken from memory of prior audits without re-verification.

## Finding 1 — the backend's own HF repo has an unused third guitar checkpoint

`xavriley/midi-transcription-models` (MIT, lastModified **2026-03-13** — after
our 2026-05-05 license audit pin) contains, per the HF API file listing:

| file | status in TabVision |
|---|---|
| `guitar-gaps.pth` | in use (ensemble member `gaps`) |
| `guitar-fl.pth` | in use (ensemble member `fl`) |
| **`guitar_kroma.safetensors`** | **never referenced — new since audit** |
| `piano.pth`, `filosax_25k.pth`, `filobass_20000_iterations.pth`, `note_F1=…pth`, `piano_kroma.safetensors`, `model.safetensors` | other instruments / not applicable |

`guitar_kroma.safetensors` header probe (first 128 KiB, safetensors JSON
index; file size 49,360,574 bytes):

- 316 tensors; **24,663,918 parameters**.
- Module names: `frame_model`, `reg_onset_model`, `reg_offset_model`,
  `velocity_model` (72 tensors each), `frame_gru`, `reg_onset_gru`, `bn0`,
  `frame_fc`, `reg_onset_fc`, `spectrogram_extractor`, `logmel_extractor`.
- This is exactly the pinned high-resolution regression-CRNN family the
  `hf_midi_transcription` package loads (the four head names match
  `_POSTERIOR_KEYS` in `tabvision/tabvision/audio/highres.py`), with no pedal
  heads — i.e. **architecture-compatible with the existing backend**; only a
  safetensors→`.pth` state-dict conversion is needed because the pinned
  package's `checkpoint_path` loader expects `.pth`.
- No README/config documentation of its training data exists on the card or in
  `xavriley/hf_midi_transcription` (the repo `config.json` belongs to the
  saxophone default). Training-corpus identity is unknown pending N1
  behavioral comparison; the "kroma" tag also appears as `piano_kroma`, so it
  is a model-generation label, not a guitar-specific corpus claim.

**Verdict: primary Program N candidate.** Same license (MIT), same
architecture family, same runtime; the registered two-member ensemble
machinery (`checkpoint_ensemble.py`, `ensemble_v1.json`) extends naturally.
Cost $0.

## Finding 2 — SynthTab is confirmed as the Program S substrate (no request gate)

From `github.com/yongyizang/SynthTab` README (ICASSP 2024, arXiv:2309.09085)
and `synthtab.dev`:

- **License: CC BY-NC 4.0** — admissible under amended §1.5, label NC.
- 60,000 tracks rendered from DadaGP tablature with commercial guitar
  plugins; acoustic and electric subsets; ~2 TB total, but **packaged as
  per-component zips each < 50 GB** ("download only the parts you need").
- **Symbolic annotations are separable and small:**
  `all_jams_midi_V2_60000_tracks.zip` ≈ **1 GB** (JAMS + per-string MIDI) —
  this alone carries the DadaGP-derived string+fret+onset sequences needed for
  S1 symbolic scale-up.
- A small **Dev set** exists for pipeline bring-up (UR Box:
  `rochester.app.box.com/v/SynthTab-Dev`; full set `…/v/SynthTab-Full`, Baidu
  mirror password `gjwq`).
- The repo also ships baseline **TabCNN / TabCNNx4 models and embeddings**
  (`demo_embedding` folder) — per-string outputs, i.e. a candidate for
  `TABVISION_STRING_EVIDENCE`-style second opinions; whether actual weight
  files are downloadable is to be confirmed at S0 acquisition.

**DadaGP itself is request-gated** (email Dadabots / P. Sarmento per the
ISMIR 2021 paper page) with research-use terms. It is **not required** for
S1: SynthTab's JAMS re-distribution already provides the derived per-string
sequences under CC-BY-NC. Requesting DadaGP directly is an optional user
action if raw GP token format ever becomes necessary.

## Finding 3 — no public MIDI-to-Tab weights; it is the S1b blueprint, not a shortcut

"MIDI-to-Tab: Guitar Tablature Inference via Masked Language Modeling"
(Edwards, Riley, Sarmento, Dixon — ISMIR 2024, arXiv:2408.05024, paper
CC-BY-NC-SA): encoder-decoder Transformer assigning string/fret to note
sequences, **pretrained on DadaGP (25k+ tabs)**, 94.35% next-note validation
accuracy. Multiple searches found **no public code or weights**. Consequence:
S1b implements originally (per the repo's architecture-audit rule), using the
paper only as design guidance, on the SynthTab JAMS substrate.

## Finding 4 — general-purpose AMT second opinions exist but are second-line

| candidate | license | notes | verdict |
|---|---|---|---|
| **MuScriptor-large** (`MuScriptor/muscriptor-large`, Kyutai/Mirelo 2026, arXiv:2607.08168) | weights **CC-BY-NC 4.0** (now admissible) | 1.3B decoder-only; pip-installable; multi-instrument (MT3_FULL_PLUS taxonomy incl. guitar); onset F1 60.4% on its own 372-track multi-instrument test — no guitar-specific number | second-line: evaluate only as offline second opinion; 1.3B on laptop CPU likely breaks the 5-min/60-s budget for shipping auto |
| **YourMT3+** (`mimbres/YourMT3`) | code GPL-3.0; weights license unverified | GuitarSet in training mix (in-domain), T5-style | third-line: process-isolated eval only if N2 leaves headroom; verify weights license first |
| trimplexx/music-transcription | MIT | CRNN, 0.87 MPE F1 claim; published weights unconfirmed | not prioritized |

## Gate outcomes

- **N0 (survey) — PASS**: ≥1 viable $0 drop-in candidate (`guitar_kroma`,
  architecture-verified) plus two NC-admissible second-line candidates.
  Branch: proceed to N1 (kroma conversion + smoke + dev complementarity).
- **S0-survey — PASS**: substrate confirmed (SynthTab, CC-BY-NC 4.0,
  no request gate), symbolic slice ≈1 GB, audio partitionable <50 GB/zip.
  Branch: S0-acquisition (download JAMS + Dev set, SHA-256 manifest,
  parse spot-check, LICENSES.md NC rows) is the next executable step. The
  UR Box `/v/` share links are browser-served; acquisition may need an
  interactive browser session rather than curl.

## Sources

- HF API + card: `huggingface.co/api/models/xavriley/midi-transcription-models`;
  `huggingface.co/xavriley/midi-transcription-models`; ranged fetch of
  `resolve/main/guitar_kroma.safetensors` (header parse, this report).
- Package: `github.com/xavriley/hf_midi_transcription` (MIT; `checkpoint_path`
  accepts local `.pth` or `user/repo/file`).
- SynthTab: `github.com/yongyizang/SynthTab`; `synthtab.dev`;
  arXiv:2309.09085.
- DadaGP: arXiv:2107.14653; ISMIR 2021 paper (QMUL repository copy).
- MIDI-to-Tab: arXiv:2408.05024.
- MuScriptor: `huggingface.co/MuScriptor/muscriptor-large`; arXiv:2607.08168.
- YourMT3: `github.com/mimbres/YourMT3`; arXiv:2407.04822.
