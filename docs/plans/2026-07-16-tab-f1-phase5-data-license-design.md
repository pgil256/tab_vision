# Sequential Tab F1 Phase 5A — data, license, and model freeze

**Date:** 2026-07-16
**Status:** Frozen before feature extraction or model fitting
**Parent plan:** `docs/plans/2026-07-15-tab-f1-accuracy-sequential-plan.md`, Phase 5

## Entry and scope

Phases 1–4 plateaued below the desired automatic string-assignment accuracy, and
the user explicitly approved entering the larger Phase 5 training project on
2026-07-16. This phase is a new direct per-string model program, not another
fusion-weight sweep. Paid/GPU work is not authorized by this approval; the first
run is local CPU-only and costs $0.

The first mandatory gate is gold-pitch ambiguous-string accuracy on players
00–04. Real-event integration and player 05 remain unopened unless that gate
passes.

## Corpus manifest and license decisions

| corpus | domain | role in Phase 5 | license/access decision |
|---|---|---|---|
| GuitarSet | clean acoustic | sole training/development core; players 00–04 | CC-BY-4.0; permitted. Local audio/JAMS only, not redistributed. |
| GuitarSet player 05 | clean acoustic | frozen confirmation, only after development gates | Reserved and excluded from feature/training manifests until authorized. |
| Guitar-TECHS | electric | separate future electric-domain track only | CC-BY-4.0, but deliberately excluded from the acoustic acceptance model. Official site: https://guitar-techs.github.io/ |
| GOAT | electric | excluded | The official paper describes 5.9 h DI plus 29.5 h amplifier-rendered audio, but as of this audit exposes no dataset download/access link or dataset-license grant. The arXiv paper license is not a data license. Excluded from training and derived weights. Paper: https://arxiv.org/abs/2509.22655 |
| SynthTab | synthetic | excluded | CC-BY-NC dataset; non-commercial data cannot enter shipping weights. |
| GAPS | classical | excluded | CC-BY-NC-SA; evaluation-only under repository policy. |
| Guitar-TECHS/EGDB/IDMT/user recordings | non-core | excluded | No cross-domain, restricted, private, or personal material enters the acoustic model. |

The evaluator must write a SHA-256 manifest for every GuitarSet input and assert
that all training/development track IDs start with players `00`–`04`. No new
data is commissioned in this phase. If later commissioned, it must be CC0 or
CC-BY and prioritize repeated positions, adjacent-string same pitches, low-E,
five/six-candidate pitches, rock/jazz, capo, multiple guitars/players, and
synchronized microphone plus per-string channels.

## Architecture/code audit

The implementation is original TabVision code using only PyTorch primitives.
No external training code, pretrained tablature weight, or model source is
copied.

- TabCNN papers may motivate the task, but no TabCNN source is reused.
- Cwitkowitz FretNet/with-inhibition repositories are MIT, but neither code nor
  architecture is reused; no published pretrained weight is available.
- `trimplexx/music-transcription` now presents an MIT license, but it is not
  reused: its larger CRNN/search setup would violate this phase's one-original-
  architecture constraint and would weaken provenance continuity.
- The existing MIT `hf-midi-transcription` checkpoint remains the accepted
  event detector and is not modified. The new network is a second opinion for
  string identity only.

## Frozen model and preprocessing

- Examples: the 35,959 frozen production-equivalent, player-held OOF,
  pitch-correct ambiguous events from Phase 0.
- Window: 512 ms (`-64 ms/+448 ms`) from the microphone waveform, resampled
  once to 16 kHz for CPU/runtime compatibility with the accepted backend.
- Input: deterministic 64-band log-mel magnitude, 512-point FFT, 128-sample hop.
- Shared encoder: three original Conv2d → GroupNorm → SiLU → 2×2 pooling blocks
  with 12/24/48 channels and adaptive global pooling.
- Heads from one shared embedding:
  - six × 88 onset logits;
  - six × 88 frame/pitch logits;
  - 88 auxiliary global-pitch logits;
  - six string-occupancy logits.
- Loss: onset BCE, frame BCE, global-pitch BCE, occupancy BCE, plus explicit
  duplicate-pitch inhibition penalizing predicted per-pitch string counts above
  annotated counts.
- Gold-pitch candidate score: onset logit + `0.5 ×` frame logit. Candidate
  likelihoods normalize only after masking to physically playable strings.
- Primary condition: direct likelihood × player-held OOF corpus prior, combined
  once in log space at fixed weight `1.0`. Direct-only is diagnostic.
- Parameter cap: `<5,000,000`; expected architecture is under 100,000.

## Frozen bounded optimization

- Outer folds: held players 00, 01, 02, 03, 04.
- Inner validation player: first cyclic player not equal to the outer player;
  the remaining three players train each grid candidate.
- Joint learning-rate/weight-decay grid only:
  - `(3e-4, 1e-4)`
  - `(1e-3, 1e-5)`
- Inner grid: two epochs per candidate; choose validation ambiguous-string
  accuracy, then lower learning rate as the deterministic tie break.
- Final outer model: selected setting, all four non-held players, three epochs.
- Batch size 256; seed 5519; CPU only; no early stopping, architecture search,
  augmentation sweep, threshold sweep, or fusion-weight sweep.
- Each training epoch draws exactly the training-set size with replacement using
  inverse-frequency weights over `(reference string, pitch octave, solo/comp)`;
  the epoch seed is fixed and recorded. Validation/test rows are never sampled
  or reweighted.

## Decision gates

The best previously measured contextual/timbral OOF ambiguous top-1 is frozen
at `0.6620595678411524` (Phase 4). The Phase 5 gold-pitch gate therefore requires
the primary direct-plus-prior OOF condition to reach at least
`0.7120595678411524` (`+0.05`) before any real-event integration.

If it fails, do not open player 05, do not map the model into runtime
`AudioEvent.fret_prior`, and do not enlarge/retrain the architecture. If it
passes, freeze the OOF configuration and proceed to the plan's real-event Tab
F1/onset/pitch/fold/runtime gates.
