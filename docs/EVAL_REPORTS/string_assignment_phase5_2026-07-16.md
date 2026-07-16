# Sequential Tab F1 Phase 5: direct per-string gold-pitch gate

## Frozen data and license design

- Acoustic acceptance core: GuitarSet microphone audio and hex-derived JAMS labels, CC-BY-4.0, players 00-04 only. Player 05 was not read.
- Guitar-TECHS remains a separate CC-BY-4.0 electric-domain track and was not mixed into this model.
- GOAT was excluded because no dataset download/license suitable for shipped derived weights was found. SynthTab, GAPS, and private data were excluded.
- The architecture and training implementation are original project code; no external model source was copied.
- The complete pre-fit protocol is frozen in `docs/plans/2026-07-16-tab-f1-phase5-data-license-design.md`.

## Fixed model and evaluation

- Model: shared three-block convolutional encoder with six onset heads, six frame/pitch heads, global-pitch head, occupancy head, and duplicate-pitch inhibition; `69,670` trainable parameters.
- Input: 512 ms event window (64 ms before, 448 ms after), resampled to 16 kHz, 64 log-mel bands, 512-sample FFT, 128-sample hop.
- Examples: 35,959 frozen production-equivalent pitch-correct ambiguous events.
- Validation: five player-held-out folds. Each outer fold selects between only the two frozen learning-rate/weight-decay settings on the next player, with the outer player excluded from all fitting and selection.
- Primary score: direct six-string event logit plus player-held-out position-prior log probability at fixed weight 1.0.

## OOF gold-pitch result

| condition | ambiguous top-1 | top-3 | wrong rate | delta vs best previous |
|---|---:|---:|---:|---:|
| production_prior | 0.6548 | 0.9967 | 0.3452 | -0.0072 |
| best_previous_contextual_timbral | 0.6621 | 0.9998 | 0.3379 | +0.0000 |
| direct_per_string_only | 0.4948 | 0.9994 | 0.5052 | -0.1673 |
| direct_per_string_plus_oof_position | 0.5920 | 0.9998 | 0.4080 | -0.0700 |

### Player folds

| held player | examples | Phase 4 best | direct only | direct + prior | delta |
|---:|---:|---:|---:|---:|---:|
| 00 | 7,947 | 0.6137 | 0.5009 | 0.5548 | -0.0589 |
| 01 | 8,560 | 0.6417 | 0.4857 | 0.5696 | -0.0721 |
| 02 | 5,671 | 0.6732 | 0.5399 | 0.6359 | -0.0374 |
| 03 | 7,053 | 0.6960 | 0.4833 | 0.6046 | -0.0915 |
| 04 | 6,728 | 0.7001 | 0.4729 | 0.6144 | -0.0856 |

### Nested selections

| outer player | inner validation | learning rate | weight decay |
|---:|---:|---:|---:|
| 00 | 01 | 1e-03 | 1e-05 |
| 01 | 02 | 1e-03 | 1e-05 |
| 02 | 03 | 1e-03 | 1e-05 |
| 03 | 04 | 3e-04 | 1e-04 |
| 04 | 00 | 1e-03 | 1e-05 |

## Gate decision

- Required top-1: best previous `0.6621` plus `0.05` = `>= 0.7121`.
- Observed primary top-1: `0.5920` (`-0.0700`).
- Player folds above their Phase 4 comparator: `0/5`.
- Gold-pitch gate: **FAIL**.
- Decision: **FAIL: close this direct-model branch. Do not open player 05, integrate the model, enlarge it, or replace the accepted high-resolution backend.**

## Runtime, reproducibility, and production safety

- Feature cache: `299179008` bytes; target cache: `353975` bytes; cache hit `false`.
- Uncached feature extraction: `35.510 s` (`0.233 s` per source minute).
- OOF training/evaluation: `925.161 s`; total command: `965.231 s`; peak process working set `1313316864` bytes.
- Deterministic complete rerun: **PASS**; prediction SHA-256 `50c0d976b8750e9e685c4205fe66c27bc2b53ae0e94ce7bb6dbe1518bcc9a14f`; model SHA-256 `213dbb122b60311e0282725d4c6a2ca4d62dbc8cfb1becd23eed786dfd80ef8e`.
- No model artifact was registered and no runtime path was changed at this gate. Until the real-event gate passes, shipping artifact size and added automatic latency are zero; onset, pitch, Tab events, and all routing behavior are unchanged.

## Reproduction

```powershell
cd tabvision
.\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase5
```
