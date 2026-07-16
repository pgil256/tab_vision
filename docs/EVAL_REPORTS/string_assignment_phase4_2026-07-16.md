# Sequential Tab F1 Phase 4: native-rate adjacent-string probe

## Fixed method

- Data: GuitarSet microphone WAV and hex-derived per-string JAMS labels; players 00-04 only. Player 05 was not read.
- Examples: 35,959 frozen production-equivalent OOF pitch-correct ambiguous events and 56,742 physically adjacent gold-vs-alternative pairs.
- Audio: native 44.1 kHz in this corpus (44.1/48 kHz accepted); no 16 kHz backend waveform and no upsampling.
- Window: 64 ms pre-onset plus 448 ms post-onset. Attack, short-sustain, and long-sustain spectra use fixed 4096/8192/16384 FFTs.
- Features: harmonic-envelope bands through Nyquist, spectral centroid and 85/95% rolloff, 6-18 kHz pick energy/flux, harmonic decay, inharmonicity, plus separately retained raw RMS and dB/octave spectral slope.
- Model: five separate class-balanced L2-linear logistic models, one per adjacent string pair; L2=1.0, 50 deterministic Newton steps.
- Fusion: pair log-odds integrate into candidate potentials and are added to the player-held OOF position prior at the fixed weight 1.0. No grid, temperature, threshold, or held-player selection.

## OOF ambiguous-note result

| condition | ambiguous top-1 | top-3 | wrong rate | delta vs production |
|---|---:|---:|---:|---:|
| `production_prior` | 0.6548 | 0.9967 | 0.3452 | +0.0000 |
| `oof_position_prior` | 0.5973 | 0.9998 | 0.4027 | -0.0575 |
| `native_audio_only` | 0.6503 | 0.9996 | 0.3497 | -0.0045 |
| `oof_position_plus_native_audio` | 0.6621 | 0.9998 | 0.3379 | +0.0072 |

Paired clip-stratified 10,000-resample interval for the combined delta: `[-0.0152, +0.0291]`.

### Player folds

| held player | examples | production | position | audio only | combined | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 00 | 7947 | 0.6451 | 0.5517 | 0.6236 | 0.6137 | -0.0315 |
| 01 | 8560 | 0.5939 | 0.5536 | 0.6603 | 0.6417 | +0.0478 |
| 02 | 5671 | 0.6313 | 0.6540 | 0.6724 | 0.6732 | +0.0420 |
| 03 | 7053 | 0.7235 | 0.6217 | 0.6661 | 0.6960 | -0.0275 |
| 04 | 6728 | 0.6916 | 0.6333 | 0.6339 | 0.7001 | +0.0085 |

### Adjacent hard-negative diagnostic

| string pair | pairs | higher-string gold | position pair acc | audio pair acc |
|---|---:|---:|---:|---:|
| 0-1 | 5283 | 0.8690 | 0.8823 | 0.8391 |
| 1-2 | 10917 | 0.7748 | 0.8439 | 0.8547 |
| 2-3 | 15642 | 0.6426 | 0.7477 | 0.7744 |
| 3-4 | 15646 | 0.5183 | 0.6669 | 0.6854 |
| 4-5 | 9254 | 0.4382 | 0.6755 | 0.8104 |

## Gate decision

- Required aggregate lift: `>= +0.05`; observed `+0.0072`.
- Required worst fold: `>= -0.03`; observed `-0.0315`.
- Free-signal gate: **FAIL**.
- Decision: **FAIL: close the compact high-frequency timbral path; do not enlarge the model or open player 05.**

## Runtime, reproducibility, and routing safety

- Feature cache: `6056155` bytes; extraction wall `240.485 s`; cache hit `false`.
- Feature cache SHA-256 `13955c3209912038aed27a5a041b3cb1f3687f134948ee278f7d30088764ccb0`; descriptor-array SHA-256 `f377a4f3a1a5435d9d1a04503707266e04769ebb8552a5c486e621035177a423`.
- Descriptor extraction rate: `1.579 s` per 60 seconds of source audio on the first uncached run.
- Two complete OOF fits/evaluations: `9.227 s`; peak process working set `118198272` bytes.
- Deterministic rerun: **PASS**; prediction SHA-256 `a8fb946ebdf06f7a2f73c543dad92dfd8c39152434b14ca83d7242a857b57a10`; model SHA-256 `b299366f6946a61205f1f79b1db58f572379ee3ea696cf93a45c67e12cb73f6f`.
- Shipping artifact size and added automatic inference time are both zero: the failed probe is not registered or integrated. Onset and pitch events are unchanged. Automatic clean-acoustic, GAPS classical, Guitar-TECHS electric, capo, alternate-tuning, distorted, and non-high-resolution routes are unchanged; their Phase 4 metric/event delta is exactly zero.

## Reproduction

```powershell
cd tabvision
.\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase4
```
