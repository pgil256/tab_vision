# String assignment Phase 2: free timbral signal probe

Date: 2026-07-14

## Method

- Data: GuitarSet players 00–04 only; player 05 was not read.
- Examples: **35959** ambiguous, pitch-correct high-resolution events from the frozen Phase 0 note table.
- Split: five leave-one-player-out folds. Every reported prediction is OOF.
- Seed: **2714**.
- Window: 512 ms at 16 kHz, -64 ms/+448 ms around the detected onset, with zero padding and training jitter sampled from the empirical capped ±50 ms onset-error distribution.
- Audio model: fixed 512/128 log-STFT, three 16/32/64-channel Conv2d+GroupNorm+SiLU+pool blocks, 96/48 candidate MLP.
- Parameters: **35905** (<250,000).
- Augmentation: gain, time-domain spectral tilt, broadband noise, mild compression, and onset shift.
- Sampling balances player, string, pitch region, and solo/comp mode.
- Frozen Phase 0 source: `string_assignment_phase0_2026-07-14_notes.csv`.
- Frozen source SHA-256: `c0f7b0861d5fe910cddc2d95f650687d12e682187b66d543eb9f7496f85f2f39`.
- Training config: feature epochs=12, audio epochs=2, train cap=16000, batch size=32.

## OOF result

| held-out player | examples | prior-only baseline | feature-only+prior | audio+prior | audio delta |
|---:|---:|---:|---:|---:|---:|
| 00 | 7947 | 0.6451 | 0.5494 | 0.5973 | -0.0478 |
| 01 | 8560 | 0.5939 | 0.5861 | 0.5940 | +0.0001 |
| 02 | 5671 | 0.6313 | 0.6253 | 0.6334 | +0.0021 |
| 03 | 7053 | 0.7235 | 0.6372 | 0.6671 | -0.0564 |
| 04 | 6728 | 0.6916 | 0.6317 | 0.6889 | -0.0027 |

- Best prior-only candidate top-1: **0.6548**
- Feature-only + prior: **0.6027** (`-0.0521`, weight=0.5, temperature=1.30)
- Compact audio + prior: **0.6331** (`-0.0218`, weight=1.0, temperature=1.50)
- Worst player-fold delta: **-0.0564**
- Calibrated ECE: **0.0597**
- Mean posterior entropy: **0.9068**
- Predicted-string distribution: s0=0.011, s1=0.109, s2=0.218, s3=0.332, s4=0.210, s5=0.120
- Training/evaluation wall time: **6.9 min** on CPU

## Free-probe gate

Required: audio delta ≥ +0.05, every player fold ≥ -0.03, and calibrated non-collapsed posteriors.

**FAIL — do not start paid training and do not enlarge the model.**
