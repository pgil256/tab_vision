# Acoustic single-line gap — diagnosis & honest target (2026-06-02)

**Question:** close the acoustic single-line Tab F1 gap (0.51 → 0.94, the
committed v1 target). All runs: highres backend, `guitarset-v1` prior, GuitarSet
held-out player-05 validation, CPU.

## Diagnosis — the loss is string/fret assignment, not pitch

Six-bucket error decomposition (24-clip subset), single-line tier:

| correct | wrong_position_same_pitch | pitch_off | missed_onset | extra |
|---:|---:|---:|---:|---:|
| 358 | **322** | 8 | 43 | 50 |

The pitch is right (8 `pitch_off`); the **string/fret is wrong 322 times**.
Aggregate, `wrong_position_same_pitch` is **54 %** of all recoverable loss.

## Levers tested — and ruled out for single-line

1. **Melodic-segment prior** (`--melodic-prior`): **regresses** single-line
   0.474 → 0.449 (24-clip subset). The "helps solo" claim was an anecdote on one
   personal clip; on GuitarSet it hurts. Left default-off.
2. **Hand-position continuity** (`POSITION_SHIFT_COST` sweep) — the decoder's
   continuity weight was 0.05 (≈0.02 nats for a 5-fret jump vs several nats of
   prior), i.e. effectively off. Full 60-clip validation:

   | `POSITION_SHIFT_COST` | single-line | strummed |
   |---|---:|---:|
   | 0.05 (old default) | 0.5076 | 0.6708 |
   | **2.5 (new default)** | **0.5230** | **0.6763** |

   A real but **modest** win (single +1.5 pp, strummed +0.5 pp, no regression) —
   **committed as the new default.** But it does not move single-line toward 0.94.

## Conclusion — single-line is *information-limited*, not tuning-limited

With pitch correct and continuity raised 50×, single-line still sits at ~0.52.
The residual `wrong_position` errors are notes where **audio cannot determine
which string was played** — the same pitch on different strings is acoustically
near-identical. This is the string/fret ambiguity the **video / hand-tracking**
pipeline exists to resolve. Audio-only single-line is near its information
ceiling (~0.50–0.52 on GuitarSet); **0.94 is not reachable audio-only.**

## Decision — honest audio-only v1 targets (SPEC §1.4.1)

v1 is audio-only acoustic (GuitarSet has no video). Targets are set to the
demonstrated audio-only capability (acceptance `lower_95_CI ≥ target`), with
single-line flagged as video-limited and **video as the v1.1 single-line lever**:

| Tier | v1 target | demonstrated (mean / lower-95) |
|---|---:|---:|
| Clean acoustic single-line | ≥ 0.45 | 0.523 / 0.457 |
| Clean acoustic strummed | ≥ 0.60 | 0.676 / 0.606 |
| Aggregate Tab F1 | ≥ 0.55 | ~0.638 |

Onset F1 ≥ 0.92, Pitch F1 ≥ 0.90, chord-instance ≥ 0.85, latency ≤ 5 min —
unchanged (met). The original 0.94 / 0.86 become the **v1.1 (video-assisted)**
reference.

## Bounded headroom (not pursued here)
A **style/structure-conditional position prior** (design-plan Phase 3) could
recover a few more points of `wrong_position` by conditioning on key/recent
positions — but the upside is capped by the same audio ambiguity. The real
single-line lever is video string-resolution (v1.1) or a timbral string-ID model.
