# Tab F1 error decomposition — A10 pitch_off semitone-delta instrumentation (val24)

**Date:** 2026-07-06
**Branch:** `v1.1/a14-a10-probes`
**Config:** accepted default — `highres` + `guitarset-v1` (+ coupled sequence
prior), val24 manifest. Baseline parity exact vs the accepted numbers
(single-line 0.5140 / lo-95 0.4144; strummed 0.7953 — see
`a10_val24_baseline_2026-07-06.md`).
**Status:** **pitch_off FORMALLY CLOSED as a fix target.** The 11.2% bucket has
no dominant fixable mode: octave errors (the classic cheap fix) are **6/117**
(~0.6% of total loss), harmonic 30, semitone 20, **other 61 (52%)** — a diffuse
−19..+24 spread with 93% of events in strummed (dense-cluster near-miss
pairings, not a detector pitch pathology). No intervention class clears the
cost bar. The instrumentation ships in the harness
(`tabvision/eval/error_decomposition.py`), so any future backend change that
re-concentrates this histogram (e.g. octave-heavy) reopens the bucket with
zero extra work.

## Aggregate (all tiers)

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 2007 | — |
| wrong_position_same_pitch | 541 | 51.7% |
| pitch_off | 117 | 11.2% |
| timing_only | 39 | 3.7% |
| missed_onset | 199 | 19.0% |
| extra_detection | 150 | 14.3% |

## Per-tier breakdown

| Tier | correct | wrong_position_same_pitch | pitch_off | timing_only | missed_onset | extra_detection |
|---|---|---|---|---|---|---|
| clean_acoustic_single_line | 392 | 288 | 8 | 7 | 43 | 50 |
| clean_acoustic_strummed | 1615 | 253 | 109 | 32 | 156 | 100 |

## pitch_off semitone-delta histogram

Signed delta = predicted − gold MIDI pitch per `pitch_off` event. Classes: octave (|Δ| ≡ 0 mod 12), harmonic (|Δ| ≡ 5/7 mod 12), semitone (|Δ| ≤ 2), other.

| Delta (semitones) | Count | Share of pitch_off | Class |
|---:|---:|---:|---|
| -19 | 4 | 3.4% | harmonic |
| -15 | 1 | 0.9% | other |
| -13 | 1 | 0.9% | other |
| -12 | 3 | 2.6% | octave |
| -11 | 2 | 1.7% | other |
| -10 | 2 | 1.7% | other |
| -9 | 2 | 1.7% | other |
| -8 | 5 | 4.3% | other |
| -7 | 3 | 2.6% | harmonic |
| -6 | 2 | 1.7% | other |
| -5 | 11 | 9.4% | harmonic |
| -4 | 1 | 0.9% | other |
| -3 | 4 | 3.4% | other |
| -2 | 3 | 2.6% | semitone |
| -1 | 10 | 8.5% | semitone |
| +1 | 6 | 5.1% | semitone |
| +2 | 1 | 0.9% | semitone |
| +3 | 7 | 6.0% | other |
| +4 | 6 | 5.1% | other |
| +5 | 6 | 5.1% | harmonic |
| +6 | 3 | 2.6% | other |
| +7 | 2 | 1.7% | harmonic |
| +8 | 4 | 3.4% | other |
| +9 | 9 | 7.7% | other |
| +10 | 5 | 4.3% | other |
| +11 | 1 | 0.9% | other |
| +12 | 2 | 1.7% | octave |
| +14 | 4 | 3.4% | other |
| +15 | 1 | 0.9% | other |
| +16 | 1 | 0.9% | other |
| +19 | 4 | 3.4% | harmonic |
| +24 | 1 | 0.9% | octave |

### Class summary (aggregate + per tier)

| Scope | octave | harmonic | semitone | other | total |
|---|---:|---:|---:|---:|---:|
| all tiers | 6 | 30 | 20 | 61 | 117 |
| clean_acoustic_single_line | 1 | 0 | 4 | 3 | 8 |
| clean_acoustic_strummed | 5 | 30 | 16 | 58 | 109 |

