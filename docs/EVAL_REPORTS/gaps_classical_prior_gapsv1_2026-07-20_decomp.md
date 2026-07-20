# Tab F1 error decomposition

## Aggregate (all tiers)

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 11714 | — |
| wrong_position_same_pitch | 2951 | 39.0% |
| pitch_off | 554 | 7.3% |
| timing_only | 65 | 0.9% |
| missed_onset | 795 | 10.5% |
| extra_detection | 3207 | 42.4% |

## Per-tier breakdown

| Tier | correct | wrong_position_same_pitch | pitch_off | timing_only | missed_onset | extra_detection |
|---|---|---|---|---|---|---|
| clean_acoustic_single_line | 11714 | 2951 | 554 | 65 | 795 | 3207 |

## pitch_off semitone-delta histogram

Signed delta = predicted − gold MIDI pitch per `pitch_off` event. Classes: octave (|Δ| ≡ 0 mod 12), harmonic (|Δ| ≡ 5/7 mod 12), semitone (|Δ| ≤ 2), other.

| Delta (semitones) | Count | Share of pitch_off | Class |
|---:|---:|---:|---|
| -31 | 1 | 0.2% | harmonic |
| -29 | 2 | 0.4% | harmonic |
| -24 | 8 | 1.4% | octave |
| -23 | 1 | 0.2% | other |
| -21 | 3 | 0.5% | other |
| -19 | 7 | 1.3% | harmonic |
| -18 | 4 | 0.7% | other |
| -17 | 3 | 0.5% | harmonic |
| -16 | 5 | 0.9% | other |
| -15 | 7 | 1.3% | other |
| -14 | 6 | 1.1% | other |
| -13 | 1 | 0.2% | other |
| -12 | 29 | 5.2% | octave |
| -11 | 2 | 0.4% | other |
| -10 | 5 | 0.9% | other |
| -9 | 6 | 1.1% | other |
| -8 | 9 | 1.6% | other |
| -7 | 15 | 2.7% | harmonic |
| -6 | 8 | 1.4% | other |
| -5 | 16 | 2.9% | harmonic |
| -4 | 17 | 3.1% | other |
| -3 | 14 | 2.5% | other |
| -2 | 8 | 1.4% | semitone |
| -1 | 17 | 3.1% | semitone |
| +1 | 36 | 6.5% | semitone |
| +2 | 9 | 1.6% | semitone |
| +3 | 27 | 4.9% | other |
| +4 | 44 | 7.9% | other |
| +5 | 35 | 6.3% | harmonic |
| +6 | 12 | 2.2% | other |
| +7 | 25 | 4.5% | harmonic |
| +8 | 16 | 2.9% | other |
| +9 | 9 | 1.6% | other |
| +10 | 10 | 1.8% | other |
| +11 | 4 | 0.7% | other |
| +12 | 41 | 7.4% | octave |
| +13 | 6 | 1.1% | other |
| +14 | 6 | 1.1% | other |
| +15 | 19 | 3.4% | other |
| +16 | 16 | 2.9% | other |
| +17 | 3 | 0.5% | harmonic |
| +18 | 10 | 1.8% | other |
| +19 | 8 | 1.4% | harmonic |
| +20 | 5 | 0.9% | other |
| +21 | 2 | 0.4% | other |
| +22 | 1 | 0.2% | other |
| +24 | 6 | 1.1% | octave |
| +26 | 1 | 0.2% | other |
| +27 | 7 | 1.3% | other |
| +28 | 2 | 0.4% | other |

### Class summary (aggregate + per tier)

| Scope | octave | harmonic | semitone | other | total |
|---|---:|---:|---:|---:|---:|
| all tiers | 84 | 115 | 70 | 285 | 554 |
| clean_acoustic_single_line | 84 | 115 | 70 | 285 | 554 |

