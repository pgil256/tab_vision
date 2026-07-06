# Tab F1 error decomposition

## Aggregate (all tiers)

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 2000 | — |
| wrong_position_same_pitch | 548 | 52.0% |
| pitch_off | 117 | 11.1% |
| timing_only | 39 | 3.7% |
| missed_onset | 199 | 18.9% |
| extra_detection | 150 | 14.2% |

## Per-tier breakdown

| Tier | correct | wrong_position_same_pitch | pitch_off | timing_only | missed_onset | extra_detection |
|---|---|---|---|---|---|---|
| clean_acoustic_single_line | 384 | 296 | 8 | 7 | 43 | 50 |
| clean_acoustic_strummed | 1616 | 252 | 109 | 32 | 156 | 100 |

