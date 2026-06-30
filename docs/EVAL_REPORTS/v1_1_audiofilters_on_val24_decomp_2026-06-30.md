# Tab F1 error decomposition

## Aggregate (all tiers)

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 932 | — |
| wrong_position_same_pitch | 401 | 19.7% |
| pitch_off | 171 | 8.4% |
| timing_only | 7 | 0.3% |
| missed_onset | 1392 | 68.2% |
| extra_detection | 69 | 3.4% |

## Per-tier breakdown

| Tier | correct | wrong_position_same_pitch | pitch_off | timing_only | missed_onset | extra_detection |
|---|---|---|---|---|---|---|
| clean_acoustic_single_line | 340 | 272 | 6 | 5 | 115 | 42 |
| clean_acoustic_strummed | 592 | 129 | 165 | 2 | 1277 | 27 |

