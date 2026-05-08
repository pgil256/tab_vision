# Position selector training — 2026-04-29

Input: `/home/gilhooleyp/projects/tab_vision/tabvision-server/tools/outputs/position_dataset.parquet`
Rows: 1319, events: 336, videos: 20

## Headline

- Heuristic baseline: 202/336 = **60.1%**
- Model LOOCV: 203/336 = **60.4%**
- Δ: **+0.3pp**
- Ship gate: ≥ +5pp + no per-video regression > 3pp
- **Decision: NO SHIP — Δ +0.3pp < gate +5pp**

## Per-video accuracy

| video | events | heuristic | model | Δpp |
|---|---:|---:|---:|---:|
| training-01 | 13 | 7/13 (53.8%) | 7/13 (53.8%) | +0.0 |
| training-02 | 16 | 1/16 (6.2%) | 2/16 (12.5%) | +6.2 |
| training-03 | 12 | 8/12 (66.7%) | 8/12 (66.7%) | +0.0 |
| training-04 | 21 | 7/21 (33.3%) | 10/21 (47.6%) | +14.3 |
| training-05 | 14 | 5/14 (35.7%) | 5/14 (35.7%) | +0.0 |
| training-06 | 23 | 15/23 (65.2%) | 15/23 (65.2%) | +0.0 |
| training-07 | 14 | 6/14 (42.9%) | 7/14 (50.0%) | +7.1 |
| training-08 | 9 | 2/9 (22.2%) | 3/9 (33.3%) | +11.1 |
| training-09 | 13 | 10/13 (76.9%) | 10/13 (76.9%) | +0.0 |
| training-10 | 21 | 17/21 (81.0%) | 17/21 (81.0%) | +0.0 |
| training-11 | 21 | 3/21 (14.3%) | 3/21 (14.3%) | +0.0 |
| training-12 | 22 | 21/22 (95.5%) | 21/22 (95.5%) | +0.0 |
| training-13 | 16 | 16/16 (100.0%) | 16/16 (100.0%) | +0.0 |
| training-14 | 15 | 4/15 (26.7%) | 4/15 (26.7%) | +0.0 |
| training-15 | 14 | 14/14 (100.0%) | 14/14 (100.0%) | +0.0 |
| training-16 | 11 | 5/11 (45.5%) | 5/11 (45.5%) | +0.0 |
| training-17 | 18 | 18/18 (100.0%) | 13/18 (72.2%) | -27.8 |
| training-18 | 20 | 11/20 (55.0%) | 11/20 (55.0%) | +0.0 |
| training-19 | 22 | 22/22 (100.0%) | 22/22 (100.0%) | +0.0 |
| training-20 | 21 | 10/21 (47.6%) | 10/21 (47.6%) | +0.0 |

## LightGBM params (ranker, margin=1.0)

```
objective: lambdarank
metric: ndcg
n_estimators: 50
max_depth: 4
num_leaves: 15
learning_rate: 0.1
min_data_in_leaf: 5
label_gain: [0, 1]
verbose: -1
deterministic: True
random_state: 42
```

## Features

- midi_note
- amplitude
- basicpitch_confidence
- is_chord
- chord_size
- chord_string_span
- num_candidates
- prev_position_string
- prev_position_fret
- seconds_since_prev
- hand_anchor_fret
- cand_string
- cand_fret
- dist_anchor_fret
- dist_prev_fret
- dist_prev_string
- heuristic_score
- is_heuristic_pick
