# Tab F1 error decomposition

## Diagnostic summary

**Dominant failure bucket on every covered tier is
`wrong_position_same_pitch`** — the audio detected the right pitch
within onset tolerance but the system placed it on the wrong
(string, fret).

| Tier | Loss share — wrong_position_same_pitch |
|---|---:|
| clean_acoustic_single_line | **77.5%** (910 / 1174 loss events) |
| clean_acoustic_strummed | **49.7%** (1548 / 3112 loss events) |
| Aggregate | **57.3%** (2458 / 4286 loss events) |

This matches the strategy doc §2 diagnostic exactly. The audio side
is at SPEC (Pitch F1 ≥ 0.90 on both covered tiers); the gap to D2
per-tier targets is almost entirely string/fret assignment, and it
gets worse on single-line passages where chord-cluster constraints
can't help the fusion.

Companion baseline report: [`composite_baseline_2026-05-13.md`](composite_baseline_2026-05-13.md).

Six-bucket port of the apr-28 7-bucket harness; the seventh apr-28
bucket (`muted_undetectable`) is deferred until the §8 `TabEvent`
contract carries a muted/X flag.

## Aggregate (all tiers)

| Bucket | Count | Share of loss |
|---|---:|---:|
| correct | 4986 | — |
| wrong_position_same_pitch | 2458 | 57.3% |
| pitch_off | 505 | 11.8% |
| timing_only | 94 | 2.2% |
| missed_onset | 672 | 15.7% |
| extra_detection | 557 | 13.0% |

## Per-tier breakdown

| Tier | correct | wrong_position_same_pitch | pitch_off | timing_only | missed_onset | extra_detection |
|---|---|---|---|---|---|---|
| clean_acoustic_single_line | 1125 | 910 | 19 | 17 | 108 | 120 |
| clean_acoustic_strummed | 3861 | 1548 | 486 | 77 | 564 | 437 |

