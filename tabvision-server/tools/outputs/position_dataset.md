# Position dataset — 2026-04-29

Step 2 output for the learned-fusion plan
(`docs/plans/2026-04-24-learned-fusion-design.md` §4.3).

Built from `position-features-2026-04-29_093154.jsonl` (502 events) by
`tools/build_position_dataset.py`.

## Summary

| metric | value |
|---|---:|
| Events labeled | 336 |
| Events dropped (no GT match — extras) | 166 |
| Rows (one per candidate) | 1319 |
| Positive labels | 336 (25.5% of rows; exactly 1 per event) |
| Events whose heuristic pick is the correct GT position | 202 (60.1%) |

Heuristic 60.1% is the ranker's baseline. The plan's ship gate is +5pp on
exact-F1 across the 20-video set (effectively ≥65% on this metric for the
labeled subset).

## Composition

| split | rows | positives | base rate |
|---|---:|---:|---:|
| single-note (`is_chord=False`) | 750 | 182 | 24.3% |
| chord (`is_chord=True`) | 569 | 154 | 27.1% |

Per-video event counts span 9 (training-08) to 23 (training-06); all 20
videos contribute, so leave-one-video-out CV is straightforward.

## Schema

One row per candidate. Group key for CV: `video_id`.

| column | source |
|---|---|
| `video_id`, `event_id` | identifiers |
| `onset_time`, `midi_note`, `amplitude`, `basicpitch_confidence` | per-event audio |
| `is_chord`, `chord_size`, `chord_string_span`, `num_candidates` | per-event chord context |
| `prev_position_string`, `prev_position_fret`, `seconds_since_prev` | per-event continuity |
| `hand_anchor_fret`, `video_hand_anchor_fret` | per-event anchor |
| `selected_string`, `selected_fret` | what fusion actually emitted |
| `cand_string`, `cand_fret` | candidate identity |
| `dist_anchor_fret`, `dist_anchor_string` | candidate vs anchor |
| `dist_prev_fret`, `dist_prev_string` | candidate vs prev pos |
| `heuristic_score`, `is_heuristic_pick` | what the heuristic thought |
| `gt_string`, `gt_fret` | ground-truth label position |
| `label` | 0/1 — this candidate is the GT position |

## Known gaps

- All 20 videos run with `audio_only=True` per `tests/fixtures/benchmarks/index.json`,
  so `video_hand_anchor_fret` is null on every event. To fill that column,
  re-dump features after enabling video. Step 3 can train without it; the
  feature is forward-compatible.
- Extras (166 events) are dropped — they have no correct candidate. Step 3
  could optionally use them as "all-zero label" group to teach the model to
  reject hallucinations, but the plan keeps that out of scope for v1.
- Muted-X GT notes (training-20) are excluded by the audio matcher (no MIDI
  → no audio detection → no fusion event). Consistent with the corrected
  apr-28 error analysis.
