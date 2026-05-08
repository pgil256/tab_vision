# Vanilla Basic Pitch baseline — 2026-04-30

Reference number for the audio fine-tune (plan §7 Week 2).
Split: **validation**.  Tracks: **60**.

## Aggregate

| Metric | Mean | Median |
|---|---:|---:|
| Frame note F1 @ 0.3 | 0.6777 | 0.6685 |
| Frame note F1 @ 0.5 | 0.6746 | 0.6727 |
| Frame note F1 @ 0.7 | 0.6172 | 0.6131 |
| Onset P @ 0.5 (±1 frame) | 0.0000 | 0.0000 |
| Onset R @ 0.5 (±1 frame) | 0.0000 | 0.0000 |
| Onset F1 @ 0.5 (±1 frame) | 0.0000 | 0.0000 |
| Note-event P (best) | 0.8828 | 0.9041 |
| Note-event R (best) | 0.8380 | 0.8543 |
| Note-event F1 (best) | 0.8580 | 0.8780 |

Best note-event threshold (track-wise mean): onset=0.22, frame=0.38.

Total reference notes across split: **8715**.
Total estimated notes at best thresholds: **8078**.

## Per-track table
See `finetune_finetune-20260430-fullsweep-2026-04-30.csv`.

## Notes

- **Note-event F1 is the headline metric.** It maps directly to how the fine-tune output will be consumed (`note_creation.model_output_to_notes` → notes → fusion engine).

- Frame F1: per-cell binary on (T, 88) note head vs densified target. Sanity reference for "is the model picking up the right pitches at the right times" — frame F1 ≤ note-event F1 is expected because frame-level disagreement around onset/offset edges contributes to FP/FN at every frame, while note-event matching is one decision per note.

- **Onset P/R/F1 is unreliable as currently implemented.** ±1-frame (~12 ms) tolerance vs the model's smoothed multi-frame onset ridges systematically under-counts TPs. Numbers reported for completeness; use note-event F1 to compare baselines and fine-tunes.

- Note-event metrics use `mir_eval.transcription.precision_recall_f1_overlap` with `offset_ratio=None` (onset+pitch only). Best F1 over a 0.05-stride sweep of (onset_thresh, frame_thresh).

- **Scope reminder.** This is *in-distribution* GuitarSet held-out (split by player). The plan §0 ship gate is on our **20-video iPhone set** (out-of-distribution), where the current exact F1 is ~0.51 and the target is ≥ 0.60. Use the present number (note-event F1 ≈ 0.87) only as the within-GuitarSet reference — improvement here is a *necessary but not sufficient* condition for OOD improvement.
