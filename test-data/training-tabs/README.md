# Training Tabs for TabVision

These tabs are designed to be recorded on acoustic and/or electric guitar
for training and fine-tuning the TabVision transcription pipeline. See the
**Instrument** section below — a few tabs reach past the 12th fret and are
marked electric-only.

## Recording Instructions

1. Use a webcam at desk, guitar in lap
2. Standard tuning (EADGBE), no capo unless noted
3. Play to a metronome at the BPM indicated
4. Clean tone on electric (no distortion/effects)
5. Keep the guitar neck visible and roughly horizontal
6. Play each note clearly with good separation
7. Record each tab as a separate video file
8. Name files: `training-XX-acoustic.mp4` / `training-XX-electric.mp4`

## Categories

| # | Category | Tabs | Focus |
|---|----------|------|-------|
| 01-05 | Position Ambiguity | 5 | Same notes at different positions |
| 06-10 | Chord Varieties | 5 | Open, barre, power, jazz, partial |
| 11-15 | Single-Note Passages | 5 | Scales, arpeggios, chromatic runs |
| 16-20 | Edge Cases | 5 | Jumps, high frets, mixed techniques |

## Instrument

Each `-info.txt` has an `Instrument:` line indicating which guitars the tab
is playable on. Any tab reaching past fret 12 is marked **electric** (acoustic
necks typically join the body at fret 12, making higher positions impractical
without a cutaway). All other tabs are **acoustic or electric**.

Electric-only tabs: **03, 04, 16, 17** (max frets 14, 14, 15, 17).

## Tab File Format

Each `-tabs.txt` file is the ground truth in standard ASCII tab format,
compatible with `evaluate_transcription.py`. The companion `-info.txt` has
BPM, description, instrument, and what the tab tests.
