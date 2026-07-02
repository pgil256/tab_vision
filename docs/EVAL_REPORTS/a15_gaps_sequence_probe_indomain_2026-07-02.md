# A15 step 1 — fingering-sequence prior probe (oracle audio)

- manifest: `data\eval\gaps.toml` (22 clips, splits: test)
- prior training data: `gaps.toml` train split (212 clips parsed, 2 skipped malformed)
- decode: gold pitches -> synthetic AudioEvents -> `fuse()` (audio errors excluded; scores are position-resolution only)
- sequence prior gated to singleton→singleton cluster moves in the decode (chord transitions stay hand-coded — A5 territory)
- smoothing: alpha=0.5, backoff_kappa=8.0

## Transition statistics (train split)

- train transitions: **97,956** (cluster-anchor to anchor)
- marginal H(Δstring) = **2.900 bits**; conditional H(Δstring | Δpitch) = **1.188 bits** (information gain **1.712 bits**)
- same-string transitions overall: **28.0%**

| Δpitch | n | P(same string) | argmax Δstring |
|---|---|---|---|
| +0 | 8,340 | 0.91 | +0 |
| -2 | 7,006 | 0.58 | +0 |
| +3 | 5,355 | 0.25 | +1 |
| +2 | 5,345 | 0.57 | +0 |
| -1 | 4,679 | 0.79 | +0 |
| +1 | 4,646 | 0.79 | +0 |
| -3 | 4,462 | 0.27 | -1 |
| +5 | 4,218 | 0.12 | +1 |
| +4 | 3,908 | 0.14 | +1 |
| -5 | 3,455 | 0.11 | -1 |

## Oracle-audio Tab F1 by config

| config | single_line | overall | chord acc (strummed) |
|---|---|---|---|
| handcoded (no unigram, no seq) | 0.7782 | 0.7782 | 0.000 |
| unigram guitarset-v1 (baseline) | 0.6213 | 0.6213 | 0.000 |
| unigram + seq delta stats-all w=1.0 | 0.6376 | 0.6376 | 0.000 |
| unigram + seq delta stats-all w=2.0 | 0.6372 | 0.6372 | 0.000 |
| unigram + seq delta stats-all w=4.0 | 0.6403 | 0.6403 | 0.000 |
| handcoded + seq delta stats-all w=1.0 (no unigram) | 0.7752 | 0.7752 | 0.000 |
| handcoded + seq delta stats-all w=2.0 (no unigram) | 0.7633 | 0.7633 | 0.000 |
| handcoded + seq delta stats-all w=4.0 (no unigram) | 0.7492 | 0.7492 | 0.000 |
| unigram + seq delta stats-singles w=1.0 | 0.6359 | 0.6359 | 0.000 |
| unigram + seq delta stats-singles w=2.0 | 0.6367 | 0.6367 | 0.000 |
| unigram + seq delta stats-singles w=4.0 | 0.6399 | 0.6399 | 0.000 |
| handcoded + seq delta stats-singles w=1.0 (no unigram) | 0.7746 | 0.7746 | 0.000 |
| handcoded + seq delta stats-singles w=2.0 (no unigram) | 0.7664 | 0.7664 | 0.000 |
| handcoded + seq delta stats-singles w=4.0 (no unigram) | 0.7507 | 0.7507 | 0.000 |
| unigram + seq delta_fret stats-all w=1.0 | 0.6524 | 0.6524 | 0.000 |
| unigram + seq delta_fret stats-all w=2.0 | 0.6648 | 0.6648 | 0.000 |
| unigram + seq delta_fret stats-all w=4.0 | 0.6721 | 0.6721 | 0.000 |
| handcoded + seq delta_fret stats-all w=1.0 (no unigram) | 0.7768 | 0.7768 | 0.000 |
| handcoded + seq delta_fret stats-all w=2.0 (no unigram) | 0.7746 | 0.7746 | 0.000 |
| handcoded + seq delta_fret stats-all w=4.0 (no unigram) | 0.7709 | 0.7709 | 0.000 |
| unigram + seq delta_fret stats-singles w=1.0 | 0.6550 | 0.6550 | 0.000 |
| unigram + seq delta_fret stats-singles w=2.0 | 0.6647 | 0.6647 | 0.000 |
| unigram + seq delta_fret stats-singles w=4.0 | 0.6715 | 0.6715 | 0.000 |
| handcoded + seq delta_fret stats-singles w=1.0 (no unigram) | 0.7778 | 0.7778 | 0.000 |
| handcoded + seq delta_fret stats-singles w=2.0 (no unigram) | 0.7774 | 0.7774 | 0.000 |
| handcoded + seq delta_fret stats-singles w=4.0 (no unigram) | 0.7740 | 0.7740 | 0.000 |
