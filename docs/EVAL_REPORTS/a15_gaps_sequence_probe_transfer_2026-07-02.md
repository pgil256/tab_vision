# A15 step 1 — fingering-sequence prior probe (oracle audio)

- manifest: `data\eval\gaps.toml` (22 clips, splits: test)
- prior training data: GuitarSet players != 05 (300 tracks)
- decode: gold pitches -> synthetic AudioEvents -> `fuse()` (audio errors excluded; scores are position-resolution only)
- sequence prior gated to singleton→singleton cluster moves in the decode (chord transitions stay hand-coded — A5 territory)
- smoothing: alpha=0.5, backoff_kappa=8.0

## Transition statistics (train split)

- train transitions: **28,576** (cluster-anchor to anchor)
- marginal H(Δstring) = **2.235 bits**; conditional H(Δstring | Δpitch) = **0.545 bits** (information gain **1.690 bits**)
- same-string transitions overall: **52.3%**

| Δpitch | n | P(same string) | argmax Δstring |
|---|---|---|---|
| +0 | 8,031 | 0.99 | +0 |
| -2 | 2,453 | 0.71 | +0 |
| +2 | 2,187 | 0.73 | +0 |
| +3 | 1,464 | 0.28 | +1 |
| -3 | 1,340 | 0.29 | -1 |
| +1 | 1,311 | 0.94 | +0 |
| -1 | 1,220 | 0.94 | +0 |
| -5 | 933 | 0.08 | -1 |
| +7 | 852 | 0.01 | +1 |
| +5 | 833 | 0.06 | +1 |

## Oracle-audio Tab F1 by config

| config | single_line | overall | chord acc (strummed) |
|---|---|---|---|
| handcoded (no unigram, no seq) | 0.7782 | 0.7782 | 0.000 |
| unigram guitarset-v1 (baseline) | 0.6213 | 0.6213 | 0.000 |
| unigram + seq delta stats-all w=1.0 | 0.6347 | 0.6347 | 0.000 |
| unigram + seq delta stats-all w=2.0 | 0.6368 | 0.6368 | 0.000 |
| unigram + seq delta stats-all w=4.0 | 0.6329 | 0.6329 | 0.000 |
| handcoded + seq delta stats-all w=1.0 (no unigram) | 0.7643 | 0.7643 | 0.000 |
| handcoded + seq delta stats-all w=2.0 (no unigram) | 0.7540 | 0.7540 | 0.000 |
| handcoded + seq delta stats-all w=4.0 (no unigram) | 0.7335 | 0.7335 | 0.000 |
| unigram + seq delta stats-singles w=1.0 | 0.6257 | 0.6257 | 0.000 |
| unigram + seq delta stats-singles w=2.0 | 0.6326 | 0.6326 | 0.000 |
| unigram + seq delta stats-singles w=4.0 | 0.6313 | 0.6313 | 0.000 |
| handcoded + seq delta stats-singles w=1.0 (no unigram) | 0.7621 | 0.7621 | 0.000 |
| handcoded + seq delta stats-singles w=2.0 (no unigram) | 0.7505 | 0.7505 | 0.000 |
| handcoded + seq delta stats-singles w=4.0 (no unigram) | 0.7309 | 0.7309 | 0.000 |
| unigram + seq delta_fret stats-all w=1.0 | 0.6330 | 0.6330 | 0.000 |
| unigram + seq delta_fret stats-all w=2.0 | 0.6422 | 0.6422 | 0.000 |
| unigram + seq delta_fret stats-all w=4.0 | 0.6375 | 0.6375 | 0.000 |
| handcoded + seq delta_fret stats-all w=1.0 (no unigram) | 0.7653 | 0.7653 | 0.000 |
| handcoded + seq delta_fret stats-all w=2.0 (no unigram) | 0.7575 | 0.7575 | 0.000 |
| handcoded + seq delta_fret stats-all w=4.0 (no unigram) | 0.7370 | 0.7370 | 0.000 |
| unigram + seq delta_fret stats-singles w=1.0 | 0.6244 | 0.6244 | 0.000 |
| unigram + seq delta_fret stats-singles w=2.0 | 0.6287 | 0.6287 | 0.000 |
| unigram + seq delta_fret stats-singles w=4.0 | 0.6262 | 0.6262 | 0.000 |
| handcoded + seq delta_fret stats-singles w=1.0 (no unigram) | 0.7637 | 0.7637 | 0.000 |
| handcoded + seq delta_fret stats-singles w=2.0 (no unigram) | 0.7474 | 0.7474 | 0.000 |
| handcoded + seq delta_fret stats-singles w=4.0 (no unigram) | 0.7299 | 0.7299 | 0.000 |
