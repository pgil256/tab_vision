# A15 step 1 — fingering-sequence prior probe (oracle audio)

- manifest: `data\eval\gaps.toml` (22 clips, splits: test)
- prior training data: *(learned sweep skipped — artifact comparison only)*
- decode: gold pitches -> synthetic AudioEvents -> `fuse()` (audio errors excluded; scores are position-resolution only)
- sequence prior gated to singleton→singleton cluster moves in the decode (chord transitions stay hand-coded — A5 territory)
- smoothing: alpha=0.5, backoff_kappa=8.0

## Transition statistics (train split)

*(no transitions extracted)*

## Oracle-audio Tab F1 by config

| config | single_line | overall | chord acc (strummed) |
|---|---|---|---|
| handcoded (no unigram, no seq) | 0.7782 | 0.7782 | 0.000 |
| unigram guitarset-v1 (baseline) | 0.6213 | 0.6213 | 0.000 |
| unigram + artifact guitarset-seq-v1 w=2.0 | 0.6287 | 0.6287 | 0.000 |
| unigram + artifact guitarset-seq-v1 w=4.0 | 0.6262 | 0.6262 | 0.000 |
| unigram + artifact guitarset-seq-v1 w=8.0 | 0.6197 | 0.6197 | 0.000 |
| handcoded + artifact guitarset-seq-v1 w=2.0 (no unigram) | 0.7474 | 0.7474 | 0.000 |
| handcoded + artifact guitarset-seq-v1 w=4.0 (no unigram) | 0.7299 | 0.7299 | 0.000 |
| handcoded + artifact guitarset-seq-v1 w=8.0 (no unigram) | 0.7042 | 0.7042 | 0.000 |
| unigram + artifact pdmx-seq-v1 w=2.0 | 0.6612 | 0.6612 | 0.000 |
| unigram + artifact pdmx-seq-v1 w=4.0 | 0.6661 | 0.6661 | 0.000 |
| unigram + artifact pdmx-seq-v1 w=8.0 | 0.6704 | 0.6704 | 0.000 |
| handcoded + artifact pdmx-seq-v1 w=2.0 (no unigram) | 0.7755 | 0.7755 | 0.000 |
| handcoded + artifact pdmx-seq-v1 w=4.0 (no unigram) | 0.7688 | 0.7688 | 0.000 |
| handcoded + artifact pdmx-seq-v1 w=8.0 (no unigram) | 0.7620 | 0.7620 | 0.000 |
| unigram + artifact guitarset-pdmx-seq-v1 w=2.0 | 0.6550 | 0.6550 | 0.000 |
| unigram + artifact guitarset-pdmx-seq-v1 w=4.0 | 0.6558 | 0.6558 | 0.000 |
| unigram + artifact guitarset-pdmx-seq-v1 w=8.0 | 0.6604 | 0.6604 | 0.000 |
| handcoded + artifact guitarset-pdmx-seq-v1 w=2.0 (no unigram) | 0.7713 | 0.7713 | 0.000 |
| handcoded + artifact guitarset-pdmx-seq-v1 w=4.0 (no unigram) | 0.7662 | 0.7662 | 0.000 |
| handcoded + artifact guitarset-pdmx-seq-v1 w=8.0 (no unigram) | 0.7452 | 0.7452 | 0.000 |
