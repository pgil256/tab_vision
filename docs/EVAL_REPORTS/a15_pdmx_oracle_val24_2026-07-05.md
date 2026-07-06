# A15 step 1 — fingering-sequence prior probe (oracle audio)

- manifest: `data\eval\local_gs_val24.toml` (24 clips, splits: validation, test)
- prior training data: *(learned sweep skipped — artifact comparison only)*
- decode: gold pitches -> synthetic AudioEvents -> `fuse()` (audio errors excluded; scores are position-resolution only)
- sequence prior gated to singleton→singleton cluster moves in the decode (chord transitions stay hand-coded — A5 territory)
- smoothing: alpha=0.5, backoff_kappa=8.0

## Transition statistics (train split)

*(no transitions extracted)*

## Oracle-audio Tab F1 by config

| config | single_line | strummed | overall | chord acc (strummed) |
|---|---|---|---|---|
| handcoded (no unigram, no seq) | 0.2019 | 0.5219 | 0.4386 | 0.472 |
| unigram guitarset-v1 (baseline) | 0.5542 | 0.8564 | 0.7777 | 0.792 |
| unigram + artifact guitarset-seq-v1 w=2.0 | 0.5989 | 0.8578 | 0.7904 | 0.795 |
| unigram + artifact guitarset-seq-v1 w=4.0 | 0.6125 | 0.8573 | 0.7936 | 0.794 |
| unigram + artifact guitarset-seq-v1 w=8.0 | 0.5935 | 0.8564 | 0.7879 | 0.792 |
| unigram + artifact pdmx-seq-v1 w=2.0 | 0.5732 | 0.8564 | 0.7826 | 0.794 |
| unigram + artifact pdmx-seq-v1 w=4.0 | 0.5366 | 0.8593 | 0.7752 | 0.801 |
| unigram + artifact pdmx-seq-v1 w=8.0 | 0.5271 | 0.8564 | 0.7706 | 0.799 |
| unigram + artifact guitarset-pdmx-seq-v1 w=2.0 | 0.5894 | 0.8564 | 0.7869 | 0.794 |
| unigram + artifact guitarset-pdmx-seq-v1 w=4.0 | 0.5881 | 0.8554 | 0.7858 | 0.792 |
| unigram + artifact guitarset-pdmx-seq-v1 w=8.0 | 0.5949 | 0.8535 | 0.7862 | 0.792 |
