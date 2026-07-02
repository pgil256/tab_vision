# A15 step 1 — fingering-sequence prior probe (oracle audio)

- manifest: `data\eval\local_gs_val24.toml` (24 clips, splits: validation)
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

| config | single_line | strummed | overall | chord acc (strummed) |
|---|---|---|---|---|
| handcoded (no unigram, no seq) | 0.2019 | 0.5219 | 0.4386 | 0.472 |
| unigram guitarset-v1 (baseline) | 0.5542 | 0.8564 | 0.7777 | 0.792 |
| unigram + seq delta stats-all w=1.0 | 0.5921 | 0.8583 | 0.7890 | 0.794 |
| unigram + seq delta stats-all w=2.0 | 0.5854 | 0.8578 | 0.7869 | 0.793 |
| unigram + seq delta stats-all w=4.0 | 0.5881 | 0.8569 | 0.7869 | 0.791 |
| unigram + seq delta stats-singles w=1.0 | 0.6003 | 0.8578 | 0.7908 | 0.793 |
| unigram + seq delta stats-singles w=2.0 | 0.5894 | 0.8578 | 0.7879 | 0.793 |
| unigram + seq delta stats-singles w=4.0 | 0.5935 | 0.8573 | 0.7886 | 0.792 |
| unigram + seq delta_fret stats-all w=1.0 | 0.5854 | 0.8578 | 0.7869 | 0.795 |
| unigram + seq delta_fret stats-all w=2.0 | 0.6030 | 0.8573 | 0.7911 | 0.794 |
| unigram + seq delta_fret stats-all w=4.0 | 0.5962 | 0.8573 | 0.7893 | 0.794 |
| unigram + seq delta_fret stats-singles w=1.0 | 0.5827 | 0.8578 | 0.7862 | 0.795 |
| unigram + seq delta_fret stats-singles w=2.0 | 0.5989 | 0.8578 | 0.7904 | 0.795 |
| unigram + seq delta_fret stats-singles w=4.0 | 0.6125 | 0.8573 | 0.7936 | 0.794 |

## Ungated application (superseded — banked negative)

A first pass applied the learned term to **every** cluster transition
(including chord-to-chord moves). Single-line improved but strummed and
chord accuracy regressed — the sequence statistics describe note-to-note
movement, not chord voicing changes:

| config | single_line | strummed | overall | chord acc (strummed) |
|---|---|---|---|---|
| unigram guitarset-v1 (baseline) | 0.5542 | 0.8564 | 0.7777 | 0.792 |
| ungated unigram + seq delta w=0.5 | 0.5732 | 0.8483 | 0.7766 | 0.780 |
| ungated unigram + seq delta w=1.0 | 0.5921 | 0.8402 | 0.7756 | 0.763 |
| ungated unigram + seq delta w=4.0 | 0.5921 | 0.8187 | 0.7597 | 0.733 |
| ungated unigram + seq delta_fret w=2.0 | 0.6030 | 0.8340 | 0.7738 | 0.748 |

This motivated the singleton→singleton gate now hard-wired in
`viterbi._viterbi_clusters` (chord transitions = A5 chord-dictionary
territory, per the A15/A5 complement design).

## Conclusions (step 1 of the A15 staging)

1. **The sequence signal is real in-domain.** Δpitch carries 1.69 bits about
   Δstring (2.235 → 0.545 conditional). With the singleton gate, every swept
   config beats the accepted baseline on oracle audio: single-line
   **0.5542 → 0.6125** (+5.8pp, delta_fret stats-singles w=4.0), strummed and
   chord accuracy preserved (0.8564 → 0.857–0.858 / 0.792 → 0.79x).
2. **It does not transfer to GAPS as a standalone term** — same lesson as the
   A2 unigram negative. Against GAPS's accepted config (prior=none) the
   GuitarSet-trained prior is a mild negative (0.7782 → 0.7653 best); even a
   GAPS-trained one is a wash (0.7778) — on classical single-line the
   hand-coded transition terms are already near-optimal
   (`a15_gaps_sequence_probe_{transfer,indomain}_2026-07-02.md`).
3. **Under the product default config (unigram on), it helps BOTH corpora**:
   val24 overall 0.7777 → 0.7936 and GAPS-22 0.6213 → 0.6422 (GuitarSet-trained)
   / 0.6721 (GAPS-trained) — the sequence term partially repairs the unigram's
   known cross-domain damage.
4. **Deployment shape:** the sequence prior rides the `guitarset-v1` config
   family — active only when the pitch-position prior is active. GAPS's
   accepted `--position-prior none` config is untouched, so its no-regression
   gate passes by construction; the val24 real-audio gate is measured next
   (step 4 of the staging: real-audio runs, artifact `guitarset-seq-v1`).

All numbers above are **oracle-audio** (gold pitches through the real decode);
real-audio deltas will be smaller — see the step-4 gated runs.
