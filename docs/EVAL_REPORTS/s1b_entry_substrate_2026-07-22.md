# Q2 / S1b-v2 — entry substrate verification + symbolic corpus

Accuracy-loop iteration 2 (ROI deep-dive §3.2, §7 row 2). No model trained
yet: this establishes that the probe's two inputs — the banked Phase 0
ambiguous lattice and a sequence-preserving SynthTab corpus — are real,
faithful, and sized for the job, and it corrects the gate's evaluation slice
before any training is spent against the wrong target.

## 1. The banked lattice is a faithful offline replay substrate

`string_assignment_phase0_2026-07-15_notes.csv` (247,520 rows, 5 conditions)
reproduces both Phase 0 headline numbers **exactly** from disk, with no audio,
no backend, and no pipeline:

| condition / split | ambiguous notes | top-1 | top-3 |
|---|---:|---:|---:|
| `production_equivalent` / `held_out_05` | 7,121 | **0.6770** | **0.9986** |
| `production_equivalent` / `development_oof` | 35,959 | **0.6548** | 0.9967 |

The held-out row matches `string_assignment_phase0_2026-07-15.md` to four
decimals. The rescoring probe is therefore pure CSV replay — seconds per
sweep, `$0`, and repeatable.

Per-row the CSV carries `track_id`, `event_index`, `cluster_index`,
`cluster_event_index`, `onset_s`, `pitch_midi`, gold `reference_string` /
`reference_fret`, `reference_rank`, `ambiguous_pitch_match`, and
`candidate_path` — the ranked lattice as `string:fret:cost_delta_from_best`
triples. That is everything a windowed contextual model needs (sequence
order, chord grouping, the exact candidate set) **plus the decoder's own
cost margin**, so the probe can score the real integration shape — blend
model log-probability with the existing cost and sweep the mixing weight
offline — rather than an approximation of it.

## 2. Gate correction: 0.6770 is the sealed player-05 slice

The deep-dive states the entry gate as "ambiguous top-1 ≥ +0.05 over
**0.6770**". That number is `held_out_05` — **player-05**, which the loop
opens only after config freeze and an explicit user proceed. Tuning a model
against it would burn the confirmation set the whole program protects.

**Working gate for Q2 development is therefore the dev-OOF slice**, carrying
the same +0.05 bar:

> ambiguous top-1 ≥ **0.7048** (from 0.6548) on `production_equivalent` /
> `development_oof`, n = 35,959.

This is strictly the better development target anyway: 5.0× the notes of the
player-05 slice, so 5× the power to separate a real effect from noise.
Player-05 stays sealed for the confirmation run.

## 3. What the gate actually asks for

Distribution of the gold candidate's rank under the current decoder, dev-OOF
ambiguous slice:

| gold rank | 1 | 2 | 3 | 4 | 5 | absent |
|---|---:|---:|---:|---:|---:|---:|
| notes | 23,547 | **10,428** | 1,867 | 15 | 2 | 100 |

- Gold is in the lattice for **99.72%** of ambiguous notes — the ceiling for
  a constrained rescorer, and it confirms the §3.2 design choice to
  redistribute mass over `candidate_positions()` rather than free-run.
- **84% of all misses are gold-at-rank-2** (10,428 of 12,412). The decoder is
  not lost; it is second-guessing a binary.
- The +0.05 gate needs **1,798 notes** flipped — i.e. **17% of the rank-2
  pile**. That is the concrete bar, and it is a far more tractable statement
  than "beat 0.6548".
- Split by tier: solo **0.5908** (n=12,663) vs comp **0.6896** (n=23,296).
  The assignment headroom is concentrated in single-line material, exactly
  where SPEC §1.4.1's weakest tier lives and where audio evidence is
  information-limited — so context is the only lever left there.

## 4. Symbolic corpus: sequences, not counts

`scripts/eval/s1b_extract_symbolic.py` extracts per-track note **sequences**
from the SynthTab archive, delegating the parse to S1a's `_track_events` so
the substrate is identical to the audited one (same tempo map, standard-
tuning filter, SynthTab 1=high-E → repo 0=low-E flip, 24-fret bound).

| quantity | value |
|---|---:|
| tracks scanned / eligible / parsed | 60,633 / 45,868 / **34,621** |
| notes | **34,063,065** |
| skipped: non-guitar program / non-standard tuning / unreadable | 14,765 / 11,247 / 0 |
| notes per track, median / p90 | 708 / 2,166 |
| **pitch-ambiguous note share** | **0.8874** |
| mean playable positions per note | 3.45 |
| clusters (80 ms grouping) / polyphonic share / mean size | 16,887,427 / 0.4735 / 2.02 |
| open-string (fret 0) share | 0.1902 |
| extraction wall-clock / output size | 279 s / 46.3 MB (`.npz`) |

**Cross-check:** 34,621 tracks / 34.06M notes reproduces the S1a audit's
all-guitar figures exactly (`s1a_synthtab_priors_2026-07-20.md`), confirming
the extractor sees the same corpus that produced the banked count priors —
the difference is entirely that sequence order and cluster structure are now
preserved rather than marginalized away.

Two properties make this usable as pretraining substrate:

- **88.7% of notes are pitch-ambiguous** — the corpus is dense in precisely
  the decision the decoder gets wrong, so a masked-string objective spends
  almost all its gradient on the target problem rather than on notes with
  one playable position.
- **47% of clusters are polyphonic** (mean size 2.02) — real voicing grammar
  is present, so the 80 ms cluster grouping the decode uses is learnable
  rather than an inference-time-only construct.

## 5. Why this is not S1a again

S1a consumed this same 34.06M-event substrate and closed CI-negative on every
arm. The difference is not scale — it is identical scale — but
representation: S1a reduced each note to a per-pitch position marginal and a
singleton Δfret transition, discarding order. The Phase 0 segment gate
measures the discarded quantity at **+0.1446** ambiguous top-1. Q2 keeps it.
If the contextual model also fails, the failure will be about model or
domain, not about the corpus, because the corpus is byte-for-byte the one
S1a used.

## 6. Next (iteration 3)

1. Tokenize windows (pitch + cluster structure), masked-string objective.
2. Train a small encoder on CPU / free Colab; hold out by track.
3. Rescore the dev-OOF lattice: blend model log-prob with
   `cost_delta_from_best`, sweep the mixing weight, report ambiguous top-1
   against **0.7048**, plus the solo/comp split and the rank-2 flip rate.
4. Fail → banked negative, close Q2. Pass → Q3 (fine-tune + integration
   behind an explicit decoder flag), still stopping before player-05.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/s1b_extract_symbolic.py --variant all \
  --json ../docs/EVAL_REPORTS/s1b_symbolic_corpus_2026-07-22.json
```

Corpus lands at `$TABVISION_DATA_ROOT/models/s1b_symbolic/synthtab_all.npz`
(flat int arrays + per-track offsets; git-ignored). SynthTab is
CC-BY-NC-4.0 — the corpus and anything trained on it inherit NC
(LICENSES.md).
