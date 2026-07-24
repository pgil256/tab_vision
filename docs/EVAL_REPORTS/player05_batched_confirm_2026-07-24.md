# Player-05 sealed confirmation — level correction REFUTED, N1 CONFIRMED

**Date:** 2026-07-24
**Script:** `tabvision/scripts/eval/player05_batched_confirm.py`
**Data:** `docs/EVAL_REPORTS/player05_batched_confirm_2026-07-24.json`
**Set:** GuitarSet player 05, 60 clips (sealed held-out; opened with user authorization)
**Design:** 2×2 over (table, isolation) plus baseline, bootstrap N=10,000, seed 42

## Verdict

| question | reading |
|---|---|
| physics channel on held-out data | **PASS** — reproduces Q6 exactly |
| +0.60 level correction | **REFUTED** — reverted |
| N1 partial-aware isolation | **PASS** — shipped |

## Results

| arm | Tab F1 |
|---|---:|
| baseline | 0.6340 |
| raw-strict | 0.7119 |
| corrected-strict | 0.7053 |
| raw-partial | 0.7346 |
| **corrected-partial** | 0.7357 |

| comparison | vs | delta | lo-95 | hi-95 | reading |
|---|---|---:|---:|---:|---|
| gate re-seal | baseline | +0.0780 | +0.0502 | +0.1078 | PASS |
| level correction | raw-strict | −0.0066 | −0.0224 | +0.0079 | **REFUTED** |
| level correction \| partial | raw-partial | +0.0012 | −0.0217 | +0.0213 | FAIL |
| N1 \| raw table | raw-strict | +0.0226 | +0.0022 | +0.0446 | PASS |
| N1 \| corrected table | corrected-strict | +0.0304 | +0.0093 | +0.0531 | PASS |
| **best (raw-partial)** | baseline | **+0.1006** | +0.0615 | +0.1416 | PASS |

## The harness reproduces Q6 bit-for-bit

`raw-strict` vs baseline measured +0.0780 [+0.0502, +0.1078], solo +0.1396,
comp +0.0164 — identical to every figure in the Q6 manifest's `held_out` block
(+0.0780 [+0.0502, +0.1078], solo 0.1396, comp 0.0164). The confirmation
harness, the cache, and the scoring path are all unchanged from the original
gate, so the two new legs sit on a substrate proven to reproduce.

## Why the level correction failed

The correction gained **+0.0160 [+0.0088, +0.0233]** on GuitarSet dev (300
clips, players 00-04) and measured **−0.0066 [−0.0224, +0.0079]** on player 05.
The intervals do not overlap: dev's lower bound is +0.0088, player-05's upper
bound is +0.0079.

The level error itself is not in doubt. It was measured three independent ways
— Q6's leave-one-player-out residual (−0.566), N5's perturbation sweep, and
N4's direct hex-pickup measurement (+0.780 median). The table really does
under-predict B.

What failed is the inference from "the error exists" to "a constant fixes it".
**N4's own data predicted this and it was under-weighted at the time**: the
per-player offsets ran +0.514, +0.748, +0.780, +0.794, +1.092 — a spread of
0.578 log-B, *wider than the correction itself*. A population that varied that
much cannot be served by one constant, and the dev median is not the
population median. This is the same finding that killed N4's per-instrument
ritual, arriving from the other direction: the level error is physically real
and instrument-specific, so neither a fixed constant nor a per-instrument
estimate beats leaving the derivation alone.

The dev gain was real but local to players 00-04. Nothing about the physics
was wrong; the generalization step was.

## N1 partial-aware isolation

Confirmed on both tables, so the result does not depend on the correction's
disposition:

- on the raw (shipped) table: **+0.0226 [+0.0022, +0.0446]**
- on the corrected table: +0.0304 [+0.0093, +0.0531]

Coverage lifts from 834 to **2227** applied notes of 8709 — 2.7×. Strict
isolation demands a note sound alone; partial-aware drops only the partials a
simultaneous note actually collides with and fits the rest.

`isolation` now travels inside the artifact alongside `weight`, `min_r2` and
`sigma`, so a change to the module default cannot alter what the registered
artifact does.

## Net effect on the shipped configuration

Player-05 Tab F1 against the no-physics baseline moves from Q6's **+0.0780** to
**+0.1006 [+0.0615, +0.1416]** — entirely from N1, none from the correction.

## Honest limits

- 60 clips, one held-out player. The correction's refutation rests on a
  point estimate that is negative but whose interval includes zero; the
  defensible claim is "no evidence of benefit out of distribution, against a
  significant in-distribution gain", not "significantly harmful".
- The 2×2's four cells were run in one pass; the first pass omitted
  `raw-partial` and was re-run complete. Both passes agree on every shared
  cell.
- Player-05 has now been opened twice for this artifact (Q6, and this run).
  Its value as a sealed set is correspondingly reduced.
- Method note earned here: a constant located on dev and validated only on dev
  is not validated. The three independent *physical* measurements of the level
  error created false confidence in a *decision-theoretic* correction — they
  are different claims, and only the first was ever tested.
