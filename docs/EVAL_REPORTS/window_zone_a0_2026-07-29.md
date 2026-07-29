# Phase A0 — the physics channel covers the window but cannot aim it

**Gate A0: FAIL.** Coverage passed comfortably; aim did not.

| leg | bar | measured (solo, 1 s, `self_seeded`) | verdict |
|---|---:|---:|---|
| windows with ≥ 1 readable note | ≥ 0.60 | **0.7940** | PASS |
| implied zone agrees with gold | ≥ 0.75 | **0.5816** | **FAIL** |

Per §7's decision tree this banks the negative and closes the program.
A1 is not built. 300 GuitarSet dev tracks (players 00, 01, 02, 03, 05),
52,223 notes, 291 s CPU, $0. Sealed player 04 was not read.

## The bet was half right

The program rested on an arithmetic claim: per-note coverage is sparse
(~34% of solo notes ring alone) but a 1 s window holds several notes, so
*window* coverage should be dense. **That part is confirmed.** Per-note
readability across dev is 12,978/52,223 = **0.2485**, yet **79.4%** of solo
1 s windows contain at least one readable note. Sparse per-note really does
become dense per-window.

It does not help, because the readings are wrong too often to propagate.

## The three arms

Per-note implied-position accuracy on readable notes:

| arm | accuracy | what it needs |
|---|---:|---|
| `reference` — shipped table, raw | 0.3437 | nothing (this is what ships) |
| `self_seeded` — label-free session refit | 0.5830 | a first decode's provisional labels |
| `gold_calibrated` — refit on gold | 0.7797 | labels; ceiling only |

Window-level zone agreement, solo, by window size:

| window | `reference` | `self_seeded` | `gold_calibrated` | coverage | hand-moved |
|---|---:|---:|---:|---:|---:|
| 1 s | 0.2911 | **0.5816** | 0.8167 | 0.7940 | 373 / 4,267 |
| 2 s | 0.1870 | 0.4847 | 0.7268 | 0.9147 | 495 / 2,286 |
| 4 s | 0.0840 | 0.3593 | 0.5902 | 0.9714 | 505 / 1,190 |

Strummed is worse everywhere (1 s: 0.1996 / 0.4267 / 0.5555), as the design
predicted — the channel is a single-line instrument by construction.

**Longer windows trade coverage for aim, and lose.** Going 1 s → 4 s buys
coverage 0.794 → 0.971 but agreement collapses 0.582 → 0.359, while
hand-moved windows rise from 8.7% to 42%. There is no window size at which
both legs pass; the frozen 1 s primary was already the best cell.

## Why it cannot aim: the table is mis-centred, not mis-shaped

Measured `log B` sits a systematic **+0.52** above the reference table's
prediction *at the gold position*, with residual std **0.582** to gold versus
**1.487** to alternatives. The shape is right — gold is three times tighter —
but +0.52 in log B is a **1.68×** ratio, almost exactly the 1.59–1.78× that
separates two candidates 4–5 frets apart
(`docs/EVAL_REPORTS/q6_separability_2026-07-22.md`). A whole-table offset of
one candidate-step moves the argmax off gold nearly every time, which is why
the shipped table's argmax is 0.3437 while a gold refit reaches 0.7797.

This is not a defect in the shipping channel. `apply_fits` multiplies a
Gaussian likelihood into the prior; **soft, mis-centred evidence still helps**
(+0.0522 on sealed player 04) even when its top-1 is poor. Propagation is
different: it would take the argmax zone and push it onto neighbours that
have no evidence of their own, so it needs the top-1 to be right, and it is
not.

Label-free calibration closes about half the gap (0.3437 → 0.5830) and no
more — consistent with `docs/EVAL_REPORTS/q6_self_calibration_2026-07-22.md`,
which found self-calibration from unlabelled audio contributes nothing at one
clip and −0.0029 pooled. Reaching 0.7797 required gold.

## What was reachable, and would have been poisoned

5,265 unreadable solo notes sit inside covered 1 s windows, **5,217 of them
ambiguous** — a large, precisely-targeted population, which is what made the
mechanism attractive. Propagating a zone that is wrong 42% of the time onto
those notes would have been actively harmful, and the design's §10 named this
("propagation would then actively mislead"). A0 cost 291 s of CPU to find out
instead of a build.

## The 0.3437 → 0.7797 gap is a banked negative, not a lead

*Corrected 2026-07-29, same day: an earlier revision of this report flagged
this gap as "worth a separate look". It is not — the experiment has already
been run and reverted, and this section now says so, so no one re-walks it.*

The gap between the shipped table's per-note top-1 (**0.3437**) and a
gold-calibrated refit (**0.7797**) is a **level** error, and the level error
is real, known, and measured three independent ways: Q6's LOPO residual
−0.566, N5's perturbation sweep, and N4's direct hex measurement +0.780.
A0's measured +0.52 is a fourth measurement of the same quantity.

**Correcting it was tried and reverted the same day (2026-07-24).** A uniform
+0.60 log-B level correction gained **+0.0160 [+0.0088, +0.0233]** on
GuitarSet dev and then measured **−0.0066 [−0.0224, +0.0079]** on sealed
player 05 — the dev and held-out intervals do not overlap
(`docs/EVAL_REPORTS/player05_batched_confirm_2026-07-24.md`).

**Why no constant can work:** N4 measured the per-player offset at
**+0.514, +0.748, +0.780, +0.794, +1.092** — a spread wider than the
correction itself. The level error is genuine but *instrument-specific*,
which is also why N4's per-instrument ritual lost to a fixed constant, and
why q6's self-calibration arms land at +0.0000 (blind), −0.0029 (pooled) and
+0.0388 (self-seeded) against an incumbent worth +0.0522.

`tabvision/tabvision/fusion/string_physics.py` carries the standing
instruction in-source: *"Do not re-derive this constant from dev clips; the
negative is banked."* A0 changes none of that — it re-measures the offset on
a fourth occasion and lands inside the known range.

## Honest limits

- **Gold-timed notes**, so coverage here is an upper bound; the detected
  stream would read fewer notes and time them worse.
- **`self_seeded` is mildly optimistic** — its provisional labels come from
  `guitarset-v1`, which is in-sample for players 00–03.
- **The gate legs were frozen before any run** (design §7, merged `80eae31`),
  and the arm definition was corrected before the full run (`08207ca`,
  `821b13e`) after a two-track harness check exposed a mis-specified anchor.
  Neither gate value was touched.
- **A0 answers coverage and aim only.** It does not measure Tab F1 and never
  built the propagation.

## Reproduction

```bash
python -m scripts.eval.window_zone_a0 --json ../docs/EVAL_REPORTS/window_zone_a0_2026-07-29.json
```

Physics settings come from the registered `acoustic-physics-v1` artifact
(`min_r2` 0.5, `partial_aware`, σ 0.35, fret exponent 1.0, min clean partials
4). Windows are `fixed_window_groups` at its default 80 ms cluster gap, so
they are bit-identical to those the +0.2756 oracle was measured over. Per-clip
records are in the JSON beside this file.
