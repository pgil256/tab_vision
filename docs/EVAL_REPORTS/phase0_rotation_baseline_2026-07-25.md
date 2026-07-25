# Phase 0 — sealed-set rotation, frozen baseline, current-default decomposition

**Date:** 2026-07-25
**Script:** `tabvision/scripts/eval/phase0_rotation_baseline.py`
**Data:** `docs/EVAL_REPORTS/phase0_rotation_baseline_2026-07-25.json`
**Design:** `docs/plans/2026-07-25-parallel-improvement-program-design.md`

Three Phase-0 items in one pass: rotate the sealed confirmation set, re-baseline
against it, and decompose the error profile of the configuration that ships.

## Verdict

The channel passes on the new sealed player (**+0.0522 [+0.0259, +0.0809]**),
`wrong_position_same_pitch` is still the largest bucket (**47.6%** dev), and all
four Phase 1 accuracy tracks remain justified.

**But the published headline was the luckiest of six players.** Player 05 gave
the channel `+0.1006`; the other five give `+0.047` to `+0.084`. On the newly
sealed player 04 the shipped default measures **0.6609 aggregate**, not the
**0.7346** the README carried. That is a −0.074 re-base, and it is the main
product of this run.

## The rotation

Player 05 was opened twice for `acoustic-physics-v1` and is spent as a sealed
set. **Sealed is now player 04**, chosen by a mechanical rule — the next player
index below the burned one — fixed *before* any per-player score was inspected.
Player 05 returns to development, so dev keeps 300 clips and only the roles swap.

    dev    = {00, 01, 02, 03, 05}   300 clips
    sealed = {04}                    60 clips

## Frozen baseline

Leave-one-player-out priors throughout; `shipped` = `acoustic-physics-v1` at its
registered settings with `partial_aware` isolation.

### Sealed (player 04, 60 clips) — the number Phase 2 confirms against

| Arm | Single-line | Strummed | Aggregate |
|---|---:|---:|---:|
| baseline (no string evidence) | 0.5854 | 0.6320 | 0.6087 |
| **shipped (current default)** | **0.6686** | **0.6533** | **0.6609** |
| Δ | +0.0832 | +0.0213 | **+0.0522** |

Aggregate Δ 95% CI `[+0.0259, +0.0809]` — PASS.
Onset F1 0.9032, pitch F1 0.8673 (identical in both arms).

### Dev (300 clips) — the number Phase 1 tracks gate against

| Arm | Single-line | Strummed | Aggregate |
|---|---:|---:|---:|
| baseline | 0.5504 | 0.6662 | 0.6083 |
| **shipped** | **0.6854** | **0.6747** | **0.6801** |
| Δ | +0.1350 | +0.0085 | **+0.0718** |

Aggregate Δ 95% CI `[+0.0558, +0.0885]`.
Onset F1 0.9270, pitch F1 0.9094 (identical in both arms).

Physics coverage: **22.4%** of events on dev (11,528 / 51,392), 21.2% on sealed.

## Per-player spread — why the headline moved

| Player | baseline | shipped | Δ | solo Δ | |
|---|---:|---:|---:|---:|---|
| 00 | 0.6055 | 0.6896 | +0.0841 | +0.1414 | |
| 01 | 0.5342 | 0.5945 | +0.0603 | +0.1245 | |
| 02 | 0.6047 | 0.6718 | +0.0671 | +0.1230 | |
| 03 | 0.6631 | 0.7098 | +0.0468 | +0.1105 | |
| **04** | 0.6087 | **0.6609** | **+0.0522** | +0.0832 | **sealed** |
| 05 | 0.6340 | **0.7346** | **+0.1006** | +0.1754 | previously sealed |

**The channel's gain varies 2.15× across players (+0.047 to +0.101), and player
05 is the maximum.** Nothing was wrong with the Q6 or 2026-07-24 measurements —
this run reproduces them exactly — but a single held-out player estimates a
population mean with a spread this wide only loosely. The defensible statement
about the physics channel is now "**worth roughly +0.05 to +0.07 aggregate**",
with +0.1006 understood as the top of the observed range rather than the
expectation.

The same caution applies to every number in this report: player 04 is one draw
too. The dev figure over five players is the better population estimate; the
sealed figure exists to catch overfitting, not to be the headline.

## Error decomposition — the current default

Shipped arm, share of total loss.

| Bucket | dev agg | dev single-line | dev strummed | sealed agg |
|---|---:|---:|---:|---:|
| `wrong_position_same_pitch` | **47.6%** | **63.6%** | 42.4% | 38.6% |
| `missed_onset` | 18.9% | 14.4% | 20.4% | 21.9% |
| `pitch_off` | 15.1% | 7.0% | 17.8% | 18.3% |
| `extra_detection` | 14.4% | 12.5% | 15.0% | 17.5% |
| `timing_only` | 4.0% | 2.5% | 4.5% | 3.8% |

**The trend, against the 2026-05-13 pre-physics decomposition:**
`wrong_position_same_pitch` has fallen **57.3% → 51.9% (baseline arm) → 47.6%
(shipped)**. It is still the largest single bucket and still dominates
single-line (63.6%), so Tracks A/B/C keep their justification — but the margin
is narrowing, and on the sealed player it is already down to 38.6%.

**Detection is now a third of the loss.** `missed_onset` + `extra_detection` =
**33.3%** on dev and **39.4%** on sealed, against 47.6% / 38.6% for wrong
position. On the sealed player the two are effectively tied. Track D is not a
minor track.

**The channel's effect is surgical, and this run proves it.** Between the two
arms, `pitch_off` (3497), `timing_only` (925), `missed_onset` (4371) and
`extra_detection` (3324) are *identical to the event*. Only
`wrong_position_same_pitch` moves (13,088 → 10,999), converting 1:1 into
`correct` (30,342 → 32,431). The README's claim that the channel "cannot add,
remove, or retime a note" is now directly verified rather than argued.

## A gate leg that would not pass on the new sealed player

Onset and pitch F1 vary by player as much as Tab F1 does. On player 04 the
current default measures **pitch F1 0.8673** — below the SPEC §1.4 gate of
**≥ 0.90** — against 0.9094 on the dev five and ~0.93 on player 05.

This does not retroactively invalidate the v1.0.0 acceptance, which was properly
run against its declared validation set. It does say the pitch leg passed with
less margin than one player suggested, and that a future acceptance run should
report the per-player spread rather than a single held-out figure.

## Methodology

**Priors.** Leave-one-player-out over all six players, at the registered
artifacts' own hyper-parameters (position `alpha=1.0, power=2.0`; sequence
`delta_fret, alpha=0.5, backoff_kappa=8.0, singleton_only=True`, weight 4.0).
Each fold trains on five players — the same number the shipped artifacts saw —
so dev and sealed folds are equally powered and no clip is ever scored under a
prior that memorized its own player. **The shipped artifacts are unmodified;**
this is a measurement substrate, not a product change.

**Arms.** Both score identical detections from the banked ensemble cache, so
every delta is attributable to the fusion stage alone. The run refuses to
re-transcribe: all 360 clips must already be banked.

**Statistics.** Paired per-clip deltas, bootstrap N=10,000, seed 42.

## Methodology note — the bug this run's check exists to catch

The first pass of this script reported itself as measuring the shipped
configuration and did not. It replayed evidence through N5's banked
`apply_banked`, whose cached fits come from `measure_events` — which takes **no
isolation argument** and is therefore always strict. So it measured the
`raw-strict` arm (player-05 aggregate 0.7119) while its header printed
`isolation=partial_aware` and its output was labelled `shipped` (true value
0.7346).

Nothing failed. The script had a self-check comparing the banked replay against
the live path — but it invoked the live path with the *default* isolation, so
both sides carried the same error and agreed perfectly.

The fix is not a better internal check. It is **pinning an external published
number**: player 05's leave-one-out fold trains on exactly the five players the
registered artifacts saw, so this run must reproduce
`player05_batched_confirm_2026-07-24`'s 0.6340 / 0.7346. It now does, to
**−0.0000 drift on both arms**, asserted at the end of every invocation with a
0.0015 tolerance.

The general lesson for every Phase 1 track: **an internal consistency check
cannot detect a whole harness measuring the wrong configuration.** Pin something
the harness did not produce.

## Known limitation handed to Track A

`measure_events` has no `isolation` parameter, so **partial-aware fits cannot be
banked** — only strict ones. This run recomputes fits live for all 360 clips
(2.9 min), and any future physics sweep pays that cost on every iteration.

Track A's first task should be to add that parameter (additively, defaulting to
current behaviour) and bank a partial-aware fit cache. Its entire workload is
sweeping admission parameters against fixed fits, so this turns its inner loop
from minutes into seconds. It was deliberately left undone here to keep Phase 0
to its scope and to avoid pre-empting the track that owns `inharmonicity.py`.
