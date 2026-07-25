## 2026-07-25 — rotate the sealed confirmation set from player 05 to player 04

**Phase:** Parallel improvement program, Phase 0 (P0.1)
**Decision tree:** Player 05 had been opened twice for `acoustic-physics-v1` and
four incoming workstreams all wanted confirmation against it. Either keep using
a spent set, relabel it development-adjacent and have no sealed set at all, or
rotate to a fresh player and accept a one-time re-base.
**Branch taken:** Rotate. Sealed = player **04**, chosen by a mechanical rule
(next player index below the burned one) fixed before any per-player score was
inspected. Player 05 returns to development, so dev keeps 300 clips and only the
roles swap. Every clip — dev and sealed alike — is now scored under
leave-one-player-out priors rebuilt from a six-player pool at the registered
artifacts' hyper-parameters, so no clip is ever scored under a prior that
memorized its own player. The shipped artifacts are unmodified.
**Evidence:** `docs/EVAL_REPORTS/phase0_rotation_baseline_2026-07-25.md` (+
`.json`). The harness reproduces `player05_batched_confirm_2026-07-24`'s
published 0.6340 / 0.7346 to −0.0000 drift on both arms, asserted at runtime.
On the new sealed player 04 the shipped default measures aggregate **0.6609**
(single-line 0.6686, strummed 0.6533) against a no-physics baseline of 0.6087 —
Δ **+0.0522 [+0.0259, +0.0809]**, PASS. Dev over the remaining five players:
0.6801 vs 0.6083, Δ +0.0718 [+0.0558, +0.0885].
**Reasoning:** A sealed set that has been opened twice cannot support a
program's worth of incoming confirmations, and the alternative — discovering
later that nothing is confirmable — is far more expensive than one re-base. The
mechanical selection rule matters as much as the rotation: a player chosen after
seeing scores would bias every confirmation the rotation exists to protect.

---

## 2026-07-25 — the physics channel is worth +0.05 to +0.07, not +0.10

**Phase:** Parallel improvement program, Phase 0 (P0.3/P0.4)
**Decision tree:** Establish the frozen baseline the parallel tracks measure
against, and decide whether the published headline can stand as the reference.
**Branch taken:** Re-base the published numbers. Report the channel's value as a
range with the per-player spread attached, and treat the dev figure over five
players as the population estimate rather than any single held-out player.
**Evidence:** Measured on all six players under leave-one-out priors, the
channel's aggregate gain is +0.0468 (03), +0.0522 (04), +0.0603 (01), +0.0671
(02), +0.0841 (00), +0.1006 (05) — a **2.15× spread, with the previously sealed
player 05 at the maximum**. The Q6 and 2026-07-24 measurements are reproduced
exactly, so nothing was mis-measured; the single-player estimate was simply the
top of a wide range. Shipped aggregate on the new sealed player is 0.6609 versus
the 0.7346 the README carried.
**Reasoning:** A held-out player estimates a population mean only as tightly as
the population is uniform, and this one is not. Publishing the maximum of six
draws as the headline overstates what a new user should expect. The range is
still a clear win and the channel still passes its gate on a fresh sealed
player; only the point estimate was too generous.

---

## 2026-07-25 — Track D promoted; no accuracy track is cut

**Phase:** Parallel improvement program, Phase 0 (P0.3)
**Decision tree:** Phase 0's decomposition was explicitly allowed to kill a
track — if `wrong_position_same_pitch` had fallen below `missed_onset`, Tracks
A/B/C would have been chasing a bucket that was no longer dominant.
**Branch taken:** Keep all four accuracy tracks; promote Track D from
"measurement, probably only measurement" to a first-class track.
**Evidence:** Shipped-arm decomposition. `wrong_position_same_pitch` is still
the largest bucket at **47.6%** of dev loss (63.6% on single-line), down from
57.3% pre-physics and 51.9% in this run's own baseline arm. But `missed_onset` +
`extra_detection` together are **33.3%** of dev loss and **39.4%** of sealed
loss, against 38.6% for wrong position on the sealed player — effectively tied.
Also measured: between the two arms `pitch_off`, `timing_only`, `missed_onset`
and `extra_detection` are identical to the event, and only
`wrong_position_same_pitch` moves (13,088 → 10,999), converting 1:1 into
`correct`. This directly verifies the claim that the channel cannot add, remove,
or retime a note.
**Reasoning:** The dominant bucket is still dominant, so the string-assignment
tracks keep their justification, but the detection buckets are within striking
distance and have had a fraction of the investment. Track D keeps its
measurement-first discipline — the A10 precedent, where decomposing `pitch_off`
closed it and saved the build, is the model — but it is no longer a side quest.
