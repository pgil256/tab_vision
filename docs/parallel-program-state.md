# Parallel improvement program — state

last_updated: 2026-07-25
design: `docs/plans/2026-07-25-parallel-improvement-program-design.md`

This file is the program's memory. Read it first. Each track owns its own
section and nobody edits another track's.

## Phase 0 — shared substrate

| id | item | status | key numbers |
|----|------|--------|-------------|
| P0.1 | rotate the sealed set | **done** | sealed `04` (pre-declared); dev `00,01,02,03,05`; leave-one-player-out priors over all six, five per fold |
| P0.2 | bank the audio stage | **done** | 360/360 clips banked; run refuses to re-transcribe |
| P0.3 | decomposition on the current default | **done** | see below |
| P0.4 | freeze baseline + publish protocol | **done** | this file + the report |

Runner: `tabvision/scripts/eval/phase0_rotation_baseline.py`.
Report: `docs/EVAL_REPORTS/phase0_rotation_baseline_2026-07-25.md` (+ `.json`).

### The rotation

Player 05 was opened twice for `acoustic-physics-v1` and is spent as a sealed
set. **Sealed is now player 04**, picked by a mechanical rule — next index down —
fixed before any per-player score was inspected. Player 05 returns to
development, so dev keeps 300 clips and only the roles swap.

Every clip is now scored under priors rebuilt without its own player, from a
six-player pool, at the registered artifacts' hyper-parameters. Dev and sealed
numbers are therefore directly comparable. **The shipped artifacts are
unmodified** — this is a measurement substrate, not a product change.

### Frozen baseline

Leave-one-player-out priors; `shipped` = the current default.

| Split | Arm | Single-line | Strummed | Aggregate |
|---|---|---:|---:|---:|
| **dev** (300) | baseline | 0.5504 | 0.6662 | 0.6083 |
| **dev** (300) | **shipped** | **0.6854** | **0.6747** | **0.6801** |
| **sealed 04** (60) | baseline | 0.5854 | 0.6320 | 0.6087 |
| **sealed 04** (60) | **shipped** | **0.6686** | **0.6533** | **0.6609** |

Δ dev `+0.0718 [+0.0558, +0.0885]`; Δ sealed `+0.0522 [+0.0259, +0.0809]`.
Physics coverage 22.4% of events. Onset/pitch are identical between arms
(dev 0.9270 / 0.9094; sealed 0.9032 / 0.8673).

**Tracks gate against the dev row and nothing else.**

### What Phase 0 changed about the plan

**The published headline was the luckiest of six players.** The channel's gain
ranges `+0.047` to `+0.101` across players — 2.15× — and player 05 was the
maximum. The honest estimate is **+0.05 to +0.07**, not +0.10. The README's
0.7346 has been re-based to 0.6609 on the new sealed player.

**Track D is promoted.** `missed_onset` + `extra_detection` = **33.3%** of dev
loss and **39.4%** of sealed loss, against 47.6% / 38.6% for
`wrong_position_same_pitch`. On the sealed player they are effectively tied. The
detection buckets are no longer a minor track, and no track is cut.

**`wrong_position_same_pitch` is still #1** (47.6% dev, 63.6% on single-line), so
Tracks A/B/C keep their justification — but it has fallen 57.3% → 47.6% and the
margin is narrowing.

**Track A has a defined first task:** add an `isolation` parameter to
`measure_events` so partial-aware fits can be banked. Today they cannot, so
every physics sweep recomputes them. See the report's closing section.

**A gate leg is at risk.** Pitch F1 on the new sealed player is **0.8673**,
below the SPEC §1.4 `≥ 0.90` gate (dev 0.9094, player 05 ~0.93). Not a
regression — player 04 is simply harder — but any future acceptance run should
report the per-player spread rather than one held-out figure. Flagged for
Track E.

### Protocol every track follows

- Gate on **dev only**. The sealed set is opened once, in Phase 2, for the
  combined configuration. Every opening spends the set's value.
- Use the leave-one-player-out priors. Dev clips scored under the registered
  artifacts are in-distribution and read optimistically.
- Paired comparisons against the frozen baseline, bootstrap N=10,000, seed 42.
- Report `lo95 > 0` as the bar, and report the point estimate even when it fails.
- **Do not sum isolated deltas across tracks.** Tracks A/B/C attack the same
  bucket and their gains overlap; Phase 2 measures the joint configuration.
- Pin an external published number where one exists. Internal consistency checks
  do not catch a whole harness measuring the wrong configuration — see the
  Phase 0 report's methodology note, which is there because it happened.

## Phase 1 tracks

| track | branch | status | headline |
|---|---|---|---|
| A — physics coverage | `accuracy/a-physics-coverage` | **result, dev leg passed** | hard thresholds refuted; confidence weighting **+0.0071 [+0.0021, +0.0122]** vs shipped at 28% coverage |
| B — timbral string classifier | `accuracy/b-timbre` | not started | needs compute approval before training |
| C — session-adaptive prior | `accuracy/c-prior-adapt` | not started | owns `inference_policy.py` |
| D — detection buckets | `accuracy/d-detection` | **result** | both buckets decomposed; see section |
| E — hygiene | `chore/e-hygiene` | **done** | 3 stale branches closed, loop-state frozen |

### Track A — physics coverage

**Dev leg PASSED; not promoted.** The motivating premise — that the 77.6% of
uncovered notes hold usable signal — is **refuted in its stated form**. Every
relaxation of `min_r2` is a CI-significant regression against shipped (0.40
−0.0050 … 0.00 −0.0206), with coverage nearly doubling to 43.1% while Tab F1
falls 0.6801 → 0.6595.

What works instead is **weighting by fit quality rather than thresholding on
it**: `confidence ≥ 0.30` gains **+0.0071 [+0.0021, +0.0122]** over shipped
*and* raises coverage to 28.0%. A marginal fit admitted at full weight asserts
confidence it cannot support; admitted at its own weight it contributes.

Shipped default unchanged. Still needs the cross-domain leg and the Phase 2
sealed confirmation. **+0.0071 is a tenth of the channel's own value** — real,
CI-significant, not to be oversold.

Prerequisite landed: `measure_events` / `apply_fits` split, verified
bit-identical against Phase 0's pinned numbers (±0.0000). Banking 22,694
partial-aware fits costs 2.9 min once; arms then sweep in seconds.

Report: `EVAL_REPORTS/a_physics_coverage_2026-07-25.md`.

### Track E — hygiene

**Done.** Three "unmerged" branches (`codex/fix-live-deployment`,
`codex/record-live-shutdown`, `docs/prod-repoint-2026-07-09`) were not pending
work — their content is already in `main` via PR #34 and other paths. Local refs
deleted; **the remote copies still exist and deleting them is the user's call**.

`accuracy-loop-state.md` frozen as historical with its wrong claims enumerated
rather than refreshed, since the program it describes is closed.

Incidental: the 2026-07-13 entry retires the *old* `pgil256` Modal workspace,
not the deployment — the README's "Modal production deploy" claim is accurate.

### Track A — physics coverage
_(nothing yet)_

### Track B — timbral string classifier
_(nothing yet)_

### Track C — session-adaptive prior
_(nothing yet)_

### Track D — detection buckets
_(nothing yet)_

### Track E — hygiene
_(nothing yet)_

## Coordination

- One worktree per track, off the frozen Phase-0 commit. Never `git checkout`
  over another worktree's branch.
- Only the integrator advances `main`.
- Decision entries go to `docs/decisions-inbox/<track>.md`, never straight into
  `docs/DECISIONS.md` — see that directory's README for why.
- Re-read `git log` and `git worktree list` before assuming a branch tip.
