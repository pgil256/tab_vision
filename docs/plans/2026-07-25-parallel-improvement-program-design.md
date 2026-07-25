# Parallel improvement program — design

**Date:** 2026-07-25
**Status:** design, awaiting approval on three Phase-0 decisions
**Base:** `main` @ `f08b31b`

Goal: run as many of the identified improvement areas as possible concurrently,
without (a) git conflicts, (b) compute contention, or (c) the measurement error
that comes from adding up effects that overlap.

---

## The three problems with naive parallelism

Before the schedule, the three things that make this non-trivial. Each one
drives a structural choice below.

**1. Effects are not additive.** Tracks A, B, and C all attack the same bucket —
`wrong_position_same_pitch`. If each measures +0.02 in isolation, the joint
result is *not* +0.06; they are competing to fix overlapping notes. Summing
isolated deltas would overstate, and this repo has been bitten by exactly this
class of error before (N5: "a shared factor is not a no-op when only one side of
a comparison moves"). **Mitigation:** every track measures against one frozen
baseline, and Phase 2 measures the *joint* configuration directly and reports
the interaction term explicitly.

**2. The eval harness is a contended resource.** A 60-clip run is ~1,054 s; dev
is 300 clips. Four tracks each gating independently would serialize on compute
anyway. **Mitigation:** Phase 0 banks the audio stage once. Every fusion-side
track is then a pure offline replay costing seconds, and they genuinely run in
parallel. This is the pattern that made the Q-program cheap; here it is the
enabler for concurrency, not just for speed.

**3. Each opening of a sealed set costs its integrity.** Four tracks confirming
separately would open the held-out set four times. **Mitigation:** tracks gate
on dev only. The sealed set is opened **once**, in Phase 2, for the combined
configuration.

---

## Phase 0 — shared substrate (SERIAL, blocks everything)

Nothing else starts until this lands. It is short, and it is what makes the rest
parallel.

### P0.1 — Rotate the sealed confirmation set ⚠️ needs a decision

Player-05 has been opened twice for the physics artifact. The reports say so
directly: *"its value as a sealed set is correspondingly reduced."* Every
headline number rides on it, and four incoming tracks all want confirmation.

| Option | Cost | Consequence |
|---|---|---|
| **Rotate to a fresh GuitarSet player** (recommended) | Priors rebuild OOF on 4 players instead of 5; one-time baseline re-base | Confirmations stay trustworthy and comparable |
| Keep player-05, relabel it "development-adjacent" | Free | No sealed set at all; no future claim is confirmable |
| Use GAPS test-22 as the sealed set | Free | Domain shift — classical, nylon; measures a different thing |

**Recommendation:** rotate. Accept the one-time re-base cost now, while there is
a program's worth of work about to be measured, rather than discovering later
that nothing can be confirmed. Re-baseline once in P0.4 and every track measures
against the new number.

### P0.2 — Bank the audio stage

Cache `AudioEvent[]` and the candidate lattice for dev + the new confirmation
set under the current default. Assert exact reproduction of the production
decode before trusting the cache — the F8 runner already does this and its
pattern should be reused. Output goes under `$TABVISION_DATA_ROOT`, git-ignored.

**This is the highest-leverage single item in the plan.** It converts four
compute-bound tracks into four CPU-free ones.

### P0.3 — Re-run the error decomposition on the current default

The reason the current prioritization is inherited rather than trustworthy. Run
on dev + confirmation, current default, all six buckets, per tier.

Its output **re-ranks Tracks A–D**, and may kill one outright — if the physics
channel already shrank `wrong_position_same_pitch` below `missed_onset`, then
Track D is the priority and A/B/C are chasing a bucket that is no longer
dominant. Do not skip this to save a day.

### P0.4 — Freeze the baseline and publish the protocol

One report: baseline numbers on the new split, current bucket shares, and the
comparison protocol every track must use (same seed, same bootstrap, same OOF
priors, paired where possible).

---

## Phase 1 — parallel tracks

Five tracks, one git worktree each, disjoint file ownership.

### Track A — physics-channel coverage
**Branch:** `accuracy/a-physics-coverage`
**Owns:** `fusion/string_physics.py`, `fusion/inharmonicity.py`,
`scripts/eval/build_acoustic_physics_v1.py`, `fusion/priors/acoustic_physics_v1.*`

Coverage is the binding constraint: 8,709 notes isolated → 2,284 fit → 2,227
applied. The loss is in *the fit succeeding*, not in isolation, so the work is
in admission and estimation quality.

Probes, cheapest first: confidence-weighted admission instead of binary
`min_r2`; aggregate `B` across a note's full duration rather than one window;
fit the decay tail where neighbours have died away.

**Gate:** offline replay on banked dev, OOF priors, ΔTab F1 lo-95 > 0, *and* no
drop in per-note accuracy among already-covered notes — coverage bought by
admitting bad fits is not a gain.

**Risk:** low. Known mechanism, proven effect, pure offline iteration.

### Track B — learned timbral string classifier
**Branch:** `accuracy/b-timbre`
**Owns:** new `fusion/timbre_*.py`, `scripts/train/`, its own artifact

Fills the empty `guitarset-timbre-v1` slot. Physics proved per-note string
evidence is the winning shape; timbre is the complement, aimed at the ~74% of
notes whose partials are unreadable.

**Must be measured conditional on physics abstaining.** A model that is merely
correlated with physics adds nothing. The two-leg second-opinion bench with its
derived break-even (`q4_breakeven_precision.py`) already exists for exactly this
question — use it before training anything large.

**Gate:** complementarity first (cheap, offline), then accuracy.
**Needs approval:** training compute beyond free local/Colab; any new dataset.
**Risk:** highest ceiling, highest chance of a banked negative. That is fine —
the program is designed to absorb negatives cheaply.

### Track C — session-adaptive position prior
**Branch:** `accuracy/c-prior-adapt`
**Owns:** `fusion/position_prior.py`, `fusion/neck_prior.py`,
`fusion/inference_policy.py`

The prior is population statistics; a player whose habits differ is
systematically mis-assigned and nothing notices. The capo case shows the
magnitude when prior and session disagree: 0.2956 against a 0.6773 control —
collapse, not degradation.

Two sub-items: (i) adapt within-session from the decoder's own high-confidence
assignments; (ii) consume the assisted-review corrections the UI already
collects and currently discards.

**Gate:** must not regress the population case. An adaptive prior that helps
unusual players and hurts typical ones is a loss.
**Note:** Track C owns `inference_policy.py`. Track B's registration lands as a
small follow-up after C merges, to keep the file single-owner.

### Track D — the detection buckets
**Branch:** `accuracy/d-detection`
**Owns:** `eval/error_decomposition.py` extensions, audio-backend probes

`missed_onset` + `extra_detection` = 28.7% of loss and have had a fraction of
the attention. Neither has been decomposed the way `pitch_off` was.

**Measurement first, and possibly measurement only.** The A10 precedent is the
model: decomposing `pitch_off` *closed* it as a fix target (52% "other", no
dominant mode) and saved the build. Do the same here — decompose `missed_onset`
by texture/velocity/masking and `extra_detection` by harmonic relationship to a
true note. **Do not build a fix until a dominant fixable mode is demonstrated.**

Hypothesis worth testing early: 564 of 672 missed onsets are in strummed, which
points at masking inside dense voicings rather than a general recall problem.

### Track E — hygiene
**Branch:** `chore/e-hygiene`
**Owns:** `docs/accuracy-loop-state.md`, the three deployment branches, `SPEC.md`

Refresh or retire the stale loop-state file; merge or close
`codex/fix-live-deployment`, `codex/record-live-shutdown`,
`docs/prod-repoint-2026-07-09`; update SPEC §1.4.1's information-limited framing.

**Needs approval:** the SPEC edit. Zero conflict with any other track.

---

## Phase 2 — integration (SERIAL)

Merge order: **E → D → A → C → B** (least to most entangled with fusion
evidence). After each merge, re-run the banked-replay measurement.

Then, once:

1. Measure the **joint** configuration on dev.
2. Report joint Δ *alongside* the sum of isolated Δs. **The gap is the
   interaction term and gets written down**, not hidden. A joint result well
   below the sum means the tracks were fixing the same notes — useful
   information, not a failure.
3. Cross-domain leg on GAPS.
4. **One** opening of the sealed confirmation set, for the combined
   configuration only.
5. Update README / NARRATIVE / CLAUDE.md and the per-artifact license map.

---

## Coordination protocol

This repo has an *observed* concurrent-session hazard — branches moved under an
active session twice on 2026-07-24/25. These rules are not theoretical.

- **One worktree per track.** `git worktree add` off the frozen Phase-0 commit.
  Never `git checkout` over another worktree's branch.
- **Only the integrator advances `main`.** Tracks never merge to main.
- **DECISIONS.md is written at merge time, not during.** Each track drops its
  entry in `docs/decisions-inbox/<track>.md`; the integrator appends them in
  order. This file caused an append-vs-append conflict on both merges performed
  on 2026-07-25 — routing around it is free.
- **Re-read `git log` before assuming a tip.** Check `git worktree list` first.
- Each track keeps its own section in a shared `docs/parallel-program-state.md`;
  nobody edits another track's section.

---

## What blocks on the user

Resolve these before Phase 1 opens, so tracks are not interrupted mid-flight:

1. **P0.1 sealed-set rotation** — methodological, re-bases every baseline.
2. **Track B compute/data** — free local + Colab is pre-approved; anything paid
   or any new dataset is not.
3. **Track E SPEC §1.4.1 edit** — spec changes need explicit approval.

---

## Honest assessment of the parallelism

Genuinely concurrent: **A, B, D, E**. Four tracks, disjoint files, no compute
contention after Phase 0.

**C is the awkward one.** It shares `inference_policy.py` with B's registration
and competes with A for the same errors. It is sequenced after A in the merge
order for that reason, and its follow-up ordering with B is explicit above.

The real limit is not machinery, it is that **A, B, and C are three attempts on
one bucket.** Phase 0's decomposition may show that bucket is no longer the
biggest, in which case the honest move is to cut one of them and move the effort
to D. That decision point is built into P0.3 deliberately.
