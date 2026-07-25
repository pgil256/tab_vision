## 2026-07-25 — coverage is not the lever; the binary admission gate is

**Phase:** Parallel improvement program, Track A (physics-channel coverage)
**Decision tree:** Phase 0 measured the channel's coverage at 22.4% of events
and named coverage the binding constraint. Test whether the 77.6% of notes
currently receiving no evidence are a reservoir of unused signal, by relaxing
the `min_r2` admission threshold; and separately whether the threshold should be
a threshold at all.
**Branch taken:** Reject every hard-threshold relaxation. Adopt
confidence-weighted admission (`confidence ≥ 0.30`) as the candidate for
promotion, pending the cross-domain leg and the Phase 2 sealed confirmation.
The shipped default is unchanged by this entry.
**Evidence:** 300 development clips, leave-one-player-out priors, banked
partial-aware fits, paired bootstrap N=10,000 seed 42, replay verified against
the Phase 0 frozen baseline at ±0.0000 drift.

Hard-threshold relaxation is **CI-significantly worse at every step**, with all
five intervals entirely below zero: `min_r2` 0.40 −0.0050 [−0.0087, −0.0014],
0.30 −0.0088, 0.20 −0.0135, 0.10 −0.0192, 0.00 −0.0206 [−0.0293, −0.0120].
Coverage nearly doubled (22.4% → 43.1%) and Tab F1 fell 0.6801 → 0.6595.

Confidence weighting (weight ∝ `(r2 − floor)/(1 − floor)`) passes:
`≥0.30` **+0.0071 [+0.0021, +0.0122]** at 28.0% coverage; `≥0.10` +0.0068
[+0.0016, +0.0122]; `≥0.00` +0.0052 [−0.0004, +0.0109], inconclusive.
Report: `docs/EVAL_REPORTS/a_physics_coverage_2026-07-25.md`.
**Reasoning:** The two results are only apparently contradictory. Admitting a
marginal fit at *full* weight asserts confidence the fit does not support, and
that corrupts more than the fit contributes — which is why every threshold
relaxation loses. Admitting the same fit at *its own* weight contributes
proportionally, which is why the soft rule gains accuracy and coverage at once.
The motivating hypothesis — that uncovered notes hold usable signal — is
therefore refuted in its stated form and true in a weaker one. The actionable
statement is not "raise coverage" but "stop treating a continuous quality
measure as a binary".

---

## 2026-07-25 — split the physics channel into a measurement half and a scoring half

**Phase:** Parallel improvement program, Track A (prerequisite)
**Decision tree:** Phase 0 found that `measure_events` had no isolation
parameter and was always strict, so banked fits could not express the shipped
`partial_aware` configuration — and that this had already caused a run to
measure the wrong arm while reporting itself as shipped. Fix the parameter
only, or split the channel at its seam.
**Branch taken:** Split. `measure_events(..., isolation=, min_clean_partials=)`
is now the whole spectral half; `apply_fits(...)` is the scoring half;
`attach_inharmonicity_evidence` is their composition. The contamination check
moves into the measurement half, where it is part of deciding whether a fit
exists. `min_r2` stays in the scoring half because it is an admission threshold
rather than a measurement, which is precisely what made this track's sweep
possible.
**Evidence:** Verified bit-identical rather than merely green — the Phase 0
runner reproduces its pinned player-05 numbers at −0.0000 drift on both arms and
the sealed decomposition matches to the event; 1109 tests, ruff, mypy clean.
Banking 22,694 partial-aware fits over 300 clips costs 2.9 min once, after which
nine admission arms swept in 2.2 min total.
**Reasoning:** Adding the parameter alone would have left two separate
implementations of the same gates — which is what allowed one of them to be
silently missing a mode. Composing the shipped path from the same two halves the
replay uses makes that class of divergence unrepresentable rather than merely
unlikely.
