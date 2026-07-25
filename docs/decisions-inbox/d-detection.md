## 2026-07-25 — close `extra_detection`, open `missed_onset`

**Phase:** Parallel improvement program, Track D (detection buckets)
**Decision tree:** Phase 0 promoted the detection buckets to a first-class track
(33.3% of dev loss, 39.4% of sealed). Following the A10 precedent, decompose
before building: look for a dominant fixable mode in each bucket and build only
if one survives.
**Branch taken:** **Close `extra_detection`** as a fix target. **Keep
`missed_onset` open**, with a masking-aware detection pass as the single build
candidate this probe supports. Build nothing yet.
**Evidence:** 300 development clips, current default, residuals read from the
shipped matcher via a new out-parameter rather than a reimplementation.
Report: `docs/EVAL_REPORTS/d_detection_probe_2026-07-25.md`.

`extra_detection` (3,324), against the interval content of the music: fifths and
fourths are 29.6% of spurious detections and **37.2% of the intervals the music
contains** — lift **0.80×**, i.e. depleted. `other` 0.92×, `semitone` 0.64×.
Only octaves (10.2%, **2.32×**) and unisons (6.9%, **1.53×**) are genuinely
enriched, together ~17% of the bucket and ~2.5% of total loss. Ring-out is 1.4%,
so offset handling is not the problem either.

`missed_onset` (4,371), against all gold notes: 3+ simultaneous neighbours
**49.0% vs 30.1% base (1.63×)**; sounding alone **13.5% vs 30.2% (0.45×)**;
short notes < 150 ms **34.0% vs 21.1% (1.61×)**. Register shows no
concentration.
**Reasoning:** The two buckets look similar in the Phase 0 totals and are not
alike at all. `extra_detection`'s attractive harmonic-leakage story is a
base-rate artifact — the largest identified class in the bucket is the one class
the model is *under*-represented in — and what remains genuinely enriched is too
small to build against. That reproduces A10's `pitch_off` outcome for the same
reason, so two of the three non-position buckets are now closed on the same
finding. `missed_onset` is the opposite: both hypotheses clear their base rates
decisively and point the same way. The detector is not failing at recall
generally — a note sounding alone is missed at less than half the base rate — it
is failing inside dense simultaneity and on very short notes.

**Method note worth carrying:** without the base-rate columns this probe would
have reported "harmonic leakage is the dominant mode of `extra_detection`",
which is precisely backwards. The repo has made the conditional-without-marginal
error before (A14 read 0.285 without its 0.382 marginal; F7 repeated the shape).
The probe now computes both by construction and prints them adjacently, so the
conditional cannot be read without its marginal.

---

## 2026-07-25 — expose matcher residuals instead of reimplementing the matcher

**Phase:** Parallel improvement program, Track D (prerequisite)
**Decision tree:** Characterising missed and spurious events needs the events
themselves; `decompose_errors` computes exactly which they are and returns only
counts. Either reimplement the matching in the probe, or have the matcher hand
them out.
**Branch taken:** Add an optional `Residuals` out-parameter to
`decompose_errors`, following the A10 precedent (which added `pitch_off_deltas`
to the same function for the same reason). Omitting it changes nothing.
**Evidence:** 39 existing `error_decomposition` tests pass unchanged, plus 4 new
ones asserting the residuals agree with the counts they accompany, that omitting
the parameter is a no-op, and that a fresh instance starts empty. 1113 tests,
ruff and mypy clean.
**Reasoning:** A second copy of the matching would drift from the scored one,
and a diagnosis computed against a different matcher describes a pipeline that
does not ship. This is the same argument that split the physics channel in
Track A: derive the diagnostic from the production path rather than beside it.
