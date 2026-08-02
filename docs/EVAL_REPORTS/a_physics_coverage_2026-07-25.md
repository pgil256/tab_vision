# Track A — physics-channel coverage: hard thresholds refuted, soft weighting wins

**Date:** 2026-07-25
**Script:** `tabvision/scripts/eval/a_physics_coverage.py`
**Data:** `docs/EVAL_REPORTS/a_physics_coverage_2026-07-25.json`
**Population:** 300 development clips (players 00-05 excluding sealed 04),
leave-one-player-out priors, banked partial-aware fits. **The sealed player was
not opened.**

## Verdict

The premise was that coverage is the binding constraint — 22.4% of events get
evidence, so the other 77.6% must be worth something. That premise is **half
wrong, and the half that is wrong is the interesting half.**

**Admitting more notes at full weight makes things worse, monotonically.**
Every reduction of `min_r2` below the shipped 0.5 loses accuracy, all the way
down: coverage nearly doubles to 43.1% and Tab F1 *falls* from 0.6801 to 0.6595.
The discarded fits are not a reservoir of unused signal. They are mostly noise,
and the shipped threshold is close to correctly placed for a hard rule.

**But the hard rule is itself the mistake.** Replacing the threshold with a
weight that scales with fit quality gains accuracy *and* coverage
simultaneously: **+0.0071 [+0.0021, +0.0122] over shipped**, paired, with
coverage rising 22.4% → 28.0%.

The two results are consistent and together they say something sharper than
either alone: **the marginal fits carry real but proportionally weaker
information.** Admitted at full confidence they corrupt; admitted at their own
confidence they contribute. A binary gate on a continuous quality signal was
discarding usable evidence at one end and over-trusting it at the other.

## Results

Development, 300 clips, paired against the Phase 0 frozen baseline.

| Arm | Tab F1 | Δ vs baseline | 95% CI | Coverage |
|---|---:|---:|---|---:|
| baseline (no channel) | 0.6083 | — | — | 0.0% |
| **shipped** (`min_r2` 0.5, hard) | 0.6801 | +0.0718 | `[+0.0558, +0.0885]` | 22.4% |
| `min_r2` 0.40 | 0.6750 | +0.0667 | `[+0.0498, +0.0845]` | 25.3% |
| `min_r2` 0.30 | 0.6713 | +0.0630 | `[+0.0453, +0.0815]` | 28.0% |
| `min_r2` 0.20 | 0.6666 | +0.0583 | `[+0.0399, +0.0774]` | 30.9% |
| `min_r2` 0.10 | 0.6609 | +0.0526 | `[+0.0333, +0.0726]` | 34.3% |
| `min_r2` 0.00 | 0.6595 | +0.0512 | `[+0.0306, +0.0724]` | 43.1% |
| **confidence ≥ 0.30** | **0.6872** | **+0.0789** | `[+0.0645, +0.0941]` | 28.0% |
| confidence ≥ 0.10 | 0.6869 | +0.0786 | `[+0.0627, +0.0952]` | 34.3% |
| confidence ≥ 0.00 | 0.6853 | +0.0770 | `[+0.0609, +0.0941]` | 43.1% |

The confidence arms use weight ∝ `(r2 − floor) / (1 − floor)`, so a fit at the
floor contributes nothing and a perfect fit contributes the shipped weight.

### Head-to-head against shipped — the gate that decides

Every arm above sits inside every other arm's interval against *baseline*, so
that column decides nothing. Paired against `shipped` on identical clips it is
unambiguous:

| Arm | Δ vs shipped | 95% CI | Verdict |
|---|---:|---|---|
| `min_r2` 0.40 | −0.0050 | `[−0.0087, −0.0014]` | **regression** |
| `min_r2` 0.30 | −0.0088 | `[−0.0139, −0.0037]` | **regression** |
| `min_r2` 0.20 | −0.0135 | `[−0.0196, −0.0074]` | **regression** |
| `min_r2` 0.10 | −0.0192 | `[−0.0263, −0.0120]` | **regression** |
| `min_r2` 0.00 | −0.0206 | `[−0.0293, −0.0120]` | **regression** |
| **confidence ≥ 0.30** | **+0.0071** | `[+0.0021, +0.0122]` | **PASS** |
| confidence ≥ 0.10 | +0.0068 | `[+0.0016, +0.0122]` | **PASS** |
| confidence ≥ 0.00 | +0.0052 | `[−0.0004, +0.0109]` | inconclusive |

All five hard-threshold relaxations are **CI-significant regressions** — not
merely unhelpful, actively worse, with intervals entirely below zero. Two
confidence arms clear the bar; the third, which admits every fit however poor,
does not, which is the expected place for the ramp to run out.

## Reading the monotonic loss

The hard-threshold sweep is worth stating plainly because it refutes the
motivating hypothesis. Coverage rose 22.4% → 43.1%, an extra ~10,000 notes
receiving evidence, and every increment cost accuracy. If the uncovered notes
had held usable signal at full weight, some threshold below 0.5 would have
beaten 0.5. None did.

This is a useful negative on its own: **"raise coverage" is not the lever.**
"Stop treating a continuous quality measure as a binary" is.

## Notes on the method

**The banked replay is exact.** The sweep reproduces Phase 0's frozen dev
numbers at ±0.0000 on both the baseline and shipped arms, asserted in the run.
That check is why the arms above can be compared at all — a replay that drifted
would make every row incommensurable with the frozen baseline.

**Banking cost 2.9 minutes once; each subsequent arm is free.** 22,694
partial-aware fits over 300 clips. Nine arms were swept in 2.2 minutes total.
This is only possible because `measure_events` gained an `isolation` parameter
in this track's first commit — before that a partial-aware cache could not be
represented, which is the defect that made Phase 0's first pass measure the
wrong arm.

## Limits

- Development only. The sealed player has not been opened and must not be until
  a configuration is frozen for promotion.
- The confidence ramp is linear in `r2` because that is the simplest form that
  expresses "trust proportionally". It has not been compared against other
  shapes, and the floor was swept only coarsely (0.30 / 0.10 / 0.00). The three
  floors land within 0.002 of each other, so the *shape* matters more than the
  floor and that is where any follow-up should look.
- `r2` is a proxy for fit trustworthiness, not a calibrated probability. A
  properly calibrated confidence would likely do better still, and would be the
  principled version of this result.
- **+0.0071 is small** — a tenth of what the channel itself is worth. It is
  CI-significant and it comes with 5.6 points of extra coverage, but it should
  not be oversold, and it has not yet faced the cross-domain leg.

## Next

Promotion needs the house two-leg gate, and only the first leg is done:

1. ✅ in-domain (this run, dev, paired vs shipped)
2. ⬜ cross-domain — GAPS no-regression. The channel's domain guard means
   classical sessions abstain by construction, so this may again be satisfiable
   by proof rather than by a run; that must be checked, not assumed.
3. ⬜ sealed-player confirmation, once at Phase 2 integration, alongside the
   other tracks rather than separately.

