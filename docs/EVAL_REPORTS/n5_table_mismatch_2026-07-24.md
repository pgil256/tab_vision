# N5 table-mismatch robustness — ROBUST; the stated Q6 caveat is discharged

Accuracy-loop post-program iteration (N5). Q6 registered
`acoustic-physics-v1` and every gate it passed was measured on GuitarSet,
whose guitars resemble the shipped table. The pending default-on decision
therefore carried one explicit caveat, quoted from the state file:

> Caveat: all validation is GuitarSet on similar steel-strings —
> portability to another guitar is argued from physics, not measured.

This study replaces that argument with a tolerance curve: **how wrong can
the table be before the channel stops helping?**

300 GuitarSet dev clips (players 00–04), LOPO position prior, frozen config
`weight 1.0 / min_r2 0.50 / sigma 0.35`, 17 arms declared in source before
the run, paired bootstrap N = 10,000 seed 42. Baseline Tab F1 **0.6031**.

## Verdict — ROBUST

The reading was pre-declared in the script docstring before any arm ran:

> **Robust** — every derived real-guitar set keeps lo-95 > 0, **and** the
> offset tolerance band that holds lo-95 > 0 covers at least ±0.10 log-B
> (the span of real acoustic scale lengths, 24.75″–25.6″).

Both legs pass, neither marginally.

- All six derived real-guitar arms hold lo-95 > 0. The weakest is
  `core:round-0.90` at **+0.0222 [+0.0105, +0.0342]** — half the shipped
  gain, still four CI-widths clear of zero.
- The offset band holding lo-95 > 0 spans **at least [−0.40, +0.60] log-B**,
  four to six times the ±0.10 required. Even the worst arm tested
  (−0.60 log-B, a 1.8× table error) is **not significantly negative**:
  +0.0120 [−0.0001, +0.0242].

**Nothing tested anywhere in the sweep is significantly negative.** There is
no table error in the range a real steel-string acoustic can produce that
turns this channel harmful.

## The control: the replay reproduces the shipped result exactly

Every arm's number comes from an offline replay, so the replay's fidelity is
load-bearing. The shipped-table arm is the control:

| | Δ Tab F1 | lo-95 | hi-95 |
|---|---:|---:|---:|
| Q6 full-dev (`q6_full_dev_2026-07-22.md`) | +0.0443 | +0.0339 | +0.0555 |
| this run, `offset+0.00` | **+0.0443** | **+0.0339** | **+0.0555** |

All three figures agree to four decimals against a run made on a different
day by different code. Two unit tests assert the same equivalence directly —
that `apply_banked` returns `fret_prior` arrays bit-identical to
`attach_inharmonicity_evidence`, both where the channel fires and where it
abstains — and the in-run `self_check` re-asserts it on the first three
clips of every sweep.

**Coverage is table-independent.** All 17 arms applied evidence to exactly
**4,354** notes. The `r² ≥ 0.50` gate is a property of the measurement, not
of the table, so the arms differ only in how the same notes are *scored* —
never in which notes are touched. This is what makes the arms comparable.

## The tolerance curve

| offset (log-B) | in σ | table error | Δ Tab F1 | lo-95 | hi-95 |
|---:|---:|---|---:|---:|---:|
| −0.60 | −1.71 | ×0.55 | +0.0120 | −0.0001 | +0.0242 |
| −0.40 | −1.14 | ×0.67 | +0.0228 | +0.0111 | +0.0346 |
| −0.20 | −0.57 | ×0.82 | +0.0363 | +0.0248 | +0.0482 |
| −0.10 | −0.29 | ×0.90 | +0.0404 | +0.0295 | +0.0518 |
| **0.00** | 0.00 | **shipped** | **+0.0443** | **+0.0339** | **+0.0555** |
| +0.10 | +0.29 | ×1.11 | +0.0484 | +0.0381 | +0.0595 |
| +0.20 | +0.57 | ×1.22 | +0.0513 | +0.0407 | +0.0628 |
| +0.40 | +1.14 | ×1.49 | +0.0575 | +0.0463 | +0.0695 |
| +0.60 | +1.71 | ×1.82 | +0.0605 | +0.0486 | +0.0737 |

The curve is **monotone increasing and does not turn over** inside the range
tested. That is not the symmetric bowl a well-centred table would give, and
it is the study's most consequential secondary result — see below.

## Real-guitar arms

Each is the shipped table moved by the *difference* the variant implies, so
the wound model's own fit residual cancels (asserted exact to 1e-12) and an
arm's effect is attributable to the string change alone.

| arm | largest shift | in σ | Δ Tab F1 | lo-95 | hi-95 |
|---|---:|---:|---:|---:|---:|
| `set:extra-light` (.010–.047) | 0.365 | 1.04 | +0.0437 | +0.0332 | +0.0550 |
| `set:medium` (.013–.056) | 0.160 | 0.46 | +0.0433 | +0.0333 | +0.0541 |
| `scale:24.75in` | 0.104 | 0.30 | +0.0487 | +0.0382 | +0.0599 |
| `scale:25.6in` | 0.031 | 0.09 | +0.0436 | +0.0330 | +0.0549 |
| `core:round-0.90` | 0.421 | 1.20 | **+0.0222** | +0.0105 | +0.0342 |
| `core:hex-1.10` | 0.381 | 1.09 | +0.0512 | +0.0407 | +0.0624 |

**Scale length is a non-issue.** The whole 24.75″–25.6″ span moves the table
by −0.03 to +0.10 log-B — a tenth of a σ to three tenths — and every arm
lands within noise of shipped. This also supplies an internal validity
check: scale length is provably a *pure* uniform offset (unit-tested to
1e-9 against `4·ln(L_ref/L)`), so those two arms must reproduce the offset
curve. They do, to **+0.0002** and **+0.0005**. The offset axis is a
faithful summary statistic, not a convenient fiction.

**Gauge is nearly free, and only touches the plain strings.** A three-step
gauge change moves the two plain strings by up to 1.04 σ and the four wound
strings by under 0.05 log-B — because a plain string's `B ∝ d²` while a
wound string's core and total mass move together and largely cancel. Both
gauge arms land within noise of shipped.

**Wound-core construction is the whole risk.** It is also the one spec
manufacturers do not publish. A ±10% core error at fixed gauge moves the
four wound strings by ∓0.42/+0.38 log-B (1.2 σ / 1.1 σ) and is the only arm
that materially costs anything: `core:round-0.90` halves the gain.

## Two findings

### 1. A uniform level error is *not* harmless — and the shipped table is biased low

The Q6 portability report states:

> **Level error is harmless** — a shared factor shifts every candidate for a
> note equally. Only *shape* can flip a decision.

**This measurement refutes that.** A shared factor shifts every candidate's
*prediction*, but the *measurement* does not move with it, so a uniform
offset is exactly equivalent to biasing every measured `log B` by −Δ. It
changes which candidate is nearest. Empirically it is worth **0.049 Tab F1
of swing** across the ±0.60 band (+0.0120 to +0.0605) — comparable to the
entire size of the channel's gain.

The claim's *conclusion* survived by luck: level error is not harmless, but
it is also not dangerous, because the curve is so flat near the middle and
never goes significantly negative.

Three independent lines here point the same way — the shipped table
**under-predicts B**:

1. The offset curve rises monotonically to +0.60 and has not turned over.
2. Q6's own residual analysis measured the physics table low by **0.566 log**
   against the LOPO-fitted table — and the sweep's optimum is somewhere at
   or beyond +0.60, i.e. consistent with ≈ +0.57.
3. `core:hex-1.10` (thicker wound cores, +0.381 on wound strings only) beats
   shipped at +0.0512, while `core:round-0.90` costs half the gain. Q6's
   residual split was −0.53/−0.81/−0.60/−0.71 on wound vs −0.15/−0.20 on
   plain, so real wound cores are thicker than the model assumes — the same
   direction, from a different measurement.

The cause is most likely wound-core geometry rather than an estimator bias,
because a measurement artefact would hit all six strings roughly equally and
the residual is 3–4× larger on wound strings than on plain ones. **This is
not certain** — nothing here separates instrument from estimator directly,
and that separation would need a rig with known core diameters.

**I am not proposing an offset default.** Picking +0.57 because it wins on
these 300 GuitarSet clips is tuning on the fast loop, and the loop's own rule
applies: in-distribution gains mean nothing until a cross-domain gate passes.
The honest route to the scalar is per-rig calibration — which is exactly what
`calibrate_from_ritual` already computes, and exactly the item (N4) that is
blocked pending public audio.

### 2. The channel's value lives on the wound strings

`core:round-0.90` moves only the four wound strings, by −0.421, and lands at
+0.0222. The pure-offset arm that moves **all six** by the same −0.42 lands
at +0.0217. The two are indistinguishable: leaving the plain strings correct
buys essentially nothing.

So the low-E/A/D/G rows carry the channel, and B/e contribute almost none of
its gain. That is consistent with the physics — the plain trebles are the
least inharmonic strings and the most crowded in candidate space — and it
sharpens where any future calibration effort should go: **get the wound cores
right and the rest can stay approximate.**

## Sigma diagnostic — direction only

Pre-declared as diagnostic and never as a proposed default; choosing σ on
this run would be tuning on the test.

| | σ = 0.35 (shipped) | σ = 0.60 |
|---|---:|---:|
| table correct (offset 0.00) | +0.0443 | +0.0300 |
| table badly wrong (offset −0.60) | +0.0120 | +0.0240 |

Widening the posterior behaves as theory says: it **doubles** the gain when
the table is badly wrong and **costs a third** of it when the table is right.
Given the verdict — real tables are never badly wrong — the trade is not
worth taking blind. Recorded as the lever to reach for *if* a future
instrument class is ever found where the table is off by more than a σ.

## Honest limits

1. **The table was varied; the guitars were not.** All 300 clips are
   GuitarSet. This measures the channel's sensitivity to table error, which
   is the right proxy and the one the caveat asked for, but a genuinely
   different guitar changes timbre, mic, and room as well as `B0`. This
   bounds one axis of portability, not all of them.
2. **The upper end of the tolerance band is unbounded by this run.** The
   curve is still rising at +0.60, so "how far positive is too far" is
   unanswered. It does not affect the verdict — real sets do not get there —
   but the band is a lower bound, not an interval.
3. **The derived sets rest on a wound model fitted to four points** (the
   shipped table's own wound strings, packing 0.860, core = 0.00725 +
   0.2052·gauge). The two `core:*` arms exist precisely as the hedge against
   that model being wrong, and they are the arms that move the result most.
4. **No cross-domain leg.** N5 changes no shipped code and proposes no
   default, so the GAPS gate is not triggered; but that also means this is
   GuitarSet evidence about a GuitarSet-derived concern.
5. Q6's self-calibration study already closed the obvious fallback:
   self-blind abstains and self-pooled is slightly negative
   (`q6_self_calibration_2026-07-22.md`). **The reference table is
   load-bearing** — there is no plan B if the table is wrong, which is what
   makes this robustness result decisive rather than merely reassuring.

## Recommendation

The stated caveat on the Q6 default-on decision — *portability argued from
physics, not measured* — is **discharged for the table axis**. Within the
full range of table error a real steel-string acoustic can produce (gauge,
scale length, and a ±10% wound-core error), the channel never becomes
harmful and in most of that range performs within noise of shipped.

**Promotion into the `auto` path remains the user's call** (SPEC §0.8, loop
STOP condition). Nothing in this iteration changes a default, registers an
artifact, or touches shipped library code; the study lives entirely in
`scripts/eval/` and `tests/unit/`.

If the answer is default-on, the follow-on with the clearest value is **N4**
— per-rig calibration of the level term, now with a measured payoff attached
(the offset curve says a correct level is worth up to +0.016 Tab F1 over the
specification-derived table) and a sharper target (the four wound strings).

## Reproduce

```bash
python -u scripts/eval/n5_table_mismatch.py \
  --json docs/EVAL_REPORTS/n5_table_mismatch_2026-07-24.json
```

7.8 min on laptop CPU for 300 clips × 17 arms, `$0`. Measurements are banked
per clip under `$TABVISION_DATA_ROOT/models/n5_fit_cache`, so re-runs with
new arms cost seconds. Tests: `pytest tests/unit/test_n5_table_mismatch.py`
(10 tests).
