# Q6 portability — a specification-derived table matches the fitted one

Accuracy-loop iteration 10 (ROI deep-dive §4.1). Self-calibration from
unlabelled audio failed (previous report), leaving the channel dependent on a
stiffness table fitted from GuitarSet's guitars. Two fixes were built:
**derive the table from string physics instead of data**, and **a guided
calibration take** that measures an instrument directly.

The physics route resolves the portability problem outright.

## The physics

Stiff-string theory plus ideal tension give

```
B = pi^3 * E * d_core^4 / (64 * T * L^2),   T = 4 * mu * L^2 * f^2
=> B = pi^3 * E * d_core^4 / (256 * mu * L^4 * f^2)
```

`E` is the core's Young modulus, `d_core` the core wire, `mu` the *total*
linear density including winding, `L` the scale length, `f` the open pitch.
All are published or measurable — nothing is fitted.

Two things fall out. Fretting gives `L_n = L·2^(-n/12)` and
`f_n = f·2^(n/12)`, so `B_n = B0·2^(4n/12)/2^(2n/12) = B0·2^(n/6)` — **the
fret law the model already used is derived, not assumed.** And winding raises
`mu` without touching `d_core`, so wound strings are much less inharmonic
than plain ones at the same pitch, which is what makes the low strings
separable.

## Result — it works with no dataset at all

20-clip bank, weight 1.0, r² ≥ 0.50, LOPO position prior.

| arm | ΔTab F1 [lo-95, hi-95] | solo Δ | requires |
|---|---|---|---|
| baseline | — | — | — |
| `lopo` | +0.0525 [+0.0208, +0.0888] | +0.1050 | labelled reference guitars |
| **`physics`** | **+0.0502 [+0.0198, +0.0853]** | +0.1003 | **published specs only** |
| **`physics+offset`** | **+0.0581 [+0.0203, +0.1052]** | **+0.1161** | specs + one scalar |
| `self-seeded` | +0.0388 [+0.0107, +0.0720] | +0.0776 | a reference table |
| `self-blind` | +0.0000 | +0.0000 | nothing (abstains) |
| `self-pooled` | −0.0029 [−0.0088, +0.0000] | −0.0058 | nothing (no help) |

**The specification-derived table is statistically indistinguishable from the
one fitted to GuitarSet**, and with a single global offset it is slightly
better. The dataset dependence is gone: GuitarSet is now a test of the table
rather than its source, which is exactly the inversion that makes this usable
on an unseen instrument.

## Why the level error does not matter but the shape would

Compared against the fitted table, the physics table is systematically **low
by 0.566 log units** (a 0.57× factor) with a residual spread of **0.249 log
units (1.28×)** after that shared offset is removed.

The level error is harmless because it is *shared*: the decision compares
candidate positions **for the same note**, and a common factor shifts every
candidate equally. Only the shape — the relative spacing between strings —
can flip a decision, and 1.28× of residual sits comfortably inside the
1.59–1.78× separation the candidates actually have. That is why raw physics
scores +0.0502 despite being wrong about the absolute level.

The residual splits informatively by construction:

| strings | type | residual vs fitted |
|---|---|---|
| 0–3 | wound | −0.53, −0.81, −0.60, −0.71 |
| 4–5 | plain | −0.15, −0.20 |

**Plain strings agree well; wound strings do not.** For a plain string
`d_core` *is* the gauge and is known exactly. For a wound string the core is
manufacturer-specific and often unpublished, and `B ∝ d_core⁴`, so a 10% core
error is a 46% error in `B`. The wound-string disagreement is a
specification-data gap, not a failure of the physics — and it is fixable once
per string set, not per instrument.

## The calibration take

`calibrate_from_ritual` is implemented for instruments that deviate from
standard specs. It takes the guided 18-note form — three frets on each of six
strings — rather than six open strings, because that makes the **fret
exponent measurable** instead of assumed: each string contributes a
least-squares slope of `log B` against fret, the shared exponent is their
median, and each string's `B0` is its intercept under that exponent. Labels
are certain, since the app asked for the string and fret, so none of the
+0.2975 bootstrap bias that sank self-calibration applies.

`StringStiffnessModel` now carries `fret_exponent` for this; the physics
table sets it to the derived 1.0.

**Not yet validated end to end.** GuitarSet cannot validate the take: usable
isolated open-string notes number 1–3 per player across all six strings, so
the dataset essentially never contains the ritual. Validation needs a real
recording, and using one for validation would make it an eval artifact under
the SPEC private-recordings ban — a deliberate exception or public audio is
required. The unit tests cover the fit's mathematics (exact `B0` and exponent
recovery, including a deliberately non-ideal 1.35 exponent), not its
behaviour on real plucks.

## Honest limits

- Still the 20-clip pilot. Full-dev OOF, GAPS clean-12 no-regression and
  player-05 have not run.
- The default spec set is typical light-gauge phosphor bronze; wound-core
  diameters are approximations, which the residual table above quantifies.
- `physics+offset` uses an offset derived from GuitarSet. A session could
  learn that scalar itself, but that has not been demonstrated — the safe
  reading is raw `physics` at **+0.0502**, which needs nothing.
- Scale length is assumed 25.4"; a materially different instrument should
  pass its own, which is a single documented parameter.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q6_physics_table.py \
  --json ../docs/EVAL_REPORTS/q6_physics_table_2026-07-22.json
python scripts/eval/q6_self_calibration.py \
  --json ../docs/EVAL_REPORTS/q6_physics_arms_2026-07-22.json
```
