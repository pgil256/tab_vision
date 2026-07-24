# N4 ritual validation — the blocker was wrong; the ritual works and is not worth shipping

Accuracy-loop post-program iteration (N4). Two results, and the second is the
useful one:

1. **N4 was not actually blocked.** GuitarSet can supply the calibration
   ritual, on public data, with no download and no exception to the
   private-recordings ban.
2. **The ritual runs, recovers sensible physics — and is dominated by a single
   global constant.** Per-instrument calibration's entire prize is **+0.0027**
   Tab F1 against a well-chosen fixed offset, and that figure is an *oracle*.
   This ritual lands **−0.0095 below** the fixed constant while chasing it.

280 scored GuitarSet dev clips, frozen config (`weight 1.0 / min_r2 0.50 /
sigma 0.35`), LOPO position prior, paired bootstrap N = 10,000 seed 42,
baseline Tab F1 **0.5974**. Player 05 untouched.

## The blocker was measured against the wrong microphone

The state file recorded:

> GuitarSet contains only 1-3 usable isolated open notes per player, so it
> cannot contain the ritual. Needs public capo/calibration audio or an
> explicit exception to the private-recordings ban.

`calibrate_from_ritual` does not need *open* strings or *temporally isolated*
notes. It needs notes whose (string, fret) is **certain** and whose `B` is
measurable. That constraint binds on the mono microphone, where a note must be
isolated in *time* to be measurable at all. GuitarSet also ships
`audio_hex-pickup_debleeded` — one channel per string — where a note is
isolated by *construction*, and the JAMS annotation supplies the label.

Measured, not assumed:

| | ritual observations | strings covered |
|---|---:|---:|
| player 00 | 18 | 6/6 |
| player 01 | 18 | 6/6 |
| player 02 | 15 | 5/6 |
| player 03 | 17 | 6/6 |
| player 04 | 18 | 6/6 |

The ritual asks for 18 (three frets × six strings). Four clips per player
supply it. **N4's blocker is retired.**

**The channel→string mapping was verified rather than assumed.** Fitting every
labelled note at its labelled `f0` in all six channels and taking the best r²
gives a diagonal confusion matrix (77% diagonal overall, 98% and 92% on the
two channels with most notes). The off-diagonal mass sits on *adjacent*
channels — the signature of residual pickup bleed, not a permutation error; a
shifted mapping would put a single off-diagonal near 100%. Because bleed is
real, a note enters the ritual only when its labelled channel is also the
best-fitting of the six.

**This substrate is pessimistic by construction.** A real ritual is 18
deliberate plucks with nothing else sounding; these are performance excerpts
bleeding through an imperfect debleed. Whatever the ritual achieves here is a
lower bound on a guided take.

## Verdict on the pre-declared reading — PARTIAL

> **Partial** — the exponent recovers but the Tab F1 delta over `shipped` has
> a CI spanning zero. The calibration math is sound; the payoff is below noise
> on this substrate.

Both halves hold. The exponent recovers (median 0.859; three of five players
within the pre-declared 0.15 of the theoretical 1.0), and `ritual` versus
`shipped` is **−0.0007 [−0.0081, +0.0061]** — indistinguishable from the
shipped table.

## Every arm, on the same 280 clips

| arm | vs baseline | vs shipped | lo-95 | hi-95 |
|---|---:|---:|---:|---:|
| `shipped` (registered spec table) | +0.0429 | — | — | — |
| `ritual` (per-player, fitted `B0` + exponent) | +0.0421 | **−0.0007** | −0.0081 | +0.0061 |
| `ritual-level` (per-player level only) | +0.0493 | +0.0065 | −0.0020 | +0.0151 |
| `global-level` (+0.780, ritual median) | +0.0514 | +0.0085 | +0.0000 | +0.0170 |
| `offset+0.40` (diagnostic) | +0.0560 | +0.0131 | +0.0071 | +0.0194 |
| **`offset+0.60`** (diagnostic) | **+0.0589** | **+0.0160** | +0.0088 | +0.0233 |

`global-level`'s lower bound is **+0.000041** — nominally above zero, but
sitting exactly on the boundary, and it is a post-hoc arm. Treat it as
suggestive, not established.

**The ordering is the finding: more calibration is worse.** A fixed constant
chosen *without* the ritual beats the ritual's own global constant, which
beats its per-player levels, which beat the full per-player fit.

## Control: N5's offset curve reproduces exactly

N5 measured `offset+0.60` at **+0.0162** over shipped on 300 clips. This run,
on a different 280-clip subset and through a different script, measures
**+0.0160**. Agreement to 0.0002 across two independent studies is the control
that lets the two be read together.

## N5's open limit is now closed

N5 reported: *"the curve is still rising at +0.60, so how far positive is too
far is unanswered; the band is a lower bound, not an interval."*

On one consistent substrate: +0.40 → +0.0131, +0.60 → +0.0160, +0.78 →
+0.0085. **The response turns over between +0.60 and +0.78.** The peak sits
near +0.6, which is where Q6's independently measured −0.566 level residual
said it would be.

## Two findings

### 1. The fret exponent is measurably below 1.0

| player | 00 | 01 | 02 | 03 | 04 |
|---|---:|---:|---:|---:|---:|
| fitted exponent | 0.859 | 0.887 | 0.835 | **0.378** | 0.869 |

Stiff-string theory derives `B(s,n) = B0·2^(n/6)`, i.e. exponent exactly 1.0,
and the shipped table assumes it. **All five players come out below it**, four
of them tightly clustered at 0.835–0.887 (median 0.859). Q6 anticipated
exactly this when it made the exponent fittable:

> a real fret and fingertip terminate the string differently from the nut, so
> a calibration ritual that measures several frets per string can fit `k`
> rather than trust it.

This is the first measurement of that `k` on real plucks. Five of five below
theory is not noise, though the estimate is thin — three frets per string, and
player 03's 0.378 is a broken fit that the median absorbs. It does not by
itself justify changing the shipped exponent: the `ritual` arm, which is the
only one that *uses* the fitted exponent, is the worst-performing arm here.

### 2. Per-player calibration is real but nearly worthless

Per-player deltas against the shipped table, uniform-offset arms only:

| player | shipped | +0.40 | +0.60 | +0.78 | best |
|---|---:|---:|---:|---:|---|
| 00 | 0.0000 | −0.0059 | −0.0120 | −0.0184 | **shipped** |
| 01 | 0.0000 | +0.0123 | +0.0220 | +0.0216 | +0.60 |
| 02 | 0.0000 | +0.0258 | +0.0384 | +0.0266 | +0.60 |
| 03 | 0.0000 | +0.0111 | +0.0098 | −0.0028 | +0.40 |
| 04 | 0.0000 | +0.0221 | +0.0218 | +0.0158 | +0.40 |

Player heterogeneity is genuine — player 00 wants **no** offset while 01/02
want +0.60 — so per-instrument calibration is not a fiction. But the prize is
tiny:

| | Tab F1 vs shipped |
|---|---:|
| **oracle per-player** (best arm per player, chosen on test) | **+0.0187** |
| best fixed constant (+0.60) | +0.0160 |
| ritual's global constant (+0.78) | +0.0085 |
| ritual's per-player levels | +0.0065 |

**Per-player calibration's entire headroom over a fixed constant is +0.0027,
and that is an upper bound** — it picks each player's best arm using the test
data. The ritual, trying to capture it honestly, lands **−0.0095 below** the
fixed constant. The variance it adds is roughly four times the signal it
chases.

Why it overshoots: the ritual's level is the **median across strings**, and
four of six strings are wound with shifts of +0.75…+0.92 against the plain
strings' +0.21/+0.34, so the median is pulled to +0.78 while the
decision-optimal offset is near +0.60. A second candidate cause is that the
ritual measures `B` on the **hex pickup** while scoring measures it on the
**mono microphone** — two transducers with different partial amplitudes, hence
a systematic difference in fitted `B`. A real ritual would use one microphone
for both and would not carry that term. Neither cause is isolated here.

Note the *shape* is not an artifact: wound-string shifts far exceeding plain
matches Q6's mic-side residual split (wound −0.53/−0.81/−0.60/−0.71, plain
−0.15/−0.20) and N5's finding that wound-core geometry dominates. Three
independent routes — Q6's LOPO residual, N5's perturbation sweep, and this
run's direct hex measurement — agree that the specification table
under-predicts `B`, most on the wound strings.

## Honest limits

1. **The oracle is not achievable.** +0.0187 picks each player's best arm on
   the data it is scored on. Any real selector must estimate it, and this
   ritual estimates it worse than a constant.
2. **Cross-transducer calibration.** Ritual on hex, scoring on mono mic. This
   is the substrate's central weakness and it plausibly costs the ritual some
   of its shortfall. It cannot be separated here without a mic-side ritual,
   which GuitarSet cannot supply (that was the original blocker, and for the
   mono mic it stands).
3. **Five instruments.** Player-level heterogeneity is estimated from five
   guitars, so "+0.0027 of headroom" is itself uncertain.
4. **The exponent rests on three frets per string** and one of five fits is
   visibly broken.
5. **No cross-domain leg.** Nothing shipped changed, so no gate triggered —
   but that also means this is GuitarSet evidence.
6. **In-distribution.** The +0.60 optimum is located on GuitarSet dev. It is
   corroborated by Q6's independent −0.566 residual and by this run's direct
   physical measurement, which is far stronger than a swept hyperparameter —
   but it is not a cross-domain result.

## Recommendation

**Do not ship a per-instrument calibration ritual.** It is buildable, it runs,
and the physics it recovers is sensible — but the measurement shows its
ceiling is +0.0027 against a constant, and every honest estimator tried here
lands below the constant rather than above it. N4 closes as a **banked
negative on the shipping question**, with the machinery retained.

**The live lever is the table's level, not per-instrument calibration.** A
uniform +0.60 log-B correction to `acoustic-physics-v1` is worth **+0.0160
[+0.0088, +0.0233]** on top of the shipped table — larger than anything the
ritual produced, and now supported by three independent measurements of the
same underlying error.

That is a change to a registered artifact on the `auto`-decision path, so it
is **a user decision under SPEC §0.8 and the loop's STOP rule, not mine.** It
would also want a cross-domain gate before shipping, which today is satisfied
for classical only by abstention. Nothing in this iteration changes a default,
registers an artifact, or touches shipped library code.

## Reproduce

```bash
python -u scripts/eval/n4_ritual_validation.py \
  --json docs/EVAL_REPORTS/n4_ritual_validation_2026-07-24.json
```

~7 min on laptop CPU, `$0`. Hex measurement is the cost; scoring replays the
banked N5 fits. Tests: `pytest tests/unit/test_n4_ritual_validation.py`
(12 tests).
