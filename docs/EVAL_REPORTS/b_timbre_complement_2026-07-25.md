# Track B — a timbral model is not the missing piece, and my mechanism was wrong

**Date:** 2026-07-25
**Scripts:** `tabvision/scripts/eval/b_timbre_complement.py`,
`tabvision/scripts/eval/b_timbre_separability.py`
**Data:** `b_timbre_complement_2026-07-25.json`,
`b_timbre_separability_2026-07-25.json`
**Population:** 300 development clips, leave-one-player-out throughout.
**The sealed player was not opened. Nothing was trained beyond per-fold linear
separators, and no artifact is registered.**

## Verdict

**Do not build `guitarset-timbre-v1`.** Track B closes, making it the third
independent closure of the timbral path — but the first with a measured account
of *why*, and that account contradicts the one this probe set out to confirm.

The short version: on the population where the physics channel abstains, the
oracle headroom is **large** (+0.1934), the timbral signal is **present**
(pairwise AUC ≈ 0.71), and yet two prior model-based probes converting that
signal into Tab F1 got **+0.0072** and **−0.0218**. The bottleneck is neither
ceiling nor signal. It is that a ~0.71 pairwise separation is not enough to
improve a position prior already at ~0.65 top-1 on the same notes.

## What was already closed, and why this is not a re-litigation

- **Phase 2 (2026-07-14):** a 35,905-parameter audio ranker over 35,959 OOF
  events scored **0.6331** against a prior-only **0.6548** — worse. A
  feature-only variant scored 0.6027. Calibration was healthy (ECE 0.0597, all
  six strings active), so the finding was explicitly *lack of transferable
  timbral lift*, not posterior collapse.
- **Phase 4 (2026-07-16):** native 44.1 kHz descriptors — multi-resolution
  harmonic envelopes through Nyquist, pick noise, centroid, rolloff, decay,
  **inharmonicity**, raw spectral slopes — over 56,742 adjacent-string pairs.
  Position + audio reached 0.6621 vs a 0.6548 comparator: **+0.0072
  [−0.0152, +0.0291]** against a +0.05 gate. It closed the path and stated: do
  not enlarge the window or model, do not tune on the failure set, do not open
  player 05.

Neither asked the question the physics channel created. Both measured timbral
lift over *all* ambiguous notes; neither measured it on the ~77% where physics
now abstains. That is the only place a timbral model could still have a niche,
and it is what this track tested. **No model was enlarged and no forbidden
retrain was run.**

## Where a timbral model would have to work

Ambiguous notes, split by whether the shipped physics channel applies:

| Population | n | share | median duration | mean concurrency | short (<150 ms) | masked (3+) |
|---|---:|---:|---:|---:|---:|---:|
| covered | 11,528 | 23.4% | 0.297 s | 2.07 | 0.0% | 35.7% |
| **abstain** | **37,740** | **76.6%** | 0.305 s | **3.72** | **17.2%** | **66.3%** |

The abstain population is not shorter in the median — it is **denser**: 1.8×
the concurrency and 1.9× the masking rate. Physics abstains under simultaneity,
not brevity (the 17.2% short share is the secondary cause, from the minimum
window the fit requires).

## The oracle ceiling — large, so it decides nothing

Giving the fusion the **gold string** for a chosen population is the absolute
upper bound for any per-note string evidence there. Paired against shipped:

| Arm | Tab F1 | Δ vs shipped | 95% CI |
|---|---:|---:|---|
| baseline (no channel) | 0.6083 | −0.0718 | `[−0.0885, −0.0558]` |
| shipped | 0.6801 | — | — |
| **oracle on abstain** | **0.8735** | **+0.1934** | `[+0.1768, +0.2101]` |
| oracle on covered | 0.7855 | +0.1055 | `[+0.0952, +0.1163]` |
| oracle on all | 0.9049 | +0.2249 | `[+0.2083, +0.2416]` |

This was pre-declared as the closing test: *a small ceiling closes Track B; a
large ceiling does not open it.* The ceiling is large. So the oracle does not
close the track and the question moves to whether the signal is reachable.

Worth noting in passing: **the oracle on the covered population is still worth
+0.1055.** The physics channel is far from perfect even where it applies, which
is Track A's territory, not Track B's.

## The hypothesis I set out to confirm — and it is wrong

The mechanistic story was: physics abstains when partials are unreadable; the
abstain notes are 66% masked; masking should destroy timbral information too;
therefore the complement is selected *against* every spectral method and no
timbral model can work there however built. That would have explained both prior
closures at once.

**It does not survive measurement.** Leave-one-player-out pairwise separability
(same pitch, different string, Fisher direction on plain spectral features):

| Population | string pairs | notes | mean AUC | median AUC |
|---|---:|---:|---:|---:|
| covered | 64 | 11,020 | 0.6633 | 0.7112 |
| **abstain** | 63 | 33,346 | **0.7060** | 0.7157 |

Separability on the abstain population is **not degraded — it is slightly
better** than on the covered one. Masking does not destroy timbral string
identity. The complement is not adversarially selected, and the tidy mechanism
is refuted by its own probe.

## What the numbers actually say

Putting the three together:

- ceiling on the complement: **large** (+0.1934)
- signal on the complement: **present** (AUC ≈ 0.71, well above chance)
- realised Tab F1 from model-based extraction: **+0.0072 and −0.0218**

So the constraint is the **conversion**, not the ingredients. A pairwise AUC of
0.71 means substantially overlapping distributions — it is a real but weak
separation, and it has to improve a position prior that already reaches ~0.65
top-1 on the very same notes. Weak evidence competing against a strong prior on
a 6-way decision is exactly the regime where added channels wash out, and
"healthy calibration, no lift" (Phase 2's finding) is its signature.

This also explains why the *physics* channel succeeded where timbral models
failed, despite inharmonicity being one of Phase 4's features. Physics is not a
better-fitted version of the same thing: it is **specification-derived**, so it
contributes an absolute physical prediction per candidate rather than a
discriminative direction learned from the same distribution the prior already
models. Its per-note evidence is independent of the prior in a way a fitted
timbral direction is not.

## Decision

Close Track B. Do not register `guitarset-timbre-v1`; the slot stays empty and
`--string-evidence auto` continues to resolve to the physics channel or `none`.
No training spend, no new dependency, no artifact.

**If it is ever reopened**, the evidence says the productive question is not
"better timbral features" — Phase 4 already went to Nyquist — but "how would
weak per-note evidence be combined so that it helps rather than washes out".
Track A's finding is the relevant precedent: the same evidence admitted at its
own confidence rather than at full weight changed a regression into a gain.
Applying that shape to a timbral channel is the only version of this idea the
measurements support, and it would need its own gate.

## Limits

- Development only; the sealed player was not opened.
- Separability is measured pairwise per (pitch, string-pair), which is the
  cheapest decisive form but is not the same as full 6-way posterior quality. A
  6-way model could in principle do better than the pairwise numbers suggest —
  though Phase 2 built one and it scored below prior-only.
- The feature set here is deliberately plain. It is not a fair test of "the best
  possible timbral features", and it is not meant to be: Phase 4 already ran the
  rich version. Plain features reaching AUC 0.71 is evidence the signal is
  *accessible*, which is the point being made.
- `pair_auc` originally ranked ties arbitrarily, so an all-tied comparison read
  as a perfect separation rather than chance. Its unit test caught this and it
  was fixed to average ranks. **Re-running produced identical figures to four
  decimals** — exact ties are vanishingly rare in continuous LDA projections, so
  the bug never bit here. It is fixed and pinned anyway, because a statistic
  that cannot represent "no information" is the wrong statistic for a probe
  whose whole job is deciding whether information exists.
