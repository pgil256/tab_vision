# Pre-registration — gating fret calibration on per-clip fit rate

**Date:** 2026-07-28 · **Status:** pre-registered, **not yet run**.
**This document is committed before any result exists.** Its thresholds and its
falsification conditions are fixed here so they cannot be adjusted afterwards.

Origin: `docs/EVAL_REPORTS/e2_fret_keypoints_2026-07-28.md` §6. While measuring
E2 the `uniform` control arm showed that on clips where `calibrate_fret_xs`
fires rarely, applying it is **net-harmful** — 0.6603 against the uncalibrated
0.6915 over 1,987 notes, and −0.129 on `118_VD1wc` where it fires on 28% of
frames. The lever this suggests needs no model at all.

---

## 1. The circularity trap this design exists to avoid

The obvious experiment is invalid, and it is worth stating plainly so nobody
runs it and believes the answer.

The wire-sparse subset was *defined* as "clips where the OBB arm fires on < 0.50
of frames", and the harm was *observed* inside that subset. If we now gate at
T = 0.50 and evaluate on the same twelve clips, then

```
pooled_gated − pooled_ungated  ==  (uniform − calibrated) summed over the sparse clips
```

which is exactly the quantity already measured as positive. The improvement is
an **arithmetic identity**, not evidence. It would be true even if per-clip fire
rate carried no predictive information whatsoever.

Any result of that form must be rejected, including if a future session produces
one.

## 2. The claim actually under test

> **Per-clip calibration fire rate predicts, out of sample, whether applying
> `calibrate_fret_xs` to that clip helps or hurts.**

That is a claim about generalisation, and it can be false. It is false if the
sparse-clip harm is idiosyncratic to particular clips rather than tracking fire
rate — in which case a threshold fitted on some clips will not transfer to
others.

## 3. Design

### 3.1 Quantities, per clip, on the Phase A 720p crop cache over clean-12

- `f` — fire rate: share of usable frames where `calibrate_fret_xs` returns a map.
- `a_cal` — ambiguous-note string accuracy, calibrated (current default).
- `a_uni` — the same, uncalibrated (uniform partition).
- `d = a_cal − a_uni` — the per-clip benefit of calibrating.
- `n` — ambiguous notes with CV evidence (the denominator).

All are already produced by `scripts/eval/e2_fret_registration_ab.py`; the
`uniform` and `obb` arms are exactly `a_uni` and `a_cal`.

### 3.2 Primary test — leave-one-clip-out

For each clip `i`:

1. choose threshold `T_i` using **only the other 11 clips** — the `T` maximising
   their pooled gated accuracy, searched over `T ∈ {0.05, 0.10, …, 0.95}`;
2. apply `T_i` to clip `i`: if `f_i < T_i` use `a_uni,i`, else `a_cal,i`;
3. accumulate correct counts across all held-out clips.

Compare that pooled LOO-gated accuracy against the ungated default. Clip `i`
never influences the threshold applied to it, so this is not circular.

### 3.3 Mechanism check

Spearman rank correlation between `f` and `d` across the 12 clips, note-weighted
mean of `d` in the low- and high-`f` halves, and the per-clip table. This is
*descriptive*: it explains any effect, it does not establish one.

### 3.4 Confirmation — source-disjoint-10

Only if the primary test passes. Requires acquiring the 10 videos at 720p and
building their rich CV cache (neither exists — verified 2026-07-28), so it is a
separate, more expensive run. **One frozen run, single threshold, no tuning**:
the `T` fitted on all twelve clean-12 clips, applied unchanged. test-22 is not
touched at all.

## 4. Pre-registered decision thresholds

| Outcome (primary, LOO on clean-12) | Action |
|---|---|
| LOO-gated pooled accuracy **> ungated by ≥ 0.010** | Proceed to the source-disjoint-10 confirmation. |
| gain in **(0, 0.010)** | Bank as "directionally right, too small to justify the confirmation spend". No pipeline change. |
| gain **≤ 0** | **Bank the negative.** Fire rate does not transfer; the E2 §6 observation was a description of these twelve clips, not a rule. The line stops. |
| any arm errors or the fire-rate statistic is degenerate (all clips one side of every T) | Stop and report; do not substitute another statistic. |

Promotion to the default pipeline requires the confirmation run **and** the
gated-Tab-F1 no-regression property, and is a separate decision that this
document does not pre-approve.

## 5. Fixed in advance, so they cannot drift

- **Statistic: per-clip fire rate.** Mean homography confidence, inlier counts
  and fit RMS are plausible alternatives and are **not** tested here. Trying
  several and reporting the best would be fishing; if fire rate fails, that is
  the recorded answer for fire rate, and any successor needs its own
  pre-registration.
- **Granularity: per clip**, not per frame. A per-frame gate is a different
  proposal and is out of scope.
- **Metric: ambiguous-note string accuracy, best orientation** — the banked
  WS1/Phase A leading indicator, so numbers are comparable to 0.543 / 0.574 /
  0.7195. Tab F1 is a *gate*, not the primary; the deployable auto-orientation
  figure is lower for every arm, as always.
- **Search grid** for `T`: 0.05 to 0.95 in 0.05 steps. Ties broken toward the
  **larger** `T` (more gating, i.e. the more conservative fallback to uniform).
- **No re-runs.** The LOO number reported is the first one produced.

## 6. Known limitations, stated before the result

- **Twelve clips.** LOO on n=12 is high-variance; a single influential clip can
  carry it. The per-clip table is reported in full so that is visible.
- clean-12 is **development data** and has been looked at repeatedly.
- Gating trades a possible dense-clip loss for a sparse-clip gain; the pooled
  number can therefore genuinely go either way, which is what makes §3.2 a real
  test rather than a restatement.
- This measures the *leading indicator*, not end-to-end Tab F1. Phase A's
  lesson — that the indicator can move while Tab F1 does not — applies with full
  force here and is why promotion is explicitly out of scope.
