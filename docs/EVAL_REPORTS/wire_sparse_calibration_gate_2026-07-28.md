# Wire-sparse calibration gate — the rule does not transfer

**Date:** 2026-07-28 · **Verdict: FAIL. Bank the negative.**
Pre-registration: `docs/plans/2026-07-28-wire-sparse-calibration-gate-preregistration.md`
(committed as `a8f5f2e`, **before** this run).

Leave-one-clip-out gating scores **0.7152** against the ungated default's
**0.7195** — a gain of **−0.0043**, on the wrong side of zero. Per the
pre-registered tree that is "fire rate does not transfer out of sample; bank the
negative. The line stops."

Reproduce:

```bash
cd tabvision
python -m scripts.eval.wire_sparse_gate_ab
```

---

## 1. What was tested

E2's report (§6) observed that on clips where `calibrate_fret_xs` fires rarely,
calibrating is net-harmful. The tempting experiment — gate at the same 0.50
threshold that *defined* that subset, then score the same twelve clips — is
circular: the pooled gain reduces to the already-measured subset difference and
would come out positive even if fire rate carried no information at all.

So the claim actually tested was **generalisation**: does per-clip fire rate
predict, *out of sample*, whether calibrating that clip helps? Leave-one-clip-out
— the threshold applied to each held-out clip fitted only on the other eleven —
answers that and can fail. It did.

## 2. Result

Pooled over 12 clips, 8,539 ambiguous notes:

| arm | str acc | correct/total |
|---|---:|---|
| uniform everywhere | 0.5365 | 4581/8539 |
| **ungated (current default)** | **0.7195** | 6144/8539 |
| LOO-gated | 0.7152 | 6107/8539 |
| | **−0.0043** | |

Per clip, sorted by fire rate (`d` = calibrated − uniform, so positive means
calibration helps):

| clip | n | fire | a_uni | a_cal | d |
|---|---:|---:|---:|---:|---:|
| 063_bV1wc | 448 | 0.000 | 0.569 | 0.569 | +0.000 |
| 212_y41wc | 294 | 0.157 | 0.503 | 0.524 | +0.020 |
| **118_VD1wc** | 768 | 0.280 | **0.895** | 0.766 | **−0.129** |
| 179_pM1wc | 477 | 0.311 | 0.595 | 0.660 | +0.065 |
| 341_1M1wc | 529 | 0.524 | 0.690 | 0.737 | +0.047 |
| 235_Ny1wc | 1482 | 0.714 | 0.372 | 0.591 | +0.219 |
| 142_GD1wc | 613 | 0.834 | 0.465 | 0.850 | +0.385 |
| 043_bc1wc | 937 | 0.851 | 0.697 | 0.755 | +0.058 |
| 027_Zpswc | 1449 | 0.882 | 0.438 | 0.774 | +0.337 |
| 031_vpswc | 734 | 0.903 | 0.429 | 0.766 | +0.337 |
| 294_BSswc | 423 | 0.929 | 0.487 | 0.771 | +0.284 |
| 104_xf1wc | 385 | 0.944 | 0.514 | 0.852 | +0.338 |

## 3. Why it fails — and it is not that the trend is absent

The trend is real and strong: **Spearman(fire, d) = +0.797**, note-weighted mean
`d` of **+0.072** in the low-fire half against **+0.281** in the high-fire half.
Calibration does help more when it fires more.

But that is not the claim a gate needs. A gate needs a fire-rate region where
calibration is *harmful*, and there isn't one — **calibration is net-positive
even in the low-fire half (+0.072)**. Exactly one clip in twelve has `d < 0`:
`118_VD1wc` at −0.129. Everything else is zero or better.

So the E2 §6 observation was an **outlier, not a threshold effect**. The
wire-sparse subset average was dragged negative by a single clip, and averaging
four clips hid that.

The LOO picks show the failure mechanically:

- for held-out `118_VD1wc` (the one harmful clip, fire 0.280) the eleven other
  clips chose **T = 0.15** — which does *not* gate it. The harm survives.
- for held-out `179_pM1wc` (fire 0.311, `d` **+0.065**) they chose **T = 0.50** —
  which *does* gate it, discarding a real benefit.

The rule transfers backwards: it misses the clip it was invented for and
penalises a clip it should have left alone. That is what an outlier masquerading
as a threshold effect looks like.

## 4. Correction to the E2 report

`e2_fret_keypoints_2026-07-28.md` §6 said "on wire-sparse clips the current OBB
calibration is net-harmful". That is accurate as a *description of that subset*,
and its §7 already warned the subset was 4 clips with 1–2 contributing — but the
framing invited the reading that sparseness causes harm. It does not. The
correct statement is: **one clip (`118_VD1wc`) is harmed by calibration, and it
happens to sit in the sparse subset.** The suggested lever built on that reading
is now tested and refuted.

## 5. `118_VD1wc` keeps appearing

Three independent measurements now single it out:

- Phase A's largest per-clip regression (−0.150);
- the only clip here where calibration hurts (−0.129, and uncalibrated it is the
  *best* clip in the set at 0.895);
- the clip where E2's keypoint model detects essentially nothing (~0.02 fret
  instances per frame, gate fires 0.000).

That is a clip-specific pathology worth one look on its own terms — a rendered
overlay would probably explain it in minutes. It is **not** evidence for any
general rule, and nothing here should be built on it.

## 6. What is now closed, and what is not

**Closed:** gating fret calibration on per-clip **fire rate**. Fire rate was
fixed as the statistic in the pre-registration precisely so a failure could be
recorded cleanly rather than converted into a search over statistics.

**Not tested, and needing separate pre-registration if anyone wants them:** mean
homography confidence, inlier counts, fit RMS, per-*frame* gating rather than
per-clip. The pre-registration deliberately excluded these; trying them now
because fire rate failed would be the fishing expedition it was written to
prevent.

**Unchanged:** Phase A's +0.151 pooled gain from calibration stands, and this
result reinforces it — calibration helps on 11 of 12 clips, by up to +0.385.

## 7. Limits

- Twelve clips; LOO on n=12 is high-variance and one clip can carry it — which
  is precisely the finding, so the small n cuts both ways here.
- clean-12 is development data, repeatedly examined.
- Leading indicator only, best-orientation convention. No Tab F1 was measured,
  because a −0.0043 leading-indicator result does not warrant the run.
- source-disjoint-10 was **not** touched: the pre-registered tree routes there
  only on a ≥ 0.010 gain, and neither those videos nor their CV cache exist.
