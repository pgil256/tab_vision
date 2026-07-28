# Phase E2 — learned fret keypoints vs `calibrate.py`'s consensus fit

**Date:** 2026-07-28 · **Verdict: the pre-registered go bar FAILS.**
**But the intended niche shows a real, mechanism-consistent signal, and a
separate actionable finding fell out: on wire-sparse clips the *current* OBB
calibration is net-harmful.**

Reproduce:

```bash
cd tabvision
python -m scripts.eval.e2_keypoint_cache --det-conf 0.10
python -m scripts.eval.e2_fret_registration_ab
```

---

## 1. What was tested, and what was held fixed

The E2 go bar (design 2026-07-27 §8) is *"keypoint-derived fret registration
beats `calibrate.py`'s consensus fit on wire-sparse clips"*. Phase A established
that this reconstruction is the binding constraint on the whole video channel,
so this is the question that decides whether E2 is worth anything.

The comparison is a deliberate **one-variable swap**. Both arms share:

- the **same cached homography**, used as-is (the keypoint arm never re-fits it);
- the **same** rule-of-18 fit (`fit_fret_map`), nut anchoring, canonical-x
  window and `_MIN_WIRES` floor — literally the same functions;
- the **same frames** — the keypoint model was run over exactly the frame
  indices already in the Phase A 720p crop cache;
- the **same detection floor**, 0.10. This was a correction: the first build
  used 0.25 while Phase A's OBB fret pass ran at 0.10, which handicapped the
  keypoint arm. The cache filename now encodes the floor so the two can never be
  silently confused.

Only the **source of the fret-wire positions** differs: YOLO-OBB box centres
versus the pose model's predicted wire↔string intersections (centroid of the six
visible keypoints).

Model: `yolo11n-pose` fine-tuned on `s-workspace-y3mjn/guitar-fret-6pt`
(CC BY 4.0), 100 epochs, pose mAP50 0.7399 on its own val split.
Keypoint cache: 8,776 frames, 81.5% carrying ≥1 fret keypoint set.

## 2. Pre-registered reading of the bar

Fixed in the script docstring **before** the numbers were seen:

1. wire-sparse = clips where the `obb` arm's calibration fires on **< 0.50** of
   usable frames. A threshold, not a quantile, so it cannot be tuned to the
   result.
2. E2 passes iff `keypoint` > `obb` on ambiguous-note string accuracy over that
   subset **and** does not regress on clean-12 overall.
3. Beating `uniform` is necessary but not sufficient — that only repeats WS1.

Primary metric is the banked WS1/Phase A leading indicator (ambiguous-note
string accuracy, best orientation), *not* registration fit rate — Phase A's
lesson is that fit rate can move while the downstream metric does not.

## 3. Results

Pooled over clean-12, micro over 8,539 ambiguous notes:

| arm | str acc | correct/total |
|---|---:|---|
| `uniform` (no calibration, control) | 0.5365 | 4581/8539 |
| `obb` (current default) | **0.7195** | 6144/8539 |
| `keypoint` (E2) | 0.6305 | 5384/8539 |

Wire-sparse subset — the 4 clips where `obb` fires on < 0.50 of frames
(`063_bV1wc`, `118_VD1wc`, `179_pM1wc`, `212_y41wc`; 1,987 notes):

| arm | str acc | correct/total |
|---|---:|---|
| `uniform` | 0.6915 | 1374/1987 |
| `obb` | 0.6603 | 1312/1987 |
| `keypoint` | **0.7222** | 1435/1987 |

Per clip (`fire` = share of usable frames where that arm produced a fret map):

| clip | amb | uniform | obb | kpt | fire obb | fire kpt |
|---|---:|---:|---:|---:|---:|---:|
| 027_Zpswc | 1449 | 0.438 | **0.774** | 0.511 | 0.882 | 0.641 |
| 031_vpswc | 734 | 0.429 | **0.766** | 0.677 | 0.903 | 0.841 |
| 043_bc1wc | 937 | 0.697 | **0.755** | 0.701 | 0.851 | 0.078 |
| 063_bV1wc | 448 | 0.569 | 0.569 | 0.569 | 0.000 | 0.000 |
| 104_xf1wc | 385 | 0.514 | **0.852** | 0.670 | 0.944 | 0.562 |
| 118_VD1wc | 768 | **0.895** | 0.766 | **0.895** | 0.280 | 0.000 |
| 142_GD1wc | 613 | 0.465 | **0.850** | 0.628 | 0.834 | 0.694 |
| 179_pM1wc | 477 | 0.595 | 0.660 | **0.704** | 0.311 | 0.796 |
| 212_y41wc | 294 | 0.503 | 0.524 | **0.534** | 0.157 | 0.356 |
| 235_Ny1wc | 1482 | 0.372 | **0.591** | 0.559 | 0.714 | 0.751 |
| 294_BSswc | 423 | 0.487 | **0.771** | 0.537 | 0.929 | 0.580 |
| 341_1M1wc | 529 | 0.690 | **0.737** | 0.673 | 0.524 | 0.409 |

## 4. Verdict

**FAIL.** Clause 2 of the bar is decisively unmet: the keypoint arm is **0.089
worse overall** (0.6305 vs 0.7195). Learned keypoints cannot replace the OBB
consensus fit.

Clause 1 *is* met (0.7222 vs 0.6603), and unlike the pre-dedupe run this is now a
real effect rather than an artifact — see §5. But the bar is a conjunction, and
a channel that loses 0.089 overall is not deployable.

## 5. A harness asymmetry found and corrected mid-run — both numbers reported

The first run scored keypoint at **0.5745** pooled / **0.6985** wire-sparse. A
mechanism diagnostic showed the keypoint arm's *detection* was fine — comparable
wire counts to OBB (e.g. `027`: 19.9 vs 18.7 per frame), nearly all inside the
canonical window — while `fit_fret_map` rejected them far more often.

The cause was an asymmetry in **this harness**, not in the model: Phase A's
crop-then-detect pass merges detections "by center distance < half the local
fret pitch" before they reach `calibrate_fret_xs`, so the OBB arm arrives
deduped. The keypoint arm had no such step. Measured: median *minimum* adjacent
canonical gap 0.003 (keypoint) vs 0.024 (OBB), with a median of one clustered
pair per frame versus zero. Near-duplicate wires break `fit_fret_map`'s
geometric-sequence consensus.

Adding an equivalent dedupe moved keypoint from 0.5745 → **0.6305** pooled and
0.6985 → **0.7222** wire-sparse. This is recorded explicitly because the fix
landed *after* a FAIL was seen: it is justified by the mechanism diagnostic and
by symmetry with what the OBB arm already gets, not by the metric moving. The
verdict is unchanged either way. **Both numbers are reported; the pre-dedupe run
is the one to cite if the fairness argument is rejected.**

The pre-dedupe wire-sparse "win" was in fact an artifact: on `118_VD1wc` the
keypoint arm never fired, so it simply inherited `uniform`'s 0.895. Post-dedupe
the subset gain is carried by `179_pM1wc`, where the keypoint model genuinely
fires more than OBB (0.796 vs 0.311) and beats both it and the control
(0.704 vs 0.660 vs 0.595) — mechanism-consistent, and the behaviour E2 predicted.

## 6. The separate finding worth acting on

> ⚠️ **Tested same day and REFUTED — read this before acting on §6.**
> `docs/EVAL_REPORTS/wire_sparse_calibration_gate_2026-07-28.md`.
> The lever proposed below does not work: leave-one-clip-out gating scores
> −0.0043 against the ungated default. The reason is that **sparseness does not
> cause the harm** — calibration is net-positive even in the low-fire half
> (+0.072 note-weighted), and exactly one clip of twelve has a negative delta,
> `118_VD1wc` at −0.129. The subset average below was dragged negative by that
> single clip. The statement "on wire-sparse clips calibration is net-harmful"
> is accurate about *this subset* but must not be read as a threshold effect.

**On wire-sparse clips the current OBB calibration is net-harmful:** 0.6603
against the uncalibrated control's 0.6915, a **−0.031** loss on 1,987 notes.
`118_VD1wc` shows it starkly — 0.766 calibrated vs **0.895** uncalibrated, a
0.129 loss, on a clip where calibration fires on only 28% of frames.

The reading: when wire evidence is thin, the maps `calibrate_fret_xs` does
produce are bad often enough to cost more than they gain, and the existing
per-frame `None` fallback does not catch it because the bad maps still pass the
consensus check. This is independent of E2 and suggests a cheap lever — gate
calibration on a per-clip fit-rate or fit-quality floor, falling back to uniform
below it. It is **not** tested here and would need its own pre-registration.

Note this is also a partial re-reading of Phase A: Phase A measured calibration
as a large net positive (+0.151 on the leading indicator), and that stands
pooled. The harm is confined to the sparse tail.

## 7. Limits — what this does not establish

- **Dev set only.** clean-12 is development data, looked at repeatedly. Nothing
  here touched source-disjoint-10 or test-22.
- **The wire-sparse subset is 4 clips, and effectively 1–2 contribute.** `063`
  and `118` have the keypoint arm firing at 0.000, so they contribute only
  `uniform`. The subset conclusion rests largely on `179_pM1wc`. This is a weak
  basis for anything beyond "worth another look".
- **One model, one seed, one configuration.** `yolo11n` (nano), 100 epochs,
  imgsz 640, a single 926-image dataset. A larger backbone or more data could
  move it; that is untested, and this report is not evidence that it would not.
- **The model fails outright on some clips.** `118_VD1wc` yields ~0.02 fret
  instances per frame and `043_bc1wc` only 3.16 (below the 4-wire minimum). Why
  is not diagnosed here.
- **Best-orientation convention**, matching Phase A's leading indicator; the
  deployable auto-orientation figure is lower for every arm.
- Registration was never compared against **ground-truth fret pixel positions** —
  no such annotation exists for GAPS. Everything is measured through the
  downstream string-accuracy proxy.

## 8. Recommendation

Do not pursue keypoint registration as a replacement for the OBB consensus fit
on this evidence. The honest banked result is: **learned fret keypoints lose to
the current geometry by 0.089 on clean-12, and the one place they help is too
thin a subset to build on.**

The wire-sparse finding in §6 is the cheaper and better-supported lever, and it
needs no model at all.
