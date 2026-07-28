# Phase A — 720p / conf-0.10 / crop-then-detect cache rebuild

Video evidence roadmap Phase A (`docs/plans/2026-07-27-video-evidence-roadmap-design.md`),
approved 2026-07-27. Claim under test: **the geometry chain's string evidence is
detection-limited, not logic-limited** — the chunk-6 WS3 analysis found ~68% of
ambiguous notes sit on clips where the full-frame YOLO pass finds ~0 fret OBBs at
360p/conf 0.25, so the WS1 rule-of-18 lever cannot fire there.

Primary decision variable: the **WS1 leading indicator** (ambiguous-note string
accuracy, clean-12, best fixed orientation per clip). Banked baselines: uniform
**0.544**, WS1 calibrated **0.574** (chunk-5 `_diag` 0.543). Audio playability
prior comparator: **0.778**.

> **STATUS: complete (2026-07-27).** Headline: the fret-detection wall is
> essentially eliminated (WS3 **0.650 → 0.081**) and ambiguous-note string
> accuracy rises **0.568 → 0.720 (+0.151)** — but **Tab F1 does not move**
> (gated 0.8147 → 0.8147; ungated −0.2006). Phase A is banked as a *channel*
> improvement, not a Tab F1 improvement. See §8 for the A5 decision and the
> orientation-selection lever this measurement uncovered.

## 1. What changed (code)

| Change | Where |
|---|---|
| `--max-height` on the GAPS video acquirer; format-18 preference demoted above 360p so the cap is honoured | `scripts/acquire/gaps_video.py` |
| Crop-then-detect: second fret/nut YOLO pass on the upscaled neck crop, merged back to full-frame coords | `scripts/eval/v1_1_gaps_video_chain_probe.py` (`--crop-detect`, `--crop-conf`, `--crop-pad`, `--crop-min-long-edge`) |
| Numpy-only crop geometry + merge/dedupe (`obb_corner_bounds`, `crop_rect_for_neck`, `map_crop_detection`, `merge_crop_predictions`), conf-and-variant-keyed cache paths | `scripts/eval/gaps_cv_cache.py` |
| `--cache-suffix` so the diagnostic can read a crop-pass cache | `scripts/eval/v1_1_gaps_string_diag.py` |
| Fret-wall diagnostic (WS3 statistic + mean homography confidence) from any rich cache | `scripts/eval/phasea_fret_wall.py` (new) |
| Crop-pass overlay renderer (the F2b guard rail) | `scripts/viz/overlay_crop_detect.py` (new) |
| 15 unit tests: OBB corner bounds incl. rotation, crop rect padding/clipping, crop↔full round trip, anisotropic mapping, dedupe, merge semantics, cache-path suffixes | `tests/unit/test_crop_detect.py` (new) |

Design invariants held: the **neck** OBB is taken from the full-frame pass only
(Phase A drops the fret/nut floor, not the neck's); crop caches carry a `.crop`
filename suffix so they cannot shadow full-frame caches; the shipping pipeline
(`tabvision/pipeline.py`, `tabvision/video/**`) is **untouched** — this is an
eval-side probe change only.

## 2. Environment reconstruction (deviation from prior runs)

This machine had **no GAPS data and no chunk-5/6 caches** — prior video work ran
elsewhere. The data root was rebuilt from scratch, which is a material difference
from "re-running the banked probe" and is recorded here in full:

- **Annotations:** the Zenodo record (`10.5281/zenodo.13962272`,
  `gaps_v1_no_audio.zip`) ships musicxml/midi/syncpoints under *different stem
  names* than the repo's `scan_gaps` layout expects (e.g. `-D1wc.xml`). The
  clean-12 `musicxml/`, `midi/`, `syncpoints/`, `audio/`, and
  `gaps_metadata_with_splits.csv` were therefore taken from the HF mirror
  `xavriley/GAPS`, which matches the documented layout exactly.
- **Licensing:** GAPS is CC-BY-NC-SA-4.0, acceptable under the 2026-07-20
  personal-use posture; media is cached under `~/.tabvision/` and **never
  committed**, per LICENSES.md:72.

**Alignment reproduces the banked table exactly.** Per-clip audio↔video offsets
came out **+0.010 to +0.050 s** with cross-correlation peak ratios **2.32–11.24**,
against the banked chunk-5 "+0.01 to +0.05 s, ratios 2.3–11.2" — all sub-frame,
confirming this is the same media the prior work used.

**Gold-note counts match on 11 of 12 clips.** `118_VD1wc` parses to **788** notes
vs the banked **678**. Cause identified, not assumed: with
`TABVISION_GAPS_NO_UNFOLD=1` it returns **exactly 678**, so the delta is entirely
the A6 repeat-unfold that landed after the WS0 table was banked. Both arms of this
A/B use the same current default, so the comparison is internally consistent; the
banked aggregate is therefore *not* bit-comparable and is quoted as context.

**`212_y41wc`** 403'd on its first 360p attempt (transient — format 18 is listed
for that video) and succeeded on retry. Both caches hold all 12 clips.

**The 360p arm is a re-derived control, not a bit-exact replay of the banked
cache.** On `027_Zpswc` the frame-level evidence count reproduces the banked WS0
table *exactly* (`haveCV` = 1450/1450), but the accuracies differ slightly:
uniform **0.422** vs banked 0.428, WS1-calibrated **0.446** vs banked 0.424.
Cause ruled in by elimination, not assumed:

- the calibration path is **unchanged since the banked run** — the last commits
  touching `video/fretboard/calibrate.py` and `video/hand/fingertip_to_fret.py`
  are the WS1/WS2 commits that produced those very numbers, and this branch's
  only diagnostic change (`--cache-suffix`) defaults to prior behaviour;
- gold parsing for `027` matches exactly (1607 notes), so the A6 unfold is not
  involved for this clip;
- `haveCV` matching exactly while per-note predictions differ means the same
  frames were sampled and a small number of *detections* differ.

The residual explanation is the media itself: YouTube re-encodes, so a clip
pulled today at format 18 need not be byte-identical to one pulled in June. The
WS1 path is more sensitive to this than the uniform path because
`calibrate_fret_xs` succeeds or fails on wire-count thresholds, so a handful of
detection differences flip whole frames between the nonlinear map and the uniform
fallback.

**Consequence for this report:** the *within-report* A/B is the valid comparison —
both arms were downloaded the same day with the same tool and scored with the
same code. The banked 0.544/0.574 figures are quoted as **context, not as a
bit-exact target**, and the 360p arm measured here is the control the A5 decision
is taken against.

## 3. Delivered resolutions

The `--max-height` fix is load-bearing: without it the format-18-first preference
would have silently returned 360p for every clip.

| clip | 360p cache | 720p cache | clip | 360p cache | 720p cache |
|---|---|---|---|---|---|
| 027_Zpswc | 640×360 | **1280×720** | 142_GD1wc | 640×360 | **1280×720** |
| 031_vpswc | 640×360 | **1280×720** | 179_pM1wc | 640×360 | **1280×720** |
| 043_bc1wc | 640×360 | **1280×720** | 212_y41wc | 480×360 | 640×480 ⚠️ source cap |
| 063_bV1wc | 640×360 | **1280×720** | 235_Ny1wc | 636×360 | 848×480 ⚠️ source cap |
| 104_xf1wc | 640×360 | **1280×720** | 294_BSswc | 640×360 | **1280×720** |
| 118_VD1wc | 640×360 | **1280×720** | 341_1M1wc | 640×360 | **1280×720** |

10/12 clips gained a true 2× linear (4× pixel-area) upgrade. Two are capped by
the source: `212_y41wc` (640×480 max) and `235_Ny1wc` (848×480 max). Frame rates
are unchanged between caches.

## 4. Protocol

Both arms run at **`--vote-frames 1`** — the leading indicator is defined at one
frame per onset (the diagnostic's default, as banked in chunk-5/6), and holding
it identical on both sides keeps the A/B apples-to-apples while cutting the CV
cost ~3× versus the probe's 3-frame default. No pre-registered threshold changes.

| | baseline arm | Phase A arm |
|---|---|---|
| media | `~/.tabvision/cache/gaps_video` (360p) | `~/.tabvision/cache/gaps_video_720` |
| YOLO conf (full frame) | 0.25 | 0.25 |
| crop pass | — | conf **0.10**, pad 12%, crop long edge ≥ 1280 px, `INTER_CUBIC` |
| rich cache | `gaps_video_chain/{stem}.rawcv.c0.25.pkl` | `gaps_video_chain_720/{stem}.rawcv.c0.25.crop.pkl` |

Measured per arm: (1) fret-wall statistic + mean homography confidence,
(2) leading indicator uniform and WS1-calibrated, (3) gated + ungated Tab F1 on
gold audio with the 12/12 no-regression check.

## 5. Guard rail: what the crop pass actually detects

Run **before** any cache build, per the plan (the F2b lesson: the largest video
error on record was a coordinate bug that a render would have caught).
`scripts/viz/overlay_crop_detect.py`, frame 900, 720p media:

| clip | full-frame frets | crop-added | merged | homography conf |
|---|---:|---:|---:|---|
| 031_vpswc | 16 | +8 | 23 | 0.884 → 0.884 |
| 063_bV1wc | 0 | +0 | 0 | 0.000 → 0.000 (no neck at this frame; crop pass correctly cannot run) |
| 235_Ny1wc | **0** | **+11** | 11 | 0.770 → **0.820** |

Visual inspection is mixed, and this is the honest read:

- On **031** (a clip where the detector already fires) the crop pass adds
  **genuine fret wires near the body joint** — the tightly-spaced high frets the
  full-frame pass misses — which is exactly the intended mechanism. It also adds
  a few false positives on the soundhole and one off the neck entirely.
- On **235** (a documented zero-fret-wall clip) the crop pass turns 0 detections
  into 11, but the neck box is poorly aligned on that frame and the additions
  cluster on the body/strings rather than on wires. Most are probably false.

Two consequences follow, and both are measured rather than assumed:

1. **`calibrate.py`'s RANSAC is the designed backstop** (≥4 wires, ≥50% inliers,
   RMS ≤ 30% of spacing): a garbage wire set should be *rejected*, falling back to
   the uniform partition — i.e. pre-Phase-A behaviour. The leading indicator is
   what reveals whether that holds in aggregate.
2. **False frets inflate confidence.** `predictions_to_homography` adds +0.05 when
   ≥4 frets are present, which is exactly the 0.770 → 0.820 shift on 235. That
   confidence feeds `vision_weight` in `playability.emission_cost` **and** the
   clip-coverage gate, so a detection that is wrong can still increase video's
   influence. The fret-wall diagnostic was extended to report mean homography
   confidence per arm for this reason.

## 6. Mechanism — measured directly, and a pre-registered prediction

The causal claim is that the crop pass helps by letting `calibrate_fret_xs`
actually **fit**, rather than silently falling back to the uniform partition.
"Frames with ≥4 wires" is only the *precondition* for that; the RANSAC consensus
fit can still reject them on inlier count or RMS. The diagnostic therefore calls
`calibrate_fret_xs` per cached frame and reports the true **fit rate**:

| clip | arm | ≥4 wires | **fit rate** | leading indicator |
|---|---|---:|---:|---:|
| 027_Zpswc | 360p | 46.3% | **36.1%** | 0.446 |
| 027_Zpswc | 720p-crop | 100.0% | **88.2%** | **0.774** |
| 031_vpswc | 360p | 66.0% | **58.0%** | 0.618 |
| 031_vpswc | 720p-crop | 100.0% | **90.3%** | **0.766** |

Two things follow. **The dose–response holds across arms and across clips**: at
360p, `031` had both the higher fit rate (58.0% vs 36.1%) and the higher accuracy
(0.618 vs 0.446); at 720p both converge to ~90% fit and ~0.77 accuracy. **And the
RANSAC backstop is working** — the fit rate sits strictly below the ≥4-wire share
in every cell (88.2% vs 100.0% on `027`), so ~10% of frames carry enough wires
yet still fail the consensus test and correctly revert to the uniform fallback.
That is the designed defence against the conf-0.10 false positives seen in §5,
observed operating rather than assumed.

**Pre-registered prediction (recorded before the remaining ten clips were
measured):** if this mechanism is the whole story, then per-clip Δ in the leading
indicator should track per-clip Δ in fit rate, and clips already near-saturated at
360p should gain little. The six documented zero-fret clips (`043`, `063`, `118`,
`179`, `235`, `294`) have the most headroom and should gain most — *unless* their
crop-pass detections are false wires, in which case fit rate rises while accuracy
does not. **A rise in fit rate without a rise in accuracy on those clips is the
falsifier**, and would mean the crop pass is manufacturing plausible-looking
geometry rather than finding real frets.

## 7. Results

### 7.1 The 360p control reproduces the banked baseline (complete, 12/12 clips)

| quantity | this control | banked | Δ |
|---|---:|---:|---:|
| leading indicator, uniform | **0.543** (4492/8266) | 0.544 | −0.001 |
| leading indicator, WS1 calibrated | **0.568** (4696/8266) | 0.574 | −0.006 |
| WS3 fret wall (ambiguous notes on zero-median-fret clips) | **0.650** (6639/10213) | ~0.68 | −0.03 |

The control is faithful in aggregate, which is what the A5 decision rests on.
Eight of twelve clips have a zero *median* fret count at 360p, confirming the
wall is a property of this corpus and not an artefact of the re-download.

**But per-clip variance is substantial, and that matters for reading per-clip
deltas.** Against the banked WS1 column: `063` +0.083, `118` +0.037, `104`
+0.028, `027` +0.022 on one side; `294` −0.137 and `212` −0.266 on the other.
`212` also shows a large evidence-count change (`haveCV` 304 here vs 79 banked)
— unsurprising, since it is the clip that 403'd and was re-fetched. These swings
largely cancel, which is why the aggregate lands within 0.006.

The interpretation is the media-drift finding of §2 operating at clip scale:
YouTube re-encodes, a handful of detections flip, and `calibrate_fret_xs`
amplifies that because it succeeds or fails on wire-count thresholds. **Both arms
of this report were fetched the same day with the same tool, so within-report
deltas remain valid; comparisons of a single clip against the 2026-06 banked
column are not reliable at better than roughly ±0.1.**

### 7.2 The detection wall is essentially eliminated

| statistic | 360p control | 720p + crop |
|---|---:|---:|
| **WS3 — ambiguous notes on zero-median-fret clips** | **0.650** (6639/10213) | **0.081** (824/10213) |
| clips with zero median fret detections | **8 / 12** | **1 / 12** (`063_bV1wc` only) |

Per-clip, the precondition moves almost everywhere. Frames carrying ≥4 wires
(`calibrate_fret_xs`'s minimum) go 0.0% → 100.0% on `043`, `179`, `294`;
1.3% → 98.6% on `235`; 16.3% → 100.0% on `142`; 5.7% → 94.2% on `341`. Two clips
resist: `118` reaches only 35.9% and `212` only 35.6%, and `063` does not move at
all (0.0% → 0.0%).

### 7.3 Leading indicator — the A5 decision variable

| | 360p control | 720p + crop | Δ |
|---|---:|---:|---:|
| uniform partition | 0.543 (4492/8266) | 0.536 (4575/8539) | **−0.007** |
| **WS1 calibrated** | **0.568** (4696/8266) | **0.720** (6146/8539) | **+0.151** |

| clip | haveCV | 360p WS1 | 720p-crop WS1 | Δ |
|---|---:|---:|---:|---:|
| 142_GD1wc | 608 | 0.503 | **0.850** | +0.347 |
| 027_Zpswc | 1450 | 0.446 | **0.774** | +0.329 |
| 294_BSswc | 423 | 0.447 | **0.771** | +0.324 |
| 235_Ny1wc | 1482 | 0.384 | **0.591** | +0.207 |
| 104_xf1wc | 377 | 0.647 | **0.852** | +0.205 |
| 031_vpswc | 753 | 0.618 | **0.766** | +0.148 |
| 179_pM1wc | 477 | 0.587 | **0.660** | +0.073 |
| 043_bc1wc | 870 | 0.699 | **0.755** | +0.056 |
| 341_1M1wc | 527 | 0.710 | **0.737** | +0.028 |
| 212_y41wc | 304 | 0.569 | 0.524 | **−0.045** |
| 063_bV1wc | 229 | 0.616 | 0.569 | **−0.047** |
| 118_VD1wc | 766 | 0.915 | 0.766 | **−0.150** |

**Nine clips gain, three regress.**

**The uniform row is the load-bearing control.** At −0.007 it is flat-to-slightly-
negative: quadrupling the pixel count buys *nothing* on its own, because the
uniform partition is the wrong model at any resolution. The entire +0.151 comes
from the calibrated map finally being able to fit. That is the §6 mechanism
confirmed at full scale, not inferred from two favourable clips.

The pre-registered prediction was **partly falsified, in exactly one place.**
`118_VD1wc` is the falsifier case: fit rate rose 0% → 35.9% and accuracy *fell*
0.915 → 0.766. Partial wire evidence produced a confidently wrong nonlinear map
on the clip that had the most to lose, and a wrong map is worse than the uniform
default it displaced. `212` shows the same signature more weakly (35.6% fit,
−0.045). `063` is the complementary case — the crop pass cannot reach it at all
(homography confidence 0.406, the lowest in the set: the *neck* is barely
detected, so there is no crop region to search) and accuracy is unchanged within
noise. **The failure mode is therefore identified and bounded: partial fret
evidence, not false evidence in general.**

### 7.4 Gated Tab F1 — unchanged, because the coverage gate blocks everything

Gold audio, clean-12, `--vote-frames 1`:

| condition | 360p control | 720p + crop |
|---|---:|---:|
| audio-only | 0.8147 | 0.8147 |
| + real video (auto orientation) | **0.8147** | **0.8147** |
| + oracle strings | 0.9728 | 0.9728 |
| per-clip lift | +0.0000 (12/12) | +0.0000 (12/12) |
| clip coverage | 0.48 – 0.52 | 0.48 – 0.52 |

**No regression on any clip in either arm** — the no-regression invariant holds
12/12. But no gain either: measured coverage sits at **0.48–0.52 against the 0.71
`min_clip_coverage` gate**, so the video channel is gated out on every clip in
both arms, exactly as chunk-6 recorded. Phase A improved the *evidence*; the
*gate* prevents that evidence from reaching fusion at all.

This is also a strong control-fidelity check: the 360p arm reproduces the banked
gated numbers essentially exactly — **0.8147 vs banked 0.8148** audio-only, and
**0.9728 vs banked 0.9726** oracle.

### 7.4b Ungated A/B — the gate is *not* the only obstacle

Removing the coverage gate entirely (`--no-gate`, 720p + crop, gold audio):

| condition | mean Tab F1 | vs audio-only |
|---|---:|---:|
| audio-only | 0.8147 | — |
| + real video, **auto** orientation | **0.6142** | **−0.2006** |
| + real video, **best fixed** orientation (diagnostic ceiling) | **0.7635** | **−0.0512** |
| + oracle strings | 0.9728 | +0.1581 |

No-regression is **VIOLATED on 10/12 clips**, worst `294_BSswc` −0.6624.

**This is the finding that matters most, and it tempers the headline.** Even at
the *best-orientation diagnostic ceiling* — which uses gold to pick the flip and
is therefore unavailable in practice — ungated video still scores **0.7635 vs
0.8147 audio-only**. The improved evidence does not convert.

The reason is visible in §7.5: the video channel is now *near-peer* with the
audio playability prior (0.720 vs 0.778) but still **below** it. Applying a
0.720-accurate string prior in place of a 0.778-accurate one costs more than it
gains, at every clip where it is applied. The coverage gate is not an
inconvenience blocking a win — it is what has been protecting Tab F1 from a
channel that is still second-best.

Compare against the banked chunk-6 sweep, which ran the same conditions on the
360p evidence: no-gate auto **0.6597** (banked) vs **0.6142** here, and no-gate
oracle-orientation **0.7632** (banked) vs best-orientation **0.7635** here. **The
lagging indicator did not move at all**, despite the leading indicator gaining
+0.151. That is the A14 complementarity problem restated: improving video's
marginal accuracy does not help when its errors remain co-located with the notes
audio already gets right.

**Orientation selection accounts for the auto-vs-ceiling gap — but fixing it
would not make ungated video a win.** Auto-orientation scores 0.6142 against a
best-fixed-orientation ceiling of 0.7635, a 0.149 Tab F1 gap; on `294` the auto
path picks `none` (0.2658) where `flip-both` scores 0.8080. A dedicated
diagnostic (`scripts/eval/phasea_orientation_diag.py`) then tested *why*, and the
result corrects the first reading:

| measure | value |
|---|---:|
| clips where the selector agrees with the gold-best orientation | **5 / 12** |
| **mean ambiguous-note string accuracy lost to the choice** | **0.031** |
| median relative spread across the four scores | 0.545 |

The initial hypothesis — that `candidate_support` is near-invariant under the
string-axis mirror, so the four scores tie and `max` falls back to
`ORIENTATIONS` order — is **wrong for most clips**. The scores are
well-separated (relative spread 0.21–1.00 on ten of twelve); only `063` (0.002)
and `212` (0.051) are genuine near-ties. The selector is making a *confident*
wrong call, not an arbitrary one.

And the cost is smaller than the Tab F1 gap suggests: orientation costs only
**0.031** of ambiguous-note string accuracy. The 0.149 appears in Tab F1 because
the ungated path applies the video posterior to *every* note, so a systematically
mirrored posterior damages notes audio would otherwise have decoded correctly —
the damage is amplified by ungated application, not caused by orientation alone.

**The decisive number: even with a perfect, gold-chosen orientation, ungated
video scores 0.7635 against audio-only 0.8147.** Orientation selection is
therefore *necessary but not sufficient* — fixing it reduces the harm from
−0.201 to −0.051, and does not turn the channel into a net contributor.

⚠️ **Consequently the §7.3 leading indicator is a best-orientation figure.** The
diagnostic selects the orientation that maximises gold string accuracy, so
**0.720 is not directly deployable**; the auto-orientation equivalent is
≈ **0.689**. The banked 0.574 baseline is computed the same way, so the +0.151
comparison stands — but the deployable channel quality is the lower number, and
the gap to the 0.778 audio prior is correspondingly wider (≈0.089, not 0.058).

### 7.5 Where the channel now stands

| comparator | value |
|---|---:|
| audio playability prior (the thing to beat) | 0.778 |
| **video, 720p + crop, WS1 calibrated** | **0.720** |
| video, 360p WS1 (previous state) | 0.568 |
| chunk-6 "competitive" target | 0.75 |

Video string resolution was **0.210 behind** the audio prior; it is now **0.058
behind**. This does not overturn the chunk-6 capstone ordering — audio still
resolves strings better than video — but it converts a decisively worse channel
into a near-peer one, which is the precondition for it contributing anything in
fusion. It remains just short of the 0.75 bar.

## 8. A5 decision

The pre-registered tree (plan §4.5), against the 0.574 baseline:

| outcome | action | fired? |
|---|---|---|
| < 0.60 | bank the negative; the geometry line stops here | no |
| 0.60–0.65 **and** zero-fret share < 30% | one further lever, then a final call | no |
| **≥ 0.65** | **proceed to WS5 gate re-derivation on clean-12, then one confirmation run** | **YES — 0.720** |
| any gated clean-12 regression | the coverage gate stays as-is regardless | no — 12/12 hold |

**Formally: the ≥ 0.65 branch fires and no gated regression occurred.**

### 8.1 What is banked

**Phase A is a real and large improvement to the video channel, and it is not a
Tab F1 improvement.** Both halves of that sentence are load-bearing:

- **Banked positive.** The fret-detection wall is essentially gone (WS3 0.650 →
  0.081; zero-median-fret clips 8/12 → 1/12), and ambiguous-note string accuracy
  rises **0.568 → 0.720 (+0.151)**, 9 clips gaining and 3 regressing. The
  mechanism is measured, not inferred: calibration fit rate rises in lockstep,
  and the uniform-partition control is flat at −0.007, so resolution alone
  contributes nothing.
- **Banked negative.** Gated Tab F1 is **unchanged** (0.8147 → 0.8147, +0.0000 on
  12/12), and ungated Tab F1 is **worse** (0.6142, −0.2006; violated on 10/12),
  with even the best-orientation ceiling at 0.7635 still below audio-only 0.8147.
  Against the banked chunk-6 no-gate sweep the lagging indicator did not move
  (oracle-orientation 0.7632 banked vs best-orientation 0.7635 here).

### 8.2 WS5 is authorized but its premise has changed — do not simply loosen the gate

The tree authorizes WS5. The ungated measurement, which the tree did not
anticipate, shows **what WS5 must not be**: lowering `min_clip_coverage` to admit
this evidence would import a −0.05 to −0.20 Tab F1 loss. The gate is not
withholding a win; it is protecting Tab F1 from a channel that, at 0.720, is
still below the 0.778 audio playability prior it would displace.

WS5 should therefore be re-scoped from *"re-derive the coverage threshold"* to
*"admit video only where it is expected to beat audio"* — and note that
confidence-keyed routing of the **360p-era** evidence is a recorded do-not-retry
(A14), so any such attempt must be justified by the new evidence quality and
pre-registered afresh rather than treated as a reopened lever.

### 8.3 Orientation selection — real, but smaller than it first looked

The ungated table made orientation look like the dominant lever (0.6142 auto vs
0.7635 ceiling). The follow-up diagnostic (§7.4b) qualifies that: the selector is
wrong on 7/12 clips but costs only **0.031** of ambiguous-note string accuracy,
and its scores are well-separated rather than tied — it is confidently wrong, not
arbitrary. The 0.149 shows up in Tab F1 because ungated application multiplies a
mirrored posterior across every note.

**The number that settles it: with a perfect gold-chosen orientation, ungated
video still scores 0.7635 vs audio-only 0.8147.** Orientation is necessary but
not sufficient; fixing it reduces harm from −0.201 to −0.051 without producing a
gain. It is therefore worth doing, but it is **not** a route to a Tab F1 win on
its own, and it should not be sold as one.

**Recommended order:** (1) fix orientation selection — cheap, offline, and it
removes a confound from every later measurement; (2) impose a minimum-support
condition on applying the calibrated map, targeting the `118`-class failure
(35.9% fit, −0.150); (3) only then consider selective admission, freshly
pre-registered given A14's do-not-retry. Source-disjoint-10 stays untouched until
clean-12 shows a **Tab F1** gain, not merely an evidence gain.

**The honest strategic read:** Phase A moved the channel from *decisively worse
than audio* to *close to audio* (deployable ≈0.689 vs 0.778), and removed the
detection wall that was the stated blocker. It did not move Tab F1, and no
combination of gate or orientation work will, while the channel remains below the
prior it displaces. Closing that last ≈0.089 — via learned fret keypoints
(Phase E2) or the partial-evidence fix — is the precondition for any Tab F1 gain,
and that is where effort should go next.

## 9. Reproduce

```bash
cd tabvision
# 360p baseline arm
python -m scripts.eval.v1_1_gaps_video_chain_probe --audio-source gold \
    --clips clean12 --vote-frames 1 --output /tmp/probe_360_gold.md
python -m scripts.eval.v1_1_gaps_string_diag --calibrate
python -m scripts.eval.phasea_fret_wall

# Phase A 720p crop-then-detect arm
python -m scripts.acquire.gaps_video --download --clips clean12 --max-height 720 \
    --cache-dir ~/.tabvision/cache/gaps_video_720 --offsets
python -m scripts.eval.v1_1_gaps_video_chain_probe --audio-source gold \
    --clips clean12 --vote-frames 1 --crop-detect \
    --video-cache ~/.tabvision/cache/gaps_video_720 \
    --cache-dir ~/.tabvision/cache/gaps_video_chain_720 \
    --output /tmp/probe_720_gold.md
python -m scripts.eval.v1_1_gaps_string_diag --calibrate \
    --video-cache ~/.tabvision/cache/gaps_video_720 \
    --cache-dir ~/.tabvision/cache/gaps_video_chain_720 --cache-suffix .crop
python -m scripts.eval.phasea_fret_wall \
    --cache-dir ~/.tabvision/cache/gaps_video_chain_720 --cache-suffix .crop
```
