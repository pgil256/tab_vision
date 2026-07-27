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

> **STATUS: measurement in progress.** Method, environment, and deviations below
> are final; result tables are filled from the frozen runs and the A5 decision is
> recorded only once both arms complete.

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

## 6. Results

_Pending — filled from the frozen runs._

## 7. A5 decision

_Pending._ The pre-registered tree (plan §4.5), applied to the WS1 leading
indicator against the 0.574 baseline:

| outcome | action |
|---|---|
| < 0.60 | bank the negative; resolution/conf/crop is not the wall; the geometry line stops here |
| 0.60–0.65 **and** zero-fret share < 30% | one further lever (conf 0.05 floor or per-clip multi-scale), then a final call |
| ≥ 0.65 | proceed to WS5 gate re-derivation on clean-12, then one confirmation run |
| any gated clean-12 regression | the coverage gate stays as-is regardless of the indicator |

Source-disjoint-10 is not touched until clean-12 passes; test-22 only for a single
final frozen confirmation.

## 8. Reproduce

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
