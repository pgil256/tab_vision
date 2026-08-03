# FretCam capo detection: giving the camera the one job audio cannot do

**Date:** 2026-07-26
**Status:** IMPLEMENTED, **reporting only**. Nothing routes on it; no default
changed; no accuracy claim.
**Code:** `fretcam/src/fretcam/capo.py`,
`tabvision/tabvision/preflight/capo.py::detect_capo_from_video`
**Control:** `fretcam/src/fretcam/capo_negative_control.py`
**Data:** `docs/EVAL_REPORTS/fretcam_capo_negative_control_2026-07-26.json`

## Why this, after six negatives

The 2026-07-25 investigation
(`fretcam_contact_evidence_2026-07-25.md`) closed six routes for making FretCam
improve Tab F1: coverage, evidence type, temporal aggregation, the pressing
gate, frame density, and the missed-onset population. The built contact channel
measured **−0.000362** end-to-end. A seventh was tried and closed for this
report: re-centring `transition_cost`'s position-shift term on FretCam's
observed hand motion. On 2,949 covered gold note pairs the correlation between
observed and actual fret motion is **+0.069**, and re-centring makes the mean
residual *worse* (1.952 → 2.051 frets). Contact-derived hand motion does not
track real fret motion at 640×360.

The common thread in all seven is that they asked the camera for **fine spatial
precision, per note, at 640×360** — the regime where it is worst.

Capo detection is the inverse ask, and Q7 already established it is worth more
than everything above combined.

## The case

From `q7_capo_detect_2026-07-23.md`:

- The capo-covariant prior is worth **~+0.37 Tab F1 to a capo user** — two
  orders of magnitude above anything measured for the position channel.
- **Audio provably cannot recover the capo.** A capo at fret `C` playing a shape
  produces exactly the pitch set of capo 0 playing the same music transposed up
  `C`. The note sets are identical. Pitch-based detection measured **1/60**, and
  the negative is theoretical, not empirical — no cleverness fixes it.
- The inharmonicity route could not be tested (the synthetic capo was
  pitch-shifted, which does not shorten a string, so the audio was a physically
  impossible instrument).
- Today the capo **must be typed in by hand**. Q7's disposition was "report the
  bound and ask", because the physical upper bound is sound but weak.

So there is a lever worth +0.37 with **no sensor attached to it**. A camera is
the obvious candidate, and a capo suits it in every way the fretting hand did
not:

| | fretting hand (six negatives) | capo |
|---|---|---|
| size | 7 mm inter-string spacing | a bar across the whole neck |
| duration | changes every note | fixed for the entire session |
| what's needed | per-note precision | one integer per recording |
| integration | fights coverage | *benefits* from long sessions |

The coverage problem that sank every other route — FretCam producing usable
output on 2.6% of notes — is irrelevant here. A static object only has to be
seen in *some* frames, and the estimate integrates over all of them.

## Method

`FrameDetection.fret_ticks` already publishes each fret wire as a segment across
the neck in image coordinates. For each candidate fret the detector samples a
thin band across the strings inside that fret's **cell**, and records its mean
darkness normalized against the neck's own brightness range. Per-frame profiles
accumulate; the session estimate is the fret that is both darkest by a margin
and **persistent**.

**Which cell, precisely.** A capo at fret `N` clamps between wire `N-1` and wire
`N`, pressed against wire `N` — that is what makes the string speak at fret `N`.
So the band runs *back* from wire `N`, not forward from it. The first
implementation sampled forward, which would have reported every capo one fret
low: a plausible-looking wrong answer, which is worse than an abstention. The
convention is now pinned by a test that draws a bar strictly between wires 2 and
3 and requires the answer to be fret 3.

Working off the published ticks means no second homography and no drift from
the geometry the rest of the chain uses.

**Coverage fallback.** `fret_ticks` is emitted only once the *calibrated* fret
map locks, which the first negative-control run showed happening on just 5 of 12
clips — the other 7 abstained blind. `neck_locked` runs far higher (97–98% on
clips where the fret map never locked), so when ticks are absent the detector
derives approximate wires from `neck_quad` plus `body_joint_fret` by rule of 18.
That geometry is coarser than the calibrated map and would be unacceptable for
assigning a note's fret; it is acceptable here because identifying which of
seven frets carries a full-width bar is a far weaker demand, and persistence
absorbs the extra jitter. Calibrated ticks always take precedence.

**Two discriminators, and the second one had to be found the hard way.**

*Persistence.* In a single frame a barre chord and a capo are nearly the same
object; a barre comes and goes and a capo never does. A fret must lead in ≥60%
of observed frames, with a median darkness margin ≥0.06, over ≥12 frames.

*Across-string width.* Persistence turned out to be **insufficient**, and the
negative control is what exposed it — see below. A player who stays in one
position darkens that cell in most frames, which is precisely a capo's temporal
signature. What a hand cannot imitate is width: a capo is a rigid bar clamping
every string, while fingers cover two or three and leave the rest bare. So a
candidate must also be darker than the frame's midpoint across ≥80% of the
string span, or it is discarded regardless of how dark or persistent it is.

Frames where *no* fret spans the strings still count toward the denominator.
They are evidence against a capo, and dropping them would let a handful of lucky
frames carry the persistence fraction.

## Validation, and its honest limits

**There is no capo ground truth in this repository.** GAPS has no capo column,
and **none of its 404 MusicXML scores encode `<capo>`** (checked, not assumed).
Private recordings are barred from eval roles. So the **true-positive rate on
real capos cannot be measured here**, and this report does not claim one.

What can be measured is the thing that actually gates safety for a reporting
feature: **does it invent capos on real footage that has none?**

GAPS clean-12 is solo classical guitar; classical players do not use capos and
none of these scores declare one. Sampled at 2 Hz, capped at 240 frames a clip.
It ran three times, and the middle run is the reason this section is long.

**Run 1 — 0/12, but on only 5 exercised clips.** Seven clips abstained because
the calibrated fret map never locked, so they produced no ticks at all. That is
blindness, not discrimination. It motivated the neck-quad fallback.

**Run 2 — 1/12.** With the fallback, coverage rose from 5 to 11 clips
(`027_Zpswc` alone went 32 → 132 usable frames) and `212_y41wc` reported **fret
1 at confidence 0.607**.

**The obvious explanation was wrong.** The natural hypothesis — fret 1's cell
abuts the nut, which is itself a permanent full-width dark bar — was tested by
dumping per-fret darkness profiles across three clips. It is refuted: fret 1 is
the darkest band on `212` (0.668) but the **lightest** on `027` (0.543) and near
the bottom on `179` (0.314).

What the profiles actually showed is worse. The leading fret is **fret 4 on both
`027` (26/90 frames) and `179` (44/90)** — the middle of the neck, where the
fretting hand spends its time. The confound is not the nut. **It is the hand**,
and it defeats persistence by construction: a player who stays in one position
produces exactly the temporal signature the detector was looking for. On
classical repertoire, which moves constantly, this surfaced as one false
positive; on open-position folk repertoire it would be far worse. 1/12 was a
floor, not a ceiling.

That is why the across-string width gate exists. It was added **after** seeing
the failure, which deserves stating plainly — but it is not a fitted threshold.
It is a physical property (a bar clamps six strings; fingers clamp two or
three), it is validated by a synthetic stationary-hand test that is independent
of this corpus, and no existing threshold was moved to accommodate the result.

**Run 3 — with the width gate:**

**False positives: 0 / 12, with 11 of 12 clips exercised.**

| clip | frames | margin | outcome |
|---|---:|---:|---|
| `179_pM1wc` | 228 | 0.111 | not_persistent |
| `104_xf1wc` | 185 | 0.027 | not_persistent |
| `027_Zpswc` | 132 | 0.108 | not_persistent |
| `294_BSswc` | 119 | 0.083 | not_persistent |
| `235_Ny1wc` | 114 | 0.062 | not_persistent |
| `118_VD1wc` | 97 | 0.164 | not_persistent |
| `031_vpswc` | 93 | 0.062 | not_persistent |
| `142_GD1wc` | 77 | 0.165 | not_persistent |
| `341_1M1wc` | 44 | 0.046 | not_persistent |
| `063_bV1wc` | 30 | 0.005 | not_persistent |
| **`212_y41wc`** | 28 | 0.073 | **not_persistent** (was fret 1 @ 0.607) |
| `043_bc1wc` | 4 | — | insufficient_frames |

Run 3 dominates run 1 rather than merely matching it: the same clean sheet, but
over **11 exercised clips instead of 5**, and the clip that broke run 2 now
abstains. The three runs read as a chain — coverage fix exposed a real failure,
the failure exposed the missing discriminator, and the discriminator holds
without giving the coverage back.

Summary of the safety claim: **0/12 false positives on real capo-free footage,
with the discriminators exercised on 11 clips.** The true-positive rate is still
unmeasured and this is still reporting-only.

**What the width gate costs is unknown.** It trades sensitivity for
specificity, and only one side of that trade is measurable here. A real capo
partly occluded by the thumb, or a slightly misplaced fallback geometry that
puts some of the twelve string samples off the bar, will be rejected. With no
capo footage there is no way to price that. The trade is still the right one for
a feature whose only consumer is a human: a wrong fret number is worse than
silence, because silence is obviously silence and a wrong number looks like an
answer.

Synthetic tests cover the mechanism separately — correct fret localisation at
frets 1–7, an oblique slanted-tick view, abstention on a bare neck, and the two
barre cases (intermittent at one fret, and wandering across frets). Those prove
the detector responds to the right *geometry*; they are not evidence about real
capos, which have shadows, varied colour, and imperfect alignment.

**A real bug the synthetic tests missed.** The first negative-control run
crashed: a sample at x=639.6 passes a `< 640` bounds check and then rounds to
640. Fixed by bounds-checking the rounded index, and pinned by a regression
test. Synthetic fixtures did not generate geometry at the frame edge; real
footage did on its first clip.

## Integration

The chain is deliberately short and ends at a human.

```
FretCam CapoDetector  (same traversal, no extra inference)
  -> VideoObservations.capo
  -> PipelineArtifacts.video_capo
  -> preflight.detect_capo_from_video   (cross-check)
  -> diagnose report: "possibly fret N ... unverified"
```

**The cross-check is the useful part.** Audio cannot locate a capo, but its
physical upper bound *can refute one*: if the recording contains a pitch below
`open_midi[0] + C`, a capo at `C` is impossible. That bound held **60/60** in
Q7. So `detect_capo_from_video` rejects any video estimate above it and reports
`video-refuted-by-bound`. The two sensors are complementary in exactly the right
way — the camera proposes, the physics disposes.

**Nothing routes on the result.** `cfg.capo` is untouched. The diagnose report
says "possibly fret N … unverified; re-run with `--capo N` if that is right".
That is deliberate: with the true-positive rate unmeasured, the only safe
consumer is a human who can look at their own guitar.

## Verification

- FretCam: `273 passed, 1 skipped, 5 subtests` (was 245; +28 capo, covering the
  barre discriminators, the neck-quad fallback, the cell convention, and the
  frame-edge regression).
- TabVision: `1149 passed, 12 skipped` (+8: video estimator, refutation,
  clamping, diagnose wording).
- Ruff check + format and mypy clean across both packages.

## What would raise this from "reporting" to "routing"

One thing: **a handful of real capo recordings with known capo positions.**
Ten clips at capos 0/2/4/5 would give a true-positive rate and a confusion
matrix, which is all that stands between this and auto-setting `cfg.capo` for a
+0.37 lever. That footage does not need to be large, licensed, or annotated
beyond a single integer per clip — which makes it by far the cheapest
outstanding data acquisition in the project.
