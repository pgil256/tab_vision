# Q6 generalization — self-calibration fails; the reference table is load-bearing

Accuracy-loop iteration 9 (ROI deep-dive §4.1). The pilot integration lifted
Tab F1 by +0.0525, but its stiffness table was fitted from **other GuitarSet
players' gold labels**. `B0` is a property of the string set and scale
length, so that table is an artefact of one dataset's instruments. The user's
requirement is that this work on *any* acoustic guitar. This asks whether the
reference can be eliminated.

The hypothesis was reasonable: `B ∝ 1/L²` and the scale length is shared by
all six strings, so a different instrument should mostly *shift* the table.
A recording ought to be able to calibrate itself — decode once without the
physics, take the provisional assignments, re-fit `B0` from the recording's
own notes, decode again.

**It does not work.**

## Result

20-clip bank, weight 1.0, r² ≥ 0.50, leave-one-player-out position prior.

| arm | Tab F1 | ΔTab F1 [lo-95, hi-95] | what it requires |
|---|---:|---|---|
| baseline | 0.6773 | — | — |
| `lopo` | 0.7298 | **+0.0525 [+0.0208, +0.0888]** | other guitars' **gold labels** |
| `self-seeded` | 0.7161 | **+0.0388 [+0.0107, +0.0720]** | reference table + session refit |
| `self-blind` (one clip) | 0.6773 | +0.0000 | nothing — abstains on everything |
| `self-pooled` (~2 min, one player) | 0.6744 | −0.0029 [−0.0088, +0.0000] | nothing — and does not help |

Only the two arms that carry a reference table work. Self-calibration from
unlabelled audio contributes nothing at one clip and slightly less than
nothing at two minutes.

## A bug this experiment exposed

The first `self-blind` run scored **−0.0329 [−0.0600, −0.0098]** — a
CI-significant *regression*. The cause was in the evidence channel, not the
calibration: `inharmonicity_matrix` scored a candidate whose string was
missing from the table at probability **zero**. A zero is a hard veto, so a
partially-calibrated table silently forced every note onto whichever strings
happened to have enough data.

Fixed: an uncalibrated candidate string now makes the channel **abstain on
that note** rather than scoring it at zero. That turned the −0.0329
regression into a clean +0.0000, and it is a latent hazard removed from the
shipping path regardless of how calibration is eventually solved — any future
sparse or partial table would have hit it.

## Why self-calibration fails

**Data volume.** The channel needs ~8 well-fitted isolated notes *per string*.
A 30-second clip yields on the order of ten fitted isolated notes across all
six strings, so `self-blind` abstains almost everywhere. Pooling four clips
(~2 minutes of one player) is still not enough to calibrate six strings.

**Bootstrap bias.** Where enough notes do exist, the labels come from the
first decode, which is only ~65% right on exactly the ambiguous notes this is
meant to fix. The measured median `log B0` shift between the self-fitted and
reference tables is **+0.2975** — roughly 35% in `B`, comparable to the
1.6–1.8× separation the whole method depends on. The errors are not random
noise that a median absorbs: the decoder's mistakes correlate with string
identity, so they bias the per-string estimate systematically.

## What this means for "works on any acoustic guitar"

The physics is instrument-general; the *calibration* is not, and it cannot
currently be recovered from unlabelled audio. Two routes remain, neither yet
tested:

1. **A physical reference table instead of a fitted one.** `B0` for a
   standard acoustic set follows from string gauge, core construction and
   scale length — published, measurable quantities. A table derived from
   manufacturer specs rather than fitted to GuitarSet would be genuinely
   instrument-general for standard light/medium sets, and GuitarSet becomes
   a *validation* set rather than the source. `self-seeded` (+0.0388) already
   shows that a reference table plus session refinement retains most of the
   benefit, so this is the highest-value next step.

2. **Anchor on label-free notes.** Pitches playable at exactly one position
   are ground truth with zero label noise — no decoder involved. They cover
   only the extreme strings directly, but combined with a fixed table *shape*
   they would pin the one parameter that actually varies between instruments
   (the shared offset). This is the automatic version of route 1.

3. **A calibration ritual.** Ask the user to play six open strings once.
   Six perfectly-labelled notes, and the instrument is calibrated exactly.
   Trivially reliable, at the cost of ten seconds of setup.

**Untested and important:** whether the GuitarSet-fitted table transfers to a
*different* acoustic guitar at all. There is no second acoustic dataset in
the repo to check it against, so `self-seeded`'s +0.0388 is demonstrated only
on instruments similar to the ones the table came from.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q6_self_calibration.py \
  --json ../docs/EVAL_REPORTS/q6_self_calibration_2026-07-22.json
```
