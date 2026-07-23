# N3 entry probe — physics is a strong wrong-position *doubt* signal

Accuracy-loop N3 (re-scope the assisted review ranker onto the Q6 physics
channel, proposed in the program summary). The Phase 6 ranker flags likely
wrong-position notes for a human (AUC 0.7127 over all ambiguous notes, 38.76%
wrong-reduction @60 s). N3 asks whether the physics channel's per-note signals
improve that. **Probe-before-build: the signal is real and strong, so the
build is justified — but this measures the signal, it does not yet beat
38.76%.**

35,227 ambiguous dev-OOF notes (base wrong rate 0.3445, matching Phase 6's
0.3452). Physics measured from GuitarSet audio, partial-aware isolation.

## Result

Physics fires on **27.4%** of ambiguous notes. As a wrong-position detector:

| detector | AUC (fired subset) | AUC (isolated only, n=4,121) |
|---|---:|---:|
| decoder margin (a Phase 6 feature) | 0.5897 | 0.5540 |
| **physics prob of decoder's string** | **0.6964** | **0.7515** |
| naive z-blend | 0.6863 | 0.6901 |

The physics score is the probability the physics posterior assigns to the
string the decoder chose; **low means physics doubts the decoder**. As a
standalone flag it reaches **AUC 0.75 on isolated notes** — far above the
decoder margin's 0.55 — and 0.70 across all fired notes.

Corroborating the direction, split by agreement:

- P(wrong | physics disagrees with decoder) = **0.3999** (n=7,197)
- P(wrong | physics agrees) = **0.3182** (n=2,461)
- base rate = 0.3445

A physics-vs-decoder contradiction raises the wrong-probability; agreement
lowers it.

## The counterintuitive part, resolved

Physics's *hard* string accuracy on these notes is only **0.29** (0.35
isolated). That looks alarming next to Q6's 0.92 — until you separate the two
quantities:

- Q6's 0.92 was over **all** notes, dominated by unambiguous pitches with one
  candidate. These are the **ambiguous** 2-3-candidate notes, the hard core.
- The physics posterior uses a deliberately wide σ = 0.35, so its *argmax* is
  a soft, noisy classifier — but its *calibration as a doubt signal* is
  strong. Physics is much better at "the decoder's choice is unlikely" than at
  "here is the right string", which is exactly what a review flag needs.

So the hard-accuracy number is not a bug and not the relevant metric; the AUC
of the continuous doubt score is.

## Honest limits — why this is an entry probe, not a win

- **Coverage is 27.4%.** On the other ~73% of ambiguous notes physics
  abstains and adds nothing, so any ranker gain is diluted to the covered
  fraction.
- **The comparison is against decoder margin, not the full Phase 6 ranker.**
  Phase 6 reaches AUC 0.7127 using nine features; this probe shows physics
  beats *one* of them (margin) by a wide margin on its covered subset, and is
  complementary (it measures the audio, not the decode). It does **not** show
  physics beats the full nine-feature MLP — that requires adding the feature
  and retraining.
- **The naive z-blend underperforms physics-alone on isolated notes** (0.6901
  vs 0.7515), which is informative: the two signals need a *learned* combiner,
  not a hand blend — precisely what the Phase 6 MLP provides. So the
  integration path is clean and the blend number here understates it.

## Verdict and next slice

**Entry gate PASS: physics_prob_decoder is a strong, complementary
wrong-position signal (AUC 0.75 isolated / 0.70 fired vs decoder margin's
0.55-0.59), available on 27% of ambiguous notes.**

The build slice, deferred under the one-slice timebox:

1. Add `physics_prob_decoder`, `r2`, and a fired/abstained indicator to the
   Phase 6 feature set (`string_assignment_phase6`).
2. Retrain the calibrated MLP with player-held folds (its existing protocol).
3. Re-run the offline correction replay and compare **wrong-reduction @60 s**
   against the shipped **38.76%** — the actual N3 ship metric. Reported
   separately from automatic Tab F1.

This iteration establishes the signal exists and is worth the feature; the
replay is the next iteration.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n3_physics_review_probe.py \
  --lattice ../docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv \
  --json ../docs/EVAL_REPORTS/n3_physics_review_probe_2026-07-23.json
```

Labels + decoder features from the banked lattice (git-ignored, 70 MB — pass
`--lattice`); physics measured from cached events + GuitarSet audio. ~2 min.
