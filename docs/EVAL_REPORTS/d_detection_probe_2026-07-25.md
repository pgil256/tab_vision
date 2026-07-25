# Track D — detection buckets: `extra_detection` closes, `missed_onset` opens

**Date:** 2026-07-25
**Script:** `tabvision/scripts/eval/d_detection_probe.py`
**Data:** `docs/EVAL_REPORTS/d_detection_probe_2026-07-25.json`
**Population:** 300 development clips, leave-one-player-out priors, current
default configuration. **The sealed player was not opened.** Nothing is tuned.

## Verdict

Phase 0 promoted this track because `missed_onset` + `extra_detection` are 33.3%
of development loss. Decomposing them splits the two cleanly:

- **`extra_detection` — CLOSE as a fix target.** Its apparent dominant mode is a
  base-rate artifact. Correcting for the interval content of the music, spurious
  detections are *less* likely to be a fifth or fourth from a real note than
  chance. Only octaves (2.32×) and unisons (1.53×) are genuinely enriched, and
  together they are ~17% of the bucket.
- **`missed_onset` — OPEN.** Masking is real and large: notes with 3+ neighbours
  sounding are missed at **1.63×** their base rate, and notes sounding **alone
  at 0.45×**. Short notes are missed at **1.61×**. Both survive their base-rate
  comparison.

## `extra_detection` (3,324) — against the interval content of the music

For each spurious detection, the interval to the nearest gold note within 250 ms.
The base column is the same statistic computed over *every* gold note and its
nearest neighbour — i.e. what intervals this music simply contains.

| Class | Observed | Base | Lift |
|---|---:|---:|---:|
| other | 39.4% | 42.9% | 0.92× |
| fifth / fourth | 29.6% | 37.2% | **0.80×** |
| **octave** | 10.2% | 4.4% | **2.32×** |
| semitone | 7.1% | 11.0% | 0.64× |
| **unison** | 6.9% | 4.5% | **1.53×** |
| orphan (no gold within 250 ms) | 5.4% | — | — |
| ring-out (decay of an ended note) | 1.4% | — | — |

**Read the second column before the first.** At face value the raw distribution
looks like a clear harmonic-leakage story: fifths and fourths are the largest
identified class at 29.6%. They are also **37.2% of the intervals the music
contains** — standard tuning and ordinary chord voicings are built from them. So
the model is *under*-represented there relative to chance, not over. Reporting
29.6% without its marginal would have pointed a build at the largest class in
the bucket, which is the one class actively depleted.

What *is* real is small: octaves at 2.32× and unisons at 1.53×, together ~17% of
the bucket, i.e. ~2.5% of total loss. Ring-out is negligible at 1.4%, so offset
handling is not the problem either.

**This reproduces A10's outcome on `pitch_off` almost exactly** — an attractive
harmonic explanation that dissolves once the base rate is included. Two of the
three non-position buckets now have the same verdict for the same reason.

## `missed_onset` (4,371) — against all gold notes

| Property | Observed | Base | Lift |
|---|---:|---:|---:|
| **3+ others sounding at onset** | 49.0% | 30.1% | **1.63×** |
| 1–2 others sounding | 37.5% | 39.7% | 0.94× |
| **sounding alone** | 13.5% | 30.2% | **0.45×** |
| **short (< 150 ms)** | 34.0% | 21.1% | **1.61×** |

Both effects clear their base rates decisively and in the same direction: a note
is missed when it is *buried* or *brief*. A note sounding alone is missed at
**less than half** the base rate — the detector is not failing at recall
generally, it is failing inside dense simultaneity.

Register is reported without a lift column because registers are not equally
populated and the comparison would be meaningless; the raw split (48.9% mid,
26.3% low, 24.8% high) shows no obvious concentration.

## What this justifies

**A masking-aware detection pass is the first Track D build candidate**, and it
is the only one this probe supports. Concretely: notes inside dense
simultaneity, and notes shorter than ~150 ms, are where the recall loss lives.
That is 49% and 34% of the bucket respectively, against a bucket worth 18.9% of
development loss.

**It does not justify anything on `extra_detection`.** Closing that as a fix
target now — before any build — is the whole point of running the probe first,
and follows the A10 precedent exactly.

## Method note

The base-rate columns are the finding. Without them this report would have said
"29.6% of spurious detections are fifths or fourths — harmonic leakage is the
dominant mode", which is precisely backwards. This repo has made the
conditional-without-marginal error before (A14's video complementarity, read at
0.285 without its 0.382 anchor marginal; the F7 re-run had the same shape). The
probe now computes both by construction, and the two are printed side by side so
a reader cannot see one without the other.

Residuals come from the shipped matcher via a new optional out-parameter on
`decompose_errors`, not a reimplementation — a second copy of that matching
would drift from the one that is scored, and the diagnosis would then describe a
pipeline that does not ship.

## Limits

- Development only; the sealed player was not opened.
- "Masking" here is measured against *gold* simultaneity, which is the right
  question for diagnosis but is not available at inference time. A build would
  have to infer density from the signal.
- The 250 ms association window and the 150 ms short-note threshold are
  reasonable but unswept. The masking effect is large enough that it is unlikely
  to be an artifact of either, but that has not been shown.
