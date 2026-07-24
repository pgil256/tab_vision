# Q7 entry probe — capo-covariant prior: mechanism valid, gap quantified

Accuracy-loop iteration 13 (ROI deep-dive §4.3). Probe-before-build entry
gate for capo support: today `resolve_inference_policy` routes any capo>0
session to `priors=none`, discarding the position-prior lift on exactly the
recordings a personal user makes with a capo. §4.3 proposes a
**capo-covariant** prior — shift the fret axis by the capo before applying
it. Does the shift actually recover the lift?

Label-level on GuitarSet dev gold (players 00-04), LOPO priors, no audio, no
pipeline change. A note `(s0, f0, pitch0)` under a capo at `C` is the same
shape `C` frets up, so gold becomes `(s0, f0+C, pitch0+C)` and the
capo-covariant score reads the capo-0 prior `C` lower:
`covariant(s, fret | P, C) = prior_capo0(s, fret-C | P-C)`.

Top-1 assignment accuracy on ambiguous notes (~51,000 per capo):

| capo | covariant | naive (capo-ignorant) | none-lowfret (today) | none-uniform |
|---:|---:|---:|---:|---:|
| 0 | 0.5960 | 0.5960 | 0.4378 | 0.2653 |
| 2 | 0.5960 | 0.5552 | 0.4378 | 0.2732 |
| 4 | 0.5959 | 0.4681 | 0.4377 | 0.2850 |
| 7 | 0.5951 | 0.4366 | 0.4366 | 0.3118 |

## What is proven, and what is not

**The transform is correctly constructed (verdict: PASS).** At capo 0
`covariant == naive` exactly (the shift is a no-op), and across capos
`covariant` is flat at ~0.596. The flatness is **partly by construction**:
shifting both the gold and the prior lookup by `C` maps the capo-`C`
fretboard onto the capo-0 one, so covariant reproduces capo-0 assignment
quality by design (modulo high-fret candidates lost to the 24-fret ceiling).
That is the right thing to confirm — it proves the index arithmetic and the
relative-fretboard equivalence — but it is not independent evidence.

**The current routing gap is real and large.** `none-lowfret` — the
lowest-fret default a capo session falls back to today with priors off — sits
at **0.438**, flat across capo. The capo-covariant prior offers **0.596**. The
**+0.158** between them is what capo sessions are currently leaving on the
table, and it is the same order as the +22 pp Tab-F1 prior lift §4.3 cites
(measured here as position-prior-alone assignment top-1, which is why the
capo-0 anchor is 0.596 rather than Q2's full-decode 0.6548 — no Viterbi,
sequence prior or playability in this probe).

**Ignoring the capo genuinely hurts — this part is not tautological.** The
`naive` arm applies the capo-0 prior at absolute shifted coordinates, i.e.
the prior used without capo awareness, and it degrades monotonically with
capo depth: 0.596 → 0.555 → 0.468 → 0.437. By capo 7 it is no better than no
prior at all. So the shift is necessary, not cosmetic — a capo-covariant
prior is the only way to keep the prior useful past capo ~2.

## The assumption this probe cannot test

Covariance assumes real capo playing follows the **same relative-fret
distribution** as capo-0 playing. GuitarSet has no capo recordings, so this
is applied to relabelled capo-0 gold and cannot be checked here. The claim is
physically reasonable — hand ergonomics are relative to the capo, and the
capo-0 prior is dominated by open-vs-fretted and low-vs-high-position
preferences that a capo shifts wholesale — but it is an assumption, and the
+0.158 is conditional on it.

## Verdict and next slice

Entry gate **PASS**: the lever exists, the transform is correct, and the gap
it would close is large and non-trivially better than the capo-ignorant
alternative. Q7 continues; it is not shipped.

The real accept/reject is the build slice §4.3 describes, deliberately left
for the next iteration under the one-slice timebox:

1. **Preflight detection** — estimate tuning offset (cent histogram of
   detected f0 vs equal temperament) and capo (minimum-fret occupancy +
   open-string pitch classes), warn or auto-set.
2. **Wire the covariant transform** behind an explicit flag, `auto`
   unchanged, widening the gated domain to capo 0-7.
3. **Validate on pitch-shifted audio** — GuitarSet pitch-shifted +2/+4/+7
   semitones with capo-shifted labels, measured as real Tab F1 through
   `fuse()`. This is the test the label-level probe cannot substitute for,
   and it is the one that decides shipping. It re-transcribes audio, so it is
   a multi-hour run of its own.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q7_capo_covariant_probe.py \
  --json ../docs/EVAL_REPORTS/q7_capo_covariant_2026-07-22.json
```
