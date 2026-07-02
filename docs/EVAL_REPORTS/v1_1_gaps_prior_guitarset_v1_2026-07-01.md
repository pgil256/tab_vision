# GAPS test-22: guitarset-v1 prior — measured NEGATIVE (−0.138 Tab F1 vs no-prior baseline)

Roadmap item **A2** (2026-07-01 day-one path): the first-ever measurement of the
`guitarset-v1` pitch-position prior on GAPS. Every prior GAPS eval ran
`--position-prior none`; on GuitarSet the prior is worth +22–29pp Tab F1, and
`wrong_position` is 34.1% of real GAPS loss — so a lift was plausible. **It is
not: the prior actively regresses GAPS.**

## Headline: delta vs the 2026-06-18 no-prior baseline

Same manifest, splits, backend, bootstrap settings, tolerance; only
`--position-prior` differs
(baseline: `v1_1_gaps_chunk5_audio_only_2026-06-18.md`).

| Metric (GAPS test-22, single-line) | none (baseline) | guitarset-v1 | Δ |
|---|---:|---:|---:|
| Tab F1 mean | 0.6468 | **0.5087** | **−0.1381** |
| Tab F1 lower-95 | 0.5734 | 0.4549 | −0.1185 |
| Chord-instance acc mean | 0.6633 | 0.5125 | −0.1508 |
| Chord-instance acc lower-95 | 0.6038 | 0.4749 | −0.1289 |
| Onset F1 | 0.8277 | 0.8277 | 0 |
| Pitch F1 | 0.8185 | 0.8185 | 0 |

The lower-95 with the prior (0.4549) sits below the baseline's (0.5734) with no
overlap ambiguity: this is a real regression, not bootstrap noise. (It also
drops the tier from comfortably passing the 0.45 target to skimming it.)

## Decomposition delta: a pure correct ↔ wrong-string exchange

Six-bucket decomposition vs `v1_1_gaps_chunk5_audio_only_decomp_2026-06-18.md`:

| Bucket | none (baseline) | guitarset-v1 | Δ |
|---|---:|---:|---:|
| correct | 10467 | 8336 | **−2131** |
| wrong_position_same_pitch | 2978 | 5109 | **+2131** |
| pitch_off | 488 | 488 | 0 |
| timing_only | 58 | 58 | 0 |
| missed_onset | 708 | 708 | 0 |
| extra_detection | 4500 | 4500 | 0 |

Every non-string bucket is bit-identical (the prior only reweights
string/fret choice among same-pitch candidates — onset/pitch F1 unchanged),
so this is a perfectly controlled measurement of the prior's string
decisions: on GAPS it net-flips **2,131 notes from the right string to the
wrong one** (~16% of all pitch-matched notes).

## Per-tier results (this run)

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 22 | 14699 | 0.5087 | 0.4549 | 0.45 | pass | 0.8277 | 0.8185 |

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 22 | 0.5125 | 0.4749 |

## Interpretation

The `guitarset-v1` prior encodes GuitarSet's position conventions (open-position
bias from pop/comping players on steel-string acoustic). GAPS is in-the-wild
classical repertoire, routinely played in higher positions — so the
cross-domain prior doesn't merely fail to help, it **overrides decode decisions
that the playability model was already getting right**. This is the same
domain-sensitivity lesson as the electric tier (0.12 cross-domain) applied to
the fusion side: positional priors transfer badly across position-convention
domains.

Note this does **not** contradict shipping `guitarset-v1` as the CLI/production
default (A1, commit `8570130`): the v1 acceptance targets and user base are
GuitarSet-domain (home acoustic recording), where the prior is +22–29pp. It
does mean the default is **domain-sensitive**: classical/GAPS-style material
currently decodes better with `--position-prior none` — worth a line in user
docs if classical support ever becomes a target.

## Conclusion / decision-tree branch

Per the A2 branch logic (2026-07-01 roadmap day-one path): **wash/negative →
A7 (GAPS-native prior) is marked SKIPPED in the roadmap.** Banked negative;
no code or default changed by this measurement.

Honest caveat for any future revisit (recorded, not actioned): this negative
is specifically a *cross-domain transfer* failure — it does not by itself
falsify an *in-domain* (GAPS-native) prior, whose mechanism-level premise
(wrong_position is the dominant fixable bucket) still holds. Reopening A7
would need its own justification against this banked result plus the A6
gold-coverage fix landing first; per the recorded branch logic it is out of
the roadmap as of 2026-07-02.

## Methodology

- Manifest: `data\eval\gaps.toml`, `--splits test` (22 clips)
- Audio backend: `highres`
- Position prior: `guitarset-v1` (baseline: `none`)
- Harness: `scripts.eval.v1_1_second_corpus_probe` (cached/resumable composite
  wrapper; scoring identical to `scripts.eval.composite_eval`)
- Eval-harness SHA: `8570130`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5
- Companion decomposition: `v1_1_gaps_prior_guitarset_v1_2026-07-01_decomp.md`
