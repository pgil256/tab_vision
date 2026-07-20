# S1a — SynthTab-scale count priors: CLOSED (bounded negative, both arms)

**Date:** 2026-07-20 · **Program:** SynthTab-scale (Program S), Phase S1a ·
**Plan:** `docs/plans/2026-07-20-nc-second-opinion-and-synthtab-program.md`

## What was built

`scripts/eval/build_synthtab_v1_prior.py` parsed the audited SynthTab
symbolic slice (S0 report; zip SHA-256 `da678dba…2dc0576d`) into the exact
registered artifact class (`guitarset-v1`/`gaps-v1` hyperparameters:
position α=1.0 power=2.0; sequence `delta_fret` α=0.5 κ=8.0), standard
tuning only, tick→seconds via per-track tempo map + MIDI-header PPQ:

| variant | GM programs | tracks parsed | position events | transition samples |
|---|---|---:|---:|---:|
| acoustic | 24–25 | 9,317 | 9,226,460 | 1,912,867 |
| all | 24–31 | 34,621 | 34,063,065 | 7,910,342 |

For scale: `gaps-v1` (which passed its classical gate the same day) was
built from 171,059 position events — the acoustic variant is **54×** that
substrate; `all` is **199×**.

## Arm 1 — swaps (`s1a_synthtab_prior_probe_2026-07-20.md`)

Oracle-pitch decode, GuitarSet dev players 00–04 (300 clips, player 05
untouched), sequence weight 4.0, paired stratified 10k bootstrap:

| condition | aggregate | Δagg [95% CI] | gate |
|---|---:|---|---|
| baseline (guitarset pair) | 0.6883 | — | — |
| st-acoustic | 0.4942 | −0.1941 [−0.2193, −0.1682] | fail |
| st-all | 0.5472 | −0.1412 [−0.1642, −0.1175] | fail |
| st-acoustic-pos (position only) | 0.5074 | −0.1809 [−0.2057, −0.1557] | fail |
| st-acoustic-seq (sequence only) | 0.6681 | −0.0202 [−0.0292, −0.0116] | fail |

The damage is concentrated in the **position prior** (−0.18); the sequence
swap is mildly negative (−0.02, though comp alone is +0.003).

## Arm 2 — count blends (`s1a_synthtab_blend_probe_2026-07-20.md`)

`build_synthtab_blend_prior.py`: guitarset counts + λ × mass-normalized
synthtab-all counts, λ ∈ {0.25, 1.0}:

| condition | aggregate | Δagg [95% CI] | gate |
|---|---:|---|---|
| blend-l0p25 (pos+seq) | 0.6761 | −0.0123 [−0.0183, −0.0060] | fail |
| blend-l1 (pos+seq) | 0.6351 | −0.0532 [−0.0685, −0.0381] | fail |
| blend-seq-l0p25 (seq only) | 0.6885 | +0.0002 [−0.0020, +0.0025] | fail |
| blend-seq-l1 (seq only) | 0.6848 | −0.0035 [−0.0070, −0.0001] | fail |

Monotone dose-response: every admixture of SynthTab counts is
neutral-at-best. The best cell (+0.0002) is a wash, nowhere near the
CI-lower > 0 gate.

## Verdict and interpretation

**S1a closes on the bounded-negative branch.** At 54–199× substrate scale
with domain-matched (guitar tablature, standard tuning) symbolic data,
count-based position/sequence priors do not transfer into GuitarSet
clean-acoustic decoding — the strongest-form replication of the A15/PDMX
"domain match beats scale" result, now with the domain-mismatch excuse
largely removed. Plausible residual mismatches, recorded for S1b design:
DadaGP repertoire skew (distortion/rock dominates; even the acoustic subset
is GP arrangements rather than GuitarSet-style comping/soloing), and
user-transcribed GP tabs encoding notation-convenient rather than
performance-typical positions. The registered `guitarset-v1` +
`guitarset-seq-v1` pair remains the correct clean-acoustic default.

Do not re-run count-prior variants (different α/power, per-mode splits,
other GM subsets) without a materially new hypothesis; the dose-response
already brackets them.

## Provenance

- Builder provenance JSONs (`synthtab_v1_{acoustic,all}.provenance.json`)
  sit beside the artifacts under
  `$TABVISION_DATA_ROOT/models/synthtab_priors/` (evaluation-only; never
  registered; NC-labeled per LICENSES.md).
- Artifact SHA-256s are embedded in both probe JSON reports.
- Probe: `scripts/eval/s1a_synthtab_prior_probe.py` (bootstrap seed
  20260720); blend builder: `scripts/eval/build_synthtab_blend_prior.py`.
- Environment: tabvision `.venv`, `TABVISION_DATA_ROOT=~/.tabvision/data`,
  `PYTHONUTF8=1`, Windows 11, 2026-07-20.
