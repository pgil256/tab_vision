# GAPS classical-route priors (gaps-v1 + gaps-seq-v1) — gate decision

**Date:** 2026-07-20
**Program:** Personal-posture Tier 2 (DECISIONS.md 2026-07-20; SPEC §1.5 amended)
**Decision:** **PASS — registered.** Classical clean/standard-tuning/capo-0
sessions auto-route to `gaps-v1` + coupled `gaps-seq-v1` instead of `none`.

## What was built

`gaps-v1` (pitch→string/fret position counts) and `gaps-seq-v1`
(Δpitch/Δstring/prev-fret singleton transitions), the exact artifact class of
the shipped `guitarset-v1`/`guitarset-seq-v1` pair, built by
`scripts/eval/build_gaps_v1_prior.py` from the **GAPS train split only**:
212 standard-tuning scores (2 malformed scores skipped and logged), 171,059
position events, 70,933 singleton transitions. The eval test split never
enters training; stems hash `67a5230b…` is recorded in both manifests.
GAPS is CC-BY-NC-SA-4.0 → the derived count tables are NC-SA-labeled
(LICENSES.md 2026-07-20 posture).

## Result on GAPS test-22 (single-line, honest post-A6 gold)

Companion reports: `gaps_classical_prior_baseline_2026-07-20.md` (baseline
rerun, reproduces the A6 0.6969 exactly),
`gaps_classical_prior_gapsv1_2026-07-20.md` (+ decomposition twin).

| condition | Tab F1 mean | lower-95 | chord-instance | onset F1 | pitch F1 |
|---|---:|---:|---:|---:|---:|
| no prior (production `auto` before this change) | 0.6969 | 0.6256 | 0.6821 | 0.8796 | 0.8703 |
| `gaps-v1` + coupled `gaps-seq-v1` | **0.7051** | 0.6339 | **0.6951** | 0.8796 | 0.8703 |

Paired clip-stratified bootstrap (10,000 resamples, seed 42, all 22 clips):

- **mean Δ Tab F1 = +0.0082, 95% CI [+0.0010, +0.0152]** — lower bound > 0.
- 16/22 clips improved, 6 regressed; worst clip −0.0338 (`031_vpswc`),
  best +0.0405 (`043_bc1wc`).
- Onset/pitch F1 byte-identical — string-assignment-only change, as designed.

## Reading

The lift is real but small compared to the +22–29 pp the GuitarSet prior gives
on GuitarSet. The reason is that the no-prior decoder is already strong on
classical repertoire: GAPS is open-position-heavy, which the decoder's
built-in playability terms (open-string bonus, low-fret bias, continuity)
capture well. The learned counts add on top of that. The important structural
change is that the classical route now has an in-domain prior instead of the
banked −13.8 pp cross-domain `guitarset-v1` regression or nothing.

## Safety

- Routing is domain-scoped: only `instrument="classical"`, `tone="clean"`,
  standard tuning, capo 0 resolve the pair; everything else is unchanged
  (verified in `tests/unit/test_inference_policy.py`).
- GuitarSet acoustic metrics are untouched by construction (different route).
- Missing/corrupt/unregistered artifacts degrade to `none`, never to the
  acoustic prior.

## Reproduction

```powershell
cd tabvision
# artifacts
.\.venv\Scripts\python.exe -m scripts.eval.build_gaps_v1_prior --gaps-root $HOME\.tabvision\data\gaps
# eval (cached, resumable)
.\.venv\Scripts\python.exe -m scripts.eval.v1_1_second_corpus_probe --manifest data\eval\gaps.toml `
  --backend highres --position-prior none --splits validation,test --output <baseline.md>
.\.venv\Scripts\python.exe -m scripts.eval.v1_1_second_corpus_probe --manifest data\eval\gaps.toml `
  --backend highres --position-prior gaps-v1 --splits validation,test --output <candidate.md>
```
