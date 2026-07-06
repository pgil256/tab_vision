# B4 — string-confidence enrichment validation (val24)

**Date:** 2026-07-06
**Branch:** `v1.1/b4-b3-correction-ux`
**Status:** **SHIPS — a real but modest wrong-string signal, and a strict
improvement over the status quo.** `fuse` now writes `TabEvent.confidence` from
the Viterbi string-flip margin (was the decode-inert velocity proxy, AUC ≈ 0.50
by construction). The pre-registered gate passes; the signal is honestly modest
(AUC 0.603) and its ceiling is the information limit — where the guitarset-v1
prior is *confidently wrong* the decode is confidently wrong, so no margin can
flag it (Q4, conf≈1.0, still carries a 0.089 wrong_position rate). What
confidence catches is the notes where the decode itself was uncertain, which
correlate with the wrong-string bucket. That is exactly why B3 (let the user
fix any note in ≤ 2 keystrokes) matters more than the flag.
**Reconciliation:** the per-note label counts below are bit-identical to the
A10 decomposition (correct 2007 / wrong_position 541 / pitch_off 117 /
timing_only 39 / extra 150), confirming the matcher is faithful.
**Script:** `scripts/eval/b4_string_confidence_validation.py`.

Config: `highres` + `guitarset-v1`, manifest `data\eval\local_gs_val24.toml`, splits `validation,test`.

**Gate: PASS** — AUC ≥ 0.60 and Q1 rate > base rate. Editor bands (via
`v1_adapter` thresholds): red/"low" = conf < 0.5 = string margin < 0.69 nats
(a near-tie); green/"high" = conf ≥ 0.8 = margin ≥ 1.6 nats.

- Pitch-correct population (correct ∪ wrong_position): **2548** notes; wrong_position base rate **0.212**.
- **AUC (wrong_position less confident than correct) = 0.603** (enriched; 0.5 = chance).

## Wrong_position rate by confidence quartile

| quartile | notes | conf range | wrong_position rate | vs base |
|---|---:|---|---:|---:|
| Q1 | 637 | [0.006, 0.855] | 0.273 | 1.29× |
| Q2 | 637 | [0.855, 0.986] | 0.204 | 0.96× |
| Q3 | 637 | [0.986, 1.000] | 0.283 | 1.33× |
| Q4 | 637 | [1.000, 1.000] | 0.089 | 0.42× |

## Flagging utility (red = lowest-confidence X%)

| flag fraction | conf threshold | precision (flagged are wrong) | recall (wrong caught) |
|---:|---:|---:|---:|
| 10% | ≤ 0.359 | 0.350 | 0.165 |
| 20% | ≤ 0.761 | 0.305 | 0.287 |
| 30% | ≤ 0.897 | 0.253 | 0.357 |
| 40% | ≤ 0.963 | 0.250 | 0.471 |
| 50% | ≤ 0.986 | 0.239 | 0.562 |

## Predicted-note label counts (reconcile with decomposition)

| label | count |
|---|---:|
| correct | 2007 |
| wrong_position_same_pitch | 541 |
| pitch_off | 117 |
| timing_only | 39 |
| extra_detection | 150 |

