# String assignment Phase 3: high-resolution uncertainty and checkpoint ensemble

## Development OOF results

| condition | solo | comp | macro Tab F1 | delta [95% CI] | onset F1 | pitch F1 | ambiguous top-1 | top-3 | wrong rate | worst player |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.5460 | 0.5702 | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.8838 | 0.8476 | 0.6659 | 0.9998 | 0.3341 | +0.0000 |
| `gaps` | 0.5460 | 0.5702 | 0.5581 | +0.0000 [+0.0000, +0.0000] | 0.8838 | 0.8476 | 0.6659 | 0.9998 | 0.3341 | +0.0000 |
| `fl` | 0.5539 | 0.6461 | 0.6000 | +0.0419 [+0.0348, +0.0489] | 0.9150 | 0.8913 | 0.6957 | 0.9999 | 0.3043 | +0.0245 |
| `union` | 0.5458 | 0.6180 | 0.5819 | +0.0238 [+0.0179, +0.0298] | 0.8974 | 0.8713 | 0.6914 | 0.9998 | 0.3086 | +0.0081 |
| `intersection` | 0.5538 | 0.5935 | 0.5736 | +0.0156 [+0.0123, +0.0189] | 0.8915 | 0.8688 | 0.6649 | 0.9999 | 0.3351 | +0.0097 |
| `confidence_winner` | 0.5577 | 0.6458 | 0.6017 | +0.0436 [+0.0376, +0.0497] | 0.9187 | 0.8956 | 0.6887 | 0.9999 | 0.3113 | +0.0246 |
| `logistic` | 0.5571 | 0.6458 | 0.6014 | +0.0433 [+0.0373, +0.0494] | 0.9187 | 0.8959 | 0.6888 | 0.9999 | 0.3112 | +0.0260 |

## Posterior oracle and calibration

- Current pitch-off or missed gold events: `10503` (pitch-off `6281`, missed `4222`).
- Gold pitch in raw top 2: `30.8%`; top 3: `42.7%`.
- Fixed two-hypothesis eligibility recovers `2.5%` and adds `6.45` false candidates per ten correct events.
- Missed notes with subthreshold onset evidence: `854/4222`; frame evidence: `133/4222`.
- Posterior-entropy error AUC: pitch `0.6269`, string `0.4322`.

## Gate decision

- Posterior/lattice gate: **fail**.
- Best checkpoint/ensemble condition: `confidence_winner` at `+0.0436` with lower bound `+0.0376`.
- Ensemble gate: **pass**.
- Projected two-checkpoint CPU pipeline runtime: `100.41` seconds per 60-second clip.

The development ensemble gate passed; player-05 confirmation is required.

## Frozen player 05 confirmation

Player 05 was opened only after `confidence_winner`, its features, and its 0.5 threshold were frozen from players 00–04. No confirmation result was used for retuning.

| condition | solo | comp | aggregate | micro | onset F1 | pitch F1 | ambiguous top-1 | top-3 | wrong rate | delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.5418 | 0.6834 | 0.6126 | 0.6279 | 0.9320 | 0.9169 | 0.6809 | 0.9997 | 0.3191 | baseline |
| `confidence_winner` | 0.5503 | 0.7175 | 0.6339 | 0.6641 | 0.9491 | 0.9403 | 0.6979 | 0.9997 | 0.3021 | +0.0213 [+0.0104, +0.0342] |

Same-pitch wrong-position relative reduction: `+5.3%`.
Phase 3 confirmation safety check: **pass**.
Cumulative automatic promotion guardrails: **fail**.
The artifact is registered for explicit clean-acoustic evaluation; automatic routing remains on the production GAPS backend pending the integrated Phase 7 gate.

## Runtime, memory, artifact, and determinism

The reproducible 60-second end-to-end audio-only pipeline benchmark ran the two checkpoints sequentially in isolated processes. It completed in `59.108 s` and `64.213 s`, with peak working sets of `2,062,364,672` and `2,061,873,152` bytes. Both runs emitted 188 audio events and 188 tab events with the identical SHA-256 `1d4ece2570ac73b99f9a825700f6aa2dd1ff9dd2dbaeab73321c012d05c11d5e3`. The measured maximum remains well below the five-minute limit; the development-cache projection of `100.41 s` is conservative.

The registered scalar-calibration artifact is `1,166` bytes with SHA-256 `b6cad83e68ad181337eb935419382610e39d40ce4acbca3720ee05377a7f8296`. GAPS is released before FL is loaded; the sequential-backend test pins maximum simultaneous checkpoint instances to one. Full benchmark provenance is in `string_assignment_phase3_2026-07-15_benchmark.json`.

## Domain routing and integration scope

| session/corpus | automatic audio backend after Phase 3 | Phase 3 ensemble automatic? | metric delta from prior route |
|---|---|---:|---:|
| clean acoustic | `highres` | no | `0.0000` |
| GAPS classical safety corpus | `highres` | no | `0.0000` |
| Guitar-TECHS clean electric safety corpus | `highres-electric` | no | `0.0000` |
| distorted acoustic/electric | existing tone/instrument route | no | `0.0000` |

No automatic safety-corpus event stream changes in this phase, so onset, pitch, and Tab F1 deltas on those routes are exactly zero. Explicit `highres-ensemble` use is restricted internally to clean acoustic sessions; non-clean or non-acoustic explicit sessions deterministically use GAPS only. Automatic activation remains false because player 05 improved solo Tab F1 by only `+0.0085` (guardrail `+0.03`) and reduced ambiguous same-pitch wrong-position errors by only `5.3%` (guardrail `10%`), despite passing the Phase 3 event-ensemble gate.
