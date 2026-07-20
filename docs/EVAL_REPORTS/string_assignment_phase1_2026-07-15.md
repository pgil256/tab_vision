# Correct-pitch / wrong-string Phase 1 segment decoder

The fixed 11-point coarse grid was selected only from player-held-out OOF predictions for GuitarSet players 00-04. Player 05 was decoded after the winning configuration and decision rules were frozen.

## Frozen selection

- Grid point: `prior_0p5` (selected within the hard comp non-inferiority set).
- Configuration: `{"max_segment_notes": 32, "max_segment_s": 4.0, "offset_weight": 1.0, "prior_weight": 0.5, "register_jump_semitones": 7, "relaxed_state_change_scale": 0.25, "repeat_weight": 0.0, "rest_boundary_s": 0.75, "state_change_weight": 1.0, "transition_weight": 1.0, "zone_centers": [2, 5, 7, 10, 13], "zone_weight": 1.0}`.
- Repeat consistency stayed disabled (`repeat_weight=0`) because no deterministic motif matcher or independent ablation was introduced.
- Production `auto` still resolves to `baseline`; `segment-v1` remains explicit.

## Development grid

| config | aggregate delta [95% CI] | solo delta | comp delta [95% CI] | wrong rate | worst player | comp eligible |
|---|---:|---:|---:|---:|---:|---:|
| `z1_o1_s1_p0_t1` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `zone_0p5` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `zone_2` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `offset_0p5` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `offset_2` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `state_0p5` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `state_2` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `prior_0p25` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `prior_0p5` | +0.0004 [-0.0005, +0.0014] | +0.0009 | -0.0001 [-0.0004, +0.0002] | 0.3338 | -0.0007 | 1 |
| `transition_0p75` | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 [+0.0000, +0.0000] | 0.3341 | +0.0000 | 1 |
| `transition_1p25` | -0.0000 [-0.0001, +0.0000] | +0.0000 | -0.0000 [-0.0001, +0.0000] | 0.3341 | -0.0001 | 1 |

## OOF and frozen confirmation metrics

| split / decoder | solo Tab F1 | comp Tab F1 | aggregate | micro | top-1 | top-3 | wrong rate | aggregate delta [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OOF baseline` | 0.5460 | 0.5702 | 0.5581 | 0.5577 | 0.6548 | 0.9967 | 0.3452 | baseline |
| `OOF segment-v1` | 0.5470 | 0.5701 | 0.5585 | 0.5579 | 0.6551 | 0.9967 | 0.3449 | +0.0004 [-0.0005, +0.0014] |
| `player05 baseline` | 0.5418 | 0.6834 | 0.6126 | 0.6279 | 0.6770 | 0.9986 | 0.3230 | baseline |
| `player05 segment-v1` | 0.5453 | 0.6834 | 0.6143 | 0.6287 | 0.6780 | 0.9986 | 0.3220 | +0.0017 [+0.0000, +0.0051] |

Top-3 exact latent/candidate paths were retained for every frozen evaluation clip. Mean second-path margin: 0.1826; mean third-path margin: 0.2962 nats.

The decoder changes only string/fret positions. Audio-event onsets and MIDI pitches are passed through unchanged; the unit suite enforces exact pitch equivalence and chord feasibility.

| unchanged audio-event metric | baseline | segment-v1 |
|---|---:|---:|
| Onset F1 (50 ms) | 0.9302 | 0.9302 |
| Pitch F1 (50 ms, no offset) | 0.9154 | 0.9154 |

The summary CSV contains error counts and rates by player, mode, style, track, MIDI pitch, candidate count/rank, reference and predicted string, string displacement, and fret displacement.

## Runtime and determinism

- Baseline decode benchmark: 4.026 s over 1828.1 s of player05 audio.
- Segment decode benchmark: 26.385 s; added 0.734 s per 60 s clip.
- Projected total: 45.73 s per 60 s clip, below the five-minute limit and below the +20% allowance.
- Learned artifact size: 0 bytes (the decoder is deterministic code and frozen constants).
- Frozen player05 top-1 prediction SHA-256: `9788d929bea9a7ca414050f8de10370a352d4fe848ed8baedb053f66cdb5d7ef`; an independent top-1 rerun matched it exactly.

## Gate decision

**`close_rule_based_segment_decoding`**

- OOF aggregate delta +0.0004
- confirmation solo delta +0.0035
- confirmation aggregate delta +0.0017, CI lower +0.0000
- confirmation wrong-position relative reduction +0.3%
- confirmation comp delta +0.0000, CI lower +0.0000
- projected 60 s runtime 45.73 s

Classical, electric, distorted, capo, and alternate-tuning requests are covered by routing tests and resolve to `baseline` before fusion. Their decoder delta is therefore exactly zero; the previously verified GAPS and Guitar-TECHS baseline paths remain unchanged.

## Reproduction

```powershell
cd tabvision
& .\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase1 `
  --data-home $HOME\.tabvision\data\guitarset `
  --output-dir ..\docs\EVAL_REPORTS
```

Full source/cache/output hashes and observational runtime data: `string_assignment_phase1_2026-07-15_provenance.json`.
