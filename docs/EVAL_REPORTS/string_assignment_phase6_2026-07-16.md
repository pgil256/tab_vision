# Sequential Tab F1 Phase 6: learned review queue and offline replay

## Frozen scope

- Automatic transcription remained frozen; all reported changes are simulated user-approved, pitch-preserving corrections.
- Development data: 35,959 production-equivalent pitch-correct ambiguous notes from GuitarSet players 00-04. Player 05 was not read.
- Features: path margin, candidate count, OOF context/timbre disagreement, native timbre strength, accepted-checkpoint posterior entropy, explicit domain score, chord size, segment inconsistency, and mode.
- Detector: fixed 10-16-8-1 MLP with `321` parameters; nested player-held Platt calibration; no grid.

## Review-detector result

- OOF ROC AUC: `0.7127` (gate `>= 0.75`).
- Global wrong-position rate: `0.3452`; highest-risk 10%: `0.6101`; enrichment `1.77x` (gate `>= 2.0x`).
- Calibrated probability ECE: `0.0303`.

| review budget | notes | precision | recall |
|---:|---:|---:|---:|
| 10% | 3,596 | 0.6101 | 0.1768 |
| 20% | 7,192 | 0.6164 | 0.3572 |
| 30% | 10,788 | 0.5875 | 0.5106 |

### Player-held folds

| player | notes | wrong rate | AUC | high-risk 10% | enrichment |
|---:|---:|---:|---:|---:|---:|
| 00 | 7,947 | 0.3549 | 0.6862 | 0.6528 | 1.84x |
| 01 | 8,560 | 0.4061 | 0.7380 | 0.7068 | 1.74x |
| 02 | 5,671 | 0.3687 | 0.7556 | 0.6919 | 1.88x |
| 03 | 7,053 | 0.2765 | 0.7348 | 0.4958 | 1.79x |
| 04 | 6,728 | 0.3084 | 0.7194 | 0.5884 | 1.91x |

Detector gate: **FAIL**.

## Offline correction replay

A review consumes two seconds per note. Wrong notes are corrected only when the gold position appears in the production decoder's displayed top three. Correct notes are rejected. No phrase/motif credit is included.

| seconds/clip | reviewed | corrections | wrong reduction | Tab F1 | solo | comp | corrections/min |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.0000 | 0.5581 | 0.5460 | 0.5702 | 0.00 |
| 10 | 1,500 | 854 | 0.0688 | 0.5812 | 0.5782 | 0.5842 | 17.08 |
| 30 | 4,500 | 2,573 | 0.2073 | 0.6281 | 0.6457 | 0.6105 | 17.15 |
| 60 | 8,999 | 4,811 | 0.3876 | 0.6873 | 0.7309 | 0.6437 | 16.04 |

- Required 60-second wrong-position reduction: `>= 0.50`; observed `0.3876`.
- Pitch-changing edits: `0`; wrong propagation: `0`; undo rate in deterministic oracle replay: `0`.
Replay gate: **FAIL**.

## Decision and reproducibility

**FAIL: do not start production UI integration or open player 05 for this fixed assisted path.**
- Deterministic complete rerun: **PASS**; prediction SHA-256 `d044a80525b4e4dc26ffd9fae40fe053023b6c65db47838c474e145fef486021`; model SHA-256 `b748a9fd97a3ec356cccfe083f6875bd3ca94f3db9d518085b1054a9369cd3dd`.
- Feature cache: `657137` bytes; cache hit `true`; feature SHA-256 `3520bd416dedc4c5ac1fd20cecf73735b19a03bd6d2994e03cb31be127c5fe00`.
- The editing core supports atomic accept/reject/undo, pitch-preserving candidate cycling and one-string phrase moves, unique K-best phrase alternatives, exact motif previews, and explicit opt-in side-information settings.
- No UI, runtime route, automatic decoder, or SPEC contract changed in this gate.

## Reproduction

```powershell
cd tabvision
.\.venv\Scripts\python.exe -m scripts.eval.string_assignment_phase6
```
