# FretCam F4e-A - public position benchmark baseline

**Date:** 2026-07-22

**Scope:** Measurement only; unchanged F4d product inference

**Verdict:** BENCHMARK FROZEN; no accuracy gate is claimed from this small set

## Frozen evidence

- Corpus: GAPS (CC-BY-NC-SA-4.0); public footage only.
- Sequences: 16 across 12 sources; dev 8 sequences/7 sources, test 8 sequences/5 sources.
- Dev/test sources are disjoint.
- Samples: 450 frames total before unlabeled ranges are omitted.
- Sample rate: 10 FPS.
- Stable positions represented: 1, 2, 3, 5, 6, 7, 9.
- Policy: Stable intervals require a visually countable nut-to-index fret relationship; shifts and deterministic public-frame occlusions have explicit ground-truth boundaries; uncertain ranges are omitted from scoring.
- This report is the one initial opening of the frozen test baseline. Future rule/threshold choices use dev only; test is rerun once for the final comparison.

## F4d baseline

| split | valid observations | displayed precision | coverage | false-lock rate |
|---|---:|---:|---:|---:|
| dev | 0.398 (64/161) | 0.825 (47/57) | 0.323 (52/161) | 0.062 (10/161) |
| test | 0.278 (32/115) | 0.667 (8/12) | 0.070 (8/115) | 0.000 (0/115) |
| all | 0.348 (96/276) | 0.797 (55/69) | 0.217 (60/276) | 0.036 (10/276) |

- Negative-control display rate: 0.031 (4/128).

### Display rate by ground-truth state

| state | displayed frames |
|---|---:|
| stable | 0.217 (60/276) |
| shifting | 0.000 (0/12) |
| dropout | 0.294 (5/17) |
| invalid | 0.031 (4/128) |

- Shift latency: not observed across 0 observed event(s); 0 censored and 3 excluded because the origin was not freshly locked.
- Dropout recovery from the annotated valid-return boundary: not observed across 0 observed event(s); 1 censored and 1 origin-not-locked.

## Position breakdown

| position | precision | coverage | false-lock rate |
|---:|---:|---:|---:|
| 1 | 0.971 (34/35) | 0.507 (35/69) | 0.014 (1/69) |
| 2 | 1.000 (19/19) | 0.179 (14/78) | 0.000 (0/78) |
| 3 | n/a | 0.000 (0/30) | 0.000 (0/30) |
| 5 | 1.000 (2/2) | 0.056 (2/36) | 0.000 (0/36) |
| 6 | 0.000 (0/9) | 0.643 (9/14) | 0.643 (9/14) |
| 7 | n/a | 0.000 (0/30) | 0.000 (0/30) |
| 9 | n/a | 0.000 (0/19) | 0.000 (0/19) |

## Technique breakdown

| technique | precision | coverage | false-lock rate |
|---|---:|---:|---:|
| barre | 0.971 (34/35) | 0.486 (35/72) | 0.014 (1/72) |
| chord | 0.591 (13/22) | 0.195 (17/87) | 0.103 (9/87) |
| note | 0.667 (8/12) | 0.068 (8/117) | 0.000 (0/117) |

## Visibility breakdown

| visibility | precision | coverage | false-lock rate |
|---|---:|---:|---:|
| close | 0.894 (42/47) | 0.473 (43/91) | 0.011 (1/91) |
| full_neck | 0.591 (13/22) | 0.092 (17/185) | 0.049 (9/185) |

## Lighting breakdown

| lighting | precision | coverage | false-lock rate |
|---|---:|---:|---:|
| bright | 0.894 (42/47) | 0.228 (43/189) | 0.005 (1/189) |
| mixed | 0.591 (13/22) | 0.195 (17/87) | 0.103 (9/87) |

## Sequence diagnostics

| sequence | valid observations | display coverage | precision |
|---|---:|---:|---:|
| `dev_031_barre_i` | 0.667 (36/54) | 0.648 (35/54) | 0.971 (34/35) |
| `dev_077_off_neck_negative` | n/a | n/a | n/a |
| `dev_104_ii_to_vi` | 0.595 (25/42) | 0.405 (17/42) | 0.591 (13/22) |
| `dev_105_boundary_negative` | n/a | n/a | n/a |
| `dev_118_note_vii` | 0.200 (3/15) | 0.000 (0/15) | n/a |
| `dev_142_chord_iii` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `dev_142_note_v` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `dev_341_shift_to_ix` | 0.000 (0/20) | 0.000 (0/20) | n/a |
| `test_027_note_ix` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `test_178_crossfade_invalid` | n/a | n/a | 0.000 (0/4) |
| `test_178_ii_to_v` | 1.000 (22/22) | 0.364 (8/22) | 1.000 (8/8) |
| `test_179_chord_i` | 0.067 (1/15) | 0.000 (0/15) | n/a |
| `test_179_note_v` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `test_235_chord_iii` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `test_235_note_vii` | 0.000 (0/15) | 0.000 (0/15) | n/a |
| `test_238_barre_ii_occlusion` | 0.500 (9/18) | 0.000 (0/18) | n/a |

## Interpretation and limits

- The baseline's main limitation is coverage: 0.217 (60/276) overall and 0.070 (8/115) held-out, alongside an overall valid-observation rate of 0.348 (96/276).
- Held-out stable false locks are 0.000 (0/115); four displays on the held-out invalid crossfade reduce held-out displayed precision to 0.667 (8/12).
- No transition yielded a numeric latency: all three shift origins lacked a fresh valid lock; one dropout origin lacked a lock and the other recovery was right-censored.
- This is the first defensible position-HUD measurement set, not a population estimate. It is intentionally small and excludes uncertain ranges.
- Two behaviors carry prior user verification; the remaining intervals were independently frame-reviewed against visible nut/fret wires.
- Dropout recovery uses manifest-annotated occlusion and valid-return boundaries. The occlusions are deterministic masks over public frames; they are not inferred from product output.
- GAPS was not recorded under FretCam's controlled-camera contract and has no native classical-position labels. A second human label review would strengthen the benchmark.
- The short dev Position-IX arrival contains only four 10 FPS samples, so that shift can be window-censored before the five-frame estimator can lock.
- No threshold was selected from these results, and no inference, dependency, model, download, training run, or TabVision package behavior changed.

## Reproduce

```powershell
.\fretcam\.venv\Scripts\python.exe -m fretcam.position_benchmark --split all --output-json <machine-local-output.json>
```

## Verification

- The frozen run completed all 450 samples and wrote the machine-local result
  to `~/.tabvision/cache/fretcam_artifacts/f4e_a_position_baseline_v1.json`;
  public media and frame artifacts remain uncommitted.
- Reconstructing every prediction from that JSON and rescoring against the
  checked-in manifest produced an exact metrics match.
- Full FretCam suite: **55 passed** (one pre-existing Starlette deprecation
  warning).
- `ruff check fretcam/src fretcam/tests`: passed.
- Ruff format check: passed for the two new Python files. The repo-wide check
  still identifies four pre-existing files outside this iteration
  (`benchmark_hud.py`, `gaps_anchor_probe.py`, `guidance.py`, and
  `test_guidance.py`); they were deliberately left untouched.
