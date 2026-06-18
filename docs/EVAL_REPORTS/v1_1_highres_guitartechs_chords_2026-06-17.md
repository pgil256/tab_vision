# Guitar-TECHS chords highres second-corpus (chunk-4)

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 0 | 0 | — | — | 0.45 | missing | — | — |
| clean_acoustic_strummed | 0 | 0 | — | — | 0.60 | missing | — | — |
| clean_electric | 12 | 1292 | 0.0700 | 0.0377 | 0.90 | fail | 0.7321 | 0.6787 |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 video-assisted target; audio-only is string-resolution-limited, like single-line Tab F1 (SPEC §1.4.1).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 0 | — | — |
| clean_acoustic_strummed | 0 | — | — |
| clean_electric | 12 | 0.0207 | 0.0000 |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_electric | GuitarTECHS | 12 | 0.0700 | 0.7321 | 0.6787 |

## Methodology

- Manifest: `data\eval\local_gt_chords.toml`
- Audio backend: `highres`
- Position prior: `none`
- Eval-harness SHA: `b25dfa9`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

