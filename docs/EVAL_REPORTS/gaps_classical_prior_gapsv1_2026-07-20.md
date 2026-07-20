# GAPS test-22 with gaps-v1 + coupled gaps-seq-v1 - 2026-07-20

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 22 | 16079 | 0.7051 | 0.6339 | 0.45 | pass | 0.8796 | 0.8703 |
| clean_acoustic_strummed | 0 | 0 | — | — | 0.60 | missing | — | — |
| clean_electric | 0 | 0 | — | — | 0.90 | missing | — | — |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 video-assisted target; audio-only is string-resolution-limited, like single-line Tab F1 (SPEC §1.4.1).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 22 | 0.6951 | 0.6489 |
| clean_acoustic_strummed | 0 | — | — |
| clean_electric | 0 | — | — |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_acoustic_single_line | GAPS | 22 | 0.7051 | 0.8796 | 0.8703 |

## Methodology

- Manifest: `data\eval\gaps.toml`
- Audio backend: `highres`
- Position prior: `gaps-v1`
- Eval-harness SHA: `<unset>`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

