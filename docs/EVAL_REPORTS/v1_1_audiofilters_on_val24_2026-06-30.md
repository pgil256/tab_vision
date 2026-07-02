# Composite per-tier baseline

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 12 | 738 | 0.4744 | 0.3735 | 0.45 | gap | 0.8706 | 0.8643 |
| clean_acoustic_strummed | 12 | 2165 | 0.4596 | 0.4133 | 0.60 | fail | 0.5822 | 0.5592 |
| clean_electric | 0 | 0 | — | — | 0.90 | missing | — | — |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 video-assisted target; audio-only is string-resolution-limited, like single-line Tab F1 (SPEC §1.4.1).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 12 | 0.4462 | 0.3539 |
| clean_acoustic_strummed | 12 | 0.2526 | 0.1974 |
| clean_electric | 0 | — | — |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_acoustic_single_line | GuitarSet | 12 | 0.4744 | 0.8706 | 0.8643 |
| clean_acoustic_strummed | GuitarSet | 12 | 0.4596 | 0.5822 | 0.5592 |

## Methodology

- Manifest: `C:\Users\patri\Documents\Projects\tab_vision\tabvision\data\eval\local_gs_val24.toml`
- Audio backend: `highres`
- Position prior: `guitarset-v1`
- Eval-harness SHA: `5091a09`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

