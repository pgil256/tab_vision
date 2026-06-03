# Composite per-tier baseline

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 0 | 0 | — | — | 0.85 | missing | — | — |
| clean_acoustic_strummed | 0 | 0 | — | — | 0.90 | missing | — | — |
| clean_electric | 12 | 1292 | 0.0784 | 0.0421 | 0.87 | fail | 0.7152 | 0.6870 |
| distorted_electric | 0 | 0 | — | — | 0.80 | missing | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_electric | GuitarTECHS | 12 | 0.0784 | 0.7152 | 0.6870 |

## Methodology

- Manifest: `data\eval\local_gt_chords.toml`
- Audio backend: `highres-fl`
- Position prior: `none`
- Eval-harness SHA: `<unset>`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

