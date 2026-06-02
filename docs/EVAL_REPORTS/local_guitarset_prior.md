# Composite per-tier baseline

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 30 | 2179 | 0.5076 | 0.4448 | 0.85 | fail | 0.9375 | 0.9304 |
| clean_acoustic_strummed | 30 | 6536 | 0.6708 | 0.6015 | 0.90 | fail | 0.9229 | 0.9005 |
| clean_electric | 0 | 0 | — | — | 0.87 | missing | — | — |
| distorted_electric | 0 | 0 | — | — | 0.80 | missing | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_acoustic_single_line | GuitarSet | 30 | 0.5076 | 0.9375 | 0.9304 |
| clean_acoustic_strummed | GuitarSet | 30 | 0.6708 | 0.9229 | 0.9005 |

## Methodology

- Manifest: `data\eval\local_guitarset.toml`
- Audio backend: `highres`
- Position prior: `guitarset-v1`
- Eval-harness SHA: `<unset>`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

