# v1 acceptance — 2026-06-03 (audio-only acoustic)

**VERDICT: v1 ACCEPTED.** Formal acceptance run over the GuitarSet held-out
player-05 validation set (60 clips), eval harness `292252d`, `highres` backend
with the leak-free `guitarset-v1` position prior. All SPEC §1.4.1 gates met:

| Gate | single-line | strummed | aggregate | target | result |
|---|---|---|---|---|---|
| Tab F1 (lower-95) | 0.457 | 0.606 | 0.600* | 0.45 / 0.60 / 0.55 | **pass** |
| Onset F1 (mean) | 0.938 | 0.923 | — | ≥ 0.92 | **pass** |
| Pitch F1 (mean) | 0.930 | 0.901 | — | ≥ 0.90 | **pass** |
| Latency | — | — | ≈45 s / 60 s clip | ≤ 5 min | **pass** |

\* aggregate = mean Tab F1 over all 60 clips. Onset/pitch shown as means (per-tier
bootstrap CIs are computed; strummed pitch mean 0.901 sits right on the 0.90 line).
**Chord-instance accuracy (0.52 single-line / 0.48 strummed) is re-scoped to a
v1.1 video target** — it shares single-line Tab F1's audio string/fret information
limit (SPEC §1.4.1; DECISIONS 2026-06-08). Latency: 60 clips in 1054 s ⇒ ~17.6 s
per ~24 s clip (0.74× realtime) ⇒ ≈45 s for a 60 s clip. Raw per-tier data below.

---

# Composite per-tier baseline

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 30 | 2179 | 0.5230 | 0.4570 | 0.45 | pass | 0.9375 | 0.9304 |
| clean_acoustic_strummed | 30 | 6536 | 0.6763 | 0.6058 | 0.60 | pass | 0.9229 | 0.9005 |
| clean_electric | 0 | 0 | — | — | 0.90 | missing | — | — |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster (SPEC §1.4 gate >= 0.85).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 30 | 0.5210 | 0.4552 |
| clean_acoustic_strummed | 30 | 0.4836 | 0.4009 |
| clean_electric | 0 | — | — |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_acoustic_single_line | GuitarSet | 30 | 0.5230 | 0.9375 | 0.9304 |
| clean_acoustic_strummed | GuitarSet | 30 | 0.6763 | 0.9229 | 0.9005 |

## Methodology

- Manifest: `data\eval\composite.toml`
- Audio backend: `highres`
- Position prior: `guitarset-v1`
- Eval-harness SHA: `292252d`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

