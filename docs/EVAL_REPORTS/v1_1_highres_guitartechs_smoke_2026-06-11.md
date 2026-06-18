# Composite per-tier baseline

## Run notes

- Purpose: highres-only second-dataset smoke after the 2026-06-11 direction to
  keep highres and avoid an audio-model switch.
- Dataset slice: shortest local Guitar-TECHS direct-input chord clip, with
  per-string MIDI labels.
- Settings: `--backend highres --position-prior none`; no GuitarSet prior and
  no alternate backend.
- A same-settings 12-clip Guitar-TECHS chord run was attempted first and
  exceeded the 30-minute interactive budget before writing a report. Use this
  smoke for quick checks; use a cached or batched runner for the full
  second-corpus gate.

## Per-tier results

| Tier | Clips | Gold notes | Tab F1 mean | Tab F1 lower-95 | Target | Status | Onset F1 | Pitch F1 |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean_acoustic_single_line | 0 | 0 | — | — | 0.45 | missing | — | — |
| clean_acoustic_strummed | 0 | 0 | — | — | 0.60 | missing | — | — |
| clean_electric | 1 | 27 | 0.0000 | 0.0000 | 0.90 | fail | 0.7187 | 0.6562 |
| distorted_electric | 0 | 0 | — | — | 0.82 | missing | — | — |

## Chord-instance accuracy

Whole-fingering recovery per chord cluster. The >= 0.85 bar is a v1.1 video-assisted target; audio-only is string-resolution-limited, like single-line Tab F1 (SPEC §1.4.1).

| Tier | Clips | Chord acc mean | Lower-95 |
|---|---:|---:|---:|
| clean_acoustic_single_line | 0 | — | — |
| clean_acoustic_strummed | 0 | — | — |
| clean_electric | 1 | 0.0000 | 0.0000 |
| distorted_electric | 0 | — | — |

## Per-source breakdown

| Tier | Source | Clips | Tab F1 mean | Onset F1 mean | Pitch F1 mean |
|---|---|---:|---:|---:|---:|
| clean_electric | GuitarTECHS | 1 | 0.0000 | 0.7187 | 0.6562 |

## Methodology

- Manifest: `data\eval\guitartechs_highres_smoke.toml`
- Audio backend: `highres`
- Position prior: `none`
- Eval-harness SHA: `b25dfa9`
- Onset tolerance: 50 ms
- Bootstrap: N=10,000, seed=42, 95% percentile interval
- Acceptance gate: `lower_95_CI >= target` per design plan §5

