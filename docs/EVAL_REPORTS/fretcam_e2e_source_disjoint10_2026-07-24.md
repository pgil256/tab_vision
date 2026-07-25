# FretCam current-solver paired end-to-end Tab F1

**Population:** source-disjoint-heldout10; **as of:** 2026-07-25T00:11:03Z.

## Result

| Metric | Audio baseline | + current FretCam | Delta |
|---|---:|---:|---:|
| Macro per-clip Tab F1 | 0.623750 | 0.624586 | +0.000836 |
| Macro lower-95 | 0.505994 | 0.506646 | — |
| Micro Tab F1 | 0.603557 | 0.604644 | +0.001087 |
| Wrong-position/same-pitch | 1,021 | 1,014 | 7 fewer |

Paired 95% bootstrap interval for the macro delta: `[+0.000000, +0.001994]`.

## Per clip

| Clip | Baseline | + FretCam | Delta | Wrong-pos Δ | Obs | Events affected | Paired runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `019_Vpswc` | 0.613668 | 0.613668 | +0.000000 | +0 | 4 | 2 | 99.2s |
| `111_hf1wc` | 0.302344 | 0.302344 | +0.000000 | +0 | 33 | 17 | 168.7s |
| `112_mf1wc` | 0.530017 | 0.535163 | +0.005146 | -6 | 524 | 298 | 277.7s |
| `126_XD1wc` | 0.321290 | 0.321290 | +0.000000 | +0 | 232 | 157 | 155.4s |
| `201_gk1wc` | 0.705128 | 0.705128 | +0.000000 | +0 | 3 | 0 | 98.5s |
| `222_W41wc` | 0.711046 | 0.711046 | +0.000000 | +0 | 26 | 16 | 86.7s |
| `247_sy1wc` | 0.730310 | 0.730310 | +0.000000 | +0 | 57 | 31 | 95.5s |
| `270_Jw1wc` | 0.643087 | 0.646302 | +0.003215 | -1 | 109 | 39 | 77.5s |
| `291_3Sswc` | 0.763359 | 0.763359 | +0.000000 | +0 | 2 | 3 | 75.4s |
| `358_441wc` | 0.917251 | 0.917251 | +0.000000 | +0 | 0 | 0 | 26.6s |

## Methodology

- Audio prediction input: current production TabEvent cache stripped only of position; baseline assignment reproduction required.
- FretCam `DetectionChain` + `PositionEstimator` runs live over each source MP4 using production demux timestamps and stride `3`.
- Cached cross-correlation offsets map FretCam observations from video time to the GAPS WAV/gold clock.
- Alignment requires onset-envelope peak ratio >= `2.0`; weaker peaks require an agreeing raw-waveform offset with the same peak-ratio floor.
- Both arms use the current clean-classical automatic policy: `gaps-v1` + `gaps-seq-v1` at weight `4.0`, assignment decoder `baseline`.
- Canonical Tab F1: exact string + fret and onset within 50 ms; macro mean and clip-stratified bootstrap use 10,000 resamples, seed 42.
- No gold pitch/string, cached CV anchor, policy tuning, download, or training enters prediction.

## Runtime and coverage

- Clips: `10`; media: `29.31 min`; paired pipeline runtime: `19.35 min`.
- Accepted observations: `990`; audio events affected: `563`.
- Direct waveform alignment checks required: `1`.
- Improved / unchanged / regressed clips: `2` / `8` / `0`.
