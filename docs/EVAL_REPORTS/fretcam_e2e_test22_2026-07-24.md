# FretCam current-solver paired end-to-end Tab F1

**Population:** GAPS-test22-combined; **as of:** 2026-07-25T00:42:41Z.

## Result

| Metric | Audio baseline | + current FretCam | Delta |
|---|---:|---:|---:|
| Macro per-clip Tab F1 | 0.705143 | 0.705438 | +0.000296 |
| Macro lower-95 | 0.636808 | 0.637262 | — |
| Micro Tab F1 | 0.694012 | 0.694359 | +0.000347 |
| Wrong-position/same-pitch | 2,951 | 2,945 | 6 fewer |

Paired 95% bootstrap interval for the macro delta: `[-0.000219, +0.000935]`.

## Per clip

| Clip | Baseline | + FretCam | Delta | Wrong-pos Δ | Obs | Events affected | Paired runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `027_Zpswc` | 0.677298 | 0.677298 | +0.000000 | +0 | 134 | 68 | 253.5s |
| `031_vpswc` | 0.825719 | 0.825719 | +0.000000 | +0 | 84 | 23 | 109.4s |
| `043_bc1wc` | 0.748845 | 0.748845 | +0.000000 | +0 | 0 | 0 | 218.9s |
| `063_bV1wc` | 0.665868 | 0.665868 | +0.000000 | +0 | 0 | 0 | 109.7s |
| `104_xf1wc` | 0.716846 | 0.716846 | +0.000000 | +0 | 17 | 6 | 113.7s |
| `118_VD1wc` | 0.854829 | 0.852336 | -0.002492 | +2 | 10 | 6 | 83.8s |
| `142_GD1wc` | 0.830570 | 0.830570 | +0.000000 | +0 | 114 | 46 | 146.7s |
| `179_pM1wc` | 0.853256 | 0.853256 | +0.000000 | +0 | 97 | 27 | 133.3s |
| `212_y41wc` | 0.690776 | 0.690776 | +0.000000 | +0 | 47 | 15 | 178.5s |
| `235_Ny1wc` | 0.628935 | 0.629571 | +0.000636 | -1 | 166 | 60 | 325.8s |
| `294_BSswc` | 0.894292 | 0.894292 | +0.000000 | +0 | 9 | 0 | 81.9s |
| `341_1M1wc` | 0.888406 | 0.888406 | +0.000000 | +0 | 9 | 6 | 65.0s |
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

- Clips: `22`; media: `76.39 min`; paired pipeline runtime: `49.69 min`.
- Accepted observations: `1677`; audio events affected: `820`.
- Direct waveform alignment checks required: `1`.
- Improved / unchanged / regressed clips: `3` / `18` / `1`.
