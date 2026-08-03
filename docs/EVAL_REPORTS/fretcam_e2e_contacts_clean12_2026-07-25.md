# FretCam current-solver paired end-to-end Tab F1

**Population:** clean12-contact-evidence; **as of:** 2026-07-25T22:44:40Z.

## Result

| Metric | Audio baseline | + current FretCam | Delta |
|---|---:|---:|---:|
| Macro per-clip Tab F1 | 0.772970 | 0.772608 | -0.000362 |
| Macro lower-95 | 0.720356 | 0.720313 | — |
| Micro Tab F1 | 0.747707 | 0.747430 | -0.000277 |
| Wrong-position/same-pitch | 1,930 | 1,934 | 4 more |

Paired 95% bootstrap interval for the macro delta: `[-0.001246, +0.000159]`.

## Per clip

| Clip | Baseline | + FretCam | Delta | Wrong-pos Δ | Obs | Events affected | Paired runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `027_Zpswc` | 0.677298 | 0.677298 | +0.000000 | +0 | 134 | 189 | 246.9s |
| `031_vpswc` | 0.825719 | 0.825719 | +0.000000 | +0 | 84 | 44 | 110.4s |
| `043_bc1wc` | 0.748845 | 0.748845 | +0.000000 | +0 | 0 | 6 | 206.2s |
| `063_bV1wc` | 0.665868 | 0.665868 | +0.000000 | +0 | 0 | 0 | 103.5s |
| `104_xf1wc` | 0.716846 | 0.716846 | +0.000000 | +0 | 17 | 17 | 105.8s |
| `118_VD1wc` | 0.854829 | 0.849844 | -0.004984 | +4 | 10 | 150 | 76.0s |
| `142_GD1wc` | 0.830570 | 0.830570 | +0.000000 | +0 | 114 | 211 | 135.4s |
| `179_pM1wc` | 0.853256 | 0.853256 | +0.000000 | +0 | 97 | 150 | 123.0s |
| `212_y41wc` | 0.690776 | 0.690776 | +0.000000 | +1 | 47 | 66 | 162.9s |
| `235_Ny1wc` | 0.628935 | 0.629571 | +0.000636 | -1 | 166 | 284 | 298.7s |
| `294_BSswc` | 0.894292 | 0.894292 | +0.000000 | +0 | 9 | 16 | 75.8s |
| `341_1M1wc` | 0.888406 | 0.888406 | +0.000000 | +0 | 9 | 38 | 56.2s |

## Methodology

- Audio prediction input: current production TabEvent cache stripped only of position; baseline assignment reproduction required.
- FretCam `DetectionChain` + `PositionEstimator` runs live over each source MP4 using production demux timestamps and stride `3`.
- Cached cross-correlation offsets map FretCam observations from video time to the GAPS WAV/gold clock.
- Alignment requires onset-envelope peak ratio >= `2.0`; weaker peaks require an agreeing raw-waveform offset with the same peak-ratio floor.
- Both arms use the current clean-classical automatic policy: `gaps-v1` + `gaps-seq-v1` at weight `4.0`, assignment decoder `baseline`.
- Canonical Tab F1: exact string + fret and onset within 50 ms; macro mean and clip-stratified bootstrap use 10,000 resamples, seed 42.
- No gold pitch/string, cached CV anchor, policy tuning, download, or training enters prediction.

## Runtime and coverage

- Clips: `12`; media: `47.08 min`; paired pipeline runtime: `28.35 min`.
- Accepted observations: `687`; audio events affected: `1171`.
- Direct waveform alignment checks required: `0`.
- Improved / unchanged / regressed clips: `1` / `10` / `1`.
