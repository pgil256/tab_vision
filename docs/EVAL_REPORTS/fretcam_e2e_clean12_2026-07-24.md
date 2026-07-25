# FretCam current-solver paired end-to-end Tab F1

**Population:** clean12-development-confirmation; **as of:** 2026-07-25T00:42:05Z.

## Result

| Metric | Audio baseline | + current FretCam | Delta |
|---|---:|---:|---:|
| Macro per-clip Tab F1 | 0.772970 | 0.772815 | -0.000155 |
| Macro lower-95 | 0.720356 | 0.720378 | — |
| Micro Tab F1 | 0.747707 | 0.747614 | -0.000092 |
| Wrong-position/same-pitch | 1,930 | 1,931 | 1 more |

Paired 95% bootstrap interval for the macro delta: `[-0.000623, +0.000159]`.

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

## Methodology

- Audio prediction input: current production TabEvent cache stripped only of position; baseline assignment reproduction required.
- FretCam `DetectionChain` + `PositionEstimator` runs live over each source MP4 using production demux timestamps and stride `3`.
- Cached cross-correlation offsets map FretCam observations from video time to the GAPS WAV/gold clock.
- Alignment requires onset-envelope peak ratio >= `2.0`; weaker peaks require an agreeing raw-waveform offset with the same peak-ratio floor.
- Both arms use the current clean-classical automatic policy: `gaps-v1` + `gaps-seq-v1` at weight `4.0`, assignment decoder `baseline`.
- Canonical Tab F1: exact string + fret and onset within 50 ms; macro mean and clip-stratified bootstrap use 10,000 resamples, seed 42.
- No gold pitch/string, cached CV anchor, policy tuning, download, or training enters prediction.

## Runtime and coverage

- Clips: `12`; media: `47.08 min`; paired pipeline runtime: `30.34 min`.
- Accepted observations: `687`; audio events affected: `257`.
- Direct waveform alignment checks required: `0`.
- Improved / unchanged / regressed clips: `1` / `10` / `1`.
