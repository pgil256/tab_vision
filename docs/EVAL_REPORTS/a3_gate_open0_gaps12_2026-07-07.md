# A3 gate — `OPEN_STRING_BONUS` 0.5 → 0.0 (gaps.toml)

Config: `highres` + `none`, splits `test`, CLEAN_12 subset. Fusion-only A/B (raw events shared).

## Per-tier Tab F1 (mean / lower-95, bootstrap N=10k)

| tier | clips | baseline mean (lo-95) | override mean (lo-95) | Δ mean | Δ lo-95 |
|---|---:|---:|---:|---:|---:|
| clean_acoustic_single_line | 12 | 0.7660 (0.7093) | 0.7562 (0.7003) | -0.0098 | -0.0091 |

## Per-clip deltas (no-regression check)

- 12 clips: 1 improved, 11 regressed, 0 unchanged.
- Worst regressions: gaps/294_BSswc (-0.023), gaps/142_GD1wc (-0.016), gaps/027_Zpswc (-0.014), gaps/118_VD1wc (-0.012), gaps/235_Ny1wc (-0.012), gaps/341_1M1wc (-0.012), gaps/179_pM1wc (-0.010), gaps/212_y41wc (-0.007)

## Verdict

**FAIL** — a tier's lower-95 REGRESSED; 11 per-clip regression(s) [HARD gate]. (bar: per-clip no-regression + lower-95)

