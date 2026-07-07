# A5 — chord-shape bonus: val24 sweep + full gate (2026-07-07)

**Lever:** `tabvision.fusion.chord_shapes.CHORD_SHAPE_BONUS` — a per-cluster
emission reward (`cost -= CHORD_SHAPE_BONUS * overlap`) when a decoded chord
cluster's `(string, fret)` positions overlap a canonical voicing (133 open /
barre / power shapes ported from v0) by ≥ 3. Backend `highres`. Raw AudioEvents
transcribed once and shared across the A/B (fusion-only override).

## 1. val24 magnitude sweep (`guitarset-v1` prior, splits validation,test)

Baseline reproduces the roadmap val24 numbers (**0.4820 / 0.7951** — harness
validation). Single-line is **exactly invariant** at every magnitude (the ≥3-note
match gate can't fire on singleton clusters), so only strummed moves.

| `CHORD_SHAPE_BONUS` | single-line | strummed | aggregate | Δ agg |
|---:|---:|---:|---:|---:|
| 0.0 (off) | 0.4820 | 0.7951 | 0.6386 | — |
| **0.1** | 0.4820 | **0.7980** | 0.6400 | **+0.0014** |
| 0.25 | 0.4820 | 0.7901 | 0.6361 | −0.0025 |
| 0.5 | 0.4820 | 0.7809 | 0.6314 | −0.0071 |
| 1.0 | 0.4820 | 0.7778 | 0.6299 | −0.0086 |

Best magnitude **0.1**; count-scaling over-biases and turns negative by 0.25.

## 2. Gate — `CHORD_SHAPE_BONUS` 0.0 → 0.1 (per A3 discipline)

### Leg 1 — in-domain, GuitarSet 60-clip player-05 (`guitarset-v1`, split validation)

| tier | clips | baseline mean (lo-95) | override mean (lo-95) | Δ mean | Δ lo-95 |
|---|---:|---:|---:|---:|---:|
| single_line | 30 | 0.5230 (0.4570) | 0.5230 (0.4570) | +0.0000 | +0.0000 |
| strummed | 30 | 0.6763 (0.6058) | 0.6816 (0.6119) | +0.0053 | +0.0061 |

Per-clip: 7 improved, 5 regressed (worst `05_SS3-84-Bb_comp` −0.028), 48 unchanged.
**PASS** — per-tier lower-95 held (the in-domain bar); per-clip churn informational.

### Leg 2 — cross-domain, GAPS clean-12 (`--position-prior none`, `--strict-per-clip`)

| tier | clips | baseline mean (lo-95) | override mean (lo-95) | Δ mean | Δ lo-95 |
|---|---:|---:|---:|---:|---:|
| single_line | 12 | 0.7660 (0.7093) | 0.7667 (0.7097) | +0.0006 | +0.0003 |

Per-clip: 3 improved, **0 regressed**, 9 unchanged. **PASS** — the hard
cross-domain bar (per-clip no-regression + lower-95) held.

*(The GAPS "single_line" tier moves at all only because dense passages
transiently form 3+-note clusters where the bonus can fire — and here it only
ever helps.)*

### (Corroborating) val24 24-clip lower-95, same override

| tier | Δ mean | Δ lo-95 |
|---|---:|---:|
| single_line | +0.0000 | +0.0000 |
| strummed | +0.0028 | +0.0024 |

## 3. Verdict

**`CHORD_SHAPE_BONUS=0.1` clears the full measurement discipline on both legs** —
60-clip in-domain lower-95 **and** GAPS clean-12 strict cross-domain
no-regression. It is the **first fusion-constant candidate to pass both**: A3's
`OPEN_STRING_BONUS=0.0` passed GuitarSet but FAILED GAPS (single-line lo-95
−0.0091); A4 washed. The difference is that this reward is grounded in canonical
voicing **geometry**, not corpus-specific prior tuning — so it is domain-neutral.

**Effect (honest):** strummed **+0.0053** (60-clip) / +0.0028 (val24), lower-95
up on both; single-line **exactly untouched** on GuitarSet (+0.0006 / 0
regressions on GAPS). This is **below** the roadmap's hoped strummed +0.01–0.04
and does **not** move the chord-accuracy 0.48→0.85 gap — but it is a real,
rigorously-gated, near-zero-downside win (env-reversible via
`TABVISION_CHORD_SHAPE_BONUS`).

**Recommendation:** ship default `0.0 → 0.1`. Note it re-bases the canonical
val24 **strummed** baseline `0.7951 → 0.7980` (single-line/0.4820 unchanged),
which all future sweeps/gates then reference. *(Default-flip is the user's call —
see DECISIONS 2026-07-07; the committed default remains 0.0 until then.)*

## Reproduce

```bash
cd tabvision   # eval env: venv + TABVISION_DATA_ROOT + ffmpeg on PATH, PYTHONUTF8=1
# sweep (CHORD_SHAPE_BONUS axis):
python -m scripts.eval.a3_fusion_sweep --output ../docs/EVAL_REPORTS/a5_chord_shape_sweep_val24_2026-07-07.md
# gate leg 1 (in-domain 60-clip):
python -m scripts.eval.a3_gate_probe --manifest data/eval/local_guitarset.toml --splits validation \
    --backend highres --position-prior guitarset-v1 --set CHORD_SHAPE_BONUS=0.1
# gate leg 2 (cross-domain GAPS clean-12, strict):
python -m scripts.eval.a3_gate_probe --manifest data/eval/gaps.toml --splits test \
    --backend highres --position-prior none --clean12 --strict-per-clip --set CHORD_SHAPE_BONUS=0.1
```
