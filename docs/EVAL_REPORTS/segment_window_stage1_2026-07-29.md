# Segment-level position-window fusion — Stage 1 ceiling probe

**Gate G1: FAIL. Mean aggregate Tab F1 delta +0.0000 on GAPS clean-12,
0 of 12 clips reranked.** Per the design's §5 decision tree this banks the
negative and closes the line. Stage 2 (real FretCam observations) is not run.

Every constant was frozen in
`docs/plans/2026-07-28-segment-position-window-design.md` §5a and committed
(`d6c4c89`) **before** this script was first run.

## Result

| arm | aggregate Tab F1 |
|---|---:|
| segment-v1 baseline (top-1 path) | 0.766423 |
| gold-window reranked | 0.766423 |
| **delta** | **+0.0000 [+0.0000, +0.0000]** |

G1 required ≥ +0.010 with no per-clip regression worse than −0.002. The
delta is exactly zero because **the reranker abstained on all twelve
clips** — not because it chose badly.

| clip | notes | segments | windows | (seg, obs) pairs | distinct assignments | notes a rerank could reach |
|---|---:|---:|---:|---:|---:|---:|
| 027_Zpswc | 1569 | 133 | 514 | 454 | 1 | 0 |
| 031_vpswc | 887 | 75 | 220 | 183 | 1 | 0 |
| 043_bc1wc | 1383 | 115 | 404 | 358 | 1 | 0 |
| 063_bV1wc | 837 | 57 | 228 | 197 | 1 | 0 |
| 104_xf1wc | 420 | 69 | 174 | 131 | 1 | 0 |
| 118_VD1wc | 725 | 29 | 187 | 178 | 1 | 0 |
| 142_GD1wc | 692 | 64 | 275 | 252 | 1 | 0 |
| 179_pM1wc | 517 | 71 | 239 | 198 | 1 | 0 |
| 212_y41wc | 976 | 77 | 483 | 458 | 2 | 2 |
| 235_Ny1wc | 1513 | 124 | 567 | 503 | 1 | 0 |
| 294_BSswc | 463 | 34 | 155 | 138 | 1 | 0 |
| 341_1M1wc | 667 | 28 | 143 | 137 | 1 | 0 |

## The oracle fired; there was nothing to choose between

The evidence channel was fully populated. Of an 11,028-tick candidate grid,
2,295 ticks carried no fretted gold note and 92 spanned more than one window
could cover; of the 8,641 eligible, 3,589 survived the frozen 0.416 coverage
degradation (measured retention 0.4153) and **all 3,589 passed the
production validity contract**, contributing 3,187 (segment, observation)
pairs. Precision was 1.0 by construction.

**The paths the reranker was asked to choose among were the same tab.** On
11 of 12 clips all three retained paths carry byte-identical string/fret
assignments; on the twelfth they differ on 2 notes out of 976. The decoder's
K-best runs over the product space of *latent hand state × chord state*, but
only the chord-state half reaches the emitted `TabEvent`. Two paths can
differ in cost and in hand-state label while assigning every note
identically — and several do so at cost delta exactly 0.0000, i.e. exact
ties between hand-state relabelings of one tab.

## Headroom: the ceiling for *any* reranker over these paths

`scripts.eval.segment_window_headroom` picks the best of the k retained
paths **with gold in hand** — an upper bound no evidence channel can beat:

| k | distinct assignments per clip | clips with any alternative | oracle best-of-k Tab F1 | oracle gain |
|---:|---:|---:|---:|---:|
| 3 (the design's k) | 1.083 | 1 / 12 | 0.766510 | **+0.000087** |
| 10 | 1.250 | 3 / 12 | 0.766685 | +0.000262 |
| 25 | 1.667 | 3 / 12 | 0.766808 | +0.000385 |

At the specified k=3 a perfect oracle wins **+0.000087**, about **115× below
the +0.010 gate**. Widening to k=25 — eight times the retained set, at real
decode cost — still only reaches +0.000385, **26× below the gate**. The
mechanism's ceiling is not low because the windows are weak; it is low
because the candidate set contains almost no alternative tabs.

## What this refutes

The design's §2 claim was that segment-level aggregation could "break
segment-path ties that per-note bonuses cannot… separate retained candidate
paths whose margins average ~0.18 nats." Those margins are real, but on this
corpus **they separate latent hand-state hypotheses, not tabs**. Reranking
them cannot move Tab F1 no matter how good the evidence is, which is why the
per-note bridge's +0.000836 and this mechanism's +0.0000 are both consistent
with the same underlying fact.

This also means Stage 1 did **not** test the strength of position-window
evidence. It tested — and refuted — the assumption that the retained path
set offers anything to choose between. That is a stronger and cheaper
negative than the design anticipated, and it applies to any reranker over
this decoder's K-best, not just a vision-driven one.

## Honest limits

- **The gold-window oracle is generous and still changed nothing.** It sees
  the hand 0.35 s ahead, has precision 1.0, and never drifts. Real FretCam
  observations can only do worse, so Stage 2 cannot rescue the mechanism.
- **A decoder whose K-best deduplicated by assignment is a different
  experiment** and is not registered here. The k=25 column is the best
  available estimate of its ceiling (+0.000385) and does not motivate one,
  but it is a diagnostic, not a gate — it was added after the frozen
  constants and is reported as context, never as a pass/fail.
- **Session routing caveat.** `segment-v1` is admitted only by
  `_automatic_acoustic_domain`, so this ran under q6's acoustic/clean
  session, not a session tagged classical/nylon — where the decoder abstains
  to `baseline` and the delta is exactly zero by construction. The finding
  concerns the decoder's path set, which is unaffected by that routing.
- **Corpus.** GAPS clean-12 is classical fingerstyle. Whether GuitarSet's
  strummed material yields more assignment-distinct K-best paths is
  untested; Phase 1 reported a larger mean second-path margin there (0.1826
  nats) than the margins seen here, several of which are exactly 0.0000.

## Reproduction

```bash
TABVISION_DATA_ROOT=~/.tabvision/data python -m scripts.eval.segment_window_stage1 --json ../docs/EVAL_REPORTS/segment_window_stage1_2026-07-29.json
```

```bash
TABVISION_DATA_ROOT=~/.tabvision/data python -m scripts.eval.segment_window_headroom --k 3 10 25 --json ../docs/EVAL_REPORTS/segment_window_headroom_2026-07-29.json
```

Audio events are `highres-ensemble`, cached once per clip at
`$TABVISION_DATA_ROOT/models/q6_gaps_cache/{clip}.ensemble.json` (90 min CPU
build, 2.7× realtime) and shared byte-identically by both arms. Priors are
`gaps-v1` / `gaps-seq-v1`. Stage 1 decode: 55.7 s; headroom sweep: 445.9 s.
No training, no GPU, no new dependencies, $0. Per-clip records including
observation statistics, path margins and raw agreement scores are in the two
JSON files above.
