# Q6 player-05 sealed confirmation — PASS

Accuracy-loop iteration 14 (ROI deep-dive §4.1). The final gate. Player-05 is
the held-out player the program keeps sealed, opened only after config freeze
and an explicit user proceed — both conditions met (full-dev passed +0.0443
with frozen config; the user authorized this run).

**Nothing was tuned.** Weight 1.0, `min_r2` 0.50, raw specification-derived
table — the identical values the full-dev run froze. The position prior is
the **registered `guitarset-v1`**, which excludes player 05 per its manifest,
so the decoder's prior never saw this player. The stiffness table is derived
from string specifications and depends on no player at all.

## Baseline reproduces the shipped production number exactly

| tier | this run's baseline | CLAUDE.md shipped |
|---|---:|---:|
| solo | 0.5503 | 0.5503 |
| comp | 0.7176 | 0.7175 |
| **aggregate** | **0.6340** | **0.6339** |

The harness reproduces the shipped player-05 decode to four decimals before
any evidence is added. That is the control this whole confirmation rests on —
the delta below is measured against the real production baseline, not a
reconstruction of it.

## Result

| metric | baseline | with channel | Δ [lo-95, hi-95] |
|---|---:|---:|---|
| **Tab F1 (aggregate)** | 0.6340 | **0.7119** | **+0.0780 [+0.0502, +0.1078]** |
| Tab F1 (solo) | 0.5503 | **0.6899** | **+0.1396 [+0.0985, +0.1806]** |
| Tab F1 (comp) | 0.7176 | 0.7340 | +0.0164 [+0.0000, +0.0458] |
| onset F1 | 0.9473 | 0.9473 | bit-identical |
| pitch F1 | 0.9386 | 0.9386 | bit-identical |

**Confirmation gate (lo-95 > 0): PASS**, lower bound +0.0502.

Solo improves **+0.1396 absolute — a 25% relative gain on the weakest tier**,
which is where SPEC §1.4.1 has always been furthest from target and where
every other lever in this program failed (Q2's contextual model moved solo by
+0.0112).

## Consistency with the development set

| set | Δ Tab F1 [lo-95, hi-95] | solo Δ |
|---|---|---|
| full dev (300 clips, players 00-04) | +0.0443 [+0.0339, +0.0555] | +0.0860 |
| **player-05 (60 clips, sealed)** | **+0.0780 [+0.0502, +0.1078]** | **+0.1396** |

The hold-out point estimate is **higher** than dev, which is the opposite of
the usual overfitting direction and worth stating plainly rather than
celebrating. The intervals do overlap (dev's upper +0.0555 against
player-05's lower +0.0502), so this is **not** a significant difference — the
honest reading is "consistent with dev, at the optimistic end." Player 05 is
also a somewhat cleaner player (baseline aggregate 0.6340 vs dev's 0.6031)
with slightly higher channel coverage (9.6% vs 8.3% of detections), which
plausibly accounts for the gap without invoking anything more interesting.

## Decomposition: one-for-one, again

| bucket | baseline | with channel | Δ |
|---|---:|---:|---:|
| correct | 5,594 | 5,961 | **+367** |
| wrong_position_same_pitch | 2,300 | 1,933 | **−367** |
| pitch_off | 273 | 273 | 0 |
| timing_only | 80 | 80 | 0 |
| missed_onset | 468 | 468 | 0 |
| extra_detection | 391 | 391 | 0 |

Exactly 367 wrong-position errors become correct notes; the other four
buckets do not move by one event across 8,709 detections. The same signature
as the pilot and the full-dev run, now on data the calibration never touched.

## Per-clip behaviour

**30 improved, 28 unchanged, 2 regressed.** The regression rate on the
hold-out (2/60 = 3.3%) is lower than dev's (25/300 = 8.3%). Coverage is
834 applied of 8,709 detections (9.6%), overwhelmingly solo.

## What this does and does not license

**Does:** the channel is confirmed on sealed data with frozen config, with
the gain provably confined to the targeted bucket and detection metrics
untouched. Every offline gate in the program is now passed.

**Does not:**

- **This is GuitarSet.** The physics table is specification-derived and so
  carries no GuitarSet fitting, but every clip here is still the same corpus,
  recorded on similar steel-string acoustics. Behaviour on a materially
  different guitar is argued from physics, not measured.
- The channel abstains outside clean steel-string acoustic, standard tuning,
  capo 0 (domain guard). Classical, electric, capo and alternate tunings get
  nothing — by construction, not by accident.
- Comp barely moves. A strum-heavy recording will see far less than +0.0780.
- The calibration take (`calibrate_from_ritual`) remains unvalidated on real
  plucks; the physics table does not depend on it.

## Remaining decision (user)

Registration and routing:

1. **Register** the channel as a `string_evidence` artifact with its manifest
   (config, gate provenance, this report), or leave it unregistered.
2. **`auto` routing** — default-on for the clean steel-string acoustic domain
   it is gated to, or keep it opt-in behind `TABVISION_STRING_EVIDENCE`.

Promotion into `auto` is a user decision per SPEC §0.8 and the loop's stop
rules; nothing in this iteration changed default behaviour.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q6_player05_confirm.py \
  --json ../docs/EVAL_REPORTS/q6_player05_confirm_2026-07-22.json
```

Config frozen in source. Ensemble events cache to
`$TABVISION_DATA_ROOT/models/q6_player05_cache/`; the run is resumable (this
one was, after the first attempt was killed at clip 50 by a session teardown —
no work lost, and the resumed run reproduced the same running means clip for
clip).
