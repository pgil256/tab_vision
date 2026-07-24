# N2 — nylon table for classical: banked negative

Accuracy-loop N2 (proposed in the program summary). Q6's physics channel
abstains on classical because no nylon table existed, so the GAPS cross-domain
gate was satisfied by *abstention*. N2 built a nylon table and tested whether,
applied to classical audio, it converts that gate into a real measurement.

**It does not. The nylon table is no help on GAPS, so classical keeps
abstaining.**

## Result — GAPS clean-12, classical routing

`gaps-v1` position prior + `gaps-seq-v1`, `instrument="classical"` (gaps
checkpoint), three arms scored on one transcription.

| arm | Tab F1 | Δ vs baseline [lo-95, hi-95] | coverage |
|---|---:|---|---:|
| baseline | 0.7733 | — | — |
| strict | 0.7742 | +0.0009 [−0.0004, +0.0023] | **1.04%** |
| partial_aware | 0.7580 | **−0.0153 [−0.0472, +0.0151]** | 19.04% |

Neither arm is a win:

- **strict is neutral but empty.** +0.0009 is not significant, and coverage is
  **1.04%** — classical is almost entirely polyphonic, so strict isolation
  finds virtually no notes to measure. It is safe because it does nothing.
- **partial_aware gets coverage but the point estimate goes negative**
  (−0.0153), and the per-clip spread shows why: some clips help
  (212_y41wc +0.090, 142_GD1wc +0.004) but others regress hard
  (294_BSswc −0.148, 118_VD1wc −0.068). The CI spans zero, so it is not
  *significantly* harmful — but it is clearly not the +0.0182 partial-aware
  delivered on steel/GuitarSet.

Onset F1 is 0.9510 in every arm, bit-identical — the channel still only
touches string assignment.

## Why it fails, and it was predicted

The nylon table's construction (recorded in `string_physics.py`) split the six
strings honestly:

- **Three trebles** (G3, B3, E4) — plain nylon monofilament, linear mass
  computed from density and gauge. First-principles.
- **Three basses** (E2, A2, D3) — nylon floss core under metal winding. The
  winding mass is essential and the *effective bending core* is ill-defined,
  so the core diameters and unit weights are **documented approximations**.
  Since `B ∝ d_core⁴`, a 20% error in a bass core diameter is a ~2× error in
  its `B`.

Classical repertoire uses the bass strings heavily, and partial-aware
measurement admits exactly the overlapped bass notes whose table entries are
roughest. So the mode that finally gets coverage on classical is the one that
most exposes the weakest rows. The regressions are concentrated where the
physics is least trustworthy, which is the honest failure mode, not a
surprise.

## What ships

**Nothing changes in the pipeline.** The routing was reverted:
`stiffness_model_for_session` still abstains on classical, restoring the
"abstain by construction" invariant that makes the GAPS gate free for the
steel `acoustic-physics-v1` artifact (verified by
`test_out_of_domain_sessions_are_bit_identical_to_baseline`, which depends on
classical → `None`).

**The machinery is kept as a banked negative:**
`classical_stiffness_model()`, `CLASSICAL_NYLON_SET`, the per-string modulus
field on `StringSpec`, and `scripts/eval/n2_nylon_gaps.py`. They are tested and
reachable by direct call, so a future session with **manufacturer bass core
diameters and unit weights** can rebuild the bass rows and re-run the eval
without redoing the trebles or the harness. The steel table already showed the
method works when the specs are good; the missing ingredient here is data, not
physics.

## Honest limits

- 12 clips, one dataset. A larger negative would be more decisive, but the
  point estimate is negative and the mechanism (rough bass rows exposed by
  partial-aware) is legible, so more clips would sharpen a conclusion whose
  sign is not in doubt.
- The trebles may well carry signal; this eval cannot isolate them because
  GAPS has no per-string ground truth. A treble-only variant (abstain on
  strings 0-2) is a possible refinement, but it would be selection on the
  eval set unless pre-registered, and the ROI is low against the bass-data gap.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n2_nylon_gaps.py \
  --json ../docs/EVAL_REPORTS/n2_nylon_gaps_2026-07-23.json
```

Classical GAPS clips are ~4 min each; the ~47-minute transcription caches per
clip and is resumable.
