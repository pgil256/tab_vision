# F9 — nut-anchored translation correction (design, pre-registered)

**Status: DESIGN — awaiting sign-off.** Scoped fix for the 031 barre
boundary regression (loop-state F9; DECISIONS 2026-07-29 correction).

## 1. Verified diagnosis

Instrumented replay of `dev_031_barre_i` on the current encode
(`f9_probe.py` / `f9_probe2.py`, scratchpad):

- Barre classification is healthy: `barre=True`, `contact_source=barre_axis`,
  curl 0.95–0.97, on every scored frame.
- The calibrated fret map is ideal: wires descend from canonical 1.0 (nut)
  with gap ratios exactly 0.944 (rule of 18) — no phantom wires, no
  distortion.
- The failure is a **coherent translation of the whole hand's canonical
  projection**: the four fingers of the labeled Position-I chord read cells
  2/3/4/5 (canonical x 0.871/0.800/0.722/0.630) instead of 1/2/3/4 — every
  finger ~50% past its wire, where the barre deadband cliff
  (`FRET_WIRE_DEADBAND_FRACTION = 0.35`) flips it. June's recorded values
  (tip-x 1.656–1.977 → Position I via F4d) imply the same content projected
  ~0.15–0.4 fret-units lower; the re-downloaded video variant shifted the
  projection. The June bytes are gone; the shift's origin (quad fit vs
  landmark placement on new pixels) cannot be decomposed further and does
  not need to be.

## 2. Rejected fixes (and why, recorded to prevent re-litigating)

- **Raise the deadband** (0.35 → 0.5): threshold fishing on the exact
  number that failed; today's values sit at ~0.50, so it just moves the
  cliff under the measurement.
- **Nearest-wire snapping for barres**: today's axis is within 0.001 of the
  exact cell midpoint — nearest-wire is the same knife edge renamed.
- Both fail the wire-sparse lesson: don't re-tune the statistic that broke;
  anchor to independent physical evidence.

## 3. The fix: correct translation from the detected nut

The nut is the strongest landmark on the neck, it is largest and most
reliable in exactly the close framing where this failure occurs, and its
canonical position is known by construction (wire 0 of the fitted map).

Mechanism: per frame, when the nut is confidently detected, compute
`delta = canonical_nut_position(map) - canonical_nut_position(detected)`
and add `delta` to contact canonical-x values before wire-cell
classification.

Gates (all required; abstain from correcting otherwise — "sees a little is
not safe"):
- nut detection present with confidence above the existing detector floor;
- `|delta|` ≤ 0.6 × the local first-cell width (a translation larger than
  half a cell is evidence of a broken fit, not a correctable bias — do not
  "correct" by more than the ambiguity we are resolving);
- the correction applies to contact classification only — never to the
  homography, the fret map, the anchor, or the FretCam window output
  (position windows are ±-tolerant by design; contacts are the knife-edge
  consumer).

Implementation verification step 0: reconcile the `nut_x` field's
convention (probe 1 read 0.03–0.05 while this clip's map places the nut at
canonical 1.0 — establish whether `nut_x` is orientation-normalized or a
different quantity before wiring anything).

## 4. Pre-registered acceptance (frozen before implementation)

On the hybrid dev cache, full dev split, benchmark harness unchanged:
1. `dev_031_barre_i`: ≥ 90% of its stable frames correct (currently 0/55).
2. The five other sequences: per-sequence correct/wrong counts unchanged
   within ±1 frame each (the fix must not move healthy sequences).
3. Negative-control displays 0/120, both with and without the F6 fallback.
4. Full fretcam suite green; new unit tests cover the gates in §3
   (correction applied; abstained when nut absent; abstained when
   |delta| exceeds the cap).
5. Re-read the F6 ON arm on the fixed solver — it must remain
   non-degrading vs the fixed baseline.
Any outcome outside these lines is banked as a negative and the fix is
reverted; no post-hoc gate edits.

## 5. Cost and scope

Prototype-only (`fretcam/src/fretcam/detection.py` + tests), ~30–60 lines,
$0, no training, no schema or bridge changes, no tabvision/ changes.
