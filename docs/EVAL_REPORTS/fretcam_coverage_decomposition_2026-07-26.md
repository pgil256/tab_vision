# FretCam position accuracy — coverage decomposition and four fixes

**Date:** 2026-07-26
**Scope:** `fretcam/` only. No dependency, model, download, training run, private
recording, §8 contract, or `tabvision/` package behaviour changed.
**Benchmark:** the frozen F4e-A public position benchmark
(`fretcam/data/position_benchmark_v1.json`, GAPS CC-BY-NC-SA-4.0, public footage
only, 16 sequences / 12 sources, 10 FPS, 450 samples).

## 1. The problem is not accuracy, it is silence

The shipped build's displayed positions are essentially never wrong. Re-running
the frozen benchmark on current `main`:

| split | stable frames | valid observations | coverage | displayed precision | false-lock rate |
|---|---:|---:|---:|---:|---:|
| dev | 161 | 0.6770 | 0.4161 | 1.000 | 0.0000 |
| test | 115 | 0.4087 | 0.0696 | 1.000 | 0.0000 |
| all | 276 | 0.5652 | 0.2717 | 1.000 | 0.0000 |

Negative-control display rate 0.000. **Every displayed position was correct**;
the system simply declines to display one on 73% of stable frames. So the
accuracy lever is coverage-at-fixed-precision, not error correction.

Coverage decomposes into two independent losses:

- **observation loss** — only 0.565 of stable frames yield a usable position
  observation at all;
- **lock loss** — of those, only ~0.48 ever reach the display, because the
  temporal estimator multiplies observation confidence by temporal agreement,
  which is itself depressed by the observation loss. The two losses compound.

Position III shows the second loss in isolation: 0.600 of its frames produce a
valid observation and **0.000** are ever displayed.

## 2. Where the observation loss comes from

`solve_hand_position` records why each frame failed. Tabulating those blockers
over stable frames with no valid observation:

| blocker | dev | test |
|---|---:|---:|
| `no_hand` | 59.6% | 67.6% |
| `too_few_contacts` | 40.4% | 32.4% |
| `no_board` | 0% | 0% |

Board geometry is **not** the bottleneck. Mean board confidence is 0.88–0.92
even on failing frames, and `geometry_status` is `tracked`/`detected` on every
stable frame in both splits. This retires the standing assumption that fret/neck
detection is what limits the HUD.

### 2.1 `no_hand` is FretCam's search strategy, not MediaPipe's recall

On the 77 stable frames the chain reports as `no_hand`, an uncropped
still-image MediaPipe pass was re-run at three detection thresholds:

| search | any hand found | ≥3 fingertips on the neck |
|---|---:|---:|
| full frame @ 0.50 (shipped threshold) | **1.000** | **0.610** |
| full frame @ 0.30 | 1.000 | 0.610 |
| full frame @ 0.15 | 1.000 | 0.610 |
| upscaled neck crop @ 0.50 | 0.909 | 0.481 |
| upscaled neck crop @ 0.15 | 0.987 | 0.520 |

Three conclusions:

1. MediaPipe finds a hand on **every** frame the product calls `no_hand`.
   Lowering the detection threshold buys nothing — recall is not the limit.
2. On 61% of those frames the recovered hand already satisfies
   `MIN_ON_NECK_FINGERTIPS`, so it would have been accepted.
3. **The neck crop is the weaker search** (0.481 vs 0.610). At this framing the
   crop removes the body context the palm detector uses, and the ≤2.5× upscale
   does not compensate.

Of the 47 recoverable frames, 36 were searched via a crop path, and the chain's
failure paths then *deferred* recovery to the next frame
(`neck_recovery_pending`, `full_recovery_pending`) — spending a whole frame to
find a hand that was visible in the one just processed.

### 2.2 `geometry_stability` is anti-correlated with the outcome it gates

`OpticalBoardTracker` scored flow quality as `ratio · exp(−error_px / 2.0)` — a
**fixed 2-pixel** scale. Measured on valid stable frames the factor has median
**0.113** and mean 0.289, on boards the tracker itself calls healthy. Worse, its
mean is *higher* on failing frames than on passing ones:

| factor | valid frames | invalid frames |
|---|---:|---:|
| `stability` (dev) | 0.294 | 0.504 |
| `stability` (test) | 0.276 | 0.362 |

It is measuring "is textured content moving in the frame" — which is maximised
exactly when a hand is working over the neck — not "has the board geometry
slipped". It enters the confidence geometric mean at exponent 0.15 and costs a
median **1.39×** on observation confidence, pushing the fraction of valid frames
clearing the estimator's 0.20 gate from 0.692 down to 0.603.

A fixed pixel scale cannot express the intended quantity: 3 px is a quarter of a
fret on a close-framed neck and two frets on a distant one.

### 2.3 The contact-support gate is calibrated for roughly three fingers

`support = clip(total_weight / 1.5)` with a `support < 0.35` blocker. Measured
over the benchmark:

- a *useful* contact carries a median weight of **0.571** (n = 258);
- **72%** of frames with any useful contact have **exactly one** (132 of 183);
- median total weight per frame is 0.629.

Each contact's weight is already a product of four [0, 1] terms, so a normaliser
of 1.5 demands roughly three strong contacts before any position may lock. This
is why barre — the one technique with a hard-coded `support = max(support,
0.85)` bypass — covered 0.75 of its frames while chord and note covered 0.08 and
0.12.

## 3. Changes

All four are in `fretcam/`; none adds a dependency, model, or training run.

| # | file | change |
|---|---|---|
| A | `tracking.py` | Scale the flow-residual stability term by 3% of the tracked neck's projected long edge (floor 2 px) instead of a fixed 2 px. Under rule-of-18 spacing the narrowest nut-to-fret-12 cell is ~6% of that edge, so the scale is about half a fret at the tightest end — the point where a residual can actually move a contact into the wrong cell. |
| B | `detection.py` | Add a reserved third per-frame MediaPipe call that a would-be-empty frame may spend on an uncropped still-image recovery pass, wired into the four paths that previously deferred to the next frame. Healthy frames still cost two calls. |
| C | `detection.py` | `CONTACT_SUPPORT_NORMALIZER = 0.9`, derived from the measured weight distribution: two typical contacts saturate support, one typical contact clears the sufficiency gate. |
| D | `position.py` | Track `_departed_position` separately from `_transition_active`: once evidence has pointed *away* from the held position, one agreeing frame may not undo the departure — it must re-acquire. A merely *rejected* spike is deliberately excluded, preserving the F4b isolated-spike guarantee. |

## 4. Result

Dev split (the tuning surface):

| build | valid obs | coverage | displayed precision | false-lock | negative control |
|---|---:|---:|---:|---:|---:|
| shipped | 0.6770 | 0.4161 | 1.000 | 0.0000 | 0.0000 |
| A+B | 0.7081 | 0.4721 | 0.9529 | 0.0000 | 0.0000 |
| **A+B+C+D** | **0.7267** | **0.4783** | 0.9535 | **0.0000** | **0.0000** |

Dev coverage by category — the gains land exactly where the losses were:

| category | shipped | fixed |
|---|---:|---:|
| chord | 0.123 (7/57) | **0.228** (13/57) |
| note | 0.280 (14/50) | **0.340** (17/50) |
| barre | 0.852 (46/54) | 0.870 (47/54) |
| full-neck framing | 0.196 (21/107) | **0.280** (30/107) |
| close framing | 0.852 (46/54) | 0.870 (47/54) |
| position II | 0.159 (7/44) | **0.295** (13/44) |
| position V | 0.533 (8/15) | **0.733** (11/15) |
| positions III / VI / IX | 0.000 | 0.000 |

Change C in isolation. Replaying the post-change dev observation stream through
the offline harness while varying only `CONTACT_SUPPORT_NORMALIZER` shows the
gate was simply too tight, with no dev-visible precision cost anywhere on the
curve:

| normaliser | dev coverage | dev precision | dev false locks |
|---:|---:|---:|---:|
| 1.5 (shipped) | 0.4348 | 0.9375 | 0 |
| 1.2 | 0.4472 | 0.9390 | 0 |
| 1.0 | 0.4596 | 0.9405 | 0 |
| **0.9 (chosen)** | **0.4783** | 0.9425 | 0 |
| 0.8 | 0.4845 | 0.9432 | 0 |
| 0.6 | 0.4907 | 0.9444 | 0 |

0.9 is the *derived* value — two typical contacts saturate — not the value that
maximises dev coverage. Lower normalisers keep gaining on dev, but the test
split's single adjacent-position error (§5) is the failure mode they amplify, so
the derived anchor is the defensible stopping point.

Held-out test split, opened once for this iteration:

| build | valid obs | coverage | displayed precision | false-lock | negative control |
|---|---:|---:|---:|---:|---:|
| shipped | 0.4087 | 0.0696 | 1.000 | 0.0000 | 0.0000 |
| fixed | 0.3739 | **0.1130** | 0.8889 | 0.0174 | 0.0000 |

Two test sequences that previously displayed **nothing at all** now display:
`test_178_ii_to_v` 0/22 → 2/22 and `test_179_chord_i` 0/15 → 3/15.

Also newly measurable: dropout display rate rose 0.294 → 0.588 and the shifting
state now yields a measured recovery, because positions lock often enough for
transition timing to exist at all.

## 5. What it costs

The change is **not free**. Every wrong display in the fixed build is one of two
localised events:

- **dev, 4 frames** (`dev_104_ii_to_vi`, t = 15.9–16.2): the origin position
  re-locks from one late agreeing frame after a rejected jump, then rides the
  0.5 s dropout hold into a labelled shift. Change D removes this on the
  pre-change observation stream but not on the post-change one: with the
  recovery pass supplying more evidence, the re-lock arrives through the
  *rejected-jump* path, which by F4b's design must **not** count as a departure.
  Closing this requires distinguishing "spike from noise" from "spike during
  real motion" — a real piece of work, not a threshold change.
- **test, 2 frames** (`test_178_ii_to_v`, t = 19.2–19.3): a single **adjacent**
  position error (predicted IV, truth V) on a single-note passage. This is the
  expected failure mode of relaxing the support gate: with one contact and no
  cross-finger corroboration, a half-fret bias in the contact coordinate flips
  the position by one.

Negative-control display rate stays **0.000** in every build — the strongest
safety property is untouched.

Both splits are small (161 / 115 stable frames). Neither the +5 covered test
frames nor the 2 false locks is individually significant; the dev direction
(+10 covered frames at zero false locks) is the stronger signal.

## 6. Measurement-hygiene disclosure

Threshold and formula selection used the **dev** split plus the negative
controls. However, an early screening sweep printed the test column before it
was removed from the output, and seeing it contributed to rejecting a blanket
`support_floor = 0.5` variant in favour of the derived normaliser. The test
figures above are therefore an **audit** number, not a clean held-out
measurement for this iteration. A future iteration should re-freeze a held-out
portion before claiming generalisation.

## 7. Open leads, in rough order of expected value

1. **Positions III, VI and IX still never display** (0.000 coverage) despite
   III producing valid observations on 0.600 of its frames. This is a pure
   lock-loss failure and is now the single largest remaining block.
2. **The calibrated fret map is absent on 51% of frames** (135/276
   `fret_map_locked`), so the chain runs on the `rule18_fret12_fallback` axis
   half the time. It has not produced a wrong display yet, but it is an
   unmodelled fret-numbering risk — and it is the one place where better
   fret-wire detection would actually pay.
3. **Detector inference scale — measured, and it is the lever on lead 2.**
   `YoloOBBBackend` runs `model.predict(frame)` at ultralytics' default
   `imgsz=640`, and `HudFrameProcessor` caps live frames at 640×480, so fret
   wires are never detected above native resolution. Holding the checkpoint
   fixed and varying only the inference scale over 54 dev stable frames:

   | framing | imgsz | neck detected | mean fret detections | ≥4 wires | fret map fitted |
   |---|---:|---:|---:|---:|---:|
   | full-neck (36) | **640** | 1.000 | **6.56** | 0.389 | 0.333 |
   | full-neck | 960 | 1.000 | **14.42** | **0.861** | 0.556 |
   | full-neck | 1280 | 0.972 | 14.31 | 0.861 | **0.667** |
   | close (18) | **640** | 1.000 | 14.44 | 1.000 | **1.000** |
   | close | 960 | 0.889 | 14.67 | 1.000 | 0.889 |
   | close | 1280 | 0.611 | 10.11 | 1.000 | **0.111** |

   At full-neck framing — two thirds of the benchmark, and the framing where
   coverage is worst — going from 640 to 960 **more than doubles** fret-wire
   detections and lifts the fitted-fret-map rate from 0.333 to 0.556 (0.667 at
   1280). At close framing the same change is actively harmful: the **neck**
   OBB itself starts being missed (1.000 → 0.611 at 1280) once a close neck is
   upscaled well past the detector's training scale, collapsing the fret map to
   0.111.

   So the fix is a **scale-adaptive** inference size — upscale when the detected
   neck is small in frame, stay native when it fills it — not a fixed larger
   `imgsz`. This is the clearest remaining path to lead 2. **It was not
   implemented here**: `imgsz` would have to be plumbed through
   `YoloOBBBackend`, which lives in `tabvision/`, and FretCam accuracy work is
   scoped to `fretcam/` unless separately approved.
4. **Rejected-jump vs real-motion disambiguation**, which closes the dev
   regression in §5.
5. **The contact classifier admits one finger per frame 72% of the time.** The
   support gate was recalibrated to match that reality; the alternative is to
   attack the `pressing` classification so chords actually register multiple
   contacts. That is the higher-ceiling fix and remains unexplored.

## 8. Verification

- `275 passed, 1 skipped` in the FretCam suite (up from 273; two tests were
  rewritten, not removed, because they encoded the crop-defer behaviour that
  change B replaces — the per-frame work bound is now asserted as
  "two calls plus at most one reserve").
- `ruff check` and `ruff format --check` pass on all changed files. One
  pre-existing `I001` import-order finding in `detection.py` was left untouched.
- The offline replay harness used for screening reproduces the shipped build's
  `observation_confidence` and `position_fret` on **450/450** frames before any
  variant was scored.

## 9. Reproduce

```bash
./fretcam/.venv/Scripts/python.exe -m fretcam.position_benchmark --split all --output-json <machine-local-output.json>
```
