# Q4 second-opinion bench — infrastructure + leg-2 calibration

Accuracy-loop iteration 4 (ROI deep-dive §3.3). Two outcomes: the Basic Pitch
probe is **blocked on this machine's Python**, and the gate leg invented in
Q1 is now **derived rather than guessed** — and it validates.

## 1. Basic Pitch — BLOCKED (environment, not evidence)

`basic-pitch>=0.3.0` is a declared optional extra (`audio-baseline` in
`pyproject.toml`), Apache-2.0, already mapped in LICENSES.md — so it is not a
new dependency in the stop-and-ask sense. It does not install here:

- This machine has **only Python 3.12** (`py -0p`: `3.12` sole entry).
- Every `basic-pitch` release (0.3.0 → 0.4.0) resolves to a dependency set
  with no 3.12 wheel path; pip falls back to building **numpy from source**,
  whose `setup.py` calls `pkgutil.ImpImporter` — removed in Python 3.12.
  `pyproject.toml` already carries the warning: *"the verified Linux fixture
  smoke uses Python 3.11."*
- The `[onnx]` extra fails the same way.

Installing a second Python runtime is a system-level dependency, so it stops
for a user decision rather than being taken inside the loop.

**Recommendation: drop Basic Pitch rather than install Python 3.11.** The
prior is poor and now quantified. Its published GuitarSet zero-shot note F1
is **66.1** against our ensemble's 0.9491 onset / 0.9403 pitch, so it is a
much weaker transcriber than MuScriptor — and MuScriptor cleared leg 1 by
3.8× and still failed leg 2 at 0.181 against a break-even of 0.528 (§2). The
deep-dive itself prices this whole row at **+0.00 – +0.02**. Spending a
runtime install to test a weaker model against a bar a stronger one missed by
3× is poor value.

## 2. The leg-2 threshold, derived

Q1 set "added-note precision ≥ 0.5" by judgement. Here is the algebra.

A merge admits `a` notes, a fraction `p` of which are real notes the ensemble
missed. Each real one converts a false negative into a true positive **only
if** the decoder also puts it on the right string — probability `α`. The rest
become false positives. With `D = 2TP + FP + FN` and `F1 = 2TP/D`:

```
TP' = TP + α·p·a        FN' = FN − p·a
FP' = FP + a − α·p·a     D'  = D + a·(1 + α·p − p)
```

Requiring `F1' > F1` and substituting `TP/D = F1/2`:

> **p > (F1/2) / (α·(1 − F1/2) + F1/2)**

Two things fall out. The volume `a` cancels — **how many notes you admit
never changes the sign, only the magnitude**, which is exactly why every one
of N2's six variants was negative and why the conservative ones merely lost
less. And the bar *rises* with the stream's own F1: the better the
transcription you are adding to, the purer an addition has to be.

### Calibration against the banked N2 sweep

`α` is measured, not assumed: real notes admitted (`added_true_notes`) versus
the rise in the decomposition's `correct` bucket.

| | value |
|---|---:|
| baseline Tab F1 (20-clip pilot) | 0.6773 |
| measured α (added real note → tab-correct) | **0.4581** |
| **break-even added-note precision** | **0.5278** |
| (same at a hypothetical α = 0.65) | 0.4406 |

| variant | added precision | break-even | predicted | observed ΔTab F1 |
|---|---:|---:|---|---:|
| `union` | 0.104 | 0.553 | negative | −0.0541 |
| `union-dur60` | 0.103 | 0.558 | negative | −0.0538 |
| `near80` | 0.166 | 0.464 | negative | −0.0234 |
| `cluster` | 0.181 | 0.538 | negative | −0.0167 |
| `cluster-dur60` | 0.181 | 0.538 | negative | −0.0167 |

**Sign agreement: 5/5.** The Q1 threshold of 0.5 sits just under the derived
0.528 — close enough that the gate's verdicts do not change, but it is now a
derived quantity with a stated dependence on `F1` and `α` rather than a
round number.

**Consequence for the bench:** leg 2 is no longer a fixed 0.5. It should be
computed per candidate from the stream it is joining:
`breakeven_precision(baseline_f1, alpha)`. On today's ensemble that is
**0.528**; if the ensemble improves, the bar for any future second opinion
rises with it.

## 3. What was built

- `scripts/eval/q4_second_opinion_probe.py` — the standing bench. Runs a
  candidate in its **own probe venv** (the N2 pattern: Basic Pitch pulls
  TensorFlow, and putting that in the shared eval venv would risk the working
  torch/numpy stack for a throwaway probe), banks its events beside the
  ensemble cache, and reports both gate legs plus the full merge-variant
  sweep, six-bucket decomposition and paired bootstrap. Reuses the N2
  machinery wholesale; a new candidate costs only its own inference.
- `scripts/eval/q4_breakeven_precision.py` — the calibration above.

The bench is untested end-to-end because no candidate would install; its
merge/scoring path is the one already exercised by the N2 pilot.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q4_breakeven_precision.py \
  --pilot-json ../docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.json \
  --json ../docs/EVAL_REPORTS/q4_breakeven_precision_2026-07-22.json
```

The bench itself, once a candidate is installable:

```
python scripts/eval/q4_second_opinion_probe.py --candidate <name> \
  --output ../docs/EVAL_REPORTS/<name>_bench.md --json <...>.json
```
