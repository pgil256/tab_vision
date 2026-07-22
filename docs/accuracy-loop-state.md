# Accuracy-loop state
last_updated: 2026-07-22
current_branch: accuracy/q2-s1b-probe

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| Q1 | N2 MuScriptor merge | **closed-negative** | best variant ΔTab F1 **-0.0167** [-0.0480, +0.0090]; Δonset **-0.0195** [-0.0325, -0.0079] (CI-sig regression); added-note precision **0.181** vs 0.6855 stream precision | none — closed | — |
| Q2 | S1b-v2 offline probe | **in-progress** | gate re-based: amb-top1 ≥ **0.7048** (from 0.6548 dev-OOF, n=35,959); corpus 34,621 tracks / 34.06M notes | train masked-string model, rescore lattice | — |
| Q3 | S1b-v2 integration | blocked | — | — | Q2 |
| Q4 | Second-opinion probes | open | **two-leg gate now** (see below) | Basic Pitch probe on cached eval audio | — |
| Q5 | Onset snapping | open | — | prototype (must re-run full onset/pitch gates) | — |
| Q6 | Inharmonicity study | open | gates 0.85 hex / 0.70 mono | hex B-estimator | — |
| Q7 | Capo/tuning preflight | open | — | design synthetic-capo eval | — |
| Q8 | Review-ranker upgrade | blocked | beat 38.76% @60s | — | Q3 |

**Topmost open unblocked item = Q2 (in progress, iteration 3 continues it).**

## Q1 — closed 2026-07-21 (bounded negative)

Report: `docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.md` (+ `.json`).
DECISIONS.md 2026-07-21 "Program N2: MuScriptor merge CLOSED".

All six predeclared merge variants lose under the shipped clean-acoustic
decode on 20 dev clips (10 comp = entry-probe slice, 10 new solo), OOF
position prior. Root cause: complementarity charges nothing for false
additions; MuScriptor's admitted notes are only 0.10-0.18 precise against a
0.6855-precision stream, costing ~7-18 new false detections per correct note
gained. No confidence-floor variant is possible (MIDI velocity is constant
100; the API exposes no per-note score). Solo complementarity is 0.1481 vs
comp's 0.3818 — the entry headline was a comp-mode artifact.

The 300-clip dev run was deliberately **not** spent (failure is structural,
not sample-size). Do not re-open without a materially new admission signal.

## Q2 — in progress (iteration 2 of N)

Report: `docs/EVAL_REPORTS/s1b_entry_substrate_2026-07-22.md`
(+ `s1b_symbolic_corpus_2026-07-22.json`). DECISIONS.md 2026-07-22.

**Done (entry substrate):**

- Banked Phase 0 lattice verified as a faithful offline replay substrate —
  reproduces both headlines exactly from CSV (held-out 0.6770 top-1 /
  0.9986 top-3), no audio or pipeline. Sweeps are seconds, $0.
- **Gate re-based off player-05.** The deep-dive's 0.6770 is the
  `held_out_05` slice, which the loop keeps sealed until config freeze +
  user proceed. Working gate is the same +0.05 on dev-OOF:
  **ambiguous top-1 ≥ 0.7048** (from 0.6548, n = 35,959 — 5× the power).
- Miss structure: gold is in the lattice for 99.72% of ambiguous notes;
  **84% of misses are gold-at-rank-2**, so the gate = flip **17% of the
  rank-2 pile** (1,798 notes). Solo 0.5908 vs comp 0.6896.
- Corpus built: 34,621 tracks / 34,063,065 notes / 46.3 MB npz / 279 s,
  reproducing the S1a audit totals exactly (same parse, order preserved).
  88.7% pitch-ambiguous notes; 47% polyphonic clusters.
- `candidate_path` carries `cost_delta_from_best`, so the probe scores the
  real integration shape (blend model log-prob with decoder cost, sweep the
  mixing weight offline) rather than a proxy.

**Next action (iteration 3):** tokenize windows (pitch + cluster structure),
masked-string objective, train a small encoder on CPU/free Colab held out by
track, then rescore the dev-OOF lattice and report ambiguous top-1 vs 0.7048
with the solo/comp split and rank-2 flip rate. Fail → banked negative, close
Q2. Pass → Q3, still stopping before player-05.

## Q4 gate revision (binding, from Q1's carry-forward)

Second-opinion candidates gate on **both** legs, measured in the same
offline replay:

1. P(candidate right | ensemble wrong) ≥ 0.10 — is there anything to gain;
2. **added-note precision ≥ 0.5** under the candidate's best admission
   rule — can the gain be separated from the noise it arrives with.

N2 passed leg 1 by 3.8× and failed leg 2 at 0.18. Leg 2 is free once events
are banked.

## Reusable harness (built for Q1, reusable for Q4)

`tabvision/scripts/eval/n2_muscriptor_merge.py`, two stages:

- `--stage cache` banks per clip: registered `highres-ensemble` AudioEvents
  (`<track>.ensemble.json`) + candidate MIDI (`<track>.<model>.mid`) under
  `$TABVISION_DATA_ROOT/models/muscriptor_probe/`. Resumable; ~50 s/clip
  ensemble + ~40-130 s/clip MuScriptor on this laptop CPU.
- `--stage sweep` is pure offline replay (seconds, no inference): per-mode
  complementarity, added-note precision, six merge variants scored through
  `fuse()` with the leave-one-player-out position prior (`--prior oof` —
  `guitarset-v1` is trained on the dev players) + `guitarset-seq-v1` @ w=4.0,
  paired bootstrap (N=10,000, seed=42) and the six-bucket decomposition.

20 clips are banked on disk, so new admission rules cost zero compute.
Reproduce (from `tabvision/`, `TABVISION_DATA_ROOT=~/.tabvision/data`):

```
python scripts/eval/n2_muscriptor_merge.py --stage sweep \
  --comp-clips 10 --solo-clips 10 --prior oof \
  --output ../docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.md \
  --json ../docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.json
```

## Questions for the user
- None blocking. Q2 (S1b-v2 offline probe) is next and is $0 / local-CPU;
  it needs no new dependency or dataset — the SynthTab JAMS are already on
  disk. If training moves to free Colab that stays inside the pre-approved
  set; anything paid stops for approval.

## Iteration log (newest first)
- 2026-07-22 — Q2 — entry substrate verified + corpus built. Gate re-based
  off the sealed player-05 slice to dev-OOF (0.6548 → target 0.7048);
  84% of misses are rank-2. Corpus 34,621 tracks / 34.06M notes matches the
  S1a audit exactly. `s1b_entry_substrate_2026-07-22.md`.
- 2026-07-21 — Q1 — **closed-negative**. Built the two-stage merge harness,
  banked 20 dev clips, measured solo complementarity 0.1481 (vs comp 0.3818)
  and all six merge variants negative (best -0.0167 Tab F1, CI-sig -0.0195
  onset F1); root cause = added-note precision 0.181.
  `n2_muscriptor_merge_pilot_2026-07-21.md`.
