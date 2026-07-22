# Accuracy-loop state
last_updated: 2026-07-22
current_branch: accuracy/q2-s1b-probe

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| Q1 | N2 MuScriptor merge | **closed-negative** | best variant ΔTab F1 **-0.0167** [-0.0480, +0.0090]; Δonset **-0.0195** [-0.0325, -0.0079] (CI-sig regression); added-note precision **0.181** vs 0.6855 stream precision | none — closed | — |
| Q2 | S1b-v2 offline probe | **closed-negative** | full recipe: top-1 **0.7015**, Δ **+0.0467** [+0.0291, +0.0640] vs target 0.7048 — short by 0.0033; comp +0.0661 vs solo +0.0112 | none — closed | — |
| Q3 | S1b-v2 integration | **dropped** | — | — | Q2 closed-negative |
| Q4 | Second-opinion probes | open | **two-leg gate now** (see below) | Basic Pitch probe on cached eval audio | — |
| Q5 | Onset snapping | open | — | prototype (must re-run full onset/pitch gates) | — |
| Q6 | Inharmonicity study | open | gates 0.85 hex / 0.70 mono | hex B-estimator | — |
| Q7 | Capo/tuning preflight | open | — | design synthetic-capo eval | — |
| Q8 | Review-ranker upgrade | **unblocked-but-orphaned** | beat 38.76% @60s | needs a posterior source; Q3 dropped, so re-scope or drop | Q3 dropped |

**Topmost open unblocked item = Q4 (second-opinion probes: Basic Pitch, then YourMT3+).**

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

## Q2 — CLOSED 2026-07-22 (banked negative, full recipe run)

Report: `docs/EVAL_REPORTS/s1b_context_probe_2026-07-22.md` (+ 3 JSON).
DECISIONS.md 2026-07-22 (second entry).

- 413,958-param transformer, 3.84M notes, pitch+gap inputs only (never sees
  a string, cannot copy the answer). In-domain val ambiguous acc **0.7679**.
- Rescoring the banked lattice as an emission term
  (`cost_delta + λ·(−log p)`): peak λ=4 → top-1 **0.6850**,
  **Δ +0.0302 [+0.0163, +0.0446]** (paired bootstrap over tracks,
  N=10,000, seed 42). Target 0.7048 → **FAIL, CI excludes the bar**.
- **`marginal` control (same corpus, counts only) is negative at every λ** and
  collapses to 0.5419 at λ=∞. So context is the active ingredient, not the
  corpus — and it independently replicates S1a via a different mechanism.
  First CI-significant positive from a SynthTab-derived artifact in this repo.
- λ sweep is smooth/unimodal; model-only (0.6496) is *worse* than the decoder,
  i.e. complementary evidence, exactly the §3.2 emission shape → a tuned λ
  ports directly to Q3. Comp peaks at λ=4 (+0.0355), solo at λ=8 (+0.0287).
**Stage 2 (fine-tune) ran on user instruction and Q2 closes:**

- Leave-one-player-out fine-tune (each dev player scored only by the fold
  that never saw it): ambiguous top-1 **0.7015**, Δ **+0.0467
  [+0.0291, +0.0640]** vs the 0.7048 gate — **short by 0.0033** on the point
  estimate, far short on the lower bound. Fine-tune contributed +0.0165,
  under half the published +4.0 pp.
- **λ was selected on the reported slice** (best of nine on the same dev-OOF
  lattice), so +0.0467 is optimistic by an uncorrected amount.
- **Key finding — context helps chords, not single lines:** comp
  0.6896 → **0.7557 (+0.0661)**, solo 0.5908 → 0.6020 (**+0.0112**), a 6×
  asymmetry. Wrong-position is 57.3% of all loss but 77.5% of *single-line*
  loss, so the tier needing this most got almost none. A chord voicing
  constrains its own members; a single line needs hand-position continuity
  over time, which `guitarset-seq-v1` already models.
- **Re-opening guidance:** target single-line disambiguation specifically,
  not a general contextual model; budget a λ-selection protocol that does not
  touch the reported slice. Masked-string / autoregressive variants are a
  different model needing their own entry gate.
- Q3 (integration) is **dropped** with Q2; Q8 depended on Q3's posteriors and
  is now orphaned — re-scope or drop when it comes up.

Harness notes: `s1b_rescore_lattice.py` refuses `--split held_out_05` —
player-05 stays sealed until config freeze + explicit proceed. The lattice
CSV (70 MB) is **git-ignored** and exists only in the main working tree, so
pass `--lattice` explicitly when running from a git worktree.

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

None blocking. Q2 closed negative after the full recipe; Q4 is next and is
$0 / local CPU on already-cached eval audio.

Note: iterations 1-3 ran on branches `accuracy/q1-n2-merge` and
`accuracy/q2-s1b-probe`, stacked in that order (the state file is a running
document, so Q2 builds on Q1). They are unpushed and must be merged in
order. A parallel session moved the shared working tree onto
`fretcam/f2-detection-chain`; iteration 3 was committed via a temporary git
worktree so that checkout was left undisturbed.

## Iteration log (newest first)
- 2026-07-22 — Q2 — **CLOSED negative** after stage 2 (LOPO fine-tune):
  top-1 0.7015, Δ +0.0467 [+0.0291, +0.0640] vs 0.7048 — short by 0.0033.
  comp +0.0661 vs solo +0.0112 → context helps chords, not single lines.
  `s1b_context_probe_2026-07-22.md` (stage 2 section).
- 2026-07-22 — Q2 — iteration 3: **gate FAILED** at top-1 0.6850,
  Δ +0.0302 [+0.0163, +0.0446] vs target 0.7048. Marginal control negative
  at every λ → context is the active ingredient. Paused on a user call
  (run recipe stage 2, or bank). `s1b_context_probe_2026-07-22.md`.
- 2026-07-22 — Q2 — entry substrate verified + corpus built. Gate re-based
  off the sealed player-05 slice to dev-OOF (0.6548 → target 0.7048);
  84% of misses are rank-2. Corpus 34,621 tracks / 34.06M notes matches the
  S1a audit exactly. `s1b_entry_substrate_2026-07-22.md`.
- 2026-07-21 — Q1 — **closed-negative**. Built the two-stage merge harness,
  banked 20 dev clips, measured solo complementarity 0.1481 (vs comp 0.3818)
  and all six merge variants negative (best -0.0167 Tab F1, CI-sig -0.0195
  onset F1); root cause = added-note precision 0.181.
  `n2_muscriptor_merge_pilot_2026-07-21.md`.
