# Accuracy-loop state
last_updated: 2026-07-22
current_branch: accuracy/q6-inharmonicity

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| Q1 | N2 MuScriptor merge | **closed-negative** | best variant ΔTab F1 **-0.0167** [-0.0480, +0.0090]; Δonset **-0.0195** [-0.0325, -0.0079] (CI-sig regression); added-note precision **0.181** vs 0.6855 stream precision | none — closed | — |
| Q2 | S1b-v2 offline probe | **closed-negative** | full recipe: top-1 **0.7015**, Δ **+0.0467** [+0.0291, +0.0640] vs target 0.7048 — short by 0.0033; comp +0.0661 vs solo +0.0112 | none — closed | — |
| Q3 | S1b-v2 integration | **dropped** | — | — | Q2 closed-negative |
| Q4 | Second-opinion probes | **dropped** (user, 2026-07-22) | leg-2 gate **derived = 0.528**, 5/5 sign agreement — kept as the standing bench for any future candidate | none — dropped | — |
| Q5 | Onset snapping | **closed-negative** | best `snap-10ms` **+0.0002** [-0.0009, +0.0016]; wider windows lose on Tab *and* onset; `timing_only` rises 15→41 | none — closed | — |
| Q6 | Inharmonicity study | **GATES PASS + transfers to detected notes** | detected-stream acc **0.9242** (vs 0.9200 gold) but coverage **10% of detections, solo-only** (solo n=208 / comp n=3); est. solo ambiguous lift **~+0.10** | user call: integrate bounded soft evidence behind a flag | user decision |
| Q7 | Capo/tuning preflight | open | — | design synthetic-capo eval | — |
| Q8 | Review-ranker upgrade | **unblocked-but-orphaned** | beat 38.76% @60s | needs a posterior source; Q3 dropped, so re-scope or drop | Q3 dropped |

**Q6 PASSED — integration is a user call (see Questions). Topmost open unblocked item otherwise = Q7 (capo/tuning preflight).**

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

## Q4 — iteration 4: bench built, Basic Pitch blocked, leg-2 derived

Report: `docs/EVAL_REPORTS/q4_second_opinion_bench_2026-07-22.md`
(+ `q4_breakeven_precision_2026-07-22.json`). DECISIONS.md 2026-07-22.

- **Basic Pitch BLOCKED — environment, not evidence.** Declared extra,
  Apache-2.0, already licensed — but this machine has only Python 3.12 and
  every release falls back to building numpy from source, whose `setup.py`
  uses `pkgutil.ImpImporter` (removed in 3.12). `[onnx]` fails the same way.
  Probe venv exists at `~/.tabvision/probe-envs/basicpitch` but is unusable.
- **Recommendation: drop rather than install Python 3.11.** Basic Pitch's
  published GuitarSet zero-shot note F1 is 66.1 vs our ensemble's 0.9491/
  0.9403; MuScriptor is far stronger, passed leg 1 by 3.8x, and still failed
  leg 2 at 0.181 vs a 0.528 break-even. Deep-dive prices the row at
  +0.00-+0.02.
- **Leg 2 is now derived, not guessed:**
  `p > (F1/2) / (α·(1 − F1/2) + F1/2)`. Volume cancels — **how many notes a
  rule admits never changes the sign, only the magnitude** (which is why all
  six N2 variants were negative and the cautious ones merely lost less).
  On the banked pilot (F1 0.6773, measured α 0.4581) break-even =
  **0.5278**, predicting the sign of all five admitting variants (**5/5**).
  Recompute per candidate: as the ensemble improves, the bar rises.

## Q5 — CLOSED 2026-07-22 (banked negative)

Report: `docs/EVAL_REPORTS/q5_onset_snapping_2026-07-22.md` (+ `.json`).
DECISIONS.md 2026-07-22.

- `snap-10ms` is a wash (+0.0002 Tab F1 [-0.0009, +0.0016]); wider windows
  lose monotonically on Tab **and** onset F1 (`snap-50ms`: -0.0047 / -0.0097).
  Strum-cluster collapsing adds nothing.
- **Mechanism is the inverse of the intent:** `timing_only` — the bucket
  snapping exists to drain — rises **15 → 41** while `correct` falls
  **1411 → 1384**; missed/extra barely move. Snapping pushes
  already-inside-tolerance notes *out*.
- **The ensemble's onsets are already better than STFT flux peaks** (mean
  shift only 5.4 ms at a 10 ms window). "Snapping Matters" (piano) assumes a
  detector whose timing is the weak link; at onset F1 0.9325 that is false
  here. The deep-dive had already flagged that result as non-transferable.
- **Re-open only** with a backend whose onset timing is worse than spectral
  flux, or a materially better onset estimator than STFT flux. This retires
  the whole "refine the onsets" class: any such method must first beat the
  backend's own timing.

## Q6 — blocked on the hex download; precursor PASSES

Report: `docs/EVAL_REPORTS/q6_separability_2026-07-22.md` (+ `.json`).
DECISIONS.md 2026-07-22.

- Gate A needs GuitarSet's **hexaphonic partition**, not on disk — the
  acquirer takes `annotations` + `audio_mic` only and explicitly skips the
  multi-GB hex-pickup/mix partitions. Mono-mic alone is 1.6 GB.
- **Precursor (no download spent) passes.** With `B(s,n) = B0_s · 2^(n/6)`,
  the assumption-free separation between two candidates for one pitch is the
  fret-difference term. On 35,959 dev-OOF ambiguous notes **every**
  rank-1/rank-2 pair is **≥4 frets apart** (99.4% at 4-5) = **1.59-1.78×
  B ratio from length alone**, before plain-vs-wound B0 differences (ignored,
  so these are lower bounds).
- Lower-bound string accuracy vs B-estimator relative error: 0.9956 @10%,
  0.9116 @20%, 0.8175 @30% → **Gate A clearable if B is estimable to better
  than ~25%**.
- **Does not** establish that B *is* estimable to 25% on real audio — that is
  the actual risk, and exactly what Gates A/B were pointed at. Literature is
  encouraging on isolated notes (Hjerrild & Christensen 1.5% string+fret),
  nothing survives dense polyphony → §4.1's single-line scope stands.

## Q6 — BOTH GATES PASS (2026-07-22)

User approved the download 2026-07-22; it failed. **zenodo.org resolves
(188.184.98.114) but TCP :443 times out**, while huggingface.co and pypi.org
connect fine, and the same timeout occurs with the sandbox disabled — a
routing/firewall block specific to Zenodo, not a repo or sandbox issue.

Hex partition acquired over VPN (3.36 GB -> 8.8 GB, 361 tracks). Both gates
run on dev players 00-04, LOPO, **isolated notes only**.

| gate | source | notes | acc @~70% coverage | unfiltered | count-prior control |
|---|---|---:|---:|---:|---:|
| A (>=0.85) | hex pickup | 6,917 | **0.8950** | 0.8234 | ~0.65 |
| B (>=0.70) | mono mic | 6,771 | **0.9200** | 0.8095 | ~0.65 |

- **The control carries the claim:** count-prior is flat ~0.65 in every arm
  (matching the decoder's own 0.6548 on the ambiguous lattice), so
  inharmonicity beats it by **+0.26-0.31**. Not "isolated notes are easy".
- r-squared is a label-free confidence signal (fit residual only); accuracy
  rises monotonically 0.81 -> 0.99 as coverage falls 100% -> 29%. A
  legitimate abstention channel.
- **The mic beats the pickup** (0.9200 vs 0.8950 at matched coverage) —
  GuitarSet's hex is a bridge-mounted Roland GK-style divided pickup,
  band-limited, and B lives in the high partials. **So hex was needed to
  validate the estimator, not to carry the signal: the channel can run on
  the mono mic the pipeline already has.**
- **Scope:** "isolated" = ~34% of solo notes, ~1.3% of comp. A single-line
  instrument, as §4.1 scoped — which is the point, since Q2 left solo at
  +0.0112 and single-line carries 77.5% of wrong-position loss.
- **Detected-notes probe (2026-07-22): the physics transfers.** On the
  20-clip ensemble bank, scoring the *detected* stream gives **0.9242 at
  r²>=0.50 vs 0.9200 on gold** — 50 ms onset error and wrong pitches do not
  degrade the estimate. Control 0.654 matches the decoder's own 0.6548.
- **Coverage is the binding constraint and it is solo-only:** of 2,105
  detections, 346 (16.4%) isolated+ambiguous, 213 (10.1%) survive the fit —
  **solo n=208 vs comp n=3**. Estimated pooled ambiguous top-1 lift
  **~+0.039**; solo-tier lift **~+0.10** (vs Q2's +0.0112 on solo). Upper
  bounds: they assume hard replacement, not the bounded soft evidence §4.1
  specifies.

`scripts/eval/q6_gate_a.py` is complete and **self-validated on synthetic
stiff strings** (recovers B at 5e-5/1e-4/5e-4 within 25%, f0 within 1%,
separates the 1.78× five-fret case, rejects silence). The synthetic test
caught a real bug: the partial-search half-width scaled as `k^1.5` and by
k≈10 swallowed the neighbouring partial, returning a confidently wrong
f0 (111.3 for 110 Hz) — invisible on real audio, and it would have made a
Gate A failure uninterpretable. Window now capped at `0.4·f0`.
Also asserts the hex channel↔string mapping empirically rather than assuming
`data_source` order.

## Q4 gate revision (binding, from Q1's carry-forward)

Second-opinion candidates gate on **both** legs, measured in the same
offline replay:

1. P(candidate right | ensemble wrong) ≥ 0.10 — is there anything to gain;
2. **added-note precision ≥ break-even** under the candidate's best
   admission rule — can the gain be separated from the noise it arrives
   with. The break-even is computed, not fixed:
   `p > (F1/2) / (α·(1 − F1/2) + F1/2)` = **0.528** against today's
   ensemble (`q4_breakeven_precision.py`).

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

**DECISION NEEDED (Q6):** both gates passed, so §4.1's offline study is
complete and the route is live. The next step is the **first item in this
loop that would touch fuse()** — a bounded emission-evidence term on
single-line segments, confidence-weighted by r-squared, zero-weighted below a
fit threshold. That is pipeline code plus the usual OOF -> GAPS
no-regression -> player-05 discipline. **Options: (a) proceed to integration
behind an explicit TABVISION_STRING_EVIDENCE flag with auto unchanged;
(b) ~~detected-notes probe~~ **DONE 2026-07-22 — it transfers (0.9242)**;
(c) bank Q6 and move to Q7.** The remaining unknown is not whether the
physics works but whether a *bounded soft* term converts a solo-only, ~10%
coverage channel into Tab F1 — which needs the integration to answer.

Q1, Q2, Q5 closed negative; Q4 dropped; Q3 dropped with Q2; Q8 orphaned.
Q7 is unblocked and needs no new data (synthetic capo shifts of GuitarSet).

Note: iterations 1-3 ran on branches `accuracy/q1-n2-merge` and
`accuracy/q2-s1b-probe`, stacked in that order (the state file is a running
document, so Q2 builds on Q1). They are unpushed and must be merged in
order. A parallel session moved the shared working tree onto
`fretcam/f2-detection-chain`; iteration 3 was committed via a temporary git
worktree so that checkout was left undisturbed.

## Iteration log (newest first)
- 2026-07-22 — Q6 — detected-notes probe: **physics transfers** (0.9242 on
  detected vs 0.9200 gold), coverage **10% of detections and solo-only**
  (solo 208 / comp 3). Est. solo ambiguous lift ~+0.10.
  `q6_separability_2026-07-22.md` (detected section).
- 2026-07-22 — Q6 — **BOTH GATES PASS**. Hex acquired over VPN. Gate A 0.8950
  @71.9% cover, Gate B 0.9200 @66.6%, count-prior control flat ~0.65
  (+0.26-0.31). Mic beats pickup -> channel can run on mono-mic.
  `q6_separability_2026-07-22.md`.
- 2026-07-22 — Q6 — Gate A **blocked on network** (zenodo.org TCP:443
  unreachable). Estimator built + self-validated on synthetic stiff strings;
  synthetic test caught a `k^1.5` search-window bug that biased f0 by +1.2%.
- 2026-07-22 — Q6 — **blocked** on the hex download; separability precursor
  **PASSES** (every ambiguous pair ≥4 frets = 1.59-1.78× B ratio; 0.9116
  lower-bound accuracy at 20% B error). `q6_separability_2026-07-22.md`.
- 2026-07-22 — Q5 — **CLOSED negative**. snap-10ms +0.0002 [-0.0009, +0.0016],
  wider windows lose on Tab and onset; `timing_only` rises 15→41 while
  `correct` falls 1411→1384 — the backend's onsets already beat flux peaks.
  `q5_onset_snapping_2026-07-22.md`.
- 2026-07-22 — Q4 — **dropped by user** (rather than install Python 3.11);
  bench + derived leg-2 threshold retained. Basic Pitch **blocked** (py3.12 vs its
  ≤3.11 dep graph); **leg-2 threshold derived** = 0.528 at measured α=0.4581,
  5/5 sign agreement on the banked N2 variants.
  `q4_second_opinion_bench_2026-07-22.md`.
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
