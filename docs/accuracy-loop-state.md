# Accuracy-loop state
last_updated: 2026-07-23
current_branch: accuracy/program-summary  (PROGRAM COMPLETE)

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| Q1 | N2 MuScriptor merge | **closed-negative** | best variant ΔTab F1 **-0.0167** [-0.0480, +0.0090]; Δonset **-0.0195** [-0.0325, -0.0079] (CI-sig regression); added-note precision **0.181** vs 0.6855 stream precision | none — closed | — |
| Q2 | S1b-v2 offline probe | **closed-negative** | full recipe: top-1 **0.7015**, Δ **+0.0467** [+0.0291, +0.0640] vs target 0.7048 — short by 0.0033; comp +0.0661 vs solo +0.0112 | none — closed | — |
| Q3 | S1b-v2 integration | **dropped** | — | — | Q2 closed-negative |
| Q4 | Second-opinion probes | **dropped** (user, 2026-07-22) | leg-2 gate **derived = 0.528**, 5/5 sign agreement — kept as the standing bench for any future candidate | none — dropped | — |
| Q5 | Onset snapping | **closed-negative** | best `snap-10ms` **+0.0002** [-0.0009, +0.0016]; wider windows lose on Tab *and* onset; `timing_only` rises 15→41 | none — closed | — |
| Q6 | Inharmonicity study | **REGISTERED (opt-in)** | player-05 **+0.0780** [+0.0502, +0.1078]; `acoustic-physics-v1` registered + wired; **`auto` still `none`** | **user call: default-on for clean steel acoustic?** | user decision |
| Q7 | Capo/tuning preflight | **transform validated; detection refuted** | covariant **+0.3870** [+0.2818, +0.4906]; capo detection from pitch **1/60** — warn only, no auto-set | **user call: route capo>0 to covariant prior?** (capo stays user-supplied) | user decision |
| Q8 | Review-ranker upgrade | **unblocked-but-orphaned** | beat 38.76% @60s | needs a posterior source; Q3 dropped, so re-scope or drop | Q3 dropped |

**PROGRAM COMPLETE — every item is closed-negative, dropped, or blocked on a user decision. The loop has produced its final summary (bottom of this file) and stopped. Two `auto`-routing decisions remain for the user; a proposed next program (N1-N4) is at the end.**

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

## Q6 fusion integration — Tab F1 lift measured (2026-07-22)

Report: `docs/EVAL_REPORTS/q6_fusion_eval_2026-07-22.md` (+ `.json`).
Module: `tabvision/tabvision/fusion/inharmonicity.py` (package code, 10 unit
tests). **`auto` untouched, nothing registered.**

| arm | Tab F1 | ΔTab F1 [lo-95, hi-95] | solo Δ | onset/pitch F1 |
|---|---:|---|---|---|
| baseline | 0.6773 | — | — | 0.9325 / 0.9131 |
| **w=1.0, r²>=0.50** | **0.7298** | **+0.0525 [+0.0208, +0.0888]** | **+0.1050 [+0.0553, +0.1537]** | 0.9325 / 0.9131 |

- All four arms CI-significantly positive. **Onset and pitch F1 bit-identical
  across every arm** — the channel rewrites `fret_prior` only, asserted by
  unit test.
- **Decomposition is a one-for-one conversion:** correct 1411 -> 1477 (+66),
  wrong_position 443 -> 377 (-66), all four other buckets **exactly 0**.
  Cleanest bucket result in the program; §6.3 leakage check passes strictly.
- The detected-notes probe predicted "~+0.10 solo" *before* this run from
  coverage and per-note accuracy alone; measured +0.1050. Out-of-sample check
  on the reasoning.
- Coverage 10.1% of detections; abstains on ~90% and on strummed material.

**Not a ship gate:** 20 clips; weight/r² chosen on the reported set so
+0.0525 is the optimistic end; and **calibration is GuitarSet-specific** —
a user's own guitar needs its own B0 via the per-session EM bootstrap §4.1
sketches and this work does not implement. That is the biggest gap between
"works on GuitarSet" and "works on your recording."

## Q6 generalization — self-calibration FAILS (2026-07-22)

Report: `docs/EVAL_REPORTS/q6_self_calibration_2026-07-22.md` (+ `.json`).

| arm | ΔTab F1 [lo-95, hi-95] | requires |
|---|---|---|
| `lopo` | **+0.0525** [+0.0208, +0.0888] | other guitars' gold labels |
| `self-seeded` | **+0.0388** [+0.0107, +0.0720] | reference table + session refit |
| `self-blind` (1 clip) | +0.0000 | nothing — abstains everywhere |
| `self-pooled` (~2 min) | -0.0029 [-0.0088, +0.0000] | nothing — and doesn't help |

**Only the arms carrying a reference table work.** Two measured causes:
*data volume* (needs ~8 fitted isolated notes **per string**; a 30 s clip
yields ~10 across all six, and 2 min pooled is still short) and *bootstrap
bias* (first-pass labels are ~65% right on exactly the ambiguous notes at
issue; measured median log-B0 shift **+0.2975**, ~35% in B, comparable to the
1.6-1.8x separation the method depends on — and systematic, since decoder
errors correlate with string).

**Bug found and fixed:** `inharmonicity_matrix` scored candidates on
uncalibrated strings at probability **zero** — a hard veto, not abstention —
so a partial table silently forced notes onto whichever strings had data.
First self-blind run regressed **-0.0329**; after the fix it is +0.0000.
Latent hazard removed from the shipping path regardless of how calibration is
solved.

**Routes to portability, none yet tested, in value order:**
1. Derive the reference table from **string-manufacturer physics** (gauge,
   core construction, scale length) instead of fitting GuitarSet — genuinely
   general for standard sets, demotes GuitarSet to validation.
   `self-seeded`'s +0.0388 shows reference-plus-refinement keeps most of it.
2. Anchor the shared offset on **unambiguous notes** (label-free, no decoder).
3. A **six-open-string calibration ritual** — perfect labels, ~10 s of setup.

**Untested and material:** whether the GuitarSet-fitted table transfers to a
different acoustic guitar at all. No second acoustic dataset exists in-repo.

## Q6 portability SOLVED — specification-derived table (2026-07-22)

Report: `docs/EVAL_REPORTS/q6_physics_table_2026-07-22.md`.
Module: `tabvision/tabvision/fusion/string_physics.py`.

`B = pi^3*E*d_core^4 / (256*mu*L^4*f^2)` — every term published or
measurable, nothing fitted.

| arm | ΔTab F1 [lo-95, hi-95] | requires |
|---|---|---|
| `lopo` | +0.0525 [+0.0208, +0.0888] | labelled reference guitars |
| **`physics`** | **+0.0502** [+0.0198, +0.0853] | **published specs only** |
| **`physics+offset`** | **+0.0581** [+0.0203, +0.1052] | specs + one scalar |

**Statistically indistinguishable from the fitted table.** GuitarSet is now a
*test* of the table, not its source — the dataset dependence is gone.

- The **fret law is derived, not assumed**: `B_n = B0*2^(n/6)` falls out of
  `L_n = L*2^(-n/12)`, `f_n = f*2^(n/12)`.
- Physics table is low by 0.566 log (0.57x) with 0.249 (1.28x) residual
  spread. **Level error is harmless** — a shared factor shifts every
  candidate for a note equally. Only *shape* can flip a decision, and 1.28x
  sits inside the 1.59-1.78x separation.
- Residual splits by construction: wound -0.53/-0.81/-0.60/-0.71, plain
  -0.15/-0.20. Plain strings' `d_core` is the gauge exactly; wound cores are
  manufacturer-specific and `B ∝ d_core^4`. Fixable once per string set.
- **Calibration take implemented** (`calibrate_from_ritual`): 18 notes, three
  frets x six strings, so the **fret exponent is measured not assumed**.
  Labels certain (the app asks), so no bootstrap bias. **Not validated end to
  end** — GuitarSet has only 1-3 usable isolated open notes per player, so it
  cannot contain the ritual; validating on a real recording would make it an
  eval artifact under the private-recordings ban.

## Q6 domain guard — GAPS gate satisfied by construction (2026-07-22)

`string_physics.stiffness_model_for_session` returns a table **only** for
clean steel-string acoustic, standard tuning, capo 0. Everything else gets
`None`, and `attach_inharmonicity_evidence` treats `None` as an explicit
no-op, so out-of-domain sessions are **bit-identical to baseline**.

- **Classical/nylon abstains.** `B` is linear in Young's modulus: polyamide
  ~3 GPa vs steel ~200 GPa, so a nylon treble is ~**65x less inharmonic**.
  Candidates separate by only 1.6-1.8x, so the steel table would be wrong by
  more than the whole signal. The physics fits nylon fine — it needs a nylon
  table, which does not exist here.
- **Capo and alternate tuning also abstain** — `B0` describes the *open*
  string, and both move speaking length and tension. Previously latent.
- **The GAPS clean-12 classical no-regression check is therefore proven, not
  measured** (`test_out_of_domain_sessions_are_bit_identical_to_baseline`).
  A ~2 CPU-hour run was started and stopped once this was clear.
  `scripts/eval/q6_gaps_no_regression.py` is retained as the empirical
  confirmation for the day a nylon table exists.

## Q6 full-dev OOF — PASSED (2026-07-22)

Report: `docs/EVAL_REPORTS/q6_full_dev_2026-07-22.md` (+ `.json`).
Runner: `scripts/eval/q6_full_dev.py` (config frozen in source).

300 GuitarSet dev clips, weight 1.0 / min_r2 0.50 / raw physics table, all
fixed **before** the run — no sweep.

| metric | baseline | arm | Δ [lo-95, hi-95] |
|---|---:|---:|---|
| **Tab F1** | 0.6031 | 0.6474 | **+0.0443 [+0.0339, +0.0555]** |
| solo | — | — | **+0.0860 [+0.0673, +0.1055]** |
| comp | — | — | +0.0026 [-0.0000, +0.0069] |
| onset / pitch | 0.9182 / 0.8951 | 0.9182 / 0.8951 | bit-identical |

**Gate lo-95 > 0: PASS** (lower bound +0.0339, far clear of zero). Sits
~0.008 below the pilot's tuned +0.0525 — expected once the weight is no
longer chosen on the reported set.

- **Decomposition one-for-one at 52k events:** correct +1,248,
  wrong_position -1,248, pitch_off/timing/missed/extra **each 0**. §6.3
  leakage check passes strictly.
- **Honest cost:** 129 improved, 146 unchanged, **25 regressed** — a mean
  improvement, not strict Pareto. Coverage 8.3% of detections, ~all solo.

## Q7 entry probe — capo-covariant prior PASSES (2026-07-22)

Report: `docs/EVAL_REPORTS/q7_capo_covariant_2026-07-22.md` (+ `.json`).
Probe: `scripts/eval/q7_capo_covariant_probe.py` (label-level, no audio).

Today capo>0 routes to `priors=none`, discarding the prior lift. §4.3's fix:
shift the fret axis by the capo. Top-1 assignment on ~51k ambiguous dev
notes:

| capo | covariant | naive | none-lowfret (today) |
|---:|---:|---:|---:|
| 0 | 0.5960 | 0.5960 | 0.4378 |
| 4 | 0.5959 | 0.4681 | 0.4377 |
| 7 | 0.5951 | 0.4366 | 0.4366 |

- **Transform correct:** capo-0 `covariant == naive`; covariant flat across
  capo. Flatness is **partly by construction** (shifting gold + lookup
  together maps capo-C onto capo-0) — confirms the arithmetic, not
  independent evidence.
- **Gap is real:** today 0.438 vs covariant 0.596 = **+0.158** conditional
  (same order as §4.3's +22 pp). Anchor 0.596 is position-prior-alone
  (below Q2's full-decode 0.6548 — no Viterbi/seq/playability here).
- **Shift is necessary:** naive degrades 0.596→0.437 by capo 7 — no better
  than no prior. Not tautological.
- **Untestable here:** assumes real capo playing follows capo-0 relative-fret
  conventions; GuitarSet has no capo audio.

**Next slice:** preflight capo/tuning detection + wire covariant behind a
flag + validate on pitch-shifted audio (real Tab F1, multi-hour). That is the
shipping gate; the label probe cannot substitute for it.

## Q6 player-05 sealed confirmation — PASS (2026-07-22)

Report: `docs/EVAL_REPORTS/q6_player05_confirm_2026-07-22.md` (+ `.json`).
Runner: `scripts/eval/q6_player05_confirm.py` (config frozen in source).

60 sealed clips, frozen config (weight 1.0 / min_r2 0.50 / raw physics
table), registered `guitarset-v1` prior (excludes player 05 by manifest).

| metric | baseline | with channel | Δ [lo-95, hi-95] |
|---|---:|---:|---|
| **aggregate** | 0.6340 | **0.7119** | **+0.0780 [+0.0502, +0.1078]** |
| **solo** | 0.5503 | **0.6899** | **+0.1396 [+0.0985, +0.1806]** |
| comp | 0.7176 | 0.7340 | +0.0164 [+0.0000, +0.0458] |
| onset / pitch | 0.9473 / 0.9386 | unchanged | bit-identical |

**Gate lo-95 > 0: PASS.** Solo gains 25% relative on the tier SPEC §1.4.1 is
weakest on and where every other lever failed (Q2 moved solo +0.0112).

- **Harness validated:** baseline reproduces the shipped production numbers
  to 4 decimals (solo 0.5503, comp 0.7176, agg 0.6340 vs CLAUDE.md's
  0.5503 / 0.7175 / 0.6339). The delta is against the real production decode.
- **Decomposition one-for-one (3rd time):** +367 correct / −367
  wrong_position, other four buckets **exactly 0** across 8,709 detections.
- 30 improved / 28 flat / **2 regressed** (3.3%, below dev's 8.3%).
- **Hold-out > dev** (+0.0780 vs +0.0443) — opposite of overfitting, but the
  CIs overlap (dev hi +0.0555 vs p05 lo +0.0502), so **not significant**;
  player 05 is simply cleaner (baseline 0.6340 vs 0.6031) with 9.6% vs 8.3%
  coverage.

**Every offline gate in the program is now passed.** What remains is product
decisions, not evidence: registration and `auto` routing.

## Q7 build slice — capo-covariant prior validated on audio (2026-07-23)

Report: `docs/EVAL_REPORTS/q7_capo_audio_2026-07-23.md` (+ `.json`).
Transform: `capo_covariant_prior` in `fusion/position_prior.py`.

20 clips pitch-shifted +2/+4 semitones with capo-shifted labels, LOPO priors,
arms paired on identical shifted audio. **Capo-0 control 0.6773.**

| capo | today (priors=none) | covariant | Δ [lo-95, hi-95] | naive |
|---:|---:|---:|---|---:|
| 2 | **0.2956** | **0.6827** | **+0.3870 [+0.2818, +0.4906]** | 0.3573 |
| 4 | **0.2875** | **0.6533** | +0.3658 [+0.2613, +0.4685] | 0.2868 |

- **Today's capo routing is a collapse, not a shortfall.** §4.3 framed it as
  losing "+22 pp"; capo sessions actually score **0.2956 vs 0.6773** — less
  than half — with wrong_position at 1182 of ~1800 notes. Without a prior the
  decoder's low-fret fallback breaks when every candidate sits above the capo.
  **This half depends on no modelling assumption.**
- **Recovery is partly by construction** (synthetic capo relabels capo-0 gold;
  the covariant prior shifts identically). What it proves is the transform is
  correct end-to-end through real audio/fuse(); it does **not** prove real
  capo playing follows capo-0 conventions.
- **One-for-one:** capo 2 +767 correct / −767 wrong_position; capo 4 +686/−686.
- **Naive decays with depth:** +0.0617 at capo 2, **−0.0007 at capo 4**.
- **Sequence prior adds nothing under capo (measured):** covariant+seq 0.6766
  vs 0.6827 at capo 2. `guitarset-seq-v1` uses `delta_fret`, whose absolute
  fret-region conditioning is capo-shifted, so it backs off to the delta
  backbone. A capo-covariant seq prior is separate, small work.
- **Methodology correction:** the first run used the registered `guitarset-v1`
  (trained on players 00-04 = these clips) and reported an in-sample +0.45;
  LOPO gives +0.387.

## Q7 capo detection — pitch route refuted (2026-07-23)

Report: `docs/EVAL_REPORTS/q7_capo_detect_2026-07-23.md` (+ `.json`).
Module: `tabvision/preflight/capo.py`.

60 cases (20 clips × capo 0/2/4), ground truth by construction.

| estimator | exact | mean signed error |
|---|---:|---:|
| pitches | **0.017** (1/60) | **+1.92** |
| physical upper bound valid | **1.000** | — |

- **Pitch-based detection is refuted, and the negative is valid.** A capo at
  `C` is pitch-identical to capo 0 transposed up `C`, so no pitch heuristic
  can separate them; occupancy guesses repertoire. Pitch-shift models pitch
  content faithfully, so the test was fair to what this estimator consumes.
- **The physics estimator could not be tested — do not quote its 0.183.**
  A real capo 2/4 should raise median `log B` by **+0.231 / +0.462**;
  measured on pitch-shifted audio it moved **−0.11 … +0.14**. Pitch-shifting
  scales frequencies without shortening the string, so the synthetic audio is
  capo-0 stiffness with capo-`C` pitches. The test is invalid, not the method.
- **Methodology rule for the program:** synthetic pitch-shift is valid for
  pitch/position evaluation, **invalid** for timbre/physics evaluation.
  Checked: Q7's covariant result stands (position priors over pitch content);
  the Q6 channel is uncontaminated (abstains at capo>0, never ran on it).
- **Conclusion:** preflight reports the bound and asks. Auto-setting the capo
  from pitch would be wrong ~98% of the time, so the capo stays user-supplied.

**Also fixed:** `acoustic_physics_v1.json` was hashed over CRLF bytes while
git stores LF, so the registered artifact failed hash verification on a fresh
checkout — broken for anyone cloning. Now written as the bytes git stores;
hash stable across a round-trip.

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

**DECISION NEEDED (Q6 gates):** portability is solved — the physics table
needs no dataset and matches the fitted one. What remains is the standard
ladder, none of which has run: **(1)** full-dev OOF (not 20 clips) with
weight/threshold fixed *before* the run; **(2)** ~~GAPS clean-12~~ **done by construction** (classical abstains,
unit-tested); **(3)** player-05 confirmation;
then registration and the `auto`-routing decision. Separately, **(4)** the
calibration take needs real-recording validation, which needs public audio or
a deliberate exception to the private-recordings ban. **Options: (a) work
down (1)-(3); (b) do (4) first; (c) bank Q6 and move to Q7.** Recommend (a) —
the channel is shippable-shaped now and the gates are what stand between it
and `auto`.

Q1, Q2, Q5 closed negative; Q4 dropped; Q3 dropped with Q2; Q8 orphaned.
Q7 is unblocked and needs no new data (synthetic capo shifts of GuitarSet).

Note: iterations 1-3 ran on branches `accuracy/q1-n2-merge` and
`accuracy/q2-s1b-probe`, stacked in that order (the state file is a running
document, so Q2 builds on Q1). They are unpushed and must be merged in
order. A parallel session moved the shared working tree onto
`fretcam/f2-detection-chain`; iteration 3 was committed via a temporary git
worktree so that checkout was left undisturbed.

## Iteration log (newest first)
- 2026-07-23 — Q7 — capo **detection** slice: pitch-based **refuted**
  (1/60 exact, +1.92 bias); physics route **untestable** on pitch-shifted
  audio (stiffness shift −0.11…+0.14 vs +0.231/+0.462 expected) — test
  invalid, not method. Preflight warns, never auto-sets. Also fixed a
  CRLF/LF artifact-hash bug that broke `acoustic-physics-v1` on fresh
  checkout. `q7_capo_detect_2026-07-23.md`.
- 2026-07-23 — Q7 — build slice: capo-covariant prior **validated on audio**.
  Today's capo routing measured at **0.2956 vs 0.6773** capo-0 control — a
  collapse, not a shortfall; covariant restores to **0.6827** (+0.3870).
  Naive decays to −0.0007 by capo 4; seq prior adds nothing under capo.
  Caught + fixed an in-sample-prior error mid-run (+0.45 → +0.387).
  `q7_capo_audio_2026-07-23.md`.
- 2026-07-23 — Q6 — **registered** `acoustic-physics-v1` (hash-verified,
  gate provenance in manifest, decode config bound to the hash) and wired
  behind explicit selection. **`auto` still resolves to `none`** (asserted by
  test); default-on remains a user decision.
- 2026-07-22 — Q6 — **player-05 sealed confirmation PASS**: agg 0.6340→0.7119
  (**+0.0780** [+0.0502, +0.1078]), solo 0.5503→0.6899 (+0.1396); +367/−367
  one-for-one, onset/pitch bit-identical; baseline reproduces shipped numbers
  to 4dp. All offline gates passed. `q6_player05_confirm_2026-07-22.md`.
- 2026-07-22 — Q7 — entry probe **PASS**: capo-covariant transform correct
  (covariant 0.596 flat vs today's 0.438, +0.158 conditional); naive
  degrades 0.596→0.437 so the shift is necessary. By-construction caveat
  stated; real gate = pitch-shifted-audio Tab F1 (next slice).
  `q7_capo_covariant_2026-07-22.md`.
- 2026-07-22 — Q6 — **FULL-DEV GATE PASSED**: 300 clips, frozen config,
  **+0.0443** [+0.0339, +0.0555] (solo +0.0860, comp +0.0026); +1248 correct
  / -1248 wrong_position, all other buckets exactly 0; onset/pitch
  bit-identical. Next gate = player-05 (user-gated).
  `q6_full_dev_2026-07-22.md`.
- 2026-07-22 — Q6 — **domain guard**: channel abstains outside clean
  steel-string acoustic (classical/nylon ~65x less inharmonic; also capo and
  alt tuning). GAPS cross-domain gate now **satisfied by construction and
  unit-tested**; the ~2 CPU-hour run was stopped as unnecessary.
- 2026-07-22 — Q6 — **portability SOLVED**: specification-derived table gives
  **+0.0502** [+0.0198, +0.0853] with no dataset, matching the fitted
  +0.0525; +offset **+0.0581**. Fret law derived not assumed. Calibration
  take (18 notes) implemented, not yet validated on real audio.
  `q6_physics_table_2026-07-22.md`.
- 2026-07-22 — Q6 — generalization: **self-calibration FAILS** (self-blind
  +0.0000, self-pooled -0.0029 vs lopo +0.0525). Causes: data volume + a
  +0.2975 log-B0 bootstrap bias. Found and fixed a hard-veto bug that had
  caused a -0.0329 regression. `q6_self_calibration_2026-07-22.md`.
- 2026-07-22 — Q6 — **INTEGRATED**: ΔTab F1 **+0.0525** [+0.0208, +0.0888],
  solo **+0.1050**; decomposition +66 correct / -66 wrong_position, all other
  buckets exactly 0; onset/pitch bit-identical.
  `q6_fusion_eval_2026-07-22.md`.
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

---

# PROGRAM COMPLETE — final summary (2026-07-23)

All eight queue items are terminal (`closed-negative`, `dropped`, or
`blocked` on a user decision), so the loop stops here per its own rule.
16 iterations, 23 commits on a stacked chain ending at
`accuracy/q7-capo-detect`, unpushed, to be merged in order.

## What the program bought

| item | outcome | number |
|---|---|---|
| Q1 second-opinion merge | closed-negative | added-note precision 0.181 vs break-even 0.528 |
| Q2 symbolic contextual assigner | closed-negative | +0.0467 vs a +0.05 bar |
| Q3 S1b integration | dropped with Q2 | — |
| Q4 Basic Pitch / YourMT3+ | dropped (user) | environment-blocked; leg-2 gate derived |
| Q5 onset snapping | closed-negative | +0.0002; `timing_only` rose 15→41 |
| **Q6 inharmonicity** | **all gates passed, registered opt-in** | **player-05 +0.0780 [+0.0502, +0.1078]** |
| **Q7 capo-covariant prior** | **validated, unwired** | **+0.3870 [+0.2818, +0.4906] at capo 2** |
| Q8 review-ranker | orphaned | needed Q3's posteriors |

**One lever worked, and it is the one nobody had tried.** Five of eight closed
negative or dropped. The survivor is physics: inharmonicity as per-note string
evidence, derived from published string specifications rather than fitted to
any dataset, taking sealed player-05 aggregate Tab F1 from **0.6340 → 0.7119**
and solo from **0.5503 → 0.6899**.

## The three findings worth carrying forward

**1. Wrong-position is the target, and single lines are where it hides.**
57.3% of all loss, 77.5% of single-line loss. Q2 then showed that *context*
resolves chords (+0.0661) far better than single lines (+0.0112) — so the
tier carrying most of the loss is the one sequence models cannot help. That is
precisely the gap the physics channel fills (solo +0.1396 on player-05).

**2. Complementarity is not sufficient for a merge — added-note precision is.**
MuScriptor cleared the ≥0.10 complementarity gate by 3.8× and still produced
no admissible merge. The derived break-even
`p > (F1/2) / (α·(1 − F1/2) + F1/2)` = **0.528** predicted the sign of all five
admitting variants (5/5). Volume cancels: how many notes a merge admits never
changes the sign, only the magnitude.

**3. Today's capo handling is a collapse, not a shortfall.** §4.3 framed capo
sessions as losing a "+22 pp" bonus. They actually score **0.2956 against a
0.6773 capo-0 control** — the no-prior fallback's low-fret heuristic breaks
when every candidate sits above the capo. This is the largest single defect
the program found, and it needs no new research to fix.

## Two decisions pending (both yours, SPEC §0.8)

1. **Default-on the Q6 channel** for clean steel-string acoustic, or keep it
   opt-in behind `--string-evidence acoustic-physics-v1`? Evidence: every
   offline gate passed, one-for-one bucket conversion on all three runs,
   onset/pitch bit-identical, domain-guarded twice. Caveat: all validation is
   GuitarSet on similar steel-strings — portability to another guitar is
   argued from physics, not measured.
2. **Route capo>0 to the covariant prior** instead of `priors=none`? Worth
   ~+0.37 to a capo user. Note the capo must still be supplied by hand;
   detection is refuted.

## Proposed next program, in ROI order

**N1 — Extend the physics channel's coverage (highest ROI).** Accuracy when it
fires is 0.92; **coverage is 8-10% of detections** and is the binding
constraint. Only 4,407 of 9,927 isolated notes (44%) produce a usable fit, so
the cheapest lever is fit *success*, not fit quality: longer adaptive windows,
better partial tracking, and replacing the hard `r² ≥ 0.50` gate with a
calibrated soft weight so marginal fits contribute proportionally instead of
nothing. Doubling coverage roughly doubles the gain. Days, `$0`, and it reuses
the shipped module.

**N2 — A nylon table for classical.** The channel abstains on classical
entirely because no nylon table exists. Nylon's `E` (~3 GPa vs steel's ~200)
is published, the machinery is identical, and GAPS is already on disk to
validate against — and classical is single-line-heavy, exactly where the
method is strongest. This also converts the GAPS cross-domain gate from
"satisfied by abstention" into a real measurement. ~1 week.

**N3 — Re-scope Q8 on the physics channel.** The review-ranker died because it
wanted Q3's posteriors. The inharmonicity fit produces exactly what it lacked:
a per-note confidence (`r²`) and a physically-grounded margin between
candidate strings. Target: beat the shipped 38.76% wrong-position reduction
@60 s. Reported separately from automatic Tab F1. Days.

**N4 — Per-instrument calibration for real use.** `calibrate_from_ritual`
(18 notes, three frets × six strings, fits `B0` *and* the fret exponent) is
built but unvalidated on real plucks — GuitarSet contains only 1-3 usable
isolated open notes per player, so it cannot contain the ritual. Needs public
capo/calibration audio or an explicit exception to the private-recordings ban.
This is the gap between "works on GuitarSet" and "works on your guitar."

**Explicitly not recommended:** more second-opinion merges (the two-leg gate
kills them cheaply — use the bench first), more context/sequence modelling for
single lines (Q2 closed it), any onset refinement (Q5: the backend already
beats spectral flux), and capo detection from pitch (Q7: impossible in
principle).

## Methodology rules earned here

- **Synthetic pitch-shift is valid for pitch/position evaluation and invalid
  for timbre/physics** — it scales frequencies without shortening the string.
  Cost: one invalidated experiment, caught by checking the measured stiffness
  shift against the physical prediction.
- **Always use leave-one-player-out priors on dev clips.** An in-sample slip
  inflated a capo result to +0.45 against a true +0.387.
- **Registered artifacts must be written as the bytes git stores (LF)**, or
  hash verification fails on a fresh checkout — the artifact is then broken
  for everyone but its author.
- **Domain guards can discharge cross-domain gates by proof.** Scoping the
  channel to instruments it has a table for satisfied the GAPS clean-12
  no-regression check by construction, replacing a ~2-hour run with a unit
  test.
- **Bank events, replay offline.** Most gates in this program cost seconds
  because inference was cached once and every variant swept against it.
