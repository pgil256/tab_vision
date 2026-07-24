# TabVision accuracy loop — operating prompt for Claude Code

Copy-paste this whole file as the prompt, or start a session with:
"Read docs/prompts/accuracy-loop.md and execute one iteration."
Headless loop: `claude -p "$(cat docs/prompts/accuracy-loop.md)"` run repeatedly.

---

You are working in the TabVision repo (this directory). Mission, sustained
across many sessions: raise automatic Tab F1 by executing the ROI-tiered
program in `docs/2026-07-21-tab-accuracy-roi-deep-dive.md`, **one bounded
work item per iteration**, under the repo's SPEC §0 discipline. You have no
memory between sessions — the state file is your memory.

## Startup (every iteration, in order)

1. Read `CLAUDE.md`, then `docs/2026-07-21-tab-accuracy-roi-deep-dive.md`
   (§3–§7 minimum), then `docs/accuracy-loop-state.md`. If the state file
   does not exist, create it from the template at the bottom of this prompt
   and treat this iteration as item Q1.
2. Read the newest entries in `docs/DECISIONS.md` and any
   `docs/EVAL_REPORTS/*` dated after the state file's `last_updated` — the
   user may have worked outside the loop.
3. Pick the **topmost `open` unblocked item** in the state-file queue. Never
   run two items at once. Never re-open anything in report §5 (closed with
   receipts) unless the state file says the user explicitly re-opened it.

## Work protocol

- **Probe before build.** Every item enters through its stated gate. A gate
  is a measurement, not an implementation: prefer offline replay against
  banked artifacts (e.g. the Phase 0 ambiguous-note lattice CSVs in
  `docs/EVAL_REPORTS/`) before touching pipeline code.
- **Fast loop = val24.** Iterate against `data/eval/local_gs_val24.toml`
  via the harness in `scripts/eval/` — replicate the exact invocation
  recorded in the most recent EVAL_REPORT rather than guessing flags.
  Remember the documented trap: val24 IS GuitarSet; in-distribution gains
  there mean nothing until the cross-domain gate passes.
- **Cross-domain gate.** Anything touching `fuse()` behavior must pass the
  GAPS clean-12 strict per-clip no-regression check before acceptance.
- **Determinism.** seed=42, bootstrap N=10,000, record prediction hashes.
  Acceptance = lower-95 CI ≥ target, per house rules.
- **Ship shape.** An accepted change ships with: unit tests, `ruff check`,
  `mypy tabvision`, `pytest` green; a dated EVAL_REPORT including the
  six-bucket decomposition delta (`tabvision/eval/error_decomposition.py`)
  — verify the gain lands in the bucket the report predicts (§6.3); a
  DECISIONS.md entry; a LICENSES.md label for any NC-derived artifact.
- **Contract safety.** Never change SPEC §8 signatures or SPEC targets.
  New decoders/evidence go behind the existing explicit channels
  (`TABVISION_ASSIGNMENT_DECODER`, `TABVISION_STRING_EVIDENCE`, backend
  registry) with `auto` behavior unchanged until a promotion decision.
- **Banked negatives are wins.** If a gate fails, write the eval report,
  mark the item `closed-negative` in the state file with the number, and
  move on. Do not iterate past a failed gate hoping it improves.
- **Timebox.** One gate or one implementation slice per iteration. If a
  compute step will clearly exceed ~2 h on this machine, checkpoint,
  record how to resume, and end the iteration.

## STOP and ask the user (end the iteration with a question) when:

- Anything costs money (Modal/Colab Pro/API). Free local CPU and free
  Colab are pre-approved. The $25 Modal cap exists but requires explicit
  per-run user approval.
- A new dependency, dataset download, or access request (e.g. DadaGP) is
  needed.
- A gate passes and the next step is a **player-05 confirmation run** —
  player-05 is opened only after config freeze AND the user says proceed.
- Promotion of anything into the `auto` default path.
- A SPEC edit or §8 contract change would be required.
- A gate fails in a way the report's decision tree doesn't cover.
- **Never**, under any circumstances, use private/user recordings in any
  training, eval, or label role.

## End of iteration (always, even if blocked)

1. Update `docs/accuracy-loop-state.md`: item status, key numbers, exact
   next action, blockers, `last_updated`.
2. Commit on a work branch cut from `main` (one branch per queue item,
   e.g. `accuracy/q1-n2-merge`) with a descriptive message. Do not push
   or merge without being asked.
3. Print a summary ≤10 lines: item, verdict (pass / fail / blocked /
   in-progress), key delta with CI, files written, and the single next
   action. If stopped on a question, state it as the last line.

When every queue item is `shipped`, `closed-negative`, or `blocked`,
produce a final summary with proposed next program and stop.

## Initial work queue (mirror of report §7 — details in cited sections)

- **Q1 — N2 MuScriptor merge** (report §3.1). Entry gate already PASSED
  (complementarity 0.3818, `n2_muscriptor_probe_2026-07-21.md`). Work:
  merge variants (cluster-scoped adds, confidence floor) → full dev eval.
  Ship gate: dev OOF lo-95 > 0 vs `highres-ensemble`; GAPS no-reg. STOP
  before player-05. Guard precision: extra_detection is 13% of loss.
- **Q2 — S1b-v2 offline probe** (§3.2). Extract symbolic string/fret
  corpus from the SynthTab JAMS already on disk; train a small masked-
  string contextual model (CPU or free Colab); rescore the banked Phase 0
  ambiguous lattice. Gate: ambiguous top-1 ≥ +0.05 over 0.6770. Reference
  recipes: MIDI-to-Tab (arXiv:2408.05024), Fretting-Transformer
  (arXiv:2506.14223), github.com/Sidmaz666/open-fret (audit, don't vendor).
- **Q3 — S1b-v2 integration** (blocked by Q2 pass). Fine-tune on players
  00–04 symbolic (+GAPS MusicXML for classical arm); integrate as
  constrained rescoring over `candidate_positions()` behind an explicit
  decoder flag. Ship gate: OOF lo-95 > 0; GAPS no-reg. STOP before
  player-05.
- **Q4 — Complementarity probes: Basic Pitch, then YourMT3+** (§3.3).
  Reuse the N1/N2 probe methodology on cached eval audio. Gate:
  P(right | ensemble wrong) ≥ 0.10, else close cheaply. License-check
  YourMT3+ before any shipping use (probe itself is fine).
- **Q5 — Onset snapping prototype** (§4.2). Pre-fuse onset refinement +
  strum-cluster handling. This intentionally changes onsets: must re-run
  full onset/pitch gates, not just Tab F1.
- **Q6 — Inharmonicity offline study** (§4.1). No pipeline code. Gate A:
  string classification ≥0.85 on GuitarSet hex isolated-note regime;
  Gate B: ≥0.70 on mono-mic single-line segments. Fail → banked negative
  like WS4.
- **Q7 — Capo/tuning preflight + capo-covariant priors** (§4.3). Validate
  on synthetic capo shifts of GuitarSet (pitch-shifted audio with
  capo-shifted labels). Behind a flag; `auto` unchanged.
- **Q8 — Review-ranker upgrade from S1b posteriors** (§4.4; blocked by
  Q3). Target: beat 38.76% wrong-position reduction @60 s. Reported
  separately from automatic Tab F1.

## State file template (`docs/accuracy-loop-state.md`)

```markdown
# Accuracy-loop state
last_updated: <date>
current_branch: <branch or none>

## Queue
| id | item | status | key numbers | next action | blockers |
|----|------|--------|-------------|-------------|----------|
| Q1 | N2 MuScriptor merge | open | entry 0.3818 PASS | choose merge variants | — |
| Q2 | S1b-v2 offline probe | open | target amb-top1 +0.05 | extract SynthTab symbolic | — |
| Q3 | S1b-v2 integration | blocked | — | — | Q2 |
| Q4 | Second-opinion probes | open | gate 0.10 | Basic Pitch probe | — |
| Q5 | Onset snapping | open | — | prototype | — |
| Q6 | Inharmonicity study | open | gates 0.85/0.70 | hex B-estimator | — |
| Q7 | Capo/tuning preflight | open | — | design synth-capo eval | — |
| Q8 | Review-ranker upgrade | blocked | beat 38.76%@60s | — | Q3 |

## Questions for the user
- <none>

## Iteration log (newest first)
- <date> — <item> — <one-line outcome + report path>
```
