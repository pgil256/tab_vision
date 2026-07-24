# Q2 / S1b-v2 — contextual string model, offline lattice gate

Accuracy-loop iteration 3 (ROI deep-dive §3.2). Pretrain-only probe: a small
transformer trained on the SynthTab symbolic corpus rescoring Phase 0's
banked ambiguous lattice. Entry gate per
`s1b_entry_substrate_2026-07-22.md` §2: ambiguous top-1 ≥ **0.7048**
(baseline 0.6548, dev-OOF, n = 35,959).

> **Final verdict (both stages): Q2 CLOSED — banked negative.** Stage 2 is
> in §"Stage 2" below; it reached **0.7015 vs the 0.7048 bar**, missing by
> 0.0033. The sections immediately below describe stage 1 (pretrain only).

## Stage 1 verdict — FAIL, with a CI-significant positive

| scorer | best λ | ambiguous top-1 | Δ vs decoder [lo-95, hi-95] | verdict |
|---|---:|---:|---|---|
| decoder (baseline) | — | 0.6548 | — | — |
| **context** | **4** | **0.6850** | **+0.0302 [+0.0163, +0.0446]** | **FAIL vs +0.05** |
| marginal (control) | 0 | 0.6548 | +0.0000 | no effect at any λ |

The gate is missed and the miss is **decisive, not marginal**: the whole 95%
interval sits below the +0.05 bar (upper bound +0.0446). This is not a
sample-size question — at this model scale the effect is real, measured, and
too small.

## The full sweep

Blend: `combined_cost = cost_delta_from_best + λ · (−log p_model(string))`.

| λ | top-1 | solo | comp | rank-2 flip rate |
|---:|---:|---:|---:|---:|
| 0 | 0.6548 | 0.5908 | 0.6896 | 0.0000 |
| 0.25 | 0.6633 | 0.5935 | 0.7012 | 0.0567 |
| 1 | 0.6772 | 0.6000 | 0.7192 | 0.1578 |
| 2 | 0.6832 | 0.6066 | 0.7248 | 0.2291 |
| **4** | **0.6850** | 0.6112 | **0.7251** | 0.3080 |
| 8 | 0.6822 | **0.6195** | 0.7163 | 0.3814 |
| ∞ (model only) | 0.6496 | 0.5849 | 0.6848 | 0.5023 |

Three things this curve says:

1. **Smooth and unimodal**, peaking at λ = 4 — the behaviour of a real signal
   being traded against a real prior, not a threshold artifact.
2. **Model-only is worse than the decoder** (0.6496 < 0.6548). The model is
   not a replacement; it is complementary evidence. That is precisely the
   §3.2 integration shape (an emission term next to the existing cost), so
   the tuned λ transfers directly to Q3 if Q2 ever passes.
3. **The tiers want different λ**: comp peaks at 4, solo at 8. Solo gains
   +0.0287 (0.5908 → 0.6195), comp +0.0355 (0.6896 → 0.7251). A per-tier λ
   is worth ~+0.002 pooled — real, but nowhere near the shortfall.

## Context is the active ingredient, not the corpus

The `marginal` control — P(string | pitch) counts from the *same* 34.06M-note
corpus — is **negative at every λ > 0** and collapses to 0.5419 at λ = ∞. It
never beats λ = 0, so its best is the baseline itself.

That contrast is the scientific content of this iteration. Same corpus, same
lattice, same blend, same code path: counts hurt, sequence context helps by a
CI-significant +0.0302. It independently replicates S1a's negative through a
different mechanism (rescoring rather than prior substitution), and it rules
out "the corpus is simply informative" as the explanation for the positive.

It is also the first time in this repo's history that a SynthTab-derived
artifact has produced a CI-significant positive on an assignment metric.

## The model

`s1b_train_context.py`: 3-layer transformer encoder, d_model 128, 4 heads,
**413,958 parameters**. Inputs are pitch and bucketed inter-onset gap only —
it never sees a string as input and so cannot copy the answer. Output is a
per-note distribution over the six strings, which is exactly what `fuse()`
consumes as an emission.

| | |
|---|---|
| training windows / notes | 60,000 × 64 = **3,840,000** |
| held out | 5% **by track** (val tracks disjoint from train) |
| val accuracy (all notes) | 0.7955 |
| **val accuracy (ambiguous notes)** | **0.7679** |
| epochs | 4 (last truncated by the 1500 s budget) |
| wall-clock | ~25 min, laptop CPU, $0 |

Note the gap between **0.7679 in-domain** (SynthTab held-out tracks) and the
**+0.0302** it delivers on GuitarSet. The model has genuinely learned
position grammar; most of that competence does not survive the transfer.

## Why stage 1 falls short — and the recipe step not yet run

The deep-dive §3.2 recipe has two training stages. **Only the first was
run.** Stage 2 — fine-tune on GuitarSet players 00–04 symbolic — is
unexecuted, and MIDI-to-Tab measures that step alone at **+4.0 pp** string
agreement. +0.0302 + ~0.04 would clear +0.05.

The domain gap is the expected culprit and matches the repo's own recorded
lesson (A15/PDMX, S1a: "domain match beats scale"). SynthTab is
notation-derived (DadaGP transcriptions); GuitarSet is what four players
actually did with their hands. A transcriber writing for readability and a
player choosing a comfortable position disagree systematically about which
of several equivalent positions to use — and that disagreement is the entire
quantity being measured here.

A second, smaller shortfall source: the model is trained on clean symbolic
sequences but scored on the *predicted* note stream, which carries the
ensemble's missed and extra notes. No attempt was made to close that gap.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/s1b_train_context.py --max-windows 60000 --epochs 4 \
  --json ../docs/EVAL_REPORTS/s1b_context_training_2026-07-22.json

python scripts/eval/s1b_rescore_lattice.py --scorer context \
  --checkpoint $TABVISION_DATA_ROOT/models/s1b_symbolic/context_v2.pt \
  --json ../docs/EVAL_REPORTS/s1b_rescore_context_2026-07-22.json
python scripts/eval/s1b_rescore_lattice.py --scorer marginal \
  --json ../docs/EVAL_REPORTS/s1b_rescore_marginal_2026-07-22.json
```

Rescoring is pure CSV replay — ~25 s per sweep, no audio, no pipeline. The
harness refuses `--split held_out_05` outright: player-05 stays sealed until
config freeze plus an explicit user proceed.

Checkpoint: `$TABVISION_DATA_ROOT/models/s1b_symbolic/context_v2.pt`
(git-ignored, CC-BY-NC-4.0 inherited from SynthTab — LICENSES.md). Nothing
registered; `auto` untouched; no SPEC or §8 change.


---

# Stage 2 — GuitarSet fine-tune (leave-one-player-out)

Recipe stage 2 from §3.2, run after the stage-1 near-miss on explicit user
instruction. **Q2 closes here.**

## Protocol — leave-one-player-out is load-bearing

The lattice is players 00-04, so a model fine-tuned on all of 00-04 would be
scored on its own training data. Instead each dev player P gets its own fold
fine-tuned on the *other four* (`context_v2_oof_<P>.pt`), and the harness
scores P's tracks only with the fold that never saw P — the same protocol as
`string_assignment_phase4.py::_oof_position_prior`. Player-05 is never read.

Fine-tune: 8 epochs, lr 5e-5, from the stage-1 checkpoint. GuitarSet dev is
small (121-147 windows per player), so all five folds together took 3m 16s.

| held-out player | 00 | 01 | 02 | 03 | 04 | mean |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous string accuracy | 0.7168 | 0.6754 | 0.7142 | 0.7546 | 0.7035 | **0.7129** |

## Result — FAIL by 0.0033

| stage | best λ | ambiguous top-1 | Δ vs decoder [lo-95, hi-95] |
|---|---:|---:|---|
| baseline (decoder) | — | 0.6548 | — |
| stage 1 (pretrain only) | 4 | 0.6850 | +0.0302 [+0.0163, +0.0446] |
| **stage 2 (+ fine-tune)** | 4 | **0.7015** | **+0.0467 [+0.0291, +0.0640]** |
| gate | | 0.7048 | +0.05 |

The fine-tune added **+0.0165** — real, but well under half the +4.0 pp
MIDI-to-Tab reports for the same step. The point estimate lands **0.0033
below the bar** and the lower bound (+0.0291) is far below it, so the gate
fails on the house rule (acceptance = lo-95 ≥ target) and on the point
estimate alike.

**Two honest deductions, both pushing the true value down:**

- **λ was selected on the evaluation set.** The sweep picks the best of nine
  λ values on the same dev-OOF lattice it reports, so +0.0467 is optimistic
  by an unmeasured amount. A nested or held-out λ selection would only lower
  it.
- The λ curve is flat near its peak (0.6996 at λ=2, 0.7015 at λ=4, 0.6970 at
  λ=8), so the peak is not a knife-edge artifact — but neither is there a
  better λ hiding between grid points.

## The finding worth keeping: context helps chords, not single lines

| tier | baseline | stage 2 | Δ |
|---|---:|---:|---:|
| comp (strummed) | 0.6896 | **0.7557** | **+0.0661** |
| solo (single-line) | 0.5908 | 0.6020 | **+0.0112** |

Comp gains **6× what solo does**. That asymmetry is the most useful thing
this probe produced, and it is bad news for the lever:

- Wrong-position is 57.3% of all Tab F1 loss but **77.5% of single-line
  loss** (deep-dive §2). The tier that needs contextual disambiguation most
  is the one that got almost none of it.
- The mechanism is legible. In a chord, the voicing constrains its own
  members — pick one note's string and the rest follow, and a model that has
  seen millions of voicings learns that grammar. A single line has no such
  simultaneous constraint; resolving it needs hand-position continuity across
  *time*, which is exactly what the existing `guitarset-seq-v1` transition
  prior already models, so the contextual model is largely re-deriving
  evidence the decoder has.
- This also explains the shortfall arithmetic: solo is 35% of the ambiguous
  notes, and it contributed +0.004 of the +0.0467.

## Verdict — Q2 CLOSED, banked negative

The full §3.2 recipe was executed: pretrain at scale on 34M symbolic notes,
then fine-tune on in-domain performance data under proper OOF. It ends
+0.0467 [+0.0291, +0.0640] against a +0.05 bar, with a λ chosen on the eval
set. The gate is not met and the remaining levers are either
already-measured-small (per-tier λ: both tiers peak at λ=4) or a different
model rather than a tweak (masked-string conditioning on neighbouring
strings, autoregressive decoding). Per house rule, the negative is banked
rather than iterated past.

**What a future session should know before re-opening:** the ceiling is not
the problem — gold is in the lattice for 99.72% of ambiguous notes — and
context is demonstrably real evidence (+0.0467 CI-significant, with the
counts control at exactly 0.0000). What failed is that the available context
signal concentrates in polyphony, where the decoder was already strongest.
Any re-opening should target **single-line** disambiguation specifically
rather than repeating a general contextual model, and should budget for a
λ-selection protocol that does not touch the reported slice.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data python scripts/eval/s1b_finetune_guitarset.py   --json ../docs/EVAL_REPORTS/s1b_finetune_2026-07-22.json

python scripts/eval/s1b_rescore_lattice.py --scorer context-oof   --json ../docs/EVAL_REPORTS/s1b_rescore_context_oof_2026-07-22.json
```

**Note:** the lattice CSV (`string_assignment_phase0_2026-07-15_notes.csv`,
70 MB) is **git-ignored** — it exists only in the working tree, not in the
repo. Pass `--lattice` explicitly when running from a git worktree.

Fold checkpoints: `$TABVISION_DATA_ROOT/models/s1b_symbolic/context_v2_oof_*.pt`
(git-ignored; NC inherited from the SynthTab-pretrained initialization —
LICENSES.md). Nothing registered; `auto` untouched; player-05 sealed.
