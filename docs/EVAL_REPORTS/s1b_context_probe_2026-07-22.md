# Q2 / S1b-v2 — contextual string model, offline lattice gate

Accuracy-loop iteration 3 (ROI deep-dive §3.2). Pretrain-only probe: a small
transformer trained on the SynthTab symbolic corpus rescoring Phase 0's
banked ambiguous lattice. Entry gate per
`s1b_entry_substrate_2026-07-22.md` §2: ambiguous top-1 ≥ **0.7048**
(baseline 0.6548, dev-OOF, n = 35,959).

## Verdict — FAIL, with a CI-significant positive

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

## Why it falls short — and the one recipe step not run

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
