# String assignment Phase 2: constrained contextual candidate reranker

## Decision

**Close symbolic-context expansion.** The fixed contextual architecture was evaluated once with player-held-out OOF predictions; no post-result model or hyperparameter search was run.

Selected predeclared composition: `context_segment`. The artifact is `unregistered` and automatic routing therefore remains on the last gate-passed baseline unless all promotion criteria passed.

## Development OOF results

| condition | aggregate macro | micro | solo macro | comp macro | ambiguous top-1 | top-3 | wrong rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.5581 | 0.5577 | 0.5460 | 0.5702 | 0.6659 | 0.9949 | 0.3341 |
| segment | 0.5585 | 0.5579 | 0.5470 | 0.5701 | 0.6662 | 0.9949 | 0.3338 |
| control_baseline | 0.5619 | 0.5622 | 0.5459 | 0.5779 | 0.6722 | 0.9998 | 0.3278 |
| context_baseline | 0.5614 | 0.5616 | 0.5456 | 0.5771 | 0.6712 | 0.9998 | 0.3288 |
| context_segment | 0.5617 | 0.5618 | 0.5464 | 0.5771 | 0.6715 | 0.9998 | 0.3285 |

- Linear-control ambiguous top-1 gain: `+0.0062`.
- Selected contextual ambiguous top-1 gain: `+0.0056`.
- Aggregate Tab F1 delta / paired 95% CI: `+0.0036` `[+0.0018, +0.0055]`.
- Comp delta / paired 95% CI: `+0.0069` `[+0.0037, +0.0104]`.
- Player-fold deltas: `00 +0.0051`, `01 -0.0014`, `02 +0.0012`, `03 +0.0133`, `04 +0.0000`.
- Fold best epochs: `{"context": [12, 1, 5, 14, 1], "control": [6, 5, 4, 8, 3]}`.
- Optional PDMX pretraining: **not run** — the GuitarSet-only aggregate gain was below +0.015
- Ambiguous-note rows mean events with at least two physically playable pitch-preserving candidates; baseline and candidates use the same pool.

Onset and pitch events are byte-for-byte unchanged by construction; their frozen benchmark values remain onset F1 `0.9302` and pitch F1 `0.9154`.

## Frozen player 05 confirmation

| condition | aggregate macro | solo macro | comp macro | ambiguous top-1 | wrong rate |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6126 | 0.5418 | 0.6834 | 0.6809 | 0.3191 |
| segment | 0.6143 | 0.5453 | 0.6834 | 0.6819 | 0.3181 |
| context_segment | 0.6152 | 0.5453 | 0.6850 | 0.6840 | 0.3160 |

## Artifact, runtime, and safety

- TorchScript artifact: `tabvision/tabvision/fusion/priors/context_v1.pt` (`414632` bytes, SHA-256 `9d0df2eba2c99f2271e2932e7b791cb9ebb228c6b1d44bb27099fcd66ce56fea`).
- Manifest: `tabvision/tabvision/fusion/priors/context_v1.manifest.json`; architecture is below 500,000 parameters.
- Added context inference: `0.520` s per 60 s; projected pipeline total `45.52` s.
- Peak process memory: `755650560` bytes.
- Frozen prediction hash: `d3ab8c8a96302e6e978374815c5e6a4caf3dcb3b50fa1ffde04a03565ef84109`; deterministic rerun: `d3ab8c8a96302e6e978374815c5e6a4caf3dcb3b50fa1ffde04a03565ef84109`.
- Context routing is restricted to clean acoustic, standard tuning, capo 0. Classical/electric/out-of-domain requests fall back to baseline.
- Missing, corrupt, incompatible, and unregistered artifacts fall back to baseline.

## Reproducibility

Training used CPU PyTorch from `tabvision/.venv`, deterministic algorithms, fixed seeds, inverse joint-frequency weighting, and early stopping on held-out macro Tab F1. Error rows are written beside this report for grouping by string, offset, fret, pitch, candidate count, style, and player.
