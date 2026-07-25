# TabVision Demo Assets

Portfolio-facing assets, all generated from automated runs and public data —
no private recordings and no hand annotation are on the critical path.

| Asset | What |
|---|---|
| `demo.gif` | Terminal capture: `tabvision transcribe` printing a confidence-graded ASCII tab from a real decoded GuitarSet excerpt. |
| `architecture-brief.md` | One-page overview of the pipeline for portfolio publication. |
| `per-tier-examples.md` | The project thesis in one table: the same piece single-line vs. strummed, ground truth vs. output. |
| `fresh-user-path.md` | The verified from-scratch install and first-run path. |
| `release-evidence.md` | Links to the automated reports and release checks that back the narrative's claims. |
| `sample-a440-ascii.tab` | Pinned synthetic-tone output used as the desktop shell's bootstrap smoke golden. |

Accuracy numbers live in `../EVAL_REPORTS/`, not here — cite them rather than
restating them, so there is exactly one place to update. The current default's
headline run is
[`player05_batched_confirm_2026-07-24.md`](../EVAL_REPORTS/player05_batched_confirm_2026-07-24.md).

Electric tiers are **v2** (SPEC §1.4.1), so there is deliberately no
clean-electric or distorted-electric example here.

Do not commit large raw clips. Keep source media under `$TABVISION_DATA_ROOT`
and commit only small, publication-ready derivatives.
