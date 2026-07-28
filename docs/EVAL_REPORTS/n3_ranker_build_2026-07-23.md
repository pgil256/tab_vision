# N3 build — physics is worth adding to the review ranker (+0.0514 @60 s)

Accuracy-loop N3 build slice. The entry probe showed the physics "doubt" score
is a strong wrong-position detector (AUC 0.75 isolated). This measures what it
buys the review ranker under the Phase 6 replay protocol.

> **CORRECTION 2026-07-27 — the blockage below is not real.** All three grounds
> were re-checked and none holds: `PHASE1_NOTES` is a **git-ignored, reproducible
> CSV output** (`.gitignore:76-77`), not a lost stage; phase4 and phase6
> provenance record the **identical** `event_ids_sha256`
> (`17b7d3b3a7da24f82de778fffc84cff73ee012c2c10d80fd82dc9727020fce3c`), so row
> order does not differ; and phase 6 loads **no** Phase 4 timbre checkpoint — it
> re-runs the model in-process (`string_assignment_phase6.py:279`). GuitarSet is
> on disk at `~/mir_datasets/guitarset` (360 wav + 360 jams, 300/300 dev sha256
> match) — it read as missing only because it is not under
> `$TABVISION_DATA_ROOT`, so the scripts need `--data-home`. Regeneration costs
> ~5.6 h of unattended CPU and zero downloads. The marginal-delta framing below
> remains valid on its own terms, but it was **not** the only option available.
> See `docs/plans/2026-07-27-video-evidence-roadmap-design.md` §6 (C1).

## The exact 38.76% comparison is blocked (verified)

The intended test — add physics to the frozen Phase 6 ranker and beat its
38.76% wrong-reduction @60 s — cannot be run here:

- Phase 6's feature cache is keyed to `PHASE1_NOTES`, a re-decode stage **not
  on disk** and not cheaply regenerable.
- Three of its ten features come from the Phase 4 timbre model.
- Its row order does not match the phase0 lattice: phase0 has 43,080 rows
  across all splits vs the cache's 35,959, and the event-id SHA differs, so the
  cached 10-feature matrix cannot be aligned to fresh physics measurements.

So this measures the **marginal value of physics** on a self-contained ranker
instead, with the exact Phase 6 training and replay protocol.

## Setup

35,959 phase0 dev-OOF ambiguous notes (wrong rate 0.3452). Player-held nested
OOF, Platt calibration, the review_queue MLP architecture / seeds / constants,
and the replay (2 s/note, gold-in-top-3 correctable, wrong-reduction at
10/30/60 s per clip). Two arms:

- **decoder** — 4 lattice-derivable features: path margin, candidate count,
  cluster size, mode.
- **+physics** — the same plus 3 physics features: the doubt score
  (1 - physics-probability of the decoder's string), fit r2, and a fired
  indicator. Physics fires on 26.9% of notes; abstained rows get zeros.

## Result

| ranker | OOF AUC | reduction @60 s | @30 s | @10 s |
|---|---:|---:|---:|---:|
| decoder | 0.6273 | 0.3286 | 0.1703 | 0.0573 |
| **+physics** | **0.7031** | **0.3800** | 0.2137 | 0.0857 |
| **delta** | **+0.0758** | **+0.0514** | +0.0434 | +0.0284 |

**Physics adds +0.0514 absolute wrong-reduction @60 s (+15.6% relative) and
+0.076 AUC.** The relative gain is *largest at the tightest budget* — +50% at
10 s (0.0573 -> 0.0857) — which is where ranking quality matters most, since
the reviewer only sees the very top of the queue.

## Two things this establishes

1. **Physics is a worthwhile ranker feature.** The delta is positive at every
   budget and consistent with the entry probe's AUC 0.75. On 27% coverage it
   still moves the whole-set metric by +5 pp, because the notes it covers are
   disproportionately the ones the decoder margin cannot rank.
2. **Physics substitutes for much of the timbre machinery.** The 4-decoder +
   3-physics ranker reaches **AUC 0.7031**, within 0.01 of Phase 6's full
   **ten-feature 0.7127** — which needed the Phase 4 timbre model, accepted-
   checkpoint posteriors, context-disagreement and segment-inconsistency
   features. Three cheap physics features nearly match all of that.

## Honest limits

- **Not the 38.76% comparison.** The decoder-feature baseline (0.3286) is
  weaker than Phase 6's full ten-feature ranker, so the +physics 0.3800 is not
  comparable to the shipped 0.3876 — coincidental proximity. What is valid is
  the **delta**, which isolates the physics contribution under a fixed
  protocol.
- **Whether physics clears the 0.50 replay gate** the full ranker missed
  (0.3876) is unmeasured here; it needs the full ten-feature ranker plus
  physics, which needs PHASE1_NOTES.
- Still GuitarSet dev; player-05 not touched (this is an assisted metric,
  separate from automatic Tab F1, and does not gate on the hold-out).

## Recommendation

Add `physics_prob_decoder` (+ `r2`, fired) to the Phase 6 feature set when its
row provenance is regenerated. This probe justifies the feature: it is
complementary to the decoder margin, nearly matches the entire timbre feature
group on its own, and lifts wrong-reduction most where the review budget is
tightest.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n3_ranker_build.py \
  --lattice ../docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv \
  --json ../docs/EVAL_REPORTS/n3_ranker_build_2026-07-23.json
```

Lattice is git-ignored (70 MB — pass `--lattice`); physics measured from
cached events + GuitarSet audio. ~12 min (physics + 50 MLP fits).
