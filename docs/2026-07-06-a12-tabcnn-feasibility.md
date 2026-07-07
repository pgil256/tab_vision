# A12 — timbral string-ID (TabCNN) feasibility verdict (2026-07-06)

**Question (roadmap A12, Tier 3 / approval-gated):** can a pretrained timbral
string-ID model be dropped in as a second audio pass, its per-string posterior
consumed through the existing `AudioEvent.fret_prior` channel, to break the
single-line information ceiling? The gate rule: *pretrained weights must exist,
else it becomes a rule-8 training spend requiring sign-off.*

## Bottom line — **BECOMES A TRAINING SPEND (soft no on current evidence)**

The deciding fact: **no pretrained weight files exist** for TabCNN or any model
in its lineage. Every candidate ships *training code only* — using it means
running the 6-fold GuitarSet training yourself, which is exactly the rule-8
spend the gate exists to catch. Recommendation: **do not authorize** on current
evidence; the expected lift is too thin to justify a training run + a new
front-end (see below).

## Weights & license (the crux)

| Model | Weights published? | License | Framework |
|---|---|---|---|
| **TabCNN** (Wiggins & Kim, ISMIR 2019) — the named candidate | **No** (`model/` = 3 `.py` files, "no releases") | **None** (no LICENSE → all rights reserved; code itself non-shippable) | Keras/TF |
| guitar-transcription-with-inhibition (successor) | **No** (train via CV script) | MIT | PyTorch |
| FretNet / guitar-transcription-continuous | **No** (train via CV script) | MIT | PyTorch |

`github.com/andywiggins/tab-cnn` · `github.com/cwitkowitz/guitar-transcription-with-inhibition` · `github.com/cwitkowitz/guitar-transcription-continuous`. No HuggingFace/Zenodo-hosted tab checkpoint surfaced. The successors are MIT (code reusable) but the original TabCNN is **unlicensed** — so even setting weights aside, TabCNN's *code* is not shippable; only the cwitkowitz repos would be a safe training base.

## Integration surface in this repo (ready — the model is the only gap)

- **`AudioEvent.fret_prior`** (`tabvision/types.py:52`) is exactly the channel the posterior would flow through — already consumed by `playability.emission_cost` (2D `[string, fret]` prior → `-log P`). **A3 adds the weight knob** on that very term (`TABVISION_FRET_PRIOR_WEIGHT`), so an A12 posterior would ride A3's infrastructure with a strict no-regression gate — the two compose exactly as the roadmap describes.
- **Backend registry** (`tabvision/audio/backend.py`): register a factory under a name, satisfy the `AudioBackend` protocol, populate `fret_prior`. Low-friction.
- **Stub discrepancy:** `tabvision/audio/tabcnn.py` is a 6-line stub whose docstring references a *"trimplexx CRNN model"*, **not** Wiggins TabCNN. Reconcile before any build — the stub's stated intent already diverges from the A12 candidate.
- **Input representation is a separate front-end:** TabCNN consumes a CQT (22 050 Hz, hop 512, 60 bins/oct, 360 bins, 7-frame context) → per-string softmax over frets 0–19 + "no-play". Our path is Basic-Pitch/highres-based, so this is a distinct feature stage, not a reuse.

## Expected value — marginal, and measured on a favorable population

Paper numbers on GuitarSet: multipitch F1 0.86; **tablature F1 0.75**, **TDR (tablature disambiguation rate) 0.84**. TDR is the direct analogue of our audio playability prior's **0.778** on contested strings, and 0.84 > 0.778 suggests a *real but modest* in-domain lift. Three caveats erode it:
1. TDR 0.84 is over **all** notes, not the hard contested-string subset where our 0.778 is measured — not apples-to-apples; the true delta on ambiguous cases is unknown and plausibly smaller.
2. TabCNN's own tab F1 (**0.75**) is **below our current in-domain ~0.815** — its value would be strictly as a string-posterior feed, never as a standalone transcriber.
3. Acoustic/GuitarSet-trained → no electric transfer (consistent with our cross-dataset findings).

Net: plausible small win, high uncertainty, and it costs a training run just to test.

## Recommendation to the user

1. **Hold A12** — the "pretrained weights confirmed to exist" gate condition is **not met**. Proceeding requires authorizing a training run (rule-8 sign-off), and on TDR 0.84-vs-0.778 the expected lift doesn't clear that bar.
2. If ever revisited: prefer the **MIT-licensed** FretNet/inhibition code over unlicensed TabCNN as a training base; and first re-scope the expected lift against the **contested-string subset specifically** (a zero-spend re-analysis of our existing GAPS/val24 caches) before spending on a train.
3. **Cheapest zero-spend next step** if string-ID is still wanted: reconcile the `tabcnn.py` stub ("trimplexx CRNN" vs Wiggins) — a 10-minute check of whether *that* named model has any downloadable weights.

**The A12 lever is not dead — it is the one bounded bet past the single-line
ceiling — but it is a spend, not a drop-in, and the current numbers don't
justify the spend.** Filed for the decision packet alongside D2 (electric).
