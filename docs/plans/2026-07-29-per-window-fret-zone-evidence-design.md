# Per-window fret-zone evidence — design (pre-registration draft)

**Status: DESIGN — awaiting sign-off. No build until approved** (SPEC §0
rule 8; plan-doc-first workflow). Gates in §7 and §8 are written before any
run and must not be edited after numbers are seen, per the wire-sparse
precedent (`a8f5f2e`) and the Stage 1 precedent (`d6c4c89`).

**Read §3 before §5.** This program was requested as "the learned per-window
fret-zone predictor". The headroom is real and the gap is real, but four
previous *fitted audio* channels aimed at the same error have failed, and the
decision log explains why in a way that predicts a fifth failure. The design
below therefore leads with a **non-learned** phase that costs an afternoon
and is a strictly better experiment; the learned model becomes Phase B, with
an explicit entry gate. If you want the learned model attempted first
regardless, that is a one-line change to §8 — but §3 is the argument against
it.

## 1. The opportunity

Same-pitch-wrong-position is the dominant Tab F1 loss, and it is
overwhelmingly a *fret-zone* ambiguity rather than an abstract string
ambiguity:

- **Every ambiguous pair is ≥ 4 frets apart; 99.4% are 4 or 5 frets apart**
  (`docs/EVAL_REPORTS/q6_separability_2026-07-22.md`, n = 35,959 dev-OOF
  ambiguous notes). Choosing the zone therefore *chooses the string*.
- **An oracle supplying one fret zone per 1 s window is worth +0.2756 macro
  Tab F1**; per 4 s joint window, +0.1446
  (`docs/EVAL_REPORTS/string_assignment_phase0_2026-07-27.md`). This is by a
  wide margin the largest unclaimed headroom measured in the repo.
- Nothing currently supplies that signal from audio. `segment-v1` has a
  latent fret-zone variable (`zone_centers = (2, 5, 7, 10, 13)` in
  `tabvision/tabvision/fusion/segment_decoder.py`) but infers it from pitch
  geometry and priors alone, and banked **+0.0004** OOF. The only real
  fret-window channel, `position_window_prior.py`, is fed by **video**.

**Honest reading of the +0.2756.** `apply_gold_oracle`
(`tabvision/scripts/eval/string_assignment_oracles.py`) selects, per window,
the state that *maximises correctness against gold* out of
`FRET_ZONES = ((0,4), (3,7), (5,9), (7,12), (10,15))` plus a neutral state.
That is an upper bound on **any** per-window zone selector, and it is
strictly above what a perfect *true-zone* predictor would score, because the
true zone is not always the error-minimising choice. Treat +0.2756 as the
ceiling of the mechanism, never as a forecast.

## 2. The graveyard — what is already refuted

Any proposal here is arguing against a long run of banked negatives. Stating
them up front is the point, not a formality.

| attempt | modality | gate | achieved | verdict |
|---|---|---|---|---|
| Phase 2 compact timbral ranker | audio | ambig top-1 ≥ +0.05 | **−0.0218** | FAIL |
| Phase 4 native-rate timbral | audio | ≥ +0.05 | +0.0072 [−0.0152, +0.0291] | FAIL |
| Phase 5 direct per-string net | audio | ambig top-1 ≥ 0.7121 | **0.5920** | FAIL |
| Phase 6 learned error detector | audio-derived | AUC ≥ 0.75 / 2.0× / 50% | 0.7127 / 1.77× / 38.76% | FAIL ×3 |
| `context-v1` reranker | symbolic | ≥ +0.02 ambig, ≥10% wrong-pos | +0.0056, 1.7% | FAIL |
| Q2/S1b context transformer | symbolic | ambig top-1 ≥ 0.7048 | 0.7015 | FAIL by 0.0033 |
| S1a SynthTab count priors | symbolic | CI-sig dev gain | −0.1941 … +0.0002 | FAIL |
| A15 PDMX n-gram | symbolic | dual no-regression | −3.6 pp / −0.9 pp | FAIL |
| WS4 / Phase D string resolver | video | val 6-way ≥ 0.45 | **0.2919** | FAIL |
| Stage 1 segment-window rerank | video/oracle | ≥ +0.010 | **+0.0000** | FAIL |

Three standing rules follow, and this design must satisfy all three:

1. **"Domain match beats scale"** — established three times independently
   (A15/PDMX at 5× data, S1a at 54–199× data, S1b's 34 M-note pretrain).
   More data is not the missing ingredient.
2. **Weak evidence washes out against a strong prior.** The position prior is
   already ~0.65 top-1 on exactly the contested notes; a channel at
   AUC ≈ 0.71 does not move it. The recorded re-opening guidance is *not*
   better features but **"how weak per-note evidence is combined"**.
3. **Reranking is dead.** `docs/EVAL_REPORTS/segment_window_stage1_2026-07-29.md`:
   a *gold* window oracle reranking `segment-v1`'s K-best returned +0.0000,
   abstaining 12/12, because the retained paths are the same tab. Ceiling for
   any reranker over that path set is **+0.000385** at k=25. **Any new
   channel must enter through the `AudioEvent.fret_prior` product-of-experts
   chain, not as a reranker.**

## 3. Why the *learned* version should not be attempted first

The one channel that worked is `acoustic-physics-v1`: **+0.0780
[+0.0502, +0.1078]** on player-05 confirmation, +0.0522 on sealed player 04
(`docs/EVAL_REPORTS/phase0_rotation_baseline_2026-07-25.md`). The decision
log's explanation of *why* it worked where every fitted audio model failed is
the load-bearing argument of this document:

> it contributes an **absolute physical prediction independent of the
> position prior**, rather than a discriminative direction learned from the
> same distribution the prior already models.

A learned zone predictor trained on GuitarSet fingerings is, structurally, a
discriminative direction learned from the same distribution `guitarset-v1`
already encodes. That is precisely the shape that failed four times. It is
not that learning is forbidden; it is that **a learned model must add
information the prior does not already have**, and a model fit to the same
labels mostly does not.

Meanwhile the physics channel already extracts exactly the right quantity and
is being under-used in a specific, fixable way.

### The mechanism nobody has tried

Inharmonicity is a function of **both** string and fret
(`tabvision/tabvision/fusion/inharmonicity.py`):

```
B(s, n) = B0_s · 2^(n/6)
```

So a readable `B` measurement localises the *position*, not merely the
string — that is why the same-pitch candidates, 4–5 frets apart, separate by
a 1.59–1.78× `B` ratio from length alone. On notes where it is readable the
channel measures **0.92 string accuracy against a 0.65 count-prior control**.

**Coverage is the entire constraint**, and the module says so:

> a note must ring alone for the partial structure to be readable, which is
> ~34% of solo notes and ~1% of strummed ones. This channel is a single-line
> instrument by construction.

And critically — **the evidence is applied per-note and never propagated.**
Every "window" in `inharmonicity.py` is an FFT analysis window; there is no
temporal aggregation across notes. A note that rings alone pins the hand
position, and the note 200 ms later, contaminated and unreadable, inherits
*nothing* — even though the hand did not move.

That is the gap, and it lines up with all three standing rules:

- It is an **absolute physical** measurement propagated, not a fitted
  discriminative direction (rule 1 and 2).
- It is precisely **"how weak per-note evidence is combined"** — the exact
  re-opening the log recommends (rule 2).
- It enters through `fret_prior`, where the physics channel already enters,
  not as a reranker (rule 3).
- It targets **single-line** specifically, which is where 77.5% of the
  single-line loss lives and where physics has its coverage — and the log's
  other explicit guidance is *"target single-line disambiguation
  specifically rather than a general contextual model."*

The arithmetic that makes this promising is one number nobody has measured:
per-note coverage is ~34% on solo, but a 1 s window contains several notes.
**If a window has 3–4 notes and each is independently ~34% readable, most
windows contain at least one readable note.** Sparse per-note coverage
becomes dense per-*window* coverage — and the oracle says the window is
exactly the scope where position knowledge is worth +0.2756. Phase A0 (§7)
measures that number first, because if it is low the whole program stops for
an afternoon's cost.

## 4. Claim under test

> Within a short window the fretting hand occupies one position. Propagating
> the physics channel's high-confidence per-note position measurements to
> their unreadable neighbours in the same window yields a Tab F1 gain that
> per-note application does not, concentrated on single-line material.

The null this must beat is not "no evidence" — it is **the shipped
`acoustic-physics-v1` channel applied per-note**, which is already in the
default pipeline. Phase A must show a gain *over that*, not over a
physics-free baseline.

## 5. Design sketch

No SPEC §8 change. No new runtime dependency. New code lives in eval scripts
until a gate passes; production wiring is a separate approval.

1. **Window partition.** Reuse the oracle's own cluster-safe partition,
   `fixed_window_groups(events, indices, window_s)` from
   `string_assignment_oracles.py`, so the measured headroom and the mechanism
   are defined over identical windows. Primary `window_s = 1.0`; 2.0 and 4.0
   reported as secondary.
2. **Per-note position posterior.** From the existing
   `measure_events` / `attach_inharmonicity_evidence` output: for each note
   with a fit clearing `min_r2`, a posterior over its playable candidates —
   already computed today, already bounded, already abstaining where
   unreadable.
3. **Window zone posterior.** Combine the per-note posteriors in a window
   into a posterior over the five `FRET_ZONES` plus neutral, weighted by each
   note's own fit quality (`r2`, partial count) — *at its own confidence, not
   full weight*, which is the recorded Track B recommendation. Abstain when a
   window contains no readable note.
4. **Propagate.** Multiply the window zone posterior into the `fret_prior` of
   **every** note in the window, including the unreadable ones, as one more
   bounded expert in the existing product-of-experts chain — the same place
   and the same order `pipeline.py` already runs physics before
   `apply_pitch_position_prior`. Cap the log-odds contribution so the channel
   can break a tie and never veto.
5. **No new learned artifact in Phase A.** Deterministic code and frozen
   constants, exactly as Phase 1 of the segment program.

## 6. Eval protocol (binding for every phase)

- **Corpus.** GuitarSet, 360 tracks, 6 players, JAMS string+fret labels.
  Both tiers reported separately — `solo` (single-line) and `comp`
  (strummed) — because this channel is solo-biased *by construction* and an
  aggregate-only number would hide that.
- **Splits.** The 2026-07-25 rotation, not the pre-rotation convention:
  `dev = {00, 01, 02, 03, 05}` (300 clips), **sealed = {04}** (60 clips).
  Development decisions are made on dev only; the sealed player is opened
  once, at the end, for confirmation.
- **Leak-free priors.** Every clip scored under priors rebuilt **without its
  own player**, per `tabvision/scripts/eval/phase0_rotation_baseline.py`.
  The shipped `guitarset-v1` / `guitarset-seq-v1` artifacts **must not** be
  used when measuring on player 04 — they were built excluding player 05 and
  have therefore *seen* player 04.
- **Statistics.** Paired, clip-stratified bootstrap, 10,000 resamples,
  seed 42; report the 95% CI on every headline number; the gate is on the
  **CI lower bound**, not the point estimate.
- **Scale.** ≥ 20 clips and ≥ 500 scored notes per tier per reported cell.
- **Pre-registration.** Every free constant frozen in this document and
  committed before the first run, Stage 1 style. Any outcome not matching a
  gate below is a banked negative — no post-hoc statistic substitution.
- **Baseline.** The current default: `highres-ensemble` + LOPO position
  prior + `acoustic-physics-v1` per-note physics with partial-aware
  isolation. Not a physics-free arm.

## 7. Phase A — non-learned, ~1 day, $0

### A0 — the coverage arithmetic (half a day, decides the program)

Pure measurement on GuitarSet dev, no mechanism, no gate on Tab F1:

1. Fraction of 1 s / 2 s / 4 s windows containing **at least one** note whose
   inharmonicity fit clears `min_r2`, split solo vs comp.
2. Among those windows, how often the readable notes' implied zone **agrees
   with the gold zone** of the window (this is the channel's realistic
   precision, replacing Stage 1's gold oracle with a physics oracle).
3. How many *unreadable* notes sit in a window with ≥ 1 readable note — the
   population the propagation would actually reach.

**Gate A0 (go/no-go for A1):** solo windows with ≥ 1 readable note
**≥ 0.60**, and implied-zone agreement with gold on those windows
**≥ 0.75**. Below either, the channel cannot cover or cannot aim, the
propagation is pointless, and the program stops here with a banked negative
for the cost of one afternoon.

#### Frozen A0 constants (written before the first run, 2026-07-29)

**Notes.** Gold-timed notes from the GuitarSet JAMS, not the detected
stream. A0 is a feasibility ceiling: gold onsets/offsets remove transcription
error and mistimed isolation as confounders, so the coverage it reports is an
**upper bound** on what the detected stream can reach. If A0 fails on gold it
cannot pass on detected. A1 measures on the detected stream.

**Physics settings** are read from the registered `acoustic-physics-v1`
artifact via `load_string_evidence`, never hardcoded, so a later artifact
change cannot silently alter what A0 measured. At time of freezing these
resolve to `min_r2 = 0.5`, `isolation = "partial_aware"`, `sigma = 0.35`,
`weight = 1.0`, `fret_exponent = 1.0`, `MIN_CLEAN_PARTIALS = 4`.

**Readable note.** `measure_events` returns a fit for the note and that fit's
`r2 >= min_r2`. Nothing else counts as readable.

**Implied position of a readable note.** The playable candidate for its pitch
minimising `|fit.log_b − model.predicted_log_b(string, fret)|`. This is the
physics channel's own scoring (`inharmonicity_matrix` is Gaussian in that
same distance, so its argmax is this argmin), so A0 cannot flatter the
channel with a different rule than production uses.

##### Amendment, 2026-07-29, before the first full run — the calibration arm

The rule above was run on two tracks as a harness check and returned
implied-position accuracy **0.1975** against the ~0.92 this section named as
the sanity anchor. Per the paragraph below, that means the harness was wrong,
and it was — in a way worth recording rather than quietly patching.

**Diagnosis.** Measured `log B` sits a systematic **+0.52** above the
reference table's prediction *at the gold position* (residual std 0.582 to
gold versus 1.487 to alternatives, so the table's shape is right and only its
offset is wrong). +0.52 in log B is a **1.68×** ratio — almost exactly the
1.59–1.78× that separates two candidates 4–5 frets apart
(`docs/EVAL_REPORTS/q6_separability_2026-07-22.md`). A whole-table offset of
one candidate-step is enough to move the argmax to the wrong candidate
nearly every time.

**Why the anchor did not apply.** The 0.92 was measured under **per-player
B0 calibration**, stated in that report's own limitations
(*"Per-player B0 calibration is leave-one-player-out but same-instrument-set"*).
The shipping table is a reference table applied raw, and
`docs/EVAL_REPORTS/q6_self_calibration_2026-07-22.md` shows why that is still
sound in production: the channel is *soft evidence multiplied into the
prior*, not an argmax, so a mis-centred table still helps (+0.0525 lopo,
+0.0388 self-seeded) even though its top-1 is poor. **0.92 was never the
shipped table's argmax accuracy, and citing it as A0's anchor was my error.**

**Amended rule — three arms, all sharing one set of spectral fits:**

- `reference` — the shipped table applied raw. The floor, and what ships today.
- `self_seeded` — `calibrate_from_session(seed=reference)` with provisional
  positions from a **physics-free first pass**: the `guitarset-v1` position
  prior's top-1 per note. **Fully label-free and deployable**; mirrors the
  `self-seeded` arm q6 measured at +0.0388, whose provisional labels likewise
  come from a first *decode* (~0.65 top-1). Seeding instead from the
  mis-centred table's own 0.20 argmax was tried first and merely recycles its
  error (agreement 0.19 vs 0.66 per-note) — recorded so the distinction is
  not lost. The prior artifact is in-sample for players 00–03, which makes
  this arm mildly optimistic; A0 is already declared an upper bound.
- `gold_calibrated` — the same refit with gold provisional positions. Uses
  labels, so it is the ceiling, reported for bracketing only.

**Gate A0 is read on `self_seeded`** — the deployable arm — at solo, 1 s.
The two gate values (0.60 / 0.75) are unchanged from the merged design and
are not being touched with a number in view; only the arm the implied zone is
computed from is being corrected, because the original rule measured a
mis-centred table's argmax and no propagation design would ever have used
that. The revised sanity anchor is `gold_calibrated`, which should approach
the ~0.92 regime; if it does not, the harness is still wrong.

**Windows.** `fixed_window_groups` from
`tabvision/scripts/eval/string_assignment_oracles.py` at its default
`cluster_gap_s = 0.080`, so A0's windows are bit-identical to the windows the
+0.2756 oracle was measured over. Primary `window_s = 1.0`; 2.0 and 4.0
reported as secondary.

**Zones.** `FRET_ZONES = ((0,4), (3,7), (5,9), (7,12), (10,15))`, unchanged
from the oracle. Open notes (`fret == 0`) are excluded from every zone
determination — the open-string exemption is inherited, not relitigated.

- *Gold zone set* of a window = the zones containing **every** gold fretted
  fret in it. Windows where no single zone covers all of them are counted and
  reported separately as **hand-moved windows**, and excluded from the
  agreement denominator — the single-zone assumption is false there by
  construction, and §10 already names this as a failure mode.
- *Implied zone set* = the zones containing every implied fret of the
  window's readable fretted notes.
- **Agreement** iff the two sets intersect.

**Splits.** dev = players {00, 01, 02, 03, 05} only. **Sealed player 04 is
not read in A0.** Tiers split by track-id suffix (`_solo` / `_comp`) and
always reported separately; the gate is read off **solo**.

**Reported alongside (not gated):** per-note implied-position accuracy on
readable notes, as a sanity check against the channel's documented 0.92 on
isolated notes — a materially different number means A0's harness is wrong,
not that the channel changed.

### A1 — the propagation itself

Implement §5 and measure on dev, both tiers.

**Gate A1:** single-line (solo) aggregate Tab F1 delta **CI lower bound
> 0** over the physics-per-note baseline, **and** strummed (comp)
non-inferiority at CI lower bound **≥ −0.002**, **and** no per-player
regression worse than −0.005. Runtime within the existing +20% decode
allowance.

**Confirmation:** only if A1 passes on dev, open sealed player 04 once and
report; promotion to the default requires the sealed delta's CI lower bound
> 0 on solo and comp non-inferiority.

## 8. Phase B — the learned model, entry-gated

**Entry gate B0: Phase A must have run and reported.** Phase B is justified
in exactly two situations, and the choice between them is a decision for
sign-off *after* A0's numbers exist:

- **A0 passes but A1 fails** — the information is present in the window and a
  hand-crafted combination could not extract it. That is the one condition
  under which "learn the combination" is a genuinely new hypothesis rather
  than a fifth repetition of §2, and the model to train is small: a
  combiner over per-note physics features → window zone posterior, tens to
  low hundreds of parameters, trained LOPO on GuitarSet.
- **A0 fails on coverage but not on agreement** — the physics is accurate
  where readable but too sparse; the learned model's job is then to predict
  zone on notes physics *cannot* read, from spectral features, using
  physics-labelled windows as supervision. This is the only version that
  needs a real network, and it is the version most exposed to §2's
  graveyard.

**Substrate.** GuitarSet (CC-BY-4.0) alone for the shipping default. GAPS
(CC-BY-NC-SA-4.0) may be added for the classical route, at the cost of an
NC-SA label on the derived artifact.

**Pre-registered bar (frozen now so B cannot be graded on a curve).** Any
Phase B artifact must clear the *same* A1 gate on dev and sealed player 04,
**and** beat the Phase A arm's point estimate. A learned model that merely
matches deterministic code does not ship — that is the `context-v1` lesson,
where a masked linear control (0.5619) beat the transformer (0.5617).

**Explicitly not proposed:** re-running count-prior variants (banked,
"do not re-run without a materially new hypothesis"), symbolic context
models (S1b, context-v1), video string resolvers (WS4/Phase D), and any
reranker over `segment-v1`'s K-best (Stage 1).

## 9. Licensing

Per SPEC §1.5 as amended 2026-07-20 and `LICENSES.md`:

| artifact | license (verbatim) | role here |
|---|---|---|
| GuitarSet | `CC-BY-4.0` | training + eval substrate. Media never redistributed or committed. |
| GAPS | `CC-BY-NC-SA-4.0` | Phase B classical route only; derived artifacts inherit and must be labeled **NC-SA** in LICENSES.md. Test split (test-22 / clean-12) never enters training manifests. |
| `acoustic-physics-v1` | specification-derived, no dataset encumbrance | the channel being extended |
| private/user recordings | **BANNED** | not used in any role — the 2026-06-11 ban is on *label quality*, not licensing, so no licensing decision can lift it |

**Phase A adds no dataset, no download, no training and therefore no new
licensing row.** Phase B on GuitarSet alone stays CC-BY-4.0 and clean; only a
GAPS-inclusive variant creates an NC-SA artifact, which must be labeled
before it is registered.

*Unrelated hygiene noticed while checking this and worth a separate small
change, not this one:* `roboflow-ghaleb-guitar-fretboard-v5` is on disk but
absent from LICENSES.md (its own `README.dataset.txt` declares
`License: CC BY 4.0`), and IDMT-SMT-Guitar is recorded as `research-use,
registration` in LICENSES.md but as **CC-BY-NC-ND-4.0** in two eval reports —
if ND is correct it cannot be a training substrate even under the amended
posture, since the 2026-07-20 amendment lifts NC and NC-SA but not ND.

## 10. How this fails (stated before the run)

- **A0 shows coverage is the wall.** Solo windows without a single readable
  note stay unreachable; if that is most of them, propagation has nothing to
  propagate. Most likely failure, cheapest to discover, and the reason A0
  exists.
- **The prior already knows.** The position prior may already place the same
  notes correctly, so the propagated evidence is redundant and the delta is
  ~0. This is the §2 rule-2 failure mode and it is not excluded by A0 passing.
- **Chords are unreachable.** ~1% strummed coverage means comp gains are
  structurally near zero; the honest expectation is a solo-only channel, and
  the comp leg of gate A1 is a non-inferiority bar for exactly that reason.
- **The hand does move within a window.** Position changes inside 1 s make
  the single-zone assumption wrong, and propagation would then actively
  mislead. Reported directly by A0's agreement measurement.
- **Double-counting.** Physics evidence applied per-note *and* propagated
  risks counting the same measurement twice; the window term must be capped
  and the combination must not simply re-multiply an already-applied posterior.

## 11. Questions for sign-off

1. **Phase order.** Approve Phase A first (recommended), or attempt the
   learned model directly despite §3?
2. **Gate A0 thresholds** — 0.60 coverage / 0.75 agreement. Both are
   judgement calls set before any measurement; they are the numbers that
   decide whether an afternoon becomes a program.
3. **Sealed-player discipline.** Confirm player 04 is opened at most once for
   this program, at the end of Phase A.
4. **Scope of A1's promotion.** If A1 passes on solo but comp is merely
   non-inferior, is a solo-only routed channel acceptable in the default, or
   must a channel be tier-neutral to ship?
