# TabVision Tab F1 Accuracy Program — Sequential Plan

**Date:** 2026-07-15
**Status:** Proposed successor to the completed 2026-07-14 wrong-string program
**Primary objective:** Raise automatic clean-acoustic Tab F1 by resolving correct-pitch / wrong-string errors without regressing onset, pitch, strummed performance, unsupported domains, or CPU feasibility.
**Canonical constraints:** `SPEC.md`, especially §0, §1.4.1, §8, and §9.3.

## 1. Starting point and central hypothesis

The deployed production-equivalent baseline on GuitarSet player 05 is:

| metric | solo | comp | aggregate |
|---|---:|---:|---:|
| macro per-clip Tab F1 | 0.5418 | 0.6834 | 0.6126 |
| micro Tab F1 | — | — | 0.6279 |
| ambiguous-note top-1 position accuracy | — | — | 0.6770 |
| ambiguous-note top-3 candidate recall | — | — | 0.9986 |
| same-pitch wrong-position rate | — | — | 0.3230 |

The current candidate generator is not the bottleneck: the annotated position is already in the first three candidates for 99.86% of ambiguous held-out notes. Of 2,300 held-out wrong-position rows, 2,271 are adjacent-string displacements. The dominant errors are the physical equivalents `+1 string / -5 frets` and `-1 string / +5 frets`.

A diagnostic gold oracle that chooses one `-1`, `0`, or `+1` predicted-string offset per track raises ambiguous-note accuracy from 0.6770 to 0.7566. Applying the same oracle independently in four-second windows raises it to 0.8245. These values are diagnostic ceilings, not shippable results, because gold labels choose the offset. They nevertheless make a persistent phrase/segment hand-position state the first hypothesis to test.

The preceding program is closed:

- Domain-aware GuitarSet position and sequence routing shipped for clean acoustic, standard-tuning, capo-zero sessions.
- Solo/comp-specific count priors failed development gates and remain unregistered.
- The 16 kHz, 512 ms compact timbral ranker regressed against the prior baseline and must not be retried unchanged.
- The one-anchor phrase-refinement oracle missed its build gate; that API and UI path did not ship.
- The current video chain is anti-complementary when audio is wrong and is closed as an automatic accuracy lever unless a genuinely new visual signal is demonstrated.

## 2. Program rules and evaluation protocol

### 2.1 Phase discipline

1. Execute one numbered phase at a time.
2. Do not enter Phase N+1 until Phase N's report is checked in, its decision branch is recorded in `docs/DECISIONS.md`, and the user explicitly says `proceed`.
3. A failed experiment is a completed result when the predeclared gate says stop. Do not tune against the failure set or enlarge a failed model without a later phase explicitly authorizing a materially different approach.
4. Any paid training, new paid service, or new dependency with licensing consequences requires a cost/license note and explicit approval before the charge or dependency is introduced.

### 2.2 Immutable and additive interfaces

- Preserve the §8 dataclasses, protocols, and `fuse(events, fingerings, cfg, session, lambda_vision) -> list[TabEvent]` signature.
- Keep `run_pipeline(...) -> list[TabEvent]` backward compatible.
- Inject learned position evidence through the existing `AudioEvent.fret_prior` field.
- A new decoder may replace the implementation behind `fuse`, but not its public contract.
- Add optional inference metadata only to `PipelineArtifacts`, `ResolvedInferencePolicy`, and API result metadata. Historical jobs and clients must continue to parse results.
- Select decoder behavior with `TABVISION_ASSIGNMENT_DECODER=baseline|segment-v1|context-v1`; `baseline` is the rollback value. Continue using `TABVISION_STRING_EVIDENCE` for optional timbral evidence.
- Resolve every automatic artifact by session domain. Unsupported sessions must fall back to the baseline or neutral evidence, never to an unvalidated learned artifact.

### 2.3 Fixed data partitions

- **Model development:** GuitarSet players 00–04 using leave-one-player-out predictions. All feature selection, thresholds, weights, architecture choices, early stopping, and ablations use only these folds.
- **Confirmation:** GuitarSet player 05 with the current cached high-resolution events. Player 05 has been reported previously, so call it a frozen historical confirmation set rather than a pristine unseen test. Inspect it only after a phase's configuration and decision rules are frozen; do not retune after seeing it.
- **Cross-domain safety:** GAPS classical clips and Guitar-TECHS electric clips are diagnostic/no-regression sets. Automatic clean-acoustic artifacts must resolve to `none` on both unless a later domain-specific phase explicitly passes its own gate.
- **No personal-data gate:** do not train, tune, or accept a release using private/user recordings. Optional personal calibration stays local and is reported separately.
- **Symbolic data:** PDMX `no_license_conflict` TAB scores may support sequence pretraining. They may not replace player-held-out real-audio evaluation.

### 2.4 Metrics

Every automatic phase report must include:

- Macro per-clip Tab F1 for solo, comp, and aggregate.
- Micro Tab F1 as a cross-check.
- Onset F1 and pitch F1.
- Ambiguous-note top-1, top-3, and same-pitch wrong-position rate.
- Error counts by reference string, predicted-minus-reference string, fret displacement, MIDI pitch, candidate count, style, and player.
- Paired clip-stratified 10,000-resample bootstrap intervals for Tab F1 deltas.
- Runtime, peak memory, artifact size, and deterministic rerun hashes.
- Domain-routing resolution and metrics on every applicable safety corpus.

The primary automatic promotion gate, unless a phase declares a stricter one, is cumulative versus the frozen production-equivalent baseline:

- Solo Tab F1: at least `+0.03` absolute.
- Aggregate Tab F1: at least `+0.02` absolute and paired 95% CI lower bound above zero.
- Same-pitch wrong-position errors: at least 10% relative reduction.
- Comp Tab F1 mean delta: at least `-0.005`; one-sided 95% non-inferiority lower bound above `-0.01`.
- Onset/pitch events unchanged for string-assignment-only phases.
- Added CPU time below 20% of current pipeline time and total 60-second-clip latency below five minutes.
- No automatic activation outside the validated domain.

Assisted editing, score-informed transcription, calibration, video-assisted capture, and hardware-assisted capture must be reported under separate named metrics. They do not count as automatic Tab F1.

## 3. Sequential execution map

| phase | work | entry condition | possible exit |
|---|---|---|---|
| 0 | Freeze benchmark, diagnostics, and ceilings | Plan approved | Reproducible evidence packet |
| 1 | Segment-level latent position decoder | Phase 0 pass | Ship, bank, or pass evidence to Phase 2 |
| 2 | Constrained contextual candidate reranker | Phase 1 report + proceed | Ship context model or close symbolic-context path |
| 3 | High-resolution uncertainty and checkpoint ensemble | Phase 2 report + proceed | Ship event improvements or preserve current events |
| 4 | High-frequency adjacent-string audio evidence | Phase 3 report + proceed | Register timbral artifact or close compact audio-string path |
| 5 | Direct per-string transcription and new data | Earlier automatic paths plateau + explicit training approval | Register second-opinion model or stop automatic model expansion |
| 6 | Assisted accuracy and optional side information | Automatic result frozen | Improve edit-time/user outcome without changing automatic score |
| 7 | Integrated verification and domain-gated rollout | At least one phase has a gate-passed result | Production promotion with independent rollback |

## 4. Phase 0 — Freeze evidence and diagnostic ceilings

### Goal

Turn the ad hoc error geometry and segment-offset oracle into a reproducible, reviewable benchmark before changing fusion.

### Implementation

1. Extend the existing string-assignment evaluation tooling rather than creating a separate metric implementation.
2. Read cached high-resolution events once and emit one deterministic note table with stable clip, cluster, event, gold-match, candidate-rank, baseline-pick, and session fields.
3. Add these diagnostics:
   - Candidate-rank histogram by player and mode.
   - String/fret displacement matrix.
   - Wrong rate by candidate count, pitch, reference string, style, and clip.
   - Per-track and fixed-window `-1/0/+1` string-offset gold oracles.
   - Window sizes of 1, 2, 4, 8, and 16 seconds.
   - Fret-zone oracle using centers 0–4, 3–7, 5–9, 7–12, and 10–15.
   - Joint `(string offset, fret zone)` oracle.
   - Existing one-anchor phrase and best-of-three oracles for comparison.
4. Report both ambiguous-note accuracy and full Tab F1 for every oracle. An oracle must never overwrite the normal baseline row.
5. Add exact provenance: dataset version/license, player split, audio-event cache key, prior artifacts/hashes, source commit, command, Python/package versions, and output hashes.
6. Check in the compact report, provenance JSON, and summary CSV. Keep the reproducible row-level table ignored if its size remains excessive.

### Tests

- Synthetic examples for every string offset and tuning boundary.
- Windows never split simultaneous onset clusters.
- Offset operations preserve MIDI pitch and discard impossible candidates rather than changing pitch.
- Gold oracles cannot score below the unmodified baseline.
- No-gold production paths cannot import or call oracle selection functions.
- Identical inputs produce identical CSV/report hashes.

### Gate and branch

- **Pass:** the production baseline reproduces within `1e-4`, provenance is complete, and the four-second joint oracle improves ambiguous-note accuracy by at least `+0.10`. Proceed to Phase 1.
- **Weak segment signal:** improvement is `+0.05` to `<+0.10`. Skip a large Phase 1 implementation; build only the smallest segment decoder probe, then require its development result before proceeding.
- **No segment signal:** improvement is `<+0.05`. Record the result and skip directly to Phase 2 after user approval.
- Any baseline mismatch blocks all later work until resolved.

## 5. Phase 1 — Segment-level latent hand-position decoder

### Goal

Recover the observed clustered adjacent-string errors using a gold-free structured decoder over the existing pitch-correct candidates.

### Decoder design

1. Keep the current chord clustering and candidate enumeration.
2. Partition cluster sequences at rests over 0.75 seconds and cap a segment at four seconds or 32 notes, cutting only between onset clusters. Treat the thresholds as the predeclared default; tune alternatives only on development folds.
3. For each segment, construct latent states from:
   - String offset hypothesis `{-1, 0, +1}` relative to the baseline candidate.
   - Fret-zone center `{2, 5, 7, 10, 13}`.
   - A neutral/open-position state that does not penalize open strings.
4. A latent state does not blindly shift notes. It changes the cost of real playable candidates that preserve the detected MIDI pitch.
5. Candidate emission cost is the existing emission plus:
   - Distance from the state's fret-zone center, exempting open strings.
   - Deviation from the state's preferred adjacent-string direction.
   - Current corpus prior negative log probability.
   - Existing chord-shape cost.
6. Segment-state transitions penalize unnecessary zone/offset changes. The penalty is reduced after long rests and large pitch-register changes.
7. Add repeat consistency: acoustically/timing-similar motifs within a clip receive a soft penalty when assigned incompatible string/fret paths. Disable this term unless a deterministic motif matcher passes unit tests and its independent ablation improves development results.
8. Decode jointly with exact dynamic programming. Retain the best three segment paths for evaluation, but return only top-1 in the automatic pipeline.
9. Derive confidence from the full-path margin between the selected string and the cheapest alternative-string path.

### Tuning

- Use a fixed coarse grid for fret-zone, offset, state-change, repeat, prior, and transition weights.
- Select by aggregate macro Tab F1 across OOF players 00–04, with comp non-inferiority as a hard constraint.
- Break ties by lower wrong-position rate, then smaller runtime, then smaller departure from existing weights.
- Freeze all weights and boundaries before running player 05.

### Integration

- Place the implementation behind the existing `fuse` contract.
- `TABVISION_ASSIGNMENT_DECODER=segment-v1` selects it explicitly.
- `auto` may resolve to `segment-v1` only after the final gate passes; until then production remains `baseline`.
- Add resolved decoder and reason to pipeline/API metadata.
- Classical, electric, capo, and alternate-tuning sessions resolve to `baseline` during this phase.

### Tests

- Exact pitch preservation for every selected candidate.
- Chord string uniqueness and hand-span constraints remain enforced.
- Segment boundaries, simultaneous clusters, open strings, short clips, long rests, and no-candidate notes.
- Deterministic path ordering and confidence margins.
- Bit-identical baseline output when the decoder is disabled.
- Concurrent jobs using different decoder settings cannot share mutable state.

### Gate and branch

- **Promote:** the cumulative automatic promotion gate in §2.4 passes. Register `segment-v1` and make it the clean-acoustic `auto` decoder in Phase 7.
- **Bank for composition:** OOF aggregate gain is at least `+0.01`, no player regresses by more than `0.02`, and the four-second oracle remains at least `+0.10`, but the final promotion gate fails. Keep the decoder available for Phase 2 composition but do not auto-enable it.
- **Close rule-based segment decoding:** OOF aggregate gain is below `+0.01` or two player folds regress by more than `0.02`. Preserve the report and proceed to Phase 2 only because the oracle shows learnable signal; do not continue hand-tuning weights.

## 6. Phase 2 — Constrained contextual candidate reranker

### Goal

Learn string/fret selection from whole-phrase context while making pitch changes impossible.

### Phase 2A: feature-only control

1. Build one candidate row per playable `(string, fret)` for each matched development event.
2. Use only deterministic features already available at inference:
   - MIDI pitch and pitch class.
   - Previous/next pitch intervals.
   - Duration and log onset gaps.
   - Chord size, note rank within chord, and bass-note flag.
   - Candidate string, fret, open flag, and candidate count.
   - Corpus prior log probability and current Viterbi emission/transition components.
   - Segment fret zone and offset hypothesis from Phase 1.
   - Session instrument, tone, and style.
3. Train a masked linear softmax scorer as the control. Use cross-entropy with inverse-frequency weighting by player, reference string, and candidate count.
4. If the control does not improve OOF ambiguous top-1 by at least `+0.01`, still continue to the contextual model; the control exists to quantify the value of sequence context.

### Phase 2B: contextual model

Use a compact candidate-conditioned Transformer:

- Two bidirectional encoder layers.
- Model width 64, four attention heads, feed-forward width 128, dropout 0.1.
- Maximum 128 events per window with overlap at inference; never split an onset cluster.
- Event tokens contain the Phase 2A event/context features.
- A shared candidate head concatenates each contextual event embedding with its candidate features and emits one score.
- Mask non-playable candidates before softmax.
- Keep the model below 500,000 parameters and export to TorchScript.
- Train with candidate cross-entropy plus a small transition-consistency term computed only from annotated adjacent events.
- Early-stop on mean OOF development Tab F1, not training loss.

Feed the normalized candidate probabilities into `AudioEvent.fret_prior`, then run the segment/baseline decoder. Evaluate four fixed compositions: baseline decoder, segment decoder, context evidence alone, and context evidence plus segment decoder.

### Optional symbolic pretraining

Run this only if the GuitarSet-only contextual model improves OOF aggregate Tab F1 by at least `+0.015` but remains data-limited by fold variance:

1. Pretrain the event encoder on PDMX `no_license_conflict` TAB sequences using masked candidate prediction.
2. Exclude audio features and copyrighted/unapproved partitions.
3. Fine-tune all weights on GuitarSet players 00–04.
4. Keep PDMX pretraining only if it adds at least `+0.01` OOF aggregate Tab F1 over the GuitarSet-only model and no player fold regresses by more than `0.02`.

### Artifact and runtime

- Artifact manifest records architecture, code/data versions, player folds, seeds, metrics, license inputs, SHA-256, and compatible prior/decoder versions.
- Missing, corrupt, incompatible, or out-of-domain artifacts fall back to the last gate-passed decoder.
- Batch all events in a clip and keep added CPU time under 20%.

### Tests

- Candidate masking and pitch preservation.
- Padding/window overlap produces one stable score per event.
- OOF split code prevents player leakage.
- Artifact hash/manifest validation and TorchScript parity.
- Neutral evidence is a practical no-op.
- Domain routing and fallback behavior.
- Deterministic fixed-seed training smoke test on a tiny fixture.

### Gate and branch

- **Promote:** the §2.4 cumulative promotion gate passes and at least four of five development player folds improve. Register `context-v1`; select the best predeclared decoder composition.
- **Useful but not shippable:** ambiguous top-1 improves by at least `+0.03`, but full Tab F1 or non-inferiority fails. Preserve the model only as an offline diagnostic and use its disagreement signal in Phase 6; do not auto-enable it.
- **Close symbolic-context expansion:** OOF ambiguous top-1 gain is below `+0.02`. Do not increase model size or run open-ended architecture search. Proceed to the independent audio-information phases.

## 7. Phase 3 — High-resolution uncertainty and checkpoint ensemble

### Goal

Recover residual onset/pitch/event errors and give fusion calibrated uncertainty instead of hard MIDI plus velocity-proxy confidence.

### Implementation sequence

1. Verify what raw onset, frame, and pitch posteriors the high-resolution backend exposes before changing code.
2. Populate the existing `AudioEvent.pitch_logits` field when real logits are available. Do not fabricate logits from MIDI velocity.
3. Add a cache format version and posterior-shape/provenance validation.
4. Measure posterior calibration and oracle recoverability:
   - Is the reference pitch in top-2/top-3 posterior choices?
   - Are missed notes visible as sub-threshold onset/frame peaks?
   - Does posterior entropy predict pitch or string errors?
5. Run the predeclared two-checkpoint comparison using `guitar_gaps` and `guitar_fl`:
   - Match same-pitch events within 50 ms.
   - Preserve agreed events.
   - For disagreements, select or merge using development-only calibrated posterior scores.
   - Evaluate union, intersection, confidence winner, and one learned logistic combiner; no additional variants.
6. If posterior alternatives have a useful oracle, allow the contextual decoder to score a small audio lattice. Limit it to at most two pitch hypotheses per onset and require the final emitted pitch to be explicit in the path.

### Gates

- Posterior/lattice work continues only if the reference is recoverable from top-2 alternatives for at least 25% of current pitch-off or missed-event errors without more than one extra false candidate per ten correct events.
- An ensemble ships only if aggregate Tab F1 improves by at least `+0.01`, onset and pitch F1 do not regress, the paired Tab F1 lower bound is above zero, and latency remains below five minutes.
- If neither gate passes, preserve the current event stream bit-for-bit and close posterior/ensemble work. Do not retune the string decoder on player 05.

### Tests

- Raw posterior extraction against a fixed backend fixture.
- Cache versioning, corrupt-cache rejection, and old-cache fallback.
- Deterministic event matching and duplicate suppression.
- Boundary onsets, chords, sustains, and overlapping checkpoint events.
- Identical downstream output when posterior use is disabled.

## 8. Phase 4 — High-frequency adjacent-string audio evidence

### Goal

Test a materially different string-identification signal while avoiding the design that already failed.

### Free signal probe

1. Preserve the demuxed 44.1/48 kHz waveform for this branch; do not upsample the 16 kHz backend signal.
2. Build same-pitch adjacent-string pairs from GuitarSet microphone audio and hexaphonic annotations.
3. Extract deterministic, interpretable features from a 64 ms pre-onset + 448 ms post-onset window:
   - Multi-resolution harmonic magnitudes through the available high-frequency range.
   - Onset spectral centroid/rolloff and pick-noise energy.
   - Harmonic decay ratios and inharmonicity.
   - RMS and spectral slope, retained separately from normalized features.
4. Train a regularized linear pairwise classifier with player-held-out folds.
5. Continue only if it beats the prior-only candidate choice by at least five percentage points on ambiguous OOF examples and no fold regresses by more than three points.

### Compact learned model

If the free probe passes, train a compact multi-resolution model:

- Three branches: onset transient, short sustain, and long sustain.
- Log-frequency harmonic input plus raw scalar descriptors.
- Shared convolutional encoder under one million parameters.
- Candidate-conditioned pairwise score, trained primarily on adjacent-string hard negatives.
- Balanced sampling by player, pitch, string, candidate count, and solo/comp mode.
- Gain, EQ, codec, room-response, mild compression, and onset-jitter augmentation fixed before confirmation evaluation.
- Temperature calibration from OOF predictions.

The model outputs candidate likelihoods only; it cannot modify onset or pitch. Combine its log probabilities with the context/corpus evidence using development-selected fixed weights, then normalize once over playable candidates.

### Cost and license gate

Before GPU training, record the exact command, estimated runtime/cost, spending cap, dataset license/attribution, artifact license, and fixed hyperparameter grid. Obtain explicit user approval. Do not use NC/research-only data to train shipping weights.

### Promotion gate

In addition to §2.4:

- Timbral evidence must improve OOF ambiguous top-1 by at least `+0.05` over the best non-timbral system.
- Worst player-fold regression must be no more than `0.03`.
- Artifact size must be below 10 MB.
- Added inference time must be below 20%.
- The model remains disabled for classical, electric, capo, alternate tuning, and non-high-resolution backends.

Failure closes this compact timbral path. Do not respond by increasing sample rate window, parameter count, or training budget without a new plan and new evidence.

## 9. Phase 5 — Direct per-string transcription and targeted data expansion

### Entry condition

Enter only if Phases 1–4 have plateaued below the desired automatic accuracy and the user approves a larger training project. This is a new model program, not another fusion-weight sweep.

### Phase 5A: data/license design

1. Use GuitarSet hexaphonic annotations as the clean-acoustic training core.
2. Audit every proposed architecture/codebase before reuse. Papers may guide an original implementation; unlicensed source code must not be copied.
3. Keep Guitar-TECHS as a separate electric-domain track; do not mix it into the acoustic acceptance model by default.
4. Recheck GOAT access and dataset license before including it. Exclude it if terms do not permit shipped derived weights.
5. Exclude SynthTab and other NC/research-only corpora from shipping-model training.
6. If new data is commissioned, prioritize:
   - Same phrase played in several fretboard positions.
   - Same pitch on adjacent reachable strings.
   - Low-E and the current worst MIDI pitches.
   - Five/six-candidate pitches, rock/jazz passages, capo, and multiple guitars/players.
   - Synchronized normal microphone and per-string/hexaphonic channels.
   Use CC0 or CC-BY terms and publish a manifest before training.

### Phase 5B: model

Implement an original six-string multi-task network:

- Shared audio encoder.
- Six onset heads and six frame/pitch heads.
- Auxiliary global-pitch head to preserve pitch quality.
- String-occupancy and duplicate-pitch inhibition loss.
- Output per-string event likelihoods mapped into `AudioEvent.fret_prior` for candidate pitches.
- Use it first as a second opinion alongside the existing high-resolution backend; replace the backend only if a later direct comparison passes all onset/pitch gates.

Start with a CPU-feasible architecture under five million parameters. Use a single fixed architecture and bounded learning-rate/regularization grid. No neural architecture search.

### Gates

- Gold-pitch string accuracy must exceed the best contextual/timbral system by at least `+0.05` OOF before real-event integration.
- Real-event cumulative aggregate Tab F1 must improve by at least `+0.04` over the frozen production baseline, with the paired lower bound above zero.
- Onset and pitch F1 may not regress by more than `0.005` absolute.
- At least four of five development player folds improve.
- CPU latency remains below five minutes for a 60-second clip; artifact size and license are documented.
- If the second-opinion configuration fails, do not replace the accepted high-resolution backend.

## 10. Phase 6 — Assisted accuracy and optional side information

### Goal

Reduce user correction effort after the automatic result is frozen. Report these outcomes separately from automatic Tab F1.

### 6A: learned review queue

1. Train an error detector from OOF predictions using path margin, candidate count, context/timbre disagreement, posterior entropy, domain score, chord size, and segment inconsistency.
2. Calibrate probabilities OOF.
3. Ship only if error-detection ROC AUC is at least 0.75 and the lowest-confidence 10% contains at least twice the global wrong-position rate.
4. Measure precision/recall at fixed review budgets of 10%, 20%, and 30% of notes.

### 6B: top-three and phrase alternatives

- Reuse the automatic decoder's K-best paths; do not build a separate incompatible correction engine.
- Offer pitch-preserving candidate cycling and `move phrase one string up/down`.
- Show up to three phrase alternatives only when they differ and all preserve pitch/playability constraints.
- Accept applies one atomic batch edit; reject changes nothing; one undo restores the prior phrase.
- Propagate a correction only to repeated, demonstrably matched motifs and always preview the affected notes.
- Persist accepted edits through the existing localStorage/job-result mechanism.

### 6C: optional calibration and score-informed modes

Treat each as an explicit user-selected mode:

- Six-open-string or known-chord calibration to estimate session timbre.
- User-provided starting hand position.
- MIDI, MusicXML, Guitar Pro, or chord-chart alignment.
- Licensed known-song reference alignment.
- Private per-user correction prior stored locally with opt-in.

Do not silently use private corrections for global training.

### Metrics and gate

- Tab F1 after fixed review budgets of 10, 30, and 60 seconds.
- Corrections per minute, notes changed per accepted action, undo rate, and wrong propagation rate.
- Target: at least 50% reduction in residual wrong-position errors within a 60-second review budget, with zero pitch-changing automatic edits.
- Production UI work proceeds only after the offline replay harness meets the target.

## 11. Deferred research branches

These are deliberately outside the main automatic sequence and require separate approval:

### Electric guitar

- Train a separate high-resolution electric checkpoint using Guitar-TECHS or another shipping-compatible corpus.
- Gate pitch/onset quality first; current electric Tab F1 is primarily blocked by pitch collapse, not only string assignment.
- Register separate electric context/string artifacts. Never route the acoustic model into electric automatically.

### New video signal

- Do not retune the current fretting-hand chain.
- Reopen video only with a pre-implementation complementarity probe for a distinct signal: picking-hand string motion near the bridge/sound hole or coarse hand-position zone.
- Continue only if `P(new video right | audio wrong)` exceeds audio top-1 by at least 10 percentage points on a held-out public corpus and a confidence router improves Tab F1.

### Hardware-assisted modes

- Hexaphonic pickup, MIDI guitar, direct per-string interface, or instrumented fretboard can provide decisive string identity.
- Report these as separate capture modes, not as blind iPhone transcription.

## 12. Phase 7 — Integrated verification and rollout

### Pre-merge verification

Run from a clean checkout:

```powershell
cd tabvision
pytest -v
ruff check .
ruff format --check .
mypy tabvision
```

Also run:

- All phase-specific deterministic eval commands.
- Full GuitarSet development OOF and frozen player-05 confirmation.
- GAPS and Guitar-TECHS routing/no-regression diagnostics.
- Server adapter and result-metadata tests.
- Web-client tests/build only if Phase 6 ships UI changes.
- Artifact manifest/hash, missing/corrupt artifact, and clean-install tests.
- 60-second CPU latency and peak-memory benchmark.
- Concurrent-job tests with different sessions and decoder policies.

### Documentation

- Append each promoted, rejected, skipped, or deferred branch to `docs/DECISIONS.md`.
- Update `SPEC.md` only for an explicitly approved contract/scope change.
- Update `LICENSES.md` for every dataset, dependency, derived model, and attribution.
- Store each phase report in `docs/EVAL_REPORTS/` with command, commit, environment, hashes, folds, metrics, and known limitations.
- Document artifact provenance, supported domain, fallback behavior, and rollback switch.

### Deployment order

1. Deploy code with every new decoder/evidence artifact disabled; verify baseline parity.
2. Enable the gate-passed position/sequence routing and verify current production behavior.
3. Upload/register one new artifact at a time without changing `auto` routing.
4. Run a public held-out fixture through production and compare result hash/metrics to the frozen expected output.
5. Enable the selected automatic decoder only for clean acoustic, standard tuning, capo zero, high-resolution sessions.
6. Confirm classical, electric, capo, alternate-tuning, missing-artifact, and corrupt-artifact requests fall back safely.
7. Monitor resolved policy metadata, fallback counts, latency, errors, and artifact hashes without retaining user media for development.
8. Enable assisted features only after automatic rollout is stable.
9. Roll back independently with `TABVISION_ASSIGNMENT_DECODER=baseline`, `TABVISION_STRING_EVIDENCE=none`, and the existing position/sequence controls.

### Completion criteria

The program is complete when:

- At least one automatic approach passes the cumulative promotion gate and is verified in production, or every bounded automatic branch has produced a documented negative result.
- Automatic, assisted, electric, video-assisted, score-informed, and hardware-assisted metrics are clearly separated.
- All §8 contracts remain intact or an explicit approved SPEC amendment exists.
- Tests, lint, types, licenses, artifact manifests, reports, docs, and rollback controls are current.
- Unsupported domains reliably resolve to the last validated neutral/baseline behavior.

## 13. Immediate first action after approval

Create the Phase 0 evaluator/report only. Do not change fusion or production routing in the same phase. Reproduce the 0.6126 baseline, formalize the offset/fret-zone/window oracles, check in the evidence packet, record the gate decision, and stop for the next explicit `proceed`.
