# Correct-Pitch / Wrong-String Accuracy Program

## Summary

The work will target string-and-fret assignment after pitch detection, leaving onset and pitch transcription unchanged. The high-value sequence is:

1. Establish a leakage-free string-assignment benchmark and quantify the available ceiling.
2. Make existing position and sequence priors session-aware instead of applying GuitarSet behavior universally.
3. Build a compact pitch-conditioned timbral string ranker for clean steel-string acoustic guitar.
4. Add correction-driven phrase re-decoding with up to three previewable alternatives.
5. Ship only the domains and components that pass their individual gates.

The final automatic system must improve held-out steel-string single-line Tab F1 by at least `+0.03` absolute, improve its lower 95% confidence bound, and remain non-inferior on strummed material.

The existing §8 dataclasses and fusion signature remain unchanged. Video stays disabled because the current video models reduce ambiguous-note string accuracy. The retired melodic prior, larger global sequence corpora, stronger prior weights, and whole-neck image classification will not be retried.

The user granted blanket execution approval on 2026-07-14. Acceptance gates
remain binding ship/skip criteria, but passing a phase no longer requires a
new permission prompt: record the result and continue automatically. Failed
components are recorded and skipped according to their decision trees. The
approval covers in-scope dependencies, local or Modal compute within the
existing `$25` total training cap, deployment, verification, and rollback
actions required by this program. Do not exceed that cap, use private/user
recordings, weaken a metric gate, or broaden production scope beyond this plan.

## Fixed Interfaces and Defaults

### Pipeline and CLI

- Preserve the existing `run_pipeline(...) -> list[TabEvent]` behavior.
- Add `run_pipeline_with_artifacts(...) -> PipelineArtifacts` for the production adapter. `PipelineArtifacts` is an additive non-§8 type containing:
  - Final `TabEvent` objects.
  - Post-audio `AudioEvent` objects with combined fret evidence.
  - Requested and resolved inference policies.
- Add CLI/config options:
  - `--position-prior auto|none|<registered-artifact>`
  - `--sequence-prior auto|none|<registered-artifact>`
  - `--string-evidence auto|none|guitarset-timbre-v1`
- Production defaults become `auto`, while explicit artifact names remain available for evaluation and rollback.
- `auto` enables the timbral model only for:
  - `instrument=acoustic`
  - `tone=clean`
  - Standard tuning
  - Capo zero
  - High-resolution audio backend
  - A performance style for which that model passed its gate
- An explicitly requested timbral artifact with an unsupported backend or configuration returns a clear configuration error. `auto` silently resolves to `none` for unsupported domains.
- Result metadata gains optional, backward-compatible fields for requested/resolved position prior, sequence prior, string-evidence model, artifact versions, and model SHA-256.

### Refinement API

Add `POST /jobs/<job_id>/refine`.

Request:

```json
{
  "anchorNoteId": "note-id",
  "string": 2,
  "fret": 7,
  "lockedNotes": [
    {"noteId": "another-id", "string": 3, "fret": 5}
  ],
  "excludedNoteIds": ["deleted-original-note-id"]
}
```

Response:

```json
{
  "phrase": {"startTime": 4.2, "endTime": 8.7},
  "alternatives": [
    {
      "id": "proposal-1",
      "scoreDeltaFromBest": 0.0,
      "changes": [
        {
          "noteId": "note-id",
          "from": {"string": 3, "fret": 12},
          "to": {"string": 2, "fret": 7}
        }
      ]
    }
  ],
  "unchangedCount": 8
}
```

- Return at most three distinct alternatives.
- Do not represent decoder scores as probabilities; expose only cost relative to the best proposal.
- Return:
  - `404` for an unknown job.
  - `409` when a historical/expired job lacks refinement context.
  - `422` for a non-pitch-preserving correction or infeasible constraints.
  - `503` when refinement execution is unavailable.
- The endpoint proposes changes only. The client remains responsible for accepting and persisting them.

## Phase 0 — Freeze the Benchmark and Measure the Ceiling

### Repository preparation

- Preserve the current unrelated media-upload work before implementation begins.
- Create a clean `codex/string-assignment` branch from current `main`.
- Record the program and frozen metrics in the current accuracy design document and `docs/DECISIONS.md`.
- Verify the provenance of every existing GuitarSet prior. Rebuild any artifact whose player split cannot be proven.

### Data policy and splits

- Use public datasets only. Do not use uploaded/user recordings for training, tuning, model selection, or acceptance.
- Use GuitarSet players `00–04` for development:
  - Perform leave-one-player-out validation within these five players.
  - Use only out-of-fold predictions for hyperparameter selection and calibration.
- Reserve player `05` untouched for one final confirmation.
- Train the final shippable artifact on players `00–04`; do not retrain it on player `05` after evaluation.
- Treat GuitarSet solo passages as the single-line set and comp passages as the strummed set.
- Use GAPS and Guitar-TECHS only for domain-routing/non-regression checks allowed by their licenses. Never train shipped weights on GAPS.

### Baselines

Cache high-resolution audio events so every fusion experiment receives identical onsets and pitches. Measure:

1. Current production-equivalent algorithm:
   - High-resolution audio.
   - Global `guitarset-v1` position prior.
   - Coupled `guitarset-seq-v1`.
   - Video and melodic prior disabled.
2. No position or sequence prior.
3. Position prior without a sequence prior.
4. Global versus solo/comp-specific priors.
5. Later phases cumulatively, both against the frozen production baseline and the previous phase.

If the existing production prior contains held-out-player leakage, report its score for context but use a leakage-free reconstruction of the same algorithm as the formal comparator.

### Required diagnostics

Report results by player, solo/comp mode, style, pitch, candidate count, predicted string, reference string, and fret displacement. Include:

- Standard Tab F1.
- Ambiguous-note candidate top-1 and top-3 accuracy.
- Same-pitch wrong-position rate:

```text
pitch-correct matched notes assigned the wrong string/fret
----------------------------------------------------------
pitch-correct matched notes with two or more playable positions
```

- String confusion matrix.
- Existing string-confidence AUC and calibration.
- Paired, clip-stratified bootstrap confidence intervals using 10,000 resamples.

### Phrase and top-K ceiling probes

Before building refinement:

- Segment phrases using the final rules from Phase 3.
- Inject one gold string/fret correction at the first ambiguous note in each phrase.
- Measure anchored top-1 phrase decoding and best-of-three decoding.
- Build the refinement feature only if one gold correction improves ambiguous-note Tab F1 by at least `+0.10` absolute.
- Expose multiple alternatives only if best-of-three improves ambiguous-note Tab F1 by at least `+0.05` over anchored top-1. Otherwise the API returns only the best proposal.

### Phase gate

Phase 0 passes when the benchmark is reproducible from scripts, all artifacts have provenance, the held-out split is clean, and the baseline report and oracle results are checked in. If these conditions fail, stop before model work.

## Phase 1 — Domain-Aware Priors and Evidence Composition

### Rebuild and partition prior artifacts

- Rebuild the global GuitarSet position prior from players `00–04`.
- Build separate solo and comp position-prior artifacts using the same smoothing and serialization as the current global artifact.
- Build solo and comp sequence artifacts from matching data rather than coupling a mode-specific position prior to the global sequence table.
- Give every artifact a manifest containing:
  - Dataset/version and license.
  - Player and mode split.
  - Construction command.
  - Smoothing/constants.
  - Source commit and SHA-256.

### Resolve priors from session context

Replace universal prior selection with this fixed policy:

| Session | Position prior | Sequence prior |
|---|---|---|
| Clean/processed acoustic, fingerstyle, standard tuning, capo 0 | Solo artifact if its gate passes; otherwise global GuitarSet | Matching solo sequence only if the pair passes |
| Acoustic, strumming, standard tuning, capo 0 | Comp artifact if its gate passes; otherwise global GuitarSet | Matching comp sequence only if the pair passes |
| Acoustic, mixed, standard tuning, capo 0 | Global GuitarSet | Existing global sequence if its paired gate passes |
| Classical, electric, nonstandard tuning, or capo > 0 | None | None |
| Explicit named artifact | Requested artifact | Only its registered compatible sequence artifact |

- `SessionConfig` must actually participate in policy resolution; remove the current behavior that discards it.
- Sequence resolution must be keyed to the resolved position artifact. Never automatically pair a GuitarSet sequence table with an unrelated prior.
- A split prior is registered only if it improves its target mode by at least `+0.01` Tab F1 and its paired bootstrap lower bound is above zero.
- If neither split passes, acoustic sessions continue using the leakage-free global GuitarSet prior.

### Combine evidence instead of overwriting it

Introduce one evidence combiner for corpus priors, timbral evidence, and future sources:

```text
log Pcombined(candidate)
  = wposition × log Pposition(candidate)
  + wtimbre   × log Ptimbre(candidate)
```

- Compute only over playable candidates and normalize afterward.
- Missing evidence contributes a neutral factor.
- Uniform or low-information evidence must be an exact practical no-op.
- Select component weights using development folds only.
- Keep the existing fusion-level prior weight fixed to avoid tuning the same evidence twice.
- Preserve bit-for-bit behavior for events with no additional evidence.

### Phase gate

- Prior-only routing must not reduce its target mode by more than `0.005` Tab F1.
- Classical GAPS must no longer receive the known harmful GuitarSet prior.
- Guitar-TECHS gold-pitch candidate accuracy must be reported before electric routing changes are accepted.
- All policy, artifact-compatibility, normalization, and fallback tests must pass.
- Record failed prior variants rather than silently choosing them.

## Phase 2 — Pitch-Conditioned Timbral String Ranker

### Training examples

Build examples only for annotated notes with two or more playable positions:

- Input audio: 512 ms at 16 kHz, from 64 ms before onset through 448 ms after onset, zero-padded at boundaries.
- Normalize with an RMS floor while retaining original RMS as a scalar feature.
- During training, jitter onset alignment using the empirical high-resolution onset-error distribution measured on development folds, capped at ±50 ms.
- Candidate features:
  - One-hot string.
  - Normalized fret.
  - Open-string flag.
  - Normalized MIDI pitch.
  - Pitch-class sine/cosine.
  - Session-style one-hot.
- Label: the annotated string/fret candidate.
- Balance batches across player, string, pitch region, and solo/comp mode.
- Include overlapping/chord notes; pitch conditioning tells the ranker which note in the shared audio patch it is scoring.

### Model

Use a compact deterministic PyTorch model:

- Compute log-magnitude STFT with `n_fft=512` and `hop_length=128`.
- Three `3×3` convolution blocks with 16, 32, and 64 channels.
- Each block uses GroupNorm, SiLU, and `2×2` pooling.
- Adaptive average pooling produces a 64-dimensional audio embedding.
- Concatenate the embedding with candidate features.
- Candidate MLP: 96 hidden units, then 48, then one score.
- Apply masked softmax across playable candidates.
- Keep the model below 250,000 parameters and export as TorchScript.
- Calibrate one softmax temperature using out-of-fold development predictions.

Use gain, EQ, broadband noise, mild compression, and onset-shift augmentation. Parameters must be fixed from public development data and the already-modeled studio degradation ranges, not from user recordings.

### Free signal probe

Before any paid training:

- Run a simple candidate-feature-only ranker and the compact audio model locally/CPU.
- Evaluate on actual pitch-correct high-resolution events, not only gold-aligned events.
- Continue to paid optimization only if the audio model:
  - Improves ambiguous-note top-1 accuracy by at least five percentage points over the best prior-only system.
  - Has no development player fold regress by more than three points.
  - Produces calibrated, non-collapsed posteriors.
- If this gate fails, record the negative result, do not enlarge the model, and proceed to correction-driven refinement.

### Paid-training stop

If the free probe passes:

- Prepare one reproducible Modal training command, expected runtime, and cost estimate.
- Continue without another permission prompt under the 2026-07-14 blanket
  approval, provided the free-probe gate passed and the fixed total cap is
  respected.
- Cap total Modal training expenditure at `$25`.
- Use a fixed small search over learning rate, weight decay, and evidence-combination weight; do not conduct an open-ended search.
- Preserve logs, configurations, seeds, fold metrics, and artifact hashes.

### Runtime integration

- Run the ranker after high-resolution pitch events and before fusion.
- Batch all note/candidate pairs for a clip.
- Convert model outputs into the existing `AudioEvent.fret_prior` matrix through the evidence combiner.
- Do not change pitch, onset, duration, or candidate-generation logic.
- In `auto` mode, fall back to prior-only fusion when:
  - The artifact is missing or invalid.
  - The input is out of domain.
  - No ambiguous candidates exist.
- Check in the TorchScript artifact and manifest only if their combined size is below 5 MB.
- Update `LICENSES.md` with GuitarSet CC-BY attribution and derived-model provenance.

### Strict ship gate

On untouched GuitarSet player `05`, the cumulative automatic system must satisfy all of:

- Single-line Tab F1 improves by at least `+0.03` absolute against the frozen production-equivalent baseline.
- The lower bound of the paired 95% bootstrap interval for that delta is above zero.
- Same-pitch wrong-position errors fall by at least 10% relative.
- Strummed Tab F1 mean delta is at least `-0.005`.
- The strummed one-sided non-inferiority lower bound is above `-0.01`.
- Onset and pitch outputs are identical to the baseline inputs to fusion.
- CPU processing for a 60-second clip remains under five minutes and adds no more than 20% to pipeline runtime.
- Artifact size remains below 5 MB.
- The model stays disabled for every unvalidated domain.

Failure means the model is not registered as an `auto` artifact; no test-set-driven retuning is allowed.

## Phase 3 — Correction-Driven Phrase Re-Decoding

### Persist compact decode context

For each new job, write a compressed, schema-versioned sidecar next to the result:

- Stable note IDs aligned with result notes.
- Onset, offset, MIDI pitch, and audio confidence arrays.
- Combined fret-evidence matrices and presence masks.
- Original selected string/fret positions.
- Guitar configuration and session policy.
- Requested/resolved artifact versions.
- No waveform or video data.

Use compressed numeric arrays with pickle disabled and validate all shapes, ranges, versions, and note IDs on load. Add an optional `decode_context_path` to job records with a backward-compatible default.

### Phrase selection

For an anchor correction:

- Group notes using the configured onset clustering tolerance so simultaneous notes are never split.
- Keep adjacent clusters in one phrase while the gap is at most `0.75` seconds.
- Limit a phrase to `8` seconds or `32` notes.
- If a limit is exceeded, center the window on the anchor and cut at the nearest large inter-cluster gaps.
- Use selected positions immediately outside the phrase as soft boundary-continuity costs.

### Anchored decoder

- The user-corrected note is a hard `(string, fret)` constraint.
- Existing pitch-preserving edited notes in the phrase are also hard constraints.
- Deleted original notes are excluded.
- Inserted client-only notes remain untouched.
- Notes with only one playable position remain naturally fixed.
- An open-string correction anchors only that note; it does not define a hand-position center.
- Reuse the existing playability and transition costs within the phrase.
- Filter impossible states rather than approximating hard constraints with very large penalties.
- Keep the best three distinct paths using K-best dynamic programming.
- If no feasible path exists, return `422` and preserve the client state.
- Every proposed position must reproduce the stored MIDI pitch under the saved tuning and capo.

### Modal and local execution

- Local development calls the refinement decoder directly.
- Production dispatches refinement to a lightweight Modal CPU function containing only the package, NumPy, and fusion dependencies.
- The web process reads job metadata and delegates the sidecar decode; it does not install process-global transition priors while serving concurrent Flask requests.
- The refinement function reloads the result volume before reading the sidecar and performs no persistent mutation.

### Client preview

- A pitch-preserving string move requests phrase refinement before committing.
- Display the phrase range and all changed notes, with Option 1/2/3 tabs only if the Phase 0 top-K gate passed.
- Highlight proposed notes on the tab canvas without altering saved state.
- Accept applies the chosen proposal as one atomic batch edit.
- Reject leaves the transcription unchanged.
- Network failure, `409`, `422`, or timeout offers the existing single-note correction as the fallback.
- Extend undo/redo with a batch action so one undo restores the entire pre-refinement phrase.
- Persist accepted batch edits through the existing localStorage mechanism.
- Preserve previously edited notes, insertions, and deletions.
- Make the preview keyboard accessible and return focus to the corrected note after accept/reject.

### Phase gate

- The Phase 0 one-anchor oracle gate must have passed.
- Unit tests must prove pitch preservation, hard locks, phrase boundaries, excluded-note handling, impossible-path behavior, and stable K-best ordering.
- Accept/reject, atomic undo/redo, persistence, timeout fallback, and historical-job behavior must pass client/API integration tests.
- Decoder CPU time must be below 100 ms for a 32-note phrase.
- Warm API latency must be below 500 ms; cold Modal execution must remain below five seconds.

## Phase 4 — Verification, Documentation, and Domain-Gated Rollout

### Execution record (2026-07-14)

- Phase 0 passed its reproducibility/provenance gate, but the correction anchor
  oracle improved only `+0.0614` versus the required `+0.10`.
- Phase 1 passed. The global GuitarSet pair is registered and hash verified;
  solo/comp pairs failed and remain unregistered. Classical/electric/capo/
  alternate-tuning routes are neutral.
- Phase 2 failed its free gate: compact audio + prior was `-0.0218` versus the
  prior-only baseline and the worst fold was `-0.0564`. No paid training was
  started and no timbral artifact was registered.
- Phase 3 was skipped because its Phase 0 oracle prerequisite failed.
- Therefore Phase 4 deploys only the gate-passed domain-aware prior routing.
  Timbral evidence and phrase refinement remain disabled, and deployment steps
  5–7 below are intentionally not executed.
- Production verification completed on 2026-07-14. The exact test, package,
  clean-checkout, deployment, and live-job evidence is recorded in
  `docs/EVAL_REPORTS/string_assignment_phase4_verification_2026-07-14.md`.
- The strict automatic improvement and correction-path completion conditions
  did not pass. This execution is therefore closed as an evidence-backed
  partial rollout, while the broader accuracy objective remains unmet.

### Automated verification

Run:

- `pytest -v`
- `ruff check .`
- `ruff format --check .`
- `mypy tabvision`
- Server route, adapter, Modal-storage, and refinement tests.
- Web-client unit tests and production build.
- Determinism checks with fixed seeds.
- Artifact manifest/hash and package-install tests.
- Full public-corpus evaluation from a clean checkout.

Add regression cases for:

- Same pitch playable on two through six strings.
- Open-string alternatives.
- High-fret alternatives.
- Chords containing repeated pitch classes.
- Capo jobs resolving automatic learned evidence to `none`.
- Classical/electric sessions never loading the acoustic model.
- Missing/corrupt model artifacts.
- Old jobs without sidecars.
- Concurrent refinement requests using different prior configurations.
- Accepted proposals never changing pitch or touching notes outside the phrase.

### Documentation

- Append every promoted or rejected experiment to `docs/DECISIONS.md`.
- Update the current accuracy design, CLI documentation, server API documentation, and result metadata schema.
- Document model provenance, dataset attribution, training command, splits, seed, metrics, known domain limits, and rollback switches.
- Preserve §8 contracts. If a genuinely unavoidable contract change is
  discovered, document the evidence, update `SPEC.md` explicitly, and continue
  under the 2026-07-14 blanket approval rather than requesting another prompt.

### Feature controls

Use three independent rollback controls:

```text
TABVISION_POSITION_PRIOR=auto|none|<artifact>
TABVISION_SEQUENCE_PRIOR=auto|none|<artifact>
TABVISION_STRING_EVIDENCE=auto|none|guitarset-timbre-v1
TABVISION_PHRASE_REFINEMENT=true|false
```

### Deployment sequence

1. Deploy backend and client code with the timbral model and phrase refinement disabled.
2. Verify health, CORS, upload/job/result flow, old-job compatibility, and result metadata.
3. Enable domain-aware position/sequence routing.
4. Verify a public held-out acoustic fixture and confirm classical/electric jobs resolve learned evidence to `none`.
5. Enable `guitarset-timbre-v1` only for validated clean acoustic sessions.
6. Verify output against the frozen expected result for a public fixture.
7. Enable phrase refinement and test accept, reject, history, refresh persistence, and expired-context behavior end to end.
8. Monitor errors, latency, model-resolution metadata, and fallback counts without retaining audio or using production recordings for model development.
9. Roll back individual components through their environment controls if error rate, latency, or domain routing deviates from the accepted report.

The program is complete only when the strict automatic gate passes, the correction path is verified in production, documentation and licenses are current, and all supported but unvalidated domains safely resolve to neutral behavior.
