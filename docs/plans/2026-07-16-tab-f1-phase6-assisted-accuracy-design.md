# Sequential Tab F1 Phase 6 - assisted-accuracy freeze

**Date:** 2026-07-16
**Status:** Frozen before Phase 6 feature extraction or detector fitting
**Parent:** `docs/plans/2026-07-15-tab-f1-accuracy-sequential-plan.md`, Phase 6

## Scope and invariants

The user authorized Phase 6 after the automatic model sequence stopped at the
Phase 5 gold-pitch gate. Automatic transcription is frozen. Phase 6 may rank
notes for review and offer pitch-preserving edits, but it may not change the
automatic Tab F1 row, audio events, decoder weights, routing, or SPEC section 8
contracts.

The development population is the 35,959 production-equivalent,
pitch-correct, physically ambiguous GuitarSet events from players 00-04. The
label is whether the automatic string/fret position is wrong. All detector
predictions are leave-one-player-out. Player 05 stays unopened until both the
review-detector gate and the offline replay gate pass with the method below.

## Frozen review features

All features are available without a gold label at inference time.

1. `path_margin`: log-scaled cost gap between the automatic candidate and the
   second min-marginal candidate from the production decoder.
2. `candidate_count`: playable string/fret candidates for the emitted pitch.
3. `context_disagreement`: whether the already evaluated OOF Phase 1 segment
   decoder chooses a different string than production.
4. `timbre_disagreement`: whether the OOF Phase 4 native-rate audio-only probe
   chooses a different string than production.
5. `timbre_strength`: maximum absolute adjacent-pair logit from that probe.
6. `posterior_entropy`: normalized pitch entropy in the accepted GAPS
   checkpoint's three-frame onset window.
7. `domain_score`: the explicit automatic-route eligibility score. It is 1.0
   for this clean-acoustic, standard-tuning, capo-zero development corpus. Its
   constancy is reported rather than hidden.
8. `chord_size`: number of events in the simultaneous onset cluster.
9. `segment_inconsistency`: absolute difference between the event's inferred
   Phase 1 string shift and the median shift in its deterministic segment,
   scaled by five. Segments split at a rest over 0.75 s, four seconds, or 32
   notes without splitting an onset cluster.
10. `mode_comp`: 1 for comp/strumming and 0 for solo/fingerstyle.

The source row, native feature cache, posterior cache, and all generated
feature arrays receive SHA-256 provenance. Player 05 and non-GuitarSet data are
asserted absent.

## Frozen detector and calibration

- Model: original 10 -> 16 -> 8 -> 1 ReLU MLP, under 500 parameters.
- Optimizer: AdamW, learning rate `3e-3`, weight decay `1e-4`.
- Training: 40 full deterministic epochs, batch 512, seed 6621 plus fold ID.
- Loss: binary cross entropy with a fixed positive weight equal to the
  training fold's correct/wrong ratio.
- Outer evaluation: leave one player out.
- Calibration inside each outer fold: obtain predictions for every outer-
  training player from four three-player inner fits, then fit a one-dimensional
  L2 Platt calibrator to those inner-OOF logits. Fit the final detector on all
  four outer-training players and apply only that training-derived calibrator
  to the held player.
- No architecture, feature, epoch, threshold, calibration, or weighting grid.

The detector gate requires both overall OOF wrong-position ROC AUC at least
`0.75` and wrong-position prevalence in the highest-risk 10% at least twice
the global prevalence. Precision and recall are also frozen at review budgets
of 10%, 20%, and 30% of notes.

## Frozen correction primitives

The reusable editing core must:

- cycle only through the production decoder's pitch-preserving min-marginal
  candidates;
- derive up to three unique phrase alternatives only from its constrained
  K-best paths;
- move a phrase one string up/down only if every resulting position is
  playable and preserves MIDI pitch;
- apply a multi-note edit atomically, reject without mutation, and restore the
  complete prior phrase with one undo;
- return repeated-motif propagation as a preview only. A repeat must exactly
  match pitch sequence and quantized inter-onset intervals; no propagation is
  automatic;
- keep optional calibration, starting-position, score-reference, and private
  correction-prior inputs disabled unless explicitly selected. A private prior
  is local-only and never a global-training input.

## Frozen offline replay

Replay is per clip and uses only the held-player detector predictions.

- Queue notes by decreasing calibrated wrong-position probability, then stable
  event index.
- A reviewed correct note is rejected and consumes two seconds.
- A reviewed wrong-position note is corrected only when its gold position is
  present in the production decoder's top three min-marginal candidates. The
  oracle represents a user choosing among the displayed pitch-preserving
  candidates; it also consumes two seconds.
- Budgets are 10, 30, and 60 seconds per clip (5, 15, and 30 reviewed notes).
- No motif propagation is credited in the acceptance metric. This makes wrong
  propagation and pitch-changing edits exactly zero by construction; phrase
  batching is reported separately only if used.
- Report residual wrong-position reduction, correction precision/recall, Tab
  F1 with corrected positions, corrections per minute, notes changed per
  accepted action, undo rate, wrong propagation, and pitch changes.

The offline gate requires at least 50% aggregate reduction of residual
wrong-position errors at 60 seconds, zero pitch changes, and zero wrong
propagation. Production UI work starts only if the detector and replay gates
both pass. A failed gate is terminal for this fixed assisted path; do not tune
the queue or timing assumptions against the result.
