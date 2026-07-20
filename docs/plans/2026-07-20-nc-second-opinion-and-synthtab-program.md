# NC second-opinion + SynthTab-scale program — plan

**Date:** 2026-07-20
**Status:** Active. Phase N0/S0 (survey) complete — see
`docs/EVAL_REPORTS/nc_checkpoint_dataset_survey_2026-07-20.md`.
**Authorization:** user directive 2026-07-20 ("Execute both"), recorded in
`docs/DECISIONS.md` same date. Successor programs to the closed 2026-07-15
sequential Tab F1 program; this plan **reuses that plan's frozen protocols by
reference** (§2.2 interfaces, §2.3 partitions, §2.4 metrics/promotion
reporting) rather than restating them.
**Posture:** SPEC §1.5 personal non-commercial. NC-derived artifacts labeled
in LICENSES.md. Private/user recordings remain banned from all
training/eval/label roles.

## Standing rules for both programs

1. Fixed partitions and metrics per sequential plan §2.3/§2.4: development =
   GuitarSet players 00–04 OOF; frozen confirmation = player 05 (inspect only
   after configs freeze); GAPS + Guitar-TECHS = cross-domain no-regression
   diagnostics; paired clip-stratified 10k bootstrap for Tab F1 deltas.
2. §8 contracts immutable; new behavior additive and explicit-only until its
   gate passes; unsupported domains resolve to current behavior; independent
   rollback env switches preserved.
3. Execution interpretation of the 2026-07-20 directive: $0, local,
   reversible development work (surveys, probes, dev-fold evals, doc/eval
   commits on a work branch) proceeds without per-step confirmation. **User
   checkpoints remain mandatory for:** (a) any paid compute or service,
   (b) any new runtime dependency of the shipping package, (c) artifact
   registration / `auto`-routing changes / production deploy, (d) bulk
   downloads beyond the ~1 GB symbolic slice and the SynthTab Dev set.
4. Baselines to beat (frozen): clean-acoustic dev aggregate `0.6017`
   (registered two-checkpoint ensemble), player-05 aggregate `0.6339`;
   ambiguous top-1 `0.6770` (baseline decoder reference frame).

## Program N — pretrained checkpoint second opinions

### N0 — survey (DONE, gate PASSED)

Candidates ranked: (1) `guitar_kroma.safetensors` — same MIT repo and
architecture family as the accepted backend, 24.66M params, verified via
safetensors header; (2) MuScriptor-large — CC-BY-NC 4.0, offline-only
candidate; (3) YourMT3+ — GPL-3.0 code, weights license unverified. SynthTab
baseline TabCNN weights tracked under Program S.

### N1 — kroma conversion + smoke ($0, local)

1. Download `guitar_kroma.safetensors` (47 MB) to the eval data root; record
   SHA-256. Convert state dict to `.pth` with a new
   `scripts/eval/convert_kroma_checkpoint.py` (safetensors → `torch.save`),
   byte-stable output, manifest with source URL/hash/license (MIT).
2. Probe **without repo runtime changes**: call the pinned
   `hf_midi_transcription` API directly with `checkpoint_path=<local .pth>` on
   (a) the repo audio fixture, (b) 5 GuitarSet dev clips (players 00–04 only).
3. Compare onset/pitch F1 and event counts against cached `guitar_gaps` /
   `guitar_fl` events on the same clips.

**Gate:** loads cleanly and mean onset F1 and pitch F1 on the 5-clip smoke are
each within `0.05` of the better of gaps/fl (sanity, not promotion).
**Pass →** N2. **Fail (load) →** attempt trivial key remap only (no
architecture surgery); **fail (quality) →** close kroma, open the MuScriptor
complementarity probe instead.

### N2 — three-member ensemble development ($0, local, CPU-hours)

1. Add `guitar_kroma` to `GUITAR_VARIANTS`/`_CHECKPOINT_FILE` (additive,
   explicit selection only; `auto` untouched).
2. Generate cached kroma events for dev players 00–04.
3. Extend the Phase 3 harness (`scripts/eval/string_assignment_phase3.py`) to
   three members: existing pair + kroma pairwise, and 3-way
   `confidence_winner` + one logistic combiner (same predeclared variant set;
   no new merge families). Report complementarity explicitly:
   `P(kroma right | registered ensemble wrong)` on matched events.
4. Freeze weights/calibrators on dev OOF only.

**Gate (mirrors the accepted Phase 3 ensemble gate):** dev aggregate Tab F1
≥ `+0.01` over `0.6017` with paired CI lower bound > 0; onset/pitch F1
non-regressing; 60-s clip latency with three sequential passes < 5 min
(measured; expect ≈ +50% over the two-pass 196–258 s → risk item, mitigation:
kroma-for-fl substitution variant is in scope).
**Pass →** N3. **Fail →** record bounded negative; open MuScriptor probe only
if complementarity ≥ 10 pp headroom was observed; else close Program N.

### N3 — confirmation + registration + deploy (user checkpoint)

Frozen config → player-05 confirmation; `ensemble_v2.json` manifest
(members, calibrators, hashes, validated domain); GAPS/Guitar-TECHS routing
no-regression; then **stop for user sign-off** before registration,
`auto`-routing change, and Modal/Vercel deploy (per prod-topology memory).
Promotion bar per the 2026-07-20 precedent: CI-significant aggregate
improvement + non-regressions (cumulative-guardrail waiver is the user's,
already exercised once; re-confirm at sign-off).

### N-branch — MuScriptor / YourMT3 (conditional)

Entry only on N2 fail-with-headroom or explicit user request. Order:
10-clip dev complementarity probe (offline, process-isolated, no shipping
dependency) → full dev eval only if `P(right | ensemble wrong)` ≥ 10 pp.
MuScriptor weights are CC-BY-NC (label NC if ever registered); YourMT3
requires weights-license verification first; GPL code stays process-isolated.

## Program S — SynthTab/DadaGP-scale training

### S0 — acquisition + license audit (≈1 GB + Dev set; browser download OK)

1. Download `all_jams_midi_V2_60000_tracks.zip` (≈1 GB) and the SynthTab Dev
   set from UR Box (browser-interactive; Box `/v/` links don't curl). Store
   under `TABVISION_DATA_ROOT`, git-ignored; SHA-256 manifest checked in.
2. LICENSES.md: SynthTab rows (CC-BY-NC 4.0, attribution, NC label; DadaGP
   provenance note).
3. Parser spot-check: JAMS/per-string MIDI → per-note `(string, fret, onset,
   duration, tuning, capo)` sequences; count tracks, notes, standard-tuning
   share; verify GuitarSet/GAPS eval material is absent (provenance:
   DadaGP-derived, disjoint by construction — assert no track-name overlap
   anyway).

**Gate:** ≥ 50k tracks parse with usable string+fret+onset in standard-tuning
6-string form. **Fail →** diagnose format; if unusable, Program S closes to a
DadaGP-request path (user action) or bounded negative.

### S1a — SynthTab-scale count priors (CPU, hours — cheapest shot first)

Build `synthtab-v1` position + `synthtab-seq-v1` sequence priors (same
artifact class as `guitarset-v1`/`gaps-v1`, reusing the
`build_gaps_v1_prior.py` tooling generalized to the S0 parser). Evaluate per
the frozen A15 protocol on dev OOF: candidate roles are (i) swap vs the
GuitarSet pair, (ii) interpolated blend (single blend weight, coarse grid on
dev only). **Gate:** A15/gaps-v1 standard — CI-significant dev aggregate
improvement, comp non-inferiority, cross-domain routing untouched. The PDMX
lesson (scale loses to domain mismatch) is the null hypothesis; SynthTab is
domain-matched tab, so a real test.

### S1b — contextual reranker, pretrain → finetune (CPU-bounded; else free-GPU ask)

Original implementation (MIDI-to-Tab as published blueprint only — no code
exists to copy). Frozen Phase 2B bounds by reference: ≤ 500k params, masked
candidate scoring into `AudioEvent.fret_prior`, TorchScript export, playable-
candidate masking, pitch preservation. Pretrain masked string/fret prediction
on SynthTab symbolic sequences (cap: 20k standard-tuning tracks, ≤ 48 h
laptop-CPU budget; if exceeded, stop and ask for a free-tier GPU session);
finetune on GuitarSet 00–04 per Phase 2B.
**Keep-rules (per Phase 2B/PDMX precedent):** pretraining is kept only if it
adds ≥ `+0.01` OOF aggregate over the GuitarSet-only ablation and no player
fold regresses > `0.02`; the composed system must clear ≥ `+0.01` dev
aggregate with CI > 0 to reach confirmation. Fail → bounded negative; this
closes the symbolic-scale hypothesis with domain-matched data (the strongest
version of the A15/Phase-2 question).

### S2 — per-string audio model on SynthTab renders (entry gated)

**Entry:** S1 report checked in + explicit user compute authorization
(free Colab/Kaggle first per operating rule 6; any paid GPU requires a cost
note + approval per rule 8). Bring-up on the SynthTab Dev set; acoustic
subsets (< 50 GB/zip) before any electric work. Reuse the frozen Phase 5
architecture/gates by reference (`2026-07-16-tab-f1-phase5-data-license-design.md`)
with one amendment: SynthTab pretraining + GuitarSet finetune replaces
GuitarSet-only training, and the corpus table's SynthTab exclusion row is
superseded by the §1.5 amendment (NC label required). Gold-pitch gate
unchanged: OOF ambiguous top-1 ≥ `0.7121` before real-event integration.

### S-deferred

Electric per-string / electric checkpoint work stays outside this program
(v2 scope). SynthTab's electric renders may not enter the acoustic acceptance
model.

## Reporting

Every phase ships an eval report + DECISIONS entry per house format;
artifacts carry manifests (source, hash, license, validated domain, folds).
Automatic vs assisted metrics stay separated per SPEC §1.4.1.
