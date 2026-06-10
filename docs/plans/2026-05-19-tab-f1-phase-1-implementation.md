# Tab F1 — Phase 1 Implementation Plan

**Date:** 2026-05-19
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — **plan-only, do not start implementation yet**
**Strategy doc:** [`docs/plans/2026-05-12-tab-f1-to-spec-design.md`](2026-05-12-tab-f1-to-spec-design.md) §6 (Phase 1)
**Predecessor:** Phase 0 — [`docs/plans/2026-05-13-tab-f1-phase-0-implementation.md`](2026-05-13-tab-f1-phase-0-implementation.md)
**Baseline this plan moves:** [`docs/EVAL_REPORTS/composite_baseline_2026-05-13.md`](../EVAL_REPORTS/composite_baseline_2026-05-13.md)
**Implementation branch:** to be cut as `impl/tab-f1-phase-1` off `main` once Phase 0 (PR #11) merges.

## 0. Gating note — do not begin implementation

Per CLAUDE.md operating rule 2 ("Phase N+1 starts only after Phase N's
acceptance gate passes AND user says 'proceed.'"):

- **No code in `tabvision/tabvision/audio/` may be modified** until
  PR #11 (Phase 0 implementation) merges into `main`.
- After #11 merges, the user must explicitly say "proceed" before
  the `impl/tab-f1-phase-1` branch is cut and Stage A begins.

This plan document itself is independent of #11 and can be reviewed
and approved in parallel. Approval here means "the Phase 1 approach
is fine"; it does **not** authorize coding to start.

## 1. Goal recap

From strategy doc §6:

| Metric | Phase 0 baseline (2026-05-13) | Phase 1 target | Δ needed |
|---|---:|---:|---:|
| Pitch F1 (clip-mean, GuitarSet validation) | ~0.915 (single-line 0.9304 / strummed 0.9005) | **≥ 0.93** | +1.5 pp |
| Onset F1 (clip-mean, GuitarSet validation) | ~0.93 (single-line 0.9375 / strummed 0.9229) | **no regression > 1 pp** | ≥ 0.92 |
| Tab F1 (aggregate) | 0.589 | no regression beyond mathematical pitch-improvement bound | — |

The Phase 0 baseline is reproducible from the manifest at
`tabvision/data/eval/composite.toml` via:

```bash
cd tabvision && ../tabvision-server/venv/bin/python -m tabvision.eval.composite \
    --manifest data/eval/composite.toml \
    --backend highres --position-prior guitarset-v1 \
    --output /tmp/baseline.md --bootstrap-n 10000 \
    --eval-harness-sha "$(git rev-parse --short HEAD)"
```

Phase 1 is a **measurement-driven tuning pass on the existing audio
backend**, not a training run. No new model weights, no Modal spend,
no §8 contract changes.

## 2. Staged approach

Two stages with a hard decision point between them.

### Stage A — Cheap deterministic post-processing (~1-2 days local CPU)

Three independent moves; each can be ablated. Run the composite eval
after each to bisect contribution.

#### A1. Voicing / silence gate on highres output

`tabvision.audio.highres.HighResBackend` currently runs with
`filter_config=False` (filters disabled). The `AudioFilterConfig`
primitives — `min_confidence`, `min_amplitude`, `min_duration_s`,
same-pitch `merge_gap_s` — are already implemented in
[`tabvision/tabvision/audio/filters.py`](../../tabvision/tabvision/audio/filters.py)
and exhaustively tested (18 tests in `test_audio_filters.py`).

**Move:** enable filters with a highres-tuned config. v0 defaults
(`min_confidence=0.3`, `min_amplitude=0.1`, `min_duration_s=0.03`)
were tuned for Basic Pitch's over-detection profile; highres
under-detects rarely but emits low-amplitude/short ghost events on
chord decays. Tune via grid sweep on the GuitarSet validation set.

**Sweep:**
- `min_confidence ∈ {0.0, 0.15, 0.3, 0.45}`
- `min_amplitude ∈ {0.0, 0.05, 0.1, 0.2}`
- `min_duration_s ∈ {0.0, 0.02, 0.05}`

Pick the (min_conf, min_amp, min_dur) tuple with the highest Pitch F1
that doesn't drop Onset F1 below 0.92. **48 combinations × 60 clips ×
~30 s inference = ~24 hr serial.** Mitigate by:

- Caching the raw highres output to disk per clip (one inference
  pass), then sweeping the filter config in-memory against the cached
  events. Reduces the sweep to seconds. New helper:
  `tabvision/scripts/eval/cache_audio_events.py`.

#### A2. Onset peak-picking adjustment

`HighResBackend.__init__` currently uses
`onset_threshold=0.3, offset_threshold=0.3, frame_threshold=0.1`.
These map directly into the underlying `hf_midi_transcription`
peak-picker.

**Move:** grid-sweep `onset_threshold ∈ {0.15, 0.3, 0.45, 0.6}` and
`frame_threshold ∈ {0.05, 0.1, 0.2}` using the cached-events trick from
A1 (caching here requires saving the model's probability outputs, not
just the decoded events — wider change). Alternative: run the 12-cell
grid live (12 × 30 min = 6 hr serial; tolerable overnight).

**Decision:** if A1 + A2 cleared the bar, skip the live sweep — A2
without live model access is a marginal lever.

#### A3. Same-pitch event merging

`AudioFilterConfig.merge_enabled=True, merge_gap_s=0.02` collapses
same-pitch events within 20 ms. Already in `filters.py`; just turn on.
Expected impact small (~0.2 pp Pitch F1) but free.

### Stage A acceptance check

After A1+A2+A3, run the full composite eval:

- **Pass:** clip-mean Pitch F1 ≥ 0.93 *and* Onset F1 ≥ 0.92 ⇒ ship
  Stage A, skip Stage B.
- **Gap:** 0.92 ≤ Pitch F1 < 0.93 ⇒ proceed to Stage B.
- **Fail:** Pitch F1 < 0.92 ⇒ diagnose; do not proceed to Stage B
  blindly. Likely a filter mis-tune that dropped real notes; revert
  and try a more conservative sweep range.

### Stage B — Basic Pitch pitch-only ensemble (~1-2 days local CPU)

Only run if Stage A leaves us 1-2 pp short.

**Move:** run [`tabvision.audio.basicpitch`](../../tabvision/tabvision/audio/basicpitch.py)
in parallel with highres on each clip. For each highres event:

- If a Basic Pitch event lands within onset tolerance with **the same
  pitch_midi** → confidence multiplier *up* (e.g. ×1.5).
- If no Basic Pitch event within onset tolerance OR all nearby BP
  events have different pitches → confidence multiplier *down* (e.g.
  ×0.5).

Then re-apply the Stage A voicing gate. Disagreement-downweighting
should drop spurious highres events more aggressively than agreement-
upweighting promotes confident ones; the asymmetry is intentional
because both backends are individually high-recall.

**Critical caveat:** ensemble onset/pitch matching uses the same
onset tolerance as the eval (50 ms). Do NOT cross-feed Basic Pitch
onsets into highres events — that taints the Onset F1 measurement.
Pitch-only, as the name says.

**Cost:** Basic Pitch is ~10s/clip on CPU (vs ~30s/clip for highres);
adds ~10 min total for 60-clip composite. Negligible.

## 3. Files to add / modify

### 3.1 New files

| Path | Purpose |
|---|---|
| `tabvision/tabvision/audio/voicing.py` | (optional) extract voicing-gate primitives from `filters.py` if we end up with a more highres-specific filter than `AudioFilterConfig` supports. If `AudioFilterConfig` covers the sweep cleanly, skip this file. |
| `tabvision/tabvision/audio/ensemble.py` | Stage B only. `pitch_agreement_reweight(highres_events, bp_events, *, onset_tolerance_s, agree_multiplier, disagree_multiplier) -> list[AudioEvent]`. Pure function; no I/O. |
| `tabvision/scripts/eval/cache_audio_events.py` | Cache raw highres output to disk per manifest clip. Speeds up filter-config sweeps from hours to seconds. Output: JSON-per-clip under `tabvision/data/cache/audio_events/`. |
| `tabvision/scripts/eval/sweep_audio_filters.py` | Read cached events + iterate over `AudioFilterConfig` grid, emit `docs/EVAL_REPORTS/audio_filter_sweep_<date>.md` with per-config Pitch / Onset F1. |
| `tabvision/tests/unit/test_audio_ensemble.py` | Stage B only. Pitch-agreement reweighting on synthetic event lists. |
| `tabvision/tests/unit/test_cache_audio_events.py` | Round-trip: cache → reload → compare. |
| `docs/EVAL_REPORTS/audio_filter_sweep_<date>.md` | Stage A output — sweep table. |
| `docs/EVAL_REPORTS/composite_baseline_phase1_<date>.md` | Final Phase 1 baseline (post-tuning). Side-by-side with the Phase 0 baseline. |

### 3.2 Modified files

| Path | Change |
|---|---|
| `tabvision/tabvision/audio/highres.py` | (a) Default `filter_config` from `False` to a tuned `AudioFilterConfig` instance (the Stage A sweep winner). (b) Possibly expose `onset_threshold` / `frame_threshold` defaults to the Stage A2 sweep winner. |
| `tabvision/tabvision/audio/filters.py` | Likely no change. If sweep reveals a needed knob the dataclass doesn't have (e.g. a percentile-based amplitude gate rather than absolute), add it as a new optional field with the current behavior as default. |
| `tabvision/tests/unit/test_audio_filters.py` | If a new knob lands in `AudioFilterConfig`, add coverage; otherwise no change. |
| `tabvision/scripts/eval/composite_eval.py` (already shipped in #11) | No change in Phase 1 itself. Phase 1 invokes the CLI as-is; the highres-side changes propagate via the backend's new defaults. |
| `LICENSES.md` | No change — Basic Pitch (Apache-2.0) is already cleared for the default pipeline. |
| `docs/DECISIONS.md` | Append a Phase 1 entry with: chosen filter config, threshold sweep winner, whether Stage B was needed, the artifact paths. |

### 3.3 NOT modified

- `tabvision/tabvision/fusion/**` — Phase 1 is purely audio-side.
- `tabvision/tabvision/eval/**` — the composite-eval harness from #11
  is the measurement substrate; it shouldn't change.
- `tabvision/tabvision/pipeline.py` — `run_pipeline` already plumbs
  the backend; no orchestrator change needed.
- `tabvision-server/**` — Phase 0 ground rule carries forward: no
  production behavior change in this phase.

## 4. Test plan

All tests under `tabvision/tests/{unit,integration}/`. Skip cleanly
when an optional dep (basic-pitch package) is missing.

### 4.1 Unit tests (new + extended)

| Test | Purpose |
|---|---|
| `test_audio_filters.py::test_highres_tuned_defaults_drop_low_confidence_chord_decay` | New: regression-locks the Stage A1 winner config so future highres updates don't silently regress. |
| `test_audio_ensemble.py::test_agreement_boosts_confidence` | Stage B: synthetic highres + BP events; agreement increases confidence by the documented multiplier. |
| `test_audio_ensemble.py::test_disagreement_drops_confidence` | Disagreement applies the disagree multiplier. |
| `test_audio_ensemble.py::test_onset_tolerance_respected` | A BP event 100 ms from a highres event is NOT counted as agreement (default tolerance 50 ms). |
| `test_audio_ensemble.py::test_no_bp_events_leaves_highres_alone` | When BP returns nothing, ensemble is a no-op. |
| `test_cache_audio_events.py::test_round_trip_preserves_event_fields` | Cached JSON → re-loaded `AudioEvent` round-trips every field. |
| `test_cache_audio_events.py::test_cache_keyed_by_clip_id_and_backend` | Re-runs use the cache; mismatched backend invalidates. |

### 4.2 Integration tests

| Test | Purpose |
|---|---|
| `test_phase1_e2e.py::test_filter_sweep_produces_report` | End-to-end: cache events for 2 fixture clips, run the sweep over a 4-cell grid, verify the report file lands with the expected columns. |
| `test_phase1_e2e.py::test_ensemble_path_runs_when_basicpitch_available` | Skip-if-no-basic-pitch; runs highres + BP ensemble on 1 fixture clip, asserts the event count is sensible. |

### 4.3 No regression on existing tests

The full Phase 0 test suite (107 tests) must still pass:

```bash
cd tabvision && ../tabvision-server/venv/bin/python -m pytest \
    tests/unit/test_bootstrap_ci.py \
    tests/unit/test_parsers_registry.py \
    tests/unit/test_parser_guitar_techs_midi.py \
    tests/unit/test_error_decomposition.py \
    tests/unit/test_eval_manifest.py \
    tests/unit/test_manifest_builder.py \
    tests/unit/test_composite_report_formatting.py \
    tests/unit/test_guitarset_audio_eval.py \
    tests/unit/test_audio_filters.py \
    tests/integration/test_composite_eval_smoke.py
```

Plus the new Phase 1 tests.

## 5. Commands to run

All from repo root, with the `tabvision-server/venv` python.

### 5.1 Cache highres events for the composite manifest (one-shot)

```bash
cd tabvision && ../tabvision-server/venv/bin/python -m tabvision.scripts.eval.cache_audio_events \
    --manifest data/eval/composite.toml \
    --backend highres \
    --output-dir data/cache/audio_events/highres/
# ~30 minutes on the 60-clip validation manifest
```

### 5.2 Filter sweep (Stage A1 + A3)

```bash
../tabvision-server/venv/bin/python -m tabvision.scripts.eval.sweep_audio_filters \
    --manifest data/eval/composite.toml \
    --cache-dir data/cache/audio_events/highres/ \
    --output ../docs/EVAL_REPORTS/audio_filter_sweep_$(date +%F).md
# Seconds, since cached events skip inference
```

### 5.3 Threshold sweep (Stage A2, only if A1 doesn't clear bar)

```bash
# Live grid over onset_threshold × frame_threshold (12 cells × ~30 min/cell = ~6 hr)
for ot in 0.15 0.3 0.45 0.6; do
  for ft in 0.05 0.1 0.2; do
    ../tabvision-server/venv/bin/python -m tabvision.eval.composite \
        --manifest data/eval/composite.toml \
        --backend highres --position-prior guitarset-v1 \
        --output /tmp/sweep_ot${ot}_ft${ft}.md \
        --bootstrap-n 1000 \
        --eval-harness-sha "phase1-sweep-ot${ot}-ft${ft}"
    # ... grep Pitch F1 / Onset F1 out of /tmp/sweep_*.md
  done
done
```

Threshold-sweep CLI argument support requires a small change to
`tabvision.eval.composite.main()` to forward `--onset-threshold` and
`--frame-threshold` into the backend constructor. Fold into Phase 1
work; not in #11.

### 5.4 Stage B ensemble (only if Stage A leaves a 1-2 pp gap)

```bash
../tabvision-server/venv/bin/python -m tabvision.eval.composite \
    --manifest data/eval/composite.toml \
    --backend highres --enable-ensemble basicpitch \
    --position-prior guitarset-v1 \
    --output ../docs/EVAL_REPORTS/composite_baseline_phase1_stage_b_$(date +%F).md \
    --bootstrap-n 10000 \
    --eval-harness-sha "$(git rev-parse --short HEAD)"
```

Requires a `--enable-ensemble` flag in `composite_eval` main —
new in Phase 1.

### 5.5 Final Phase 1 baseline

After whichever stage cleared the bar:

```bash
../tabvision-server/venv/bin/python -m tabvision.eval.composite \
    --manifest data/eval/composite.toml \
    --backend highres --position-prior guitarset-v1 \
    --output ../docs/EVAL_REPORTS/composite_baseline_phase1_$(date +%F).md \
    --decomposition-output ../docs/EVAL_REPORTS/tab_f1_error_decomposition_phase1_$(date +%F).md \
    --bootstrap-n 10000 \
    --eval-harness-sha "$(git rev-parse --short HEAD)"
```

## 6. Acceptance outputs

These are the artifacts whose existence + content gates Phase 2.

### 6.1 `docs/EVAL_REPORTS/composite_baseline_phase1_<date>.md`

Same format as the Phase 0 baseline. Headline numbers must show:

- Clip-mean Pitch F1 **≥ 0.93** (lower-95 CI ≥ 0.93 if achievable; mean
  is the headline given Stage A's small-effect-size sweep).
- Clip-mean Onset F1 **≥ 0.92** (no regression > 1 pp from Phase 0).
- Per-tier table same as Phase 0; expect both covered tiers' Tab F1 to
  rise modestly with the Pitch headroom.

### 6.2 `docs/EVAL_REPORTS/audio_filter_sweep_<date>.md`

Stage A artifact. Sweep grid + winner highlighted. Justification trail
for the new `HighResBackend` defaults.

### 6.3 `docs/EVAL_REPORTS/tab_f1_error_decomposition_phase1_<date>.md`

Companion six-bucket decomposition for the Phase 1 baseline. Expect
`pitch_off` and `missed_onset` to shrink relative to Phase 0;
`wrong_position_same_pitch` stays roughly flat (Phase 1 doesn't touch
fusion).

### 6.4 `docs/DECISIONS.md` Phase 1 entry

Append a single dated entry summarising the sweep winner, whether
Stage B ran, and the artifact paths.

### 6.5 CI verification

`pytest tabvision/tests/unit tabvision/tests/integration` passes
on the impl branch.

## 7. Decision tree

| Result | Action |
|---|---|
| Stage A: Pitch F1 ≥ 0.93, Onset F1 ≥ 0.92 | Ship Stage A. Skip Stage B. Open Phase 1 PR. |
| Stage A: Pitch F1 in [0.92, 0.93) | Run Stage B. Re-check. If now ≥ 0.93, ship the ensemble. |
| Stage A: Onset F1 < 0.92 (regression) | Revert the filter changes that dropped onsets (likely a too-aggressive `min_amplitude`). Re-sweep narrower. |
| Stage A: Pitch F1 < 0.92 (regression) | Revert. The filter sweep is at the wrong grid; widen `min_confidence` toward 0 and reduce `min_amplitude` toward 0. The default highres is already at ~0.915, so we should never get *worse* by adding filters; if we do, the cache is stale or the filter is buggy. |
| Stage B: still short of 0.93 | Phase 1 is exhausted on the cheap path. Open a DECISIONS.md entry calling Phase 1 *partial-pass*. Phase 2 (Lightning fine-tune) becomes mandatory rather than optional. |

## 8. Time + compute budget

| Item | Effort | Compute |
|---|---|---|
| Caching script + tests (3.1) | 0.5 day | none |
| Filter-sweep script + tests (3.1) | 0.5 day | none |
| Cache the 60-clip events (5.1) | 0.5 hr wall-clock | local CPU |
| Filter sweep (5.2) | seconds | local CPU |
| Threshold sweep (5.3, if needed) | ~6 hr wall-clock | local CPU |
| Stage A regression / locking tests (4.1) | 0.5 day | none |
| Stage B implementation + tests (3.1, conditional) | 1 day | none |
| Stage B eval run (5.4) | 1 hr wall-clock | local CPU |
| Final baseline + decomposition runs (5.5) | 1 hr wall-clock | local CPU |
| DECISIONS.md + report writing | 0.5 day | none |
| **Total (no Stage B)** | **~2 days engineering** | **~$0** |
| **Total (with Stage B)** | **~3-4 days engineering** | **~$0** |

## 9. Out of scope for Phase 1

- Anything in `tabvision/tabvision/fusion/`. Position priors, melodic
  prior, learned fusion all land in Phases 3 / 5 / 7.
- Anything in `tabvision/tabvision/video/`. Video is qualitative-only
  (D5) and ships separately in Phase 6.
- Any model training. Phase 1 is cheap deterministic tuning; if it
  exhausts without hitting 0.93, Phase 2 picks up with the Lightning
  free-tier fine-tune.
- Guitar-TECHS or EGDB tier coverage. Those land via Phase 0 §8 user
  actions, not in Phase 1.

## 10. Done definition

Phase 1 is **done** when:

- All §3.1 / §3.2 files exist on `impl/tab-f1-phase-1` with green
  tests.
- `docs/EVAL_REPORTS/composite_baseline_phase1_<date>.md` exists and
  meets §6.1.
- `docs/EVAL_REPORTS/audio_filter_sweep_<date>.md` exists and meets
  §6.2.
- `docs/DECISIONS.md` entry per §6.4.
- Pitch F1 ≥ 0.93 *or* DECISIONS.md entry calling Phase 1 partial-pass
  with the path forward to Phase 2 articulated.

Phase 2 (highres fine-tune on Lightning free tier) gets its own
implementation plan after Phase 1's evidence is in, mirroring this
plan-doc-first cycle.

## 11. Dependencies on PR #11 (Phase 0)

This plan assumes the following land on `main` first:

- `tabvision.eval.composite` (the harness + CLI used by every Phase 1
  measurement command).
- `tabvision/data/eval/composite.toml` (the 60-clip portable manifest).
- `tabvision.eval.bootstrap` (CI computation).
- `tabvision.eval.error_decomposition` (the six-bucket port).
- The 2026-05-13 Phase 0 baseline in `docs/EVAL_REPORTS/`.

If any of those don't make it into `main`, this plan needs revision
before code starts. **The gating note in §0 is therefore literal**:
Phase 1 is not just "blocked on user proceed," it's "blocked on the
Phase 0 substrate existing on the branch we're about to cut from."
