# Phase 5 — Fusion (Viterbi + chord-aware) Design

**Date:** 2026-05-06
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — pending sign-off
**Spec source:** `SPEC.md` §5 Phase 5, §8 module contracts.
**Branch:** `claude/refactor-eval` (forked from `refactor/v1`); merge back to `refactor/v1` on green.

## 0. Status snapshot

What `tabvision.fusion` looks like right now on `refactor/v1`:

| Module | Lines | State |
|---|---:|---|
| `candidates.py` | 50 | **Done.** `candidate_positions(pitch, cfg) → list[Candidate]`. Used by Phase 1 audio-only fusion. |
| `viterbi.py` | 119 | **Phase-1 placeholder.** `fuse(...)` raises `FusionError` whenever any `FrameFingering` carries non-zero logits ("video-aware fusion not implemented — this is a Phase 5 deliverable"). Greedy lowest-fret + continuity decoder works for the audio-only path (5 tests passing). |
| `playability.py` | 9 | **Stub.** Docstring only. |
| `chord.py` | 9 | **Stub.** Docstring only. |
| CLI | — | `--fusion-lambda-vision` flag not yet exposed. |

Phase 4 already produces `FrameFingering.marginal_string_fret() → (6, 25)` softmax per frame (`tabvision.video.hand.fingertip_to_fret`). Phase 5 consumes that.

Legacy reference: `tabvision-server/app/fusion_engine.py` (2,216 lines, 23 functions) and `tabvision-server/app/chord_shapes.py` (790 lines). Per the SPEC §3.3 module-boundary plan, we **port selectively** (hand-span, slide, monophony heuristics) rather than wholesale-translate. The Apr-24 learned-fusion attempt (LightGBM ranker) **did not ship** (LOOCV +0.3 pp vs +5 pp gate per `tools/outputs/position_selector_report-2026-04-29.md`); the lesson is that small ML on top of weak features doesn't beat structured search with informative evidence — Phase 5 takes the structured-search path.

## 1. Goal & acceptance bars

From SPEC §5 Phase 5:

- **Tab F1 ≥ 0.85** on the user eval set. Target 0.88 by Phase 9.
- **Chord-instance accuracy ≥ 0.80**. Target 0.85 by Phase 9.
- **Audio+vision must beat audio-only by ≥ 8 pp on Tab F1** (ablation report).

The user eval set = the 20-video iPhone-recorded training set, plus whatever Phase 1.5 annotation tooling adds to the four difficulty tiers. Today's audio-only baseline on that set is **exact F1 ≈ 0.51** (per `errors-2026-04-28_185743.md`). Phase 5's 0.85 bar therefore needs both (a) better audio (Phase 2 SOTA backbone) and (b) the audio+vision boost. Phase 5 alone is on the hook for the **+8 pp audio+vision delta**, not the absolute number — that's the readable signal that the fusion is doing real work.

## 2. Cost function

We score a sequence of decoded `(string, fret)` picks by a sum of **emission** terms (per pick) and **transition** terms (between consecutive picks). Lower total cost wins. All terms are negative log-probs (or proportional to them) — i.e. dimensionally consistent.

### 2.1 Emission cost per `Candidate c = (s, f)` for `AudioEvent ev`

```
E(c | ev, fingering_at_t) =
      -log P_audio(c | ev)              # audio prior on string/fret
  +   -λ_v · log P_vision(c | t)        # vision marginal at event time
  +   α_open · 1[f == 0] · open_bonus   # negative if c is on an open string
  +   α_low · f                          # mild lower-fret bias
```

- `P_audio(c | ev)`:
  - If `ev.fret_prior` is provided (Phase 2's `tabcnn` backend, when present), use it directly. Otherwise uniform over candidates.
  - Multiply by `ev.confidence` (the model's pitch posterior).
- `P_vision(c | t)`:
  - Look up the `FrameFingering` whose `t` is closest to `ev.onset_s`. Linear-interpolate between two adjacent frames if the gap is small (< 1 / fps).
  - `marginal_string_fret()[s, f]` is the per-(string, fret) cell of the (6, 25) softmax.
  - If no fingering carries evidence (`finger_pos_logits.size == 0` or all-zero) → fallback to uniform; `λ_v` is effectively zero for this event.
- `λ_v`: tunable, default `1.0`, exposed as `--fusion-lambda-vision` (CLI) and `lambda_vision` kwarg on `fuse()`.
- `open_bonus`: small constant (e.g. 0.5). Open strings are systematically under-represented in MediaPipe-derived `marginal_string_fret` because no fingertip is pressing — so we re-introduce them via this bonus.
- `α_low`: lower-fret bias (e.g. 0.05/fret). Keeps the decoder honest when audio + vision are both flat across candidates.

### 2.2 Transition cost between `prev = (s_p, f_p)` and `curr = (s_c, f_c)`

```
T(prev → curr) =
      β_shift · |f_c - f_p| / span_norm        # position-shift penalty
  +   β_span · max(0, |f_c - f_p| - max_span)  # hard hand-span barrier (kicks in beyond ~5 frets)
  -   β_string · 1[s_c == s_p]                 # same-string continuity bonus
```

- `span_norm = 12` (one octave), `max_span = 5` frets — calibrated from the legacy `fusion_engine.py` anchor system.
- `β_string` ≈ 0.5 — direct port of the existing `STRING_CONTINUITY_BONUS`.
- A "muted" / X transition is permitted by skipping cost contribution (technique flag set on the `TabEvent`).

### 2.3 Per-string monophony

Hard constraint baked into the **chord cluster** state space (§3.2), not a soft cost. Single-line Viterbi (§3.1) is monophonic by construction.

## 3. State spaces

### 3.1 Single-line Viterbi (`viterbi.py`)

Triggered when consecutive events are > 80 ms apart.

- States at event `i`: `candidate_positions(events[i].pitch_midi, cfg)` — typically 2–6 per pitch.
- Initial cost: `E(c_0)`.
- Recurrence: `cost[i, c] = E(c) + min_{c'} (cost[i-1, c'] + T(c' → c))`.
- Termination: pick the lowest-cost terminal state, backtrack.
- Worst case: `O(N × K^2)` for `N` events, `K ≤ 6` candidates per event. `N` is hundreds; trivial.

### 3.2 Chord cluster decode (`chord.py`)

A **chord cluster** is a maximal run of consecutive `AudioEvent`s pairwise within 80 ms onset distance. (SPEC §5: "simultaneous events ≤ 80 ms apart".)

For a cluster of `m` events:

- A **chord state** is an ordered tuple of m candidates `(c_1, …, c_m)` with:
  - **Per-string monophony:** all `s_i` distinct.
  - **Hand-span constraint:** `max(f_i for f_i > 0) - min(f_i for f_i > 0) ≤ max_span` (open strings exempt).
  - Order convention: low-pitch first (so the spelling is reproducible).
- State enumeration: cartesian product of candidates, filtered by the two constraints. With `m ≤ 6` (six-string guitar) and `K ≤ 6` per pitch, worst case `6^6 = 46 656` raw tuples — pruned aggressively to a few hundred valid ones.
- Emission cost for a chord state = sum of per-event emission costs.
- Transition between two chord clusters: collapse each cluster to its **lowest-fret pressed note** (the natural anchor point) and apply `T(prev → curr)` from §2.2 — keeps the inter-chord cost compatible with single-line transitions.
- Optional: `chord_shapes.py` templates from the legacy code give a prior over common shapes (open chords, barre, power). **Deferred to Step D below** — start without templates and only add if F1 demands.

The chord-cluster decode is itself a Viterbi over chord-states between clusters; single-line events are degenerate clusters of size 1.

## 4. Module responsibilities

```
tabvision.fusion.candidates   -- (done) candidate_positions, Candidate dataclass.
tabvision.fusion.playability  -- emission + transition cost helpers (pure functions, fully unit-tested).
tabvision.fusion.viterbi      -- (a) the public fuse() entrypoint; (b) single-line Viterbi; (c) dispatcher to chord.
tabvision.fusion.chord        -- chord cluster grouping + chord-state Viterbi.
```

`viterbi.fuse(events, fingerings, cfg, session, lambda_vision=1.0)` stays as the single public entrypoint per SPEC §8; behaviour switches internally based on whether `fingerings` carry evidence and whether events fall into chord clusters.

## 5. Port mapping (legacy → new)

| Legacy (`tabvision-server/app/fusion_engine.py`) | New | Notes |
|---|---|---|
| `_score_position_heuristic` | `playability.emission_cost` | Drop hand-anchor side-channel; subsume into structured Viterbi. |
| `_select_best_position` | replaced by single-line Viterbi | The greedy logic was the source of `wrong_position_same_pitch` errors. |
| `_optimize_chord_positions` | `chord.decode_chord_state` | The legacy version is greedy with backtracking; the new version is exhaustive over the (already-small) feasible set. |
| `_correct_slide_positions` | `playability.transition_cost` (built-in) | Slide/legato preference falls out of the same-string continuity bonus and the position-shift penalty — no separate post-pass. |
| `_correct_melodic_segments` | not ported; subsumed by Viterbi | Subsumed. Confirm via ablation. |
| `_postfilter_tab_notes` | not ported (yet) | Dedup + low-confidence isolated filter. Defer; revisit if Phase 5 has visible artifacts of this kind. |
| `_detect_techniques` | shallow port | Hammer-on / pull-off / slide tag inference based on consecutive same-string events. Spec §5 leaves bend/vibrato to Phase 7. |
| `chord_shapes.py` (templates) | optional Step D in `chord.py` | Defer — only adopt if needed. |
| `fuse_audio_only` | already ported (Phase 1 path) | Keep. |
| `fuse_audio_video` | replaced wholesale | The legacy version is the worst-performing module per `errors-2026-04-28_185743.md` (35.2% of loss is `wrong_position_same_pitch`). |

## 6. Step-by-step phasing within Phase 5

Each step is independently mergeable; each lands tests before behaviour.

### Step A — `playability.py`: pure cost helpers (~½ day)

Implement:
- `emission_cost(candidate, event, fingering_at_t, cfg, *, lambda_vision=1.0) → float`
- `transition_cost(prev, curr, cfg) → float`
- Constants for the weight hyperparameters (named, documented).

Tests (`tabvision/tests/unit/test_playability.py`, new):
- Emission: pure-audio (no fingering) reproduces the existing greedy decoder's preferences.
- Emission: vision evidence pulls a candidate that audio is indifferent on.
- Emission: open-string bonus correctly recovers fret 0 when MediaPipe marginal is uniform.
- Transition: same-string is cheaper than string-jump.
- Transition: hand-span barrier triggers only past `max_span`.

**Acceptance:** All new unit tests green. No change to `viterbi.fuse()` behaviour (Phase 1 tests still pass).

### Step B — single-line Viterbi (~1 day)

Replace `viterbi._greedy_audio_only` with a single-line Viterbi using `playability` costs. Keep the public `fuse()` signature.

Tests (extend `test_fusion_audio_only.py`):
- All five existing tests still pass (regression gate).
- Add: 4-event sequence where greedy picks the wrong string at event 3 but Viterbi recovers it via lookahead.
- Add: vision-uniform fingerings produce same output as no fingerings (sanity).
- Add: vision-decisive fingering moves the pick to a non-lowest-fret candidate.

**Acceptance:** All tests green. Run `tabvision/tests/eval/test_phase4_eval.py` (or its Phase 5 sibling, see Step E) and confirm no regression on the audio-only path.

### Step C — chord cluster decode (~1–1½ days)

Implement `chord.cluster_events(events, max_gap_ms=80)` and `chord.decode_clusters(clusters, fingerings, cfg, lambda_vision)` returning the per-event picks. Wire `viterbi.fuse()` to dispatch.

Tests (`tabvision/tests/unit/test_chord_fusion.py`, new):
- Two simultaneous events on the same string get one moved (per-string monophony).
- A 3-note chord has all picks within `max_span` of each other (hand-span constraint).
- A chord cluster with vision evidence prefers the vision-supported voicing.
- An open-chord shape (open strings present) is preferred over a barre when both are reachable and vision is uniform.

**Acceptance:** All tests green. Single-line tests still pass.

### Step D — CLI integration & lambda sweep (~½ day)

- Add `--fusion-lambda-vision FLOAT` to `tabvision.cli`. Default `1.0`. Pass through to `fuse()`.
- Document in CLI `--help`.
- Add `tabvision/tests/unit/test_cli_fusion_flag.py`: smoke that the flag round-trips into `fuse()`.

### Step E — Phase 5 acceptance eval (~1 day)

Add `tabvision/tests/eval/test_phase5_eval.py` modelled on `test_phase4_eval.py`. It:

1. Runs the full pipeline (audio + video) on each video in the user eval set.
2. Computes Tab F1 (string + fret + onset within ±50 ms) and chord-instance accuracy.
3. Runs the audio-only ablation (`λ_v = 0`) on the same set.
4. Asserts:
   - `tab_f1 >= 0.85` (the §5 bar) — **may be marked `xfail` until Phase 2 SOTA backbone lands**, with the understanding that today's audio is the bottleneck.
   - `tab_f1_audio_video - tab_f1_audio_only >= 0.08` — **the Phase-5-specific bar; this is the gate for "fusion is doing real work"**.
   - `chord_accuracy >= 0.80`.
5. Writes a markdown report to `tabvision-server/tools/outputs/phase5_eval-YYYY-MM-DD.md` summarising the ablation per video (mirrors the `finetune_baseline-*.md` convention).

**Acceptance for Phase 5 as a whole:** the `tab_f1_audio_video - tab_f1_audio_only >= 0.08` assertion passes. The absolute-Tab-F1 bar may be deferred to Phase 7 if audio is still the bottleneck — but if it is, that's a material finding and should land in `DECISIONS.md`.

## 7. Risks & open questions

- **Risk:** `λ_v = 1.0` may be wrong by an order of magnitude. Mitigation: Step E sweeps `λ_v ∈ {0, 0.5, 1, 2, 5}` and reports best per video and aggregate. If best is `0`, vision evidence is genuinely uncalibrated → SPEC §5 decision tree's `C2` branch (return to Phase 4).
- **Risk:** chord-state explosion on dense voicings. Mitigation: 6-string max plus monophony pruning bounds cardinality at 720 raw tuples; in practice the constraint cuts to <100. If a real video produces a worst-case cluster (>100 tuples), beam-search is a 5-line addition.
- **Risk:** open-string bonus over-fires when the player is fingering a fret-0 chord (e.g. capo-0 G major shape) and MediaPipe correctly says "no fingertip on the low strings." Mitigation: chord-cluster decode considers the whole shape — bonus is per-event, but the chord-state's hand-span constraint pulls the rest of the shape into a coherent fingering.
- **Open:** does Step C need `chord_shapes.py` templates as a prior? Plan says no — start without and add only if F1 demands. Tracked as a Step-C-follow-up if needed.
- **Open:** what's "the user eval set" for Step E? Today: the 20-video iPhone training set. Phase 1.5's annotation tool will add labelled clips across four difficulty tiers — those should fold into the same eval as they land.

## 8. Estimated effort

Steps A → E total **~4 working days** of implementation + writeup. Acceptance eval (Step E) is the slowest because it requires running the full pipeline on the eval set, which is gated on Phase 4's video stack working end-to-end on the iPhone videos (probably true today but worth confirming as Step 0 below).

## 9. Pre-flight (before Step A)

A quick 15-min sanity check before any code:

- Run `tabvision/tests/eval/test_phase4_eval.py` end-to-end on at least one iPhone video and confirm we get a non-empty `list[FrameFingering]` with non-uniform `marginal_string_fret`. If we don't, Step E is going to be useless and we should fix Phase 4's eval path first.

---

**For sign-off:** confirm (a) cost-function shape (§2), (b) module split (§4), (c) phasing/order of A–E. If those look right I'll start with Step A.
