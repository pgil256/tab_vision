# Segment-level position-window fusion — design (pre-registration draft)

**Status: STAGE 1 APPROVED 2026-07-29 (user sign-off). Stage 2 still
requires G1 to pass first.** Gates in §5 are written before any run and
must not be edited after numbers are seen, per the wire-sparse precedent
(`a8f5f2e`). §5a below freezes every remaining free parameter — the
aggregation form §3.2 deferred "to implementation time", the bonus cap §3.3
referred to §5, and the gold-window degradation of §4 — and was committed
**before the Stage 1 script was run**, so no constant in it can have been
chosen with a number in view.

## 1. Problem

Same-pitch-wrong-position remains the dominant Tab F1 loss. Three
measurements frame the opportunity:

1. **The headroom is temporal.** `string_assignment_phase0_2026-07-27.md`:
   an oracle supplying one fret zone per 1 s window is worth **+0.2756 macro
   Tab F1** on held-out player 05; per 4 s joint window **+0.1446**.
2. **Audio alone did not extract it.** `string_assignment_phase1_2026-07-27.md`:
   the rule-based segment decoder banked **+0.0017**. It retains top-3 exact
   candidate paths per clip with mean second-path margin **0.1826 nats** —
   the alternatives are close, and the decoder is guessing between them.
3. **The per-note video bridge structurally cannot supply it.**
   `fretcam_audio_bridge_fixed_policy_2026-07-24.md`: +0.000836 [0.000000,
   0.001994] on source-disjoint-10. The bonus is capped at 1 nat and scaled
   by instantaneous observation confidence, applied note-by-note.

**Anecdotal illustration (not eval evidence).** A 28 s personal acceptance
clip (2026-07-28, per-user request; private recordings remain banned from
all measured training/eval roles per DECISIONS 2026-06-11 — this clip
informs design only and appears in no gate). Instrumenting
`apply_position_window_priors` showed FretCam's stabilized position track
was zone-correct for the full clip (V→IV, II ×15 consecutive locks,
V, III→I) on oblique webcam framing, with confidence crushed to 0.20–0.46
by foreshortening. At every fixable wrong-position note the window pushed
the correct direction at ±0.03–0.11 nats — and flipped nothing, because
audio-side preference gaps exceed that. One wobbly observation (VII for V)
pushed the wrong way and was contained by the cap. Conclusion: the
*consistency* of consecutive agreeing observations carries the signal; the
per-note instantaneous-confidence weighting discards it.

## 2. Claim under test

Aggregating stabilized position windows over a segment's time span can
break segment-path ties that per-note bonuses cannot, yielding a Tab F1
lift on GAPS clean-12 that the per-note bridge does not.

The break-even bar is explicit: at segment level, video does not need to
beat the 0.778 audio playability prior note-by-note; it needs to separate
retained candidate paths whose margins average ~0.18 nats.

## 3. Design sketch

Offline (cache-only) first; no §8 changes; fretcam stays quarantined with
the existing one-way import through `tabvision.video.position`.

1. **Segment paths.** Reuse the Phase 1 segment decoder's retained top-k
   exact paths per segment (already produced and cached for clean-12-style
   runs; k=3).
2. **Window agreement score.** For each retained path and each causally
   valid observation in the segment's span (same causality rule as the
   bridge: lookback ending 30 ms before onset), score agreement as the
   fraction of the path's fretted notes whose fret lies in the observation's
   window (open/capo notes excluded from both numerator and denominator —
   the open-string exemption is inherited unchanged, not relitigated here).
   Aggregate per path with a robust statistic over observations (median
   agreement × log(1 + n_obs) — exact form frozen at implementation time,
   before any eval run), so a single wobbly observation cannot dominate:
   the personal-clip probe showed exactly one wrong-direction observation
   in 44.
3. **Rerank.** Add a capped segment-level bonus (cap pre-registered in §5)
   to each retained path's score; re-select the winning path. Abstain —
   bit-identical output — when a segment has zero locked observations, when
   all retained paths agree equally, or when the winning path is unchanged.
4. **No new learned artifacts.** Deterministic code and frozen constants,
   as in Phase 1.

## 4. Two-stage plan

**Stage 1 — ceiling probe (gold windows, an afternoon, $0).** On GAPS
clean-12, synthesize position windows from gold annotations, degraded to
FretCam-like statistics (precision 1.0, stable coverage ~0.4, observation
cadence and confidence distribution matched to the F8 cache). Rerank the
retained paths per §3. This bounds what real FretCam observations could
ever deliver through this mechanism.

**Stage 2 — real observations (only if Stage 1 fires).** Replace gold
windows with the actual cached FretCam observations from the F8 evaluation
runs (no new inference needed). Same reranker, same gates.

## 5. Pre-registered gates

- **G1 (Stage 1 go/no-go):** gold-window rerank aggregate Tab F1 delta on
  clean-12 ≥ **+0.010** with no per-clip regression worse than −0.002.
  Below that: bank the negative, close the line — the mechanism's ceiling
  is too low regardless of detector quality.
- **G2 (Stage 2 verdict):** real-observation delta with paired
  clip-stratified bootstrap **CI lower bound > 0**, no per-clip regression
  worse than −0.002, and delta strictly greater than the per-note bridge's
  +0.000836 on the same clips. Runtime within the +20% decode allowance.
- **G3 (generalization, required before any default-path discussion):**
  repeat Stage 2 on the ten source-disjoint clips (the F8 primary set).
  Same CI and no-regression bars. Passing G2 but not G3 ships nothing.
- Decision tree is exhaustive: any outcome not matching a gate above is a
  banked negative. No post-hoc statistic substitution (wire-sparse rule).

## 5a. Frozen Stage 1 constants (written before the first run, 2026-07-29)

**Corpus and inputs.** GAPS clean-12, all twelve clips present (47.1 min of
audio); gold from the `gaps_musicxml_tab` parser. Audio events come from
`highres-ensemble` and are cached once at
`$TABVISION_DATA_ROOT/models/q6_gaps_cache/{clip}.ensemble.json` (the q6
gate's convention); **both arms read the identical cached events**, so the
audio half cannot differ between them.

**Session and routing.** `SessionConfig(instrument="acoustic", tone="clean",
style="fingerstyle")`, `GuitarConfig()` (standard tuning, capo 0), `gaps-v1`
pitch-position prior, `gaps-seq-v1` sequence context — q6's exact
configuration. Recorded caveat: `segment-v1` is admitted only by
`_automatic_acoustic_domain`, so this probe measures the mechanism under the
q6 session, **not** under a session explicitly tagged classical/nylon, where
the decoder abstains to `baseline` by construction and the delta is exactly
zero.

**Decoder.** `decode_segment_v1_with_analysis(..., k_paths=3)` with
`DEFAULT_SEGMENT_CONFIG`, which is bit-identical to the frozen Phase 1
winner `prior_0p5`. Baseline arm = `paths[0].events`, unmodified.

**Gold-window synthesis (the oracle, degraded).** Candidate observations on a
fixed **4.0 Hz** grid. At timestamp `t` the gold notes with onset in
`[t-0.25, t+0.35]` and `fret > 0` define the hand: `P = min(fret)`; the
observation is **dropped** when that set is empty or when `max(fret) > P+4`
(one window cannot cover the span, where real FretCam destabilises). Emitted
window is exactly the validity contract
`(0, *range(max(1, P-1), min(max_fret, P+4)+1))`, `state="locked"`,
`confidence=0.26` — the documented FretCam median from Appendix A.
Confidence enters **only** the `>= 0.20` validity gate and never the
reranker score, which is the design's whole point (§1.3), so this constant is
inert by construction. Coverage is then degraded to the F5c frozen figure
**0.416** by a deterministic Bresenham retention (`floor((i+1)*0.416) >
floor(i*0.416)`) — no RNG, no seed. Precision is 1.0 by construction.

**Reranker (the form §3.2 deferred).** Observation `o` is attributed to
segment `s` when `s.start_onset_s - 0.18 <= o.timestamp_s <= s.end_onset_s`
(0.18 = the bridge's 0.03 lead + 0.15 lookback, so an observation just before
a segment's first onset still counts). For retained path `p`:

```
agreement(p, s, o) = |{fretted notes of p in s with fret in o.window_frets}|
                     / |{fretted notes of p in s}|        # skip if denom 0
raw(p)             = median over all (s, o) of agreement  *  log(1 + n_obs)
```

Open/capo notes are excluded from numerator and denominator, inherited
unchanged from the bridge. `n_obs` is the number of contributing `(s, o)`
pairs and is common to all paths, so the log factor scales the *separation*
between paths rather than any single path's score — which is the
"consistency of consecutive agreeing observations" the design argues carries
the signal.

**Rerank and cap.** Applied as a capped *penalty* relative to the
best-agreeing path, so the cap bounds video's total influence without
saturating every path into a tie:

```
penalty(p)       = min(CAP, WEIGHT * (max_q raw(q) - raw(p)))
adjusted_cost(p) = p.cost + penalty(p)
CAP = 1.0 nat   (inherited from MAX_POSITION_LOG_BONUS)
WEIGHT = 1.0
```

Winner is `argmin adjusted_cost`, ties resolved to the lowest original index
so an exact tie keeps the baseline path. **Abstain — bit-identical output —**
when the clip yields no valid observations, when all `raw(p)` are equal, or
when the winner is already `paths[0]`.

**Scoring.** Per-clip `tab_f1(predicted, gold)`; aggregate = unweighted mean
over the twelve clips; delta = arm − baseline. G1 is read off exactly these
numbers, with no post-hoc statistic substitution.

## 6. Non-goals (standing evidence against each)

- Raising the per-note cap or weight globally — ungated/overweighted video
  measured **−0.15 to −0.20** (`v1_1_gaps_video_chain_2026-06-22.md`,
  `phaseA_720p_cache_rebuild_2026-07-27.md`).
- Lowering `min_clip_coverage` — measured cost 0.05–0.20.
- Confidence-keyed routing — A14 do-not-retry.
- Changing the open-string exemption — real limitation (it makes
  open-string misassignments unfixable by this channel), but it needs its
  own pre-registration; not bundled here.
- Any use of private/user recordings in the gates.

## 7. Data, licensing, cost

GAPS clean-12 + existing F8 observation caches (CC-BY-NC-SA media, never
committed; acceptable under the 2026-07-20 personal-noncommercial posture).
No training, no GPU, no new dependencies, CPU-only, $0. New code lives in
eval scripts until G2/G3 pass; production wiring is a separate approval.

## 8. Relationship to other open work

- The eval/production parity fix (separate session, 2026-07-28) is
  orthogonal but should land first so the legacy path cannot contaminate
  any A/B.
- L2/C2 (live product value) proceed independently — this design addresses
  the automatic pipeline only.
- If G1 fails, the honest conclusion is that position windows carry too
  little information at segment granularity too, and video's remaining
  value is entirely in the live/review surfaces.

---

## Appendix A — 2026-07-28 personal-clip instrumentation record

Design context only; nothing here enters a gate (private-recording posture,
DECISIONS 2026-06-11 / 2026-07-20). Clip: 28 s, 720p webcam, oblique
lap/couch framing; two `transcribe` arms (`--no-video` and
`--video-backend fretcam`) produced **byte-identical** tabs; the probe
wrapped `apply_position_window_priors` on a third run.

**FretCam's stabilized track was zone-correct on framing this oblique.**
44 locked/holding observations; confidence 0.20–0.46 (median ≈0.26),
crushed by foreshortening — but the position story matched the playing:

| t (s) | FretCam | actual zone |
|---|---|---|
| 0.4–4.4 | V → IV | frets 5–8 |
| 7.7–13.6 | II (15 consecutive locks) | frets 1–4 |
| 16.3–17.9 | one VII wobble, then V | frets 5–8 |
| 22.5–27.8 | III → I | open/low |

**Per-error record** (16 prior mutations across 14 ambiguous events;
0 decode changes):

- G3 played at D-5, decoded A-10: window boosted fret 5 (+0.109), penalized
  fret 10 (−0.108). No flip.
- F#4 played at e-2, decoded B-7 (×2): fret 2 boosted (+0.095 / +0.057),
  fret 7 penalized. No flip.
- G3 decoded as open-G: unfixable by construction — the open-string
  exemption keeps fret 0 supported under every window.
- The lone VII wobble nudged fret 10 over fret 5 (+0.054) — wrong
  direction, contained by the cap.

Reading: right evidence, right notes, right direction, ±0.03–0.11 nats of
weight — below the audio prior's preference gaps. Per-note
instantaneous-confidence weighting discards the consistency (15 agreeing
consecutive locks) that carries the real signal. Hence §3's segment-level
aggregation.

## Appendix B — issue ledger from the same session (for tracking, not gating)

| issue | disposition |
|---|---|
| Same-pitch-wrong-position (~8–10 of ~70 notes on the clip) | this design |
| Production legacy path applies ungated `FrameFingering` evidence (`vision_evidence.py` protections unwired) | fix in flight, separate session 2026-07-28 |
| First-pass descending bass (A3, G#3, F#3) missed; caught on the repeat | undiagnosed; investigate onset/level thresholds after `diagnose` is fixed |
| `tabvision diagnose` hard-requires basic-pitch; empty report under the default `auto` backend | queued small fix: route through the same backend resolution as `transcribe` |
| `--json` envelope carries only status/timings — no events, no video diagnostics; `position_window_prior` logs nothing even at `-v` | queued small fix |
| Preflight passes severely oblique framing (`GUITAR_VISIBLE` is the only check) | queued: surface the 118-forensics (wires/frame, nut visibility, orientation determinacy) as preflight findings |
| ASCII renderer: no slide semantics; omits empty string rows per system | queued cosmetic fixes |
| FretCam live UX (from L2 attempt 1, see `docs/fretcam-loop-state.md`): no board re-acquire control; guidance text conflates no-detection with fingertip-gate rejection | queued prototype fixes before the clean L2 re-run |
