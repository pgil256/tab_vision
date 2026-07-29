# Segment-level position-window fusion — design (pre-registration draft)

**Status: DESIGN — awaiting sign-off. No build until approved** (SPEC §0
rule 8; plan-doc-first workflow). Gates in §5 are written before any run and
must not be edited after numbers are seen, per the wire-sparse precedent
(`a8f5f2e`).

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
