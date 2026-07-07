# D1 decision packet — SPEC §1.4.1 remainder + §15 (2026-07-06)

**For:** the user. One batched set of decisions; each item names the exact
SPEC edit implied. Nothing here changes the binding v1 gates (single-line
≥ 0.45, strummed ≥ 0.60, aggregate ≥ 0.55 — all ACCEPTED 2026-06-03).
**Already resolved (2026-07-02):** the 0.94 single-line video-assisted
reference is RETIRED (SPEC §1.4.1 amended, A2 attached).
**Evidence attached:** A2 (`v1_1_gaps_prior_guitarset_v1_2026-07-01.md`),
chunk-6 capstone (DECISIONS 2026-06-29), **A14
(`a14_video_complementarity_2026-07-06.md`, DECISIONS 2026-07-06)** — the
probe this packet was waiting on.

---

## D1-a. Retire the strummed 0.86 and chord-instance 0.85 video-assisted references — **recommend: RETIRE**

**What SPEC says now (§1.4.1):** "The strummed 0.86 and chord-instance 0.85
video-assisted references remain open pending the chord-frame video probe
(roadmap A14) … chord-frame video is the one axis where video plausibly beats
audio and is still unmeasured."

**A14 measured it. The hypothesis is refuted, not just unproven:**

- On chord-member notes audio resolves contested strings **better** than on
  singletons (0.819 vs 0.779) and video **worse** (0.542 vs 0.600). The
  "a chord shape is one static frame video can read" premise fails on real
  footage — occlusion and hand crowding hurt the CV chain most exactly there.
- No routed hybrid exists anywhere: video is right on only **5.8%** of the
  notes audio gets wrong, and P(video right | audio wrong) = **0.285** —
  *half* video's own marginal (0.574). Audio-uncertainty-keyed routing (the
  B4 trellis margin) loses at every threshold.
- Every axis is now measured: aggregate (capstone), fusion at any
  λ/gate/orientation (capstone), uncertainty-routed hybrid (A14), chord axis
  (A14).

**Proposed SPEC edit:** in §1.4.1, replace the "remain open pending A14"
sentence with a retirement mirroring the 0.94 one: the 0.86/0.85
video-assisted references are retired (A14 attached); audio-only chord
baselines stay recorded (0.52/0.48 at acceptance; val24 chord acc currently
0.51/0.61); **no new stretch numbers until demonstrated** — remaining paths
are audio-side (A5 chord-shape dictionary for chords; A12 timbral string-ID
for single-line, approval-gated).

**Cost of deciding wrong:** none binding — references are aspirational. The
risk of *keeping* them is planning against a number three independent
negatives say the current sensor chain cannot reach.

## D1-b. Expressive markings stretch (≥ 0.70 technique F1) — in or out — **recommend: OUT until baselined (measure-first)**

GuitarSet JAMS carries technique labels and `TabEvent.techniques` exists, so
a baseline eval is fully automated and free — but **no number has ever been
measured**, and rule 7 (2026-07-02 precedent) says don't publish stretch
numbers nothing has demonstrated. Recommendation: do NOT add ≥ 0.70 to SPEC
now; instead queue the free baseline (hours-class, rides the composite
harness) and set a stretch only from its measured value. If you prefer it in
SPEC today, mark it explicitly "unbaselined reference — not a gate."

## D1-c. Studio-condition eval tier (A8 harness) — **recommend: IN as a diagnostic tier, not a gate**

Every accuracy number in the repo is on clean corpus WAVs; the product
ingests MediaRecorder webm from laptop mics. A8 (week 3) re-encodes val24
through the real capture chain (ffmpeg-only, gold labels carry over).
Recommendation: add a named diagnostic tier to §1.4.1 ("studio-condition
val24 — reported, not gated") once A8's harness exists, so degradation is
tracked release-over-release. No gate until we see the first number.

## D1-d. SPEC §15 "Live open questions" is stale — **recommend: replace wholesale**

All five questions were written pre-Phase-1.5 and are answered by events:

| § | Question (short) | Status |
|---|---|---|
| 1 | Audit → promote work early? | Answered by Phases 0–5 landing; moot. |
| 2 | Eval tiers 4/4/4/3 incl. electric | Superseded: acoustic scope (§1.4.1), electric → v2; current sets are val24/60-clip GuitarSet + GAPS + Guitar-TECHS manifests. |
| 3 | Annotator CLI vs GUI | Built: `scripts/annotate/` (label_clips.py, frames.py); moot. |
| 4 | Preflight lenient vs strict | v1 shipped lenient + `tabvision check`/`diagnose`; web surfacing is roadmap B9. Decide there, not in §15. |
| 5 | NARRATIVE.md author | `docs/NARRATIVE.md` exists; final pass is Phase 9 (D4). |

**Proposed edit:** replace §15's list with the *actual* live questions — i.e.
this packet's D1-b/D1-c if you defer them, plus D2 (electric v2 sequencing),
D3 (export-deps license review), D4 (Phase 9 "proceed"). §15's framing
("ask after Phase 1.5") is long-expired.

---

## What this packet does NOT cover

- **D2** electric v2 go/no-go sequencing (spend-gated) — separate call.
- **D3** music21 + PyGuitarPro license review for exports (MIDI needs
  nothing) — separate call.
- **D4** Phase 9 kickoff — needs your explicit "proceed" per SPEC §0.
- SPEC edits themselves: per operating rule 3/§0.5 they happen only after
  your approval of the recommendations above.

## Suggested reply format

`D1-a: retire | keep` · `D1-b: out-until-baselined | in-as-reference` ·
`D1-c: in-as-diagnostic | out` · `D1-d: replace §15 | leave` — one line is
enough; I'll make the SPEC edits and log DECISIONS accordingly.
