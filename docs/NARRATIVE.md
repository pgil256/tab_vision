# TabVision — Project Narrative

*The story of the project: what it set out to do, what turned out to be hard,
what actually worked, and what's next. Written to be honest first — it doubles
as the source for a portfolio write-up or blog post. Every number below is
measured; see `docs/EVAL_REPORTS/` and `docs/DECISIONS.md` for the receipts.*

## What it's trying to do

TabVision turns a video of someone playing solo guitar into **tablature** — the
string-and-fret notation a guitarist actually reads. That sounds like "run a
transcription model," but the interesting part is exactly the part a pitch
model doesn't solve:

> **Pitch does not determine position.** The same note — say, E4 — can be
> played in five or six places on the neck. A correct transcription gets the
> pitch; a correct *tab* gets the specific string and fret the player used.

So the whole project lives or dies on **string assignment**, and that turns out
to be where audio alone runs out of information. TabVision v1 is a Python CLI,
scoped to **acoustic** guitar, audio-first, built as a set of swappable modules
behind stable contracts so each source of evidence can improve without
entangling the rest.

## The architecture, in one paragraph

The pipeline is split into modules with strict dataclass contracts (SPEC §8):
**audio** transcription → note events; **video** (guitar detector, fretboard
geometry, hand tracking) → per-frame position evidence; **fusion** (a Viterbi
playability model plus learned priors) → the actual string/fret decisions;
**render** → ASCII / MIDI / MusicXML / Guitar Pro. It was built in phases, one
at a time, each gated on a held-out eval set before the next began. The result
ships three ways: a local CLI, a Modal production deploy, and a one-command
"studio" loop that records from the browser and prints tab end-to-end.

## What was hard

**1. The string-assignment ceiling is real, and it's information-theoretic.**
Audio-only single-line Tab F1 tops out around **0.52**. When we decomposed the
errors, the loss was overwhelmingly one failure mode:
`wrong_position_same_pitch` — **322 of ~380 errors, with the pitch correct**.
The model hears the right note and puts it on the wrong string, because the same
pitch is acoustically near-identical across strings. This is a limit of the
input, not a bug to fix, and naming it honestly reshaped the entire v1 scope.

**2. Video was the obvious rescue — and it didn't work.** If audio can't see
the fretting hand, watch it. We built the whole chain: a YOLO-OBB fretboard
detector, MediaPipe hand tracking, and a geometric fingertip-to-fret map. And
the lever is real: feed the fusion *gold* string labels and Tab F1 jumps to
**0.973** (the oracle probe) — string assignment is the whole ballgame. But on
real, in-the-wild video (the GAPS classical-guitar corpus), the calibrated
video chain resolves contested strings at **0.574** — while the audio
playability prior already gets **0.778**. Video was *worse*, so fusing it in
degraded Tab F1 at any non-trivial weight, and no confidence gate recovered a
lift. A learned video model did worse still. A later probe (A14) checked the one
scenario we'd held open — chord frames — and found video is *anti-enriched*
exactly where audio fails: `P(video correct | audio wrong) = 0.285`. Video
string-resolution was **closed as a lever**, and every "video-assisted" stretch
target was retired. The chain stays in the repo as runnable, measured evidence —
just not as an acceptance target.

**3. Electric guitar is a different instrument to the model.** The transcription
backbone is acoustic-trained. Pointed at electric (Guitar-TECHS), its pitch F1
collapses **0.93 → 0.73** and clean-electric Tab F1 is **0.12**. The
off-the-shelf alternate checkpoint didn't help. With no training code in the
repo, closing electric means a fine-tune — a bounded, *paid* v2 project, not a
v1 gate. So v1 was scoped to acoustic on the strength of that measurement.

**4. The hardest discipline was not publishing numbers we hadn't earned.** The
project's original targets (0.94 single-line, 0.86 strummed, 0.85 chord
accuracy, 0.70 technique detection) were aspirations, not measurements. One by
one they were replaced with what the evidence supported or retired outright.
Most recently the technique-detection target — the last unmeasured one — was
baselined and came back at **0.00**: there is no technique detector in the
pipeline at all, and GuitarSet can't even label bends/slides discretely to
train one against. So "≥ 0.70" was retired rather than quietly carried. The rule
throughout: a target that has never been measured is not a target.

## What worked

**Honest scope, and it passed.** v1 narrowed to acoustic, audio-only — an
evidence-based decision, not a retreat — and then cleared its gates. Formal
acceptance (2026-06-03, GuitarSet held-out player 05, 60 clips):

| Metric | Gate | Measured (mean / lower-95) |
|---|---:|---:|
| Single-line Tab F1 | ≥ 0.45 | **0.523** / 0.457 |
| Strummed Tab F1 | ≥ 0.60 | **0.676** / 0.606 |
| Aggregate Tab F1 | ≥ 0.55 | **0.600** |
| Onset F1 | ≥ 0.92 | 0.94 / 0.92 |
| Pitch F1 | ≥ 0.90 | 0.93 / 0.90 |
| Latency (60 s clip, laptop CPU) | ≤ 5 min | ~45 s (0.74× realtime) |

**The fusion layer earns small, real gains — gated twice.** Turning raw note
events into playable tab is where judgment lives: a Viterbi playability model
plus priors. The wins here are honestly modest (~1–5 pp total across all
audio-side tuning), but each one has to clear **two** gate legs — in-domain
(GuitarSet) *and* cross-domain (GAPS) — before it ships. The chord-shape bonus
was the first constant to clear both; plenty of plausible ideas cleared one and
were rejected. That two-legged gate is what keeps the accuracy claims durable.

**The eval harness is the real deliverable.** Every claim rides a reproducible
run with bootstrap confidence intervals and a cross-domain check. The repo is
full of probes that *refuted* attractive ideas (video fusion, a bigger n-gram
corpus, an open-string bonus, melodic priors) — and keeping those refutations
visible is the point, not an embarrassment.

**Input robustness turned out to be a non-issue.** The product ingests
Opus-in-webm from whatever laptop or phone mic the user has; a degradation study
(A8) measured the eval-vs-product gap at **~0** across the capture chain. Effort
that would have gone to denoising went back into the model.

**License-clean by construction.** Permissive defaults, with the one AGPL
dependency (the YOLO detector) accepted deliberately and documented prominently;
non-commercial datasets kept out of any shipped weights; and the TAB / Guitar
Pro export dependencies (music21 BSD, PyGuitarPro LGPL) cleared with the copyleft
one held to an opt-in extra. The default pipeline is safe to demo in public.

## What's next

- **Phase 9 (now):** the finalization pass — this narrative, README accuracy
  claims wired to the eval reports, demo assets, the license CI check, and the
  `v1.0.0` release, all from automated and public-data evidence (private/manual
  annotation was deliberately removed from the critical path).
- **v1.1, audio-side only:** the remaining single-line levers are a *timbral
  string-ID* model and style/position priors — the latter measured as
  domain-sensitive (+22–29 pp on GuitarSet but −0.138 on GAPS classical), so it
  ships behind honest guardrails. Technique detection is greenfield from the
  0.00 baseline. Upside here is bounded and we say so.
- **v2, electric:** a spend-gated fine-tune of a separate `highres-electric`
  checkpoint. The "tone toggle" is already wired, so the electric model drops in
  without disturbing the acoustic one — deferred by budget, not by design.

## The takeaway

The story of TabVision isn't a leaderboard number. It's the engineering
judgment around one: measure before you claim, scope to what the evidence
supports, and let refuted ideas stay refuted. The finished v1 knows exactly what
it can do (acoustic, audio-only, ~0.52 single-line / ~0.68 strummed, real-time
on a laptop), exactly what it can't (resolve strings audio can't hear, transcribe
electric, detect techniques), and — importantly — *why* for each one.
