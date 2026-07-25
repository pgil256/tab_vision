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

So the whole project lives or dies on **string assignment**, and that is where
the interesting engineering — and the project's two most instructive mistakes —
turned out to be. TabVision is a Python CLI, scoped to **acoustic** guitar,
audio-first, built as a set of swappable modules behind stable contracts so each
source of evidence can improve without entangling the rest.

## The architecture, in one paragraph

The pipeline is split into modules with strict dataclass contracts (SPEC §8):
**audio** transcription → note events; **video** (guitar detector, fretboard
geometry, hand tracking) → per-frame position evidence; **fusion** (a Viterbi
playability model, learned priors, and a physics-derived string-evidence
channel) → the actual string/fret decisions; **render** → ASCII / MIDI /
MusicXML / Guitar Pro. It was built in phases, one at a time, each gated on a
held-out eval set before the next began — and those contracts earned their keep
later, when the largest accuracy change in the project's history was added
without altering a single one. It ships four ways: a local CLI, a Modal
production deploy, a one-command "studio" loop that records from the browser and
prints tab end-to-end, and a Windows desktop shell.

## What was hard

**1. The string-assignment ceiling is real — but it was not where we said it
was.** Audio-only single-line Tab F1 sat around **0.52**. Decomposing the
errors, the loss was overwhelmingly one failure mode:
`wrong_position_same_pitch` — **322 of ~380 errors, with the pitch correct**.
The model hears the right note and puts it on the wrong string. That much held
up. The explanation attached to it did not: this document used to call the
ceiling *information-theoretic*, on the reasoning that the same pitch is
acoustically near-identical across strings.

It isn't. **Real strings are stiff**, and stiffness makes a note's overtones
stretch progressively sharp — an inharmonicity coefficient `B` that depends on
the string's gauge, tension, and speaking length. A given pitch played on a
thick low string and a thin high one has measurably different partial spacing.
That information was in the audio the entire time; we were not reading it.

Adding a physics-derived inharmonicity channel moves aggregate Tab F1 by
**+0.05 to +0.07**, and single-line — the tier the ceiling was supposedly about —
by **+0.08 to +0.18** depending on the player. The channel is derived from
published string specifications rather than fitted to the eval set, and abstains
per note when the partials are unreadable, so its failure mode is "no evidence,"
not "wrong evidence."

The lesson is not "we were wrong about the ceiling." It's **that
"information-theoretic" is a claim about the world, and it was being used as a
claim about our features.** The honest version was always "we can't currently
extract it." That gap sat unexamined for two months because the stronger
phrasing sounded more rigorous.

**And then the same instinct nearly bit twice.** The first measurement of this
channel came from a single held-out player and read `+0.1006` — it went into the
README as *the* number. Measuring all six players later showed the gain ranges
`+0.047` to `+0.101` and that player was the maximum. Nothing was mis-measured;
the estimate was just built on one draw from a population that turns out to vary
by a factor of two. A single held-out set protects against overfitting. It does
not, on its own, tell you the size of an effect.

**2. Video was the obvious rescue — and it still doesn't pay, but not for the
reason we first published.** If audio can't see the fretting hand, watch it. We
built the whole chain: a YOLO-OBB fretboard detector, MediaPipe hand tracking,
and a geometric fingertip-to-fret map. The lever looked real: feed the fusion
*gold* string labels and Tab F1 jumps to **0.973** (the oracle probe). But on
real, in-the-wild video (the GAPS classical-guitar corpus), the calibrated video
chain resolved contested strings at **0.574** while the audio playability prior
already got **0.778**. Video was *worse*, so fusing it in degraded Tab F1 at any
non-trivial weight. A learned video model did worse still. A probe (A14) then
found video appeared *anti-enriched* exactly where audio fails:
`P(video correct | audio wrong) = 0.285`.

**That 0.285 was a bug.** Re-running the identical probe — same clips, same
cached frames, same decoder, window, and timestamp protocol — while fixing a
fret-map error (the adapter mapped a unit-neck body joint to fret 24 instead of
using the calibrated fret map) moved the primary result from 0.247 to
**0.763** [0.741, 0.783]. Video is not anti-informative. It was mis-projected.

And yet the honest end-to-end verdict is unchanged. Most of that 0.763 is the
window simply being wide — the enrichment over the anchor's own marginal is only
**+0.048**. Running the live position solver against real audio predictions on
ten source-disjoint clips moved macro Tab F1 by **+0.000836**, with a 95%
interval whose lower bound is exactly zero; the development set moved
**−0.000155**. So video ships as explicit opt-in and stays out of the default
path — but on evidence of *negligible effect*, not of harm. The distinction
matters, because the earlier framing would have permanently closed a lever that
turns out to be merely weak.

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

**Then a structured search found the one big lever.** After the release, an
ROI-tiered accuracy program worked a queue of candidate routes, one bounded item
per iteration, each ending in a written verdict. Fourteen items ran. **Eleven
were negatives** — a second-opinion model merge (admitted notes only 0.18
precise), a symbolic contextual assigner (+0.0467 against a +0.05 bar), onset
snapping (the backend already beat spectral flux), a nylon table for classical,
per-instrument calibration, capo detection from pitch (refuted in principle — a
capo is pitch-identical to a transposition). One was the inharmonicity channel
above, worth more than everything else attempted, before or since, combined.

That ratio is the point. The program's value came from **killing candidates
cheaply** — banking model outputs to disk so variant sweeps cost seconds, and
deriving break-even thresholds before building anything. One example worth
keeping: rather than ask "does per-instrument calibration work?", we computed
the *oracle* — the best achievable score if you could pick each player's
calibration perfectly, on the test set. It was +0.0027. No estimator could
rescue that, so a multi-week build closed in a single iteration.

**The fusion layer earns small, real gains — gated twice.** Turning raw note
events into playable tab is where judgment lives: a Viterbi playability model
plus priors. Most individual wins here are modest, but each has to clear **two**
gate legs — in-domain (GuitarSet) *and* cross-domain (GAPS) — before it ships.
The chord-shape bonus was the first constant to clear both; plenty of plausible
ideas cleared one and were rejected. That two-legged gate is what keeps the
accuracy claims durable. The physics channel cleared its cross-domain leg *by
construction*: it is scoped to instruments it has a table for, so classical,
electric, capo, and alternate-tuning sessions abstain by proof rather than by
measurement — a unit test replacing a two-hour run.

**Held-out data is worth the discipline it costs.** The player-05 split stayed
sealed and was opened only at pre-declared checkpoints. It earned that: a
uniform level correction to the physics table measured **+0.0160 [+0.0088,
+0.0233]** on development data and **−0.0066 [−0.0224, +0.0079]** on the
held-out player — non-overlapping intervals. It had been built, tested, and
written up as a shipping candidate. The held-out run reverted it. The effect was
physically real but instrument-specific, and development data could not see the
difference.

**The eval harness is the real deliverable.** Every claim rides a reproducible
run with bootstrap confidence intervals and a cross-domain check. The repo is
full of probes that *refuted* attractive ideas (video fusion, a bigger n-gram
corpus, an open-string bonus, melodic priors) — and keeping those refutations
visible is the point, not an embarrassment.

**Input robustness turned out to be a non-issue.** The product ingests
Opus-in-webm from whatever laptop or phone mic the user has; a degradation study
(A8) measured the eval-vs-product gap at **~0** across the capture chain. Effort
that would have gone to denoising went back into the model.

**License posture is tracked per artifact, not asserted in aggregate.** The
project is personal and non-commercial (SPEC §1.5), which permits NC-licensed
datasets and weights. Default *code* dependencies stay permissive with the one
AGPL dependency (the YOLO detector) accepted deliberately and held to an opt-in
extra. Default *artifacts* are mixed: the classical route's priors are derived
from the GAPS train split and inherit CC-BY-NC-SA, so the default pipeline is
not commercially redistributable as-is. Every NC-derived artifact is labeled in
`LICENSES.md` precisely so a future commercialization knows what to replace, and
a CI check enforces the dependency half. Private and user recordings are banned
from every training, eval, and labeling role.

## What shipped since v1.0.0

- **The physics string-evidence channel**, on by default for clean steel-string
  acoustic — the single largest accuracy change in the project's history.
- **Capo handling.** With `--capo N`, the position prior transforms
  covariantly, worth **+0.387** at capo 2 over the previous routing, which was
  not a shortfall but an outright collapse. Capo stays user-supplied because
  detecting it from audio is impossible in principle.
- **A Windows desktop shell** (WPF/.NET 8) over the CLI: pinned installer,
  first-run environment and weight bootstrap that resumes after interruption,
  audited offline-after-bootstrap transcription, and in-app camera + microphone
  recording. It is **thin and disposable by design** — the pipeline is a moving
  target and the shell is expected to be rebuilt.
- **FretCam**, a live fretboard/hand-position HUD, and the opt-in bridge that
  feeds its coarse position windows into fusion.

## What's next

- **Widen the physics channel.** It currently applies to about a quarter of
  notes — the ones whose partials are readable. Coverage, not effect size, is
  now the binding constraint on the single-line tier.
- **Technique detection** is greenfield from a measured 0.00, and needs a
  technique-labelled corpus to score against at all.
- **v2, electric:** a spend-gated fine-tune of a separate `highres-electric`
  checkpoint. The "tone toggle" is already wired, so the electric model drops in
  without disturbing the acoustic one — deferred by budget, not by design.
- **Video** stays opt-in unless a materially better position solver appears. The
  mechanism is sound and the end-to-end effect is ~0; those are compatible.

## The takeaway

The story of TabVision isn't a leaderboard number. It's the engineering judgment
around one: measure before you claim, scope to what the evidence supports, and
write down what failed. The current default knows what it can do (acoustic,
audio-first, **0.73 single-line / 0.74 strummed** on a held-out player, faster
than realtime on a laptop CPU), what it can't (transcribe electric, detect
techniques, resolve a capo it wasn't told about), and *why* for each.

If there's one lesson worth extracting, it's about the two mistakes this project
made in opposite directions. It called a limitation *information-theoretic* when
it was merely unextracted, and nearly left 0.10 Tab F1 on the table. And it
banked a refutation — "video is anti-informative" — that was a coordinate bug,
and came within one re-run of permanently closing a live lever. Both errors were
confident, well-written, and supported by real measurements. What caught them
was the same thing in both cases: **being able to state the mechanism, not just
the number.** A result whose mechanism you can't articulate is a result you
haven't finished checking.
