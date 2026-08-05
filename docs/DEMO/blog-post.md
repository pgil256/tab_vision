# The Same Note, Six Places: Seven Months of Trying to Read a Guitarist's Hands

*The full story of TabVision — every hypothesis, every failed experiment, the one
that worked, and why the biggest accuracy win came from 19th-century physics
instead of a bigger model. Every number in this post traces to a reproducible
eval report in the repo.*

---

Play an E4 on a guitar. Now play it again, somewhere else — there are five or
six places on the neck that produce that exact pitch. A transcription model can
tell you *what* you played with better than 90% accuracy. It cannot tell you
*where*. And "where" is the entire point of tablature, the notation guitarists
actually read.

That gap is the whole project. TabVision started in January 2026 as a weekend
shaped idea — point a camera and a microphone at someone playing guitar, get tab
out — and became 556 commits, four shipped frontends, one formal release, a
fourteen-item experiment queue with eleven negative results, and a decoder that
gets its biggest single boost from string stiffness physics derived from
manufacturer spec sheets.

This is the start-to-finish account: the hypotheses, the failures, the pivots,
and what I'd tell anyone attempting an applied-ML project of their own.

## Act I: The naive build (January)

The first commit is an Electron + Flask demo: drag in a video, fake a
processing job, show a tab editor. Within a week it had a real audio pipeline,
a video pipeline, and a fusion stage — and a set of accuracy targets I would
spend months living down: 0.94 single-line accuracy, 0.86 strummed, 0.70
technique detection.

None of those numbers had ever been measured. They were aspirations written in
the confident voice of specifications, and the project's later rule — **a
target that has never been measured is not a target** — exists because of
them. When the technique-detection target was finally baselined, it came back
at exactly 0.00: there was no technique detector in the pipeline at all, and
the eval dataset couldn't even label bends and slides to train one against.

The original hypothesis stack looked like this:

1. Audio gets you pitch (mostly solved, use an off-the-shelf model).
2. Video gets you position (watch the fretting hand).
3. Fuse them, render tab, done.

Hypothesis 1 held. Hypothesis 2 consumed roughly three months and is one of
the most instructive failures I've had. Hypothesis 3 turned out to hide the
real problem entirely.

## Act II: The video era, or, the obvious idea (spring)

If audio can't see the fretting hand, watch it. I built the full chain: a
YOLO-OBB oriented-bounding-box detector for the fretboard, MediaPipe hand
tracking for fingertips, and a geometric map from fingertip coordinates to
fret positions, calibrated on the rule of 18 (the geometric series that
spaces frets on a neck).

The lever was provably real. An oracle probe — feed the fusion stage *perfect*
string labels and see what happens — jumped Tab F1 to **0.973**. The
information, if you could extract it, was worth nearly everything.

The extraction failed. On real in-the-wild video, the calibrated chain
resolved contested strings at **0.574** accuracy — while the audio-only
playability prior, the thing video was supposed to rescue, already scored
**0.778** on the same decisions. Fusing in a signal worse than your baseline
degrades the result at any weight. A learned video model did worse still. A
later probe appeared to deliver the coup de grâce: video was *anti-enriched*
exactly where audio failed — P(video correct | audio wrong) = 0.285.

Hold that number. It comes back in Act V, and it's wrong.

Two structural facts doomed the era regardless. First, the labelled video
corpora that exist are 640×360 — at that resolution there is provably no
string-level signal to learn (a classifier's likelihood ratio came out at
1.03, indistinguishable from "no information"). Second, self-occlusion: the
fretting hand hides its own fingertips from most camera angles at exactly the
moments that matter.

**The pivot:** scope v1 to acoustic guitar, audio-first, and rebuild as a
disciplined CLI — swappable modules behind strict dataclass contracts, each
phase gated on a held-out eval before the next began. Electric was measured
(pitch F1 collapses 0.93 → 0.73, Tab F1 to 0.12 on an acoustic-trained
backbone) and explicitly deferred to a paid fine-tune rather than papered
over.

In June, v1.0.0 passed formal acceptance on a sealed held-out player:
single-line Tab F1 0.523, strummed 0.676, onset F1 0.94, pitch F1 0.93,
faster than realtime on a laptop CPU. Honest numbers, none of them
impressive. The interesting part is what the error decomposition said:
**of ~380 single-line errors, 322 were the same failure** — right pitch,
wrong position. The model heard the note and put it on the wrong string.

I wrote in the docs that this ceiling was "information-theoretic": the same
pitch is acoustically near-identical across strings, so audio fundamentally
can't tell. That sentence sat in the repo for two months. It was wrong, and
the way it was wrong is the best lesson in the project.

## Act III: The experiment queue (July)

Post-release, I ran a structured accuracy program: a ROI-ranked queue of
candidate improvements, one bounded experiment per iteration, each ending in
a written verdict — shipped, or closed-negative with the reason banked.
Fourteen items ran. **Eleven were negatives.** A sample, because the negatives
are where the craft lives:

- **Second-opinion model merges.** Merge notes from a second transcription
  model (MuScriptor) where the primary is uncertain. It passed the
  "is there anything to gain" gate by 3.8× — and failed on arrival: its
  admitted notes were only 0.18 precise against a derived break-even of
  0.528. The derivation is the keeper: the break-even precision for admitting
  outside notes is computable from your own F1 and match rate, and *volume
  cancels* — how many notes a rule admits never changes the sign of the
  merge, only its magnitude. Every later candidate got priced against that
  bar before anyone built anything. A TabCNN-family retry in August failed
  the same gates.
- **A symbolic sequence model.** A 414k-parameter transformer trained on 34
  million notes of symbolic tab, rescoring the decoder's candidate lattice.
  It cleared its control (proving *context*, not corpus statistics, was the
  active ingredient) and still missed the pre-declared bar by 0.0033 — and
  the miss had structure: it helped chords six times more than single lines,
  and single lines were 77.5% of the loss. Banked, closed.
- **Onset snapping.** Align note starts to spectral-flux peaks, a published
  win on piano. Here it *created* timing errors (the timing-error bucket rose
  15 → 41): the ensemble's onsets were already better than the flux peaks I
  was snapping to. The published result assumed a detector whose timing was
  the weak link; mine wasn't.
- **Capo detection from audio.** Refuted in principle, not just in practice:
  a capo at fret 2 produces the identical pitch set to no-capo transposed up
  two semitones, so no pitch-based detector can ever separate them. The probe
  recovered 1 case in 60. The capo stays a user-supplied input forever, and
  that's a proof, not a shrug.

And then there was Q6.

## Act IV: The physics (the one that worked)

Real strings are stiff. Stiffness resists bending, and that pushes a
vibrating string's overtones progressively *sharp* — the 10th partial of a
note might land at 10.04× the fundamental instead of 10×. The size of the
stretch is captured by one number per string, the inharmonicity coefficient
**B**, and B depends on the string's core diameter, its material, and its
speaking length. Here is the punchline: **the same pitch played on two
different strings has measurably different B** — my candidate pairs differed
by 1.6–1.8× — so the overtone spectrum of a single note carries a fingerprint
of which string produced it.

The information was in the audio the entire time. My "information-theoretic
ceiling" was actually "features I hadn't extracted." The claim about the
world had been quietly substituted for a claim about my code, and it survived
two months because the stronger phrasing *sounded* more rigorous.

The build ran as a gate ladder, and every rung taught something:

1. **Separability precursor** (no data spent): every ambiguous candidate pair
   sits ≥4 frets apart, worth a 1.59–1.78× B ratio before any per-string
   differences. Clearable if B is estimable to ~25%.
2. **Estimator, self-validated on synthetic strings first** — which caught a
   real bug (a partial-search window that grew as k^1.5 and swallowed
   neighbouring partials by the 10th, returning confidently wrong
   fundamentals). Invisible on real audio; fatal to interpreting a failure.
3. **The gates.** String classification from B: 0.895 accuracy on hexaphonic
   pickup data, **0.920 on the ordinary mono microphone** — against a
   count-prior control flat at 0.65. The mic *beating* the dedicated
   per-string pickup was the best surprise of the project: the pickup is
   band-limited and B lives in the high partials. It meant the channel could
   run on audio the pipeline already had.
4. **Integration** as soft, abstaining evidence: the channel measures its own
   fit quality (r²) and contributes nothing below threshold. Its failure mode
   is "no evidence," never "wrong evidence." The decomposition was one-for-one
   at 52,000 events — every gained note came out of the wrong-position
   bucket, all other error buckets moved by exactly zero, onset and pitch
   bit-identical.
5. **Self-calibration failed** — fitting B from the user's own recording
   needs ~8 clean notes per string (a 30-second clip yields ~10 total) and
   bootstrapping from the decoder's own labels injects a bias comparable to
   the whole signal. This looked like the ship-blocker: was the channel
   forever chained to the lab dataset?
6. **The portability solve:** derive B from *published string
   specifications* — `B = π³E·d⁴/(256μL⁴f²)`, every term a number off a spec
   sheet, the fret law falling out of geometry rather than being assumed.
   The spec-derived table scored **+0.0502**, statistically indistinguishable
   from the dataset-fitted table's +0.0525. The training dataset was demoted
   from source to test. Nothing was fitted to anything.

Full-development gate: **+0.0443** [+0.0339, +0.0555] over 300 clips, with
config frozen before the run. Sealed-player confirmation: **+0.0780**, solo
tier **+0.1396** — a 25% relative gain on exactly the tier the project had
declared information-limited. It shipped as the default for clean
steel-string acoustic, with a domain guard proven by unit test rather than
measured by eval run: nylon, capo, and alternate tunings abstain by
construction, because a wrong table would be worse than none.

A robustness study then mapped the failure envelope across 17 pre-declared
table perturbations: string gauge and scale length are noise; the entire risk
concentrates in wound-core construction — the one spec manufacturers don't
publish — and the gain lives almost entirely on the four wound strings.

The capo work rode the same wave: routing declared-capo sessions through a
coordinate-shifted ("covariant") prior recovered an outright collapse —
**0.296 → 0.683** at capo 2. The old behavior hadn't been a shortfall; it
was the decoder breaking when every playable candidate sat above the capo.

## Act V: Humility, measured (late July)

Three episodes from the same two weeks, all of them course corrections the
process caught before I could publish something wrong.

**The headline was the luckiest player.** The +0.0780 sealed result went into
the README as ~0.73 aggregate. Measuring all six players showed the channel's
gain ranges +0.047 to +0.101 — a factor of two — and my sealed player sat at
the maximum. Nothing was mis-measured; a single held-out set protects against
overfitting, but one draw from a varying population cannot tell you an effect
size. The headline was re-based downward to 0.66, the sealed player was
rotated, and the README now explains why.

**The sealed set earned its keep.** A uniform level correction to the physics
table — supported by three independent measurements of the same error —
scored +0.0160 on development data and **−0.0066 on the held-out player**,
non-overlapping intervals. It had been built, tested, and written up as a
shipping candidate. One sealed run reverted it: the effect was physically
real but instrument-specific, and development data structurally could not see
the difference.

**And the video verdict got corrected — in both directions.** Remember the
0.285 "video is anti-informative" result? Re-running the *identical* probe
while fixing a fret-mapping bug (a unit-neck coordinate had been projected to
fret 24 instead of through the calibrated map) moved it to **0.763**. Video
was never anti-informative; it was mis-projected. And yet the end-to-end
verdict stood: against real audio predictions, the corrected video channel
moved Tab F1 by +0.000836 with a confidence interval touching zero — because
a probe showed even *gold* position windows could only add +0.00009. The
decoder's surviving candidates mostly produce the same tab; there was almost
nothing left for a coarse position window to fix. Video ships as opt-in on
evidence of *negligible effect*, not of harm. The distinction matters: the
buggy framing would have permanently closed a lever that is merely weak.

Both of the project's worst errors — "the ceiling is information-theoretic"
and "video is anti-informative" — were confident, well-written, and supported
by real measurements. What caught them both was the same discipline: **being
able to state the mechanism, not just the number.** A result whose mechanism
you can't articulate is a result you haven't finished checking.

## Act VI: The pivot that stuck (August)

A five-track parallel program priced everything that remained. The most
interesting output wasn't a shipped feature — it was a price list:

- A timbral string classifier: ceiling large (+0.19 oracle), signal present
  (AUC 0.71), and *conversion* impossible — weak per-note evidence washes out
  against a prior already at 0.65 on the same notes. Closed for the third and
  final time, this time with the mechanism.
- The detection buckets: the attractive "harmonic leakage" story about
  spurious detections turned out to be a base-rate artifact (fifths and
  fourths are 29.6% of false detections — and 37.2% of *all* intervals in
  this music; enrichment 0.8×, i.e., anti-enriched). Missed notes, by
  contrast, showed real structure: missed at 1.6× base rate inside dense
  chords and on very short notes, at *half* base rate when isolated. One
  bucket closed, one legitimate build candidate opened.
- **A personal position prior: +0.0305** [+0.0183, +0.0430] — give a player a
  prior built from their own playing and accuracy rises for all five players,
  most for the most idiosyncratic (+0.076). Players genuinely differ in where
  they play the same notes. That's a small empirical fact about guitarists,
  and it became the product direction.

Because here is where the failed video chain finally found its job. FretCam —
the live fretboard HUD built from the video era's wreckage — has a measured
profile of "position on 27% of frames, *correct on 100% of them*." That
profile is wrong for a decode-time witness (needs coverage) and exactly right
for a **labeller** (needs precision). So the camera's role inverted: instead
of arguing with the decoder in real time, it silently harvests confirmed
(pitch, string, fret) labels when its evidence pins a unique candidate.

The studio app closed the loop end-to-end: transcribe a take, fix it in the
review UI (every note confidence-graded, doubtful ones queued for
one-keystroke triage), press **Bank gold** — and the corrected take becomes
perfect training labels for your personal prior, because a human-confirmed
correction needs no alignment guesswork. The system that spent seven months
learning what it couldn't detect now learns *you*.

## The scoreboard

**Datasets:** GuitarSet (the workhorse — 360 acoustic clips, mono mic + a
hexaphonic partition that validated the physics estimator), GAPS (classical,
the cross-domain gate), EGSet12 (reproduction checks), Guitar-TECHS (the
electric measurement that scoped v1), SynthTab and PDMX (symbolic corpora —
both closed negative for priors), 640×360 video corpora (structurally
insufficient for string resolution).

**Models that shipped:** the highres-ensemble audio backend; a Viterbi
playability decoder; GuitarSet- and GAPS-trained position/sequence priors; the
spec-derived physics table; a capo-covariant prior transform; the personal
prior builder.

**Models that didn't:** the YOLO-OBB + MediaPipe video chain (opt-in only), a
learned string-resolver CNN, a fret-keypoint model (lost to its geometric
baseline by 0.089), a 414k-parameter symbolic transformer, two TabCNN
variants, MuScriptor as a second opinion, an n-gram corpus swap, and every
form of per-instrument calibration.

That ratio — roughly one success per ten attempts, each failure cheap and
each verdict written down — *is* the method. The single success was worth
more than everything else attempted, combined.

## What I'd tell you if you're building something like this

1. **Decompose errors before hypothesizing.** "Accuracy is 0.52" suggests
   nothing. "322 of 380 errors are the same pitch on the wrong string" is a
   research program.
2. **Price the oracle first.** Before building any estimator, compute what a
   *perfect* one would be worth. A perfect per-player calibration was worth
   +0.0027 — that killed a multi-week build in one afternoon. A perfect video
   window was worth +0.00009 — that ended the video era with arithmetic.
3. **Never let a claim about your features masquerade as a claim about the
   world.** "Information-theoretic ceiling" cost this project two months and
   nearly 0.10 of accuracy.
4. **Sealed data is a tool you spend, not a formality.** It reverted a
   shipping candidate that three independent measurements supported. It was
   right.
5. **Write down the mechanism of every result, positive or negative.** Both
   of my worst errors were caught the same way: a number I couldn't attach a
   mechanism to turned out to be a number I hadn't finished checking.
6. **Failed components can succeed in a different role.** The video chain was
   a bad witness and is a good labeller. Precision and coverage are different
   products.

The pipeline today transcribes clean acoustic guitar at ~0.93 onset F1 and
~0.66–0.68 tab accuracy on players it has never seen, faster than realtime on
a laptop CPU, and it knows — with a written measurement behind each item —
exactly what it cannot do. In applied ML, I've come to think that second
property is the rarer one.

---

*The repo's `docs/EVAL_REPORTS/` directory contains the reproducible run
behind every number above; `docs/DECISIONS.md` is the full decision log,
negatives included.*
