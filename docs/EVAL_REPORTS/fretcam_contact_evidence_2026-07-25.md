# FretCam: why the bridge measures flat, and why fixing it does not help

**Date:** 2026-07-25
**Status:** DIAGNOSTIC + a new **opt-in, default-off** channel. No default
changed, no promotion claim; the new channel measured **negative** and is not a
gate candidate.
**Scripts:** `fretcam/src/fretcam/contact_evidence_probe.py` (diagnostic),
`tabvision/scripts/eval/fretcam_end_to_end.py --contact-evidence` (frozen paired
end-to-end).
**Population:** GAPS clean-12 (development), all 12 clips. The source-disjoint
split was not opened.

**Headline:** the shipped position window reaches 2.6% of the notes it targets.
Replacing it with the per-finger contacts the same detection chain already
computes reaches **4.6× more notes** — and moves macro Tab F1
**−0.000362 `[−0.001246, +0.000159]`**. The channel is not starved for evidence
or coverage; it injects at the emission term, where two better-calibrated priors
outweigh it 2.5–5.5:1. **Video has to enter at the position/transition path or
not at all.**

## The question

The 2026-07-24 end-to-end run measured `--video-backend fretcam` at **+0.000836**
macro Tab F1 on ten source-disjoint clips (95% CI lower bound exactly 0) and
**−0.000155** on clean-12. That result is real but it does not say *why*. A
channel can measure flat because its evidence is worthless, or because its
evidence never arrives. Those have opposite remedies, and the e2e harness
cannot separate them.

This probe separates them on two axes:

- **coverage** — how often does evidence exist at the causal pre-onset instant
  the bridge reads?
- **strength** — when it exists, how much does it discriminate the gold
  string/fret from the one the audio decoder actually chose?

Strength is a likelihood ratio, `P(hit | gold) / P(hit | audio's wrong choice)`,
reported in nats so it is directly comparable with the one-nat cap in
`tabvision/fusion/position_window_prior.py`.

## Verdict

**Both, and the ceiling is low either way.**

The shipped evidence type reaches **2.6%** of the notes it targets — 52 notes
out of 1,987 — and on those it measures **0.25 nats**, a quarter of what the
implementation is already willing to grant. The per-finger contacts the same
detection chain computes and the adapter discards reach **33.6%** — thirteen
times the coverage — over 668 notes, at **2.13 nats**.

Un-gating the contacts is therefore a real improvement, but a small one, for
two reasons.

**The string component carries no information.** With the correct convention
applied, the string channel measures a likelihood ratio of **1.03**. `CP`'s 8.45
comes from the conjunction being *rare* — it fires on 13.9% of covered notes and
is silent on 84.4% — not from either component being accurate.

**And the upside is partly cancelled by exposure.** Contacts name the gold
position and not the decoder's on **93** audio-wrong notes; on **68** notes the
decoder already had *right*, they name a rival position and not the gold. A
prior applied to every ambiguous note is exposed to both:

- best case **+93** notes (every rescuable note flips, nothing breaks)
- worst case **+14** notes (every exposed note also breaks)

Against 10,096 matched notes that projected roughly +0.001 to +0.008 Tab F1 on
video-bearing clips. **That projection was wrong.** The channel was built and
run through the frozen paired harness, and the measured value is
**−0.000362 `[−0.001246, +0.000159]`** — see "Built and measured end-to-end"
below. Counting available evidence does not predict decisions.

**Recommendation: do not invest further in FretCam for accuracy on the strength
of this corpus** — but note what the corpus is. Every GAPS video in the cache is
**640×360**, which puts inter-string spacing at a few pixels and is very likely
why the string channel reads 1.03. The finding is therefore "video string
assignment carries no information *at 640×360*", not "video string assignment
does not work". The repository has no labelled video above that resolution, so
the general question is currently unmeasurable in-repo. See
"The scope limit on all of this" below — it is the most actionable item here.

Against that, the `acoustic-physics-v1` inharmonicity string-evidence channel is
worth **+0.05 to +0.07 aggregate** (phase-0 rotation, 2026-07-25) on every clip,
video-bearing or not.

## Results

All twelve development clips, 10,096 matched notes, 1,987 audio-wrong ambiguous
notes. Contacts read with the corrected string convention (below).

| hypothesis | coverage | P(hit \| gold) | P(hit \| audio) | LR | nats |
|---|---:|---:|---:|---:|---:|
| **W** window `{0} ∪ [N-1,N+4]` — **shipped** | **2.6%** | 0.865 | 0.673 | 1.29 | 0.25 |
| **CF** contact frets | 33.6% | 0.490 | 0.376 | 1.30 | 0.26 |
| **CP** contact `(string, fret)` | 33.6% | 0.139 | 0.016 | **8.45** | 2.13 |
| **CS** contact strings | 33.6% | 0.563 | 0.548 | **1.03** | 0.03 |

`CS` reads **1.03**: the set of strings FretCam says are under a finger is
essentially no more likely to contain the gold string than the one the decoder
wrongly chose. `CF` at 1.30 retains a little along-neck signal. `CP`'s 8.45 comes
from the conjunction being *rare*, not from either component being accurate.

**In absolute terms the shipped channel supplied usable evidence on 52 notes**
out of 1,987 audio-wrong ambiguous notes, across twelve clips and 10,096 matched
notes. Contacts supplied it on 668. Any end-to-end delta computed over a
population that small is dominated by which handful of notes happened to be
covered — the real explanation for a `+0.000836` with a CI touching zero, and
for the original run's split verdict (2 improved / 8 unchanged on held-out,
1 improved / 1 regressed on clean-12).

Per-clip, the shipped window covered **zero** notes on `043`, `063`, `104`,
`294` and `341`, and one note each on `179` and `212`.

### Rescue, harm, and exposure — the actual bound

A re-ranker helps where contacts name the gold position and not the decoder's,
hurts in the mirror case, and — the term it is easy to forget — is also applied
to the notes the decoder already got right.

| outcome | count | share of covered |
|---|---:|---:|
| gold named, audio not → **rescue** | 93 | 13.9% |
| audio named, gold not → **harm** | 11 | 1.6% |
| both named → no gain | 0 | 0.0% |
| neither named → **silent** | 564 | 84.4% |
| **net rescuable** (audio-wrong only) | **+82** | 4.1% of all audio-wrong |
| **exposure** on audio-*right* notes | **−68** | of 7,527 correct ambiguous |
| **net after exposure** | **+14** | worst case |

The exposure row counts currently-correct ambiguous notes where the contacts
name some *other* playable position for that pitch and not the gold one — the
notes a contact prior is positioned to break. At 68 against 93 rescues it does
not cancel the gain, but it consumes most of it in the worst case, and it is the
term that a naive "un-gate the contacts, it's free coverage" change would miss
entirely. Whether the realised net lands nearer +14 or +93 depends on the
decoder's confidence margins on each population, which this probe does not
model.

**Aggregating the whole lookback does not rescue the silence.** Unioning
contacts across every frame in the 150 ms window, instead of reading the single
frame the shipped policy reads, moves net rescuable only 1.8% → 2.0% on the
5-clip prefix and leaves the silent share at 86.0%. The channel is not silent
because it is sampled too sparsely in time; it is silent because the contact
set genuinely does not contain the sounding position. Four fingers cover at
most four of six strings, and roughly a fifth of notes on this corpus are open
strings the fretting hand cannot express at all.

### Coverage is lost in the estimator, not the detector

Instrumented over the first 40 s of three development clips:

| clip | neck locked | frames with contacts | **accepted by the bridge** |
|---|---:|---:|---:|
| `031_vpswc` | 98.5% | 72.8% | 20.7% |
| `118_VD1wc` | 97.0% | 58.7% | **0.3%** (1 of 334) |
| `142_GD1wc` | 56.2% | 45.3% | **2.2%** (7 of 320) |

On `118_VD1wc` — the clip that regressed in the clean-12 arm — the bridge
accepted a single frame in 40 seconds while finger contacts were available on
196 of them. Across the six scored clips the bridge produced **zero** usable
observations on four. A measured e2e delta on such clips is not a statement
about video's value; it is a statement about a channel that is switched off.

**Frame density is not the cause.** Re-running `118_VD1wc` at stride 1 instead
of the production stride 3 moved acceptance **0.3% → 0.0%**: with 1001 frames
the estimator reached only `lost` (557) and `acquiring` (444), never `locked`.
Tripling the video budget makes it strictly worse.

The cause is the lock criterion. `PositionEstimator` commits only when
`observation_confidence × temporal_agreement ≥ 0.20` *and* the same integer
position has held for `acquisition_duration_s = 0.25`. On `118_VD1wc` the hand
spans frets 7–12, so the centroid-derived integer position oscillates and
temporal agreement never accumulates — while the individual contacts stay
legible at fret 11, fret 12, and so on.

This is a design mismatch, not a bug. A HUD must abstain when uncertain: a
number flickering at a human is worse than no number. A Bayesian prior wants
the opposite — emit soft evidence continuously and let the decoder weigh it.
The bridge routes a soft-evidence consumer through a hard-decision producer
built for the HUD, and inherits the HUD's abstention.

### The string index is inverted

`FingerContact.string` is a one-based index off FretCam's canonical board axis;
`TabEvent.string_idx` is zero-based from the low E. The correct transform turns
out to be `n_strings - string` — i.e. **FretCam numbers strings the way tab
notation does, 1 = high E**, which is the same convention the GAPS parser
already normalizes with `our_string_idx = 6 - musicxml_string`. On the 2-clip
prefix:

| mapping | CP likelihood ratio | CS likelihood ratio |
|---|---:|---:|
| direct `string - 1` | 1.75 | **1.00** |
| flipped `n_strings - string` | **7.33** | 1.18 |

The direct mapping reads **exactly 1.00** — literally zero information — and
flipping it recovers the signal. This is latent, not a live defect: the bridge
never exposed the string field, so nothing shipped is wrong today. It does mean
any future consumer must use the flipped convention, and it is the same class of
error as the F2b coordinate repair that moved the F7 anchor probe from 0.247 to
0.763. The probe reports both mappings so the convention is established by
measurement rather than assumed.

Even with the correct convention, `CS` settles at **1.03** over all twelve
clips. **The string estimate is close to uninformative on this footage**, and
that — not coverage — is what caps this channel. See the resolution section
below before generalising it.

The `pressing` gate was also tested and is not an improvement: restricting to
pressing contacts drops coverage 42.2% → 36.2% and CP strength 7.33 → 4.00 on
the 2-clip prefix.

## Missed notes

The stated goal names missed notes first. The measurement closes this route.

| | count | share |
|---|---:|---:|
| gold notes with no audio detection | 759 | — |
| a video frame exists in the causal window | 759 | 100.0% |
| video names the exact `(string, fret)` | 37 | **4.9%** |
| open string — fretting hand blind by construction | 170 | 22.4% |

The reason is structural: **the fretting hand answers "what is fretted", never
"when it sounds."** Note onsets live in the picking hand.

Two consequences:

- The current bridge is architecturally incapable of touching `missed_onset`
  (18.9% of dev loss, 21.9% of sealed loss) or `extra_detection`, because it
  only re-weights priors on events audio already emitted. The phase-0 run
  verified this to the event: between the two arms only
  `wrong_position_same_pitch` moves.
- The only video route to onsets is the picking hand. MediaPipe is already
  configured with `max_num_hands = 2` and already detects it; FretCam then
  selects one hand and discards the other, and F5b explicitly fixed "a
  picking-hand boundary false lock" — the onset sensor is currently treated as
  a nuisance to be rejected. Building that channel is new capability with
  unmeasured value, not a tuning change. Even then it yields an onset without a
  pitch, so it could only act as a gate on sub-threshold audio detections.

**For missed notes, FretCam is the wrong lever.** Track D's audio-side
`missed_onset` work (masking 1.63×) addresses the same 18.9% with a sensor that
can see it.

## What none of this changes

**GuitarSet has no video.** `~/.tabvision/data/guitarset/` contains
`audio_hex-pickup_debleeded`, `audio_mono-mic`, and `annotation`. Every
published headline — sealed 0.6609, dev 0.6801 — is GuitarSet. **No FretCam
work can move those numbers.** The addressable population is video-bearing
sessions: GAPS clips and the user's own FretCam/desktop recordings. Any
promotion gate for this channel has to be stated on that population, and any
README claim scoped to it.

## Where the camera *is* worth something

The conclusion is "not as a real-time evidence channel", not "not at all".

Track C priced giving a player their own prior at **+0.0305 [+0.0183, +0.0430]**
(`docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.md`, on `main`), and observed
that the assisted-review queue already collects the corrections that would build
it and currently discards them. That is **ten times** the ceiling measured here
for the contact channel, on every clip rather than video-bearing ones only.

The camera's leverage on accuracy is therefore as a **labelling aid**, not a
sensor: a live fretboard HUD beside the review queue makes a human's
string/fret correction faster and more reliable, and those corrections are worth
+0.0305 once they are persisted instead of thrown away. That routes FretCam's
existing, working capability — a precise position display, 67/67 displayed
precision on its own dev benchmark — at the lever with the best measured return,
and it needs no fusion change, no §8 change, and no new gate on the video path.

## Built and measured end-to-end

The bracket above is an extrapolation from coverage and likelihood ratios. It
is not a Tab F1 delta, so the channel was implemented and run through the frozen
paired harness to get one.

**What shipped, opt-in only:**

- `tabvision/video/position.py` — `FingerContactObservation` and a
  `ContactAwarePositionAnalyzer` protocol, both implementation-only records
  outside the §8 contracts.
- `fretcam/src/fretcam/tabvision_adapter.py` — `analyze_all()` returns windows
  *and* contacts from a single traversal, so contacts cost no extra inference.
  Contacts are emitted whenever they exist; they do **not** inherit the position
  estimator's `locked`/`holding` gate. The string index is converted with
  `n_strings - string`.
- `tabvision/fusion/contact_prior.py` — a capped causal `(string, fret)`
  likelihood, same 150 ms pre-onset lookback as the window prior, with open and
  capoed strings unconditionally supported. `MAX_CONTACT_LOG_BONUS = 2.0` nats,
  floored from the measured ratio of 8.45 and **fixed before the run** — not
  swept against Tab F1.
- `--video-backend fretcam` gains `contact_evidence`, default **off**. `legacy`
  remains the CLI default. No default changed.

**Not done, deliberately:** the window prior still runs alongside the contact
prior when both are enabled, so the two channels partially double-count on the
2.6% of notes where both fire. Given that overlap, this was left as-is rather
than complicating the code ahead of a result.

### The result: the channel reaches 4.6× more notes and makes things slightly worse

Frozen paired harness, GAPS clean-12, production tab cache, `gaps-v1` +
`gaps-seq-v1` at weight 4.0, baseline assignment decoder — identical to the
2026-07-24 window-only run.

| arm | macro Tab F1 | Δ vs audio baseline |
|---|---:|---:|
| audio baseline | 0.772970 | — |
| window only (published 2026-07-24) | 0.772815 | −0.000155 |
| **window + contacts (this run)** | **0.772608** | **−0.000362** |

Paired 95% bootstrap interval for the macro delta: `[−0.001246, +0.000159]`.
Improved / unchanged / regressed: **1 / 10 / 1**.

**The experiment is controlled to the observation.** This run produced **687
accepted window observations — exactly the published count** — so the window arm
is bit-identical and every difference is attributable to contacts alone. Events
affected went **257 → 1,171, a 4.6× increase in reach**, for a Tab F1 movement
of −0.000362.

| clip | Δ Tab F1 | window obs | events affected |
|---|---:|---:|---:|
| `118_VD1wc` | **−0.004984** | 10 | 150 |
| `235_Ny1wc` | **+0.000636** | 166 | 284 |
| the other ten | +0.000000 | 511 | 737 |

Ten of twelve clips did not move a single note despite 737 events being
enriched. The two that moved are the exposure and rescue populations
materialising — and the exposure was six times larger.

**The sharpest statement of the result is on the target metric itself.** This
channel exists to reduce `wrong_position_same_pitch`. It went **1,930 → 1,934**:
four *more* wrong positions than the audio baseline. Micro Tab F1 −0.000277.

Full paired output: `docs/EVAL_REPORTS/fretcam_e2e_contacts_clean12_2026-07-25.md`.

### Correction: the projected bracket was wrong

The rescue/harm counts above projected **+0.001 to +0.008**. The measured value
is **−0.000362** — outside that bracket, and on the other side of zero.

The projection failed for the reason its own caveat named: it counted notes
where contacts *name* the gold position, without modelling whether the decoder
would act. It turns out the decoder overwhelmingly does not act at all, and
where it does, the notes that flip are disproportionately the exposure cases —
because those are exactly the notes where the audio prior was least certain and
therefore most movable. Counting available evidence is not the same as
predicting decisions, and the gap between them is the whole result.

### Why: 2 nats loses to two independently calibrated priors

This is the finding that matters, and it generalises past FretCam. The contact
evidence has to win twice, and it loses both times.

**First, before the decoder runs at all.** `gaps-v1` already writes a fret prior,
and the contact prior multiplies into it. Measuring that prior's own
concentration over its 39 ambiguous pitches — the log-odds gap between its top
and second candidate:

| statistic | nats |
|---|---:|
| p25 | 1.76 |
| **median** | **3.10** |
| p75 | 5.74 |

A 2-nat bonus can only reach the prior's argmax on **33.3%** of ambiguous
pitches. On the other two thirds `gaps-v1` alone outweighs the entire capped
contact channel before any transition term is considered.

**Second, at the decode.**

`playability.emission_cost` applies the fret prior — where every vision channel
in this codebase injects — at `FRET_PRIOR_WEIGHT = 1.0`, **once per note**.
Hand-position continuity is enforced in `transition_cost`, **twice per note**
(each note participates in an incoming and an outgoing transition), by three
terms:

| term | default | per transition |
|---|---:|---|
| `TRANSITION_PRIOR_WEIGHT` (the `gaps-seq-v1` sequence prior) | **4.0** | `4.0 × −log P(prev→curr)` |
| `POSITION_SHIFT_COST` | 2.5 | `2.5 × |Δfret| / 12` |
| `SAME_STRING_BONUS` | 0.5 | forfeited when the string changes |
| `HAND_SPAN_BARRIER` | 5.0 | `5.0 × max(0, |Δfret| − 5)` |

For the canonical `wrong_position_same_pitch` swap — high-E fret 5 → B fret 10,
exactly the case the bridge's own regression test targets — the decoder pays,
across both transitions:

| sequence-prior disagreement | total resistance | vs 2-nat contact gain |
|---|---:|---:|
| `Δ(−log P) = 0.25` | 5.08 nats | **2.5 : 1** |
| `Δ(−log P) = 0.50` | 7.08 nats | **3.5 : 1** |
| `Δ(−log P) = 1.00` | 11.08 nats | **5.5 : 1** |

**Compounded: the channel can reach the prior's argmax on a third of pitches,
and is then outgunned 2.5–5.5:1 at the transition.** The joint probability of
actually moving a decision is small enough to round to the measured +0.000000.

The one-nat cap was never the constraint; neither was coverage. Even evidence
calibrated at its own measured strength and delivered to 2.8× more notes is
structurally incapable of moving this decoder, because it is pushing on a term
that two better-calibrated priors already dominate.

That also retro-explains the shipped window prior's +0.000836 and the
recurrence of the "signal is there, the conversion fails" pattern elsewhere in
this program. It is not a coincidence and it is not evidence quality — it is the
force balance between a weight-1.0 emission term and a weight-4.0 transition
term applied twice.

### What this implies for using video at all

**Video must inject at the position path, not at per-note emissions.** The
quantity FretCam actually measures — where the hand is over time — is the same
quantity `TRANSITION_PRIOR_WEIGHT` models. Delivered there, vision would compete
with the sequence prior on equal footing instead of at 4:1 against; delivered as
an emission prior it is arithmetically inert no matter how good it gets.

**Scope, if anyone attempts it.** `playability.transition_cost(prev, curr, cfg,
*, use_sequence_prior, gap_s)` takes candidates and an inter-onset gap but **no
timestamp**, so it cannot currently look up a time-indexed observation. Adding a
`VISION_POSITION_WEIGHT × −log P(position | video)` term in the same weight
class as `TRANSITION_PRIOR_WEIGHT` means threading an optional onset time (or
the observation itself) through its four call sites: `viterbi`,
`segment_decoder`, `context_reranker`, and `melodic_prior`. None of that touches
the §8 contracts — `playability` is an implementation module — but it does
change a widely-used internal signature, so it is a bounded refactor rather than
a drop-in.

Two honest caveats. That redesign is a real change to the decoder's transition
model, not a plumbing change, and it would need its own gate. And on *this*
corpus it would still be reading a position signal whose string component
measures 1.03 — so it should not be attempted before the resolution question
below is settled.

## The scope limit on all of this: the corpus is 640×360

**Every GAPS video in the cache is 640×360.** Checked directly with OpenCV:
`027` 640×360, `031` 640×360, `104` 640×360, `294` 640×360, `235` 636×360. They
are YouTube-sourced.

Two consequences, and the second is the important one.

First, the analyzer's `max_frame_width=640, max_frame_height=480` downscale
**never fires on this corpus** — `scale = min(1.0, 640/640, 480/360) = 1.0`.
Frames reach the detection chain at native resolution. A control run at
`--max-width 1280 --max-height 960` produced **byte-identical trace files**
(99,672 / 98,509 / 55,284 bytes on `031` / `104` / `294`), confirming the
resize is a no-op rather than assuming it. The resolution hypothesis could not
be tested, because there is no higher resolution to test with.

Second, and this bounds every conclusion above: a guitar neck occupies a few
hundred pixels of a 640-pixel-wide frame, and six strings sit inside roughly
40 mm of that. **Inter-string spacing on this footage is on the order of a few
pixels** — comparable to MediaPipe's landmark jitter. `CS = 1.03` is therefore
a measurement of *this corpus*, not a property of video. Along-neck fret pitch
is four to five times larger, which is consistent with `CF` retaining a little
signal while `CS` retains none.

**So the honest verdict is narrower than "video string assignment does not
work."** It is:

> On 640×360 footage, video string assignment carries no usable information,
> and every FretCam-versus-audio result this repository has produced — the
> chunk 5/6 video chain, F7, F8, and this probe — was measured on 640×360
> footage.

The repository has **no labelled video corpus above 640×360**, and the standing
constraint bans private recordings from eval roles, so the question cannot
currently be settled in-repo. That is the actual blocker: not the fusion
plumbing, not the estimator gate, but the absence of a corpus at the resolution
the string axis needs. Live FretCam capture from a modern webcam is typically
720p or 1080p — two to three times the linear resolution — so **the user's own
recordings plausibly sit in a regime this corpus cannot speak to.**

If someone wants video to matter for Tab F1, the first step is therefore not
integration work and not estimator tuning. It is acquiring or synthesising a
labelled corpus at 1080p, and re-running exactly this probe on it. If `CS`
stays near 1.00 there, the channel is genuinely dead and the matter is closed
for good. If it moves, everything above should be re-measured, because the
ceiling if string assignment worked is real: `wrong_position_same_pitch` is
47.6% of dev loss.

## Caveats

- **Twelve development clips.** `027_Zpswc` and `235_Ny1wc` together supply
  roughly 38% of the audio-wrong notes. Estimates moved as clips accumulated —
  `CP` ran 7.33 → 4.00 → 3.43 → 4.90 → 8.45 across the 2-, 4-, 6-, 9- and
  12-clip prefixes, and worst-case net moved −12 → +14 between the 9- and
  12-clip reads. Treat the sign as established and the magnitude as loose.
- **The aggregate hides large per-clip variance, and per-clip counts are noise.**
  On the `031`/`104`/`294` subset every ratio inverts — W 0.50, CP 0.75 — and net
  rescuable is −1. That subset contributes 42 covered notes and a 3-vs-4
  rescue/harm split, so it is not evidence that the channel is harmful there; it
  is evidence that per-clip conclusions cannot be drawn at these counts. Only the
  aggregate is readable, and even that carries a wide interval this probe does
  not compute.
- Likelihood ratios and the rescue/harm/exposure counts measure discrimination
  and exposure available to a re-ranker, **not an end-to-end Tab F1 delta**.
  Converting them into one requires the decoder's confidence margins on each
  population, which this probe deliberately does not model. The frozen paired
  harness (`scripts/eval/fretcam_end_to_end.py`) is the instrument for that, and
  it has already been run on the shipped configuration.
- The exposure count is an upper bound on damage in the same way the rescue
  count is an upper bound on gain: neither conditions on whether the capped bonus
  is actually large enough to move the decision. They are comparable to each
  other, which is the point, but neither is a prediction on its own.
- `CP` is measured raw, with no open-string carve-out: a gold note at fret 0 can
  never match a contact, so open strings count against it on both arms. This is
  the honest form for a likelihood ratio, but any implementation must give
  open/capoed strings the same unconditional support the window prior already
  grants at `position_window_prior.py:91`. Roughly a fifth of notes here are
  open, so `CP`'s attainable hit rate is bounded near 0.8, not 1.0.
- `barre` fires on 48–60% of contacts on these clips, high for fingerstyle
  classical, suggesting the barre heuristic over-fires. Not investigated.
- No threshold, window, weight, clip, or orientation was tuned against a metric.
  Both string conventions, both contact gates, and both temporal policies are
  reported.

## Reproduction

From `tabvision/`, with the sibling FretCam package installed in the same
environment. The first run pays video inference and caches a per-frame trace
under `~/.tabvision/cache/fretcam_contact_trace`; later runs take seconds.

```powershell
$env:PYTHONPATH = ((Resolve-Path '../fretcam/src').Path + ';' + (Get-Location).Path)
.\.venv\Scripts\python -m fretcam.contact_evidence_probe --stems clean12 --flip
```

Verification: FretCam suite `240 passed, 1 skipped, 5 subtests` — unchanged from
the F8 baseline. Ruff check and format pass on the probe.

## Disposition

- Leave `--video-backend fretcam` exactly as it is: explicit opt-in, `legacy`
  default. Nothing here justifies promotion, and nothing here justifies removal.
- **Do not raise the one-nat cap.** It is the obvious move and it is wrong: the
  shipped window earns 0.25 nats while the implementation already grants up to
  1.0. The channel is over-weighted relative to its measured discrimination.
- Close the missed-note route via the fretting hand; return that bucket to
  Track D.
- **Get a labelled video corpus above 640×360**, then re-run this probe. That is
  the one action that could change any of the above, and until it happens no
  conclusion here generalises past low-resolution YouTube footage.
- **Redirect the camera at the review queue.** Persisting the corrections it
  already collects is worth +0.0305 with a CI clear of zero, against this
  channel's measured −0.000362, on every clip rather than video-bearing ones
  only.
- **Do not promote `contact_evidence`.** It ships opt-in and off. It measured
  −0.000362 on the development set and it is not a candidate for a gate.

The contact channel is now implemented, tested, and measured, so the question it
existed to answer is closed: **un-gating FretCam's finger contacts into the
emission prior does not improve Tab F1, and slightly harms it.** The
implementation stays in the tree, off by default, because it is the instrument
that would re-answer the question on a higher-resolution corpus — not because it
is a candidate for promotion.

The next FretCam accuracy attempt, if there is one, should change the
*injection point* (transition term, not emission) and the *corpus* (above
640×360). Repeating this experiment with better contacts at the same injection
point would be a predictable waste: the force balance above says the decoder
would not act on them either.
