# Tab accuracy deep dive — ROI-tiered program (2026-07-21)

**Purpose.** Brainstorm-to-plan survey of every credible route to higher Tab F1,
grounded in (a) the repo's measured state and closed experiments, and (b) a
sweep of the 2019–2026 literature, datasets, and tooling. Solutions are tiered
by expected ΔTab F1 per unit effort, highest ROI first. Nothing here changes
SPEC scope; anything promoted out of this doc goes through the usual
entry-gate → OOF → player-05 lower-95 discipline (SPEC §0, §9.3).

**Posture assumed:** personal non-commercial (SPEC §1.5, 2026-07-20). NC data
and weights allowed with LICENSES.md labels; private/user recordings banned
from all training/eval roles; free tools first; $25 Modal cap (unspent);
laptop-CPU inference, ≤5 min / 60 s clip.

---

## 1. Executive summary

Current shipped state (player-05, 60 clips, `highres-ensemble` +
`guitarset-v1`/`guitarset-seq-v1`): **aggregate Tab F1 0.6339** (solo 0.5503 /
comp 0.7175), onset F1 0.9491, pitch F1 0.9403
(`docs/EVAL_REPORTS/string_assignment_phase3_2026-07-15.md`). v1 gates
(≥0.45 / ≥0.60 / ≥0.55) are all passed; the question is where the next
+0.05–0.15 lives.

The error decomposition answers it. Of all Tab F1 loss events
(`docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md`):

| Bucket | Share of loss | Status |
|---|---|---|
| wrong_position_same_pitch | **57.3%** (77.5% of single-line loss) | THE target |
| missed_onset | 15.7% | partially addressable (second opinions) |
| extra_detection | 13.0% | partially addressable (merge precision) |
| pitch_off | 11.8% | closed — no dominant fixable mode (A10) |
| timing_only | 2.2% | small; onset snapping candidate |

Three internal measurements bound the opportunity:

1. **Oracle strings ⇒ 0.9726** on GAPS gold-pitch (audio-only assignment
   0.8148) — the fusion resolver is near-perfect *given* string evidence
   (`docs/EVAL_REPORTS/v1_1_gaps_video_chain_2026-06-22.md`). Assignment, not
   detection, is the ceiling-setter.
2. **Ambiguous-note top-3 recall 0.9986** but top-1 only **0.6770** — the
   correct answer is almost always in the candidate lattice; the decoder picks
   wrong ~1/3 of the time (`string_assignment_phase0_2026-07-15.md`).
3. **Context signal exists and is large**: the Phase 0 segment-signal gate
   measured ambiguous top-1 0.6770 → 0.8217 (**+0.1446**) from four-second
   joint decoding — headroom the Phase 1–5 implementations (tiny models,
   GuitarSet-only training) captured almost none of (+0.0004 … +0.0036).

The literature sweep says the same thing from the outside: pitch/onset stages
are at published SOTA (supervised GuitarSet note-onset F50 ≈ 0.90–0.92
[MT3, YourMT3+, Riley 2024] vs our 0.9491/0.9403), while the best string
assignment results come from **symbolically-pretrained contextual models**
(MIDI-to-Tab MLM, ISMIR 2024; Fretting-Transformer, ICMC 2025) that beat
hand-crafted Viterbi/A*-style decoders by ~10 pp string agreement — precisely
the gap between our decoder and our own measured context headroom.

**Top of the stack, in ROI order:**

| # | Lever | Expected ΔTab F1 (aggregate) | Effort | Cost | Risk |
|---|---|---|---|---|---|
| 1 | Finish N2 MuScriptor merge (gate PASSED 2026-07-21) | +0.01 – +0.03 | days | $0 | low |
| 2 | Symbolically-pretrained contextual string assigner (S1b done right) | +0.02 – +0.06 | 1–3 wks | $0–25 | medium |
| 3 | Second-opinion bench (Basic Pitch / YourMT3+ complementarity probes) | +0.00 – +0.02 | days each | $0 | low |
| 4 | Inharmonicity soft evidence for single-line segments | +0.01 – +0.04 | 1–2 wks | $0 | high |
| 5 | Onset refinement (snapping) + strum-cluster timing | +0.005 – +0.015 | days | $0 | low |
| 6 | Capo/tuning auto-detect → prior coverage off-domain | 0 on GuitarSet; large in real use | ~1 wk | $0 | low |
| 7 | Review-queue ranker upgrade (assisted metric, reported separately) | n/a (assisted) | days | $0 | low |

Deliberately **not** recommended (closed with strong in-repo evidence, and the
literature offers no counter-example): video fusion, direct audio→tab
end-to-end models, more count-statistic priors at scale, naive augmentation
for the GuitarSet metric, audio-filter re-enablement, more seed-level
ensembling of the same checkpoint family. §5 documents why, so future
sessions don't re-litigate.

---

## 2. Where the loss actually is (grounding)

### 2.1 Stage-by-stage ceiling accounting

Pipeline: demux → audio backend → priors → fuse() (candidates + Viterbi +
chord shapes) → render. Measured ceilings at each stage:

| Stage | Metric today | Practical ceiling | Headroom for aggregate Tab F1 |
|---|---|---|---|
| Onset detection | 0.9491 (ensemble) | ~0.97 (best published GuitarSet-supervised ≈ 0.91–0.92, different splits — not directly comparable) | ~+0.03 via merges, mostly in dense comp |
| Pitch given onset | 0.9403 | ~0.96 | small; pitch_off bucket formally closed (A10) |
| String/fret given correct note | ambiguous top-1 0.677; all-note assignment ~0.81 on gold-pitch GAPS | **0.97 (oracle)**; 0.82+ demonstrated by segment gate | **the majority of remaining loss** |
| Chord clusters (comp tier) | chord-instance acc 0.4836–0.5210 | ≥0.85 (SPEC target, unmet) | large, overlaps string/fret row |

Two structural facts to keep front-of-mind:

- Tab F1 gates on onset match: every missed/extra onset caps the achievable
  score regardless of assignment quality. ~29% of loss (missed 15.7% +
  extra 13.0%) is onset-side; the ensemble already bought +0.017 onset F1,
  and N2 shows a second-opinion model rescues 38% of what the ensemble still
  misses.
- Wrong-position errors are *pitch-preserving*: the note sounds right and
  renders wrong. They are invisible to onset/pitch metrics and to casual
  listening — exactly why the review queue ranks them for human eyes, and why
  automatic gains here are worth more to output quality than the number
  suggests.

### 2.2 What the single-line information limit does and does not mean

Repeated finding (DECISIONS 2026-06-29 capstone, WS4, A14): audio-only string
identity for isolated notes is under-determined, and every video/learned
attempt to import external string evidence lost to the audio playability
prior (0.778) on real footage. But the limit is about *per-note, context-free*
evidence. Two escape routes remain open and are the core of this report:

1. **Context**: the 4-second joint-decoding gate (+0.1446 ambiguous top-1)
   proves that *sequential* information — hand position trajectories, phrase
   conventions, voicing grammar — resolves a large fraction of per-note
   ambiguity. Our current sequence prior (`guitarset-seq-v1`) is a singleton
   Δstring|Δpitch n-gram; it captures a sliver of this.
2. **Physics**: string identity leaves a physical trace in the signal
   (inharmonicity coefficient B, plucking-position comb filtering). Untried
   in this repo; ~90–98% per-note accuracy in the literature on clean
   isolated notes with per-instrument calibration. §3.4.

---

## 3. Tier 1 — do these next (highest ROI)

### 3.1 Finish the N2 MuScriptor merge (entry gate already PASSED)

**What.** `n2_muscriptor_probe_2026-07-21.md`: P(MuScriptor right | ensemble
wrong) = **0.3818** (63/165), gate ≥0.10 → PASS, rescues concentrated in
dense Jazz2 comp — the exact material where residual missed-onset/extra
loss lives. Remaining work per the probe doc: full dev eval + merge-variant
comparison (union-with-veto vs confidence-weighted vs cluster-scoped), then
player-05 confirm.

**Why ROI-first.** The expensive part (license approval, weight download,
MIDI caching, complementarity math) is done. Runtime ~3–4× realtime keeps it
out of `auto` but comfortably inside the 5-min gate for an offline/explicit
`--audio-backend` arm; latency budget has ~4× headroom
(`v1_acceptance_2026-06-03.md`).

**Design cautions.**
- Merge must be precision-guarded: extra_detection is already 13% of loss;
  a naive union will convert rescued recall into new false positives.
  Recommend cluster-scoped adds only (accept a MuScriptor note only inside
  ensemble-detected chord clusters where the ensemble's local recall is
  low), plus a per-note confidence floor, mirroring the
  `confidence_winner` merge that already passed CI on Phase 3.
- MuScriptor emits MIDI, not string/fret — rescued notes enter fuse() as
  ordinary AudioEvents and inherit the assignment problem. Expect its gain
  in onset/pitch buckets, then multiply by ~0.68 top-1 assignment.
- Keep the NC label chain intact (MuScriptor CC-BY-NC-4.0, LICENSES.md).

**Gate.** House standard: dev OOF lo-95 > 0 vs `highres-ensemble` baseline;
GAPS clean-12 strict no-regression (merge is backend-side, so classical
routing is untouched, but run it anyway); player-05 confirm.

### 3.2 Symbolically-pretrained contextual string assigner (S1b, done at literature scale)

**What.** Replace/rescore the per-note prior + n-gram transition with a small
encoder(-decoder) transformer that reads a *window* of detected notes
(pitch, onset, duration, cluster structure) and outputs per-note string/fret
distributions **restricted to the existing candidate lattice**. Integrate as
either (a) an emission-cost term (`FRET_PRIOR`-channel, contract-safe), or
(b) an explicit decoder behind `TABVISION_ASSIGNMENT_DECODER`, exactly like
`segment-v1`/`context-v1` were wired. No §8 change.

**Recipe (from the two 2024–2025 papers that own this problem):**

1. **Pretrain** on symbolic tabs at scale — masked-string prediction
   (MIDI-to-Tab style: mask string tokens, predict from pitch+context) or
   seq2seq token translation (Fretting-Transformer style). Substrate is
   **already in hand**: the SynthTab JAMS acquisition (DadaGP-derived
   string/fret for ~26k songs; the count-prior build consumed 9.2M–34M
   position events, `s1a_synthtab_priors_2026-07-20.md`). No new data needed;
   CC-BY-NC label already in LICENSES.md. Optionally request DadaGP proper
   (richer than the rendered subset) — free, research-use, by request.
2. **Fine-tune** on GuitarSet players 00–04 symbolic (train folds only),
   plus GAPS MusicXML tab for the classical arm. MIDI-to-Tab measured the
   fine-tune step alone at **+4.0 pp** string agreement.
3. **Constrain and rescore, don't free-run**: at inference the model only
   redistributes mass over `candidate_positions()` output. This sidesteps
   the failure mode TART found when replicating Fretting-Transformer
   unconstrained (42.1% vs the original's 81.6%) and keeps pitch exactness
   guaranteed by construction — the same reason the assisted candidates are
   pitch-preserving.
4. Decode with the existing Viterbi using the model's per-note distributions
   as emissions (keeps playability barriers, capo awareness, chord clusters).

**Why believe it, given five failed learned-assignment phases?** Every prior
failure was data- or representation-starved, and the repo's own diagnostics
say so:

| Attempt | Training data | Result | Missing ingredient |
|---|---|---|---|
| `synthtab-v1` count priors | 9.3M–34M events | CI-negative | context (counts are per-pitch marginals) |
| PDMX n-gram | 71,527 transitions | −0.036 val24 | domain (piano-derived movements) |
| `context-v1` (82k params) | GuitarSet OOF only (~4 players symbolic) | +0.0036 | pretraining scale |
| Phase 5 direct 6-string net | GuitarSet OOF audio | −0.0700 | both, plus fights the info limit head-on |
| Segment-signal *gate* (oracle-ish joint decode) | — | **+0.1446 ambiguous top-1** | ← the signal is real |

MIDI-to-Tab (ISMIR 2024): 27,619-tab pretrain → 94.35% teacher-forced /
82.52% autoregressive next-string accuracy; 73.6% full-piece agreement vs
Guitar Pro 8's 62.3% on professional jazz tabs. Fretting-Transformer
(ICMC 2025, Klangio's own research): 81.6% tab accuracy on DadaGP-acoustic,
beating A* (62.6%) and GP8 (56.0%) — i.e., **the measured gap between
learned-contextual and our style of hand-tuned decoding is ~10–20 pp**, on
exactly this task. Both are half-size BERT/T5-small models — CPU-trivial at
inference, Colab-trainable, within the $25 Modal cap with room to spare.

**Code paths.** MIDI-to-Tab has no public code (verified 2026-07-21) but the
method section is complete; `open-fret` (github.com/Sidmaz666/open-fret) is a
working Fretting-Transformer reimplementation trained on DadaGP+SynthTab —
audit before trusting, but it de-risks tokenization and training loop;
`amt-tools` (MIT) has the GuitarSet symbolic plumbing.

**Honest failure modes.** (i) Fine-tune domain: GuitarSet fingerings are
YouTube-session players, not pro-transcription conventions — the PDMX lesson
("domain match beats scale") says the fine-tune step carries the risk, so
gate on val24 *and* GAPS strict no-regression with the classical arm using
GAPS fine-tune data. (ii) TART's failed replication shows the architecture
is not turnkey — hence constrained-rescoring integration, which caps downside
at "prior contributes nothing" rather than "decoder emits unplayable tabs."
(iii) Windowing at inference must respect the 80 ms cluster grouping or chord
voicings fragment.

**Expected value.** Wrong-position is 57.3% of loss ≈ 0.21 absolute F1 mass
on the 60-clip set. Recovering a quarter of it — consistent with top-1
0.677 → ~0.76 on ambiguous notes, still below the measured 0.822 segment
ceiling — is ≈ +0.05 aggregate. Range stated as +0.02–+0.06 to respect the
fine-tune-domain risk.

**Gates.** Entry: offline replay probe (like S1a) — rescore Phase 0's banked
ambiguous-note lattice, require ambiguous top-1 ≥ +0.05 over 0.6770 before
any pipeline integration (this was Phase 4's gate; the timbral ranker died
here at +0.0072 — a fair bar). Then standard OOF → player-05 lo-95.

### 3.3 Standing second-opinion bench (complementarity-gated merges)

**What.** N1/N2 built the methodology (cache second-opinion MIDI, compute
P(right | ensemble wrong), gate ≥0.10 before any merge work). Formalize it as
a bench and run the two cheap untested candidates:

- **Basic Pitch** (Apache-2.0, already a Phase 1 backend, CPU-real-time):
  architecturally maximally different from the CRNN family — the diversity
  N1 proved the kroma checkpoint lacks (bit-identical outputs, zero
  ensemble value).
- **YourMT3+** (seq2seq token decoder, GuitarSet-supervised onset F1 91.65;
  CPU demo exists on HF Spaces): different error surface again; check
  license before shipping, probe is license-free.

**Why Tier 1.** Each probe is ~a day on cached eval audio, $0, and the gate
either kills it cheaply or hands Tier-1.1's merge machinery a new member.
Expected value per member is small-positive (+0 – +0.02) but the marginal
cost after N2 lands is near-zero.

---

## 4. Tier 2 — promising, more risk or narrower scope

### 4.1 Inharmonicity + plucking-physics soft evidence (the untried audio route past the info limit)

**What.** Per-note string evidence from string physics: stiff-string partials
deviate as f_k ≈ k·f0·√(1+B·k²) with B (inharmonicity coefficient) set by
string gauge/length — B differs across strings for the *same pitch* (fretted
position changes effective length). Plucking point adds a comb-filter
signature. Estimate B per detected note from the highres/ensemble-localized
partials; convert to a per-string likelihood; feed as a bounded emission
bonus on clean, monophonic, low-reverb segments only (segment router reuses
the B4 string-margin confidence + tier detector).

**Literature anchors.** Barbancho et al. TASLP 2012 (inharmonicity-based tab
for up to 4-note polyphony); Abeßer 2013 (isolated-note string ID F ≈ 0.90
six-string); Hjerrild & Christensen ICASSP 2019 (string+fret error 1.5% on
isolated notes, few-seconds-per-string calibration, pure DSP, realtime on
CPU); Bastas et al. ICASSP 2022 (few-sample inharmonicity adaptation +
playability constraints). Nothing published survives dense polyphonic mixes
— hence single-line scope.

**Per-session calibration without user burden.** Bootstrap from the decode
itself (EM flavor): first pass assigns strings via current decoder; take
high-margin notes as provisional labels; fit per-string B(fret) curves;
re-decode with the physics term; iterate once. GuitarSet hex-debleeded stems
give free ground truth to validate the B-estimator offline before any
pipeline wiring; the mono-mic mix then measures bleed degradation honestly.

**Why only Tier 2.** High variance: partial-tracking SNR on real acoustic
mixes, capo/aged-string B drift, and the Phase 4 timbral-ranker precedent
(learned timbre features: only +0.0072). The counterargument for trying
anyway: Phase 4 learned generic timbre embeddings from data; this is a
*structured physical* feature with a per-clip calibration loop — a different
mechanism, and the only audio-side evidence source that is *causally* tied
to string identity rather than correlated with position conventions.

**Gate.** Offline first, no pipeline code: B-estimator string-classification
accuracy on GuitarSet hex (isolated-note regime) ≥0.85, then ≥0.70 on
mono-mic single-line segments. Below that, close it with a banked negative
like WS4. Cost: 1–2 weeks CPU, $0.

### 4.2 Onset refinement + strum-timing handling

**What.** (a) Post-hoc onset snapping: context-aware refinement of onset
times against local energy/spectral-flux structure ("Snapping Matters,"
arXiv 2606.11903, +2.6 note F1 on piano benchmarks in 2026); directly
converts timing_only (2.2%) and boundary cases inside missed_onset at the
50 ms gate. (b) Strum-aware matching/decoding: comp-tier strums spread
30–60 ms across strings; per-string onset jitter around a strum center is
physically expected. A strum-cluster-aware onset assignment (snap cluster
members to a shared strum envelope before the 80 ms chord grouping) targets
the dense-comp residue N2 also lives in.

**Why Tier 2 not 1.** Bounded upside (~+0.005–0.015 aggregate) since onset
F1 is already 0.9491; but cheap, CPU-only, contract-safe (pre-fuse event
surgery), and stacks with every other lever. Note A15's onset/pitch
bit-identity discipline: this one *intentionally* changes onsets, so it
must re-run the full onset/pitch gates, not just Tab F1.

### 4.3 Real-world coverage: capo/tuning auto-detection and capo-shifted priors

**What.** Today `resolve_inference_policy()` routes *any* capo>0 or
non-standard-tuning session to `priors=none` — the +22 pp prior lift
(DECISIONS 2026-05-07) silently disappears on exactly the sessions a
personal user records with a capo. Two pieces: (1) estimate tuning offset
(global cent histogram of detected f0 vs equal temperament) and capo
(minimum-fret occupancy + open-string pitch classes) in preflight, warn or
auto-set; (2) make `guitarset-v1`/seq priors capo-covariant (shift the fret
axis by capo before application) so the gated domain widens to capo 0–7.

**Why it matters despite ΔGuitarSet = 0.** The acceptance metric can't see
it (player-05 is all capo-0 standard tuning), but output accuracy on real
sessions is the project's actual goal (project instruction, 2026-07-20
posture). SPEC §11 already earmarks capo detection for v1.1. Ship behind a
flag; validate on synthetic capo shifts of GuitarSet (pitch-shift audio +2/+4
semitones with capo-shifted labels — label-exact augmentation, no NC issues).

### 4.4 Assisted-review ranker upgrade (separate metric, cheap compounding)

The review queue's error detector sits at AUC 0.7127 and 38.76%
wrong-position reduction at 60 s/clip (`string_assignment_phase6_2026-07-16.md`).
Any Tier-1.2 model emits exactly the two signals the ranker lacks: per-note
posterior entropy and top1–top2 margin from a *context-aware* model (current
margin comes from the context-free Viterbi). Re-rank the queue and re-measure
the reduction@60s curve; also use the model's posterior to reorder C-key
candidate cycling (top-3 recall is already 0.9986 — ordering is the whole
game there). Reported separately from automatic Tab F1 per the 2026-07-20
decision; it compounds with, rather than competes against, Tier 1.

---

## 5. Tier 3 — low ROI now: closed, capped, or out-of-scope (with receipts)

**5.1 Video fusion — closed, keep closed.** In-repo: geometric chain 0.574 vs
audio prior 0.778; learned WS4 −0.117; A14 anti-enrichment
P(video right | audio wrong) = 0.285 < 0.574 marginal; every hybrid router
lost (DECISIONS 2026-06-29, 2026-07-06). Literature check (2026-07): **no
published controlled Tab-F1 gain over audio-only on GuitarSet from video
exists** — TapToTab 2024, ISVC 2023, LNCS 2025 are vision-only or
non-comparable proofs-of-concept. Re-open only under a changed capture
contract (user-owned fixed neck-cam), which is a different product.

**5.2 Direct audio→tab end-to-end — capped below current system.** Best
published joint models: TabCNN note-level string-dependent F1 0.430, FretNet
0.506 (six-fold GuitarSet) vs our decomposed pipeline at 0.6126/0.6339.
Phase 5's direct 6-string net (−0.0700) agrees. The decompose-then-assign
architecture is the right one; leave `audio/tabcnn.py` a stub.

**5.3 More count-prior scale — closed.** `synthtab-v1` every arm CI-negative;
PDMX n-gram −0.036 (S1a, A15). Scale without context or domain doesn't move
this decoder. Superseded by Tier 1.2.

**5.4 Audio-backbone fine-tuning — marginal for this metric.** Pitch F1
0.9403 with pitch_off formally closed (A10: 5% octave, no dominant mode).
Published guitar fine-tune ceiling ≈ F50 0.90–0.92 (Riley 2024 87.3–89.7;
GAPS 88.1–91.2; YourMT3+ 91.65) — we already ensemble past the CRNN family's
solo numbers. A GuitarSet-supervised fine-tune of the Riley checkpoint (MIT,
HF `xavriley/midi-transcription-models`) is the *v2-electric* enabler
(clean-electric 0.12 needs it) — park it there, not on the acoustic path.

**5.5 Augmentation for robustness — wrong metric.** A8: Opus/laptop-mic/noise
degradation ≈ wash on val24 (+0.0015–0.0085); DAFx-2024 found the same
(in-domain flat, OOD +10–14 pp tab F1). Keep the recipe on file
(audiomentations MIT + MIT IR Survey CC-BY + pedalboard GPL-internal) for
the day real-room complaints appear or the electric tier opens; it buys
nothing on player-05.

**5.6 Semi-supervised pseudo-labeling — posture-capped.** Strahl & Müller
ISMIR 2024 works on piano, but the private-recordings ban shrinks the
unlabeled pool to public NC audio we'd have to license-vet clip-by-clip, for
a pitch stage that isn't the bottleneck. Revisit for v2 electric where the
backbone genuinely lacks domain data (pool: TONE3000/NAM re-amps of EGDB DI,
GOAT amp renders).

**5.7 Electric (v2) — documented path, not now.** When opened: fine-tune
Riley checkpoint on EGDB + Guitar-TECHS (CC-BY-4.0) + GOAT (CC-BY-NC-4.0,
request) with NAM/TONE3000 re-amping augmentation (GOAT's `render_amp.ipynb`
is the reference recipe); expect the Tier-1.2 assigner to transfer better
than priors did (forced-acoustic prior on Guitar-TECHS was *below uniform* —
DECISIONS 2026-07-14 — but voicing grammar is genre-, not mic-chain-,
dependent).

**5.8 Micro-knob re-sweeps — closed.** onset/frame thresholds (wash / inert),
A4 τ-transitions (wash), OPEN_STRING_BONUS=0 (GAPS-negative), melodic prior
(single-line regression), prior-trust movers (val24-overfit trap — val24 IS
GuitarSet). The A3/A5 program mined this vein to depletion; only the
cross-domain-gated chord-shape bonus survived. Do not re-run without a new
evidence channel in the costs.

---

## 6. Cross-cutting: measurement hygiene for the next program

1. **Guard the val24 trap.** Both remaining big levers (1.2, 4.1) are trained
   or calibrated on GuitarSet-adjacent data; the prior-trust incident showed
   val24 gains can be pure in-distribution leakage. Keep: strict GAPS
   clean-12 per-clip no-regression for anything touching fuse(); player-05
   only after config freeze; lo-95 acceptance at N=10,000/seed=42.
2. **Add one honest OOD-acoustic slice.** We have no acoustic eval outside
   GuitarSet+GAPS. IDMT-SMT-Guitar subset 4 (64 short pieces, string+fret
   XML, CC-BY-NC-ND — eval-only use, label it) or a hand-built 10-clip
   AnimeTAB-rendered set would catch convention-overfit from Tier 1.2
   cheaply. ND forbids redistribution, not internal eval.
3. **Track the decomposition, not just F1.** Every accepted change should
   ship its six-bucket delta (`eval/error_decomposition.py`) so we can see
   *which* loss it bought — Tier 1.1 should move missed_onset/extra,
   Tier 1.2 wrong_position, 4.2 timing_only. A gain in the wrong bucket is
   a red flag for leakage.
4. **Ambiguous top-1 is the program KPI** for assignment work (0.6770 →
   0.8217 measured ceiling). It moves faster than Tab F1, isolates the
   decoder from backend noise, and Phase 0 banked the lattice artifacts to
   compute it offline.

---

## 7. Sequencing proposal (phase-style)

| Order | Item | Gate to enter | Gate to ship | Est. wall time |
|---|---|---|---|---|
| 1 | N2 merge variants → full dev → player-05 | done (0.3818) | lo-95 > 0 agg., GAPS no-reg | 2–4 days |
| 2 | S1b-v2 offline probe (rescore banked lattice) | SynthTab JAMS symbolic extract | ambiguous top-1 ≥ +0.05 | 3–5 days |
| 3 | S1b-v2 fine-tune + integration behind explicit decoder flag | probe pass | OOF lo-95 > 0; GAPS no-reg; player-05 | 1–2 wks |
| 4 | Basic Pitch + YourMT3+ complementarity probes | cached eval audio | P(right\|wrong) ≥ 0.10 | 1 day each |
| 5 | Onset snapping prototype | — | full onset/pitch/Tab gates | 2–3 days |
| 6 | Inharmonicity offline study (hex → mono-mic) | — | ≥0.85 hex / ≥0.70 mono | 1–2 wks |
| 7 | Capo/tuning preflight + capo-covariant priors | — | synthetic-capo eval + no-reg | ~1 wk |
| 8 | Review-ranker upgrade from S1b posteriors | S1b shipped | reduction@60s > 38.76% | 2–3 days |

Items 1, 4, 5 are independent of 2–3 and can interleave. Everything is $0
except possible Colab-Pro/Modal time in item 3 (≤ the standing $25 cap).
Stop-and-ask triggers per SPEC §0.8: DadaGP access request (new dependency),
any paid training, any SPEC-target change.

---

## 8. Resource appendix (verified 2026-07-21)

### 8.1 Methods most relevant to Tier 1–2

| Work | Venue | Key number | Code/weights | License |
|---|---|---|---|---|
| MIDI-to-Tab (Edwards, Riley, Sarmento, Dixon) | ISMIR 2024, arXiv:2408.05024 | 82.5% autoregressive string acc; beats GP8 by ~11 pp | none public | paper CC-BY |
| Fretting-Transformer (Hamberger, Murgul/Klangio) | ICMC 2025, arXiv:2506.14223 | 81.6% tab acc DadaGP-acoustic; A* 62.6% | open-fret reimpl: github.com/Sidmaz666/open-fret | CC-BY paper |
| TART pipeline (incl. failed FT replication 42.1%) | arXiv:2510.02597 | F50 0.838 GuitarSet after fine-tune | none | CC-BY |
| Hjerrild & Christensen string/fret physics | ICASSP 2019 | 1.5% isolated-note error, realtime CPU | — | — |
| Bastas et al. few-sample inharmonicity + playability | ICASSP 2022 | few-sample per-instrument adaptation | — | — |
| Snapping Matters (onset refinement) | arXiv:2606.11903 | +2.6 note F1 (piano) | — | — |
| SynthTab pretraining | ICASSP 2024, arXiv:2309.09085 | zero-shot F50 70.2 GuitarSet | github.com/yongyizang/SynthTab | CC-BY-NC-4.0 |
| Riley domain-adapted guitar CRNN | ICASSP 2024, arXiv:2402.15258 | zero-shot F50 87.3 GuitarSet | HF xavriley/midi-transcription-models | MIT weights |
| GAPS dataset + model | ISMIR 2024, arXiv:2408.08653 | zero-shot F50 88.1 | HF xavriley/GAPS | CC-BY-NC-SA |
| YourMT3+ | MLSP 2024, arXiv:2407.04822 | GuitarSet onset F1 91.65 | github.com/mimbres/YourMT3 (+CPU HF Space) | check repo |
| MuScriptor | arXiv:2607.08168 | large OOD gains; N2 probe passed in-repo | HF MuScriptor/* (gated per size) | CC-BY-NC-4.0 |
| Basic Pitch | ICASSP 2022 | GuitarSet F50 66.1 zero-shot | github.com/spotify/basic-pitch | Apache-2.0 |
| Semi-supervised piano (Strahl & Müller) | ISMIR 2024 | beats supervised on 3 piano sets | github.com/groupmm/onsets_frames_semisup | — |
| DAFx-2024 tone-robustness augmentation | DAFx 2024, arXiv:2405.14679 | OOD tab F1 0.447→0.585; in-domain flat | robust-guitar-tabs.github.io | CC-BY |

### 8.2 Data (beyond what LICENSES.md already tracks)

| Resource | Contents | String/fret GT | License | Role |
|---|---|---|---|---|
| DadaGP (by request, Sarmento/Dadabots) | 26,181 GP songs, tokenizer | yes (symbolic) | research-use, request | Tier-1.2 pretrain (superset of SynthTab symbolic) |
| IDMT-SMT-Guitar | 4 subsets, ~4.7k events + 64 pieces | yes (XML) | CC-BY-NC-ND 4.0 | OOD eval slice (§6.2) |
| EGSet12 | 12 real electric perf., 380 s | yes (JAMS) | CC-BY 4.0 | v2 OOD eval + permissive TabCNN weights |
| GOAT (ISMIR 2025) | 5.9 h DI + 29.5 h amp-rendered, GP tabs | yes | CC-BY-NC-4.0, request | v2 electric train |
| Guitar-TECHS | electric, multi-perspective, techniques | per-string MIDI (≤100 ms align caveat) | CC-BY 4.0 | already eval; v2 train |
| AnimeTAB | 412 fingerstyle arrangements, MusicXML | yes (symbolic) | CC (repo) | OOD symbolic / render-eval |
| SCORE-SET (2025) | GP5 re-fretted from MAESTRO MIDI | synthetic symbolic | arXiv:2507.18723 | low priority (PDMX lesson) |
| MIT IR Survey / OpenAIR / TONE3000-NAM | IRs, amp captures | — | CC-BY / per-capture | robustness & v2 augmentation |
| amt-tools (Cwitkowitz) | GuitarSet wrappers + tablature evaluator | — | MIT | cross-check our tab_f1 scorer |

### 8.3 Claims deliberately hedged

- MuScriptor GuitarSet-specific published numbers: not verifiable from the
  paper surface as of 2026-07-21; our own N2 probe is the authoritative
  local evidence.
- open-fret code quality: existence verified, quality unaudited — treat as
  reference, not dependency.
- "Snapping Matters" and 2606-series 2026 arXiv results: recent preprints,
  piano-domain; treat the +2.6 as directional, not transferable.
- GAPS licensing: Zenodo says CC-BY-NC-SA, the HF audio mirror is tagged
  MIT — repo already treats it as NC (keep doing that).

---

## 9. Bottom line

The pitch/onset stack is at or past published SOTA and formally
closed-bucketed; the decoder leaves a measured **0.145 of ambiguous top-1**
(and ~0.21 absolute F1 mass) on the table against its own context ceiling.
The one lever with both strong external evidence (two independent 2024–2025
papers, ~10–20 pp over hand-tuned decoding) and strong internal evidence
(segment gate +0.1446; top-3 recall 0.9986) is the symbolically-pretrained
contextual assigner, constrained to the existing candidate lattice. Finish
the already-gated MuScriptor merge first (days, cheapest confirmed win),
then run the S1b-v2 probe before committing to training. Everything else in
Tier 2 stacks independently; everything in Tier 3 stays closed unless its
entry conditions change.
