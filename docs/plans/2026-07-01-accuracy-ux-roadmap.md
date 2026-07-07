# Accuracy + UX roadmap (2026-07-01)

**Goal:** (1) make the tablature as accurate as possible, (2) make the app as
user-friendly as possible — grounded in the measured state of the repo as of
`9925bdf` on `v1.1/oracle-string-resolution`.

**Method:** multi-agent map of SPEC/DECISIONS, eval reports, audio+fusion code,
video-chain state, UX surfaces, and loop history; three independent roadmap
drafts (decoding-first / data-domain-first / product-first); adversarial
completeness critique. All three drafts converged on the same top items.

---

## 1. Where we actually stand

| Metric | GuitarSet (60-clip, accepted config) | GAPS test-22 | Notes |
|---|---|---|---|
| Tab F1 single-line | 0.523 (lo-95 0.457) | 0.647 (no prior!) | clean-12 honest: 0.761 |
| Tab F1 strummed | 0.676 (lo-95 0.606) | — | val24 fast loop: 0.795 |
| Onset / Pitch F1 | 0.938 / 0.930 | 0.828 / 0.819 | strummed pitch 0.9005 sits on the 0.90 gate |
| Chord-instance acc | 0.52 / 0.48 | 0.66 | v1.1 reference 0.85 |
| Latency | ~0.74× realtime | — | huge headroom vs 5-min gate |

**The error structure (val24 decomposition, 2026-06-30):** wrong-string-right-pitch
= **52.9%** of all loss (75% of single-line loss; **98.3%** on GAPS gold-audio);
missed onsets 18.6%; extra detections 14.0%; pitch 10.9%. Oracle string info
drives Tab F1 to **0.995/0.978** — the fusion resolver is already perfect when
fed good string evidence; the evidence is what's missing, and audio-only cannot
fully supply it (information-limited).

**What has worked historically (100% hit rate):** positional/playability priors
(+22–29pp), physics-based calibration, bug fixes (`e5ea355`). **What has never
worked:** threshold sweeps (wash), filter toggles with v0 constants (severe
regression), learned models (WS4 −0.117), video-as-fusion-evidence in the wild
(chunk-6 capstone: audio prior 0.778 > best video 0.574 — closed).

**The two biggest gaps aren't model gaps:**
1. **Users don't get the accepted pipeline.** `tabvision transcribe` defaults to
   `basicpitch` + `--position-prior none` (`cli.py:127-130,174-177`) while
   acceptance was measured with `highres` + `guitarset-v1` (prior alone worth
   +22–29pp Tab F1).
2. **The editor can't fix the #1 error.** Wrong-string-right-pitch is the
   dominant residual, and the web editor is fret-only — the one edit users need
   is impossible; confidence colors carry zero string-uncertainty signal
   (highres confidence is a velocity proxy, decode-inert).

---

## 2. Track A — Accuracy

Every item measures on `data/eval/local_gs_val24.toml` (baseline 0.4820 /
0.7951) with 60-clip player-05 lower-95 confirmation before any default change;
result banked in `docs/EVAL_REPORTS/` + `DECISIONS.md` either way.

### Tier 0 — hygiene (do first)
- **A0. Land the branch.** Push/PR the 7 unpushed commits (`29e19a8..9925bdf`,
  incl. the `e5ea355` threshold-wiring fix). Tree is clean, 551 tests green.

### Tier 1 — hours-class, no approval, high expected value
- **A1. Ship the accepted config as the CLI default** (`highres` +
  `guitarset-v1`, keep `basicpitch` behind a flag; document the one-time ~37s
  model download). Largest real-world accuracy delta in the repo for zero
  research. *(Studio + Modal already use the good config.)*
- **A2. Run the guitarset-v1 prior on GAPS** — never measured; every GAPS eval
  ran `--position-prior none` while wrong_position is 34.1% of real GAPS loss.
  One command via `scripts.eval.v1_1_second_corpus_probe`. Expect 0.647 →
  0.68–0.70+; a null still banks the generalization data point.
- **A3. Fusion-constants sweep + `FRET_PRIOR_WEIGHT` knob** — **DONE 2026-07-06.**
  All constants env-overridable + runtime-rebindable; in-process sweep harness
  (`scripts.eval.a3_fusion_sweep`) validated (reproduces val24 baseline exactly).
  **No default changed.** The big val24 movers (`LOW_FRET_BIAS=0.0` +0.039,
  `FRET_PRIOR_WEIGHT=1.5` +0.030, prior `power=3.0` +0.030) are all the
  *trust-the-prior-more* lever — but val24 IS GuitarSet, and that prior is −0.138
  on GAPS (A2), so they're a GuitarSet-overfit trap that must clear GAPS
  clean-12 first (expected to fail). Safest domain-neutral candidate:
  `OPEN_STRING_BONUS=0.0` (strummed 0.7951→0.8140, single-line flat); prior
  `alpha` inert. (`a3_fusion_sweep_val24_2026-07-06.md`, DECISIONS 2026-07-06.)
  **Gate CLOSED 2026-07-07 — candidate REJECTED, no default change:** the
  `OPEN_STRING_BONUS=0.0` candidate PASSES GuitarSet 60-clip (both tiers' lo-95
  up: single-line +0.0057, strummed +0.0117) but FAILS GAPS clean-12 (single-line
  lo-95 −0.0091, 11/12 clips regress). Even a hand-coded bonus is corpus-coupled —
  audio-fusion tuning is domain-sensitive across the board; defaults hold.
  (`a3_gate_open0_{gs60,gaps12}_2026-07-07.md`, DECISIONS 2026-07-07;
  gate harness `scripts/eval/a3_gate_probe.py`.)
- **A4. Time-scaled Viterbi transitions** — **DONE 2026-07-06: measured WASH.**
  `transition_cost` now takes an inter-onset `gap_s`; hand-continuity terms decay
  by `exp(-gap/TRANSITION_GAP_TAU)` (default `inf` = bit-identical no-op). Sweep:
  best TAU=1.0 at +0.0005, most values negative → keep off. Banked negative for
  the A4 hypothesis; knob retained as env-overridable evidence.
- **A10. Instrument `pitch_off` with semitone-delta histograms** — **DONE
  2026-07-06, bucket FORMALLY CLOSED:** no dominant fixable mode (octave 6/117
  ≈ 0.6% of loss, harmonic 30, semitone 20, other 61; 93% in strummed
  dense-cluster near-misses). Instrumentation is permanent in the harness —
  a future backend that re-concentrates the histogram reopens it for free.
  (`a10_val24_pitch_off_decomposition_2026-07-06.md`, DECISIONS 2026-07-06.)
- **A14. Cache-only video complementarity probe** — **DONE 2026-07-06,
  decisive negative; video CLOSED on every measured axis:** audio-wrong ∩
  video-right = 5.8%; P(video right | audio wrong) = 0.285 vs marginal 0.574
  (anti-enriched); chord axis refuted (audio 0.819 > video 0.542 on chord
  members); margin-keyed routing never beats audio-only at any threshold.
  (`a14_video_complementarity_2026-07-06.md`, DECISIONS 2026-07-06; attach
  to D1.)

### Tier 2 — days-class, no approval
- **A5. Port v0's 790-line chord-shape dictionary**
  (`tabvision-server/app/chord_shapes.py` → per-cluster shape bonus in
  `viterbi.py` state emission — the Phase 5 port that never happened). Targets
  strummed wrong_position (254 events) + chord acc (0.48 vs 0.85 reference).
  Expect strummed +0.01–0.04. Won't close the 0.85 gap by itself — say so.
  **Mechanism landed 2026-07-07** (`tabvision.fusion.chord_shapes`): 133 voicings
  (22 open + 72 barre + 39 power) ported to the `0=low E` convention +
  `chord_shape_cost` emission term wired into `viterbi.state_emission`;
  `CHORD_SHAPE_BONUS` env-overridable, **default 0.0 = bit-identical no-op**; A3
  sweep axis + 16 unit tests; ruff/mypy/fusion-tests green. By construction the
  ≥3-note match gate leaves **single-line decode invariant** — the term can only
  move strummed/chord clusters. **Measured + gated 2026-07-07:** best magnitude
  **0.1** (val24 strummed 0.7951→0.7980, single-line exactly 0.4820; ≥0.25
  negative). At 0.1 it **clears the full A3 gate on BOTH legs** — the first
  fusion constant to do so: in-domain 60-clip strummed +0.0053 / **lo-95 +0.0061**
  (single-line +0.0000), cross-domain GAPS clean-12 strict **0 regressions**.
  Domain-neutral (voicing geometry, not corpus prior) — vs A3 `OPEN_STRING_BONUS`
  (failed GAPS) / A4 (wash). Below the +0.01–0.04 hope but real. **Default flip
  0.0→0.1 is user-gated** (re-bases the val24 strummed baseline 0.7951→0.7980);
  in-tree default stays 0.0. (`a5_chord_shape_gate_2026-07-07.md`, DECISIONS
  2026-07-07.)
- **A6. Fix the GAPS gold coverage artifact** — **DONE 2026-07-06.** Repeat/volta
  unfolding in `gaps_musicxml_tab.py` (gated on a syncpoint-span length match;
  11/14 test-22 repeat clips unfolded, 3 nonstandard fall back). Controlled A/B:
  honest GAPS single-line Tab F1 **0.6468 → 0.6969** (+0.050, +1,380 gold notes)
  — a coverage-accounting correction (credits the model for repeat notes it
  always transcribed), **not** a model gain. All GAPS tuning now re-bases on
  0.6969. (`a6_gaps_unfold_{on,off}_2026-07-06.md`, DECISIONS 2026-07-06.)
- ~~**A7. Build a GAPS-native position prior**~~ — **SKIPPED** (A2 branch
  logic, 2026-07-02): guitarset-v1 on GAPS test-22 is a measured negative
  (0.6468 → 0.5087, a pure correct↔wrong-string exchange of 2,131 notes; see
  `docs/EVAL_REPORTS/v1_1_gaps_prior_guitarset_v1_2026-07-01.md`). The
  cross-domain-transfer caveat is recorded in that report's conclusion;
  reopening needs its own justification + A6 first.
- **A8. Studio-condition degradation eval.** Re-encode val24 through the real
  capture chain (opus-in-webm 48k, laptop-mic lowpass, noise floor, light
  compression — ffmpeg only, gold labels carry over, fully automated). Every
  accuracy number in the repo is on clean corpus WAVs while the product ingests
  MediaRecorder webm from laptop mics; the only end-to-end datapoint is
  anecdotal. Either result reshapes priorities. Diagnostic, not a gate.
- **A9. Extract real highres posteriors** (replace the `max(velocity,0.5)`
  proxy via the same attribute-patch style as `e5ea355`), then — and only
  then — retune the `audio/filters.py` constants for highres (mechanism proven:
  extra_detection 150→69; only the v0 constants are banked negative). Timebox:
  this intervention class is 0-for-3.
- **A11. Two-checkpoint onset ensemble** (`guitar_gaps` + `guitar_fl` voters,
  both already registered): attacks the ~32% missed+extra loss family
  orthogonally to the string wall. 2× transcription time — re-verify the
  latency gate (headroom exists at 0.74×).
- **A13. Capo/tuning mismatch preflight flag** (flag-only per SPEC rule 7;
  validate on synthetic pitch-shifted val24). Today a capo silently corrupts
  every fret number and no eval can see it.

### Tier 3 — approval-gated
- **A12. tabcnn timbral string-ID backend** — **feasibility DONE 2026-07-06:
  BECOMES A TRAINING SPEND, soft no (filed, not pursued).** No pretrained weights
  exist in the TabCNN lineage (TabCNN unlicensed; MIT successors ship no
  checkpoints); expected lift thin (TDR 0.84 on a favorable population vs our
  0.778; TabCNN tab F1 0.75 < our 0.815). `fret_prior` + A3's `FRET_PRIOR_WEIGHT`
  knob are wired if ever pursued with rule-8 sign-off.
  (`docs/2026-07-06-a12-tabcnn-feasibility.md`, DECISIONS 2026-07-06.) Original
  framing below. Second pass
  alongside highres, posterior consumed through the already-proven
  `AudioEvent.fret_prior` channel with A3's weight knob + strict no-regression
  gate. The **only** item with a path past the single-line information ceiling
  (+0.05–0.15 if real; cheap definitive negative if not). Needs: pretrained
  weights confirmed to exist (else it becomes a rule-8 spend), license check,
  user sign-off. The WS4 negative does not cover it (visual vs timbral).
- **A15. Fingering-sequence prior** *(added 2026-07-02, user-proposed;
  dataset candidates supplied by user same day)*: a sequence/convention prior
  learned from real fingerings — line passages, arpeggios, barre-vs-open
  voicings, not just chords — biasing the Viterbi string/fret decode before
  the user ever corrects. Attacks the same wrong-string bucket as A12 but from
  convention (statistics) rather than timbre (physics) — they compose.
  **Dataset map (user-supplied candidates, triaged):**
  - *GuitarSet* — already licensed + in hand (the shipped unigram prior is
    built from it). **Step 1 is free:** upgrade unigram → sequence/transition
    statistics on data we already have; tests the mechanism in-domain with
    zero acquisition. Small (360 excerpts) but n-grams are sample-efficient.
  - *DadaGP* (26,181 GP songs, 739 genres, tokenizer included) — the **right
    data type** (full per-note string/fret + techniques; voicings and lines
    emerge from the statistics, no labels needed). Request-access research
    terms → likely **experiment-only, not shippable** under the repo's
    NC-artifact policy (same treatment as GAPS-trained WS4 weights); still
    decisive as a corpus-scale signal probe + genre conditioning.
  - *PDMX* (250k+ public-domain MusicXML) — the **shippable-corpus
    candidate**: filter to parts carrying `<technical><string>/<fret>` (the
    in-repo GAPS MusicXML tab parser already extracts exactly this); use the
    `no_license_conflict` subset per the dataset's own caveat. Subset size
    unknown → needs a feasibility scan. Classical-skewed → doubles as a
    *shippable* classical-convention prior (the niche skipped-A7 could never
    ship for; would need its own approval to pursue as such).
  - *Chordonomicon* (666k chord progressions) — **not for this item**: chord
    symbols carry no fingering/voicing info, so it can't teach string
    resolution. Parked for later harmony-context features (e.g. a
    chord-conditioned position prior — speculative, second-order).
  - *Guitar-TECHS* — already in the local data root; **evaluation** asset
    (electric v2, techniques stretch), not prior-training material.
  **Staging:** (1) GuitarSet sequence-prior probe (free, no new data);
  (2) license/feasibility review of DadaGP access terms + PDMX tab-subset
  yield + Chordonomicon/Guitar-TECHS terms (read-only, NO downloads before it
  clears — §1.5 gate); (3) DadaGP n-gram probe (experiment-only) +
  PDMX extraction if the yield is real; (4) neural sequence model only on
  measured n-gram signal (Modal spend → rule-8 sign-off).
  **Status 2026-07-02 (user-directed execution, PDMX-only for stage 2;
  DadaGP deferred):** stage 1 **POSITIVE** — `guitarset-seq-v1` transition
  prior (`fusion/transition_prior.py`, decode-gated to singleton→singleton
  moves; ungated = banked strummed negative) lifts oracle single-line
  0.5542→0.6125 and **real-audio val24 single-line 0.4820→0.5140 (lo-95
  0.3761→0.4144), strummed wash** at w=4.0. GAPS standalone =
  wash-to-negative even GAPS-trained → ships tied to the guitarset-v1
  config family (GAPS accepted config untouched). PDMX license review:
  **CLEAR-WITH-CONDITIONS** (CC-BY; MXL files, not MusicRender JSON); yield
  fully **RESOLVED: 3,435** clean+MXL guitar songs, of which **734 carry a
  TAB staff (21.4%)** — ~340k tab notes ≈ 20× the guitarset-seq-v1 sample
  base; walk-validated, standard tunings
  (`docs/2026-07-02-pdmx-license-yield-review.md`,
  `scripts/acquire/pdmx_tab_scan.py`). Step-4 gates banked:
  60-clip confirm **PASS** (lo-95 0.457→0.4748), uncoupled GAPS **FAIL**
  (0.6468→0.5931) → **default-on SHIPPED as a coupled default**
  (`--sequence-prior auto`: active iff the position prior is; DECISIONS.md
  2026-07-02 step-4b). Neural step awaits rule-8 sign-off — recommendation:
  no spend yet, PDMX n-gram extraction first (corpus scale, not model
  capacity, is the bottleneck). Reports: `a15_*_2026-07-02.md` in
  EVAL_REPORTS. **Hard gates from
  the 2026-07-02 A2 negative:** priors are domain-sensitive → no-regression
  required on BOTH val24 and GAPS clean-12; key on the existing
  `--style`/`--instrument` inputs (and corpus genre metadata) rather than one
  global prior. Not covered by banked negatives (melodic prior was
  hand-coded, WS4 was visual).
- **Electric v2 fine-tune** — largest absolute gap (0.12 vs 0.90), design doc
  exists, spend-gated. Sequence the go/no-go explicitly (see D2) rather than
  leaving it unowned while the UI offers an "Electric" option.

### Expectations, honestly
Tier 1+2 audio work is worth maybe **+1–5pp** combined on current eval sets —
the single-line ceiling is real. The plan therefore treats **A1 (shipped
config), the correction UX (Track B), and eval honesty (A6, A8)** as the
highest-total-value accuracy moves, with A12 as the one bounded bet past the
ceiling.

---

## 3. Track B — UX

Journey today: install (Windows-only studio, hand-built venv) → record/upload
(rejects audio files and its own webm) → wait (fake 25% for the whole job,
video stages that never run) → view (confidence colors that can't indicate the
real error) → correct (fret-only, lost on refresh) → export (.txt only from
web; 3 of 4 renderers dead code) → learn nothing (README documents frozen v0).

### Tier 1 — hours-class
- **B1. Real progress + humane errors.** Thread a progress callback through
  `run_pipeline` (optional kwarg — verify §8 first), emit 4–5 real stages from
  `v1_adapter.process_v1_job` (today exactly two: 0.25, 0.9); filter video
  stages when `videoEnabled=false`; map common failures (silent audio, ffmpeg
  missing, codec) to short human messages instead of the verbatim traceback.
- **B2. Accept audio uploads + the recorder's own webm.**
  `UploadPanel.tsx ALLOWED_TYPES` blocks all audio and `video/webm` — the app
  rejects its own recordings. Add wav/mp3/m4a/webm client + server.
- ~~**B8. Remove vestigial video UI**~~ — **DROPPED per user decision
  (2026-07-02):** the video UI stays; the user values video playback as
  correction context (watch the take while fixing the tab). B1 already stopped
  the fake "Tracking fingers" progress stage for audio-only runs, which was
  the actively-misleading part.
- **B11a. Rewrite the root README** around the three real entry points (live
  site, `studio.ps1`, CLI) — it currently misdirects every new user to frozen
  v0. Pairs with D4/Phase 9.

### Tier 2 — days-class, the correction loop (the heart of the plan)
- **B4. String-assignment confidence from the Viterbi margin** (best vs
  next-best candidate cost, already in the trellis), blended into the existing
  `TabEvent.confidence` (no §8 change). Validate on val24: flagged notes must
  be enriched for wrong_position before shipping. Makes red mean "check the
  string" — SPEC rule 7 finally pointing at the error that exists.
- **B3. String-level correction in the editor.** Shift+Up/Down = move note to
  adjacent string, fret recomputed to preserve pitch; plus true delete +
  insert. Generalize `updateNoteFret` → `updateNotePosition`; reuse
  selection/undo machinery. With B4: "red note → Shift+Up → fixed." Target:
  wrong-string fix ≤ 2 keystrokes, full-clip cleanup < 1 min.
- **B5. Persist edits + job history.** localStorage autosave + restore banner
  first; then `PATCH /jobs/:id/result` + a recent-transcriptions list. Today
  every correction is destroyed on refresh. Prerequisite for correction
  telemetry (corrections-per-100-notes = the product's true accuracy metric).
  The prior-refit "flywheel" half stays deferred until a measurement design
  exists (critique: unmeasurable as specified; user data can never be gate
  evidence per the 2026-05-07 override).
- **B9. "Completed but garbage" banner.** Quality threshold on results (few
  notes / mostly-red → "check input level, see tips") + surface preflight
  checks in the web flow; today `tabvision check`/`diagnose` are CLI-only and
  a silent-input job succeeds into an empty tab.
- **B10. Audio-only playback UX** — waveform strip instead of the black
  `<video>` box in the flagship record→tabs flow.
- **B11b. Mobile audit (30 min)** — the production site is live; phone capture
  is plausibly the most common real input path and has never been assessed.

### Tier 3 — approval-gated
- **B6. Export API** (`GET/POST /jobs/:id/export?fmt=…`, POST accepts *edited*
  notes so corrections survive into the file). **MIDI can ship immediately**
  (mido already in the worker image); GP5 (PyGuitarPro, LGPL) + MusicXML
  (music21) need the §1.5 license review → D3. Add an automated render→reparse
  round-trip test — durations/offsets are currently unvalidated anywhere.
- **B7. BPM → beat quantization + bar-aware rendering.** The recorder knows
  BPM/beats-per-bar and discards them; v0's `beat_quantization.py` is the
  porting source; renderers hardcode 120 BPM. Explicitly Phase 9 territory →
  needs the user's "proceed" (D4). Keep quantization render-side only.

---

## 4. Decisions needed from the user (one packet)

- **D1. SPEC §1.4.1 revision** — **partially resolved 2026-07-02:** the 0.94
  single-line video-assisted reference is RETIRED (user-approved; SPEC §1.4.1
  amended, A2 attached as evidence; binding gate stays ≥0.45, no new stretch
  number until demonstrated). **A14 now attached (2026-07-06):** the
  chord-frame hypothesis is refuted (audio beats video *more* on chord
  members, not less; no routed hybrid exists) — recommendation: retire the
  0.85 chord-instance and 0.86 strummed video-assisted references like the
  0.94. **Still open (user decision):** that retirement; expressive markings
  (≥0.70 F1 stretch, GuitarSet JAMS labels exist, fully automated) in or out;
  optionally a studio-condition eval tier defined by A8's harness; the stale
  §15 open questions. Full packet:
  `docs/2026-07-06-d1-decision-packet.md`.
- **D2. Electric v2 go/no-go sequencing** — spend-gated fine-tune; decide when
  (or whether) to schedule the zero-spend feasibility step.
- **D3. Export deps license review** — music21 + PyGuitarPro into the Modal
  worker image (MIDI needs nothing).
- **D4. Phase 9 kickoff ("proceed")** — README rewrite, license CI check,
  fresh-clone install test, diagnose polish, rhythm rendering (B7), v1.0.0 tag.

---

## 5. Sequencing

- **Week 1** (all hours-class, independent, no approvals):
  A0 land branch → A1 CLI defaults, A2 GAPS prior run, B1 progress/errors,
  B2 upload types, ~~B8~~ (dropped per user 2026-07-02), A10 pitch_off
  instrumentation, A14 complementarity probe; draft the D1 packet with A2/A14
  attached. *(Status 2026-07-06: **Week 1 COMPLETE** — A0–A2, B1, B2 done
  2026-07-02; A15 ran ahead of queue and CLOSED 2026-07-05; A10 + A14 done +
  D1 packet drafted 2026-07-06.)*
- **Week 2**: A3+A4 sweep harness (the load-bearing infra — A5's bonus
  magnitude and per-tier configs ride it); B4 margin confidence → B3 string
  editor (ship as a pair); A6 GAPS gold fix.
- **Week 3**: A5 chord-shape port; B5 persistence + history; A8
  studio-condition eval. *(A7 dropped — skipped per the A2 negative,
  2026-07-02.)*
- **Week 4+**: A9 posteriors → filter retune; A11 onset ensemble; B9/B10/B11;
  then approval-gated work as decisions land: B6 export (MIDI first), A12
  tabcnn, B7 quantization (Phase 9).

**Coordination:** the unattended accuracy loop commits to the same
audio/fusion paths — keep its commit policy (verified work each round), don't
double-assign A2/A3-class measurements, and land A0 first so the loop and this
plan share the same HEAD.

**Measurement discipline:** val24 fast loop → 60-clip lower-95 confirm before
any default change; per-clip no-regression on GAPS clean-12 for anything
touching fusion; every result (positive, wash, or negative) banked in
`docs/EVAL_REPORTS/` + `DECISIONS.md`.

---

## 6. Do-not-retry appendix (banked negatives/washes)

`--audio-filters` on with v0 constants (strummed 0.795→0.460) · onset_threshold
0.2 (wash; intermediate sweep deliberately declined) · frame/offset thresholds
(structurally inert for Tab F1) · melodic prior default-on (0.474→0.449) ·
guitar_fl swap for electric (noise) · per-clip highres time calibration
(overfits) · video fusion evidence on in-the-wild GAPS under any
gate/orientation/λ (capstone) · WS2 nut-axis re-fit (0.574→0.547) · WS4 learned
neck-crop string model (−0.117; tighter-crop retry explicitly not authorized) ·
coverage gate loosened 0.71→0.5 (leaked −0.05) · UT-Austin real-audio corpus
(corpus-broken) · 720p GAPS re-acquire (weak EV + cache footgun) ·
audio-uncertainty/confidence-keyed routing of the current CV chain's string
evidence (A14: video anti-enriched where audio fails; every τ loses) ·
pitch_off bucket interventions (A10: no concentrated mode; octave ≈ 0.6% of
loss).
