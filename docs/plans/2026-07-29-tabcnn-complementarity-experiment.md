# TabCNN Complementarity Experiment

**Status:** protocol frozen on 2026-07-29, before any TabCNN-assisted score
was computed. Implementation and artifacts remain evaluation-only.

## Decision this experiment answers

Can a framewise TabCNN contribute string/fret evidence when TabVision's
current decoder has the right onset and pitch but chooses the wrong playable
position?

This is a deliberately narrow second-opinion experiment:

- TabVision's banked highres events remain the sole onset/pitch source.
- TabCNN may populate only the existing `AudioEvent.fret_prior` channel.
- It may not add, remove, retime, repitch, split, or merge an event.
- No model, dependency, artifact, inference route, or default is registered or
  shipped by this work.
- A negative result completes the experiment. It does not authorize tuning on
  a failed confirmation or transfer corpus.

## Frozen model families

| Family | Frozen artifact | Identity | Role and license posture |
|---|---|---|---|
| SynthTab TabCNN x4 | `SynthTab-Pretrained.pt` from `yongyizang/SynthTab` | source commit `6136f79d04d8627f1fec57d31cd5667db9854bbc`; 52,573,995 bytes; SHA-256 `a5a0812844edd1dd9540170d2bcadb543b83de2066bd18b18ac13d666d511318` | Synthetic-only pre-trained checkpoint. The dataset is described as CC BY-NC 4.0 while the repository has a CC BY license; treat the weight license as non-commercial/ambiguous and internal-evaluation-only unless separately cleared. |
| GuitarProFX TabCNN | DAFx-24 `best_TabCNN_tablature_trancription_model`, executed through the hash-pinned `tabcnn-gpfx.onnx` export | official source commit `f50309ad06dc734ddae5e3a0eda756fca221e2e7`; official checkpoint MD5 `ce168b2cd426f81a2a78499214e40605`; ONNX SHA-256 `8d9ce59157bdab37fb4816d32d7f29f3da0cdbf3c7876707c819af4d1f88e6b7` | Zenodo record 11406378 is CC BY 4.0 and the source repository is CC0. GuitarSet was used for validation/model development, so its GuitarSet result is overlap-labelled and cannot be the promotion gate. |

The hash-pinned CQT reference artifact is `tabcnn-cqt.bin`, SHA-256
`4e5dfa1f10f76545a30cbfd3224431503dbad943b1def78624632284e6df597a`.
It is retained to validate and document the shared geometry; the experiment
executes `librosa.cqt` and records the exact librosa version. The binary is
not loaded for inference.
The ONNX export is accepted only after numerical contract tests against the
official class order; it is a transport format, not a third family.

Model files live below `$TABVISION_DATA_ROOT/models/` and never enter Git.
The experiment records source URL, source revision, byte count, digest,
front-end identity, runtime versions, and load path in its manifest.

## Frozen front end and posterior mapping

Both families execute the authors' 22,050 Hz CQT geometry through librosa:

- hop 512; `fmin=C1`; 192 bins; 24 bins per octave;
- nine-frame context, with four zero-padded frames on each edge;
- deterministic CPU inference in ordered, bounded batches;
- nearest frame to each highres onset, with no temporal averaging.

SynthTab uses peak-normalized waveform input and the official
`amplitude_to_db(ref=max) / 80 + 1` feature post-processing. GuitarProFX uses
the export's documented per-clip dB-to-[0,1] front end. Native class layouts
are verified and mapped to one internal order: six strings, frets 0–19, plus
silence.

For a highres event with MIDI pitch `p`, only standard-tuning playable
`(string, fret)` candidates for `p` are inspected. The per-string value is
the conditional active probability
`P(fret) / max(1 - P(silence), epsilon)`. Supported candidates are converted
to likelihood ratios relative to their supported-candidate mean. A candidate
above fret 19 receives likelihood 1.0, so unsupported high frets are neutral
rather than silently suppressed. If there are fewer than two supported
candidates, no finite nearest frame, or no playable candidates, the model
abstains for that event.

The returned prior has the immutable `(n_strings, max_fret + 1)` shape and is
zero outside playable positions. Tests must prove that all other
`AudioEvent` fields are byte-for-byte/equality unchanged.

## Frozen arms

Each family is evaluated independently in three descriptive arms:

1. `current`: the decoder that currently routes for that corpus.
2. `posterior_only`: TabCNN position evidence without current position,
   sequence, or physics evidence; this diagnoses the model but cannot promote.
3. `current_plus_tabcnn`: the current evidence product multiplied by TabCNN
   likelihood with frozen exponent **0.35**.

There is no posterior-confidence threshold and no post-result alpha sweep.
Only structural abstention described above is permitted. Evidence composition
uses the existing weighted product-of-experts helper.

Current routing is also frozen:

- GuitarSet: leave-one-player-out `guitarset-v1`-equivalent position and
  sequence priors plus shipped `acoustic-physics-v1`, using the current
  player rotation.
- GAPS: registered `gaps-v1` plus `gaps-seq-v1`; no steel-string physics.
- EGSet12 and Guitar-TECHS: production-safe electric routing—no corpus prior,
  no sequence prior, no acoustic physics, baseline assignment decoder.

## Data order, isolation, and overlap

Evaluation proceeds once, in this order:

| Stage | Fixed material | Use |
|---|---|---|
| GuitarSet development | players 00, 01, 02, 03, 05; 300 clips; solo and comp | implementation smoke, complementarity diagnosis, and only pre-confirmation debugging |
| GuitarSet sealed | player 04; 60 clips | one confirmation after tests/config/manifest pass |
| GAPS | the previously banked clean-12 subset | primary classical cross-domain confirmation |
| EGSet12 | the 12 official WAV/JAMS pairs from Zenodo 11406378 | electric reproduction/transfer |
| Guitar-TECHS | the fixed 82 WAV/JAMS `guitar-techs` slice in `ryangowe/guitar-chord-mix`, repository commit `4448053ced18e67a9f66bfab47ac2de3cc0b4521` | independent electric technique/chord transfer; mirror provenance is explicit |

Every GuitarSet clip uses priors trained on the other five players. Player 04
remains sealed until the development output, source manifest, unit tests, and
frozen constants are complete. A cache is accepted only if all expected clips
are present and the player-05 fold reproduces the published current baseline
within ±0.0015 Tab F1; otherwise the run stops.

Here, sealing applies specifically to **candidate-assisted player-04
inference and scoring**. The all-six-player LOPO rule above necessarily uses
player-04 annotations when constructing priors for each development player;
that dependency predates this experiment and player 04's current-only
baseline was already published by the frozen Phase-0 rotation. Before the
development unlock, code may read player-04 annotations only inside the
deterministic LOPO-prior builder. It may not bank or infer player-04 audio,
cache a TabCNN posterior for player 04, pair player-04 gold with predictions,
or compute/expose any player-04 candidate-assisted metric. The manifest must
record this `LOPO_training_only` annotation dependency and its hashes.

GuitarSet is development-overlapped for GuitarProFX and is therefore
descriptive for that family. EGSet12 is a published GuitarProFX benchmark, so
it is labelled reproduction rather than independent confirmation.
Guitar-TECHS is the independent electric transfer check. No corpus is used to
fit weights or abstention thresholds.

## Frozen measurements

All summaries retain per-clip rows and report macro (mean per clip) and micro
event totals, aggregate and by source/tier/player where applicable:

- onset F1, pitch F1, and Tab F1 at the repository's 50 ms matching rule;
- paired `current_plus_tabcnn - current` Tab-F1 delta with 10,000 paired
  bootstrap samples, seed 42, and percentile 95% interval;
- exact wrong-position count and relative reduction, plus every other error
  bucket so improvement cannot hide a pitch/timing trade;
- the 2×2 current-correct/TabCNN-correct table,
  `P(TabCNN correct)`, and
  `P(TabCNN correct | current wrong-position)`;
- model coverage, structural abstention reasons, and an oracle ceiling that
  chooses TabCNN only when it would correct the current position;
- CPU wall time split into CQT, model, mapping, and decode; peak RSS; artifact
  sizes; 60-second equivalent latency; and repeat-run determinism.

Onset and pitch predictions must be exactly invariant between `current` and
`current_plus_tabcnn`; this is an assertion, not merely a reported metric.

The frozen current-path latency baseline is **262.495 CPU seconds per 60 s**:
the cold registered two-checkpoint backend measurement of 258.045 s from
`docs/EVAL_REPORTS/string_assignment_phase7_2026-07-16.md`, plus the
worst measured dense-clip partial-aware physics cost of 4.45 s/60 s from
`docs/EVAL_REPORTS/n1_partial_aware_isolation_2026-07-23.md`. This conservative
cold-start value was fixed before scoring. Candidate total latency is this
baseline plus measured CQT, model, mapping, and incremental decode time; the
added/current ratio uses the same baseline.

## Promotion gate

A family is **evidence-positive** only if all of these predeclared checks pass:

1. On the eligible acoustic/classical confirmation pool, macro Tab F1 improves
   by at least **+0.020** and the paired 95% lower bound is greater than zero.
2. Single-line/solo macro Tab F1 improves by at least **+0.030**, and
   wrong-position errors on that same solo population fall by at least **10%**.
3. Comp/strummed Tab F1 is non-inferior: delta at least **-0.005**.
4. Neither cross-domain electric set regresses by more than **-0.005** in
   aggregate; no reported tier/player with at least ten clips regresses by
   more than **-0.020**; unsupported positions activate no evidence.
5. Onset/pitch invariance, determinism, provenance, and license checks pass.
6. Total 60-second CPU latency remains below the SPEC's five-minute limit.
   Added CPU time above 20% of the current path blocks automatic promotion and
   requires a separate performance decision.

For GuitarProFX, GuitarSet cannot satisfy check 1 because of development
overlap; its eligible pool begins with GAPS. EGSet12 can reproduce the
published family behavior but cannot replace the independent Guitar-TECHS
check.

Passing these gates still does **not** integrate a model. It authorizes a new
checkpoint covering dependency choice, artifact redistribution/license
clearance, production packaging, routing, and a fresh sealed acceptance run.
Any failed gate means `do_not_integrate` for this experiment.

## Reproducibility and evidence packet

The runner is resumable and separates posterior caching from scoring. Cached
posteriors are content-addressed by audio digest, model digest, front-end
configuration, and code revision. Partial or mismatched entries are rejected.

Completion requires:

- unit tests for front-end shape/order, class remapping, high-fret neutrality,
  immutable-event preservation, metrics, bootstrap determinism, cache
  invalidation, and gate logic;
- JSON manifest and per-clip CSV/JSON suitable for recomputation;
- a dated report under `docs/EVAL_REPORTS/` containing the full result,
  limitations, overlap labels, and the explicit `evidence_positive` or
  `do_not_integrate` decision;
- dependency/license notes in `LICENSES.md` and a dated decision entry in
  `docs/DECISIONS.md`;
- the normal repository unit/lint/type checks for touched code.

No first scored run may begin until this protocol, constants, model hashes,
corpus manifests, and tests are committed in the working tree.
