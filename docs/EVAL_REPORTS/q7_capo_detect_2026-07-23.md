# Q7 capo detection — pitch-based refuted; physics route untestable on this data

Accuracy-loop iteration 16 (ROI deep-dive §4.3, piece 1). The capo-covariant
prior is worth ~+0.37 Tab F1 to a capo user, but only if something knows the
capo — today it must be passed by hand. This asks whether it can be inferred.

60 cases: 20 GuitarSet clips at capo 0, 2 and 4, ground truth known by
construction. Two estimators, no re-transcription (cached events reused).

## Result

| estimator | exact | within ±1 | mean signed error | MAE |
|---|---:|---:|---:|---:|
| pitches | **0.017** (1/60) | 0.200 | **+1.92** | 2.52 |
| inharmonicity | 0.183 | 0.250 | +0.85 | 2.85 |
| physical upper bound valid | **1.000** (60/60) | — | — | — |

## Pitch-based detection is refuted, and the negative is valid

1 case in 60. The error is not noise but a systematic over-estimate
(+1.92 semitones): with no low notes to constrain it, open-string occupancy
drifts toward high capo hypotheses.

This confirms the theoretical argument from first principles: **a capo at `C`
playing a shape produces exactly the pitches of capo 0 playing the same music
transposed up `C`.** The note sets are identical, so no amount of cleverness
applied to pitch content can separate them. Occupancy heuristics are guessing
at repertoire, not measuring the instrument.

This negative is trustworthy because pitch-shifting reproduces pitch content
faithfully — which is exactly what this estimator consumes.

## The physics estimator could not be tested — the synthetic capo is not physical

The inharmonicity estimator scored 0.183, but **that number is meaningless**
and should not be quoted as its accuracy.

A real capo shortens every string, and `B ∝ 2^(n/6)` in the absolute fret, so
a capo at 2 should raise median `log B` by **+0.231** and a capo at 4 by
**+0.462**. Measured on the pitch-shifted audio:

| clip | capo 2 shift | capo 4 shift |
|---|---:|---:|
| 00_BN1-129-Eb_solo | −0.113 | +0.111 |
| 01_Jazz2-187-F#_solo | −0.055 | +0.138 |
| 04_Jazz2-187-F#_solo | +0.070 | +0.009 |
| **physically expected** | **+0.231** | **+0.462** |

The measured shifts hover around zero and bear no relation to the prediction.
Pitch-shifting scales all frequencies uniformly; it does **not** shorten the
string. So the synthetic capo audio carries capo-0 stiffness with capo-`C`
pitches — a physically impossible instrument.

The detector predicting capo 0 for 10/20 cases at every true capo is
therefore it reading the stiffness correctly. **The test is invalid, not the
method.** Testing it needs real capo recordings, or resynthesis that actually
models string shortening; neither exists in this repo.

## Scope of the methodology gap (checked, not assumed)

Synthetic pitch-shift is **valid for pitch- and position-based evaluation**
and **invalid for timbre/physics-based evaluation**. Consequences:

- **Q7's covariant-prior result stands.** Its arms were position priors over
  pitch content, which pitch-shift models faithfully.
- **The Q6 inharmonicity channel is uncontaminated.** Its domain guard makes
  it abstain at capo>0, so it never ran on this audio.

Recorded so a future session does not repeat the experiment or, worse, trust
a physics result measured this way.

## Conclusion: report the bound, do not guess the capo

The physical upper bound held in **60/60** cases. It is sound but weak — a
bound, not an estimate; a piece that avoids low notes permits a high capo it
does not have.

So preflight should **report the bound and ask**, not auto-set. §4.3 offered
"warn or auto-set"; on this evidence only the warn branch is supportable.
Auto-setting from pitch would be wrong ~98% of the time.

## Files

- `tabvision/tabvision/preflight/capo.py` — both estimators + the bound.
- `tabvision/scripts/eval/q7_capo_detect_eval.py` — this evaluation.
- `tabvision/tests/unit/test_capo_detection.py` (5 tests).

## Separate fix in this iteration: a registered artifact that broke on checkout

`acoustic_physics_v1.json` was written with `write_text`, which emits CRLF on
Windows, and its hash was computed over those bytes. `.gitattributes` stores
the repo as LF, so a fresh checkout produced different bytes and
`load_artifact_manifest` failed with a hash mismatch — the artifact was
broken for anyone cloning the repo. Found because a new worktree could not
load it.

Fixed by writing the exact bytes git stores (`write_bytes` with LF).
Verified: 0 CRLF bytes, hash `3af0274a…` stable across a git round-trip, and
matching the convention the existing `guitarset_v1.json` already follows.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q7_capo_detect_eval.py \
  --json ../docs/EVAL_REPORTS/q7_capo_detect_2026-07-23.json
```
