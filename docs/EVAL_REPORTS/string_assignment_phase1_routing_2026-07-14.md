# String assignment Phase 1: domain-routing guardrails

Date: 2026-07-14

## Guitar-TECHS gold-pitch candidate accuracy

This is a string-axis isolation check on the public CC-BY-4.0 Guitar-TECHS corpus. It supplies each note's gold MIDI pitch to the checked-in `guitarset-v1` position table and scores its ranked playable candidates. No Guitar-TECHS audio or annotations train the artifact.

- Pairable non-technique clips: **94**
- Gold notes: **9654**
- Ambiguous playable notes: **9653**
- Forced `guitarset-v1` top-1 candidate accuracy: **0.2027**
- Forced `guitarset-v1` top-3 candidate accuracy: **0.5409**
- Uniform-candidate expected top-1: **0.2173**
- Candidate-count distribution: 2 candidates: 29, 3 candidates: 564, 4 candidates: 2164, 5 candidates: 6153, 6 candidates: 743
- Annotation aggregate SHA-256: `3f53b76dc456d792f6b1b75e486eedf51bab1fa008c71fe2743e31a090ce5f6c`

| Guitar-TECHS group | ambiguous notes | top-1 | top-3 |
|---|---:|---:|---:|
| P1_chords | 2716 | 0.2305 | 0.5560 |
| P1_scales | 1507 | 0.1858 | 0.6317 |
| P1_singlenotes | 63 | 0.2063 | 0.6349 |
| P2_chords | 2702 | 0.2135 | 0.4996 |
| P2_scales | 1408 | 0.1960 | 0.6044 |
| P2_singlenotes | 59 | 0.2203 | 0.6271 |
| P3_music | 1198 | 0.1436 | 0.4015 |

The forced acoustic prior is reported for diagnosis only. Electric auto routing remains neutral because this artifact was trained on GuitarSet acoustic behavior and has no electric promotion gate.

## Automatic routing assertions

| declared session | position | sequence | string evidence |
|---|---|---|---|
| clean acoustic | `guitarset-v1` | `guitarset-seq-v1` | `none` |
| classical / GAPS | `none` | `none` | `none` |
| clean electric / Guitar-TECHS | `none` | `none` | `none` |

The known GAPS cross-domain regression remains banked in `v1_1_gaps_prior_guitarset_v1_2026-07-01.md` (-0.138 Tab F1). The classical route now resolves both learned GuitarSet priors to `none`, so the harmful condition is unreachable in `auto` mode.

## Gate decision

**PASS.** Acoustic routing reproduces the accepted global pair, classical and electric routing are neutral, Guitar-TECHS gold-pitch candidate accuracy is reported before accepting the electric routing change, and rejected split artifacts remain unregistered.
