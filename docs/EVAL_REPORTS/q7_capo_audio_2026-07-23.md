# Q7 build slice — capo support validated on audio; today's routing is worse than assumed

Accuracy-loop iteration 15 (ROI deep-dive §4.3). The entry probe validated the
capo-covariant transform at label level. This is the build slice's gate: real
Tab F1 through `fuse()`, on audio that actually sounds capoed.

20 GuitarSet clips (10 comp + 10 solo), pitch-shifted +2 and +4 semitones with
capo-shifted labels, **leave-one-player-out priors**, arms paired on identical
shifted audio so pitch-shift artifacts hit every arm equally.

## Result

Capo-0 control (unshifted, full priors): **0.6773**.

| capo | arm | Tab F1 | Δ vs today [lo-95, hi-95] |
|---:|---|---:|---|
| 2 | today (priors=none) | **0.2956** | — |
| 2 | **covariant** | **0.6827** | **+0.3870 [+0.2818, +0.4906]** |
| 2 | covariant+seq | 0.6766 | +0.3810 [+0.2615, +0.5015] |
| 2 | naive | 0.3573 | +0.0617 [+0.0219, +0.1081] |
| 4 | today (priors=none) | **0.2875** | — |
| 4 | **covariant** | **0.6533** | **+0.3658 [+0.2613, +0.4685]** |
| 4 | covariant+seq | 0.6530 | +0.3655 [+0.2488, +0.4848] |
| 4 | naive | 0.2868 | −0.0007 [−0.0116, +0.0118] |

## The finding that is *not* by construction: today's capo behaviour is a collapse

§4.3 framed this as capo sessions losing the "+22 pp prior lift". That
understates it badly. A capo session today scores **0.2956** against the
capo-0 control's **0.6773** — less than half. The mechanism is visible in the
decomposition (capo 2): `correct` 615, `wrong_position_same_pitch` **1182**.
The decoder is getting the *string* wrong on roughly two thirds of notes.

Without a prior the decoder falls back to a low-fret preference. At capo 0
that heuristic is decent, because guitarists genuinely do use open strings and
low positions. At capo `C` every candidate sits at fret ≥ `C`, the heuristic's
implicit assumption breaks, and it mispicks systematically. So the current
`priors=none` routing does not merely forgo a bonus — it puts capo sessions
into a regime the fallback was never good in.

**This half of the result is a genuine measurement**, independent of any
synthetic-capo modelling assumption: it is what a capo user gets today.

## The transform works end-to-end, and part of that is by construction

`covariant` restores **0.6827** (capo 2) and **0.6533** (capo 4) against the
capo-0 control's 0.6773 — i.e. the capo-covariant prior recovers essentially
all of the capo-0 decode quality, through the real `fuse()` on real audio.

**Stated plainly: this is partly guaranteed.** The synthetic capo relabels
capo-0 gold, and the covariant prior shifts the capo-0 prior by the same
amount, so the two are matched by construction. What the run genuinely
establishes is that the transform is *correct and complete end-to-end* — it
survives real transcription, real candidate generation, the real Viterbi and
the real metric, with no bug that the label-level probe could have hidden.
What it cannot establish is that real capo playing follows capo-0
relative-fret conventions; GuitarSet has no capo recordings, so that remains
an assumption.

The decomposition is a clean one-for-one conversion at both capos:

| capo | bucket | today | covariant | Δ |
|---:|---|---:|---:|---:|
| 2 | correct | 615 | 1382 | **+767** |
| 2 | wrong_position_same_pitch | 1182 | 415 | **−767** |
| 4 | correct | 610 | 1296 | **+686** |
| 4 | wrong_position_same_pitch | 1141 | 455 | **−686** |

`pitch_off`, `missed_onset` and `extra_detection` are identical across arms at
each capo, as they must be for a prior that only reweights string choice.

## Two secondary results

**Naive application is useless and gets worse with depth.** Applying the
capo-0 prior without the shift gives +0.0617 at capo 2 and **−0.0007 at capo
4** — by capo 4 it is worth exactly nothing. This matches the entry probe's
0.596 → 0.437 decay and confirms the shift is load-bearing, not cosmetic.

**The sequence prior contributes nothing under a capo — measured, not
assumed.** `covariant+seq` is 0.6766 vs 0.6827 at capo 2 (slightly *worse*)
and 0.6530 vs 0.6533 at capo 4. The registered `guitarset-seq-v1` uses the
`delta_fret` scheme, which conditions on the *absolute* previous-fret region;
under a capo those regions are shifted, so it backs off to the delta backbone
and adds nothing. It does not meaningfully hurt either. A capo-covariant
sequence prior would need the same fret-region shift — a separate, small piece
of work, and worth only what the seq prior contributes at capo 0.

## Methodology correction made mid-run

The first attempt used the registered `guitarset-v1` prior and reported
+0.45. That artifact was trained on players 00-04, and these clips come from
those same players, so every prior arm was scoring in-sample. Switched to
leave-one-player-out (the protocol the Q6 runs used), which brought the figure
to +0.387. The corrected number is the one reported above.

## Honest limits

- 20 clips, two capos. Directional; not the 300-clip treatment Q6 got.
- Pitch-shifting costs some transcription accuracy (pitch F1 0.9052 at capo 2,
  0.8874 at capo 4). It is applied identically to every arm, so the paired
  deltas are unaffected, but the absolute levels are mildly pessimistic.
- A capo mechanically shrinks the candidate set — positions below the capo
  become unplayable — so capo-`C` Tab F1 is not strictly comparable to capo-0.
  Within-capo comparisons are unaffected.
- Zero gold notes were dropped for exceeding `max_fret`, so that confound is
  absent at these capos.

## What this unlocks, and what it needs

The fix is a routing change: `resolve_inference_policy` currently sends any
capo>0 session to `priors=none`; it would instead apply
`capo_covariant_prior(guitarset-v1, capo)`. On this evidence that is worth
roughly **+0.37 Tab F1** to a capo user.

That is an **`auto`-path change and therefore a user decision** (SPEC §0.8).
Nothing in this iteration altered routing.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/q7_capo_audio_eval.py --stage shift
python scripts/eval/q7_capo_audio_eval.py --stage eval \
  --json ../docs/EVAL_REPORTS/q7_capo_audio_2026-07-23.json
```

The two stages must not share a process — interleaving librosa's pitch shift
with repeated highres backend loads segfaults on this machine (exit 139);
neither operation alone does.
