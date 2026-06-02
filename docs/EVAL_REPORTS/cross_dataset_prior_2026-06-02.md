# Cross-dataset prior generalization (#2) — 2026-06-02

**Question:** the `guitarset-v1` position prior gave **+22 pp** Tab F1 on GuitarSet.
Is that a real prior over guitar physics, or memorization of GuitarSet's
distribution? Test it on a different corpus + instrument (Guitar-TECHS, electric)
that the GuitarSet-trained prior has never seen.

**Setup:** highres audio backend, CPU, laptop (i7-1185G7). Prior ON
(`guitarset-v1`) vs OFF (`none`), audio-only. GuitarSet = player-05 validation
(60 clips). Guitar-TECHS = 58 clean-electric clips (P1+P2 chords + 2 all-note
recordings; direct-input audio). Acceptance gate is `lower_95_CI ≥ target`.

## Results

| Corpus (domain) | Tier | Onset F1 | Pitch F1 | Tab F1 OFF | Tab F1 ON | Prior lift |
|---|---|---:|---:|---:|---:|---:|
| GuitarSet (acoustic, **in-domain**) | single-line | 0.94 | 0.93 | 0.219 | 0.508 | **+28.9 pp** |
| GuitarSet (acoustic, **in-domain**) | strummed | 0.92 | 0.90 | 0.475 | 0.671 | **+19.6 pp** |
| Guitar-TECHS (electric, **out-of-domain**) | clean-electric | **0.75** | **0.73** | 0.110 | 0.124 | **+1.3 pp** |

Bootstrap 95% CIs (clips): GT prior-ON Tab F1 lower-95 = 0.110; prior-OFF
lower-95 = 0.094. The +1.3 pp electric lift is **within CI noise** — not
significant.

## Verdict

**Two findings, one confounding the other:**

1. **The position prior does not measurably generalize to electric.** Its lift
   collapses from ~+22 pp (acoustic) to **+1.3 pp** (electric, within noise). On
   the runbook's decision table this is the "lift shrinks / partly
   GuitarSet-specific" branch — *not* a clean regression, but no useful transfer.

2. **The dominant, clean finding is upstream: the highres audio backbone does not
   generalize to electric guitar.** Onset/Pitch F1 drop from ~0.92/0.93 (acoustic)
   to **0.75/0.73** (electric). Tab F1 is bounded by pitch F1, so it is capped
   near ~0.12 *regardless of the prior* — the prior has almost nothing correct to
   re-assign. We therefore **cannot cleanly separate** "the prior is
   acoustic-specific" from "the prior has nothing to work with on poorly
   transcribed electric audio." The transcription gap is the real bottleneck.

## Implications

- The committed **SPEC §1.4 clean-electric (0.90) and distorted-electric (0.82)
  targets are far out of reach** with the current acoustic-trained (GAPS)
  backbone — measured clean-electric Tab F1 is **0.12**. The blocker is the audio
  backbone's lack of electric coverage, not fusion or the prior.
- **#3 as planned (GuitarSet-only fine-tune for solo acoustic) will not help the
  electric tiers** and may worsen cross-domain transfer. Before chasing electric,
  the project needs an electric-capable audio backbone — e.g. the
  `hf_midi_transcription` **`guitar_fl`** checkpoint (electric/jazz, flagged in
  AUDIT.md as a complementary backbone), or a highres fine-tune on
  Guitar-TECHS/EGDB electric audio.
- The prior stays justified for **acoustic** (in-domain +22 pp).

## Caveats

- GT eval = 58 clips (chord-dominant; 2 pedagogical "all single notes"
  recordings); no P3 / scales / excerpts / EGDB (download incomplete — resumable
  acquirer landed; re-run `acquire guitar-techs` to complete). Single electric
  corpus.
- GT clips are long continuous recordings (harder onset alignment than GuitarSet
  excerpts), which may depress onset F1 somewhat independent of timbre.

## Reproduce

```bash
# data already local; ffmpeg on PATH; venv at tabvision/.venv
python -m scripts.eval.composite_eval --manifest data/eval/local_guitarset.toml \
  --backend highres --position-prior guitarset-v1 --output docs/EVAL_REPORTS/local_guitarset_prior.md
python -m scripts.eval.composite_eval --manifest data/eval/local_guitarset.toml \
  --backend highres --position-prior none --output docs/EVAL_REPORTS/local_guitarset_noprior.md
python -m scripts.eval.composite_eval --manifest data/eval/local_guitar_techs.toml \
  --backend highres --position-prior guitarset-v1 --splits train --output docs/EVAL_REPORTS/local_guitartechs_prior.md
python -m scripts.eval.composite_eval --manifest data/eval/local_guitar_techs.toml \
  --backend highres --position-prior none --splits train --output docs/EVAL_REPORTS/local_guitartechs_noprior.md
```

## Update — off-the-shelf backbone swap (`guitar_fl`) does NOT help (2026-06-02)

Tested whether the package's other guitar checkpoint, `guitar-fl.pth` (Francois
Leduc / "broader timbre"), closes the electric gap with zero training. Paired,
same 12 Guitar-TECHS chord clips, prior-OFF:

| Backbone | Onset F1 | Pitch F1 | Tab F1 |
|---|---:|---:|---:|
| `guitar_gaps` (current) | 0.732 | 0.679 | 0.074 |
| `guitar_fl` | 0.715 | 0.687 | 0.078 |

`guitar_fl` ≈ `guitar_gaps` (pitch +0.8 pp, onset −1.7 pp — within noise). **The
cheap checkpoint-swap lever fails**; both shipped guitar checkpoints sit at ~0.68
pitch on electric (vs ~0.93 on acoustic). Closing the electric tiers therefore
requires **fine-tuning** the highres backbone on electric audio (Guitar-TECHS
CC-BY; EGDB if the grant permits training) — not a free swap.

Note: this required a backend fix — `highres-fl` was dead code (it passed a
non-existent `instrument="guitar_fl"`; the package only knows
saxophone/bass/guitar/piano). It now loads `guitar-fl.pth` via `checkpoint_path`
(`tabvision/audio/highres.py`).
