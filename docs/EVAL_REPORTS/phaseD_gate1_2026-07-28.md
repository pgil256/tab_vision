# Phase D Gate 1 — the WS4 retrain does not clear the bar

**Date:** 2026-07-28 · **Verdict: FAIL. Bank the negative.**
Best clip-disjoint validation 6-way accuracy **0.2919** against the
pre-registered bar of **≥ 0.45** (chance 0.167).

Pre-registration: `docs/plans/2026-07-27-video-evidence-roadmap-design.md` §7 —
*"Gate 1 (local, $0 beyond extraction): clip-disjoint val 6-way accuracy ≥ 0.45.
Below that, bank the negative; no pipeline A/B, no further spend."*

Reproduce:

```bash
cd tabvision
python -m scripts.acquire.datasets gaps-annotations --split train \
    --only-cached-video ~/.tabvision/cache/gaps_video_720
python -m scripts.train.extract_string_dataset --clips train --hand-tight --sustain \
    --video-cache ~/.tabvision/cache/gaps_video_720
modal run scripts/train/string_resolver_modal.py --epochs 20 --batch 128
```

---

## 1. The claim under test

The banked WS4 negative (net −0.117 Tab F1; val 6-way accuracy plateau ~0.30)
had a documented root cause and a documented, committed-but-unauthorised fix:

1. **the whole-neck crop starves the model** → crop hand-tight instead;
2. **onset-frame label alignment noise** → sample inside the note's sustain.

Both were implemented (`--hand-tight`, `--sustain`, 11 unit tests) and this run
is the first time they have been executed. Everything else was held frozen:
clip-disjoint split, `peak_ratio ≥ 2.0` alignment filter, **no flips**.

## 2. Dataset

| quantity | value |
|---|---:|
| clips extracted | **241** (of 252 with cached 720p video) |
| crops | 188,783 |
| median alignment `peak_ratio` | 4.495 |
| crops surviving the `≥ 2.0` filter | 181,937 (**96.4%**) |
| train / val after the clip-disjoint split | 159,381 / 22,556 |

Seven clips failed extraction across the two run segments (gold-parse errors such
as `KeyError: ' '`); they were skipped by design and are a rounding error against
241 successes.

## 3. Result

Modal L4, ResNet-18, 20 epochs, batch 128, lr 3e-4, seed 0.

| epoch | train_loss | val_acc |
|---:|---:|---:|
| 0 | 1.6208 | 0.2634 |
| 1 | 1.5442 | 0.2895 |
| **3** | 1.4084 | **0.2919** ← best |
| 7 | 0.8540 | 0.2571 |
| 11 | 0.5163 | 0.2598 |
| 15 | 0.3983 | 0.2504 |
| 19 | 0.3371 | 0.2619 |

**Best 0.2919, at epoch 3.** Training loss falls monotonically 1.62 → 0.34 while
validation accuracy peaks early and then drifts *down*. That is textbook
overfitting: the model is memorising crops, not learning string identity.

## 4. What this refutes

The banked WS4 run plateaued at **~0.30**. This run reaches **0.2919** — the same
plateau, within noise, on 159k hand-tight sustain-sampled crops.

**So the documented root cause is not the explanation.** "The whole-neck crop
starves the model" and "onset-frame labels are misaligned" were plausible, were
committed as the fix, and have now been tested: fixing both moves the plateau by
approximately nothing. Whatever limits this model, it is not crop framing or
onset-frame label noise.

Worth noting against over-reading an early signal: the partial-data smoke run
earlier the same day (30 clips, 2 epochs) reached **0.3135**, *higher* than this
full run's best. That was a small, easy validation split, and it was explicitly
reported at the time as "not a Gate 1 reading". It is a good example of why the
gate was pre-registered on the full clip-disjoint split.

## 5. Consequences, per the pre-registered tree

- **No pipeline A/B.** Gate 2 (cache-only gated clean-12 A/B) does not open.
- **No further spend.** The Modal fine-tune this gate was to authorise is not
  justified; total spend on Phase D was the single ~$1 Gate 1 run itself.
- The learned string resolver stays an **eval-only artifact**, exactly as WS4 v1
  did.

## 6. What is *not* established

- **This is not "learned string resolution is impossible".** It is one backbone
  (ResNet-18), one crop policy, one sampling policy, one seed, 20 epochs. The
  overfitting signature suggests capacity/regularisation and augmentation are
  untested variables, not that the signal is absent.
- The early peak at epoch 3 means a shorter schedule with stronger
  regularisation is the obvious next configuration — **untested**, and it would
  need its own pre-registration rather than an unbounded hyperparameter search.
- No Tab F1 was measured. The gate is deliberately upstream of it.
- GAPS is classical/nylon; nothing here transfers automatically to the
  steel-string acoustic default.

## 7. Operational note

The extraction was interrupted twice by WSL restarts (14:36 and ~16:54) and
recovered both times — the second automatically, by the supervisor installed
after the first. Total: 241 clips, no manual intervention after the supervisor
was in place. Manifest-level resumability meant each interruption cost at most
one clip.
