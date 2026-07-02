# v1.1 chunk-6 WS4 — learned string-resolution model: MEASURED NEGATIVE

**Date:** 2026-06-29
**Branch:** `v1.1/oracle-string-resolution`
**Status:** **WS4 trained + evaluated — a clear negative.** The learned model does
**not** beat the geometric chain; it is net-negative as string evidence. Chunk-6's
positive deliverable remains the geometry WS1 fret-map (0.544 → 0.574, committed
`587c174`). The full WS4 pipeline (acquire → extract → train → eval) is committed
and reusable; this report records the outcome so the bet isn't re-run blind.
**Approved HARD-STOP:** user-authorized training run (DECISIONS 2026-06-25; full
270-clip GAPS train split, Modal L4, NC license accepted).
**Design:** `docs/plans/2026-06-25-v1.1-ws4-learned-string-model-design.md`.

## 1. Result — go/no-go eval (held-out clean-12, gold frame)

`scripts/eval/v1_1_gaps_learned_probe.py`, ungated, the trained checkpoint:

| | Tab F1 |
|---|---:|
| audio-only | **0.8148** |
| **+learned** | **0.6974** |
| +oracle | 0.9726 |

**The learned string evidence is net-negative: it drags Tab F1 *down* by −0.117**
(0.8148 → 0.6974) — it makes more strings wrong than right. For contrast, the
geometric chain (gated) held 0.8148 with no regression, and WS1's fret-map lifted
the *string-accuracy* leading indicator 0.544 → 0.574. The learned model clears
neither bar.

## 2. Why — the training plateaued (overfit to players)

ResNet-18 over the YOLO neck-crop, 6-way string head, trained on **153,482 labeled
crops** from **251 GAPS train clips** (official train/test split — clean-12 ⊂ test,
no leakage), clip-disjoint val:

```
epoch 0:  val_acc 0.252   train_loss 1.63
epoch 8:  val_acc 0.297   train_loss 0.95
epoch 12: val_acc ~0.30   train_loss 0.66   ← val flat from epoch 8, train still falling
```

Raw 6-way val_acc plateaued at **~0.30** (chance 0.167) by epoch 8 while train loss
kept dropping — **overfitting to the training players**, not learning transferable
string cues. Pitch-restricting to each note's 2–3 candidate strings at inference
wasn't enough to recover a useful signal (hence the net-negative Tab F1).

**Likely root causes** (not pursued — see §4):
- **Whole-neck crop starves the model.** The fretting hand is small in a wide neck
  crop squished to 224²; the across-string finger position (the actual signal) is
  too coarse. A crop tight to the hand is the obvious next lever.
- **Label noise from onset-frame alignment** — the onset frame can catch the hand
  in transition rather than a settled press.
- Generalising string-from-image across players/guitars/angles is simply hard.

## 3. The wall this confirms (SPEC §0 rule 7)

WS1/WS2 (geometry) plateaued ~0.57 because ~68% of ambiguous notes are on clips with
~0 detected frets. WS4 was the approved escalation to the highest-ceiling lever (a
learned model resolving strings from pixels). It did not clear the bar in this form.
The honest state: **audio-only single-line string resolution on in-the-wild GAPS
footage remains information-limited, and neither the geometric chain nor a
GAPS-trained ResNet-18 neck-crop classifier reliably beats the audio playability
prior.** Geometry WS1 is the one measurable, no-regression positive.

## 4. Decision — bank the negative, stop (user, 2026-06-29)

Documented and stopped. The gap is large (net-negative, val ~0.30), not marginal, so
a tighter-hand-crop retry — the one promising fix — is speculative against that gap
and was not authorized for further spend. If revisited, the recipe is intact: the
extraction (`scripts/train/extract_string_dataset.py`), training
(`train_string_resolver.py` + `string_resolver_modal.py`, Modal L4), and this eval
probe are committed and reusable; the change would be a hand-tight crop in extraction
+ re-train. GAPS media + the trained checkpoint are NC, offline-only — never
committed.

## 5. Reproduce

```bash
cd tabvision
export TABVISION_DATA_ROOT=~/.tabvision/data
export PATH=~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin:$PATH
# (training: modal volume put the dataset tarball, then)
modal run scripts/train/string_resolver_modal.py --epochs 12 --batch 128
# eval the checkpoint on held-out clean-12:
python -m scripts.eval.v1_1_gaps_learned_probe \
  --checkpoint <best.pt> \
  --yolo-checkpoint ~/.tabvision/data/models/guitar-yolo-obb-finetuned.pt
```
