# Electric backbone fine-tune — design & prep (2026-06-02)

**Status:** prep / design. The fine-tune itself is free-tier **GPU** work; not
runnable on the laptop (no CUDA).
**Motivation:** `docs/EVAL_REPORTS/cross_dataset_prior_2026-06-02.md` showed the
highres backbone (acoustic GAPS) collapses on electric (pitch F1 0.93 → 0.73) and
the off-the-shelf `guitar_fl` swap doesn't help. Electric needs a fine-tune.

## Decision — a SEPARATE electric checkpoint, routed by tone

(Answers "should we tune electric on a different model so the current one isn't
confused?" — **yes**.)

Train a separate **`guitar-electric`** checkpoint; do NOT fine-tune one shared
model to cover both:

1. **Catastrophic forgetting is real.** Fine-tuning the acoustic checkpoint on
   electric would likely erode its 0.93 acoustic pitch F1 (negative transfer).
   A separate checkpoint preserves acoustic for free.
2. **The architecture already routes by checkpoint.** The package ships
   per-instrument checkpoints; the project already selects `guitar-gaps` vs
   `guitar-fl` via `--backend highres` / `highres-fl`. `guitar-electric.pth` +
   a `highres-electric` backend is the same pattern.
3. **The UI already has the signal.** Guided upload collects instrument/tone, so
   at inference you know electric vs acoustic and route — no one model has to
   disambiguate.
4. **Specialists beat a generalist on limited data.** Fine-tune *from* gaps
   (transfer learning, not from scratch): gaps already learned general
   guitar/pitch features; adapt the timbre-sensitive layers to electric.

Trade-off: a router that trusts the declared tone (mitigate with a cheap timbre
auto-detect or a sensible default when mislabeled). Two checkpoints to store —
trivial.

## Honest starting point (the real blocker)

- **No highres training code exists in this repo or the installed packages.**
  `hf_midi_transcription` / `piano_transcription_inference` are **inference-only**
  (no optimizer / loss / training loop). `scripts/train/audio_finetune.py` is a
  **scaffold** that writes a plan JSON, not a trainer. The existing fine-tune
  design (`2026-04-24-audio-backbone-finetune-design.md`) targets **Basic Pitch
  (TF)** — a different, older model.
- So fine-tuning highres requires the **upstream training code** for its
  architecture (xavriley/`hf_midi_transcription` source + the underlying
  hFT-Transformer / bytedance `piano_transcription` training repo). **Step 0 is
  to locate and stand that up.** This is the one thing between here and a run.

## Data (already on disk)

- **Guitar-TECHS** (CC-BY): electric, per-string 6-track MIDI → onset/pitch
  targets via the existing `guitar_techs_midi` parser. Split **by performer**:
  P1+P2 → train, **P3 → validation** (download P3 first — resumable
  `acquire guitar-techs`). ~5 h electric.
- Optional: **EGDB** (author-granted; distorted electric — for that tier) if the
  grant permits *training*; **EGFxSet** (electric + effects).
- Augmentation (per 2026-04-24 §7): SpecAugment + amp/cab IR convolution to span
  tones and reduce overfit to Guitar-TECHS's specific rigs.

## Two paths

- **Primary — fine-tune highres → `guitar-electric.pth`.** Best acoustic model,
  adapted to electric. Blocked on Step 0 (upstream training loop). Init from
  `guitar-gaps.pth`, unfreeze, lr ~1e-5–1e-4, batch 8, ~10–20 epochs.
- **Fallback — fine-tune Basic Pitch on electric.** The project already has TF
  fine-tune infra (`tabvision-server/tools/build_guitarset_tfrecords.py`,
  `app.training.*`) and Basic Pitch training is documented. If the highres
  training loop can't be stood up in a ~1-week timebox, fine-tune Basic Pitch on
  Guitar-TECHS electric and compare. (Weaker on acoustic, but on electric the gap
  may not matter — and it routes the same way.)

## Compute

Free-tier GPU per SPEC §6.3 / D6: **Lightning (22 GPU-hr/mo)** or Colab/Kaggle.
Est. ~3–8 GPU-hr for a first fine-tune. **Not the laptop.** W&B for tracking.

## Acceptance

- Electric **pitch F1 0.73 → ≥ 0.88** and onset F1 ≥ 0.88 on held-out
  Guitar-TECHS (P3).
- Clean-electric tier **Tab F1 materially up from 0.12**, iterating toward the
  SPEC §1.4 0.90.
- **No acoustic regression** — guaranteed by construction (separate checkpoint;
  gaps untouched). Sanity: re-run `local_guitarset.toml` with `--backend highres`
  → numbers unchanged.

## Integration (once the checkpoint exists)

Mirror the `highres-fl` wiring just landed in `tabvision/audio/highres.py`:
- add `"guitar_electric"` to `GUITAR_VARIANTS` + `_CHECKPOINT_FILE` (point
  `checkpoint_path` at the `guitar-electric.pth` — local path or HF `repo/file`);
- `register("highres-electric", ...)` in `tabvision/audio/backend.py`;
- route by the session's declared tone (electric → `highres-electric`, else
  `highres`) in `pipeline.run_pipeline` and the Modal adapter.

## Next actions to make it runnable

1. **Locate the upstream highres training code** (xavriley repo / hFT-Transformer
   / piano_transcription training) — the one real blocker.
2. `acquire guitar-techs` (resumes) to pull **P3** for a clean by-performer split.
3. Write the Guitar-TECHS → training-tensor data loader against that training
   code's expected input/label format.
4. Stand up a Colab/Lightning notebook: install training repo → prep data →
   fine-tune from gaps → export `guitar-electric.pth`.
5. Wire `highres-electric` + tone routing; validate on held-out Guitar-TECHS.
