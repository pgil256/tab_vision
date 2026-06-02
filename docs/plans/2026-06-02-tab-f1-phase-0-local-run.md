# Phase 0 — local run: cross-dataset prior check (#2)

**Date:** 2026-06-02
**Hardware:** ThinkPad T14 (i7-1185G7, 4c/8t, 32 GB, **no CUDA GPU**, 393 GB free).
**Scope:** Run the #2 cross-dataset prior-generalization check **locally on CPU**.
The #3 fine-tune is **not** here — it goes to free GPU (Lightning/Colab) per
SPEC §6.3 / design-doc D6. This is the "you run / I prepped it" half of the split.

**The question #2 answers:** the `guitarset-v1` position prior gave **+22 pp Tab
F1 on GuitarSet** (0.388 → 0.610). Is that a real prior over guitar physics, or
did it memorise GuitarSet's distribution? We test it on **Guitar-TECHS** (a
different corpus, *electric* guitar) — which the GuitarSet-trained prior has
never seen. If the lift holds, the prior generalises; if it vanishes or
regresses, the headline number is GuitarSet-specific and the accuracy story
needs reframing before we build on it.

> ⚠️ The Guitar-TECHS scanner (`manifest_builder.scan_guitar_techs`) infers the
> on-disk layout from arXiv:2501.03720 + the project page. **After the first
> download, eyeball the tree the acquirer prints and confirm the manifest shows
> non-zero `GuitarTECHS` clips.** If it shows 0, adjust the globs/keywords in
> `scan_guitar_techs` (see `tests/unit/test_scan_guitar_techs.py` for the
> assumed shape).

---

## 0. Install (one time)

CPU torch + the highres backend + eval + mirdata (for GuitarSet):

```bash
cd tabvision
python -m pip install -e '.[audio-highres,eval,train]'
# (Windows: use `py -3 -m pip ...`; WSL/venv: `python -m pip ...`)
```

Pick a data root and export it (the acquirers + the checked-in manifests use it):

```bash
export TABVISION_DATA_ROOT="$HOME/.tabvision/data"        # bash / WSL
# PowerShell:  $env:TABVISION_DATA_ROOT = "$HOME\.tabvision\data"
```

## 1. Acquire the data (CPU, just downloads)

```bash
python -m scripts.acquire.datasets guitarset       # mirdata → $TABVISION_DATA_ROOT/guitarset
python -m scripts.acquire.datasets guitar-techs    # Zenodo  → $TABVISION_DATA_ROOT/guitar-techs
```

Both are CC-BY-4.0 and idempotent (re-run = skip). GuitarSet ≈ a few GB;
Guitar-TECHS ≈ 5 h of audio. The `guitar-techs` command prints its top-level
tree at the end — **use it to sanity-check the scanner assumption.**

## 2. Build the manifests

```bash
# (a) GuitarSet-only — reproduce the 0.61 baseline locally (player 05 = validation)
python -m scripts.eval.build_composite_manifest \
  --guitarset "$TABVISION_DATA_ROOT/guitarset" \
  --data-root "$TABVISION_DATA_ROOT" \
  --output data/eval/local_guitarset.toml

# (b) Guitar-TECHS-only — the cross-dataset target (no GuitarSet → no prior leak)
python -m scripts.eval.build_composite_manifest \
  --guitar-techs "$TABVISION_DATA_ROOT/guitar-techs" \
  --data-root "$TABVISION_DATA_ROOT" \
  --output data/eval/local_guitar_techs.toml
```

> Each build prints a per-tier × source coverage summary, then runs manifest
> validation. **Expect a non-zero exit + "missing required tier" warning** —
> these single-source manifests don't cover all four tiers (distorted-electric
> needs EGDB). The TOML is still written and is fine for #2.

## 3. Run #2 — prior ON vs OFF

`guitarset-v1` was trained only on GuitarSet, so **all** Guitar-TECHS clips are
held out w.r.t. it → it's safe to evaluate the whole Guitar-TECHS set (incl. its
`train` split). For GuitarSet we keep the leak-free **player-05 validation**
split only.

```bash
# --- GuitarSet baseline (sanity: should reproduce ~0.61 vs ~0.39) ---
python -m scripts.eval.composite_eval --manifest data/eval/local_guitarset.toml \
  --backend highres --position-prior guitarset-v1 \
  --output docs/EVAL_REPORTS/local_guitarset_prior.md
python -m scripts.eval.composite_eval --manifest data/eval/local_guitarset.toml \
  --backend highres --position-prior none \
  --output docs/EVAL_REPORTS/local_guitarset_noprior.md

# --- Guitar-TECHS cross-dataset (the actual #2 question) ---
python -m scripts.eval.composite_eval --manifest data/eval/local_guitar_techs.toml \
  --backend highres --position-prior guitarset-v1 --splits validation,test,train \
  --output docs/EVAL_REPORTS/local_guitartechs_prior.md
python -m scripts.eval.composite_eval --manifest data/eval/local_guitar_techs.toml \
  --backend highres --position-prior none --splits validation,test,train \
  --output docs/EVAL_REPORTS/local_guitartechs_noprior.md
```

CPU note: the highres transformer runs ~real-time-to-a-few×-slower per clip on
4 cores. Subset with `--max-clips-per-tier` / `--limit` at build time for a
same-day read; run the full set overnight.

## 4. Read the verdict

Compare the **clean_electric (GuitarTECHS) Tab F1, prior ON − prior OFF**:

| Outcome | Δ Tab F1 on Guitar-TECHS | Reading |
|---|---|---|
| Lift holds | ≳ +10 pp (lower 95% CI > 0) | Prior generalises — safe to build on; proceed to #3 on GPU |
| Lift shrinks | small +, CI crosses 0 | Partly GuitarSet-specific — keep prior, but expect tier-specific work |
| **Regression** | ≤ 0 | Prior is GuitarSet-memorised — **stop and reframe** before #3; the +22 pp is not a general result |

Paste the four reports back here and I'll do the comparison + write the decision
into `docs/DECISIONS.md`.

## Later / not in this run

- **EGDB** (distorted-electric tier): `pip install gdown` then
  `python -m scripts.acquire.datasets egdb`. Folds in a 4th tier; not needed for #2.
- **#3 fine-tune:** free GPU only. After #2's verdict, I'll prep the Lightning/Colab job.
