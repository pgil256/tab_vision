# Tab F1 — Phase 0 Implementation Plan

**Date:** 2026-05-13
**Author:** Patrick (brainstormed with Claude)
**Status:** Proposed — pending sign-off
**Strategy doc:** `docs/plans/2026-05-12-tab-f1-to-spec-design.md`
**Implementation branch:** to be cut as `impl/tab-f1-phase-0` off `main`
              after the strategy / SPEC amendment lands.

## 0. Phase 0 goal recap

Establish the per-tier baseline and error decomposition needed to
sequence Phases 1+. **No production code changes; no shipped behavior
changes; no compute spend on training.**

Acceptance, copied from the strategy doc §6:

- Per-tier baseline numbers for ≥ 3 of 4 D2 tiers with **bootstrap
  95% CIs**, on the composite eval set.
- Per-tier six-bucket error decomposition on the same set
  (port of the apr-28 7-bucket harness; ``muted_undetectable`` deferred
  until the §8 ``TabEvent`` contract carries a muted/X flag).
- Free-tier compute accounts (Local / Colab / Kaggle / Lightning / W&B)
  verified.
- EGDB author email sent; reply tracked in `docs/DECISIONS.md`.

## 1. Files to add / modify

### 1.1 New files

| Path | Purpose |
|---|---|
| `tabvision/tabvision/eval/parsers/__init__.py` | Parser registry |
| `tabvision/tabvision/eval/parsers/guitarset_jams.py` | JAMS → `list[TabEvent]` |
| `tabvision/tabvision/eval/parsers/guitar_techs_midi.py` | 6-track MIDI → `list[TabEvent]` |
| `tabvision/tabvision/eval/parsers/egdb_gp.py` | GuitarPro tab + MIDI → `list[TabEvent]` (skipped at import-time if PyGuitarPro not installed; runs only when EGDB license clears) |
| `tabvision/tabvision/eval/composite.py` | `run_composite_eval(manifest_path) -> CompositeReport` — dispatches to per-source parsers and aggregates per-tier |
| `tabvision/tabvision/eval/bootstrap.py` | Bootstrap CI helper: `bootstrap_ci(values, statistic=mean, n=10_000, seed=int) -> tuple[float, float, float]` returning `(mean, lower_95, upper_95)` |
| `tabvision/tabvision/eval/error_decomposition.py` | Port of `tabvision-server/tools/error_analysis.py` (apr-28 7-bucket harness) targeting `list[TabEvent]` pairs |
| `tabvision/scripts/eval/composite_eval.py` | CLI wrapper: `tabvision-composite-eval --manifest data/eval/composite.toml --output docs/EVAL_REPORTS/composite_baseline_<date>.md` |
| `tabvision/scripts/eval/decompose_tab_errors.py` | CLI wrapper for error_decomposition.py |
| `tabvision/data/eval/composite.toml` | Composite-eval manifest (live; populated incrementally as datasets arrive) |
| `tabvision/data/fixtures/eval/guitarset_05_BN1-129-Eb_comp.jams` | Single-clip JAMS fixture for parser round-trip test |
| `tabvision/data/fixtures/eval/guitar_techs_sample.mid` | Single-clip 6-track MIDI fixture |
| `tabvision/tests/unit/test_parser_guitarset_jams.py` | JAMS parser round-trip test |
| `tabvision/tests/unit/test_parser_guitar_techs_midi.py` | MIDI parser round-trip test |
| `tabvision/tests/unit/test_bootstrap_ci.py` | CI helper correctness on known distributions |
| `tabvision/tests/unit/test_error_decomposition.py` | Per-bucket assignment correctness on synthetic predicted/gold pairs (six buckets populated) |
| `tabvision/tests/integration/test_composite_eval_smoke.py` | End-to-end smoke: 5-clip manifest → tier numbers exist + CIs computed |
| `docs/EVAL_REPORTS/composite_baseline_2026-05-13.md` | First baseline report (output of Phase 0E) |
| `docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md` | First six-bucket decomposition (output of Phase 0D) |

### 1.2 Modified files

| Path | Lines | Change |
|---|---|---|
| `tabvision/tabvision/eval/manifest.py` | the `REQUIRED_CLIP_FIELDS` block (currently ~lines 21-28) | Add `annotation_format` field so parser-dispatch can route by source |
| `tabvision/tabvision/eval/manifest.py` | `validate_manifest()` | Reject any clip whose `source` indicates synthetic origin (e.g. starts with `synthtab/` or `dadagp/`) from a non-train split. This is the R8 cross-contamination guard from the strategy doc. |
| `LICENSES.md` | datasets table | Add Guitar-TECHS (CC-BY-4.0), EGDB (pending), free IR packs as they're acquired |
| `docs/DECISIONS.md` | append | D1–D11 from strategy doc §1 |
| `pyproject.toml` (in `tabvision/`) | `[project.optional-dependencies]` | Add `eval` extra with `pretty_midi`, `pyguitarpro`, `jams` (already used elsewhere — verify before adding) |

### 1.3 NOT modified

- `tabvision/tabvision/pipeline.py` — no behavior change in Phase 0.
- `tabvision/tabvision/fusion/**` — no fusion changes.
- `tabvision-server/modal_app.py`, `tabvision-server/app/v1_adapter.py` — no production changes.
- `tabvision-server/app/v1_adapter.py:91` `videoIgnoredByQualityGate` — flagged in strategy doc as a faked diagnostic, but the fix is Phase 6's job, not Phase 0's.

## 2. Test plan

Every test must be runnable via `pytest tabvision/tests/...` and skip
cleanly when an optional dependency is missing (PyGuitarPro, jams).
Fixtures go under `tabvision/data/fixtures/eval/`.

### 2.1 Unit tests

| Test name | Fixture | Assertion |
|---|---|---|
| `test_parser_guitarset_jams.py::test_jams_round_trip_pitch_string_fret` | `guitarset_05_BN1-129-Eb_comp.jams` (small, ~50 notes) | Every emitted `TabEvent` has `0 ≤ string_idx ≤ 5`, `0 ≤ fret ≤ 24`, monotonically non-decreasing `onset_s`. Total event count matches the JAMS namespace's note count. |
| `test_parser_guitarset_jams.py::test_jams_pitch_consistency` | same | For each emitted event, MIDI pitch implied by `(string_idx, fret)` matches the JAMS-reported pitch. |
| `test_parser_guitar_techs_midi.py::test_midi_round_trip_per_string` | `guitar_techs_sample.mid` (6 tracks, 1 per string) | Track index → `string_idx` mapping correct: track 0 → low E (`string_idx=0`), track 5 → high E (`string_idx=5`). |
| `test_parser_guitar_techs_midi.py::test_midi_pitch_to_fret` | same | Per-string MIDI pitch → fret derivation matches expected standard-tuning offsets: E2=40 → fret 0 string 0, A2=45 → fret 5 string 0, etc. |
| `test_bootstrap_ci.py::test_ci_known_normal` | synthetic Gaussian N(0.85, 0.05), n=100 | Returned 95% CI brackets the true mean ≥ 95% of the time over 1000 trials (calibration check). |
| `test_bootstrap_ci.py::test_ci_handles_small_samples` | n=5 | No exception; CI width sane (≥ standard error). |
| `test_bootstrap_ci.py::test_ci_deterministic_with_seed` | any | Same seed → same CI. |
| `test_error_decomposition.py::test_seven_buckets_assigned` | synthetic gold + predicted `TabEvent` lists, one per bucket | Each ground-truth event lands in the expected bucket: `correct`, `wrong_position_same_pitch`, `pitch_off`, `timing_only`, `missed_onset`, `muted_undetectable`, `extra_detection`. |
| `test_error_decomposition.py::test_share_of_loss_sums_to_one` | mixed gold + predicted | Per-bucket share-of-loss percentages sum to 100% (excluding the `correct` bucket). |

### 2.2 Integration tests

| Test name | Setup | Assertion |
|---|---|---|
| `test_composite_eval_smoke.py::test_five_clip_manifest` | A 5-clip composite manifest using checked-in fixtures (3 GuitarSet, 2 Guitar-TECHS) | `run_composite_eval(manifest)` returns a `CompositeReport` whose tiers include both `clean_acoustic_single_line` and `clean_acoustic_strummed`. Each tier has a non-null `tab_f1_mean` and `tab_f1_ci_95`. |
| `test_composite_eval_smoke.py::test_synthetic_clip_rejected_from_eval` | A manifest with one clip whose `source = "synthtab/test"` and `split = "test"` | `validate_manifest()` raises with a message mentioning the cross-contamination guard. |
| `test_composite_eval_smoke.py::test_egdb_skipped_when_pyguitarpro_missing` | Manifest with an EGDB clip but PyGuitarPro not installed | Run completes successfully; the EGDB clip is reported as `skipped` with reason `parser_dependency_missing`. Other clips still evaluated. |

### 2.3 What's NOT tested in Phase 0

- The actual D2 acceptance numbers — those are the *output* of running
  the harness, not a unit-test assertion. The CI gate is what's tested;
  whether the system *hits* 0.85/0.90/0.87/0.80 is a question Phases
  1-8 answer.
- Bootstrap confidence on real production data — covered by the
  smoke test on fixtures; running on production data is a one-shot
  command, not a CI test.

## 3. Commands

All commands run from repo root, in the WSL Ubuntu shell, with the
`tabvision` venv active (`source tabvision/venv/bin/activate` or
`pip install -e tabvision[dev,eval]`).

### 3.1 One-time setup

```bash
# Install eval extras (PyGuitarPro, pretty_midi, jams)
cd tabvision && pip install -e '.[dev,eval]' && cd -

# Verify tests pass on the base
pytest tabvision/tests/unit/test_parser_guitarset_jams.py -v
pytest tabvision/tests/unit/test_bootstrap_ci.py -v
```

### 3.2 Acquire Guitar-TECHS

```bash
# Guitar-TECHS is CC-BY-4.0, hosted on Zenodo (see strategy doc §4.1)
mkdir -p ~/mir_datasets/guitar_techs
# Download the dataset archive from the URL in arXiv:2501.03720
# (resolved at acquisition time; not committed to repo)
# Extract into ~/mir_datasets/guitar_techs/
ls ~/mir_datasets/guitar_techs/
```

### 3.3 Build the manifest

```bash
# Generate composite.toml from on-disk datasets
python tabvision/scripts/eval/build_composite_manifest.py \
  --guitarset ~/mir_datasets/guitarset \
  --guitar-techs ~/mir_datasets/guitar_techs \
  --output tabvision/data/eval/composite.toml

# Validate it
python -c "from tabvision.eval.manifest import validate_manifest; print(validate_manifest('tabvision/data/eval/composite.toml'))"
```

### 3.4 Run the baseline composite eval

```bash
python tabvision/scripts/eval/composite_eval.py \
  --manifest tabvision/data/eval/composite.toml \
  --backend highres \
  --position-prior guitarset-v1 \
  --bootstrap-n 10000 \
  --bootstrap-seed 42 \
  --output docs/EVAL_REPORTS/composite_baseline_2026-05-13.md
```

### 3.5 Run the error decomposition

```bash
python tabvision/scripts/eval/decompose_tab_errors.py \
  --manifest tabvision/data/eval/composite.toml \
  --backend highres \
  --position-prior guitarset-v1 \
  --output docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md
```

### 3.6 Verify free-tier compute accounts

```bash
# W&B: confirm login + a tiny no-op run
wandb login
python -c "import wandb; r = wandb.init(project='tabvision-phase0', mode='online'); r.log({'hello': 1}); r.finish()"

# Lightning Studios: open a Studio in the browser, run `nvidia-smi`, screenshot for the DECISIONS.md log

# Kaggle: open a notebook in the browser, run `!nvidia-smi`

# Colab: same

# Modal: skip — used only as last resort per D6
```

### 3.7 Send the EGDB email

User action — not a command. Template in strategy doc; log the
date sent and the reply (when it arrives) in `docs/DECISIONS.md`.

## 4. Acceptance outputs

These are the artifacts whose existence + content gates Phase 1.

### 4.1 `docs/EVAL_REPORTS/composite_baseline_2026-05-13.md`

Must contain:

- A per-tier table:
  - Tier name
  - Clip count (≥ 20 for any tier claimed against D2)
  - Mean Tab F1
  - **95% bootstrap CI lower bound**
  - Mean Onset F1
  - Mean Pitch F1
- Per-source breakdown within each tier (GuitarSet / Guitar-TECHS /
  EGDB) so we can see whether a tier number is dominated by one
  source.
- A "Status vs D2 target" column with one of: **pass** (CI lower ≥
  target), **gap** (mean ≥ target but CI lower below), **fail** (mean
  below target).
- Methodology footer: bootstrap N, seed, parser versions, backend +
  prior versions, eval-harness commit SHA.

### 4.2 `docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md`

Must contain:

- Aggregate six-bucket table (counts + share-of-loss).
- Per-tier six-bucket table.
- A "biggest lever per tier" callout: which bucket dominates each
  tier's loss. Phase 1+ priorities derive from this.

### 4.3 `tabvision/data/eval/composite.toml`

Must satisfy `validate_manifest()` and contain:

- ≥ 20 clips for each of: `clean_acoustic_single_line`,
  `clean_acoustic_strummed`. (Guitar-TECHS additions may bring
  `clean_electric` to ≥ 20 in Phase 0E; if not, that tier waits for
  EGDB.)
- `clean_electric` and `distorted_electric` populated as much as
  Guitar-TECHS + EGDB-license-resolved allow.
- No `source = synthtab/...` or `source = dadagp/...` rows in `split =
  validation` or `split = test`.

### 4.4 `docs/DECISIONS.md` entries

D1–D11 from strategy doc §1, dated 2026-05-13. EGDB email send-date
and reply (when it arrives) as a separate entry.

### 4.5 CI verification

`pytest tabvision/tests/unit tabvision/tests/integration -v` passes
on `main` HEAD plus this Phase 0 branch.

## 5. Decision tree

What to do after Phase 0E baseline is in:

- **All four tiers' CI lower bound clears D2** — surprising; sanity
  check the eval harness, then declare v1 acceptance and skip to
  Phase 9. This is unlikely given the 2026-05-08 0.61 aggregate.
- **Strummed CI lower bound clears D2, other tiers gap or fail** —
  expected case. Proceed to Phase 1 (pitch ceiling lift). The
  error-decomposition report tells us whether Phase 2 (fine-tune) or
  Phase 3 (style priors) is the next priority after Phase 1.
- **All tiers fail** — Phase 0 implementation has a bug, or the
  highres backend regressed on the broader corpus. Inspect 3-5
  worst-case clips by hand before any further compute spend.
- **`distorted_electric` has < 20 clips** — EGDB license is the
  blocker. Set the tier aside; document the gap in the report; do not
  publish D2 acceptance until the EGDB row clears.

## 6. Time + compute budget

| Item | Effort | Compute |
|---|---|---|
| Parser implementations + tests (1.1) | 1.5 days | none |
| Manifest extensions + validator hardening (1.2) | 0.5 day | none |
| Composite + bootstrap + error-decomposition modules (1.1) | 1 day | none |
| Guitar-TECHS acquisition + manifest population | 0.5 day | none |
| Baseline + decomposition runs (3.4 + 3.5) | 4-8 wall-clock hours | local CPU |
| Free-tier compute account verification | 0.5 day | none |
| EGDB email + DECISIONS.md updates | 15 minutes | none |
| Report writing | 0.5 day | none |
| **Total** | **4-5 days engineering** | **~$0** |

## 7. Out of scope for Phase 0

- Any production-pipeline change. No edits to `pipeline.py`, `fusion/`,
  `audio/`, `video/`, `tabvision-server/`.
- Fine-tuning, training, or model weight changes.
- Anything depending on the EGDB license reply (defer to Phase 8 or
  later).
- Style-conditional priors (Phase 3).
- Video pipeline experiments (Phase 6).
- Synthetic-data generation (research/dev only; not part of Phase 0).

## 8. Done definition

Phase 0 is **done** when:

- All items in §1.1 and §1.2 exist on the impl branch.
- All tests in §2.1 and §2.2 pass green.
- `docs/EVAL_REPORTS/composite_baseline_2026-05-13.md` exists and meets
  §4.1.
- `docs/EVAL_REPORTS/tab_f1_error_decomposition_2026-05-13.md` exists
  and meets §4.2.
- `tabvision/data/eval/composite.toml` exists and validates.
- `docs/DECISIONS.md` includes D1–D11.
- EGDB email send-date recorded.
- Free-tier compute accounts verified (W&B at minimum; Lightning /
  Kaggle / Colab logged in `docs/DECISIONS.md`).

Then — and only then — the Phase 1 implementation plan gets written.
