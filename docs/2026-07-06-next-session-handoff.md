# Next-session handoff prompt (from the 2026-07-06 "a14+a10 → B4/B3 → A6 → A3/A4" chunk)

Paste the block below into a fresh Claude Code session. It is self-contained; it
does not rely on the prior conversation. Verify the "current state" against git
before acting — background jobs from the last session do **not** survive it, so
the A6 eval and A3 sweep were almost certainly interrupted and must be re-run.

---

You are picking up a multi-item accuracy/UX chunk on the TabVision repo. Read
`CLAUDE.md`, `docs/plans/2026-07-01-accuracy-ux-roadmap.md`, and the tail of
`docs/DECISIONS.md` first. Local eval setup (non-obvious, Windows) is in your
memory note `tabvision-eval-env`: use `tabvision/.venv/Scripts/python.exe`,
export `TABVISION_DATA_ROOT=~/.tabvision/data`, prepend
`~/.tabvision/tools/ffmpeg-master-latest-win64-gpl/bin` to PATH, set
`PYTHONUTF8=1`, and write eval reports to the **repo-root** `docs/EVAL_REPORTS/`
(from the `tabvision/` dir that means `..\docs\EVAL_REPORTS\...`).

## Current state (verify with `git branch`, `gh pr list`, `git status`)

**All six items from the 2026-07-06 chunk are DONE and PR'd** (five branches off
`main`, b46c175). Merge discipline: run `pytest -v`, `ruff check .`,
`ruff format --check .`, `mypy tabvision` green before any PR; per SPEC §8 the
`fuse`/`transition_cost` signatures are contracts.

| PR | Branch | Item | State |
|---|---|---|---|
| #22 | `v1.1/a14-a10-probes` | A10 + A14 | open — includes the D1 packet still awaiting the user |
| #23 | `v1.1/b4-b3-correction-ux` | B4 + B3 | open — B4 confidence + B3 string editor |
| #24 | `v1.1/a6-gaps-gold-coverage` | A6 | open — GAPS gold 0.6468→0.6969 (honest, coverage-accounting) |
| #25 | `v1.1/a3-a4-fusion-sweep` | A3 + A4 + A12 | open — sweep infra, A4 wash, A12 verdict; **no default changed** |

Everything the last session was mid-running is finished. The A3 sweep's one
**domain-neutral candidate worth gating** is `OPEN_STRING_BONUS=0.0` (strummed
0.7951→0.8140, single-line flat) — see P0 below.

## STATUS 2026-07-07 — the whole chunk is MERGED; P0/P1/P3 below are DONE

All six items (A10/A14, B4/B3/B5, A6, A3/A4/A12) plus the D1 SPEC retirement are
merged to `main` (PRs #22–#26). **P0 and P1 and P3 below are complete** — kept
here only as the record of what was done. The genuine remaining next work is
**P2 (A8)** and the **D1 remainder**. See the memory note
`project_2026-07-06-b4b3-a6-a3a4-chunk` for the current state.

## Then — the prioritized next work (informed by a blindspot audit; do in order)

### P0. Gate the one clean A3 candidate — **DONE 2026-07-07 (#26): REJECTED.**
`OPEN_STRING_BONUS=0.0` PASSED GuitarSet 60-clip (both tiers' lo-95 up) but
FAILED GAPS clean-12 (single-line lo-95 −0.0091, 11/12 clips regress) — even a
hand-coded bonus is corpus-coupled. No default change. The prior-trust movers
(`FRET_PRIOR_WEIGHT`/`LOW_FRET_BIAS`/`power`) remain the expected-to-fail
GuitarSet-overfit trap; not pursued. (`a3_gate_open0_*_2026-07-07.md`,
DECISIONS 2026-07-07.) Gate harness: `scripts/eval/a3_gate_probe.py`.

### P1. Land the shipped PRs safely + resolve the one real merge conflict — **DONE (#22→#24→#23→#25).**
Merged in that order. The #23↔#25 `viterbi.py`/`playability.py` conflict was
hand-resolved (B4's forward+backward margin + A4's `gap_s` threaded through both
passes; `playability.py` auto-merged cleanly). A latent OOM bug in
`a3_fusion_sweep._raw_events_cached` (backend rebuilt per clip) was found and
fixed during the gate run (`make_shared_audio_backend`, +4 regression tests).
Original guidance kept below for reference:
#22 (A10/A14) and A6 are isolated — merge cleanly anytime. **#23 (B4) and the
A3/A4 branch BOTH rewrite `tabvision/tabvision/fusion/viterbi.py` `_viterbi_clusters`
and edit `playability.py` (constants block, `transition_cost`, `__all__`).**
Merge one, then rebase the other and **combine** the changes by hand: keep B4's
forward+backward margin decode **and** A3/A4's `gap_s` param on `transition_cost`
plus the inter-cluster gap computation (must be threaded through **both** the
forward and backward passes), and union both sets of new constants/`__all__`
entries. Re-run `tests/unit/test_fusion_audio_only.py`, `test_string_confidence.py`,
`test_fusion_constants_a3.py`. A careless auto-merge can silently drop `gap_s` or
the margin confidence — do not trust a clean-looking auto-resolution.

### P2. A8 — studio-condition degradation eval (the highest-value item left)
**Why it's #1 on merit:** every accuracy number in the repo is on clean corpus
WAVs, but the product ingests Opus-in-webm from laptop mics — the eval-vs-product
gap is the biggest unmeasured risk. A8 re-encodes `data/eval/local_gs_val24.toml`
through the real capture chain (opus-in-webm 48k, laptop-mic lowpass, noise floor,
light compression — **ffmpeg only**, gold labels carry over, fully automated) and
re-scores. It's an afternoon and it *decides the fork*: if accuracy holds, keep
tuning; if it craters, pivot to input robustness (denoise/AGC, B9 bad-input banner,
preflight) instead of fusion constants. Diagnostic tier, **not** a gate (don't edit
SPEC §1.4 targets). Bank the result either way. Roadmap: A8 (Tier 2).

### P3. Harden the correction UX into the real value story — **B5 DONE (#23); capo guard SKIPPED per user.**
B4+B3 is the plan's centerpiece but shipped soft on two axes the audit confirmed:
- **B5 — persistence: DONE (#23).** localStorage autosave + `RestoreBanner`
  implemented (edits survive refresh); 21 headless checks. The `PATCH
  /jobs/:id/result` + recent-transcriptions-list half is still deferred.
- **B3 capo/alt-tuning guard: SKIPPED** — user 2026-07-07 said not to worry about
  alternate tunings. `pitchPreservingFret` in
  `web-client/src/store/appStore.ts` hardcodes a **standard-tuning** interval
  table (`STRING_OPEN_MIDI`). A capo cancels (both strings shift equally) so that's
  fine, but an *alternate tuning* (DADGAD/drop-D/open) makes Shift+Up/Down produce
  a silently-wrong fret with no warning. `TabDocument.tuning` is emitted by
  `tabvision-server/app/v1_adapter.py` as `["E","B","G","D","A","E"]`; guard the
  string-move (disable + tooltip, or recompute from the real tuning) when it isn't
  standard. Small fix; ship with B5.

## Also on the table (don't lose these)
- **D1 decision packet** (`docs/2026-07-06-d1-decision-packet.md`, on PR #22)
  still awaits the user: retire the 0.86 strummed / 0.85 chord video-assisted
  references (A14 refuted them), expressive-markings stretch, studio tier, stale
  SPEC §15. Don't edit SPEC without the user's answer.
- **A5 (chord-shape dictionary port)** is the next-biggest *accuracy* lever and
  rides the A3 sweep harness — but sequence it **after** A8, since A8 may redirect
  effort away from clean-corpus accuracy entirely.
- Honest ceiling reminder: video (A14) and cross-domain priors (A2) are closed;
  the single-line audio ceiling is real. Product upside is now measurement (A8),
  fixability (B4/B3/B5), and input robustness — not another decimal of clean Tab F1.

Measurement discipline throughout: val24 fast-loop → 60-clip lower-95 confirm
before any default change; per-clip no-regression on GAPS clean-12 for anything
touching fusion; bank every result (positive/wash/negative) in `docs/EVAL_REPORTS/`
+ `DECISIONS.md`.
