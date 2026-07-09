# Phase 9 (Polish) — kickoff

**Date:** 2026-07-09
**Authorization:** User gave the explicit "proceed" required by SPEC §0 rule 2
to start Phase 9 (decision item **D4**). This document opens the phase and
records its starting state; it is not itself the phase work.

## Scope

Phase 9 = **ship-quality v1** (SPEC §7 Phase 9). The user framed D4 as the
"final documentation / narrative pass" — that is the headline remaining item —
but the phase's acceptance test is broader (fresh-clone install works, all eval
targets met, user sign-off). Consistent with `docs/NARRATIVE.md` and
`docs/DEMO/README.md`: **no new money, no new dependencies, no private/manual
annotation gates** — finalize using automated + public-dataset evidence only.

## Deliverable state (SPEC §7 Phase 9)

| # | Deliverable | State (2026-07-09) | Remaining |
|---|---|---|---|
| 1 | `legacy/` removed | ✅ done — no `legacy/` dir exists | — |
| 2 | `README.md` rewrite (install, quickstart, cookbook, accuracy claims w/ eval evidence, GIF demo) | ◑ README exists (root + `tabvision/`) | Fold in the §1.4 / §1.4.1 acoustic-scope accuracy claims **with report links**; add the demo GIF once recorded; AGPL-contagion notice (LICENSES.md) prominent |
| 3 | Confidence-aware ASCII output (color-graded) | ✅ scaffolded (per NARRATIVE) | Verify color-grading path end-to-end on a real clip |
| 4 | `tabvision diagnose input.mov` → HTML report | ◑ command exists (`cli.py` `_cmd_diagnose`), scaffold per NARRATIVE | Verify the HTML report renders overlay + waveform + tab + confidence on a sample clip |
| 5 | `LICENSES.md` finalized + CI check | ◑ `LICENSES.md` current (incl. **D3** export deps 2026-07-09); `scripts/check_default_licenses.py` guards `[project].dependencies` | Expand the check to compare **loaded model artifacts** against the ✅ list (SPEC §7 / LICENSES.md action item); run once all default backends are pinned |
| 6a | `docs/DEMO/` screen recording (30–60 s) | ✗ placeholder (`screen-recording.*` not present) | Record from the existing CLI + a public/fixture clip (no private media) |
| 6b | Per-tier side-by-side GT vs output | ◑ `docs/DEMO/` has architecture-brief, fresh-user-path, release-evidence, sample tab; per-tier dirs are placeholders | Generate one automated example per **acoustic** tier (single-line, strummed); electric tiers are v2 — mark N/A for v1 |
| 6c | One-page architecture brief | ✅ `docs/DEMO/architecture-brief.md` exists | Light refresh to match shipped §3.1 + fusion state |
| 6d | `docs/NARRATIVE.md` — the project story | ◑ **29-line stub** | **Final pass (the D4 headline): "what it's trying to do / what was hard / what worked / what's next"** — expand with the string-ambiguity problem, the audio-ceiling + video-lever findings, the acoustic-scope decision, the fusion-constant work (A5 etc.), and honest limits |
| 7 | Tagged `v1.0.0` release | ◑ `v1.0.0` tag already exists | Re-cut / confirm only after 2–6 land and user signs off |

Legend: ✅ done · ◑ partial · ✗ not started

## Acceptance test (SPEC §7 Phase 9)

1. Fresh clone → `pip install -e .` → `tabvision transcribe sample.mov` works on
   a clean machine (modulo dataset/model downloads). — `docs/DEMO/fresh-user-path.md`
   is the script; **verify it end-to-end**.
2. All eval targets met (§1.4 acoustic scope; §1.4.1 tiers). — cite the current
   `docs/EVAL_REPORTS/` baselines; no regressions.
3. User signs off.

## Suggested sequence

1. **`docs/NARRATIVE.md` final pass** (the D4 headline; free, high-value, no deps).
2. **README accuracy section** — wire §1.4/§1.4.1 claims to the eval reports +
   AGPL notice; this is the portfolio-facing artifact.
3. **Verify runtime deliverables** — diagnose HTML + confidence-graded ASCII +
   fresh-user-path install, on one public/fixture clip each.
4. **Per-tier automated examples** (acoustic single-line + strummed).
5. **Expand the license CI check** to loaded-artifact comparison (LICENSES.md item).
6. **Demo recording**, then **confirm `v1.0.0`** + user sign-off.

Items 1–5 are automated/local and need no spend. Item 6 (recording) and the
final sign-off need the user. **D2 (electric v2) is explicitly out of Phase 9**
— electric tiers are v2 scope.
