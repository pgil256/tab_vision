# Loop prompt — WPF desktop shell build

This file IS the prompt. In each fresh session, paste only:

> Read docs/plans/2026-07-22-wpf-desktop-shell-loop-prompt.md and follow it.
> Execute exactly one loop iteration.

It is stateless and picks up wherever the last session stopped (state lives
in `desktop-client/PROGRESS.md`).

---

Work on the TabVision WPF desktop shell. One increment per session: pick the
smallest next unit of work, finish it, verify it, commit it, stop.

## Read first (in order)

1. `CLAUDE.md` — posture, operating rules, frozen dirs.
2. `docs/plans/2026-07-22-wpf-desktop-shell-plan.md` — the plan. Phases
   D0 → D1 → D1.5. D2 (editor) is OUT OF SCOPE — never start it.
3. `desktop-client/PROGRESS.md` — loop state. If missing, this is the first
   iteration: create `desktop-client/` and `PROGRESS.md` with a checklist of
   every D0–D1.5 deliverable and gate from the plan, all unchecked, then
   proceed to the first item.
4. `docs/DECISIONS.md` entry 2026-07-22 — the rebuild-expected decision.

## Loop body

1. **Locate state.** First unchecked item in `PROGRESS.md`. If a gate item is
   next, run the gate, don't build past it.
2. **Implement** that one item only. Resist scope creep; the shell is
   disposable by design — cheap and thin beats polished.
3. **Verify.** Whatever the item touched must prove itself:
   - Python changes: `cd tabvision && pytest -v && ruff check . && mypy tabvision`.
   - C# changes: `dotnet build` + `dotnet test` in `desktop-client/`.
   - Gate items: run the gate exactly as written in the plan (§4) and record
     the measured result in `PROGRESS.md`, pass or fail.
4. **Record.** Check the item off in `PROGRESS.md` with a one-line result.
   Non-obvious choices → append to `docs/DECISIONS.md` per its format.
5. **Commit** with a `desktop-shell:` prefix. Never commit a failing build.
6. Stop. One increment per session keeps every iteration reviewable.

## Hard rules

- **Zero transcription/ranking logic in C#.** The shell spawns
  `tabvision transcribe` and parses its output. If an item seems to need
  pipeline logic in C#, stop and ask — the plan is wrong or the item is.
- **Python changes are additive only** (`--json`, `--progress` per plan §2).
  No §8 contract changes, no default-behavior changes, no SPEC edits.
- **Frozen dirs stay frozen:** `tabvision-server/`, `tabvision-client/`,
  `web-client/`.
- **New dependencies:** C#-side NuGet packages need a one-line justification
  in `PROGRESS.md`. Python-side deps beyond the plan → stop and ask.
- **Money:** anything that costs money → stop and ask.

## Stop-and-ask conditions

- A gate fails in a way the plan doesn't cover.
- The pinned pipeline commit no longer matches the CLI surface the shell
  expects (rebuild trigger, plan §5) — report the drift; do not silently
  adapt the shell to unpinned pipeline changes.
- All D1.5 items are checked: report completion, remind that D2 stays
  blocked until the web editor stabilizes, and stop the loop.

## Reminder

This shell is expected to be rebuilt as the pipeline develops (DECISIONS.md
2026-07-22). Optimize every choice for cheap rebuild, not longevity.
