## 2026-07-25 — close three deployment branches as already-merged-by-content

**Phase:** Parallel improvement program, Track E (hygiene)
**Decision tree:** Three branches had sat unmerged for two weeks
(`codex/fix-live-deployment`, `codex/record-live-shutdown`,
`docs/prod-repoint-2026-07-09`). Merge them, or establish that they are not
pending work.
**Branch taken:** Close all three without merging. Their content is already in
`main`, so the merges were noise — an attempt to merge the first produced a
2,700-line `docs/DECISIONS.md` conflict purely because the branch predates every
subsequent entry. Local branches deleted (SHAs recorded below; all three still
exist on `origin`, so deleting the remote copies is left to the user).
**Evidence:** `git cherry -v main codex/record-live-shutdown` reports `3992d88`
as already upstream by patch-id. The other two are upstream by *content* rather
than patch-id, having landed via PR #34 (`6a34c26`, "fix(web): refresh
production and keep landing controls reachable"): `main` contains the
landing-scroll and tooltip-placement fixes in both `web-client/src/App.tsx` and
`index.css`, the full 2026-07-09 production-repoint entry, and the 2026-07-13
Modal-retirement entry. Deleted refs: `codex/fix-live-deployment` @ `43b77a4`,
`codex/record-live-shutdown` @ `3992d88`, `docs/prod-repoint-2026-07-09` @
`87ec076`.
**Reasoning:** A branch whose content has landed by another route is not
outstanding work, and treating it as such generates conflicts that look like
real disagreements. Patch-id equality is the cheap test; when it fails because a
PR squashed or amended the change, the content test is the correct fallback
rather than a merge attempt.

**Incidental finding worth recording:** the retirement entry retires the *old*
`pgil256` Modal workspace, not the deployment. The active `pgilhooley95` app is
unaffected, so the README's claim that the project ships via a Modal production
deploy remains accurate. This was checked because the branch name
("record-live-shutdown") suggested otherwise.

---

## 2026-07-25 — freeze `accuracy-loop-state.md` as historical rather than update it

**Phase:** Parallel improvement program, Track E (hygiene)
**Decision tree:** The accuracy loop's state file is stale and self-
contradicting. Either bring it current, or mark it closed.
**Branch taken:** Freeze it with a prominent header enumerating exactly which
claims are wrong, and point readers at `parallel-program-state.md` and the
Phase 0 report. Do not update the queue table — the program it describes is
closed, and a refreshed table would imply it is live.
**Evidence:** The header names `accuracy/n4-ritual-validation` as the current
branch; the real tip was `accuracy/level-correction-0.60`, four commits later.
The Q6 row still reads "opt-in" though the channel became the default on
2026-07-24, and the Q7 row still requests a capo-routing decision that shipped
the same day. Nothing in the file records the +0.60 level correction being
built, refuted on held-out data, and reverted, nor N1 being confirmed and
shipped.
**Reasoning:** The file's remaining value is its account of what was tried and
why, which is intact and worth keeping. Its queue state is worse than useless
because it reads as current. Marking the boundary precisely is more honest than
a refresh that would blur which parts were contemporaneous. The successor
program keeps its numbers in a generated report and uses prose only for
narrative, so it cannot drift the same way.
