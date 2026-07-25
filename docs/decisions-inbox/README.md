# Decisions inbox

`docs/DECISIONS.md` is append-only and every track wants to append to its tail.
That guarantees a merge conflict per track, and those conflicts are pure noise —
two additions that never disagree, that git cannot order for us. Two of them
happened on 2026-07-25 alone.

So during a parallel program, tracks **do not edit `DECISIONS.md`**. Each track
writes its entries here, one file per track:

```
docs/decisions-inbox/<track>.md
```

The integrator appends them to `docs/DECISIONS.md` in merge order and deletes
the inbox file in the same commit. The decision log ends up identical to what it
would have been; only the conflicts disappear.

## Format

Same as `DECISIONS.md` — one or more entries in the format its header defines:

```
## YYYY-MM-DD — <short title>
**Phase:** <track id and name>
**Decision tree:** <what was being decided>
**Branch taken:** <what was chosen>
**Evidence:** <metric values, report paths>
**Reasoning:** <one paragraph>
```

Separate multiple entries with `---` exactly as the main log does.

## Rules

- One file per track; never edit another track's file.
- Write the entry when the decision is made, not at merge time — the reasoning
  is perishable and reconstructing it later produces a worse record.
- A banked negative gets an entry too. The refutations are as much of this
  project's evidence as the wins, and a track that closes negative has still
  produced its deliverable.
