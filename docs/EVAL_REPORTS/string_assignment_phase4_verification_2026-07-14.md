# String assignment Phase 4: verification and rollout

Date: 2026-07-14

## Outcome

Phase 4 shipped only the gate-passed domain-aware position/sequence routing.
The timbral ranker remains unregistered, paid training was not started, and
phrase refinement remains disabled because their fixed prerequisite gates
failed. This is a safe partial improvement, not a claim that the program's
strict automatic accuracy objective passed.

## Local verification

All commands ran from commit `661de16` or its source-identical working tree.

| check | result |
|---|---|
| v1 package tests | 748 passed, 12 skipped |
| server tests | 296 passed, 3 skipped |
| Ruff | pass |
| Ruff format check | pass, 215 files already formatted |
| mypy | pass, 67 source files |
| web API-URL guard and production build | pass |
| deterministic smoke, twice and byte-diff | `deterministic=true` |
| wheel build/install/import | pass |
| wheel SHA-256 | `95d9fb92389781acab19b4ad09ed5fe90c276704ccf377af3f9584852aa8775f` |

The installed wheel independently loaded and hash-verified:

- `guitarset-v1`: `71f491cb7d377c163b5d08cbea69ebc7c47783059bf06eb727b9a66e8f7fd003`
- `guitarset-seq-v1`: `3c657db2891f6e22f4fae4c6c9025551b197218c7165779fa8712ce9f40f5e8e`

Regression coverage includes two-through-six-string candidate sets, open and
high-fret alternatives, repeated-pitch-class chords, domain-neutral routes,
invalid and corrupt artifacts, and concurrent acoustic/classical decodes.

## Clean-checkout corpus replay

Commit `661de160c3f10db583b9637ce9a5fd3a7c118346` was mounted as a detached,
clean worktree. The complete cached high-resolution GuitarSet benchmark replayed
all 360 tracks, five player-held-out development folds, and the frozen player-05
confirmation. It reproduced:

- development production-equivalent solo/comp/all: `0.5460 / 0.5702 / 0.5581`
- player-05 production-equivalent solo/comp/all: `0.5418 / 0.6834 / 0.6126`
- phrase anchor oracle: `0.6770 -> 0.7384` (`+0.0614`, prerequisite fail)

The independent Guitar-TECHS replay reproduced 94 clips, 9,653 ambiguous
notes, forced-prior top-1 `0.2027`, top-3 `0.5409`, and neutral electric
routing. No corpus content was copied into the repository.

## Production rollout

- Modal workspace/app: `pgilhooley95/tabvision-api`
- Backend: `https://pgilhooley95--tabvision-api-flask-app.modal.run`
- Vercel deployment: `dpl_FXsvyENE4eGsh6db77yn2Va2yjZJ`
- Production frontend: `https://tabvision.patbuilds.dev`
- Production asset: `index-BaZ3eyic.js`

Post-deploy health returned HTTP 200 and allowed the production origin. The
production bundle contains the replacement API URL and contains neither the
retired `pgil256` URL nor `localhost:5000`. Vercel Preview and Production
`VITE_API_URL` values were pulled independently and both matched the replacement
backend.

Two real jobs used the public repository fixture
`tabvision-server/tests/fixtures/test_a440.mp4`:

| job | declared domain | result | resolved evidence | fallback |
|---|---|---|---|---|
| `ec4fc771-f976-425b-8a8a-3e1628785e5a` | clean acoustic | completed | `guitarset-v1` + `guitarset-seq-v1`; string evidence `none` | false |
| `ea8fd514-2e4c-40b6-9400-9a41d8bb0987` | classical | completed | position `none`; sequence `none`; string evidence `none` | false |

The acoustic result reported both exact artifact hashes above. Modal logs show
normal GPU inference and successful result reads, with no job error or fallback.
The tiny A440 fixture produced zero note events; it is an upload/job/result and
policy-metadata fixture, not an accuracy sample.

## Compatibility and rollback

Result metadata is additive and the legacy `positionPrior` field remains. The
package and server suites cover the legacy list-returning pipeline entrypoint
and documents without the optional policy metadata. Two historical production
job IDs from the prior deployment were no longer present in the persistent job
dictionary (HTTP 404), so a live historical-record read could not be repeated.

Independent rollback controls remain:

```text
TABVISION_POSITION_PRIOR=auto|none|guitarset-v1
TABVISION_SEQUENCE_PRIOR=auto|none|guitarset-seq-v1
TABVISION_STRING_EVIDENCE=auto|none
TABVISION_PHRASE_REFINEMENT=false
```

## Final decision

Keep domain-aware routing enabled. Do not enable timbral evidence or phrase
refinement. The strict automatic `+0.03` improvement and verified correction
path completion conditions were not met, so the broader accuracy objective
remains open for a future evidence-backed approach.
