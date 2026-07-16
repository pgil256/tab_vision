# Sequential Tab F1 Phase 7: integrated verification and program closure

## Outcome

**Phase 7 passes the bounded-negative completion branch.** Every predeclared
automatic branch has a reproducible decision, and none passed the cumulative
automatic promotion guardrails. Production `auto` therefore remains on the
last validated behavior: the accepted GuitarSet position/sequence pair only
for supported clean-acoustic sessions, the single-checkpoint GAPS/high-resolution
audio backend, `string_evidence=none`, and `assignment_decoder=baseline`.

The Phase 3 two-checkpoint `confidence_winner` ensemble remains registered only
as an explicit clean-acoustic evaluation backend. It passed its narrower
ensemble-development gate, but its frozen player-05 aggregate lift was
`+0.0213`, solo lift was `+0.0085`, and same-pitch wrong-position reduction was
`5.3%`; these miss the broader automatic guardrails. No new automatic decoder,
assisted UI, electric model, video route, score-informed mode, or hardware mode
was enabled or deployed.

## Clean replay identity

- Evaluation source commit: `f30546096f987cfcef986bb746b39c95e38186b9`.
- Checkout: detached clean worktrees; mutating Phase 2 and Phase 3 evaluators
  ran in disposable worktrees so their rejected/generated artifacts could not
  alter the verification checkout.
- Platform: Windows 11 Pro `10.0.26200`, Intel i7-1185G7, 8 logical CPUs,
  34,048,368,640 bytes visible memory.
- Python `3.12.10`, NumPy `2.4.6`, PyTorch `2.12.0+cpu`, mir_eval `0.8.2`,
  hf-midi-transcription `0.1.1`.
- GuitarSet split: players 00-04 development OOF (300 tracks); player 05
  frozen confirmation (60 tracks). Development annotation aggregate SHA-256:
  `ae67ab869c542c9b22586464f78e555beb796b4016787391228b91dd5ce14d71`.
- The regenerated Phase 0 and Phase 1 stable note tables were byte-identical
  to the frozen inputs at SHA-256
  `6f06758525f0908be82b1eeb5d74731f1ee647fba74a5c5111e8bf44e231a530`
  and `541220a6edd0341fa1799d92313a180a8e7a20139be1c93dc5b52a6d94c9b3e1`.

## Sequential branch replay

| phase | bounded result reproduced | decisive metric / hash | automatic action |
|---|---|---|---|
| 0 | baseline and oracle packet pass | player-05 Tab F1 `0.6126`; four-second joint oracle `+0.1446` | proceed to bounded sequence work |
| 1 | close rule-based segment decoding | dev `+0.0004`, player-05 `+0.0017`; prediction `9788d929bea9a7ca414050f8de10370a352d4fe848ed8baedb053f66cdb5d7ef` | keep `auto=baseline` |
| 2 | close symbolic-context expansion | dev `+0.0036`, ambiguous top-1 `+0.0056`; prediction/rerun `d3ab8c8a96302e6e978374815c5e6a4caf3dcb3b50fa1ffde04a03565ef84109` | unregistered diagnostic only |
| 3 | posterior lattice fails; explicit ensemble gate passes; cumulative automatic gate fails | dev `+0.0436`, player-05 `+0.0213`; onset/pitch `0.9491/0.9403` | explicit `highres-ensemble`; `auto` stays GAPS |
| 4 | close native-rate timbral path | combined top-1 `+0.0072`, CI `[-0.0152,+0.0291]`, worst fold `-0.0315`; prediction `a8fb946ebdf06f7a2f73c543dadd92dfd8c39152434b14ca83d7242a857b57a10` | no artifact/runtime path |
| 5 | close direct per-string model | direct+prior `0.5920` vs required `0.7121`, `0/5` folds improve; prediction `50c0d976b8750e9e6885c4205fe66c27bc2b53ae0e94ce7bb6dbe1518bcc9a14f` | no artifact/runtime path |
| 6 | close assisted review path before UI | AUC `0.7127`, 10% enrichment `1.77x`, 60-second wrong reduction `38.76%`; prediction `d044a80525b4e4dc266ffd9fae40fe053023b6c65db47838c474e145fef486021` | no UI or persistence integration |

The Phase 2 replay reproduced decisions and prediction bytes. Its newly
serialized, unregistered TorchScript ZIP had environment-sensitive container
bytes (`d79cb...`) rather than the checked diagnostic's `9d0df...`; the
registered-artifact tests verified the checked file and the disposable replay
artifact was not copied back or promoted.

## Routing and domain safety

The real Guitar-TECHS diagnostic paired 94 non-technique clips and 9,653
ambiguous notes. Forcing the acoustic prior scored top-1 `0.2027` and top-3
`0.5409`, confirming that it must not route to electric sessions. Automatic
routing resolved as follows:

| session/domain | position | sequence | string evidence | assignment decoder |
|---|---|---|---|---|
| clean acoustic, standard tuning, capo 0 | `guitarset-v1` | `guitarset-seq-v1` | `none` | `baseline` |
| classical / GAPS | `none` | `none` | `none` | `baseline` |
| electric / Guitar-TECHS | `none` | `none` | `none` | `baseline` |
| distorted, capo, or alternate tuning | `none` | `none` | `none` | `baseline` |
| missing/corrupt registered artifact | `none` | `none` | `none` | `baseline` |

Targeted policy, adapter, artifact, registry, and concurrency tests passed 151
core cases and 27 server cases. They include simultaneous acoustic/classical
sessions with different resolved priors and simultaneous baseline/segment
decoder requests; both request-local contexts remained isolated. Server result
metadata retained requested/resolved policy names, versions, and hashes.

## 60-second public CPU benchmark

Source:
`00_Jazz1-200-B_solo_mic.wav`, SHA-256
`d8df477ac8d17da5df76bbfafe94449b9f1c398e120db99d5765deeab59fbfb6`.
The evaluator tiles it deterministically to exactly 60 seconds at 44.1 kHz and
runs the explicit registered two-checkpoint backend sequentially.

| isolated run | wall time | peak working set | events (audio/tab) | output SHA-256 |
|---|---:|---:|---:|---|
| cold model resolution | `258.045 s` | `1,925,480,448` bytes | `188/188` | `1d4ece2570ac73b99f9a825700f6aa2dd1ff9dd2dbaeab73321c012d05c11d5e3` |
| warm cache | `196.001 s` | `1,917,591,552` bytes | `188/188` | `1d4ece2570ac73b99f9a825700f6aa2dd1ff9dd2dbaeab73321c012d05c11d5e3` |

Both runs are below the five-minute laptop-CPU limit and reproduce the frozen
event hash exactly. The committed LF ensemble artifact is 1,111 bytes at
SHA-256 `1caaa87676b0849922fac82c65472ad6a88f09be925b14514b4ed8a5faa6a0f2`.
The older Phase 3 report's CRLF working-copy hash was corrected; parsed JSON and
event output were unchanged.

## Release verification

- v1 package: `845 passed, 12 skipped` from `pytest -v` after the Phase 7
  missing-artifact test was added.
- Frozen server: `296 passed, 3 skipped`.
- Ruff lint: pass; Ruff format: pass; mypy: no issues in 78 source files.
- Default dependency and loaded-artifact license policies: pass.
- Artifact coverage: registered manifest/hash load, unregistered rejection,
  missing file rejection, corrupt hash rejection, TorchScript load/masking,
  unsupported-domain and invalid-artifact neutral fallback: pass.
- Fresh `.[dev]` install: CLI version, dependency/artifact license policy, and
  render smoke pass. Tests that require optional PyTorch now skip during
  collection when the `train` extra is absent.
- Web-client tests/build: not applicable; Phase 6 failed before UI integration
  and changed no web-client files.
- `SPEC.md` and `tabvision/tabvision/types.py` are unchanged from the Phase 0
  evidence commit; no Section 8 contract amendment was required.
- `LICENSES.md` records the Phase 7 no-new-input audit. No paid run, dependency,
  dataset, trained weight, or media redistribution was added.

## Mode separation and rollout disposition

| mode | measured result | disposition |
|---|---|---|
| automatic clean acoustic | production baseline `0.5581` dev / `0.6126` player 05; explicit ensemble `0.6017` / `0.6339` but automatic guardrails fail | do not change `auto` |
| assisted | simulated 60-second review Tab F1 `0.6873`, wrong-position reduction `38.76%` | offline only; no UI |
| electric | acoustic-prior gold-pitch top-1 diagnostic `0.2027`; no electric end-to-end claim | neutral auto; separate future approval |
| video-assisted | no new-signal complementarity probe approved or run | deferred; no metric claim |
| calibration / score-informed | not opened because the Phase 6 offline prerequisite failed | explicit future mode only |
| hardware-assisted | outside blind iPhone transcription and not evaluated | separate capture mode only |

Because no automatic candidate passed, deployment steps that enable a selected
decoder or assisted feature are intentionally inapplicable. The disabled-code,
current-routing, explicit-artifact, public-fixture, fallback, metadata, and
rollback checks passed locally and in the production adapter suite. No live
production configuration was changed. Independent rollback remains available
with `TABVISION_ASSIGNMENT_DECODER=baseline`,
`TABVISION_STRING_EVIDENCE=none`, and the existing position/sequence controls.

## Commands and evidence

The phase commands were rerun with their frozen arguments and external outputs
under `C:\Users\patri\.tabvision\phase7`; concise hashes and exit status are in
`string_assignment_phase7_2026-07-16_provenance.json`. The large stable note
tables remain reproducible and git-ignored. Phase-specific commands are also
preserved in each evaluator's provenance JSON. Known limitations are the
absence of a promoted automatic improvement, the environment-sensitive bytes
of the rejected Phase 2 TorchScript serialization, and the fact that no live
rollout was appropriate after the automatic gate failed.
