# Personal position-prior labeller — design (2026-08-02)

**Status: implemented.** SPEC §1.5 carve-out authorized by user directive
2026-08-02; DECISIONS.md entry same date.

## 1. What this is

FretCam re-purposed from a decode-time witness (measured ≈0 three ways) to a
label source for the one lever Track C priced and nobody built: the personal
position prior, in-sample ceiling **+0.0305 [+0.0183, +0.0430]** aggregate
Tab F1 (`docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.md`).

The camera's measured profile — a position window on 27% of stable frames,
correct on 100% of them — is exactly a labeller's requirement (precision,
not coverage) and exactly not an evidence channel's (the decoder's retained
paths are the same tab; even a gold-window oracle gains +0.000087,
`segment_window_stage1_2026-07-29.md`).

## 2. Mechanism

```
recording ──► audio backend ──► pitches (conf ≥ 0.5)
     │                                   │  join: nearest locked window
     └──► FretCam analyzer ──► windows ──┘  within 0.25 s, conf ≥ 0.5
                                         │
                     exactly one playable (string, fret) consistent?
                                         │ yes (else abstain)
                                         ▼
                            STORE.jsonl  (accumulates across sessions)
                                         │ scripts/train/build_personal_prior.py
                                         ▼
                            personal.json (schema-1 counts artifact)
                                         │ --position-prior personal.json
                                         ▼
                            the shipped prior machinery, unchanged
```

**Consistency rule:** for the event's pitch, a candidate `(string, fret)` is
consistent when `fret` is inside the window, or `fret == 0` — an open string
is playable from any hand position. A label is emitted only when exactly one
candidate is consistent; any open/fretted ambiguity abstains. Precision
therefore inherits from window precision × audio pitch precision instead of
diluting either.

## 3. Pieces

| piece | where |
|---|---|
| harvest + store + builder library | `tabvision/fusion/personal_prior.py` |
| CLI harvest flag | `tabvision transcribe --harvest-personal-labels STORE.jsonl` (requires `--video-backend fretcam`, capo 0) |
| artifact builder | `python -m scripts.train.build_personal_prior STORE.jsonl -o personal.json` |
| consumption | `--position-prior personal.json` — new policy branch, file sha256 in the policy's artifact identities |
| observations plumbing | additive `PipelineArtifacts.position_observations` |

## 4. Frozen-by-honesty constants (never swept)

- confidence floors 0.5 on both channels (Track C's harvest floor);
- onset↔window gap 0.25 s; `state == "locked"` only;
- `min_labels_per_pitch = 5` for per-pitch switching (personal counts
  replace population counts per pitch; no blend weight exists to tune).

## 5. Posture guarantees

Label stores and personal artifacts are local. Never shipped, never
registered as defaults, never in eval corpora or published figures, never a
substrate for other artifacts. Sequence prior is forced off with a personal
position prior (no validated pairing). Harvest refuses capo sessions (the
artifact is capo-0 indexed; covariant re-indexing happens at load time).

## 6. What would prove it works

Not an eval run — the user's own recordings are banned from eval roles even
under the carve-out. The honest signal is the assisted-review queue: a
working personal prior should reduce the wrong-position correction rate on
the user's own sessions over time. Secondary sanity: `personalized_pitches`
in the artifact should grow with sessions, and labels should concentrate
where the user actually plays.
