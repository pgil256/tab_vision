# Gold-tab sessions → local video-training corpus — design (2026-08-02)

**Status: implemented.** Second SPEC §1.5 widening, user directive
2026-08-02; DECISIONS.md entry same date. Strictly local-only.

## 1. What this is

The user records a practice take in good conditions and supplies the exact
tab they played. The ingest turns that pair into two growing local assets:

1. **ground-truth labels** for the personal position prior (better than the
   FretCam window harvest: every note, no camera in the loop), and
2. **a labelled frame corpus** — JPEG frames around every note onset, each
   carrying gold `(string, fret)` — which is precisely the "labelled corpus
   at real resolution" that the banked video string-resolution negative
   named as its blocker.

No model is trained by this change. The corpus accumulates option value
from sessions the user records anyway; any future training run must be
pre-registered with a selection-bias control first.

```
take.mp4 ──► demux ──► audio backend ──► pitch sequence ─┐
                                                          │ Needleman–Wunsch,
take.tab.json ──► load_gold_tab ──► gold pitch sequence ─┤ exact matches only
                                                          ▼
                                            onset-stamped gold notes
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                          frames @ onset +40/120/200 ms        --prior-store append
                          <corpus>/<stem>-<hash8>/             (source "gold-tab")
                            frames/*.jpg + rows.jsonl
```

## 2. Pieces

| piece | where |
|---|---|
| tab format + validation | `tabvision/personal/gold_tab.py` — `{"notes": [{"string": 1-6 (tab conv., 1=high E), "fret": n, "pitch_midi"?: p}]}`; pitch cross-check catches string off-by-ones |
| pitch alignment | `tabvision/personal/alignment.py` — NW over pitch sequences; exact matches only; insertions/deletions absorbed |
| frame corpus | `tabvision/personal/video_corpus.py` — single pass over the demuxer's frames; nearest frame within 0.05 s per instant; no interpolation |
| CLI | `python -m scripts.train.ingest_gold_session take.mp4 take.tab.json [--prior-store …]` |

## 3. Honesty gates

- `--min-match 0.7`: a take that diverges from its tab is **refused whole**,
  not salvaged. Gold data is all-or-nothing.
- Wrong pitch never matches; missing frames never interpolate.
- Capo refused; standard tuning assumed; session dir keyed by content hash
  (idempotent re-ingest). The `--prior-store` append is not deduplicated —
  first ingest only.

## 4. Posture

Everything written lives under the user's data root. Never shipped, never a
default, never in eval corpora or published figures; usable as training
substrate for **local-only** video-analysis artifacts per the widened §1.5
carve-out. The natural first consumer, when volume exists: a personal
string resolver at real resolution, evaluated on camera-unlocked frames to
control the selection bias the FretCam-window harvest could never escape.
