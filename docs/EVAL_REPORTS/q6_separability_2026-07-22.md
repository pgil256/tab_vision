# Q6 inharmonicity — separability precursor (no download spent)

Accuracy-loop iteration 6 (ROI deep-dive §4.1). Gate A as written needs
GuitarSet's **hexaphonic** partition, which is not on disk: the repo's
acquirer takes `annotations` + `audio_mic` only and explicitly *"Skip[s] the
multi-GB hex-pickup + mix partitions"* (`scripts/acquire/datasets.py`).
Mono-mic alone is 1.6 GB, so hex is a comparable multi-GB fetch — a dataset
download, which stops for the user.

Rather than stall, this answers the question that decides whether the
download is worth it, from data already banked.

## The question

Inharmonicity can only resolve string identity if two candidate positions for
the *same pitch* actually have different B. For a stiff string
`f_k = k·f0·√(1 + B·k²)` with `B ∝ 1/L²`, and fretting shortens the speaking
length by `2^(−n/12)`, so

> **B(s, n) = B0_s · 2^(n/6)**

Two candidates for one pitch differ in log B by
`(log B0_s1 − log B0_s2) + (n1 − n2)/6 · log 2`. The **second term is
assumption-free** — it needs only the fret difference the banked lattice
already records. The first term (plain vs wound construction) is genuinely
large in practice and is **ignored here**, so every number below is a
*lower bound* on discriminability.

## Result — the candidates are far apart

Fret gap between the rank-1 and rank-2 candidates, dev-OOF ambiguous notes
(n = 35,959):

| fret gap | notes | share | B ratio from length alone |
|---:|---:|---:|---:|
| 4 | 10,440 | 29.0% | **1.59×** |
| 5 | 25,326 | 70.4% | **1.78×** |
| 9 | 180 | 0.5% | 2.83× |
| 10 | 10 | 0.0% | 3.17× |
| 14 | 3 | 0.0% | 5.04× |

**Every ambiguous pair differs by at least 4 frets**, and 99.4% by 4 or 5.
That is the geometry of the instrument: the same pitch on adjacent strings is
5 frets apart (4 across the G–B pair), which is exactly the span that makes
the length term large. There is no cluster of near-degenerate pairs to worry
about.

Treating the rank-1/rank-2 decision as a Gaussian mean-separation problem
with relative error `sigma` on the B estimate, accuracy is
`Phi(dlogB / (2·sigma))`:

| B-estimator relative error | lower-bound string accuracy | vs Gate A (0.85) |
|---:|---:|---|
| 2% | 1.0000 | clears |
| 5% | 1.0000 | clears |
| 10% | 0.9956 | clears |
| 20% | 0.9116 | clears |
| 30% | 0.8175 | misses |

**Gate A is clearable if B can be estimated to better than roughly 25%
relative error** — and this is the pessimistic reading, since per-string B0
differences (plain vs wound) add separation on top.

## What this does and does not establish

**Does:** the route is not blocked by degeneracy. The deep-dive's worry that
same-pitch candidates might be physically indistinguishable is answered — they
are 1.6–1.8× apart in B before construction differences.

**Does not:** say anything about whether B is *estimable* to 25% on real
audio. That is the entire risk, and it is where the two gates were pointed:
Gate A (hex, isolated notes) asks whether the estimator works at all; Gate B
(mono-mic, single-line) asks whether it survives bleed, reverb and
polyphony. Literature is encouraging on isolated notes — Hjerrild &
Christensen (ICASSP 2019) report 1.5% string+fret error with per-instrument
calibration — but nothing published survives dense polyphony, which is why
§4.1 scoped this to single-line segments.

## Why this matters more after Q2

Q2 closed with context helping **comp +0.0661 but solo only +0.0112**, while
single-line carries 77.5% of wrong-position loss. Sequence context is
exhausted as a single-line lever. Inharmonicity is per-note and physical — it
does not care whether the note is in a chord or a melody — so it is the one
remaining evidence channel aimed at the tier that still needs it.

## Reproduce

```
cd tabvision && python scripts/eval/q6_separability_precursor.py \
  --lattice ../docs/EVAL_REPORTS/string_assignment_phase0_2026-07-15_notes.csv \
  --json ../docs/EVAL_REPORTS/q6_separability_2026-07-22.json
```

## Blocked on

GuitarSet hexaphonic partition (multi-GB, CC-BY-4.0, already-approved
dataset). Acquire with a `partial_download` of the hex-pickup partition via
the existing mirdata path in `scripts/acquire/datasets.py`.
