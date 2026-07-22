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

---

# Gate A attempt — estimator built and self-validated; data unreachable

User approved the hex download (2026-07-22). It could not be fetched, and the
failure is external to this repo.

## The blocker: zenodo.org is unroutable from this machine

```
DNS   zenodo.org      -> 188.184.98.114   (CERN range, correct)
TCP   zenodo.org:443  -> TimeoutError
TCP   huggingface.co:443 -> OK
TCP   pypi.org:443       -> OK (HTTP 200)
```

`mirdata` fails with `URLError: [WinError 10051] A socket operation was
attempted to an unreachable network`. DNS resolves correctly and other hosts
connect, and the same timeout occurs with the sandbox disabled — so this is
neither a sandbox egress restriction, nor DNS, nor a stale URL. Zenodo is
simply not reachable from this network right now.

**To resume**, fetch out-of-band and unzip into
`$TABVISION_DATA_ROOT/guitarset/audio_hex-pickup_debleeded/`:

- URL: `https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1`
- MD5: `c31d97279464c9a67e640cb9061fb0c6` (from mirdata's index — verify it)

Gate A then runs immediately; nothing else is missing.

## What was built and proven

`scripts/eval/q6_gate_a.py` is complete and **self-validated on synthetic
stiff strings**, so when the audio lands a failure will be attributable to
the signal rather than the code.

The estimator linearises the stiff-string law — `(f_k/k)² = f0² + (f0²·B)·k²`
— so B falls out of a straight-line fit as `slope/intercept`, with the
residual doubling as a quality signal. It runs two passes: the first assumes
a harmonic series, the second re-centres the partial search on the positions
the fitted B predicts.

**A real bug the synthetic test caught.** The first version scaled the
partial search half-width by `k^0.5` on a *relative* tolerance, so the
absolute window grew as `k^1.5`. By k≈10 it had swallowed the neighbouring
partial, the fit locked onto the wrong peaks, and it returned a confidently
wrong answer — f0 recovered as 111.3 Hz for a 110 Hz input, a bias that would
have been invisible on real audio and would have made a Gate A failure
uninterpretable. The window is now capped at `0.4·f0`.

Validated behaviour (`tests/unit/test_q6_gate_a.py`, 7 tests):

| property | result |
|---|---|
| recovers known B at 5e-5, 1e-4, 5e-4 | within 25% relative, f0 within 1% |
| separates a 1.78× B ratio (the 5-fret case) | ≥1.3× measured separation |
| rejects silence / too-short segments | returns `None` |
| candidate enumeration respects tuning + 24-fret bound | passes |

The 25% recovery tolerance is not arbitrary — it is the threshold the
separability precursor identified as sufficient for Gate A (0.9116 predicted
accuracy at 20% error). The estimator meets it on clean synthetic input,
which is the necessary condition; whether it holds on real pickup audio is
exactly what Gate A will measure.

## Also built

An empirical **channel↔string mapping check**: for each measured note it
compares energy across the six hex channels and aborts if the annotated
string's channel is not dominant in >50% of notes. GuitarSet's `data_source`
is documented as low-E→high-E `0..5`, matching v1 `string_idx`, but a silent
channel-order mismatch would produce a plausible-looking Gate A failure, so
it is asserted rather than assumed.

---

# Gates A and B — BOTH PASS

Hex partition acquired over VPN (3.36 GB → 8.8 GB extracted, 361 tracks).
Both gates run on dev players 00–04, leave-one-player-out, **isolated notes
only** — notes with no other gold note sounding during the analysis window,
which is Gate A's stated "isolated-note regime".

## Results

**Gate A — hex pickup** (6,917 notes; channel↔string check 0.9868):

| min r² | coverage | ambiguous acc | count-prior control |
|---:|---:|---:|---:|
| 0.00 | 100% | 0.8234 | 0.6592 |
| 0.50 | 71.9% | **0.8950** | 0.6473 |
| 0.70 | 56.4% | 0.9244 | 0.6483 |
| 0.90 | 27.6% | 0.9669 | 0.6672 |
| 0.95 | 11.6% | 0.9850 | 0.6934 |

**Gate B — mono mic** (6,771 notes):

| min r² | coverage | ambiguous acc | count-prior control |
|---:|---:|---:|---:|
| 0.00 | 100% | 0.8095 | 0.6612 |
| 0.50 | 66.6% | **0.9200** | 0.6512 |
| 0.70 | 50.2% | 0.9580 | 0.6646 |
| 0.90 | 29.1% | 0.9898 | 0.7318 |
| 0.95 | 18.2% | 0.9984 | 0.7470 |

**Gate A: PASS** (0.85 → 0.8950 at 71.9% coverage).
**Gate B: PASS** (0.70 → 0.9200 at 66.6% coverage, and 0.8095 even with no
quality filter at all).

## The control is the point

The count-prior column — the most-frequent string for that pitch, learned
from the same training players — sits **flat at ~0.65 across every arm**.
Inharmonicity beats it by **+0.26 to +0.31 absolute** on exactly the decision
the decoder gets wrong. This is not "isolated notes are easy": the same notes
scored by a context-free prior land at 0.65, which is precisely the 0.6548
the production decoder achieves on the full ambiguous lattice.

`r²` is a genuine confidence signal — accuracy rises monotonically with it,
from 0.81 at full coverage to 0.99 at 29%. It is computed from fit residuals
alone and never touches the label, so it is a legitimate abstention channel:
the estimator can decline the notes it cannot fit.

## The surprise: the mic beats the pickup

Gate B was expected to be the *hard* one — mono-mic adds room, body
resonance and sympathetic ringing, and the two-gate split exists to isolate
that penalty. It does not appear. At matched thresholds the mic **equals or
beats** the hex pickup (0.9200 vs 0.8950 at ~70% coverage; 0.9580 vs 0.9244
at ~50%).

The likely reason is instrumentation, not physics: GuitarSet's hexaphonic
track is a Roland GK-style divided pickup mounted near the bridge —
band-limited, and bridge-proximate pickups capture a weaker, more sharply
rolled-off partial series. B estimation depends entirely on resolving high
partials (the `k²` term), so a full-bandwidth condenser mic is simply a
better instrument for this measurement than a divided pickup.

**Consequence:** the hex partition was necessary to *establish* the mapping
and validate the estimator against known string identity, but it is not
needed for the actual signal. The evidence channel can run on the mono mic
the pipeline already has.

## The scope caveat that governs everything

"Isolated" means no other string sounding in the 120–400 ms window. That is
**34% of solo notes and ~1.3% of comp notes** (measured on a 00_* sample).
This is not a general-purpose string detector — it is a single-line
instrument, exactly as §4.1 scoped it.

That is also why it matters. Q2 closed with context worth +0.0661 on comp but
only **+0.0112 on solo**, while single-line carries **77.5% of
wrong-position loss**. Inharmonicity is per-note and physical, and it lands
on the tier that context could not reach.

## Honest limits

- **Not yet a Tab F1 number.** This measures string classification on gold
  notes with known onsets. Wiring it into `fuse()` as bounded emission
  evidence, on *detected* notes with detected pitches, is a different and
  larger step — and the A14/WS4 precedent is that promising per-note evidence
  can still fail to lift the decoder.
- **Per-player B0 calibration is leave-one-player-out but same-instrument-set.**
  All GuitarSet players use similar acoustic guitars; a personal user's
  instrument would need its own calibration, which §4.1's EM bootstrap
  sketch addresses but this study does not test.
- The estimator abstains on ~30–50% of notes at the useful operating points.
  Coverage is a first-class number for any integration.

## Reproduce

```
cd tabvision && python scripts/eval/q6_gate_a.py \
  --data-home ~/.tabvision/data/guitarset --source hex \
  --json ../docs/EVAL_REPORTS/q6_gate_a_2026-07-22.json
cd tabvision && python scripts/eval/q6_gate_a.py \
  --data-home ~/.tabvision/data/guitarset --source mono \
  --json ../docs/EVAL_REPORTS/q6_gate_b_2026-07-22.json
```
