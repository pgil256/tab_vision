# Track C — player adaptation has a real ceiling; within-session self-adaptation is uneven

**Date:** 2026-07-25
**Script:** `tabvision/scripts/eval/c_prior_adaptation.py`
**Data:** `docs/EVAL_REPORTS/c_prior_adaptation_2026-07-25.json`
**Population:** 300 development clips, leave-one-player-out priors, shipped
physics evidence from banked fits. **Sealed player not opened.** Frozen-baseline
drift ±0.0000.

## Verdict

**The player-level lever is real and, uniquely among this program's findings,
directly achievable in production.** Giving a player their own prior is worth
**+0.0305 [+0.0183, +0.0430]**. TabVision is a *personal* application — the same
person records session after session, and the assisted-review queue already
collects their corrections and currently discards them. That is exactly the
ingredient this measurement prices.

**Within-session self-adaptation gains +0.0101 and does not clear its gate.** It
is genuinely session-specific — the mismatched control regresses hard — but it
helps three players and hurts two, and the pre-declared gate required no
regression on players who already do well.

## Results

| Arm | Tab F1 | Δ vs shipped | 95% CI | Verdict |
|---|---:|---:|---|---|
| shipped | 0.6801 | — | — | — |
| **oracle_player** (own in-sample prior) | 0.7106 | **+0.0305** | `[+0.0183, +0.0430]` | PASS |
| oracle_clip (clip's own gold) | 0.8328 | +0.1527 | `[+0.1359, +0.1699]` | PASS |
| self_adapt λ=0.15 | 0.6823 | +0.0022 | `[−0.0010, +0.0054]` | inconclusive |
| self_adapt λ=0.30 | 0.6841 | +0.0040 | `[−0.0000, +0.0080]` | inconclusive |
| self_adapt λ=0.50 | 0.6863 | +0.0062 | `[+0.0014, +0.0111]` | PASS |
| self_adapt λ=0.75 | 0.6888 | +0.0087 | `[+0.0031, +0.0143]` | PASS |
| self_adapt λ=1.00 | 0.6902 | +0.0101 | `[+0.0036, +0.0166]` | PASS |
| **mismatched λ=0.50** (control) | 0.6551 | **−0.0250** | `[−0.0334, −0.0171]` | **regression** |

Harvest rate at confidence ≥ 0.5: 91.6% of decoded notes.

## The control is the load-bearing arm

The self-adaptive prior is learned from the decoder's own confident output, and
at a 92% harvest rate that prior is close to a restatement of the decode.
Blending it back in and re-decoding could therefore be pure self-confirmation —
re-deciding in favour of what you already decided — which would inflate Tab F1
without any session knowledge being involved.

The mismatched arm settles it. Blending in **another clip's** session prior at
the identical weight gives **−0.0250 [−0.0334, −0.0171]**, a decisive
regression, against **+0.0062** for the matched prior at the same weight. The
mechanism is specific to the session it came from. Sharpening alone does not
explain it.

That also reads the other way, and is worth stating: **a wrong session prior is
about four times as harmful as a right one is helpful.** Any production
adaptation must be certain it is adapting to the right session.

## Why λ=1.00 is not the degenerate arm it looks like

λ=1.00 discards the population prior for the second pass and scores best. That
looks like evidence the population prior is worthless, which would contradict
Phase 0 directly.

It is not, because **pass 1 used the population prior**. The session prior is
learned from a decode the population prior produced, so its information is baked
in rather than discarded. λ=1.00 means "trust the consistent answer you already
reached", not "ignore the prior". The monotone trend from 0.15 to 1.00 is
therefore a statement about intra-session consistency: a player stays in one
region of the neck within a piece, the first pass already reflects that, and
re-decoding against it propagates the confident notes' evidence to the
uncertain ones.

## Per-player — where the gate fails

| Player | shipped | oracle_player | self_adapt λ=1.00 | Δ self-adapt |
|---|---:|---:|---:|---:|
| 00 | 0.6896 | 0.6906 | 0.7167 | **+0.0271** |
| 01 | 0.5945 | 0.6707 | 0.6161 | **+0.0216** |
| 02 | 0.6718 | 0.7218 | 0.6679 | **−0.0039** |
| 03 | 0.7098 | 0.7168 | 0.7215 | +0.0117 |
| 05 | 0.7346 | 0.7529 | 0.7286 | **−0.0060** |

Self-adaptation helps 00, 01 and 03 and **hurts 02 and 05** — including 05, the
strongest player. The pre-declared gate required no regression on players who
already do well, so despite a CI-significant aggregate gain this arm **does not
pass** as a default. It is a candidate for an opt-in or a
confidence-gated variant, not for promotion.

The `oracle_player` column tells a cleaner story: it is positive for every
player, and largest exactly where the shipped system is weakest (player 01,
**+0.0762**, from the worst baseline of the five). Player-level knowledge helps
the players who need it most, which is the opposite of the self-adaptive arm's
shape.

Note also that player 00's own prior is worth only **+0.0010** — that player is
essentially the population average. An earlier 10-clip smoke run of this probe
covered only player 00 and consequently showed the player oracle at *−0.0084*,
which is why the full five-player run matters and why a single-player pilot
would have closed this track wrongly.

## How different are players, really?

Mean total-variation distance between per-player in-sample priors, averaged over
shared pitches:

| | 01 | 02 | 03 | 05 |
|---|---:|---:|---:|---:|
| **00** | 0.190 | 0.281 | 0.131 | 0.164 |
| **01** | — | 0.177 | 0.179 | 0.174 |
| **02** | — | — | 0.250 | 0.280 |
| **03** | — | — | — | 0.142 |

Mean across pairs **0.197** — players genuinely differ in where they play the
same pitch, but not enormously. That magnitude is consistent with the +0.0305
ceiling: real, worth having, not transformative.

## Decision

**Sub-item (ii) — build a personal prior from accumulated user data — is
justified and priced at +0.0305.** It is the only lever in this program whose
production form is *easier* than its experimental form: a personal application
sees the same player repeatedly, and the assisted-review queue is already
collecting exactly the confirmed (string, fret) labels the prior is built from.
That work is not in this track's scope and needs its own design, but it now has
a measured ceiling instead of an assumption.

**Sub-item (i) — within-session self-adaptation — is not promoted.** Real,
session-specific, +0.0101 aggregate, but it regresses two of five players
including the strongest, and the gate said no.

Shipped default unchanged. No artifact registered.

## Limits

- Development only; the sealed player was not opened.
- `oracle_player` is in-sample: it is the ceiling for a *perfectly* estimated
  personal prior, not what any finite amount of accumulated user data would
  reach. A real personal prior built from a handful of sessions would land
  somewhere below it, and how far below is unmeasured.
- The confidence floor (0.5) and blend weights were swept coarsely, and the
  monotone λ trend means the best setting may lie beyond 1.00 in some other
  parameterisation — an iterated scheme, for instance. Not explored.
- The per-player regressions are small and the sample is five players. The gate
  is failed on a strict reading; a larger player pool might change the shape.
