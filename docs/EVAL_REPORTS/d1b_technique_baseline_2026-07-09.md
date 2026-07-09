# D1-b — Expressive-markings (technique) baseline on GuitarSet

**Diagnostic baseline, not a gate.** First-ever technique-F1 measurement, to replace the unbaselined `>= 0.70` stretch in SPEC §1.4 with an honest restatement (§0 rule 7).

## Headline

- **Operational technique-detection F1 = `0.00`** (structural, not stochastic). The shipping `highres` backend emits no technique tags (`highres.py` builds every `AudioEvent` with empty `tags`; fusion copies `tags`->`TabEvent.techniques` unchanged). The only tag-emitting path (`basicpitch.py` bend heuristic) is not installed. Zero detections -> zero recall -> F1 = 0.00 against any non-empty gold.
- **GuitarSet cannot baseline hammer-ons / pull-offs.** They are articulation, not pitch, and GuitarSet has no discrete technique labels — only `pitch_contour`. HO/PO need a technique-labelled corpus (Guitar-TECHS, electric -> v2).
- **Bends and slides are derivable as proxies** from `pitch_contour`; their **support** (below) bounds how precise any future technique target could be.

## Proxy support — split `all` (360 tracks, 62476 notes)

| Technique | Proxy source | Count | % of notes | Wilson 95% half-width @ that support |
| --- | --- | ---: | ---: | ---: |
| Bend, clear (>= 1.0 st sustained shift) | `pitch_contour` within-note | 3595 | 5.75% | +/- 0.015 |
| Bend, incl. microbend (>= 0.5 st sustained shift) | `pitch_contour` within-note | 9420 | 15.08% | +/- 0.009 |
| Slide (legato glide 1-7 st) | `pitch_contour` cross-note | 1831 | 2.93% | +/- 0.021 |
| Hammer-on / pull-off | **not derivable** | n/a | n/a | n/a |

## Proxy support — split `validation` (60 tracks, 8715 notes)

| Technique | Proxy source | Count | % of notes | Wilson 95% half-width @ that support |
| --- | --- | ---: | ---: | ---: |
| Bend, clear (>= 1.0 st sustained shift) | `pitch_contour` within-note | 635 | 7.29% | +/- 0.036 |
| Bend, incl. microbend (>= 0.5 st sustained shift) | `pitch_contour` within-note | 1687 | 19.36% | +/- 0.022 |
| Slide (legato glide 1-7 st) | `pitch_contour` cross-note | 190 | 2.18% | +/- 0.065 |
| Hammer-on / pull-off | **not derivable** | n/a | n/a | n/a |

## Interpretation — honest stretch

On the canonical validation split (player 05) there are ~635 clear-bend + ~190 slide proxies across 8715 notes — enough support for a ~+/- 0.04 F1 CI on bends (slides thinner, +/- 0.06). **Support is not the blocker; label quality is.** The 'gold' is a threshold-sensitive `pitch_contour` heuristic — the bend count nearly triples (7.3% -> 19.4% of notes) between the 1.0-st and 0.5-st thresholds — not human technique annotation, so scoring a detector against it measures agreement-with-a-heuristic, not true technique F1. Combined with a **0.00** detector and **unmeasurable** hammer-ons/pull-offs, a numeric technique target (the old 0.70) is not yet defensible.

**Recommended restatement for SPEC §1.4 / §15:**

1. Baseline (2026-07-09): operational technique-detection F1 = **0.00** — no detector is wired into the default path.
2. GuitarSet baselines **bends & slides only** (proxy, via `pitch_contour`); **hammer-ons/pull-offs are out until a technique-labelled corpus is in scope** (Guitar-TECHS -> v2).
3. First honest milestone = **build any bend/slide detector and beat 0.00** on this proxy. Defer a numeric F1 target (the old 0.70) until a detector exists *and* is measured against **human** technique labels — not this threshold-sensitive `pitch_contour` heuristic, which measures agreement-with-a-heuristic rather than true technique F1.

_Proxy thresholds are documented in the script; counts are threshold-sensitive (two settings shown). The 0.00 baseline is not — it is a structural absence of any detector._
