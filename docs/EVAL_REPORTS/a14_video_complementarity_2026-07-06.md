# A14: cache-only video complementarity probe (GAPS clean-12)

**Date:** 2026-07-06
**Branch:** `v1.1/a14-a10-probes`
**Status:** **CLOSED — decisive negative.** No routed audio/video hybrid exists on
this corpus: the notes video gets right are overwhelmingly notes audio already
gets right. P(video right | audio wrong) = **0.285**, *half* the video marginal
(0.574) — video is anti-enriched exactly where audio fails. The chord axis
(D1's open 0.85 reference) is refuted, not just unproven: on chord-member notes
audio is *stronger* (0.819) and video *weaker* (0.542) than on singletons.
Margin-keyed routing never beats audio-only at any threshold.
**Script:** `tabvision/scripts/eval/a14_video_complementarity_probe.py`
(reproduce command in its docstring; probe self-checks its decode against
`fuse()` on every clip and reproduces the capstone aggregates).
**Companions:** chunk-6 capstone (`v1_1_gaps_chunk6_ws1_2026-06-25.md` §7,
DECISIONS 2026-06-29); D1 packet.

Video mode: **WS1 calibrated**, best fixed orientation per clip (diag convention — video's ceiling); audio: `fuse(events, [], cfg)` on gold-pitch events (capstone convention). vote-frames=1, window=0.06s, conf=0.25.

- Ambiguous notes: **10072**; with CV evidence: **7666**
- Audio prior string accuracy, all ambiguous: **7859/10072 = 0.780** (capstone parity: 0.778)
- On the joined subset (ambiguous ∩ CV): audio **0.797**, video **0.574** (capstone parity: 0.574 calibrated / 0.544 baseline)

## Per-note confusion (ambiguous ∩ CV evidence)

| | video right | video wrong | total |
|---|---:|---:|---:|
| **audio right** | 3954 | 2157 | 6111 |
| **audio wrong** | 443 | 1112 | 1555 |
| **total** | 4397 | 3269 | 7666 |

- P(video right) = **0.574**; P(video right | audio wrong) = **0.285** — no enrichment (complementarity requires the conditional to exceed the marginal)
- Oracle-router ceiling (audio right OR video right): **0.855** (+0.058 over audio-only — the max ANY router could add)

## Singleton vs chord-member split (the D1 chord axis)

Chord member = note decoded in a cluster of ≥ 2 simultaneous events. If video were to beat audio anywhere, the chord-frame hypothesis says it would be here (a chord shape is one static frame).

| subset | notes | audio acc | video acc | audio-wrong ∩ video-right | oracle ceiling |
|---|---:|---:|---:|---:|---:|
| singleton | 4163 | 0.779 | 0.600 | 252 (6.1%) | 0.840 |
| chord member | 3503 | 0.819 | 0.542 | 191 (5.5%) | 0.873 |

## Audio-uncertainty-keyed routing (string-flip margin)

Margin = cost gap (nats) between the decoded state and the cheapest state putting the note on a different string (neighbours fixed) — the B4 trellis confidence. Routing rule: margin < τ → take video.

| margin quartile | notes | audio acc | video acc | video − audio |
|---|---:|---:|---:|---:|
| Q1 [-0.00, 1.00] | 1887 | 0.695 | 0.510 | -0.185 |
| Q2 [1.00, 2.25] | 1887 | 0.821 | 0.557 | -0.264 |
| Q3 [2.25, 7.67] | 1887 | 0.815 | 0.575 | -0.240 |
| Q4 [7.67, 86.73] | 1887 | 0.846 | 0.650 | -0.196 |
| ∞ (no string alternative) | 118 | 0.983 | 0.610 | -0.373 |

| routing threshold τ (nats) | routed→video | accuracy | Δ vs audio-only |
|---:|---:|---:|---:|
| 0.000 | 8 | 0.7968 | -0.0004 |
| 0.500 | 651 | 0.7832 | -0.0140 |
| 1.000 | 1562 | 0.7602 | -0.0369 |
| 2.250 | 3732 | 0.6876 | -0.1096 |
| 7.667 | 5654 | 0.6274 | -0.1697 |
| 17.583 | 6784 | 0.6003 | -0.1968 |

**Best routed accuracy = 0.7972 at τ = 0.000** vs audio-only 0.7972 (Δ = +0.0000). Routing NEVER beats audio-only — the margin does not identify a subpopulation where video is the better source.

## Per-clip breakdown

| clip | ambig | haveCV | audio acc | video acc | both | audio-only | video-only | neither | best orient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 027_Zpswc | 1443 | 1433 | 0.749 | 0.424 | 514 | 557 | 93 | 269 | flip-string |
| 031_vpswc | 827 | 751 | 0.856 | 0.635 | 449 | 195 | 28 | 79 | flip-string |
| 043_bc1wc | 1352 | 708 | 0.753 | 0.719 | 493 | 61 | 16 | 138 | none |
| 063_bV1wc | 824 | 182 | 0.705 | 0.533 | 92 | 16 | 5 | 69 | none |
| 104_xf1wc | 398 | 375 | 0.794 | 0.616 | 191 | 104 | 40 | 40 | none |
| 118_VD1wc | 658 | 622 | 0.932 | 0.878 | 518 | 59 | 28 | 17 | flip-both |
| 142_GD1wc | 663 | 612 | 0.830 | 0.526 | 296 | 215 | 26 | 75 | flip-string |
| 179_pM1wc | 484 | 478 | 0.853 | 0.598 | 249 | 158 | 37 | 34 | none |
| 212_y41wc | 886 | 79 | 0.685 | 0.835 | 64 | 8 | 2 | 5 | flip-string |
| 235_Ny1wc | 1471 | 1471 | 0.686 | 0.417 | 465 | 544 | 149 | 313 | flip-both |
| 294_BSswc | 423 | 423 | 0.915 | 0.584 | 244 | 143 | 3 | 33 | flip-both |
| 341_1M1wc | 643 | 532 | 0.896 | 0.742 | 379 | 97 | 16 | 40 | flip-both |

