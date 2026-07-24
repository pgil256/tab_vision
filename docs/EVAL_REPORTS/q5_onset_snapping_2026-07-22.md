# Q5 onset snapping — pre-fuse onset refinement

20 GuitarSet dev clips (10 comp + 10 solo), offline replay of the banked ensemble events; shipped clean-acoustic decode with the leave-one-player-out position prior + `guitarset-seq-v1` @ w=4.0. Snapping targets half-wave-rectified STFT spectral flux (n_fft=1024, hop=256).

**This changes onsets by construction**, so onset and pitch F1 are reported as first-class gates alongside Tab F1.

| variant | mean \|shift\| | Tab F1 | ΔTab [lo, hi] | onset F1 | Δonset [lo, hi] | pitch F1 | Δpitch [lo, hi] |
|---|---:|---:|---|---:|---|---:|---|
| `baseline` | 0.0 ms | 0.6773 | +0.0000 [+0.0000, +0.0000] | 0.9325 | +0.0000 [+0.0000, +0.0000] | 0.9131 | +0.0000 [+0.0000, +0.0000] |
| `snap-10ms` | 5.4 ms | 0.6775 | +0.0002 [-0.0009, +0.0016] | 0.9320 | -0.0005 [-0.0016, +0.0000] | 0.9136 | +0.0005 [-0.0009, +0.0025] |
| `snap-20ms` | 8.4 ms | 0.6772 | -0.0001 [-0.0024, +0.0025] | 0.9305 | -0.0020 [-0.0050, +0.0011] | 0.9122 | -0.0009 [-0.0042, +0.0026] |
| `snap-30ms` | 9.9 ms | 0.6739 | -0.0034 [-0.0074, +0.0006] | 0.9285 | -0.0040 [-0.0073, -0.0004] | 0.9088 | -0.0043 [-0.0089, +0.0004] |
| `snap-50ms` | 10.8 ms | 0.6726 | -0.0047 [-0.0133, +0.0041] | 0.9228 | -0.0097 [-0.0167, -0.0035] | 0.9038 | -0.0093 [-0.0170, -0.0026] |
| `strum-20ms` | 8.6 ms | 0.6772 | -0.0001 [-0.0024, +0.0025] | 0.9305 | -0.0020 [-0.0050, +0.0011] | 0.9122 | -0.0009 [-0.0042, +0.0026] |
| `strum-30ms` | 10.1 ms | 0.6739 | -0.0034 [-0.0074, +0.0006] | 0.9274 | -0.0051 [-0.0090, -0.0010] | 0.9088 | -0.0043 [-0.0089, +0.0004] |

## Six-bucket decomposition

| variant | correct | wrong_position | pitch_off | timing_only | missed_onset | extra_detection |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 1411 | 443 | 132 | 15 | 196 | 102 |
| `snap-10ms` | 1402 | 443 | 138 | 18 | 196 | 102 |
| `snap-20ms` | 1395 | 439 | 141 | 26 | 196 | 102 |
| `snap-30ms` | 1389 | 439 | 141 | 30 | 198 | 104 |
| `snap-50ms` | 1384 | 434 | 140 | 41 | 198 | 104 |
| `strum-20ms` | 1395 | 438 | 142 | 26 | 196 | 102 |
| `strum-30ms` | 1389 | 439 | 140 | 31 | 198 | 104 |

Bootstrap: paired per-clip deltas, N=10000, seed=42. Acceptance would need lo-95 > 0 on Tab F1 *and* no CI-significant regression on onset or pitch F1.

## Verdict — CLOSED, banked negative

No variant is admissible. `snap-10ms` is a wash (**+0.0002 Tab F1
[-0.0009, +0.0016]**, CI spanning zero) and every larger window loses
monotonically, on Tab F1 *and* on the onset gate:

| variant | mean \|shift\| | ΔTab F1 | Δonset F1 |
|---|---:|---:|---:|
| `snap-10ms` | 5.4 ms | +0.0002 | −0.0005 |
| `snap-20ms` | 8.4 ms | −0.0001 | −0.0020 |
| `snap-30ms` | 9.9 ms | −0.0034 | −0.0040 |
| `snap-50ms` | 10.8 ms | −0.0047 | −0.0097 |

## Why — the backend's onsets are already better than the flux peaks

The decomposition names the mechanism exactly, and it is the opposite of the
intended one:

| variant | correct | timing_only | missed_onset | extra_detection |
|---|---:|---:|---:|---:|
| baseline | **1411** | **15** | 196 | 102 |
| `snap-10ms` | 1402 | 18 | 196 | 102 |
| `snap-20ms` | 1395 | 26 | 196 | 102 |
| `snap-30ms` | 1389 | 30 | 198 | 104 |
| `snap-50ms` | 1384 | **41** | 198 | 104 |

`timing_only` — the bucket snapping exists to drain — **rises monotonically,
15 → 41**, while `correct` **falls, 1411 → 1384**. `missed_onset` and
`extra_detection` barely move. Snapping is not converting near-misses into
hits; it is taking notes that were already inside the 50 ms window and
pushing some of them out.

The ensemble's onsets are, in other words, **already more accurate than
spectral-flux peaks are**. Half-wave-rectified STFT flux is a noisier
estimator of the true attack than a CRNN trained on onset targets, so
"refining" against it can only add variance. The ~5 ms mean shift at a 10 ms
window is the giveaway: the flux peak is essentially already sitting on the
detected onset, which is why that variant is a wash rather than a win.

**The strum variant adds nothing.** `strum-20ms` is numerically identical to
`snap-20ms` and `strum-30ms` is slightly worse on onset F1 (−0.0051 vs
−0.0040). Collapsing cluster members onto a shared median does not help
because the members were not meaningfully dispersed to begin with — the
80 ms grouping already tolerates the physical spread of a strum.

## Relation to the literature claim

The deep-dive sourced this item from "Snapping Matters" (arXiv 2606.11903,
+2.6 note F1) and explicitly hedged it: *"recent preprints, piano-domain;
treat the +2.6 as directional, not transferable."* It does not transfer. The
piano result improves a detector whose onsets are *worse* than the audio
evidence; ours are better than it. That precondition — a transcriber whose
timing is the weak link — does not hold at onset F1 0.9325.

**Do not re-open** without a backend whose onset timing is measurably worse
than spectral flux, or a fundamentally better onset estimator than STFT flux
(the same bar the deep-dive set for re-opening any Tier-3 item).
