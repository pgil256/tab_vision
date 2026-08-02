# TabCNN complementarity experiment — final report

**Decision: `do_not_integrate` for both model families.** Neither DAFx-24
GuitarProFX TabCNN nor SynthTab TabCNN x4 is evidence-positive under the
protocol frozen on 2026-07-29. No model, dependency, artifact, inference
route, or default was registered or shipped.

The sealed packet contains 932 unique model/clip rows: 466 clips for each
family across GuitarSet development (300), GuitarSet sealed player 04 (60),
GAPS clean-12 (12), EGSet12 (12), and Guitar-TECHS (82). All 932 rows preserve
onset and pitch exactly, and all 932 repeat posteriors are deterministic.

## Technical summary

DAFx looks strong only when its development-overlapped GuitarSet results and
published-benchmark reproduction are mixed into the descriptive aggregate:
`+0.0658` macro Tab F1. The only eligible acoustic/classical confirmation set
for that family is GAPS, where the gain is just `+0.0052` with a 2.06%
wrong-position reduction. Both miss the predeclared `+0.020` and 10% gates.
The executed ONNX export also lacks verified numerical equivalence to the
unavailable official checkpoint, so provenance independently blocks a
positive result.

SynthTab is cleanly evaluated but not useful enough. Its eligible GAPS plus
sealed-GuitarSet pool gains `+0.0048` with a 95% interval crossing zero; its
solo gain is `+0.0028`, and solo wrong-position errors fall only 1.09%.
Projected 60-second CPU latency rises from 262.495 s to 324.646 s, exceeding
both the five-minute limit and the 20% added-CPU checkpoint.

## Key findings

| family | eligible pool | clips | current | current + TabCNN | delta (95% CI) | solo delta | solo wrong-position reduction | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DAFx-24 GuitarProFX | GAPS | 12 | 0.7670 | 0.7722 | +0.0052 [+0.0012, +0.0089] | +0.0052 | 2.06% | `do_not_integrate` |
| SynthTab x4 | GAPS + GuitarSet sealed | 72 | 0.6786 | 0.6834 | +0.0048 [-0.0046, +0.0156] | +0.0028 | 1.09% | `do_not_integrate` |

The important contrast is overlap versus transfer. DAFx gains `+0.0690` on
GuitarSet development and `+0.0608` on sealed GuitarSet, but GuitarSet was
used in that family's model development and cannot promote it. EGSet12 is a
published-family reproduction rather than an independent confirmation.
Guitar-TECHS is independent electric transfer: DAFx gains `+0.0670`, but its
result does not replace the eligible acoustic/classical check. Its 9.12%
wrong-position reduction is descriptive; the 10% gate is evaluated on the
eligible solo population, where DAFx reaches only 2.06%.

## Corpus results

Macro Tab F1 uses the repository's 50 ms matching rule. Intervals are paired
10,000-sample percentile bootstrap intervals with seed 42.

| family | corpus role | clips | current | current + TabCNN | delta (95% CI) | wrong-position reduction |
|---|---|---:|---:|---:|---:|---:|
| DAFx | GuitarSet dev — overlapped development | 300 | 0.6801 | 0.7491 | +0.0690 [+0.0586, +0.0801] | 40.94% |
| DAFx | GuitarSet sealed — overlapped confirmation | 60 | 0.6609 | 0.7217 | +0.0608 [+0.0431, +0.0801] | 44.59% |
| DAFx | GAPS — eligible independent confirmation | 12 | 0.7670 | 0.7722 | +0.0052 [+0.0012, +0.0089] | 2.06% |
| DAFx | EGSet12 — reproduction | 12 | 0.4842 | 0.5488 | +0.0645 [+0.0206, +0.1249] | 11.22% |
| DAFx | Guitar-TECHS — independent electric transfer | 82 | 0.3120 | 0.3790 | +0.0670 [+0.0493, +0.0867] | 9.12% |
| SynthTab | GuitarSet dev — development | 300 | 0.6801 | 0.6911 | +0.0110 [+0.0047, +0.0179] | 7.68% |
| SynthTab | GuitarSet sealed — eligible confirmation | 60 | 0.6609 | 0.6661 | +0.0052 [-0.0063, +0.0181] | 10.19% |
| SynthTab | GAPS — eligible independent confirmation | 12 | 0.7670 | 0.7699 | +0.0029 [-0.0001, +0.0065] | 1.06% |
| SynthTab | EGSet12 — electric transfer | 12 | 0.4842 | 0.5063 | +0.0220 [+0.0025, +0.0457] | 2.20% |
| SynthTab | Guitar-TECHS — independent electric transfer | 82 | 0.3120 | 0.3188 | +0.0068 [+0.0035, +0.0105] | 1.14% |

Across all 466 clips, which is descriptive rather than the promotion pool,
DAFx moves 0.6100 to 0.6759 (`+0.0658` [0.0575, 0.0743]) and SynthTab moves
0.6100 to 0.6196 (`+0.0096` [0.0052, 0.0143]). The corresponding aggregate
wrong-position reductions are 26.66% and 5.02%.

The diagnostic `posterior_only` arm excludes the current position, sequence,
and physics evidence and is not promotion-eligible:

| family | all | GuitarSet dev | GuitarSet sealed | GAPS | EGSet12 | Guitar-TECHS |
|---|---:|---:|---:|---:|---:|---:|
| DAFx | 0.6563 | 0.7194 | 0.6840 | 0.7001 | 0.5773 | 0.4102 |
| SynthTab | 0.4458 | 0.4628 | 0.4454 | 0.7699 | 0.5065 | 0.3278 |

Because onset and pitch are banked, their all-corpus scores are identical in
`current` and `current_plus_tabcnn`: macro onset F1 0.9161 (micro 0.9117) and
macro pitch F1 0.8991 (micro 0.8930), with an exact paired delta of zero for
both models.

## Complementarity diagnosis

| family | coverage | P(TabCNN correct) | P(TabCNN correct \| current wrong-position) | oracle ceiling | current-only events | TabCNN-only events |
|---|---:|---:|---:|---:|---:|---:|
| DAFx | 95.95% | 78.51% | 71.22% | 90.86% | 6,599 | 17,975 |
| SynthTab | 95.95% | 57.58% | 36.53% | 79.85% | 14,483 | 9,221 |

DAFx has real position information, but this experiment's decision is about
eligible confirmation, not whether the model can ever assign strings. Its
independent acoustic gain is an order of magnitude below the gate. SynthTab's
oracle headroom is also real, but its fixed `alpha = 0.35` evidence does not
convert enough of that headroom into corrected tabs.

## Promotion-gate audit

| frozen check | DAFx | SynthTab |
|---|---|---|
| eligible delta >= +0.020 | FAIL (+0.0052) | FAIL (+0.0048) |
| eligible paired 95% lower bound > 0 | PASS (+0.0012) | FAIL (-0.0046) |
| solo delta >= +0.030 | FAIL (+0.0052) | FAIL (+0.0028) |
| solo wrong-position reduction >= 10% | FAIL (2.06%) | FAIL (1.09%) |
| comp/strummed delta >= -0.005 | PASS (+0.1044) | PASS (+0.0076) |
| EGSet12 and Guitar-TECHS each >= -0.005 | PASS | PASS |
| every tier/player group with >= 10 clips >= -0.020 | PASS | PASS |
| unsupported positions neutral | PASS | PASS |
| onset/pitch exactly invariant | PASS | PASS |
| repeat posterior deterministic | PASS | PASS |
| provenance verified | **FAIL** | PASS |
| evaluation license acceptable | PASS | PASS |
| added CPU <= 20% of current | PASS (1.34%) | **FAIL (23.68%)** |
| total 60-second CPU < 300 s | PASS (266.011 s) | **FAIL (324.646 s)** |

DAFx status is `blocked_protocol_evidence`, with the explicit blocker
`dafx_official_checkpoint_to_onnx_equivalence_not_verified`. SynthTab status
is `evaluated`; it needs no provenance caveat to fail the accuracy and
performance gates. Passing checks do not offset a failed check because the
protocol requires all checks to pass.

## Scope, data, and metric definitions

The experiment changes only string/fret evidence. Banked highres events are
the sole onset/pitch source; TabCNN may populate only `AudioEvent.fret_prior`.
It cannot add, remove, retime, repitch, split, or merge events. Candidate
evidence is a likelihood ratio over playable same-pitch string/fret positions,
combined through the existing product-of-experts helper at frozen exponent
0.35. Frets above the supported 0–19 range are neutral, and structurally
uninformative events abstain.

The three frozen arms are `current`, diagnostic `posterior_only`, and
`current_plus_tabcnn`. The promotion statistic is paired per-clip macro Tab
F1. A true positive requires string, fret, and onset within 50 ms. The
wrong-position statistic counts events where onset and pitch are right but
the playable position is wrong.

## Method and evidence integrity

The development run passed its unlock before any sealed candidate score was
computed. Scoring then ran once in the frozen order: GuitarSet sealed, GAPS,
EGSet12, and Guitar-TECHS. All five corpus manifests are complete. The event
bank ledgers, player-05 q6 reproduction, and Guitar-TECHS source revision are
verified. Posterior caches are content-addressed and repeat-run checked.

Frozen identities at scoring time:

- protocol SHA-256: `7d7aa1dd080f68e1672df4834bb8a874cd5ad3726e22bb45bea777bbf91a94d5`
- scoring code SHA-256: `f6b6f476dac5b33dc72613c8af27c07b3553dd44dd267297e94e59ee2fb17e8f`
- posterior code SHA-256: `f326875670b5cf4f56ace26b395e2f494e2d5b8a683ece277f20ad0e7ce6154c`
- score-time `LICENSES.md` SHA-256: `4512e606f39cc5fb9b12bf6825940573c88f719071560e1aea4fe37c835b18f8`
- Git revision recorded by the packet: `febd38c2d57c6409a1451e8b8ac5ffc958ea45a9`

The score-time license hash intentionally authenticates the frozen pre-run
document. This dated result update changes the working document afterward and
does not retroactively alter the scored packet.

## Performance

The current path's frozen cold baseline is 262.495 CPU s per 60 s of audio.
DAFx adds 3.516 s (1.34%) for a projected 266.011 s total. SynthTab adds
62.151 s (23.68%) for 324.646 s total. Model-load times are 0.073 s and
3.578 s, respectively. These are explicit projections because evaluation
consumes banked highres events; they combine the frozen current full-pipeline
measurement with candidate CQT, inference, mapping, and incremental decode.

SynthTab's cache job was raised from the default scheduler priority to
`nice -5` partway through the long cache run at the user's request. The
posterior bytes and decisions are deterministic; scheduler priority can
affect observed wall time, so the recorded performance receipt should be
read as the measured high-priority run, not as a hardware-independent
benchmark. Even with that favorable scheduling, SynthTab fails both latency
checks.

## Limitations and uncertainty

- DAFx GuitarSet results are development-overlapped and cannot support
  promotion; its large aggregate is therefore descriptive, not decisive.
- EGSet12 reproduces the DAFx family on a published benchmark and is not an
  independent confirmation for that family.
- Guitar-TECHS is the independent electric set, but it does not replace the
  predeclared acoustic/classical promotion pool.
- DAFx executed a hash-pinned ONNX transport whose numerical equivalence to
  the unavailable official checkpoint could not be verified.
- SynthTab's published weight licensing remains ambiguous beyond the approved
  personal, non-commercial evaluation posture; redistribution is not cleared.
- The current-path latency is an external frozen baseline rather than a
  simultaneous end-to-end run, because this experiment intentionally reuses
  banked onset/pitch events.
- Twelve-clip GAPS and EGSet12 intervals are necessarily wide. The decision
  does not depend on treating their point estimates as precise.

## Reproducibility packet

- result JSON: `$TABVISION_DATA_ROOT/tabcnn-complementarity/results.json`,
  SHA-256 `b7ccf23e36e6b95ba4eabe0b015271fd533ff5e656c74a7de2f47ef75734246a`
- per-clip CSV: `$TABVISION_DATA_ROOT/tabcnn-complementarity/per-clip.csv`,
  SHA-256 `a4b67aa5a271fa3aeaff58005ce486f2f9254f05fd87e0ddc9cd8238c80ccdda`
- DAFx cache receipt: `posterior-cache-summary-dafx-df87d32d33085d8c.json`,
  SHA-256 `df87d32d33085d8c08dfe697c8f64991707d754817595ebcb32a0c865b303fde`
- SynthTab cache receipt:
  `posterior-cache-summary-synthtab-bc2c67e2a61df48a.json`, SHA-256
  `bc2c67e2a61df48a6037e8047ebd766e2d435bdc4ba488e3abf6617a8f732618`

The JSON holds every per-clip row, all error buckets, source/tier/player
breakdowns, 2x2 complementarity tables, abstentions, runtime package versions,
cache identities, manifests, and gate booleans. The CSV is the compact
recomputation surface.

## Recommended next steps

1. Close this experiment with no integration, registration, packaging,
   routing, or deployment work.
2. Retain the external evidence packet and this report as the banked negative.
3. Do not tune alpha, thresholds, or model selection on the failed sealed and
   transfer corpora. Any future TabCNN attempt needs a new pre-registration
   and a materially different hypothesis.

## Further questions

No follow-up is required for the current TabVision plan. If a future project
needs the DAFx family for a different purpose, first obtain the official
checkpoint and verify export equivalence. If commercial or redistributed use
of SynthTab weights is contemplated, obtain a separate written license
clearance before any technical work.

**Spend:** $0. All computation was local CPU.
