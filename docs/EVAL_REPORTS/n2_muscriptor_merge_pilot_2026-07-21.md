# N2 MuScriptor merge-variant pilot — GuitarSet dev (solo + comp)

Model: muscriptor-medium (isolated venv) vs registered `highres-ensemble` | 20 clips | offline replay of banked events; clean-acoustic decode with the leave-one-player-out position prior + `guitarset-seq-v1` @ w=4.0

## Complementarity by mode — P(MuScriptor right | ensemble wrong)

| mode | clips | gold notes | ensemble wrong | rescued | complementarity | gate ≥ 0.10 |
|---|---:|---:|---:|---:|---:|---|
| solo | 10 | 710 | 81 | 12 | 0.1481 | PASS |
| comp | 10 | 1487 | 165 | 63 | 0.3818 | PASS |
| pooled | 20 | 2197 | 246 | 75 | 0.3049 | PASS |

## Merge variants — shipped decode, paired bootstrap vs ensemble alone

| variant | added notes | of which real | added precision | Tab F1 | Tab P | Tab R | ΔTab F1 [lo-95, hi-95] | onset F1 | Δonset F1 [lo-95, hi-95] | pitch F1 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| `ensemble` | 0 | 0 | — | 0.6773 | 0.6855 | 0.6704 | +0.0000 [+0.0000, +0.0000] | 0.9325 | +0.0000 [+0.0000, +0.0000] | 0.9131 |
| `union` | 675 | 70 | 0.104 | 0.6232 | 0.5871 | 0.6732 | -0.0541 [-0.0989, -0.0160] | 0.8648 | -0.0678 [-0.1061, -0.0339] | 0.8465 |
| `union-dur60` | 671 | 69 | 0.103 | 0.6235 | 0.5880 | 0.6728 | -0.0538 [-0.0987, -0.0156] | 0.8651 | -0.0674 [-0.1059, -0.0335] | 0.8468 |
| `near80` | 326 | 54 | 0.166 | 0.6539 | 0.6403 | 0.6711 | -0.0234 [-0.0551, +0.0038] | 0.8993 | -0.0332 [-0.0490, -0.0181] | 0.8831 |
| `cluster` | 227 | 41 | 0.181 | 0.6606 | 0.6564 | 0.6669 | -0.0167 [-0.0480, +0.0090] | 0.9130 | -0.0195 [-0.0325, -0.0079] | 0.8948 |
| `cluster-dur60` | 227 | 41 | 0.181 | 0.6606 | 0.6564 | 0.6669 | -0.0167 [-0.0480, +0.0090] | 0.9130 | -0.0195 [-0.0325, -0.0079] | 0.8948 |

## Six-bucket decomposition (counts over the same clips)

| variant | correct | wrong_position | pitch_off | timing_only | missed_onset | extra_detection |
|---|---:|---:|---:|---:|---:|---:|
| `ensemble` | 1411 | 443 | 132 | 15 | 196 | 102 |
| `union` | 1440 | 475 | 102 | 31 | 149 | 620 |
| `union-dur60` | 1439 | 475 | 102 | 31 | 150 | 617 |
| `near80` | 1443 | 456 | 92 | 18 | 188 | 310 |
| `cluster` | 1429 | 455 | 98 | 17 | 198 | 227 |
| `cluster-dur60` | 1429 | 455 | 98 | 17 | 198 | 227 |

Bootstrap: paired per-clip ΔTab F1, N=10000, seed=42. Acceptance for a ship decision is lo-95 > 0 on the full dev set plus the GAPS clean-12 strict no-regression check; this pilot is a variant filter, not the ship gate.

## Verdict — the merge is a bounded negative; Q1 closes

**No merge variant is admissible.** The best of the six (`cluster` /
`cluster-dur60`, identical because the 60 ms floor removes only 0-4 notes)
loses **-0.0167 Tab F1 [-0.0480, +0.0090]** and regresses onset F1 by
**-0.0195 [-0.0325, -0.0079]** — a CI-significant regression in the exact
metric the merge existed to improve. The permissive variants are
CI-significantly negative on Tab F1 as well (`union` -0.0541
[-0.0989, -0.0160]). The ordering is monotone in how many notes a variant
admits, in both modes and on every metric: admitting fewer notes loses less.
The best available merge is the empty one.

## Why — the number the entry gate could not see

Complementarity P(MuScriptor right | ensemble wrong) counts rescues among
gold notes the ensemble missed. It charges **nothing** for the notes a merge
admits that are not real. That second number is decisive here:

| variant | notes admitted | real (rescues) | **added precision** | ΔTab-correct | Δextra_detection | false adds per correct gained |
|---|---:|---:|---:|---:|---:|---:|
| `union` | 675 | 70 | **0.104** | +29 | +518 | ≈ 18 |
| `near80` | 326 | 54 | **0.166** | +32 | +208 | ≈ 6.5 |
| `cluster` | 227 | 41 | **0.181** | +18 | +125 | ≈ 7 |

The ensemble stream those notes join decodes at **0.6855 Tab precision**.
Admitting notes at 0.10-0.18 precision into a 0.69-precision stream cannot
be a net win at any mixing ratio — the structural-gate variants only reduce
the exposure, they do not change its sign. The six-bucket decomposition
confirms the mechanism rather than a scoring artifact: `union` does buy the
predicted bucket (missed_onset 196 → 149) but pays 102 → 620 in
extra_detection; `cluster` does not even buy the bucket (missed_onset
196 → **198**) while still paying 102 → 227.

## Solo coverage — the entry estimate was a comp-mode artifact

The 0.3818 headline reproduces **exactly** on the same 10 comp clips (a
clean replication check on a harness that reaches it by a different code
path). The 10 new solo clips measure **0.1481** (12/81) — 2.6× weaker, and
only just above the 0.10 gate. Pooled: 0.3049 (75/246). Solo material is
where the ensemble already has the least to be rescued from and where
MuScriptor's additions are least discriminable.

## Why no confidence-floor variant exists

The ROI deep-dive §3.1 prescribed "cluster-scoped adds **plus a per-note
confidence floor**". The floor is **not implementable**: MuScriptor's MIDI
carries constant velocity 100 on every note (verified across the cache), and
its supported API (`TranscriptionModel.transcribe`) yields
`NoteStartEvent`/`NoteEndEvent` with no score field. It is an autoregressive
token decoder, so any per-note confidence would require patching a
third-party generation loop to expose token log-probabilities — which are
sequence-model likelihoods, not calibrated detection confidences. Only the
structural gates in this pilot were available, and all of them fail.

## Methodological carry-forward for the second-opinion bench (report §3.3)

The ≥0.10 complementarity gate is **necessary but not sufficient**, and N2 is
the counter-example: it passed the entry gate by 3.8× and still produced no
admissible merge. Any future second-opinion candidate (Q4: Basic Pitch,
YourMT3+) should be gated on **both** legs, measured in the same offline
replay:

1. P(candidate right | ensemble wrong) ≥ 0.10 — is there anything to gain;
2. **added-note precision ≥ 0.5** under the candidate's best admission rule —
   can the gain be separated from the noise it arrives with.

Leg 2 costs nothing extra once the events are banked (`--stage sweep`
computes it), and it is the leg that kills merges.

## Caveats

- **20 clips, dev players only.** This is a variant filter, not the ship
  gate. The full 300-clip dev run was **not** spent: the failure is
  structural (added-note precision, consistent across both modes, all six
  variants, and both metrics), not a sample-size question, and house rule
  "do not iterate past a failed gate" applies.
- **Prior:** leave-one-player-out position prior (`--prior oof`), since
  `guitarset-v1` is trained on exactly these players. The sequence prior
  `guitarset-seq-v1` is shared by all arms and is in-distribution for the dev
  players; it inflates the absolute level identically in every row and cannot
  move a paired delta.
- `pitch_logits` are dropped by the event cache. The default decoder does not
  read them (only `context_reranker`, a non-default decoder, does), so the
  replay is faithful for `auto`.
- Model is `muscriptor-medium`; `large` remains unrunnable on this machine
  (fp32 load exceeds the commit limit, pagefile disabled). `large` could have
  a better added-note precision — but it is 4× the runtime of a backend
  already at 3-4× real time, so it cannot reach `auto` regardless.

## Reproduce

```
cd tabvision && TABVISION_DATA_ROOT=~/.tabvision/data \
python scripts/eval/n2_muscriptor_merge.py --stage sweep \
  --comp-clips 10 --solo-clips 10 --prior oof \
  --output ../docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.md \
  --json ../docs/EVAL_REPORTS/n2_muscriptor_merge_pilot_2026-07-21.json
```

Banked artifacts (`$TABVISION_DATA_ROOT/models/muscriptor_probe/`): 20
`<track>.ensemble.json` + 20 `<track>.medium.mid`. The sweep is pure replay —
no model inference — so re-running it is seconds, and new merge variants can
be tested against the same bank at zero compute cost.
