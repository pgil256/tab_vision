# N1 kroma smoke — onset/pitch F1 on 5 GuitarSet dev clips

| condition | clip | onset F1 | pitch F1 | events | gold | s |
|---|---|---:|---:|---:|---:|---:|
| guitar_gaps | 00_BN1-129-Eb_comp | 0.9070 | 0.8837 | 125 | 133 | 25.125 |
| guitar_gaps | 01_BN1-129-Eb_comp | 0.8768 | 0.8571 | 98 | 105 | 14.054 |
| guitar_gaps | 02_BN1-129-Eb_comp | 0.8477 | 0.8278 | 148 | 154 | 13.198 |
| guitar_gaps | 03_BN1-129-Eb_comp | 0.9128 | 0.8926 | 141 | 157 | 13.41 |
| guitar_gaps | 04_BN1-129-Eb_comp | 0.9562 | 0.9243 | 124 | 127 | 12.762 |
| guitar_fl | 00_BN1-129-Eb_comp | 0.9272 | 0.9195 | 128 | 133 | 15.16 |
| guitar_fl | 01_BN1-129-Eb_comp | 0.9245 | 0.9245 | 107 | 105 | 12.931 |
| guitar_fl | 02_BN1-129-Eb_comp | 0.9062 | 0.9000 | 166 | 154 | 12.525 |
| guitar_fl | 03_BN1-129-Eb_comp | 0.9620 | 0.9620 | 159 | 157 | 12.479 |
| guitar_fl | 04_BN1-129-Eb_comp | 0.9615 | 0.9615 | 133 | 127 | 12.976 |
| guitar_kroma | 00_BN1-129-Eb_comp | 0.9070 | 0.8837 | 125 | 133 | 13.463 |
| guitar_kroma | 01_BN1-129-Eb_comp | 0.8768 | 0.8571 | 98 | 105 | 12.756 |
| guitar_kroma | 02_BN1-129-Eb_comp | 0.8477 | 0.8278 | 148 | 154 | 12.375 |
| guitar_kroma | 03_BN1-129-Eb_comp | 0.9128 | 0.8926 | 141 | 157 | 14.924 |
| guitar_kroma | 04_BN1-129-Eb_comp | 0.9562 | 0.9243 | 124 | 127 | 14.498 |

| condition | mean onset F1 | mean pitch F1 |
|---|---:|---:|
| guitar_gaps | 0.9001 | 0.8771 |
| guitar_fl | 0.9363 | 0.9335 |
| guitar_kroma | 0.9001 | 0.8771 |

Gate (±0.05 vs best registered member): onset Δ -0.0362, pitch Δ -0.0564 → **FAIL**

## Verdict — `guitar_kroma` is a near-duplicate of `guitar-gaps`; branch CLOSED

The gate arithmetic above is not the real story. Every kroma row is
**bit-identical** to the `guitar_gaps` row for the same clip (same F1s, same
event counts), which prompted checkpoint forensics:

- Live-model check: after `MidiTranscriptionModel(instrument="guitar",
  checkpoint_path=<guitar-kroma.pth>)`, the in-memory weights equal the kroma
  file exactly (`torch.equal` true) — the local-path loader works; this is
  **not** a silent-fallback bug.
- Tensor diff vs `guitar-gaps.pth` (316/316 keys and shapes align): only
  **16/316 tensors differ by more than 1e-2, all of them BatchNorm
  `running_var`/`running_mean` statistics** (max 4.39e-1 on
  `velocity_model.bn5.running_var`). The **median max-difference across
  learned `.weight` tensors is 4.6e-4** — fp16-round-trip scale.
- Conclusion: `guitar_kroma.safetensors` is the same trained network as the
  registered `gaps` member, re-exported (updated BN stats, likely an
  EMA/re-save of the identical run). It contributes **zero ensemble
  diversity**: thresholded event streams are bit-identical on all five smoke
  clips.

**Branch decision (plan N1 fail branch):** close `guitar_kroma` as a bounded
negative — redundant with the registered `gaps` member, not a third opinion.
Program N continues with the MuScriptor complementarity probe (CC-BY-NC
weights, offline/process-isolated) as the next candidate.

## Provenance

- Command: `python scripts/eval/n1_kroma_smoke.py --output
  ../docs/EVAL_REPORTS/n1_kroma_smoke_2026-07-20.md --json
  ../docs/EVAL_REPORTS/n1_kroma_smoke_2026-07-20.json` (tabvision `.venv`,
  `TABVISION_DATA_ROOT=~/.tabvision/data`, `PYTHONUTF8=1`).
- Clip slice: deterministic sorted-stride rule landed on the same piece
  (`BN1-129-Eb_comp`) across all five development players — a controlled
  same-piece, cross-player comparison. Player 05 untouched.
- Metric: `tabvision.eval.metrics.event_f1`, 50 ms onset tolerance;
  raw event streams (no fusion/priors involved).
- Checkpoint conversion: `scripts/eval/convert_kroma_checkpoint.py`; source
  `xavriley/midi-transcription-models/guitar_kroma.safetensors` SHA-256
  `26919a2f…f57f2de0`, converted `guitar-kroma.pth` SHA-256 `5c43b6ab…b2ebd63e`
  (full hashes in `~/.tabvision/data/models/guitar-kroma.manifest.json`).
- Probe mechanism: `TABVISION_HIGHRES_ELECTRIC_CKPT` env path +
  `checkpoint="guitar_electric"` — no runtime code changed.
