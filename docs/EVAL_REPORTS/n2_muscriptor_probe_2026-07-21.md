# N2 MuScriptor complementarity probe — GuitarSet dev clips

Model: muscriptor-medium (isolated venv) vs registered `highres-ensemble` | 10 clips | pitch-exact 50 ms greedy matching

| clip | gold | ens recall | ms recall | rescued/ens-wrong | ens onset/pitch F1 | ms onset/pitch F1 | ms s |
|---|---:|---:|---:|---:|---|---|---:|
| 00_BN1-129-Eb_comp | 133 | 0.887 | 0.865 | 3/15 | 0.926/0.918 | 0.904/0.881 | 0.0 |
| 00_Jazz2-187-F#_comp | 242 | 0.818 | 0.760 | 13/44 | 0.923/0.870 | 0.765/0.693 | 97.5 |
| 01_BN1-129-Eb_comp | 105 | 0.933 | 0.848 | 3/7 | 0.947/0.938 | 0.695/0.679 | 92.1 |
| 01_Jazz2-187-F#_comp | 166 | 0.873 | 0.645 | 3/21 | 0.898/0.868 | 0.542/0.495 | 99.0 |
| 02_BN1-129-Eb_comp | 154 | 0.929 | 0.877 | 4/11 | 0.911/0.905 | 0.883/0.857 | 56.9 |
| 02_Jazz2-187-F#_comp | 130 | 0.823 | 0.892 | 17/23 | 0.880/0.856 | 0.773/0.741 | 56.1 |
| 03_BN1-129-Eb_comp | 157 | 0.949 | 0.981 | 6/8 | 0.961/0.961 | 0.920/0.914 | 95.0 |
| 03_Jazz2-187-F#_comp | 102 | 0.990 | 0.922 | 1/1 | 0.981/0.981 | 0.797/0.780 | 48.8 |
| 04_BN1-129-Eb_comp | 127 | 0.984 | 0.882 | 0/2 | 0.973/0.973 | 0.921/0.839 | 72.8 |
| 04_Jazz2-187-F#_comp | 171 | 0.807 | 0.789 | 13/33 | 0.894/0.862 | 0.702/0.649 | 75.1 |

**Complementarity P(MuScriptor right | ensemble wrong) = 0.3818** (63/165; gate ≥ 0.1 → **PASS — full dev eval justified**)

## Reading

- Overall pitch recall: ensemble 0.8890 vs MuScriptor 0.8346 — MuScriptor is
  the weaker transcriber in absolute terms on every clip, which is exactly
  why it was probed as a *second opinion*, not a replacement. What matters
  is that its errors are differently distributed: it recovers 38% of the
  notes the ensemble misses, concentrated in the hardest chordal material
  (Jazz2 comp: 13/44, 17/23, 13/33 rescued).
- Every emitted note carried GM program 24 (acoustic guitar) — correct
  instrument identification, no cross-instrument leakage to filter.
- Runtime ≈ 50–100 s per 20–35 s clip on laptop CPU (~3–4× real time) —
  fine for offline second-opinion evaluation; far above the 5-minute/60-s
  shipping budget, so any production use would be an offline/explicit mode,
  consistent with the plan's offline-only framing.

## Setup notes and caveats

- **Model: `muscriptor-medium`** (the package default), not `large`. Large's
  fp32 load transiently exceeds this machine's memory commit (pagefile
  disabled; 22.1/31.7 GB committed at attempt time → OS error 1455). The
  `--dtype` flag cannot help because conversion happens after the full
  fp32 `safetensors` load. Large remains licensed and downloadable (5.1 GB
  cached) if a pagefile is ever enabled.
- **Gated weights**: each MuScriptor size is licensed separately on HF; the
  user accepted large + medium on account `pgil256` and authenticated the
  machine (`hf auth login`; token never handled in-session).
- **Local venv patch** (isolated probe venv only, not shipping code):
  `muscriptor/accelerator.py::synchronize` calls
  `torch.accelerator.synchronize()` whenever the API exists, which raises
  on CPU-only torch builds instead of no-opping (contradicting its own
  docstring). Patched with a `torch.accelerator.is_available()` guard —
  worth filing upstream.
- **Clip slice is comp-only**: the deterministic sorted-stride selection
  landed on BN1/Jazz2 `comp` clips across the five dev players. The 0.38
  complementarity is therefore a comp-mode estimate; the full dev
  evaluation must cover solo clips before any merge decision.
- Probe: `scripts/eval/n2_muscriptor_probe.py` (`--model medium --device
  cpu`), tabvision `.venv` + isolated `~/.tabvision/probe-envs/muscriptor`;
  MuScriptor MIDIs cached under
  `$TABVISION_DATA_ROOT/models/muscriptor_probe/`. Weights CC-BY-NC-4.0
  (admissible under SPEC §1.5; NC label required if anything derived is
  ever registered).
