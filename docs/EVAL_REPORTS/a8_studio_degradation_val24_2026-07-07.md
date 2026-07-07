# A8 — studio-condition degradation eval (val24)

Config: `highres` + `guitarset-v1` prior, splits `validation`, 24 clips. Gold labels unchanged (audio-only degradation). roadmap val24 clean baseline: single-line 0.4820 / strummed 0.7951 — clean rows should reproduce it.

**Diagnostic tier, NOT a gate** (roadmap A8 / D1-c): this does not touch SPEC §1.4 targets. It measures the clean-WAV-eval → real-product-capture gap to decide whether to keep tuning clean-corpus accuracy or pivot to input robustness.

## Profiles

- **`opus_128`** — Opus/webm 48k @128k, no mic model — codec-only floor (good mic, quiet room)
- **`opus_64`** — Opus/webm 48k @64k, no mic model — codec stress (constrained bitrate)
- **`laptop_mic`** — Built-in laptop mic: HP70/LP8000 + pink noise ~-44 dBFS (SNR~30 dB) + Opus 96k; AGC off (faithful to the app)
- **`noisy_room`** — Worst-case: HP90/LP7000 + pink noise ~-34 dBFS (SNR~16 dB) + light compression + Opus 64k

The product records `audio/webm;codecs=opus` at 48 kHz with echoCancellation / noiseSuppression / autoGainControl **disabled** (`web-client/.../RecordPanel.tsx`), so the codec is the only guaranteed degradation; mic band + noise are environmental. Noise levels are approximate (SNR vs a ~-18 dBFS-RMS signal).

## Tab F1 (mean / lower-95, bootstrap N=10k)

| condition | clean_acoustic_single_line mean (lo95) | clean_acoustic_strummed mean (lo95) | aggregate mean (lo95) | Δ agg |
|---:|---:|---:|---:|---:|
| `clean` | 0.4820 (0.3761) | 0.7951 (0.7565) | 0.6386 (0.5537) | — |
| `opus_128` | 0.4826 (0.3783) | 0.7975 (0.7589) | 0.6401 (0.5555) | +0.0015 |
| `opus_64` | 0.4812 (0.3775) | 0.8014 (0.7652) | 0.6413 (0.5561) | +0.0027 |
| `laptop_mic` | 0.4861 (0.3828) | 0.8036 (0.7654) | 0.6448 (0.5607) | +0.0063 |
| `noisy_room` | 0.4904 (0.3865) | 0.8038 (0.7661) | 0.6471 (0.5627) | +0.0085 |

## Onset F1 (detection — where codec/noise hits note recall)

| condition | clean_acoustic_single_line mean (lo95) | clean_acoustic_strummed mean (lo95) | aggregate mean (lo95) | Δ agg |
|---:|---:|---:|---:|---:|
| `clean` | 0.9227 (0.8974) | 0.9359 (0.9062) | 0.9293 (0.9101) | — |
| `opus_128` | 0.9238 (0.8987) | 0.9367 (0.9076) | 0.9303 (0.9111) | +0.0010 |
| `opus_64` | 0.9238 (0.8998) | 0.9380 (0.9079) | 0.9309 (0.9120) | +0.0016 |
| `laptop_mic` | 0.9229 (0.8996) | 0.9412 (0.9122) | 0.9320 (0.9136) | +0.0027 |
| `noisy_room` | 0.9267 (0.9046) | 0.9407 (0.9149) | 0.9337 (0.9167) | +0.0044 |

## Pitch F1 (detection + pitch)

| condition | clean_acoustic_single_line mean (lo95) | clean_acoustic_strummed mean (lo95) | aggregate mean (lo95) | Δ agg |
|---:|---:|---:|---:|---:|
| `clean` | 0.9140 (0.8833) | 0.9184 (0.8877) | 0.9162 (0.8949) | — |
| `opus_128` | 0.9151 (0.8861) | 0.9186 (0.8876) | 0.9168 (0.8956) | +0.0006 |
| `opus_64` | 0.9137 (0.8824) | 0.9202 (0.8887) | 0.9170 (0.8952) | +0.0008 |
| `laptop_mic` | 0.9142 (0.8866) | 0.9227 (0.8922) | 0.9184 (0.8984) | +0.0022 |
| `noisy_room` | 0.9180 (0.8904) | 0.9218 (0.8911) | 0.9199 (0.8994) | +0.0037 |

## Worst per-clip Tab F1 drops

- **`laptop_mic`**: guitarset/05_Funk3-112-C#_solo (-0.031), guitarset/05_Funk3-112-C#_comp (-0.010), guitarset/05_BN1-147-Gb_comp (-0.006), guitarset/05_BN3-119-G_solo (-0.005)
- **`noisy_room`**: guitarset/05_Funk3-112-C#_comp (-0.070), guitarset/05_Funk1-114-Ab_solo (-0.034), guitarset/05_BN2-131-B_comp (-0.029), guitarset/05_BN3-119-G_solo (-0.016), guitarset/05_BN1-147-Gb_comp (-0.010), guitarset/05_BN1-129-Eb_comp (-0.000)

## Verdict — fork

**robust → keep tuning.** Accuracy holds through the whole capture chain (codec floor Δagg +0.0015, realistic Δagg +0.0063). The eval-vs-product gap is small — clean-corpus accuracy work transfers to the product; keep tuning.

(Heuristic bands on Δ aggregate Tab F1 vs clean: holds ≥ -0.03, craters ≤ -0.08. Diagnostic only.)

