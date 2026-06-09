# v1.1 oracle string-resolution probe — 2026-06-03

**Question.** v1 single-line Tab F1 is capped at ~0.52 by *string* ambiguity
(audio can't tell which string a pitch was played on). v1.1's thesis: the
fretting-hand video resolves the string. Before building any video or eval data,
does the *existing* fusion actually consume a per-note string signal and resolve
it?

**Method.** Pure fusion over GuitarSet gold labels — no audio model, no video, no
rendering, no inference (runs in seconds). For each player-05 validation clip:

- Build `AudioEvent`s from gold **pitch + onset only** (perfect audio; string/fret
  stripped — that is precisely the audio limit).
- Apply the leak-free `guitarset-v1` position prior (in **both** conditions).
- `audio`  = `fuse(events, [])`.
- `+oracle` = `fuse(events, oracle_fingerings)`, where each oracle `FrameFingering`
  is peaked on the true `(string, fret)` (plus any chord-mates within
  `CHORD_MAX_GAP_S`).

Script: `tabvision/scripts/eval/v1_1_oracle_string_probe.py`
(`python -m scripts.eval.v1_1_oracle_string_probe --manifest data/eval/composite.toml`).

**Result.**

| Tier | audio | +oracle | Δ |
|---|---:|---:|---:|
| clean_acoustic_single_line | 0.568 | **0.995** | +0.427 |
| clean_acoustic_strummed | 0.747 | **0.978** | +0.231 |
| aggregate (60 clips) | 0.657 | **0.986** | +0.329 |

**Conclusions.**

1. **The resolver already exists and is correctly wired.** The path is
   `fuse → playability.find_fingering_at(onset) → emission_cost`'s
   `lambda_vision · -log(marginal_string_fret[s, f])` term, candidate-restricted by
   the Viterbi state space. Given a perfect hand signal it drives single-line to
   **0.995** (> the 0.94 v1.1 target) and strummed to **0.978** (> 0.85). The
   2026-06-03 design doc §4 ("the string-discriminative signal is not consumed by
   the per-note resolver") was **inaccurate** — that described the *neck-anchor*
   (fret-only) path; the `FrameFingering` path was already live. No new resolver
   module is needed.
2. **String is the entire lever.** Perfect string info ⇒ near-perfect tab.
3. **v1.1 P1 (resolver) is effectively done; the milestone reduces to P0 eval
   data** — a corpus with fretting-hand video + frame/note string labels to drive
   the resolver: synthetic-from-GuitarSet (design §6.1) to prove it on clean
   video, or a license-clean public video+string dataset (§6.2, the real gate).

**Caveats.** The `audio` column (0.57 / 0.75) uses *perfect* pitch+onset, so it is
higher than the v1 acceptance (0.52 / 0.68, which carries real audio errors); this
probe isolates the *string* axis only. The 0.995 (not 1.000) single-line residual
is a handful of candidate edge cases (e.g. enharmonic max-fret ties), not a
systematic miss.
