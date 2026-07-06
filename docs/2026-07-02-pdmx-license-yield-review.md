# PDMX license + tab-subset yield review (A15 staging step 2, read-only)

**Date:** 2026-07-02 · **Scope:** PDMX only, per user direction (DadaGP deferred).
**Constraint honored:** no dataset content downloaded; sources were the paper,
GitHub repo + raw source files, and search-indexed Zenodo record snippets.

## License verdict: CLEAR-WITH-CONDITIONS for a shippable derived prior

- **Dataset release license: CC-BY** — the ICASSP paper states "PDMX is released
  under a CC-BY license as a fully commercially viable dataset"
  (arXiv 2409.10831). CC-BY is in SPEC §1.5's explicitly preferred set, and the
  shipped `guitarset-v1` prior (GuitarSet, CC-BY-4.0) is direct in-repo
  precedent for this artifact class.
- **Per-score licenses: Public Domain Mark or CC0 only** — verified in the
  dataset build code (`wrangling/full.py` filters on exactly those two license
  URLs).
- **`no_license_conflict` subset:** 12.29% (31,221) of songs have internal
  MuseScore-file copyright metadata contradicting the public PD claim; the
  clean subset is **222,856 songs**. Use only this subset (the dataset's own
  caveat, already assumed in the roadmap).
- **Conditions:** (1) attribute the dataset (Long, Novack, Berg-Kirkpatrick,
  McAuley, ICASSP 2025; DOI 10.5281/zenodo.15571083) in README + LICENSES.md;
  (2) `no_license_conflict` subset only; (3) never redistribute score content —
  only derived count statistics (unprotectable facts) get committed.
- **Residual risk:** PD status is uploader-self-declared on MuseScore; for a
  count-table artifact with no content redistribution this is about as low as
  dataset risk gets.
- **Open verification to-do:** load the Zenodo record page directly once
  reachable (unreachable from this network on 2026-07-02, twice confirmed) to
  eyeball the license field; the paper + search-indexed snippets already agree.

## Yield assessment: format risk resolved; guitar count RESOLVED (metadata scan, 2026-07-02 retry)

**`PDMX.csv` fetched on the 2026-07-02 retry (225 MB, metadata only — no score
content). Scan (`tracks` MIDI programs 24–31 × `subset:no_license_conflict` ×
has-MXL):**

| Filter | Songs |
|---|---|
| Total rows | 254,077 |
| `no_license_conflict` | 222,856 (matches the dataset's own count) |
| Guitar program in `tracks` | 4,446 |
| × `no_license_conflict` | 3,437 |
| × has MXL | **3,435** |
| — of which all tracks are guitar | 1,068 |
| — of which single-track solo guitar | 798 |

Per-program (clean+MXL songs, a song can carry several): nylon 1,628 · steel
755 · clean-electric 832 · jazz 208 · distortion 183 · overdrive 151 · muted
120 · harmonics 16. Genres: 58% untagged, then classical (524), soundtrack
(199), rock (150), folk (111) — the predicted classical/acoustic lean is real
but rock/pop is not zero. **3,435 lands mid-range of the prior "hundreds to a
few thousand" estimate → the mxl.tar.gz step is justified.**

### TAB-staff scan (archive fetched + scanned 2026-07-02, user-approved)

`scripts/acquire/pdmx_tab_scan.py` streamed `mxl.tar.gz` (1.89 GB, local data
root only) and opened exactly the 3,435 filtered members (all present, none
unreadable):

- **734 TAB-bearing scores (21.4%)** — squarely inside the predicted 10–50%
  band. Genres: 368 untagged · classical 147 · rock 46 · soundtrack 43 ·
  folk 34 · pop 11 · jazz 9 · metal 7.
- **Validation (10-sample, GAPS MusicXML tab walk — the extraction code
  path):** 10/10 parse; 9/10 with per-note ``pitch == open_string + fret``
  consistency 1.000, one at 0.942 (an extraction-time per-note filter
  handles such notes). All 10 declare `<staff-tuning>` (MuseScore always
  writes it) and **all 10 declared tunings are standard EADGBE**.
- **Corpus scale:** sampled scores carry ~200–1,050 tab notes (mean ≈ 460)
  → **~340k tab notes across the 734 scores, vs 14,003 transition samples
  behind the shipped `guitarset-seq-v1`** — a ~20× corpus bump, and full
  pieces rather than 30 s excerpts.

**Verdict: the PDMX n-gram corpus is real.** Next step is extraction +
n-gram build through the existing probe harness with the same val24 + GAPS
no-regression gates (step 3 below).

### Original pre-scan assessment (kept for the record)

- **Do NOT use the primary JSON ("MusicRender") files** — verified from
  `reading/classes.py`: the Note class has no string/fret/tablature fields;
  repo-wide code search for `fret`/`tablature` has zero hits. The docs' "no
  information loss" claim is false for tab data.
- **The MXL (compressed MusicXML) files are the target** — verified from
  `wrangling/pdmx.py`: MXLs are MuseScore 3.6.2's own native export of the
  original `.mscz` uploads (not derived from MusicRender), so
  `<technical><string>/<fret>` survives wherever the score has a TAB staff.
  The in-repo GAPS MusicXML tab parser applies directly.
- **Corpus skew:** piano/choral-dominated (no guitar in the top-10
  instrumentations), classical/folk-leaning, 67% untagged genre. The PD filter
  structurally excludes most pop/rock tab. Expect a **classical-leaning**
  guitar prior — the A15 dual no-regression gates and config-keying are the
  guard (and the A15 probe already measured that classical single-line
  transitions are near-optimally handled by the hand-coded terms; see
  `EVAL_REPORTS/a15_gaps_sequence_probe_indomain_2026-07-02.md`).
- **Best estimate:** guitar-bearing songs plausibly low single-digit percent of
  222,856 (~2k–12k); TAB-staff-bearing subset of those unknown (10–50%) →
  **order hundreds to a few thousand tab scores**. Sufficient for n-gram
  statistics if real; not yet verified.
- **TAB-staff presence is not in the metadata CSV** — the CSV's `tracks` column
  (sorted MIDI programs, e.g. `24`–`31` = guitars) resolves the guitar count;
  the TAB-staff count needs a sample of actual MXLs.

## Next acquisition steps (license-cleared; smallest-first)

1. ~~**`PDMX.csv` only**~~ **DONE (2026-07-02 retry)** — fetched and scanned;
   results in the yield table above. The scan script is
   session-scratch (guitar programs 24–31 over the `tracks` column ×
   `no_license_conflict` × has-MXL); counts are derived facts, safe to commit.
2. ~~Fetch `mxl.tar.gz` + TAB-staff scan~~ **DONE (2026-07-02,
   user-approved)** — archive is 1.89 GB, lives in the local data root;
   results in the TAB-staff section above (`scripts/acquire/pdmx_tab_scan.py`
   is the committed, tested scanner). Nothing from the dataset committed.
3. Only then: extraction + n-gram build through the same probe harness
   (`scripts/eval/a15_sequence_prior_probe.py`) with the val24 + GAPS gates.

## Sources

- https://github.com/pnlong/PDMX (README; `reading/classes.py`,
  `wrangling/pdmx.py`, `wrangling/full.py`, `wrangling/instruments.py` @ main)
- https://arxiv.org/abs/2409.10831 (ICASSP 2025 paper)
- https://zenodo.org/records/15571083 (record page itself unreachable today;
  corroborated via paper + search-indexed snippets)
- https://www.w3.org/2021/06/musicxml40/tutorial/tablature/ and
  https://musescore.org/en/node/15795 (MusicXML string/fret export capability)
