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

## Yield assessment: format risk resolved, count still unquantified

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

1. **`PDMX.csv` only** (`https://zenodo.org/records/15571083/files/PDMX.csv?download=1`)
   — metadata, no score content. Filter guitar programs (24–31) ×
   `no_license_conflict` × has-MXL for the exact guitar count. **Attempted
   2026-07-02: Zenodo unreachable from this network (connect timeout); retry
   later.**
2. If the guitar count is real: fetch the `mxl.tar.gz` archive once (per-score
   fetch impossible on Zenodo; total size unverified), extract only the
   CSV-filtered guitar paths locally, grep for `<technical>`/`<fret>` to get
   the true tab-bearing count and validate the MuseScore 3.6.2 exporter
   behavior on ~10 samples. Nothing committed to the repo.
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
