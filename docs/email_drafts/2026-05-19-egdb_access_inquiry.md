# EGDB access + license inquiry — email draft

**Drafted:** 2026-05-19
**Phase:** 0 (Phase 0 acceptance — strategy doc §6 / Phase 0 impl plan §3.7)
**Status:** unsent. Track send-date and reply in `docs/DECISIONS.md` per the
              format in `SPEC.md` §0.5.
**Expected turnaround:** ~1 week (per strategy doc §9 + R1 fallback).

## Why this email exists

EGDB is the only public dataset with the right labels to evaluate our
**distorted electric** tier (strategy doc §1, D2 acceptance target 0.80).
The strategy doc §0 license table flags EGDB as **license-pending** —
the repo at https://ss12f32v.github.io/Guitar-Transcription/ does not
carry an explicit license, so portfolio-default use (SPEC §1.5) is
gated on a written reply from the author.

Without resolution the strategy doc R1 fallback kicks in: free-IR-augmented
GuitarSet for the distorted-electric tier, explicitly flagged as
synthesized in reports.

## Recipient

- **To:** Yu-Hua Chen — `f08946011@ntu.edu.tw`
  (contact on the EGDB project page; verified 2026-05-19 by Yu-Hua Chen
  still being the lead author on EGDB-PG, arXiv:2504.07406, April 2025).

## Subject

`EGDB dataset access and license inquiry — portfolio research project`

## Body

> Dear Yu-Hua,
>
> I'm a software engineer building TabVision, an open-source guitar
> transcription pipeline that I plan to publish as a portfolio project.
> Onset and pitch F1 are already at spec on GuitarSet (≥ 0.92 / ≥ 0.90),
> but string/fret assignment on distorted electric content is unmeasured
> — there's no public corpus with the right labels for that tier. EGDB
> is the only dataset I've found that fits, and it's referenced in our
> license map as the gating resource for the distorted-electric
> evaluation tier.
>
> Could I ask a few questions before relying on it?
>
> 1. **Availability.** Is the Google Drive link from
>    https://ss12f32v.github.io/Guitar-Transcription/ still the canonical
>    download location, and is it currently active?
> 2. **License.** The repo doesn't carry an explicit license file. For
>    a public-portfolio project (public GitHub repo, possible blog post,
>    possible recorded demo) I'd need to know what terms apply —
>    particularly whether the data permits redistribution of evaluation
>    reports that quote per-clip metrics, and whether portfolio use is
>    in scope or whether the dataset is research-only.
> 3. **EGDB-PG.** I saw the April 2025 EGDB-PG paper. For my
>    distorted-electric tier eval, would you recommend the original
>    EGDB, EGDB-PG, or both? And does EGDB-PG carry the same access
>    path and license as EGDB?
>
> I'm happy to attribute the dataset per any citation form you prefer
> and to flag any conditions in the project's `LICENSES.md`. No rush
> from my side — a one- or two-week reply window is fine. Thanks for
> putting EGDB out into the community; it's been hard to find an
> analogue.
>
> Best,
> Patrick Gilhooley
> pgilhooley95@gmail.com

## After sending

1. Append a `DECISIONS.md` entry dated to the send date, format per
   SPEC §0.5:

   ```
   ## YYYY-MM-DD — EGDB author email sent
   **Phase:** 0
   **Decision tree:** strategy doc §0 license gate / EGDB row
   **Branch taken:** Send license + access inquiry to Yu-Hua Chen
                     (f08946011@ntu.edu.tw).
   **Evidence:** docs/email_drafts/2026-05-19-egdb_access_inquiry.md
   **Reasoning:** EGDB license unresolved on repo; portfolio-default
                  use blocked until written reply. R1 fallback
                  (free-IR-augmented GuitarSet) deferred pending reply.
   ```

2. When the reply arrives, append a second `DECISIONS.md` entry with
   the verdict (license terms granted / refused / no reply).

3. If no reply after **14 days**, send a single follow-up. If still no
   reply after another 14 days, trigger the R1 fallback per strategy
   doc §7.
