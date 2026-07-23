# Local FretCam development evaluation set

This workflow builds a machine-local dataset for diagnosing FretCam position
and finger-contact errors. The media may be cached locally, but every source
must be public-licensed or reproducibly synthetic.

Private and user recordings are prohibited in every role: they cannot be
registered, annotated, scored, used for threshold tuning, used for training,
or used as release evidence. Participant permission does not create an
exception to this repository rule.

The workflow is intentionally separate from the live browser:

- It does not open a camera, record video, save browser frames, or extract
  still images.
- It registers only public-licensed or synthetic video already placed under
  the dataset's `media/` directory.
- The default root is
  `~/.tabvision/cache/fretcam_local_eval/`, outside the repository.
- A root anywhere inside a Git worktree is rejected, even when supplied
  explicitly.
- It stores no names, email addresses, camera identifiers, or absolute paths.

The checked-in JSON Schema is
`src/fretcam/schemas/local_eval_manifest_v1.schema.json`. Runtime validation
also enforces rules JSON Schema cannot: file containment, symlink rejection,
SHA-256 integrity, source/license verification, unique clips and timestamps,
contact semantics, and coverage.

## Initialize

```powershell
fretcam-local-eval init --dataset-id fretcam-public-local-01
```

To use a different machine-local cache, put the global option before the
command:

```powershell
fretcam-local-eval --root D:\local-data\fretcam-eval init `
  --dataset-id fretcam-public-local-01
```

Initialization creates only `manifest.json` and an empty `media/` directory.
Download or generate an eligible video separately, verify its license, and
place it in that directory yourself. Use a non-identifying lowercase filename
containing letters, numbers, `_`, or `-`.

## Register public-licensed metadata

```powershell
fretcam-local-eval add-clip `
  --clip-id public-full-neck-01 `
  --media media/public-full-neck-01.mp4 `
  --duration-ms 8000 `
  --provenance public_licensed `
  --source-uri https://example.org/source-page `
  --license CC-BY-4.0 `
  --confirm-rights `
  --handedness left `
  --framing full_neck `
  --lighting cool `
  --guitar electric_solid `
  --sleeve long_sleeve `
  --background cluttered
```

`--confirm-rights` means the source URL and license were checked for local
annotation and evaluation. The command hashes the existing file and stores
only its path relative to the dataset root. Validation fails if it later moves
or changes.

The CLI has no private, self-recorded, user-recorded, or consented-third-party
provenance option. The only accepted provenance values are
`public_licensed` and `synthetic`.

For synthetic media, `--source-uri` must identify the public, reproducible
recipe and `--license` must identify the recipe/output license:

```powershell
fretcam-local-eval add-clip `
  --clip-id synthetic-close-01 `
  --media media/synthetic-close-01.mp4 `
  --duration-ms 6000 `
  --provenance synthetic `
  --source-uri https://example.org/reproducible-recipe `
  --license CC0-1.0 `
  --confirm-rights `
  --handedness right `
  --framing close `
  --lighting bright `
  --guitar acoustic_steel `
  --sleeve bare_arm `
  --background studio
```

Clips containing minors are rejected.

### Appearance-diversity metadata

Skin-tone metadata is optional and must never be inferred casually from an
unlabeled public video. It is accepted only when the label is:

- declared by the public source (`source_declared`);
- supplied by a licensed dataset (`licensed_dataset_label`); or
- defined by a reproducible synthetic specification
  (`synthetic_specification`).

The applicable source/license must permit using that label, acknowledged with
`--confirm-appearance-metadata-rights`:

```powershell
  --skin-tone medium `
  --appearance-basis licensed_dataset_label `
  --confirm-appearance-metadata-rights
```

Omit all three options when the source does not provide an eligible label.

## Add finger annotations

One annotation labels a video timestamp, position, technique, and all five
fretting-hand fingers:

```powershell
fretcam-local-eval add-annotation `
  --clip-id public-full-neck-01 `
  --annotation-id frame-001 `
  --timestamp-ms 1250 `
  --phase stable `
  --position 1 `
  --technique barre `
  --thumb out_of_frame:unknown:- `
  --index visible:pressing:1@1,2@1,3@1,4@1,5@1,6@1 `
  --middle visible:hovering:- `
  --ring partially_visible:hovering:- `
  --pinky occluded:unknown:-
```

Finger labels use:

```text
visibility:pressing-state:string@fret,string@fret
```

- Visibility: `visible`, `partially_visible`, `occluded`, `out_of_frame`.
- Pressing state: `pressing`, `hovering`, `unknown`.
- Contacts: strings `1` through `6` and frets `0` through `24`; use `-`
  when there is no contact.

Visible fingers require an explicit pressing/hovering decision. Occluded and
out-of-frame fingers require `unknown`. A pressing finger requires at least
one contact, while a barre requires at least two strings on one finger.

Positions are restricted to I-XII. Stable annotations require a position;
`shifting` and `invalid` annotations must omit it. Available techniques are
`note`, `chord`, `barre`, `stretch`, and `slide`.

## Validate and inspect coverage

During incremental labeling, validate schema, source rights, paths, and media
integrity without requiring a complete matrix:

```powershell
fretcam-local-eval validate --schema-only
fretcam-local-eval coverage
```

The final strict command is:

```powershell
fretcam-local-eval validate
```

It requires:

- Stable labels for every position I-XII.
- All five techniques.
- Close and full-neck framing.
- Left- and right-handed players.
- Bright, dim, warm, cool, and uneven lighting.
- Pressing examples for index, middle, ring, and pinky.
- At least three eligible source-provided or synthetic skin-tone groups.
- At least two guitars, sleeve conditions, and backgrounds.

Coverage is a collection-health check, not evidence that FretCam has achieved
an accuracy target. Keep cached media and manifests outside the repository.
