# TabVision Backend

This Flask API is deployed to Modal for production video processing. The Vercel
frontend calls the Modal HTTPS endpoint directly.

## Modal Deploy

```bash
cd tabvision-server
modal setup
modal deploy modal_app.py
```

After deploy, verify the API:

```bash
curl https://<modal-api-url>/health
```

Expected response:

```json
{"status":"ok"}
```

## Learned-Evidence Policy

Production defaults to domain-aware routing:

```text
TABVISION_POSITION_PRIOR=auto
TABVISION_SEQUENCE_PRIOR=auto
TABVISION_STRING_EVIDENCE=auto
TABVISION_PHRASE_REFINEMENT=false
```

`auto` loads the hash-verified GuitarSet position/sequence pair only for clean
acoustic, standard-tuning, capo-zero jobs. Classical, electric, alternate
tuning, and capo jobs resolve learned position evidence to `none`. No timbral
model or phrase-refinement API is registered because both failed their fixed
development gates. Operators can independently roll position or sequence back
to `none`; explicit registered artifact names are for reproducible evaluation.

Completed result metadata remains backward compatible and now may include:

| field | meaning |
|---|---|
| `requestedPositionPrior` / `resolvedPositionPrior` | requested policy and domain-resolved artifact |
| `requestedSequencePrior` / `resolvedSequencePrior` | requested policy and compatible resolved sequence |
| `requestedStringEvidence` / `resolvedStringEvidence` | requested timbral policy and resolved model (`none` today) |
| `artifactVersions` | map of loaded artifact name to version |
| `artifactSha256` | map of loaded artifact name to verified SHA-256 |

The legacy `positionPrior` field remains and reports the resolved value.

## Vercel Frontend Env

Set the Modal API URL on the Vercel frontend project:

```bash
printf "%s" "https://<modal-api-url>" | vercel env add VITE_API_URL production
printf "%s" "https://<modal-api-url>" | vercel env add VITE_API_URL preview
```

Redeploy the frontend after changing Vercel env vars. Existing deployments keep
the bundle they were built with.
