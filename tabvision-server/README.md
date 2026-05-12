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

## Vercel Frontend Env

Set the Modal API URL on the Vercel frontend project:

```bash
printf "%s" "https://<modal-api-url>" | vercel env add VITE_API_URL production
printf "%s" "https://<modal-api-url>" | vercel env add VITE_API_URL preview
```

Redeploy the frontend after changing Vercel env vars. Existing deployments keep
the bundle they were built with.
