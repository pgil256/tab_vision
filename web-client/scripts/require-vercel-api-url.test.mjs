import assert from 'node:assert/strict';

import { requiresApiUrl, assertVercelApiUrl } from './require-vercel-api-url.mjs';

assert.equal(
  requiresApiUrl({ VERCEL: '1', VERCEL_ENV: 'production', VITE_API_URL: '' }),
  true,
  'production Vercel builds require VITE_API_URL'
);

assert.equal(
  requiresApiUrl({ VERCEL: '1', VERCEL_ENV: 'preview', VITE_API_URL: '   ' }),
  true,
  'preview Vercel builds require VITE_API_URL'
);

assert.equal(
  requiresApiUrl({ VERCEL: '1', VERCEL_ENV: 'development', VITE_API_URL: '' }),
  false,
  'development Vercel builds do not require VITE_API_URL'
);

assert.equal(
  requiresApiUrl({ VERCEL: undefined, VERCEL_ENV: 'production', VITE_API_URL: '' }),
  false,
  'local production builds do not require VITE_API_URL'
);

assert.doesNotThrow(() => {
  assertVercelApiUrl({
    VERCEL: '1',
    VERCEL_ENV: 'production',
    VITE_API_URL: 'https://example.modal.run',
  });
});

assert.throws(
  () => assertVercelApiUrl({ VERCEL: '1', VERCEL_ENV: 'production', VITE_API_URL: '' }),
  /VITE_API_URL is required/
);
