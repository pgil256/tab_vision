import { pathToFileURL } from 'node:url';

const guardedVercelEnvs = new Set(['production', 'preview']);

export function requiresApiUrl(env = process.env) {
  const isVercelBuild = env.VERCEL === '1' || env.VERCEL === 'true';
  const isGuardedEnv = guardedVercelEnvs.has(env.VERCEL_ENV ?? '');
  const hasApiUrl = Boolean(env.VITE_API_URL?.trim());

  return isVercelBuild && isGuardedEnv && !hasApiUrl;
}

export function assertVercelApiUrl(env = process.env) {
  if (!requiresApiUrl(env)) {
    return;
  }

  throw new Error(
    'VITE_API_URL is required for Vercel production and preview builds. Set it to the Modal API URL.'
  );
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  assertVercelApiUrl();
}
