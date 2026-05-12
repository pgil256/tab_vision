import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const distAssets = join(process.cwd(), 'dist', 'assets');
const forbiddenApiBase = 'http://localhost:5000';

if (!existsSync(distAssets)) {
  throw new Error('dist/assets does not exist. Run a production build first.');
}

const offendingFiles = readdirSync(distAssets)
  .filter((file) => file.endsWith('.js'))
  .filter((file) => readFileSync(join(distAssets, file), 'utf8').includes(forbiddenApiBase));

if (offendingFiles.length > 0) {
  throw new Error(
    `Production bundle must not contain ${forbiddenApiBase}. Found in: ${offendingFiles.join(', ')}`
  );
}
