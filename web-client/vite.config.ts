import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // PORT lets a second dev instance run alongside the default one.
    port: Number(process.env.PORT) || 5173,
  },
});
