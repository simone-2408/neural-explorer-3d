import type { ProxyOptions } from 'vite';
import { defineConfig } from 'vitest/config';

// The LLM tab talks to the local Python backend (python/server, `npm run llm`) through this proxy.
const api: Record<string, ProxyOptions> = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    configure(proxy) {
      // Backend not running: answer the status checks quietly (the tab then shows recorded examples).
      proxy.on('error', (_error, request, response) => {
        if (!('writeHead' in response) || response.headersSent) return;
        const status = request.url?.endsWith('/status');
        response.writeHead(status ? 200 : 503, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify(status ? { offline: true } : { detail: 'The Python backend is not running: start it with npm run llm' }));
      });
    },
  },
};

export default defineConfig({
  // Relative base so the production build works from any static host or sub-folder.
  base: './',
  server: {
    port: 5173,
    strictPort: true,
    proxy: api,
  },
  preview: {
    proxy: api,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    // three.js + KaTeX are intentionally shipped as regular dependencies.
    chunkSizeWarningLimit: 1600,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
