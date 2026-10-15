import { defineConfig } from 'vitest/config';

/** Integration tests that call Python/PyTorch (`npm run test:parity`). */
export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
