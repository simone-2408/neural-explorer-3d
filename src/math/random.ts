/**
 * Seeded pseudo-random number generator (mulberry32).
 * Reproducibility matters for an educational tool: the same configuration
 * and seed always produce the same initial weights.
 */
export interface Rng {
  /** Uniform in [0, 1). */
  next(): number;
  uniform(min: number, max: number): number;
  /** Gaussian sample (Box–Muller). */
  normal(mean?: number, std?: number): number;
}

export function createRng(seed: number): Rng {
  let state = seed >>> 0;
  let spare: number | null = null;

  const next = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    uniform: (min, max) => min + (max - min) * next(),
    normal: (mean = 0, std = 1) => {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return mean + std * value;
      }
      let u = 0;
      while (u <= Number.EPSILON) u = next();
      const v = next();
      const radius = Math.sqrt(-2 * Math.log(u));
      spare = radius * Math.sin(2 * Math.PI * v);
      return mean + std * radius * Math.cos(2 * Math.PI * v);
    },
  };
}

/** Deterministically mixes several integers into one 32-bit seed. */
export function mixSeed(...parts: number[]): number {
  let h = 0x811c9dc5;
  for (const part of parts) {
    h = Math.imul(h ^ (part | 0), 0x01000193);
    h ^= h >>> 13;
  }
  return h >>> 0;
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 1_000_000);
}
