/**
 * From logits to the next token: temperature, top-k, top-p (nucleus) and
 * sampling. Applied in the browser so every change of the sliders is
 * immediate.
 */
import type { Rng } from '../math/random';

export interface SamplingOptions {
  /** > 0. Lower = sharper distribution; near 0 = always the most likely token. */
  temperature: number;
  /** Keep only the k most likely tokens (0 = no limit). */
  topK: number;
  /** Keep the smallest set of tokens whose probability sums to ≥ p (1 = no limit). */
  topP: number;
}

export const DEFAULT_SAMPLING: SamplingOptions = { temperature: 1, topK: 0, topP: 1 };

export interface Distribution {
  /** Probabilities after temperature, top-k and top-p (filtered tokens have 0). */
  probs: Float64Array;
  /** Probabilities with temperature only (before filtering). */
  raw: Float64Array;
  /** Token indices sorted by decreasing probability. */
  order: number[];
  /** Number of tokens that survive the filters. */
  kept: number;
}

export function softmaxWithTemperature(logits: ArrayLike<number>, temperature: number): Float64Array {
  const t = Math.max(temperature, 1e-3);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t);
  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] / t - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function distribution(logits: ArrayLike<number>, options: SamplingOptions): Distribution {
  const raw = softmaxWithTemperature(logits, options.temperature);
  const order = Array.from(raw.keys()).sort((a, b) => raw[b] - raw[a]);
  const keep = new Uint8Array(raw.length);
  let kept = 0;
  let cumulative = 0;
  for (const i of order) {
    if (options.topK > 0 && kept >= options.topK) break;
    if (options.topP < 1 && cumulative >= options.topP) break;
    keep[i] = 1;
    kept++;
    cumulative += raw[i];
  }
  const probs = new Float64Array(raw.length);
  let sum = 0;
  for (let i = 0; i < raw.length; i++) if (keep[i]) sum += raw[i];
  for (let i = 0; i < raw.length; i++) probs[i] = keep[i] ? raw[i] / sum : 0;
  return { probs, raw, order, kept };
}

export function sample(dist: Distribution, rng: Rng): number {
  let r = rng.next();
  for (const i of dist.order) {
    r -= dist.probs[i];
    if (r <= 0 && dist.probs[i] > 0) return i;
  }
  return dist.order[0];
}

/** Shannon entropy in bits (how uncertain the prediction is). */
export function entropyBits(probs: ArrayLike<number>): number {
  let h = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] > 0) h -= probs[i] * Math.log2(probs[i]);
  return h;
}
