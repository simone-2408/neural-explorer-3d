/** Small numeric helpers shared by the model, the encodings and the inspectors. */

export function maxAbs(values: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < values.length; i++) {
    const v = Math.abs(values[i]);
    if (v > max) max = v;
  }
  return max;
}

export interface Summary {
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  maxAbs: number;
}

export function summarize(values: ArrayLike<number>): Summary {
  const n = values.length;
  if (n === 0) return { count: 0, min: 0, max: 0, mean: 0, std: 0, maxAbs: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (values[i] - mean) ** 2;
  return { count: n, min, max, mean, std: Math.sqrt(variance / n), maxAbs: Math.max(Math.abs(min), Math.abs(max)) };
}

export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
