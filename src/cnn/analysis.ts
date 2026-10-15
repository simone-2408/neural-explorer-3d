/**
 * Read-outs of the CNN for the inspectors: receptive fields, the image
 * patches that excite a filter most, and how much a learned kernel resembles
 * classic hand-made kernels. These describe numbers; they do not prove what
 * a filter is "for".
 */
import type { ImageSample } from './datasets';
import { runCnn } from './forward';
import type { CnnModel } from './model';

export interface Region {
  /** Rows / columns of the input image, inclusive, clipped to the image. */
  top: number;
  left: number;
  bottom: number;
  right: number;
}

/**
 * The input pixels that can influence the output cell (i, j) of conv layer
 * `layer` (before its pooling): each convolution widens the window by its
 * kernel, each 2×2 pooling doubles it.
 */
export function receptiveField(model: CnnModel, layer: number, i: number, j: number): Region {
  let top = i;
  let bottom = i;
  let left = j;
  let right = j;
  for (let l = layer; l >= 0; l--) {
    const s = model.shapes[l];
    const p = (s.kernel - 1) / 2;
    top -= p;
    left -= p;
    bottom += p;
    right += p;
    if (l > 0 && model.shapes[l - 1].pool !== 'none') {
      top *= 2;
      left *= 2;
      bottom = 2 * bottom + 1;
      right = 2 * right + 1;
    }
  }
  const max = model.config.size - 1;
  return { top: Math.max(0, top), left: Math.max(0, left), bottom: Math.min(max, bottom), right: Math.min(max, right) };
}

export interface TopActivation {
  sample: number;
  i: number;
  j: number;
  value: number;
  region: Region;
}

/** The positions in the samples where filter `filter` of `layer` responds most (one per sample). */
export function topActivations(model: CnnModel, samples: readonly ImageSample[], layer: number, filter: number, k = 6): TopActivation[] {
  const s = model.shapes[layer];
  const S = s.size;
  const best: TopActivation[] = [];
  samples.forEach((sample, n) => {
    const act = runCnn(model, sample.pixels).layers[layer].act;
    let value = -Infinity;
    let at = 0;
    for (let q = 0; q < S * S; q++) {
      const v = act[filter * S * S + q];
      if (v > value) {
        value = v;
        at = q;
      }
    }
    const i = Math.floor(at / S);
    const j = at % S;
    best.push({ sample: n, i, j, value, region: receptiveField(model, layer, i, j) });
  });
  return best.sort((a, b) => b.value - a.value).slice(0, k);
}

/** Classic 3×3 kernels, for comparison with learned ones. */
export const REFERENCE_KERNELS: { name: string; kernel: number[] }[] = [
  { name: 'vertical edge', kernel: [1, 0, -1, 2, 0, -2, 1, 0, -1] },
  { name: 'horizontal edge', kernel: [1, 2, 1, 0, 0, 0, -1, -2, -1] },
  { name: 'diagonal ↘ line', kernel: [2, -1, -1, -1, 2, -1, -1, -1, 2] },
  { name: 'diagonal ↗ line', kernel: [-1, -1, 2, -1, 2, -1, 2, -1, -1] },
  { name: 'horizontal line', kernel: [-1, -1, -1, 2, 2, 2, -1, -1, -1] },
  { name: 'vertical line', kernel: [-1, 2, -1, -1, 2, -1, -1, 2, -1] },
  { name: 'spot (centre vs surround)', kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1] },
];

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/**
 * Cosine similarity of a 3×3 kernel with the reference kernels, sorted by
 * |similarity| (a negative value means the same pattern with the sign flipped).
 */
export function kernelSimilarities(kernel: ArrayLike<number>): { name: string; similarity: number }[] {
  if (kernel.length !== 9) return [];
  return REFERENCE_KERNELS.map((r) => ({ name: r.name, similarity: cosine(kernel, r.kernel) })).sort((a, b) => Math.abs(b.similarity) - Math.abs(a.similarity));
}

/** One channel of one filter as a k×k array. */
export function kernelOf(model: CnnModel, layer: number, filter: number, channel: number): Float64Array {
  const s = model.shapes[layer];
  const W = model.convWeight(layer);
  const k = s.kernel;
  const out = new Float64Array(k * k);
  for (let u = 0; u < k; u++) for (let v = 0; v < k; v++) out[u * k + v] = W[model.weightIndex(layer, filter, channel, u, v)];
  return out;
}
