/**
 * Loss landscape: the loss evaluated on a 2-D *slice* of parameter space,
 *
 *     L(α, β) = L(θ_c + α·d₁ + β·d₂)
 *
 * The real function has one dimension per parameter (37 for the default
 * network): this is only a plane through it. Two choices of plane:
 *  - random directions, normalized per neuron like the weights (Li et al.,
 *    "Visualizing the Loss Landscape of Neural Nets", 2018);
 *  - the two principal components of the training trajectory, so that the
 *    path of the optimizer lies (approximately) in the plane.
 */
import { pca } from '../math/pca';
import { createRng } from '../math/random';
import type { NetworkSpec } from '../neural/config';
import { createTrace, forwardInto } from '../neural/forward';
import type { NetworkModel } from '../neural/NetworkModel';
import type { Dataset } from './datasets';
import { lossFor, sampleLoss } from './loss';

export interface Plane {
  kind: 'random' | 'pca';
  center: Float64Array;
  d1: Float64Array;
  d2: Float64Array;
  /** Fraction of the trajectory variance captured by the plane (PCA plane only). */
  explained?: number;
}

/** Parameters as one vector: [W1, b1, W2, b2, …]. */
export function flattenParameters(model: NetworkModel): Float64Array {
  const parts: number[] = [];
  for (let l = 1; l < model.layerCount; l++) {
    parts.push(...model.weights(l), ...model.biases(l));
  }
  return Float64Array.from(parts);
}

/** Splits a flat vector into per-layer weights and biases (allocating). */
export function unflatten(spec: NetworkSpec, flat: ArrayLike<number>): { weights: Float64Array[]; biases: Float64Array[] } {
  const weights = [new Float64Array(0)];
  const biases = [new Float64Array(0)];
  let k = 0;
  for (let l = 1; l < spec.layers.length; l++) {
    const rows = spec.layers[l].size;
    const cols = spec.layers[l - 1].size;
    weights.push(Float64Array.from({ length: rows * cols }, () => flat[k++]));
    biases.push(Float64Array.from({ length: rows }, () => flat[k++]));
  }
  return { weights, biases };
}

/**
 * Random plane through the current parameters. Each direction is rescaled
 * neuron by neuron to the norm of that neuron's incoming weights + bias, so
 * "one unit" along a direction is a comparable change for every neuron.
 */
export function randomPlane(model: NetworkModel, seed: number): Plane {
  const center = flattenParameters(model);
  const rng = createRng(seed);
  const make = () => {
    const d = new Float64Array(center.length);
    let k = 0;
    for (let l = 1; l < model.layerCount; l++) {
      const rows = model.spec.layers[l].size;
      const cols = model.spec.layers[l - 1].size;
      const W = model.weights(l);
      const b = model.biases(l);
      const wStart = k;
      const bStart = k + rows * cols;
      for (let j = 0; j < rows; j++) {
        let paramNorm = b[j] * b[j];
        let dirNorm = 0;
        const values: number[] = [];
        for (let i = 0; i < cols; i++) {
          paramNorm += W[j * cols + i] ** 2;
          const v = rng.normal();
          values.push(v);
          dirNorm += v * v;
        }
        const vb = rng.normal();
        dirNorm += vb * vb;
        const scale = Math.sqrt(paramNorm) / Math.sqrt(dirNorm || 1);
        for (let i = 0; i < cols; i++) d[wStart + j * cols + i] = values[i] * scale;
        d[bStart + j] = vb * scale;
      }
      k = bStart + rows;
    }
    return d;
  };
  return { kind: 'random', center, d1: make(), d2: make() };
}

/** Plane of the two principal components of a trajectory, centred on its mean. */
export function trajectoryPlane(trajectory: readonly Float64Array[]): Plane | null {
  if (trajectory.length < 3) return null;
  const result = pca(trajectory, 2);
  if (result.components.length < 2) return null;
  // Scale the unit directions to the spread of the trajectory so that α, β ∈ [-1, 1] covers it.
  const s1 = Math.sqrt(result.variances[0]) * 2.2 || 1;
  const s2 = Math.sqrt(result.variances[1]) * 2.2 || 1;
  const spread = Math.max(s1, s2);
  return {
    kind: 'pca',
    center: result.mean,
    d1: result.components[0].map((v) => v * spread),
    d2: result.components[1].map((v) => v * spread),
    explained: result.totalVariance > 0 ? (result.variances[0] + result.variances[1]) / result.totalVariance : 1,
  };
}

/** (α, β) of a parameter vector: least-squares projection onto the plane. */
export function projectOnPlane(plane: Plane, point: ArrayLike<number>): [number, number] {
  let a11 = 0;
  let a12 = 0;
  let a22 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let k = 0; k < plane.center.length; k++) {
    const r = point[k] - plane.center[k];
    a11 += plane.d1[k] * plane.d1[k];
    a12 += plane.d1[k] * plane.d2[k];
    a22 += plane.d2[k] * plane.d2[k];
    b1 += plane.d1[k] * r;
    b2 += plane.d2[k] * r;
  }
  const det = a11 * a22 - a12 * a12 || 1e-12;
  return [(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det];
}

export interface LandscapeGrid {
  plane: Plane;
  size: number;
  /** α and β span [-range, range]. */
  range: number;
  /** size × size losses, row = β index, column = α index. */
  values: Float64Array;
  min: number;
  max: number;
  /** Rows computed so far (the grid can be filled progressively). */
  rowsDone: number;
}

export function createGrid(plane: Plane, size: number, range: number): LandscapeGrid {
  return { plane, size, range, values: new Float64Array(size * size), min: Infinity, max: -Infinity, rowsDone: 0 };
}

/** Mean loss over `indices` for the parameters θ_c + α d₁ + β d₂. */
export function lossOnPlane(spec: NetworkSpec, dataset: Dataset, indices: ArrayLike<number>, plane: Plane, alpha: number, beta: number): number {
  const flat = new Float64Array(plane.center.length);
  for (let k = 0; k < flat.length; k++) flat[k] = plane.center[k] + alpha * plane.d1[k] + beta * plane.d2[k];
  return meanLoss(spec, dataset, indices, flat);
}

export function meanLoss(spec: NetworkSpec, dataset: Dataset, indices: ArrayLike<number>, flat: ArrayLike<number>): number {
  const { weights, biases } = unflatten(spec, flat);
  const trace = createTrace(spec.layers.map((l) => l.size));
  const loss = lossFor(spec);
  const x = new Float64Array(2);
  let total = 0;
  for (let b = 0; b < indices.length; b++) {
    const k = indices[b];
    x[0] = dataset.x[2 * k];
    x[1] = dataset.x[2 * k + 1];
    forwardInto(spec, weights, biases, x, trace);
    total += sampleLoss(loss, spec, trace.a[spec.layers.length - 1], dataset.y[k]);
  }
  return total / Math.max(1, indices.length);
}

/** Computes grid rows until `budgetMs` is spent; returns true when the grid is complete. */
export function fillGrid(grid: LandscapeGrid, spec: NetworkSpec, dataset: Dataset, indices: ArrayLike<number>, budgetMs: number): boolean {
  const start = performance.now();
  const { size, range, plane } = grid;
  while (grid.rowsDone < size) {
    const r = grid.rowsDone;
    const beta = -range + (2 * range * r) / (size - 1);
    for (let c = 0; c < size; c++) {
      const alpha = -range + (2 * range * c) / (size - 1);
      const v = lossOnPlane(spec, dataset, indices, plane, alpha, beta);
      const value = Number.isFinite(v) ? v : 1e6;
      grid.values[r * size + c] = value;
      grid.min = Math.min(grid.min, value);
      grid.max = Math.max(grid.max, value);
    }
    grid.rowsDone++;
    if (performance.now() - start > budgetMs) break;
  }
  return grid.rowsDone >= size;
}

/** Bilinear interpolation of the grid at (α, β). */
export function sampleGrid(grid: LandscapeGrid, alpha: number, beta: number): number {
  const { size, range, values } = grid;
  const fx = Math.max(0, Math.min(size - 1, ((alpha + range) / (2 * range)) * (size - 1)));
  const fy = Math.max(0, Math.min(size - 1, ((beta + range) / (2 * range)) * (size - 1)));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(size - 1, x0 + 1);
  const y1 = Math.min(size - 1, y0 + 1);
  const tx = fx - x0;
  const ty = fy - y0;
  const top = values[y0 * size + x0] * (1 - tx) + values[y0 * size + x1] * tx;
  const bottom = values[y1 * size + x0] * (1 - tx) + values[y1 * size + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}
