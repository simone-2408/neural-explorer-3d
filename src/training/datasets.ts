/**
 * Small 2-D datasets generated locally with a seed (deterministic).
 * Features live roughly in [-1, 1]²; labels are class indices.
 */
import { createRng, type Rng } from '../math/random';

export type DatasetId = 'xor' | 'and' | 'or' | 'linear' | 'circles' | 'moons' | 'spiral' | 'blobs';

export interface DatasetInfo {
  id: DatasetId;
  label: string;
  classes: number;
  description: string;
  /** Default noise level (0 … 1). */
  noise: number;
}

export const DATASETS: readonly DatasetInfo[] = [
  { id: 'xor', label: 'XOR', classes: 2, noise: 0.15, description: 'Class 1 where exactly one coordinate is positive. Not linearly separable: needs a hidden layer.' },
  { id: 'and', label: 'AND', classes: 2, noise: 0.15, description: 'Class 1 only where both coordinates are positive. Linearly separable.' },
  { id: 'or', label: 'OR', classes: 2, noise: 0.15, description: 'Class 1 where at least one coordinate is positive. Linearly separable.' },
  { id: 'linear', label: 'Linear', classes: 2, noise: 0.1, description: 'Two classes separated by a straight line (with some label noise).' },
  { id: 'circles', label: 'Circles', classes: 2, noise: 0.1, description: 'A disc surrounded by a ring: the boundary must be curved and closed.' },
  { id: 'moons', label: 'Two moons', classes: 2, noise: 0.12, description: 'Two interleaving half-circles.' },
  { id: 'spiral', label: 'Spiral', classes: 2, noise: 0.08, description: 'Two intertwined spirals: hard, needs a larger or deeper network.' },
  { id: 'blobs', label: 'Three blobs', classes: 3, noise: 0.2, description: 'Three Gaussian clusters: a 3-class problem (softmax output).' },
];

export interface DatasetOptions {
  samples: number;
  /** 0 … 1 */
  noise: number;
  /** Fraction of samples held out as test set (0 … 0.5). */
  testFraction: number;
  seed: number;
}

export const DEFAULT_DATASET_OPTIONS: DatasetOptions = { samples: 200, noise: 0.15, testFraction: 0.2, seed: 1 };

export interface Dataset {
  id: DatasetId;
  info: DatasetInfo;
  options: DatasetOptions;
  /** n × 2 features, row-major. */
  x: Float64Array;
  /** Class index per sample. */
  y: Uint8Array;
  classes: number;
  size: number;
  train: Uint32Array;
  test: Uint32Array;
}

export function datasetInfo(id: DatasetId): DatasetInfo {
  return DATASETS.find((d) => d.id === id)!;
}

function generate(id: DatasetId, n: number, noise: number, rng: Rng): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  const push = (px: number, py: number, label: number) => {
    x.push(Math.max(-1.2, Math.min(1.2, px)), Math.max(-1.2, Math.min(1.2, py)));
    y.push(label);
  };
  switch (id) {
    case 'xor':
    case 'and':
    case 'or': {
      for (let k = 0; k < n; k++) {
        const sx = k % 2 === 0 ? 1 : -1;
        const sy = Math.floor(k / 2) % 2 === 0 ? 1 : -1;
        const px = sx * 0.5 + rng.normal(0, 0.12 + noise * 0.5);
        const py = sy * 0.5 + rng.normal(0, 0.12 + noise * 0.5);
        const a = sx > 0;
        const b = sy > 0;
        const label = id === 'xor' ? Number(a !== b) : id === 'and' ? Number(a && b) : Number(a || b);
        push(px, py, label);
      }
      break;
    }
    case 'linear': {
      for (let k = 0; k < n; k++) {
        const px = rng.uniform(-1, 1);
        const py = rng.uniform(-1, 1);
        let label = Number(0.8 * px - 0.6 * py + 0.1 > 0);
        if (rng.next() < noise * 0.5) label = 1 - label;
        push(px, py, label);
      }
      break;
    }
    case 'circles': {
      for (let k = 0; k < n; k++) {
        const label = k % 2;
        const r = label === 0 ? rng.uniform(0, 0.42) : rng.uniform(0.62, 0.95);
        const t = rng.uniform(0, 2 * Math.PI);
        push(r * Math.cos(t) + rng.normal(0, noise * 0.4), r * Math.sin(t) + rng.normal(0, noise * 0.4), label);
      }
      break;
    }
    case 'moons': {
      for (let k = 0; k < n; k++) {
        const label = k % 2;
        const t = rng.uniform(0, Math.PI);
        const px = label === 0 ? Math.cos(t) : 1 - Math.cos(t);
        const py = label === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        // sklearn-like moons are in [-1, 2] × [-0.5, 1]: rescale to about [-1, 1]
        push((px - 0.5) * 0.66 + rng.normal(0, noise * 0.5), (py - 0.25) * 1.05 + rng.normal(0, noise * 0.5), label);
      }
      break;
    }
    case 'spiral': {
      for (let k = 0; k < n; k++) {
        const label = k % 2;
        const t = (Math.floor(k / 2) / Math.ceil(n / 2)) * 1.0;
        const r = 0.08 + t * 0.92;
        const angle = t * 3.2 * Math.PI + label * Math.PI;
        push(r * Math.cos(angle) + rng.normal(0, noise * 0.35), r * Math.sin(angle) + rng.normal(0, noise * 0.35), label);
      }
      break;
    }
    case 'blobs': {
      const centers = [
        [-0.55, -0.45],
        [0.55, -0.4],
        [0, 0.6],
      ];
      for (let k = 0; k < n; k++) {
        const label = k % 3;
        push(centers[label][0] + rng.normal(0, 0.12 + noise * 0.6), centers[label][1] + rng.normal(0, 0.12 + noise * 0.6), label);
      }
      break;
    }
  }
  return { x, y };
}

export function makeDataset(id: DatasetId, options: Partial<DatasetOptions> = {}): Dataset {
  const info = datasetInfo(id);
  const o: DatasetOptions = { ...DEFAULT_DATASET_OPTIONS, noise: info.noise, ...options };
  const n = Math.max(8, Math.min(1000, Math.round(o.samples)));
  const rng = createRng(o.seed * 7919 + id.length);
  const { x, y } = generate(id, n, o.noise, rng);

  // Shuffled split into train / test.
  const order = Array.from({ length: n }, (_, k) => k);
  for (let k = n - 1; k > 0; k--) {
    const j = Math.floor(rng.next() * (k + 1));
    [order[k], order[j]] = [order[j], order[k]];
  }
  const testCount = Math.round(n * Math.max(0, Math.min(0.5, o.testFraction)));
  return {
    id,
    info,
    options: { ...o, samples: n },
    x: Float64Array.from(x),
    y: Uint8Array.from(y),
    classes: info.classes,
    size: n,
    test: Uint32Array.from(order.slice(0, testCount)).sort(),
    train: Uint32Array.from(order.slice(testCount)).sort(),
  };
}
