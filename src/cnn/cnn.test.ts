import { describe, expect, it } from 'vitest';
import { DEFAULT_OPTIMIZER } from '../training/optimizers';
import { kernelSimilarities, receptiveField, topActivations } from './analysis';
import { drawSample, IMAGE_SIZE, makeImageDataset } from './datasets';
import { backwardCnn, runCnn, zeroGrads } from './forward';
import { CnnModel, DEFAULT_CNN, type CnnConfig } from './model';
import { CnnTrainer, evaluate } from './trainer';

const config = (patch: Partial<CnnConfig> = {}): CnnConfig => ({ ...structuredClone(DEFAULT_CNN), classes: 4, ...patch });

describe('datasets', () => {
  it('are reproducible, balanced 16×16 images in [0, 1]', () => {
    const a = makeImageDataset('digits', 3, 40, 10);
    const b = makeImageDataset('digits', 3, 40, 10);
    expect(a.train[7].pixels).toEqual(b.train[7].pixels);
    expect(a.train.length).toBe(40);
    expect(a.train.filter((s) => s.label === 2)).toHaveLength(4);
    for (const s of a.train) {
      expect(s.pixels).toHaveLength(IMAGE_SIZE * IMAGE_SIZE);
      expect(Math.min(...s.pixels)).toBeGreaterThanOrEqual(0);
      expect(Math.max(...s.pixels)).toBeLessThanOrEqual(1);
    }
    // a horizontal stroke has more ink along one row than along any column
    const h = drawSample('lines', 0, 11);
    const rows = Array.from({ length: 16 }, (_, r) => h.slice(r * 16, r * 16 + 16).reduce((x, y) => x + y, 0));
    const cols = Array.from({ length: 16 }, (_, c) => Array.from({ length: 16 }, (_, r) => h[r * 16 + c]).reduce((x, y) => x + y, 0));
    expect(Math.max(...rows)).toBeGreaterThan(Math.max(...cols) * 2);
  });
});

describe('CNN', () => {
  it('has the shapes and parameter counts of the PyTorch model', () => {
    const model = new CnnModel(config());
    expect(model.shapes.map((s) => [s.inChannels, s.outChannels, s.size, s.pooledSize])).toEqual([
      [1, 4, 16, 8],
      [4, 8, 8, 4],
    ]);
    expect(model.flatSize).toBe(128);
    // conv1 4·1·3·3 + 4, conv2 8·4·3·3 + 8, dense 4·128 + 4
    expect(model.parameterCount()).toBe(40 + 296 + 516);
    expect(model.fullyConnectedEquivalent(0)).toBe(256 * 1024 + 1024);
    const trace = runCnn(model, drawSample('lines', 1, 5));
    expect(trace.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('computes a convolution cell exactly as the formula says', () => {
    const model = new CnnModel(config({ convs: [{ filters: 2, kernel: 3, pool: 'none' }] }));
    const image = drawSample('shapes', 2, 9);
    const trace = runCnn(model, image);
    const W = model.convWeight(0);
    const [i, j, f] = [5, 7, 1];
    let z = model.convBias(0)[f];
    for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) z += W[model.weightIndex(0, f, 0, u, v)] * image[(i + u - 1) * 16 + (j + v - 1)];
    expect(trace.layers[0].pre[(f * 16 + i) * 16 + j]).toBeCloseTo(z, 12);
    // a corner cell only sees the pixels inside the image (zero padding)
    let corner = model.convBias(0)[0];
    for (let u = 1; u < 3; u++) for (let v = 1; v < 3; v++) corner += W[model.weightIndex(0, 0, 0, u, v)] * image[(u - 1) * 16 + (v - 1)];
    expect(trace.layers[0].pre[0]).toBeCloseTo(corner, 12);
  });

  const variants: [string, Partial<CnnConfig>][] = [
    ['max pooling, ReLU', {}],
    ['average pooling, tanh, 5×5 kernels', { activation: 'tanh', convs: [{ filters: 3, kernel: 5, pool: 'avg' }, { filters: 2, kernel: 3, pool: 'avg' }] }],
    ['no pooling, one layer', { convs: [{ filters: 2, kernel: 3, pool: 'none' }] }],
  ];
  for (const [name, patch] of variants) {
    it(`backpropagation matches finite differences (${name})`, () => {
      const model = new CnnModel(config(patch), 4);
      const image = drawSample('shapes', 3, 21);
      const label = 2;
      const grads = zeroGrads(model);
      backwardCnn(model, runCnn(model, image), label, grads);
      const loss = () => -Math.log(runCnn(model, image).probs[label]);
      const h = 1e-6;
      for (let k = 0; k < model.params.length; k++) {
        const p = model.params[k];
        for (const idx of [0, Math.floor(p.length / 2), p.length - 1]) {
          const keep = p[idx];
          p[idx] = keep + h;
          const plus = loss();
          p[idx] = keep - h;
          const minus = loss();
          p[idx] = keep;
          expect(grads[k][idx]).toBeCloseTo((plus - minus) / (2 * h), 5);
        }
      }
    });
  }

  it('learns the lines dataset', () => {
    const dataset = makeImageDataset('lines', 2, 400, 100);
    const model = new CnnModel(config(), 3);
    const trainer = new CnnTrainer(model, dataset, { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.01 }, 16, 5);
    const before = evaluate(model, dataset.test);
    for (let s = 0; s < trainer.stepsPerEpoch * 6; s++) trainer.trainStep();
    const after = evaluate(model, dataset.test);
    expect(after.loss).toBeLessThan(before.loss * 0.5);
    expect(after.accuracy).toBeGreaterThan(0.9);
  });
});

describe('analysis', () => {
  it('receptive fields grow with depth and pooling', () => {
    const model = new CnnModel(config());
    expect(receptiveField(model, 0, 5, 5)).toEqual({ top: 4, left: 4, bottom: 6, right: 6 });
    // conv2 cell (2, 2): 3×3 in pool1 coordinates (1..3) → conv1 rows 2..7 → image rows 1..8
    expect(receptiveField(model, 1, 2, 2)).toEqual({ top: 1, left: 1, bottom: 8, right: 8 });
  });

  it('compares kernels with reference kernels and finds top activations', () => {
    const sims = kernelSimilarities([1, 0, -1, 2, 0, -2, 1, 0, -1]);
    expect(sims[0]).toEqual({ name: 'vertical edge', similarity: 1 });
    const model = new CnnModel(config());
    const dataset = makeImageDataset('lines', 1, 20, 4);
    const top = topActivations(model, dataset.train, 0, 1, 3);
    expect(top).toHaveLength(3);
    expect(top[0].value).toBeGreaterThanOrEqual(top[2].value);
  });
});
