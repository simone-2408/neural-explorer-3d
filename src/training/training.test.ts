import { describe, expect, it } from 'vitest';
import { ACTIVATION_IDS, type ActivationId } from '../math/activations';
import { createRng } from '../math/random';
import { specFromConfig, type NetworkConfig } from '../neural/config';
import { createTrace, forwardInto } from '../neural/forward';
import { NetworkModel } from '../neural/NetworkModel';
import { backpropSample, createGradients, createScratch } from './backprop';
import { DATASETS, makeDataset } from './datasets';
import { lossFor, sampleLoss } from './loss';
import { DEFAULT_OPTIMIZER, Optimizer } from './optimizers';
import { Trainer } from './Trainer';

function randomModel(config: NetworkConfig, seed: number): NetworkModel {
  const model = NetworkModel.fromConfig(config, { init: { scheme: 'uniform', seed } });
  return model;
}

/** Loss of one sample as a function of the parameters (for finite differences). */
function lossAt(model: NetworkModel, x: number[], label: number): number {
  const { spec } = model;
  const trace = createTrace(spec.layers.map((l) => l.size));
  const W = Array.from({ length: spec.layers.length }, (_, l) => model.weights(l) as Float64Array);
  const B = Array.from({ length: spec.layers.length }, (_, l) => model.biases(l) as Float64Array);
  forwardInto(spec, W, B, x, trace);
  return sampleLoss(lossFor(spec), spec, trace.a[spec.layers.length - 1], label);
}

describe('backpropagation', () => {
  const hidden = ACTIVATION_IDS.filter((a) => a !== 'softmax' && a !== 'relu' && a !== 'leaky_relu') as ActivationId[];
  const outputs: { activation: ActivationId; size: number }[] = [
    { activation: 'sigmoid', size: 1 }, // BCE
    { activation: 'softmax', size: 3 }, // cross-entropy
    { activation: 'tanh', size: 2 }, // MSE
    { activation: 'linear', size: 1 }, // MSE
  ];

  for (const out of outputs) {
    for (const act of hidden) {
      it(`matches finite differences (hidden ${act}, output ${out.activation}×${out.size})`, () => {
        const model = randomModel(
          {
            layers: [
              { type: 'input', size: 2 },
              { type: 'dense', size: 3, activation: act },
              { type: 'dense', size: 3, activation: act },
              { type: 'output', size: out.size, activation: out.activation },
            ],
          },
          11,
        );
        const { spec } = model;
        const x = [0.37, -0.81];
        const label = out.size === 3 ? 2 : 1;
        const trace = createTrace(spec.layers.map((l) => l.size));
        const W = Array.from({ length: spec.layers.length }, (_, l) => model.weights(l) as Float64Array);
        const B = Array.from({ length: spec.layers.length }, (_, l) => model.biases(l) as Float64Array);
        forwardInto(spec, W, B, x, trace);
        const grads = createGradients(spec);
        backpropSample(spec, W, trace, lossFor(spec), label, grads, createScratch(spec));

        const h = 1e-6;
        for (let l = 1; l < spec.layers.length; l++) {
          const cols = spec.layers[l - 1].size;
          for (let j = 0; j < spec.layers[l].size; j++) {
            for (let i = 0; i < cols; i++) {
              const w = model.weight(l, j, i);
              model.setWeight(l, j, i, w + h);
              const plus = lossAt(model, x, label);
              model.setWeight(l, j, i, w - h);
              const minus = lossAt(model, x, label);
              model.setWeight(l, j, i, w);
              expect(grads.dW[l][j * cols + i]).toBeCloseTo((plus - minus) / (2 * h), 5);
            }
            const b = model.bias(l, j);
            model.setBias(l, j, b + h);
            const plus = lossAt(model, x, label);
            model.setBias(l, j, b - h);
            const minus = lossAt(model, x, label);
            model.setBias(l, j, b);
            expect(grads.db[l][j]).toBeCloseTo((plus - minus) / (2 * h), 5);
          }
        }
      });
    }
  }
});

describe('optimizers', () => {
  it('SGD, momentum and Adam follow the PyTorch update rules', () => {
    const g = [Float64Array.from([0.5, -1])];
    const sgd = new Optimizer({ ...DEFAULT_OPTIMIZER, id: 'sgd', learningRate: 0.1 });
    const p1 = [Float64Array.from([1, 1])];
    sgd.step(p1, g);
    expect(Array.from(p1[0])).toEqual([0.95, 1.1]);

    const momentum = new Optimizer({ ...DEFAULT_OPTIMIZER, id: 'momentum', learningRate: 0.1, momentum: 0.9 });
    const p2 = [Float64Array.from([1, 1])];
    momentum.step(p2, g); // v = g
    momentum.step(p2, g); // v = 0.9 g + g = 1.9 g
    expect(p2[0][0]).toBeCloseTo(1 - 0.1 * 0.5 - 0.1 * 0.95, 12);

    const adam = new Optimizer({ ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.01 });
    const p3 = [Float64Array.from([1, 1])];
    adam.step(p3, g); // first Adam step moves each parameter by ≈ lr · sign(g)
    expect(p3[0][0]).toBeCloseTo(0.99, 6);
    expect(p3[0][1]).toBeCloseTo(1.01, 6);
  });
});

describe('Trainer', () => {
  it('learns XOR with a small MLP', () => {
    const dataset = makeDataset('xor', { samples: 200, noise: 0.05, seed: 3 });
    const model = NetworkModel.fromConfig(
      {
        layers: [
          { type: 'input', size: 2 },
          { type: 'dense', size: 6, activation: 'tanh' },
          { type: 'output', size: 1, activation: 'sigmoid' },
        ],
      },
      { init: { scheme: 'xavier', seed: 5 } },
    );
    const trainer = new Trainer(model, dataset, { optimizer: { ...DEFAULT_OPTIMIZER, id: 'adam', learningRate: 0.05 }, batchSize: 16, seed: 1 });
    const before = trainer.evaluate();
    trainer.runSteps(600);
    const after = trainer.evaluate();
    expect(after.loss).toBeLessThan(before.loss);
    expect(after.accuracy).toBeGreaterThan(0.95);
    expect(trainer.evaluate(dataset.test).accuracy).toBeGreaterThan(0.9);
  });

  it('visits every training sample once per epoch', () => {
    const dataset = makeDataset('moons', { samples: 50, testFraction: 0.2, seed: 2 });
    const model = NetworkModel.fromConfig({ layers: [{ type: 'input', size: 2 }, { type: 'output', size: 1, activation: 'sigmoid' }] });
    const trainer = new Trainer(model, dataset, { optimizer: DEFAULT_OPTIMIZER, batchSize: 7, seed: 4 });
    trainer.recordBatches = true;
    const steps = trainer.stepsPerEpoch;
    for (let k = 0; k < steps; k++) trainer.step();
    const seen = trainer.batchLog.flat().sort((a, b) => a - b);
    expect(seen).toEqual(Array.from(dataset.train));
    expect(trainer.epochCount).toBe(1);
  });

  it('generates every dataset deterministically', () => {
    for (const info of DATASETS) {
      const a = makeDataset(info.id, { seed: 9 });
      const b = makeDataset(info.id, { seed: 9 });
      expect(Array.from(a.x)).toEqual(Array.from(b.x));
      expect(new Set(a.y).size).toBe(info.classes);
      expect(a.train.length + a.test.length).toBe(a.size);
    }
    const rng = createRng(1);
    expect(rng.next()).toBeGreaterThanOrEqual(0);
  });

  it('builds specs for multi-class datasets with softmax and cross-entropy', () => {
    const spec = specFromConfig({ layers: [{ type: 'input', size: 2 }, { type: 'output', size: 3, activation: 'softmax' }] });
    expect(lossFor(spec)).toBe('cross_entropy');
  });
});
