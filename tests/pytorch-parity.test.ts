/**
 * Cross-checks the TypeScript math core against PyTorch.
 * Run with `npm run test:parity` (needs python3 with torch; skipped otherwise).
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { NetworkConfig } from '../src/neural/config';
import { NetworkModel } from '../src/neural/NetworkModel';
import { PRESETS } from '../src/neural/presets';
import { applyLoadedModel, deserializeModel, serializeModel, stringifyModel } from '../src/neural/serialization';
import { makeDataset, type DatasetId } from '../src/training/datasets';
import { DEFAULT_OPTIMIZER, type OptimizerId } from '../src/training/optimizers';
import { Trainer } from '../src/training/Trainer';
import { drawSample, type ImageDatasetId } from '../src/cnn/datasets';
import { backwardCnn, runCnn, zeroGrads } from '../src/cnn/forward';
import { CnnModel, type CnnConfig } from '../src/cnn/model';
import { backwardRnn, runRnn, zeroRnnGrads } from '../src/rnn/forward';
import { RnnModel, type CellType } from '../src/rnn/model';

const python = process.env.PYTHON ?? 'python3';
const hasTorch = spawnSync(python, ['-c', 'import torch'], { stdio: 'ignore' }).status === 0;
const bridge = join(process.cwd(), 'python', 'mlp_bridge.py');
const convBridge = join(process.cwd(), 'python', 'cnn_rnn_bridge.py');

/** Runs cnn_rnn_bridge.py on case files and returns the largest differences found for each. */
function verifyCases(files: string[]): Record<string, { logits: number; loss: number; grads: number }> {
  const result = spawnSync(python, [convBridge, 'verify', ...files], { encoding: 'utf8' });
  expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  console.log(result.stdout);
  return JSON.parse(result.stdout);
}

/** Uses every activation function, including the ones no preset uses. */
const ALL_ACTIVATIONS: NetworkConfig = {
  name: 'All activations',
  layers: [
    { type: 'input', size: 3 },
    { type: 'dense', size: 6, activation: 'leaky_relu' },
    { type: 'dense', size: 5, activation: 'tanh' },
    { type: 'dense', size: 5, activation: 'gelu' },
    { type: 'dense', size: 4, activation: 'sigmoid' },
    { type: 'dense', size: 4, activation: 'linear' },
    { type: 'output', size: 3, activation: 'softmax' },
  ],
};

describe.skipIf(!hasTorch)('PyTorch parity', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ne3d-parity-'));

  it('the browser forward pass equals PyTorch for every preset and activation', () => {
    const files: string[] = [];
    const configs = [...PRESETS.map((p) => ({ id: p.id, config: p.config, input: p.input?.(), init: p.init })), { id: 'all-activations', config: ALL_ACTIVATIONS, input: [0.9, -1.3, 0.4], init: { seed: 3 } }];
    for (const { id, config, input, init } of configs) {
      const model = NetworkModel.fromConfig(config, { input, init });
      // Use larger weights in one layer so activations leave their linear regime.
      model.setWeight(1, 0, 0, 2.5);
      const file = join(dir, `${id}.json`);
      writeFileSync(file, stringifyModel(serializeModel(model)));
      files.push(file);
    }
    const result = spawnSync(python, [bridge, 'verify', ...files], { encoding: 'utf8' });
    console.log(result.stdout);
    expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
  }, 120_000);

  it('training steps (loss, backprop, SGD / momentum / Adam) equal PyTorch', () => {
    const cases: { optimizer: OptimizerId; config: NetworkConfig; dataset: DatasetId }[] = [
      { optimizer: 'sgd', dataset: 'xor', config: { layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 5, activation: 'tanh' }, { type: 'output', size: 1, activation: 'sigmoid' }] } },
      { optimizer: 'momentum', dataset: 'moons', config: { layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 6, activation: 'relu' }, { type: 'dense', size: 4, activation: 'gelu' }, { type: 'output', size: 1, activation: 'sigmoid' }] } },
      { optimizer: 'adam', dataset: 'blobs', config: { layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 5, activation: 'leaky_relu' }, { type: 'output', size: 3, activation: 'softmax' }] } },
      { optimizer: 'adam', dataset: 'circles', config: { layers: [{ type: 'input', size: 2 }, { type: 'dense', size: 4, activation: 'sigmoid' }, { type: 'output', size: 1, activation: 'tanh' }] } },
    ];
    for (const [n, c] of cases.entries()) {
      const dataset = makeDataset(c.dataset, { samples: 60, seed: n + 1 });
      const model = NetworkModel.fromConfig(c.config, { init: { seed: 10 + n } });
      const initial = serializeModel(model);
      const trainer = new Trainer(model, dataset, { optimizer: { ...DEFAULT_OPTIMIZER, id: c.optimizer, learningRate: 0.05, weightDecay: n === 1 ? 0.01 : 0 }, batchSize: 8, seed: 3 });
      trainer.recordBatches = true;
      const losses: number[] = [];
      for (let k = 0; k < 25; k++) losses.push(trainer.step());
      const file = join(dir, `train-${n}.json`);
      writeFileSync(
        file,
        JSON.stringify({ model: initial, x: Array.from(dataset.x), y: Array.from(dataset.y), batches: trainer.batchLog, optimizer: trainer.config.optimizer }),
      );
      const result = spawnSync(python, [bridge, 'train', file], { encoding: 'utf8' });
      expect(result.status, result.stderr).toBe(0);
      const torchResult = JSON.parse(result.stdout) as { losses: number[]; parameters: { layer: number; weight: number[][]; bias: number[] }[] };
      torchResult.losses.forEach((loss, k) => expect(losses[k]).toBeCloseTo(loss, 9));
      for (const p of torchResult.parameters) {
        p.weight.flat().forEach((w, k) => expect(model.weights(p.layer)[k]).toBeCloseTo(w, 9));
        p.bias.forEach((b, k) => expect(model.biases(p.layer)[k]).toBeCloseTo(b, 9));
      }
    }
  }, 120_000);

  it('the CNN equals PyTorch: logits, loss and every gradient', () => {
    const configs: [ImageDatasetId, CnnConfig][] = [
      ['lines', { size: 16, activation: 'relu', classes: 4, convs: [{ filters: 4, kernel: 3, pool: 'max' }, { filters: 8, kernel: 3, pool: 'max' }] }],
      ['digits', { size: 16, activation: 'tanh', classes: 10, convs: [{ filters: 3, kernel: 5, pool: 'avg' }, { filters: 5, kernel: 3, pool: 'max' }] }],
      ['shapes', { size: 16, activation: 'relu', classes: 4, convs: [{ filters: 2, kernel: 3, pool: 'none' }] }],
    ];
    const files = configs.map(([dataset, config], n) => {
      const model = new CnnModel(config, 20 + n);
      const labels = [0, 1, 2, 3, 1];
      const images = labels.map((label, k) => drawSample(dataset, label, 100 * n + k));
      const grads = zeroGrads(model);
      let loss = 0;
      const logits: number[] = [];
      images.forEach((image, k) => {
        const trace = runCnn(model, image);
        logits.push(...trace.logits);
        loss += backwardCnn(model, trace, labels[k], grads, 1 / images.length) / images.length;
      });
      const file = join(dir, `cnn-${n}.json`);
      writeFileSync(file, JSON.stringify({ kind: 'cnn', config, params: model.params.map((p) => Array.from(p)), images: images.map((i) => Array.from(i)), labels, expected: { logits, loss, grads: grads.map((g) => Array.from(g)) } }));
      return file;
    });
    for (const [file, diff] of Object.entries(verifyCases(files))) {
      expect(diff.logits, file).toBeLessThan(1e-12);
      expect(diff.loss, file).toBeLessThan(1e-12);
      expect(diff.grads, file).toBeLessThan(1e-12);
    }
  }, 120_000);

  it('the vanilla RNN and the GRU equal torch.nn.RNN / nn.GRU: logits, loss, BPTT gradients', () => {
    const files = (['rnn', 'gru'] as CellType[]).map((cell, n) => {
      const model = new RnnModel({ cell, inputSize: 6, hiddenSize: 7, outputSize: 5, updateGateBias: cell === 'gru' ? 1 : 0 }, 30 + n);
      const inputs = [
        [0, 3, 5, 1, 2, 4, 4, 0],
        [5, 5, 1, 0, 3, 2, 1, 1],
      ];
      const targets = [
        [1, -1, 3, 0, -1, 2, 4, 1],
        [-1, -1, -1, -1, -1, -1, -1, 3],
      ];
      const scored = targets.flat().filter((t) => t >= 0).length;
      const grads = zeroRnnGrads(model);
      let loss = 0;
      const logits: number[] = [];
      inputs.forEach((seq, b) => {
        const trace = runRnn(model, seq);
        for (const step of trace.steps) logits.push(...step.logits);
        loss += backwardRnn(model, trace, targets[b], grads, 1 / scored).loss;
      });
      const file = join(dir, `rnn-${cell}.json`);
      writeFileSync(file, JSON.stringify({ kind: 'rnn', config: model.config, params: model.params.map((p) => Array.from(p)), inputs, targets, expected: { logits, loss, grads: grads.map((g) => Array.from(g)) } }));
      return file;
    });
    for (const [file, diff] of Object.entries(verifyCases(files))) {
      expect(diff.logits, file).toBeLessThan(1e-12);
      expect(diff.loss, file).toBeLessThan(1e-12);
      expect(diff.grads, file).toBeLessThan(1e-12);
    }
  }, 120_000);

  it('a model created in PyTorch loads in the app with the same output', () => {
    const file = join(dir, 'from-torch.json');
    const result = spawnSync(python, [bridge, 'export-demo', file, '--seed', '5'], { encoding: 'utf8' });
    expect(result.status, result.stderr).toBe(0);
    const raw = JSON.parse(readFileSync(file, 'utf8'));
    const loaded = deserializeModel(raw);
    expect(loaded.ok).toBe(true);
    if (!loaded.ok) return;
    const model = NetworkModel.fromConfig({ layers: [{ type: 'input', size: 1 }, { type: 'output', size: 1 }] });
    applyLoadedModel(model, loaded.model);
    raw.output.forEach((value: number, j: number) => expect(model.output[j]).toBeCloseTo(value, 12));
  }, 60_000);
});
