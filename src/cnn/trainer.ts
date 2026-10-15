/**
 * Mini-batch training of the CNN with the optimizers of the Training tab
 * (PyTorch semantics), mean cross-entropy over the batch like
 * nn.CrossEntropyLoss.
 */
import { createRng, type Rng } from '../math/random';
import { Optimizer, type OptimizerSettings } from '../training/optimizers';
import type { ImageDataset, ImageSample } from './datasets';
import { backwardCnn, runCnn, zeroGrads } from './forward';
import type { CnnModel } from './model';

export interface CnnMetrics {
  loss: number;
  accuracy: number;
}

export interface CnnHistoryPoint {
  step: number;
  epoch: number;
  train: CnnMetrics;
  test: CnnMetrics;
}

export function evaluate(model: CnnModel, samples: readonly ImageSample[]): CnnMetrics {
  let loss = 0;
  let correct = 0;
  for (const s of samples) {
    const { probs } = runCnn(model, s.pixels);
    loss -= Math.log(Math.max(probs[s.label], 1e-300));
    let best = 0;
    for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
    if (best === s.label) correct++;
  }
  return { loss: loss / Math.max(1, samples.length), accuracy: correct / Math.max(1, samples.length) };
}

/** Confusion counts [true class][predicted class]. */
export function confusion(model: CnnModel, samples: readonly ImageSample[]): number[][] {
  const C = model.config.classes;
  const m = Array.from({ length: C }, () => new Array<number>(C).fill(0));
  for (const s of samples) {
    const { probs } = runCnn(model, s.pixels);
    let best = 0;
    for (let k = 1; k < C; k++) if (probs[k] > probs[best]) best = k;
    m[s.label][best]++;
  }
  return m;
}

export class CnnTrainer {
  readonly model: CnnModel;
  readonly dataset: ImageDataset;
  /** Replaceable: changing the optimizer keeps the weights and the history. */
  optimizer: Optimizer;
  batchSize: number;
  step = 0;
  epoch = 0;
  readonly history: CnnHistoryPoint[] = [];
  /** Gradients of the last step (batch mean), for the inspectors. */
  lastGrads: Float64Array[] | null = null;
  private order: number[] = [];
  private cursor = 0;
  private readonly rng: Rng;

  constructor(model: CnnModel, dataset: ImageDataset, optimizer: OptimizerSettings, batchSize = 16, seed = 1) {
    this.model = model;
    this.dataset = dataset;
    this.optimizer = new Optimizer(optimizer);
    this.batchSize = batchSize;
    this.rng = createRng(seed);
    this.shuffle();
  }

  private shuffle(): void {
    this.order = this.dataset.train.map((_, i) => i);
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng.next() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
    this.cursor = 0;
  }

  get stepsPerEpoch(): number {
    return Math.ceil(this.dataset.train.length / this.batchSize);
  }

  /** One mini-batch step; returns the mean loss of the batch. */
  trainStep(): number {
    const grads = zeroGrads(this.model);
    const batch: ImageSample[] = [];
    for (let b = 0; b < this.batchSize; b++) {
      if (this.cursor >= this.order.length) {
        this.epoch++;
        this.shuffle();
      }
      batch.push(this.dataset.train[this.order[this.cursor++]]);
    }
    let loss = 0;
    for (const s of batch) loss += backwardCnn(this.model, runCnn(this.model, s.pixels), s.label, grads, 1 / batch.length);
    this.optimizer.step(this.model.params, grads);
    this.lastGrads = grads;
    this.step++;
    return loss / batch.length;
  }

  /** Evaluates on (a subset of) the train and test sets and appends a history point. */
  record(subset = 200): CnnHistoryPoint {
    const point = {
      step: this.step,
      epoch: this.epoch + this.cursor / this.order.length,
      train: evaluate(this.model, this.dataset.train.slice(0, subset)),
      test: evaluate(this.model, this.dataset.test),
    };
    this.history.push(point);
    return point;
  }
}
