/**
 * Trainer — mini-batch gradient descent on a NetworkModel.
 *
 * Pure logic (no DOM, no three.js): it can run in tests, in a Web Worker or
 * be reproduced step by step in PyTorch (the batch order is recorded).
 * Parameters are updated in place through NetworkModel.updateParameters(), so
 * the 3D view and the inspector see every step.
 */
import { createRng, type Rng } from '../math/random';
import { createTrace, forwardInto, type ForwardTrace } from '../neural/forward';
import type { NetworkModel } from '../neural/NetworkModel';
import { backpropSample, createGradients, createScratch, scaleGradients, zeroGradients, type Gradients } from './backprop';
import type { Dataset } from './datasets';
import { lossFor, predictedClass, sampleLoss, type LossId } from './loss';
import { Optimizer, type OptimizerSettings } from './optimizers';

export interface TrainerConfig {
  optimizer: OptimizerSettings;
  /** Samples per step (≥ the training set size means full-batch gradient descent). */
  batchSize: number;
  /** Seed of the shuffling of the training set. */
  seed: number;
}

export interface Metrics {
  loss: number;
  accuracy: number;
}

export interface HistoryPoint {
  step: number;
  epoch: number;
  train: Metrics;
  test: Metrics | null;
  /** ‖∂L/∂W^(l)‖ per layer at that step ([0] unused). */
  gradNorms: number[];
}

export class Trainer {
  readonly model: NetworkModel;
  private datasetValue: Dataset;
  private configValue: TrainerConfig;
  private optimizer: Optimizer;
  private grads: Gradients;
  private scratch: Float64Array[];
  private trace: ForwardTrace;
  private order = new Uint32Array(0);
  private cursor = 0;
  private rng: Rng;
  private hasGradients = false;
  private prevWeights: Float64Array[] | null = null;
  private prevBiases: Float64Array[] | null = null;

  stepCount = 0;
  /** Completed passes over the training set. */
  epochCount = 0;
  readonly history: HistoryPoint[] = [];
  lastBatch: Uint32Array = new Uint32Array(0);
  lastBatchLoss = NaN;
  /** When true, every batch (as sample indices) is appended to `batchLog`. */
  recordBatches = false;
  readonly batchLog: number[][] = [];

  constructor(model: NetworkModel, dataset: Dataset, config: TrainerConfig) {
    this.model = model;
    this.datasetValue = dataset;
    this.configValue = { ...config, optimizer: { ...config.optimizer } };
    this.optimizer = new Optimizer(config.optimizer);
    this.rng = createRng(config.seed);
    this.grads = createGradients(model.spec);
    this.scratch = createScratch(model.spec);
    this.trace = createTrace(model.spec.layers.map((l) => l.size));
    this.reshuffle();
  }

  get dataset(): Dataset {
    return this.datasetValue;
  }

  get config(): TrainerConfig {
    return this.configValue;
  }

  get loss(): LossId {
    return lossFor(this.model.spec);
  }

  get optimizerSteps(): number {
    return this.optimizer.steps;
  }

  optimizerState(): ReturnType<Optimizer['exportState']> {
    return this.optimizer.exportState();
  }

  /** Restores counters, history and optimizer state saved with a session. */
  restore(state: { stepCount: number; epochCount: number; history: HistoryPoint[]; optimizer?: ReturnType<Optimizer['exportState']> }): void {
    this.stepCount = state.stepCount;
    this.epochCount = state.epochCount;
    this.history.length = 0;
    this.history.push(...state.history);
    if (state.optimizer) this.optimizer.importState(state.optimizer);
  }

  /** Last gradients (batch averages) or null before the first step. */
  gradients(): Gradients | null {
    return this.hasGradients ? this.grads : null;
  }

  previousWeights(): Float64Array[] | null {
    return this.prevWeights;
  }

  previousBiases(): Float64Array[] | null {
    return this.prevBiases;
  }

  setDataset(dataset: Dataset): void {
    this.datasetValue = dataset;
    this.reshuffle();
  }

  /** Changing the optimizer type restarts its internal state (momentum, Adam moments). */
  setConfig(config: TrainerConfig): void {
    const restart = config.optimizer.id !== this.configValue.optimizer.id;
    this.configValue = { ...config, optimizer: { ...config.optimizer } };
    if (restart) this.optimizer = new Optimizer(config.optimizer);
    else Object.assign(this.optimizer.settings, config.optimizer);
  }

  /** Forget the progress (after the architecture or the initial weights changed). */
  resetState(): void {
    this.optimizer = new Optimizer(this.configValue.optimizer);
    this.rng = createRng(this.configValue.seed);
    this.grads = createGradients(this.model.spec);
    this.scratch = createScratch(this.model.spec);
    this.trace = createTrace(this.model.spec.layers.map((l) => l.size));
    this.hasGradients = false;
    this.prevWeights = null;
    this.prevBiases = null;
    this.stepCount = 0;
    this.epochCount = 0;
    this.history.length = 0;
    this.batchLog.length = 0;
    this.lastBatch = new Uint32Array(0);
    this.lastBatchLoss = NaN;
    this.reshuffle();
  }

  /** One optimization step on the next mini-batch. Returns the batch loss. */
  step(): number {
    const batch = this.nextBatch();
    return this.stepOn(batch);
  }

  /** Runs `count` steps; listeners of the model are notified only once. */
  runSteps(count: number): void {
    this.model.batch(() => {
      for (let k = 0; k < count; k++) {
        this.step();
        if (!Number.isFinite(this.lastBatchLoss)) break;
      }
    });
  }

  /** One optimization step on the given samples (e.g. a single point for the backprop animation). */
  stepOn(batch: ArrayLike<number>): number {
    const model = this.model;
    const { spec } = model;
    const loss = this.loss;
    const ds = this.datasetValue;
    zeroGradients(this.grads);
    let total = 0;
    const weights = Array.from({ length: spec.layers.length }, (_, l) => model.weights(l) as Float64Array);
    const biases = Array.from({ length: spec.layers.length }, (_, l) => model.biases(l) as Float64Array);
    const x = new Float64Array(2);
    for (let b = 0; b < batch.length; b++) {
      const k = batch[b];
      x[0] = ds.x[2 * k];
      x[1] = ds.x[2 * k + 1];
      forwardInto(spec, weights, biases, x, this.trace);
      total += sampleLoss(loss, spec, this.trace.a[spec.layers.length - 1], ds.y[k]);
      backpropSample(spec, weights, this.trace, loss, ds.y[k], this.grads, this.scratch);
    }
    scaleGradients(this.grads, 1 / Math.max(1, batch.length));
    this.hasGradients = true;
    this.prevWeights = weights.map((w) => w.slice());
    this.prevBiases = biases.map((b) => b.slice());

    model.updateParameters((W, B) => {
      const params: Float64Array[] = [];
      const grads: Float64Array[] = [];
      for (let l = 1; l < spec.layers.length; l++) {
        params.push(W[l], B[l]);
        grads.push(this.grads.dW[l], this.grads.db[l]);
      }
      this.optimizer.step(params, grads);
    }, 'train');

    this.stepCount++;
    this.lastBatch = Uint32Array.from(batch);
    this.lastBatchLoss = total / Math.max(1, batch.length);
    if (this.recordBatches) this.batchLog.push(Array.from(batch));
    return this.lastBatchLoss;
  }

  /** Loss and accuracy on a set of samples (default: training set). */
  evaluate(indices: ArrayLike<number> = this.datasetValue.train): Metrics {
    const { spec } = this.model;
    const ds = this.datasetValue;
    const loss = this.loss;
    const weights = Array.from({ length: spec.layers.length }, (_, l) => this.model.weights(l) as Float64Array);
    const biases = Array.from({ length: spec.layers.length }, (_, l) => this.model.biases(l) as Float64Array);
    const x = new Float64Array(2);
    let total = 0;
    let correct = 0;
    const L = spec.layers.length - 1;
    for (let b = 0; b < indices.length; b++) {
      const k = indices[b];
      x[0] = ds.x[2 * k];
      x[1] = ds.x[2 * k + 1];
      forwardInto(spec, weights, biases, x, this.trace);
      total += sampleLoss(loss, spec, this.trace.a[L], ds.y[k]);
      if (predictedClass(spec, this.trace.a[L]) === ds.y[k]) correct++;
    }
    const n = Math.max(1, indices.length);
    return { loss: total / n, accuracy: correct / n };
  }

  /** ‖∂L/∂W^(l)‖ (Frobenius) per layer of the last step. */
  gradientNorms(): number[] {
    if (!this.hasGradients) return this.grads.dW.map(() => 0);
    return this.grads.dW.map((g) => Math.sqrt(g.reduce((s, v) => s + v * v, 0)));
  }

  /** Appends a history point evaluated on the full train/test sets. */
  record(): HistoryPoint {
    const point: HistoryPoint = {
      step: this.stepCount,
      epoch: this.epochCount,
      train: this.evaluate(this.datasetValue.train),
      test: this.datasetValue.test.length ? this.evaluate(this.datasetValue.test) : null,
      gradNorms: this.gradientNorms(),
    };
    this.history.push(point);
    if (this.history.length > 4000) this.history.splice(0, this.history.length - 4000);
    return point;
  }

  /** Steps per epoch with the current batch size. */
  get stepsPerEpoch(): number {
    return Math.max(1, Math.ceil(this.datasetValue.train.length / this.batchSize));
  }

  private get batchSize(): number {
    return Math.max(1, Math.min(this.configValue.batchSize, this.datasetValue.train.length));
  }

  private nextBatch(): Uint32Array {
    if (this.order.length === 0) return new Uint32Array(0);
    const size = this.batchSize;
    const batch = this.order.slice(this.cursor, Math.min(this.order.length, this.cursor + size));
    this.cursor += size;
    if (this.cursor >= this.order.length) {
      // This batch completed an epoch: reshuffle for the next one.
      this.epochCount++;
      this.reshuffle();
    }
    return batch;
  }

  private reshuffle(): void {
    const order = Uint32Array.from(this.datasetValue.train);
    for (let k = order.length - 1; k > 0; k--) {
      const j = Math.floor(this.rng.next() * (k + 1));
      const t = order[k];
      order[k] = order[j];
      order[j] = t;
    }
    this.order = order;
    this.cursor = 0;
  }
}
