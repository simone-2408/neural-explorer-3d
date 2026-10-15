/**
 * Training of the recurrent network: mini-batches of sequences, mean
 * cross-entropy over the scored steps (like nn.CrossEntropyLoss with
 * ignore_index), gradient clipping, the optimizers of the Training tab.
 */
import { createRng, type Rng } from '../math/random';
import { Optimizer, type OptimizerSettings } from '../training/optimizers';
import { backwardRnn, clipGradients, runRnn, zeroRnnGrads } from './forward';
import type { RnnModel } from './model';
import type { RnnTask, Sequence } from './tasks';

export interface RnnMetrics {
  loss: number;
  /** Share of scored steps whose most likely output is the target. */
  accuracy: number;
}

export interface RnnHistoryPoint {
  step: number;
  train: RnnMetrics;
  test: RnnMetrics;
  /** Global gradient norm before clipping. */
  gradNorm: number;
}

export function evaluateRnn(model: RnnModel, sequences: readonly Sequence[]): RnnMetrics {
  let loss = 0;
  let correct = 0;
  let count = 0;
  for (const seq of sequences) {
    const { steps } = runRnn(model, seq.inputs);
    seq.targets.forEach((target, t) => {
      if (target < 0) return;
      const p = steps[t].probs;
      loss -= Math.log(Math.max(p[target], 1e-300));
      let best = 0;
      for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
      if (best === target) correct++;
      count++;
    });
  }
  return { loss: loss / Math.max(1, count), accuracy: correct / Math.max(1, count) };
}

export class RnnTrainer {
  readonly model: RnnModel;
  readonly task: RnnTask;
  /** Replaceable: changing the optimizer keeps the weights and the history. */
  optimizer: Optimizer;
  batchSize: number;
  /** Maximum global gradient norm (0 = no clipping). */
  clip: number;
  step = 0;
  readonly history: RnnHistoryPoint[] = [];
  readonly testSet: Sequence[];
  /** ‖∂L/∂h_t‖ per step, averaged over the last batch (backpropagation through time). */
  lastHiddenGradNorms: number[] = [];
  lastGradNorm = 0;
  private readonly rng: Rng;

  constructor(model: RnnModel, task: RnnTask, optimizer: OptimizerSettings, o: { batchSize?: number; clip?: number; seed?: number } = {}) {
    this.model = model;
    this.task = task;
    this.optimizer = new Optimizer(optimizer);
    this.batchSize = o.batchSize ?? 16;
    this.clip = o.clip ?? 5;
    this.rng = createRng(o.seed ?? 1);
    const testRng = createRng((o.seed ?? 1) + 9973);
    this.testSet = Array.from({ length: 128 }, () => task.sample(testRng));
  }

  trainStep(): number {
    const grads = zeroRnnGrads(this.model);
    const batch = Array.from({ length: this.batchSize }, () => this.task.sample(this.rng));
    const scored = batch.reduce((n, s) => n + s.targets.filter((t) => t >= 0).length, 0);
    let loss = 0;
    const norms = new Array<number>(this.task.length).fill(0);
    for (const seq of batch) {
      const result = backwardRnn(this.model, runRnn(this.model, seq.inputs), seq.targets, grads, 1 / scored);
      loss += result.loss;
      result.hiddenGradNorms.forEach((v, t) => (norms[t] += v / batch.length));
    }
    this.lastGradNorm = clipGradients(grads, this.clip);
    this.optimizer.step(this.model.params, grads);
    this.lastHiddenGradNorms = norms;
    this.step++;
    return loss;
  }

  record(): RnnHistoryPoint {
    const trainRng = createRng(this.step + 1);
    const point = {
      step: this.step,
      train: evaluateRnn(this.model, Array.from({ length: 64 }, () => this.task.sample(trainRng))),
      test: evaluateRnn(this.model, this.testSet),
      gradNorm: this.lastGradNorm,
    };
    this.history.push(point);
    return point;
  }
}

/** Appends `count` symbols sampled from the model's predictions (temperature 0 = always the most likely). */
export function generate(model: RnnModel, prefix: readonly number[], count: number, temperature: number, rng: Rng): number[] {
  const out = [...prefix];
  for (let k = 0; k < count; k++) {
    const { steps } = runRnn(model, out);
    const p = steps[steps.length - 1].probs;
    let choice = 0;
    if (temperature <= 0) {
      for (let i = 1; i < p.length; i++) if (p[i] > p[choice]) choice = i;
    } else {
      const w = Array.from(p, (v) => Math.pow(v, 1 / temperature));
      const total = w.reduce((a, b) => a + b, 0);
      let r = rng.next() * total;
      choice = w.length - 1;
      for (let i = 0; i < w.length; i++) {
        r -= w[i];
        if (r <= 0) {
          choice = i;
          break;
        }
      }
    }
    out.push(choice);
  }
  return out.slice(prefix.length);
}
