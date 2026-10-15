/**
 * Live training diagnostics: detect the classic failure modes and explain
 * them with the numbers that reveal them. Each check is a measurable
 * condition, not a guess: the message states what was measured.
 */
import { ACTIVATIONS } from '../math/activations';
import { createTrace, forwardInto } from '../neural/forward';
import type { NetworkModel } from '../neural/NetworkModel';
import { formatValue } from '../utils/format';
import type { Dataset } from './datasets';
import type { Trainer } from './Trainer';

/** What the diagnostics need to look at. */
export interface TrainingState {
  model: NetworkModel;
  dataset: Dataset;
  trainer: Trainer;
}

export interface Diagnosis {
  id: 'diverged' | 'exploding' | 'vanishing' | 'dead-relu' | 'saturated' | 'symmetric' | 'plateau' | 'overfitting' | 'slow';
  severity: 'info' | 'warning' | 'stop';
  title: string;
  detail: string;
}

export interface UnitStats {
  /** Fraction of hidden ReLU-like units that are inactive on every training sample. */
  deadFraction: number;
  deadCount: number;
  reluCount: number;
  /** Fraction of sigmoid/tanh units saturated (slope f′(z) below 5% of its maximum) on most samples. */
  saturatedFraction: number;
  squashCount: number;
}

export function unitStats(ws: TrainingState): UnitStats {
  const { model, dataset } = ws;
  const { spec } = model;
  const trace = createTrace(spec.layers.map((l) => l.size));
  const weights = Array.from({ length: model.layerCount }, (_, l) => model.weights(l) as Float64Array);
  const biases = Array.from({ length: model.layerCount }, (_, l) => model.biases(l) as Float64Array);
  const everActive: Uint8Array[] = spec.layers.map((l) => new Uint8Array(l.size));
  const saturatedCount: Uint32Array[] = spec.layers.map((l) => new Uint32Array(l.size));
  const x = new Float64Array(2);
  const n = Math.min(dataset.train.length, 300);
  for (let s = 0; s < n; s++) {
    const k = dataset.train[s];
    x[0] = dataset.x[2 * k];
    x[1] = dataset.x[2 * k + 1];
    forwardInto(spec, weights, biases, x, trace);
    for (let l = 1; l < spec.layers.length - 1; l++) {
      const z = trace.z[l];
      const activation = ACTIVATIONS[spec.layers[l].activation];
      const flat = activation.kind === 'elementwise' ? 0.05 * activation.df(0) : 0;
      for (let j = 0; j < z.length; j++) {
        if (z[j] > 0) everActive[l][j] = 1;
        if (activation.kind === 'elementwise' && activation.df(z[j]) < flat) saturatedCount[l][j]++;
      }
    }
  }
  let reluCount = 0;
  let deadCount = 0;
  let squashCount = 0;
  let saturated = 0;
  for (let l = 1; l < spec.layers.length - 1; l++) {
    const activation = spec.layers[l].activation;
    for (let j = 0; j < spec.layers[l].size; j++) {
      if (activation === 'relu') {
        reluCount++;
        if (!everActive[l][j]) deadCount++;
      } else if (activation === 'sigmoid' || activation === 'tanh') {
        squashCount++;
        if (saturatedCount[l][j] > 0.6 * n) saturated++;
      }
    }
  }
  return {
    deadFraction: reluCount ? deadCount / reluCount : 0,
    deadCount,
    reluCount,
    saturatedFraction: squashCount ? saturated / squashCount : 0,
    squashCount,
  };
}

/** True if all neurons of some hidden layer have identical incoming weights and biases. */
export function symmetricLayer(ws: TrainingState): number {
  const { model } = ws;
  for (let l = 1; l < model.layerCount - 1; l++) {
    const rows = model.spec.layers[l].size;
    const cols = model.spec.layers[l - 1].size;
    if (rows < 2) continue;
    const W = model.weights(l);
    const b = model.biases(l);
    let identical = true;
    for (let j = 1; j < rows && identical; j++) {
      if (Math.abs(b[j] - b[0]) > 1e-9) identical = false;
      for (let i = 0; i < cols && identical; i++) if (Math.abs(W[j * cols + i] - W[i]) > 1e-9) identical = false;
    }
    if (identical) return l;
  }
  return -1;
}

export function diagnose(ws: TrainingState): Diagnosis | null {
  const { trainer, model } = ws;
  const history = trainer.history;
  const last = history[history.length - 1];
  const first = history[0];
  if (trainer.stepCount > 0 && !Number.isFinite(trainer.lastBatchLoss)) {
    return {
      id: 'diverged',
      severity: 'stop',
      title: 'Training diverged',
      detail: 'The loss became NaN or infinite: the updates were so large that the weights overflowed. Lower the learning rate and reset.',
    };
  }
  if (!last || trainer.stepCount === 0) return null;

  if (history.length > 3 && last.train.loss > Math.max(3 * first.train.loss, first.train.loss + 2)) {
    return {
      id: 'exploding',
      severity: 'warning',
      title: 'The loss is growing',
      detail: `Training loss went from ${formatValue(first.train.loss, 3)} to ${formatValue(last.train.loss, 3)}: the steps overshoot the minimum. A smaller learning rate usually fixes it.`,
    };
  }

  const symmetric = symmetricLayer(ws);
  if (symmetric > 0) {
    return {
      id: 'symmetric',
      severity: 'warning',
      title: 'Symmetric neurons',
      detail: `All neurons of ${model.spec.layers[symmetric].name} have identical weights: they compute the same thing and receive the same gradient, so they can never become different. Use a random initialization.`,
    };
  }

  const norms = last.gradNorms;
  const firstNorm = norms[1] ?? 0;
  const lastNorm = norms[norms.length - 1] ?? 0;
  if (model.layerCount > 3 && lastNorm > 1e-9 && firstNorm / lastNorm < 1e-3) {
    return {
      id: 'vanishing',
      severity: 'warning',
      title: 'Vanishing gradients',
      detail: `‖∂L/∂W‖ of the first layer is ${formatValue(firstNorm / lastNorm, 6)} times that of the last: each layer multiplies the error signal by small factors (f′ ≤ 0.25 for sigmoid), so the early layers barely learn.`,
    };
  }

  const units = unitStats(ws);
  if (units.reluCount > 0 && units.deadFraction >= 0.5) {
    return {
      id: 'dead-relu',
      severity: 'warning',
      title: 'Dead ReLU neurons',
      detail: `${units.deadCount} of ${units.reluCount} ReLU neurons are inactive (z ≤ 0) for every training point: their gradient is always 0 and they cannot recover.`,
    };
  }
  if (units.squashCount > 0 && units.saturatedFraction >= 0.5) {
    return {
      id: 'saturated',
      severity: 'warning',
      title: 'Saturated neurons',
      detail: `${Math.round(units.saturatedFraction * 100)}% of the sigmoid/tanh neurons work where their curve is flat (slope f′(z) below 5% of its maximum) on most points: gradients through them are tiny.`,
    };
  }

  if (last.test && last.train.accuracy - last.test.accuracy > 0.15 && history.length > 30) {
    return {
      id: 'overfitting',
      severity: 'warning',
      title: 'Overfitting',
      detail: `Accuracy is ${Math.round(last.train.accuracy * 100)}% on the training set but ${Math.round(last.test.accuracy * 100)}% on unseen test points: the network is memorizing the training data.`,
    };
  }

  if (history.length > 60) {
    const past = history[history.length - 31];
    const improvement = (past.train.loss - last.train.loss) / Math.max(past.train.loss, 1e-12);
    if (improvement < 0.01 && last.train.accuracy < 0.9) {
      return {
        id: 'plateau',
        severity: 'info',
        title: 'Training has stalled',
        detail: `The loss improved by less than 1% in the last 30 epochs while accuracy is ${Math.round(last.train.accuracy * 100)}%: the network may be too small for this data (underfitting), or the learning rate unsuitable.`,
      };
    }
  }

  if (trainer.stepCount > 150 && history.length > 5) {
    const change = Math.abs(first.train.loss - last.train.loss) / Math.max(first.train.loss, 1e-12);
    if (change < 0.005) {
      return {
        id: 'slow',
        severity: 'info',
        title: 'Almost no progress',
        detail: `After ${trainer.stepCount} steps the loss changed by ${formatValue(change * 100, 2)}%: the learning rate may be too small.`,
      };
    }
  }
  return null;
}
