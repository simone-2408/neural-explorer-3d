/**
 * Loss functions and the error signal of the output layer δ = ∂L/∂z.
 *
 * The loss is chosen from the output layer, like in practice:
 *  - one sigmoid output        → binary cross-entropy   (δ = ŷ − y)
 *  - softmax output            → cross-entropy          (δ = ŷ − one_hot(y))
 *  - anything else             → mean squared error     (δ = (ŷ − y) ⊙ f′(z))
 */
import { ACTIVATIONS } from '../math/activations';
import type { NetworkSpec } from '../neural/config';

export type LossId = 'bce' | 'cross_entropy' | 'mse';

export interface LossInfo {
  id: LossId;
  label: string;
  /** TeX of the loss of one sample. */
  tex: string;
  description: string;
}

export const LOSSES: Readonly<Record<LossId, LossInfo>> = {
  bce: {
    id: 'bce',
    label: 'Binary cross-entropy',
    tex: 'L = -\\left[\\,y \\log \\hat{y} + (1-y)\\log(1-\\hat{y})\\,\\right]',
    description: 'Measures how surprised a probability ŷ is by the true class y ∈ {0, 1}: confident wrong answers are punished heavily.',
  },
  cross_entropy: {
    id: 'cross_entropy',
    label: 'Cross-entropy',
    tex: 'L = -\\log \\hat{y}_{c}\\quad (c = \\text{true class})',
    description: 'Minus the log-probability the softmax assigns to the true class.',
  },
  mse: {
    id: 'mse',
    label: 'Mean squared error',
    tex: 'L = \\tfrac{1}{2}\\sum_j (\\hat{y}_j - y_j)^2',
    description: 'Half the squared distance between the output and the target vector.',
  },
};

export function lossFor(spec: NetworkSpec): LossId {
  const output = spec.layers[spec.layers.length - 1];
  if (output.activation === 'softmax') return 'cross_entropy';
  if (output.activation === 'sigmoid' && output.size === 1) return 'bce';
  return 'mse';
}

const EPS = 1e-12;

/** Target value of output neuron j for class `label`. */
export function targetValue(spec: NetworkSpec, label: number, j: number): number {
  const output = spec.layers[spec.layers.length - 1];
  if (output.size === 1) {
    // Single output: class 1 → 1; class 0 → 0 (or −1 for tanh, whose range is (−1, 1)).
    if (label > 0) return 1;
    return output.activation === 'tanh' ? -1 : 0;
  }
  return j === label ? 1 : 0;
}

/** Loss of one sample, given the output activations. */
export function sampleLoss(id: LossId, spec: NetworkSpec, output: ArrayLike<number>, label: number): number {
  switch (id) {
    case 'bce': {
      const p = Math.min(1 - EPS, Math.max(EPS, output[0]));
      return label > 0 ? -Math.log(p) : -Math.log(1 - p);
    }
    case 'cross_entropy':
      return -Math.log(Math.max(EPS, output[label]));
    case 'mse': {
      let sum = 0;
      for (let j = 0; j < output.length; j++) sum += (output[j] - targetValue(spec, label, j)) ** 2;
      return 0.5 * sum;
    }
  }
}

/** δ of the output layer (∂L/∂z), written into `out`. */
export function outputDelta(
  id: LossId,
  spec: NetworkSpec,
  z: ArrayLike<number>,
  a: ArrayLike<number>,
  label: number,
  out: Float64Array,
): Float64Array {
  if (id === 'bce' || id === 'cross_entropy') {
    // The derivative of the loss and of the sigmoid/softmax combine into ŷ − y.
    for (let j = 0; j < a.length; j++) out[j] = a[j] - targetValue(spec, label, j);
    return out;
  }
  const activation = ACTIVATIONS[spec.layers[spec.layers.length - 1].activation];
  for (let j = 0; j < a.length; j++) {
    const slope = activation.kind === 'elementwise' ? activation.df(z[j]) : 1;
    out[j] = (a[j] - targetValue(spec, label, j)) * slope;
  }
  return out;
}

/** Predicted class from the output activations. */
export function predictedClass(spec: NetworkSpec, output: ArrayLike<number>): number {
  const layer = spec.layers[spec.layers.length - 1];
  if (output.length === 1) return output[0] >= (layer.activation === 'sigmoid' ? 0.5 : 0) ? 1 : 0;
  let best = 0;
  for (let j = 1; j < output.length; j++) if (output[j] > output[best]) best = j;
  return best;
}

/** Probability of class 1 (binary) or of the predicted class, for colouring decision maps. */
export function classScore(spec: NetworkSpec, output: ArrayLike<number>): number {
  const layer = spec.layers[spec.layers.length - 1];
  if (output.length === 1) return layer.activation === 'tanh' ? (output[0] + 1) / 2 : layer.activation === 'sigmoid' ? output[0] : 1 / (1 + Math.exp(-output[0]));
  return output[predictedClass(spec, output)];
}
