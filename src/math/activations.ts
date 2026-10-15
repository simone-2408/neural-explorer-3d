/**
 * Activation functions: numeric definitions + educational metadata (TeX,
 * output range, description). Pure TypeScript — no DOM, no three.js — so the
 * same definitions are used by the forward pass, by future training code
 * (derivatives) and by the UI (formulas and plots).
 *
 * Conventions match PyTorch defaults so results can be compared 1:1 with a
 * PyTorch model (see python/mlp_bridge.py):
 *   - GELU is the exact erf formulation (nn.GELU())
 *   - Leaky ReLU uses negative_slope = 0.01 (nn.LeakyReLU())
 */

export type ActivationId = 'linear' | 'relu' | 'leaky_relu' | 'sigmoid' | 'tanh' | 'gelu' | 'softmax';

/** Formats a number for use inside a TeX expression (provided by the caller). */
export type TexNumber = (value: number) => string;

interface ActivationBase {
  readonly id: ActivationId;
  readonly label: string;
  /** TeX name, e.g. `\operatorname{ReLU}` */
  readonly tex: string;
  /** TeX definition (right-hand side) as a function of z */
  readonly texDefinition: string;
  /** Human readable output range */
  readonly range: string;
  readonly description: string;
}

export interface ElementwiseActivation extends ActivationBase {
  readonly kind: 'elementwise';
  f(z: number): number;
  /** Derivative da/dz (used by backpropagation). */
  df(z: number): number;
  /** TeX of the derivative f′(z). */
  readonly texDerivative: string;
  /** TeX of the definition with a concrete z substituted, e.g. `\max(0,\,0.38)`. */
  texEval(z: number, num: TexNumber): string;
}

/** Activations that couple all the neurons of a layer (softmax). */
export interface VectorActivation extends ActivationBase {
  readonly kind: 'vector';
}

export type ActivationInfo = ElementwiseActivation | VectorActivation;

export const LEAKY_RELU_SLOPE = 0.01;

const SQRT1_2 = Math.SQRT1_2;
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
const INV_SQRT_PI = 1 / Math.sqrt(Math.PI);

/**
 * Error function with ~1e-15 accuracy: Taylor series for |x| < 3,
 * continued fraction for erfc beyond. (Approximations such as
 * Abramowitz–Stegun 7.1.26 are only accurate to ~1e-7.)
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const ax = Math.abs(x);
  if (ax < 3) {
    // erf(x) = 2/√π · Σ (-1)^n x^(2n+1) / (n! (2n+1))
    const x2 = x * x;
    let term = x; // x^(2n+1) (-1)^n / n!
    let sum = x;
    for (let n = 1; n < 100; n++) {
      term *= -x2 / n;
      const contribution = term / (2 * n + 1);
      sum += contribution;
      if (Math.abs(contribution) < 1e-17 * Math.abs(sum)) break;
    }
    return 2 * INV_SQRT_PI * sum;
  }
  // erfc(x) = exp(-x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + …)))))
  let t = ax;
  for (let k = 60; k >= 1; k--) t = ax + k / 2 / t;
  const erfc = (Math.exp(-ax * ax) * INV_SQRT_PI) / t;
  return x > 0 ? 1 - erfc : erfc - 1;
}

/** Standard normal CDF Φ(z). */
export function normalCdf(z: number): number {
  return 0.5 * (1 + erf(z * SQRT1_2));
}

/** Numerically stable logistic sigmoid. */
export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

const paren = (num: TexNumber, v: number) => (v < 0 ? `\\left(${num(v)}\\right)` : num(v));

export const ACTIVATIONS: Readonly<Record<ActivationId, ActivationInfo>> = {
  linear: {
    id: 'linear',
    kind: 'elementwise',
    label: 'Linear (identity)',
    tex: '\\operatorname{id}',
    texDefinition: 'z',
    texDerivative: '1',
    range: '(−∞, +∞)',
    description:
      'No non-linearity: the output equals the weighted sum. Stacking only linear layers is still a single linear function, so hidden layers normally use a non-linear activation.',
    f: (z) => z,
    df: () => 1,
    texEval: (z, num) => num(z),
  },
  relu: {
    id: 'relu',
    kind: 'elementwise',
    label: 'ReLU',
    tex: '\\operatorname{ReLU}',
    texDefinition: '\\max(0,\\, z)',
    texDerivative: '\\begin{cases} 1 & z > 0 \\\\ 0 & z \\le 0 \\end{cases}',
    range: '[0, +∞)',
    description:
      'Rectified Linear Unit: keeps positive values unchanged and outputs 0 when z < 0. When z < 0 the neuron is inactive for that input and its local gradient is 0.',
    f: (z) => (z > 0 ? z : 0),
    df: (z) => (z > 0 ? 1 : 0),
    texEval: (z, num) => `\\max(0,\\, ${num(z)})`,
  },
  leaky_relu: {
    id: 'leaky_relu',
    kind: 'elementwise',
    label: 'Leaky ReLU',
    tex: '\\operatorname{LeakyReLU}',
    texDefinition: `\\max(${LEAKY_RELU_SLOPE}\\,z,\\; z)`,
    texDerivative: `\\begin{cases} 1 & z > 0 \\\\ ${LEAKY_RELU_SLOPE} & z \\le 0 \\end{cases}`,
    range: '(−∞, +∞)',
    description: `Like ReLU, but negative inputs are multiplied by a small slope (${LEAKY_RELU_SLOPE}) instead of being set to 0, so the gradient never becomes exactly zero.`,
    f: (z) => (z > 0 ? z : LEAKY_RELU_SLOPE * z),
    df: (z) => (z > 0 ? 1 : LEAKY_RELU_SLOPE),
    texEval: (z, num) => `\\max(${LEAKY_RELU_SLOPE} \\cdot ${paren(num, z)},\\; ${num(z)})`,
  },
  sigmoid: {
    id: 'sigmoid',
    kind: 'elementwise',
    label: 'Sigmoid',
    tex: '\\sigma',
    texDefinition: '\\dfrac{1}{1 + e^{-z}}',
    texDerivative: '\\sigma(z)\\,\\bigl(1 - \\sigma(z)\\bigr)',
    range: '(0, 1)',
    description:
      'Squashes any z into (0, 1); common for binary-classification outputs, read as a probability. For large |z| the curve is flat (saturation), which makes gradients very small.',
    f: sigmoid,
    df: (z) => {
      const s = sigmoid(z);
      return s * (1 - s);
    },
    texEval: (z, num) => `\\dfrac{1}{1 + e^{${num(-z)}}}`,
  },
  tanh: {
    id: 'tanh',
    kind: 'elementwise',
    label: 'Tanh',
    tex: '\\tanh',
    texDefinition: '\\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}',
    texDerivative: '1 - \\tanh^2(z)',
    range: '(−1, 1)',
    description:
      'Squashes z into (−1, 1) and is centred on 0. Like the sigmoid, it saturates for large |z|.',
    f: Math.tanh,
    df: (z) => {
      const t = Math.tanh(z);
      return 1 - t * t;
    },
    texEval: (z, num) => `\\tanh(${num(z)})`,
  },
  gelu: {
    id: 'gelu',
    kind: 'elementwise',
    label: 'GELU',
    tex: '\\operatorname{GELU}',
    texDefinition: 'z\\,\\Phi(z)',
    texDerivative: '\\Phi(z) + z\\,\\varphi(z)',
    range: '[≈ −0.17, +∞)',
    description:
      'Gaussian Error Linear Unit: z multiplied by Φ(z), the standard normal CDF. A smooth relative of ReLU, used in Transformer MLP blocks (GPT-2, BERT).',
    f: (z) => z * normalCdf(z),
    df: (z) => normalCdf(z) + z * INV_SQRT_2PI * Math.exp(-0.5 * z * z),
    texEval: (z, num) => `${paren(num, z)}\\,\\Phi(${num(z)})`,
  },
  softmax: {
    id: 'softmax',
    kind: 'vector',
    label: 'Softmax',
    tex: '\\operatorname{softmax}',
    texDefinition: '\\dfrac{e^{z_j}}{\\sum_k e^{z_k}}',
    range: '(0, 1), Σ = 1',
    description:
      'Turns the vector of scores (logits) of a layer into probabilities that sum to 1. Unlike the other activations, each output depends on all the z values of the layer.',
  },
};

export const ACTIVATION_IDS = Object.keys(ACTIVATIONS) as ActivationId[];

export function isActivationId(value: unknown): value is ActivationId {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(ACTIVATIONS, value);
}

/** Numerically stable softmax (subtracts the max before exponentiating). */
export function softmax(z: ArrayLike<number>, out: Float64Array): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > max) max = z[i];
  let sum = 0;
  for (let i = 0; i < z.length; i++) {
    const e = Math.exp(z[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < z.length; i++) out[i] /= sum;
  return out;
}

/** Applies an activation to a whole layer: out = f(z). `out` may alias `z` only for elementwise functions. */
export function applyActivation(id: ActivationId, z: Float64Array, out: Float64Array): Float64Array {
  const info = ACTIVATIONS[id];
  if (info.kind === 'vector') return softmax(z, out);
  for (let i = 0; i < z.length; i++) out[i] = info.f(z[i]);
  return out;
}
