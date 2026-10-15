/**
 * Weight initialization schemes. Initialization matters a lot for training
 * (the "Break it" mode of the Training tab shows what bad choices do), so each
 * scheme carries its formula and a short explanation.
 */
import type { Rng } from '../math/random';

export type InitScheme = 'pytorch' | 'xavier' | 'he' | 'uniform' | 'zeros' | 'large' | 'negative-bias';

export interface InitSettings {
  scheme: InitScheme;
  seed: number;
}

export const DEFAULT_INIT: Readonly<InitSettings> = { scheme: 'pytorch', seed: 42 };

export interface InitSchemeInfo {
  id: InitScheme;
  label: string;
  /** TeX describing the weight distribution (n_in = fan-in, n_out = fan-out). */
  tex: string;
  description: string;
}

export const INIT_SCHEMES: Readonly<Record<InitScheme, InitSchemeInfo>> = {
  pytorch: {
    id: 'pytorch',
    label: 'PyTorch default (nn.Linear)',
    tex: 'W, b \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{n_{in}}},\\, \\tfrac{1}{\\sqrt{n_{in}}}\\right)',
    description: 'What torch.nn.Linear does by default: small uniform weights and biases scaled by the fan-in.',
  },
  xavier: {
    id: 'xavier',
    label: 'Xavier / Glorot uniform',
    tex: 'W \\sim \\mathcal{U}\\!\\left(-\\sqrt{\\tfrac{6}{n_{in}+n_{out}}},\\, \\sqrt{\\tfrac{6}{n_{in}+n_{out}}}\\right),\\; b = 0',
    description: 'Keeps the variance of signals roughly constant across layers; designed for tanh/sigmoid networks.',
  },
  he: {
    id: 'he',
    label: 'He / Kaiming normal',
    tex: 'W \\sim \\mathcal{N}\\!\\left(0,\\, \\tfrac{2}{n_{in}}\\right),\\; b = 0',
    description: 'Variance scaled for ReLU networks (ReLU zeroes about half of the signal).',
  },
  uniform: {
    id: 'uniform',
    label: 'Naive uniform [−1, 1]',
    tex: 'W, b \\sim \\mathcal{U}(-1,\\, 1)',
    description: 'Ignores the layer size. Fine for tiny networks, but signals can grow or vanish in wide/deep ones.',
  },
  zeros: {
    id: 'zeros',
    label: 'All zeros (broken)',
    tex: 'W = 0,\\; b = 0',
    description:
      'Every neuron in a layer computes the same thing and would receive the same gradient: training cannot break the symmetry.',
  },
  large: {
    id: 'large',
    label: 'Too large: N(0, 3²) (broken)',
    tex: 'W, b \\sim \\mathcal{N}(0,\\, 3^2)',
    description: 'Weights far too large for the layer size: weighted sums are huge, sigmoid/tanh neurons saturate (flat region) and ReLU outputs explode.',
  },
  'negative-bias': {
    id: 'negative-bias',
    label: 'Negative biases b = −2 (broken)',
    tex: 'W \\sim \\mathcal{U}\\!\\left(-\\tfrac{1}{\\sqrt{n_{in}}},\\, \\tfrac{1}{\\sqrt{n_{in}}}\\right),\\; b = -2',
    description: 'With inputs in [−1, 1] every weighted sum starts negative: ReLU neurons are inactive for all data and get no gradient ("dead ReLU").',
  },
};

export function isInitScheme(value: unknown): value is InitScheme {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(INIT_SCHEMES, value);
}

/** Fills W (shape [fanOut, fanIn], row-major) and b in place. */
export function initializeLayer(
  scheme: InitScheme,
  fanIn: number,
  fanOut: number,
  rng: Rng,
  W: Float64Array,
  b: Float64Array,
): void {
  switch (scheme) {
    case 'pytorch': {
      const bound = 1 / Math.sqrt(fanIn);
      for (let k = 0; k < W.length; k++) W[k] = rng.uniform(-bound, bound);
      for (let k = 0; k < b.length; k++) b[k] = rng.uniform(-bound, bound);
      return;
    }
    case 'xavier': {
      const bound = Math.sqrt(6 / (fanIn + fanOut));
      for (let k = 0; k < W.length; k++) W[k] = rng.uniform(-bound, bound);
      b.fill(0);
      return;
    }
    case 'he': {
      const std = Math.sqrt(2 / fanIn);
      for (let k = 0; k < W.length; k++) W[k] = rng.normal(0, std);
      b.fill(0);
      return;
    }
    case 'uniform': {
      for (let k = 0; k < W.length; k++) W[k] = rng.uniform(-1, 1);
      for (let k = 0; k < b.length; k++) b[k] = rng.uniform(-1, 1);
      return;
    }
    case 'zeros':
      W.fill(0);
      b.fill(0);
      return;
    case 'large':
      for (let k = 0; k < W.length; k++) W[k] = rng.normal(0, 3);
      for (let k = 0; k < b.length; k++) b[k] = rng.normal(0, 3);
      return;
    case 'negative-bias': {
      const bound = 1 / Math.sqrt(fanIn);
      for (let k = 0; k < W.length; k++) W[k] = rng.uniform(-bound, bound);
      b.fill(-2);
      return;
    }
  }
}
