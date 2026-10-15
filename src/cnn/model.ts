/**
 * A small convolutional network for 16×16 grayscale images:
 *
 *   image → [conv k×k ("same" padding) → activation → 2×2 pooling] × 1–2 → flatten → dense → softmax
 *
 * Parameters use PyTorch's layouts so a model can be checked against
 * torch.nn (see tests/pytorch-parity.test.ts):
 *   conv weight  [out channels, in channels, k, k]   (nn.Conv2d.weight)
 *   dense weight [classes, flattened size]           (nn.Linear.weight)
 * and the flatten order is PyTorch's (channel, row, column).
 */
import { createRng, mixSeed } from '../math/random';

export type ConvActivation = 'relu' | 'tanh';
export type PoolKind = 'max' | 'avg' | 'none';

export interface ConvLayerConfig {
  filters: number;
  kernel: 3 | 5;
  pool: PoolKind;
}

export interface CnnConfig {
  /** Width = height of the input image. */
  size: number;
  convs: ConvLayerConfig[];
  activation: ConvActivation;
  classes: number;
}

export const CNN_LIMITS = { maxConvs: 2, minFilters: 1, maxFilters: 8 };

export const DEFAULT_CNN: Omit<CnnConfig, 'classes'> = {
  size: 16,
  convs: [
    { filters: 4, kernel: 3, pool: 'max' },
    { filters: 8, kernel: 3, pool: 'max' },
  ],
  activation: 'relu',
};

export interface ConvShape {
  inChannels: number;
  outChannels: number;
  kernel: number;
  /** Spatial size of the input and of the convolution output ("same" padding keeps it). */
  size: number;
  /** Spatial size after pooling. */
  pooledSize: number;
  pool: PoolKind;
}

export function convShapes(config: CnnConfig): ConvShape[] {
  const shapes: ConvShape[] = [];
  let channels = 1;
  let size = config.size;
  for (const c of config.convs) {
    const pooledSize = c.pool === 'none' ? size : Math.floor(size / 2);
    shapes.push({ inChannels: channels, outChannels: c.filters, kernel: c.kernel, size, pooledSize, pool: c.pool });
    channels = c.filters;
    size = pooledSize;
  }
  return shapes;
}

export class CnnModel {
  readonly config: CnnConfig;
  readonly shapes: ConvShape[];
  /** Length of the flattened vector entering the dense layer. */
  readonly flatSize: number;
  /** [conv₁ W, conv₁ b, conv₂ W, conv₂ b, …, dense W, dense b] — the order of PyTorch's state_dict. */
  readonly params: Float64Array[];
  readonly seed: number;

  constructor(config: CnnConfig, seed = 1, params?: Float64Array[]) {
    this.config = structuredClone(config);
    this.shapes = convShapes(config);
    const last = this.shapes[this.shapes.length - 1];
    this.flatSize = last.outChannels * last.pooledSize * last.pooledSize;
    this.seed = seed;
    this.params = params ?? this.initialize(seed);
  }

  /** PyTorch's default initialization of Conv2d and Linear: U(−1/√fan_in, 1/√fan_in) for weights and biases. */
  private initialize(seed: number): Float64Array[] {
    const params: Float64Array[] = [];
    const uniform = (count: number, fanIn: number, rngSeed: number) => {
      const rng = createRng(rngSeed);
      const bound = 1 / Math.sqrt(fanIn);
      return Float64Array.from({ length: count }, () => rng.uniform(-bound, bound));
    };
    this.shapes.forEach((s, l) => {
      const fanIn = s.inChannels * s.kernel * s.kernel;
      params.push(uniform(s.outChannels * fanIn, fanIn, mixSeed(seed, l, 0)), uniform(s.outChannels, fanIn, mixSeed(seed, l, 1)));
    });
    const L = this.shapes.length;
    params.push(uniform(this.config.classes * this.flatSize, this.flatSize, mixSeed(seed, L, 0)), uniform(this.config.classes, this.flatSize, mixSeed(seed, L, 1)));
    return params;
  }

  convWeight(layer: number): Float64Array {
    return this.params[2 * layer];
  }

  convBias(layer: number): Float64Array {
    return this.params[2 * layer + 1];
  }

  get denseWeight(): Float64Array {
    return this.params[2 * this.shapes.length];
  }

  get denseBias(): Float64Array {
    return this.params[2 * this.shapes.length + 1];
  }

  /** Index of W[f, c, u, v] in the weight array of a conv layer. */
  weightIndex(layer: number, f: number, c: number, u: number, v: number): number {
    const s = this.shapes[layer];
    return ((f * s.inChannels + c) * s.kernel + u) * s.kernel + v;
  }

  parameterCount(): number {
    return this.params.reduce((n, p) => n + p.length, 0);
  }

  /** Parameters of conv layer `layer` (weights + biases). */
  convParameterCount(layer: number): number {
    return this.convWeight(layer).length + this.convBias(layer).length;
  }

  /**
   * Parameters a fully connected layer would need to compute an output of the
   * same size from the same input: every output value with its own weights
   * for every input value. The convolution shares one small kernel instead.
   */
  fullyConnectedEquivalent(layer: number): number {
    const s = this.shapes[layer];
    const inputs = s.inChannels * s.size * s.size;
    const outputs = s.outChannels * s.size * s.size;
    return inputs * outputs + outputs;
  }

  clone(): CnnModel {
    return new CnnModel(this.config, this.seed, this.params.map((p) => Float64Array.from(p)));
  }
}
