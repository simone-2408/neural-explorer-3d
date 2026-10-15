/**
 * NetworkModel — the single source of truth for an MLP:
 * architecture (spec), parameters (weights, biases), current input and the
 * cached forward trace.
 *
 * It is a plain mutable object backed by typed arrays (cheap to update 60
 * times per second during training) that announces changes through events.
 * It knows nothing about three.js or the DOM: it can run headless in tests,
 * in a Web Worker, or be mirrored from a PyTorch backend.
 */
import { Emitter } from '../core/Emitter';
import { mixSeed, createRng } from '../math/random';
import { specFromConfig, type NetworkSpec } from './config';
import { createTrace, forwardInto, type ForwardTrace } from './forward';
import { DEFAULT_INIT, initializeLayer, type InitSettings } from './init';
import { Topology } from './topology';

export type ChangeReason = 'init' | 'edit' | 'input' | 'reinit' | 'config' | 'preset' | 'import' | 'train';

export type ModelEvents = {
  /** Architecture changed (layers, sizes, activations). Views must rebuild. */
  structure: { reason: ChangeReason };
  /** Weights or biases changed. `layer` is set when exactly one layer changed. */
  params: { reason: ChangeReason; layer?: number };
  /** The forward pass has been recomputed (after any input/params/structure change). */
  activations: { reason: ChangeReason };
};

export interface ParameterCount {
  layer: number;
  weights: number;
  biases: number;
  total: number;
}

export interface ArchitectureOptions {
  /** Keep the weights of the overlapping part of each layer (default true). */
  preserveParams?: boolean;
  /** New input vector; otherwise the overlapping part of the old input is kept. */
  input?: ArrayLike<number>;
  reason?: ChangeReason;
}

export class NetworkModel {
  readonly events = new Emitter<ModelEvents>();

  private specValue: NetworkSpec;
  private topologyValue: Topology;
  private weightsValue: Float64Array[] = [];
  private biasesValue: Float64Array[] = [];
  private inputValue: Float64Array;
  private traceValue: ForwardTrace;
  private initValue: InitSettings;

  private batchDepth = 0;
  private pending = { structure: false, params: false, activations: false, layer: null as number | 'many' | null };
  private pendingReason: ChangeReason = 'init';
  private pendingRank = 0;

  constructor(spec: NetworkSpec, options: { init?: Partial<InitSettings>; input?: ArrayLike<number> } = {}) {
    this.specValue = spec;
    this.topologyValue = Topology.fromSpec(spec);
    this.initValue = { ...DEFAULT_INIT, ...options.init };
    this.allocateParameters();
    this.inputValue = new Float64Array(spec.layers[0].size);
    if (options.input) this.inputValue.set(Array.from(options.input).slice(0, this.inputValue.length));
    this.traceValue = createTrace(this.topologyValue.sizes);
    this.recompute();
  }

  /** Validates a JSON configuration and builds a model (throws ConfigError when invalid). */
  static fromConfig(config: unknown, options?: { init?: Partial<InitSettings>; input?: ArrayLike<number> }): NetworkModel {
    return new NetworkModel(specFromConfig(config), options);
  }

  // ------------------------------------------------------------------ reads

  get spec(): NetworkSpec {
    return this.specValue;
  }

  get topology(): Topology {
    return this.topologyValue;
  }

  /** Current forward trace (always consistent with params + input). */
  get trace(): ForwardTrace {
    return this.traceValue;
  }

  get input(): Readonly<Float64Array> {
    return this.inputValue;
  }

  get output(): Readonly<Float64Array> {
    return this.traceValue.a[this.traceValue.a.length - 1];
  }

  get init(): Readonly<InitSettings> {
    return this.initValue;
  }

  get layerCount(): number {
    return this.specValue.layers.length;
  }

  /** Row-major weight matrix of layer l, shape [n_l, n_{l-1}] (PyTorch nn.Linear layout). */
  weights(layer: number): Readonly<Float64Array> {
    return this.weightsValue[layer];
  }

  biases(layer: number): Readonly<Float64Array> {
    return this.biasesValue[layer];
  }

  weight(layer: number, target: number, source: number): number {
    return this.weightsValue[layer][target * this.specValue.layers[layer - 1].size + source];
  }

  bias(layer: number, index: number): number {
    return this.biasesValue[layer][index];
  }

  parameterCounts(): ParameterCount[] {
    const counts: ParameterCount[] = [];
    for (let l = 1; l < this.specValue.layers.length; l++) {
      const weights = this.specValue.layers[l].size * this.specValue.layers[l - 1].size;
      const biases = this.specValue.layers[l].size;
      counts.push({ layer: l, weights, biases, total: weights + biases });
    }
    return counts;
  }

  totalParameters(): number {
    return this.parameterCounts().reduce((sum, c) => sum + c.total, 0);
  }

  // ----------------------------------------------------------------- writes

  setWeight(layer: number, target: number, source: number, value: number, reason: ChangeReason = 'edit'): void {
    this.weightsValue[layer][target * this.specValue.layers[layer - 1].size + source] = value;
    this.markParams(layer, reason);
  }

  setBias(layer: number, index: number, value: number, reason: ChangeReason = 'edit'): void {
    this.biasesValue[layer][index] = value;
    this.markParams(layer, reason);
  }

  setInput(index: number, value: number): void {
    this.inputValue[index] = value;
    this.mark({ activations: true }, 'input');
  }

  setInputs(values: ArrayLike<number>): void {
    const n = Math.min(values.length, this.inputValue.length);
    for (let i = 0; i < n; i++) this.inputValue[i] = values[i];
    this.mark({ activations: true }, 'input');
  }

  /** Re-draws all parameters from the initialization scheme (deterministic for a given seed). */
  reinitialize(settings: Partial<InitSettings> = {}, reason: ChangeReason = 'reinit'): void {
    this.initValue = { ...this.initValue, ...settings };
    this.initializeAll();
    this.markParams('many', reason);
  }

  /** Replaces all parameters (e.g. imported from a file or from PyTorch). Shapes must match the spec. */
  setParameters(weights: readonly Float64Array[], biases: readonly Float64Array[], reason: ChangeReason = 'import'): void {
    for (let l = 1; l < this.specValue.layers.length; l++) {
      if (weights[l].length !== this.weightsValue[l].length || biases[l].length !== this.biasesValue[l].length)
        throw new Error(`Parameter shape mismatch in layer ${l}.`);
      this.weightsValue[l].set(weights[l]);
      this.biasesValue[l].set(biases[l]);
    }
    this.markParams('many', reason);
  }

  /**
   * Mutable access to all parameters (e.g. for an optimizer step). Listeners
   * are notified once afterwards; inside batch() only at the end of the batch.
   */
  updateParameters(fn: (weights: Float64Array[], biases: Float64Array[]) => void, reason: ChangeReason = 'train'): void {
    fn(this.weightsValue, this.biasesValue);
    this.markParams('many', reason);
  }

  /**
   * Changes the architecture. By default the overlapping block of every
   * weight matrix is preserved, so adding a neuron does not scramble the rest
   * of the network; new entries come from the initialization scheme.
   */
  setArchitecture(spec: NetworkSpec, options: ArchitectureOptions = {}): void {
    const { preserveParams = true, reason = 'config' } = options;
    const oldSpec = this.specValue;
    const oldWeights = this.weightsValue;
    const oldBiases = this.biasesValue;
    const oldInput = this.inputValue;

    this.specValue = spec;
    this.topologyValue = Topology.fromSpec(spec);
    this.allocateParameters();

    if (preserveParams) {
      const shared = Math.min(oldSpec.layers.length, spec.layers.length);
      for (let l = 1; l < shared; l++) {
        const oldIn = oldSpec.layers[l - 1].size;
        const newIn = spec.layers[l - 1].size;
        const rows = Math.min(oldSpec.layers[l].size, spec.layers[l].size);
        const cols = Math.min(oldIn, newIn);
        for (let j = 0; j < rows; j++) {
          for (let i = 0; i < cols; i++) this.weightsValue[l][j * newIn + i] = oldWeights[l][j * oldIn + i];
          this.biasesValue[l][j] = oldBiases[l][j];
        }
      }
    }

    this.inputValue = new Float64Array(spec.layers[0].size);
    const source = options.input ?? oldInput;
    for (let i = 0; i < Math.min(source.length, this.inputValue.length); i++) this.inputValue[i] = source[i];

    this.traceValue = createTrace(this.topologyValue.sizes);
    this.mark({ structure: true, params: true, activations: true }, reason, 'many');
  }

  /**
   * Groups several mutations so listeners are notified (and the forward pass
   * recomputed) only once at the end.
   */
  batch(fn: () => void): void {
    this.batchDepth++;
    try {
      fn();
    } finally {
      this.batchDepth--;
      if (this.batchDepth === 0) this.flush();
    }
  }

  // --------------------------------------------------------------- internals

  private allocateParameters(): void {
    const layers = this.specValue.layers;
    this.weightsValue = [new Float64Array(0)];
    this.biasesValue = [new Float64Array(0)];
    for (let l = 1; l < layers.length; l++) {
      this.weightsValue.push(new Float64Array(layers[l].size * layers[l - 1].size));
      this.biasesValue.push(new Float64Array(layers[l].size));
    }
    this.initializeAll();
  }

  private initializeAll(): void {
    const layers = this.specValue.layers;
    for (let l = 1; l < layers.length; l++) {
      // One RNG stream per layer: changing one layer's size does not change
      // the initial weights of the other layers.
      const rng = createRng(mixSeed(this.initValue.seed, l));
      initializeLayer(this.initValue.scheme, layers[l - 1].size, layers[l].size, rng, this.weightsValue[l], this.biasesValue[l]);
    }
  }

  private recompute(): void {
    forwardInto(this.specValue, this.weightsValue, this.biasesValue, this.inputValue, this.traceValue);
  }

  private markParams(layer: number | 'many', reason: ChangeReason): void {
    this.mark({ params: true, activations: true }, reason, layer);
  }

  private mark(
    flags: { structure?: boolean; params?: boolean; activations?: boolean },
    reason: ChangeReason,
    layer?: number | 'many',
  ): void {
    const p = this.pending;
    p.structure ||= !!flags.structure;
    p.params ||= !!flags.params;
    p.activations ||= !!flags.activations;
    if (layer !== undefined) p.layer = p.layer === null || p.layer === layer ? layer : 'many';
    // Inside a batch the most significant reason wins (structure > params > input).
    const rank = flags.structure ? 3 : flags.params ? 2 : 1;
    if (rank >= this.pendingRank) {
      this.pendingRank = rank;
      this.pendingReason = reason;
    }
    if (this.batchDepth === 0) this.flush();
  }

  private flush(): void {
    const { structure, params, activations, layer } = this.pending;
    const reason = this.pendingReason;
    this.pending = { structure: false, params: false, activations: false, layer: null };
    this.pendingRank = 0;
    if (!structure && !params && !activations) return;
    this.recompute();
    if (structure) this.events.emit('structure', { reason });
    if (params) this.events.emit('params', typeof layer === 'number' ? { reason, layer } : { reason });
    this.events.emit('activations', { reason });
  }
}
