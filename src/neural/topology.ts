/**
 * Index bookkeeping for a fully-connected feed-forward network.
 *
 * Global indices let renderers store per-neuron / per-connection data in flat
 * typed arrays (one InstancedMesh instance per element) without duplicating
 * the model:
 *
 *   neuron (layer l, index j)            → neuronOffsets[l] + j
 *   connection (layer l, target j, src i) → connectionOffsets[l] + j · n_{l-1} + i
 *
 * The connection index inside a layer equals the position of W^{(l)}[j, i] in
 * the row-major weight array, so instance buffers mirror parameter buffers.
 */
import type { NetworkSpec } from './config';

export interface NeuronRef {
  layer: number;
  index: number;
}

export interface ConnectionRef {
  /** Layer that owns the weight (the target layer, ≥ 1). */
  layer: number;
  /** Neuron index in `layer`. */
  target: number;
  /** Neuron index in `layer - 1`. */
  source: number;
}

export class Topology {
  readonly sizes: readonly number[];
  readonly neuronOffsets: readonly number[];
  readonly connectionOffsets: readonly number[];
  readonly neuronCount: number;
  readonly connectionCount: number;

  constructor(sizes: readonly number[]) {
    this.sizes = [...sizes];
    const neuronOffsets = [0];
    for (const size of sizes) neuronOffsets.push(neuronOffsets[neuronOffsets.length - 1] + size);
    // connectionOffsets[l] = first weight of layer l; layer 0 owns none.
    const connectionOffsets = [0, 0];
    for (let l = 1; l < sizes.length; l++) connectionOffsets.push(connectionOffsets[l] + sizes[l] * sizes[l - 1]);
    this.neuronOffsets = neuronOffsets;
    this.connectionOffsets = connectionOffsets.slice(0, sizes.length + 1);
    this.neuronCount = neuronOffsets[sizes.length];
    this.connectionCount = this.connectionOffsets[sizes.length];
  }

  static fromSpec(spec: NetworkSpec): Topology {
    return new Topology(spec.layers.map((l) => l.size));
  }

  get layerCount(): number {
    return this.sizes.length;
  }

  /** Number of weights owned by layer l (0 for the input layer). */
  weightCount(layer: number): number {
    return layer === 0 ? 0 : this.sizes[layer] * this.sizes[layer - 1];
  }

  neuronIndex(layer: number, index: number): number {
    return this.neuronOffsets[layer] + index;
  }

  neuronAt(globalIndex: number): NeuronRef {
    let layer = 0;
    while (layer < this.sizes.length - 1 && globalIndex >= this.neuronOffsets[layer + 1]) layer++;
    return { layer, index: globalIndex - this.neuronOffsets[layer] };
  }

  connectionIndex(layer: number, target: number, source: number): number {
    return this.connectionOffsets[layer] + target * this.sizes[layer - 1] + source;
  }

  connectionAt(globalIndex: number): ConnectionRef {
    let layer = 1;
    while (layer < this.sizes.length - 1 && globalIndex >= this.connectionOffsets[layer + 1]) layer++;
    const local = globalIndex - this.connectionOffsets[layer];
    const fanIn = this.sizes[layer - 1];
    return { layer, target: Math.floor(local / fanIn), source: local % fanIn };
  }
}

// ---------------------------------------------------------------------------
// Naming helpers (human-readable names and TeX symbols, 1-based like textbooks)
// ---------------------------------------------------------------------------

/** "I1", "H1-3", "O1" (layer prefix + 1-based index). */
export function neuronName(spec: NetworkSpec, layer: number, index: number): string {
  const l = spec.layers[layer];
  return l.role === 'hidden' ? `${l.prefix}-${index + 1}` : `${l.prefix}${index + 1}`;
}

/** Optional user label of a neuron (from the config), e.g. "cat". */
export function neuronLabel(spec: NetworkSpec, layer: number, index: number): string | null {
  return spec.layers[layer].labels?.[index] ?? null;
}

/** Name plus label when present: `O2 “dog”`. */
export function neuronTitle(spec: NetworkSpec, layer: number, index: number): string {
  const label = neuronLabel(spec, layer, index);
  const name = neuronName(spec, layer, index);
  return label ? `${name} “${label}”` : name;
}

export function connectionName(spec: NetworkSpec, ref: ConnectionRef): string {
  return `${neuronName(spec, ref.layer - 1, ref.source)} → ${neuronName(spec, ref.layer, ref.target)}`;
}

/** TeX symbol of a neuron output: x_{i}, a^{(l)}_{j} or \hat{y}_{j}. */
export function outputSymbol(spec: NetworkSpec, layer: number, index: number): string {
  const role = spec.layers[layer].role;
  if (role === 'input') return `x_{${index + 1}}`;
  if (role === 'output') return `\\hat{y}_{${index + 1}}`;
  return `a^{(${layer})}_{${index + 1}}`;
}

/** TeX symbol of a pre-activation: z^{(l)}_{j}. */
export function preActivationSymbol(layer: number, index: number): string {
  return `z^{(${layer})}_{${index + 1}}`;
}

/** TeX symbol of a weight: W^{(l)}_{j,i}. */
export function weightSymbol(ref: ConnectionRef): string {
  return `W^{(${ref.layer})}_{${ref.target + 1},${ref.source + 1}}`;
}

/** TeX symbol of a bias: b^{(l)}_{j}. */
export function biasSymbol(layer: number, index: number): string {
  return `b^{(${layer})}_{${index + 1}}`;
}

/** TeX symbol of a whole layer's output vector: x, a^{(l)} or \hat{y}. */
export function layerVectorSymbol(spec: NetworkSpec, layer: number): string {
  const role = spec.layers[layer].role;
  if (role === 'input') return 'x';
  if (role === 'output') return '\\hat{y}';
  return `a^{(${layer})}`;
}
